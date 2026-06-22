"""Tiny W&B + tqdm logging helper shared by every training script.

Centralizes:
- Initialization of a Weights & Biases run (project: ``lattice``).
- ``run.log_code`` on the repo root after ``wandb.init``.
- A ``log`` call that updates W&B and (optionally) a tqdm postfix in one go.
- Raises if wandb is not installed (training runs always log to W&B).

Future training scripts (Stages 4, 5) should import ``RunLogger`` rather than
re-implementing wandb wiring.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - wandb is a soft dep
    _wandb = None
    _WANDB_AVAILABLE = False


def _code_log_root() -> str:
    """Repo root for ``run.log_code``."""
    return "."


def log_wandb_code(
    loggers: list[Any] | None = None,
    *,
    root: str | Path = ".",
) -> bool:
    """Snapshot Python sources under ``root`` to the active W&B run(s).

    Works with Lightning ``WandbLogger`` instances (``train.py``) or the global
    ``wandb.run`` after ``wandb.init`` (``RunLogger`` / precompute CLIs).
    """
    root = str(root)
    logged = False
    if loggers:
        from lightning.pytorch.loggers import WandbLogger as _WandbLogger

        for lg in loggers:
            if not isinstance(lg, _WandbLogger):
                continue
            exp = lg.experiment
            log_code = getattr(exp, "log_code", None) if exp is not None else None
            if callable(log_code):
                log_code(root)
                logged = True
                logger.info(
                    "wandb code logged from %s (run %s)",
                    root,
                    getattr(exp, "id", "?"),
                )
    elif _WANDB_AVAILABLE and _wandb.run is not None:
        log_code = getattr(_wandb.run, "log_code", None)
        if callable(log_code):
            log_code(root)
            logged = True
            logger.info("wandb code logged from %s", root)
    if not logged:
        logger.debug("wandb code logging skipped (no active W&B run)")
    return logged


class RunLogger:
    """Bundles wandb logging with optional tqdm postfix updates.

    Args:
        project: W&B project name. Defaults to ``"lattice"``.
        run_name: Optional human-readable run name.
        config: Hyperparameter dict (typically ``vars(cfg)``); shown in the W&B UI.
        tags: Optional list of tags.
    """

    def __init__(
        self,
        project: str = "lattice",
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        if not _WANDB_AVAILABLE:
            raise RuntimeError("wandb is required; install with `pip install wandb`")
        # ``reinit=True`` lets one process create multiple runs across test sessions.
        self.run = _wandb.init(
            project=project,
            name=run_name,
            config=dict(config) if config else None,
            mode=os.environ.get("WANDB_MODE", "online"),
            tags=tags or [],
            reinit=True,
        )
        self.enabled = True
        log_wandb_code(root=_code_log_root())
        if self.run is not None:
            logger.info("wandb run: %s", self.run.url)

    @property
    def run_id(self) -> str:
        if self.run is None:
            raise RuntimeError("wandb run not initialized")
        return str(self.run.id)

    def checkpoint_dir(self, output_dir: Path | str) -> Path:
        """Per-run checkpoint root: ``{output_dir}/checkpoints/{wandb_run_id}/``."""
        ckpt = Path(output_dir) / "checkpoints" / self.run_id
        ckpt.mkdir(parents=True, exist_ok=True)
        return ckpt

    def watch(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        """Forward to ``wandb.watch`` if enabled (collects gradient/param histograms)."""
        if self.enabled and _wandb is not None:
            _wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, metrics: Mapping[str, Any], *, step: int | None = None,
            pbar=None) -> None:
        """Log metrics to W&B (if enabled) and update a tqdm bar's postfix (if given)."""
        if self.enabled and _wandb is not None:
            _wandb.log(dict(metrics), step=step)
        if pbar is not None:
            pbar.set_postfix(
                {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in metrics.items()},
                refresh=False,
            )

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()
            self.enabled = False
            self.run = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

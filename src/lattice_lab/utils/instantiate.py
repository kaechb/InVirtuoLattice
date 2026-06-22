"""Instantiate lists of callbacks / loggers from a Hydra config node.

Both helpers accept a ``DictConfig`` whose values are themselves config nodes
carrying a ``_target_``. Missing / empty nodes are tolerated so a config can
disable all callbacks or loggers with ``callbacks: null`` / ``logger: null``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def checkpoint_dir_for_run(base_dirpath: str | Path | None, run_id: str) -> Path:
    """Return ``{base}/{run_id}``, idempotent if ``base`` already ends with ``run_id``."""
    base = Path(base_dirpath or "checkpoints")
    if base.name == run_id:
        return base
    return base / run_id


def wandb_run_id(loggers: list[Logger]) -> str | None:
    """Return the active W&B run id from the first ``WandbLogger``, if any."""
    for lg in loggers:
        if isinstance(lg, WandbLogger):
            return str(lg.experiment.id)
    return None


def wire_checkpoint_dirs_to_wandb(
    loggers: list[Logger],
    callbacks: list[Callback],
) -> str | None:
    """Append ``/{wandb_run_id}`` to every ``ModelCheckpoint.dirpath``.

    W&B run ids are only known after the logger initializes, so this runs after
    ``instantiate_loggers`` / ``instantiate_callbacks`` and before ``Trainer.fit``.
    """
    run_id = wandb_run_id(loggers)
    if run_id is None:
        return None
    for cb in callbacks:
        if isinstance(cb, ModelCheckpoint):
            ckpt_dir = checkpoint_dir_for_run(cb.dirpath, run_id)
            cb.dirpath = str(ckpt_dir)
            logger.info("ModelCheckpoint dirpath → %s", ckpt_dir)
    return run_id


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Callback]:
    callbacks: list[Callback] = []
    if not callbacks_cfg:
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("callbacks config must be a DictConfig")
    for name, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info("instantiating callback %s <%s>", name, cb_conf._target_)
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    loggers: list[Logger] = []
    if not logger_cfg:
        return loggers
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("logger config must be a DictConfig")
    for name, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info("instantiating logger %s <%s>", name, lg_conf._target_)
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers

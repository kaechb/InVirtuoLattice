"""Stage-2 freeze gate as a Lightning callback.

After ``trainer.fit`` finishes, this runs the three Stage-2 sanity checks
(val-alignment, bioisostere retrieval, QM9 linear probe) on the *best* adapter
(``ModelCheckpoint.best_model_path``) and, only when every check passes,
promotes it to ``adapter_v1.pt``. This replaces the inline promotion block at
the bottom of the original ``train_adapter.train``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from lightning.pytorch import Callback, LightningModule, Trainer

from lattice_lab.eval.bioisostere_retrieval import DEFAULT_BIOISOSTERE_CSV
from lattice_lab.eval.sanity_check import run_sanity_checks

logger = logging.getLogger(__name__)


class SanityGateCallback(Callback):
    def __init__(
        self,
        *,
        skip_sanity: bool = False,
        val_ratio: float = 0.005,
        test_ratio: float = 0.005,
        split_seed: int = 0,
        val_max_pairs: int | None = None,
        bioisostere_csv: str | Path = DEFAULT_BIOISOSTERE_CSV,
        qm9_csv: str | Path | None = "artifacts/preprocessing/raw/qm9.csv",
        qm9_n_subset: int | None = None,
        batch_size: int = 64,
        include_baselines: bool = True,
    ) -> None:
        self.skip_sanity = skip_sanity
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.val_max_pairs = val_max_pairs
        self.bioisostere_csv = Path(bioisostere_csv)
        self.qm9_csv = Path(qm9_csv) if qm9_csv else None
        self.qm9_n_subset = qm9_n_subset
        self.batch_size = batch_size
        self.include_baselines = include_baselines

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.skip_sanity:
            logger.warning("skip_sanity set: not running Stage-2 sanity checks")
            return
        ckpt_cb = trainer.checkpoint_callback
        best = ckpt_cb.best_model_path if ckpt_cb else ""
        encoder = pl_module.encoder  # type: ignore[attr-defined]
        if best:
            state = torch.load(best, map_location=pl_module.device, weights_only=True)
            adapter_state = state.get("state_dict", state)
            adapter_state = {
                k.replace("encoder.adapter.", ""): v
                for k, v in adapter_state.items()
                if k.startswith("encoder.adapter.")
            }
            if adapter_state:
                encoder.adapter.load_state_dict(adapter_state)
                logger.info("loaded best adapter from %s for sanity checks", best)

        shards = trainer.datamodule.shards  # type: ignore[attr-defined]
        encoder.adapter.eval()
        report = run_sanity_checks(
            encoder,
            val_shards=shards,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            split_seed=self.split_seed,
            val_max_pairs=self.val_max_pairs,
            bioisostere_csv=self.bioisostere_csv,
            qm9_csv=self.qm9_csv if self.qm9_csv and self.qm9_csv.exists() else None,
            qm9_n_subset=self.qm9_n_subset,
            device=pl_module.device,
            batch_size=self.batch_size,
            include_baselines=self.include_baselines,
        )
        for lg in trainer.loggers:
            lg.log_metrics(report.as_metrics(), step=trainer.global_step)
        logger.info("sanity all-passed=%s", report.all_passed)

        if report.all_passed and best:
            v1 = Path(best).parent / "adapter_v1.ckpt"
            v1.unlink(missing_ok=True)
            v1.symlink_to(Path(best).name)
            logger.info("promoted %s → %s", Path(best).name, v1)
        elif not report.all_passed:
            logger.warning("adapter NOT promoted to v1: a sanity check failed")

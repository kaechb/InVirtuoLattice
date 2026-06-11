"""Hydra + Lightning evaluation entrypoint.

    python -m lattice_lab.eval ckpt_path=/path/to/last.ckpt

Runs ``Trainer.validate`` so the same ``validation_step`` /
``on_validation_epoch_end`` used during training produces the held-out ranking
metrics (``val/ef1``, ``val/ef5``, ``val/top1``, ``val/bedroc``). For the full
LIT-PCBA / DUD-E external benchmarks, keep using the proven CLI in
``lattice.eval.lit_pcba`` — those harnesses are already self-contained.
"""

from __future__ import annotations

import logging

import hydra
import lightning as L
from omegaconf import DictConfig

from lattice_lab.utils import instantiate_loggers, seed_everything

logger = logging.getLogger(__name__)


def evaluate(cfg: DictConfig) -> dict[str, float]:
    if cfg.get("seed") is not None:
        seed_everything(int(cfg.seed))
    if not cfg.get("ckpt_path"):
        raise ValueError("eval requires ckpt_path=<checkpoint.ckpt>")

    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    loggers = instantiate_loggers(cfg.get("logger"))
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

    trainer.validate(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    logger.info("eval metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    return metrics


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()

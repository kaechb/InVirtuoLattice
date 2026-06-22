"""Seeding + a tiny hyperparameter-logging helper."""

from __future__ import annotations

import logging
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed python / numpy / torch + cudnn determinism flags."""
    import lightning.pytorch as pl

    pl.seed_everything(seed, workers=True)


def log_hyperparameters(
    *,
    cfg: DictConfig,
    model: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
    loggers: list[Logger],
) -> None:
    """Push the resolved run config + parameter counts to every logger.

    Mirrors the lightning-hydra-template helper: without this, ``Trainer`` only
    records ``LightningModule.hparams``, dropping the data / trainer config.
    """
    if not loggers:
        logger.warning("no logger configured; run config will not be tracked")
        return

    hparams: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    for lg in loggers:
        lg.log_hyperparams(hparams)

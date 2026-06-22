"""W&B code snapshot helper."""

from __future__ import annotations

from unittest.mock import MagicMock

import lattice_lab.training.run_logger as run_logger_mod
from lattice_lab.training.run_logger import log_wandb_code
from lightning.pytorch.loggers import WandbLogger


def test_log_wandb_code_uses_lightning_wandb_logger() -> None:
    exp = MagicMock()
    lg = MagicMock(spec=WandbLogger)
    lg.experiment = exp

    assert log_wandb_code([lg], root="/repo") is True
    exp.log_code.assert_called_once_with("/repo")


def test_log_wandb_code_uses_global_run(monkeypatch) -> None:
    run = MagicMock()
    wandb = MagicMock(run=run, __version__="0.16")
    monkeypatch.setattr(run_logger_mod, "_wandb", wandb)
    monkeypatch.setattr(run_logger_mod, "_WANDB_AVAILABLE", True)

    assert log_wandb_code(root=".") is True
    run.log_code.assert_called_once_with(".")


def test_log_wandb_code_noop_without_wandb() -> None:
    assert log_wandb_code([], root=".") is False

"""Checkpoint dir wiring against W&B run ids."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from lightning.pytorch.callbacks import ModelCheckpoint

from lattice_lab.utils import (
    checkpoint_dir_for_run,
    wandb_run_id,
    wire_checkpoint_dirs_to_wandb,
)
from lattice_lab.models.builders import _checkpoint_state_dict, _denoising_jepa_init_kwargs


def test_checkpoint_dir_for_run_appends_run_id() -> None:
    assert checkpoint_dir_for_run("artifacts/adapter/checkpoints", "abc12345") == Path(
        "artifacts/adapter/checkpoints/abc12345"
    )


def test_checkpoint_dir_for_run_is_idempotent() -> None:
    base = Path("artifacts/adapter/checkpoints/abc12345")
    assert checkpoint_dir_for_run(base, "abc12345") == base


def test_wandb_run_id_from_logger() -> None:
    lg = MagicMock()
    lg.experiment.id = "zrk1ptsh"
    # isinstance check needs a real WandbLogger subclass or we patch isinstance
    from lightning.pytorch.loggers import WandbLogger

    lg.__class__ = WandbLogger
    assert wandb_run_id([lg]) == "zrk1ptsh"
    assert wandb_run_id([]) is None


def test_wire_checkpoint_dirs_to_wandb() -> None:
    from lightning.pytorch.loggers import WandbLogger

    lg = MagicMock()
    lg.experiment.id = "73miv4j1"
    lg.__class__ = WandbLogger

    ckpt = ModelCheckpoint(dirpath="logs/train/checkpoints")
    run_id = wire_checkpoint_dirs_to_wandb([lg], [ckpt])
    assert run_id == "73miv4j1"
    assert Path(ckpt.dirpath) == checkpoint_dir_for_run(
        "logs/train/checkpoints", "73miv4j1"
    ).resolve()


def test_wire_checkpoint_dirs_noop_without_wandb() -> None:
    ckpt = ModelCheckpoint(dirpath="logs/train/checkpoints")
    assert wire_checkpoint_dirs_to_wandb([], [ckpt]) is None
    assert Path(ckpt.dirpath) == Path("logs/train/checkpoints").resolve()


def test_checkpoint_state_dict_strips_student_prefix() -> None:
    import torch

    raw = {
        "state_dict": {
            "student.encoder.pool.query": torch.zeros(1),
            "student.denoiser.blocks.0.qw.weight": torch.zeros(1),
        }
    }
    state = _checkpoint_state_dict(raw)
    assert "encoder.pool.query" in state
    assert "denoiser.blocks.0.qw.weight" in state
    assert not any(k.startswith("student.") for k in state)


def test_denoising_jepa_init_kwargs_fills_required_defaults() -> None:
    kwargs = _denoising_jepa_init_kwargs(
        {
            "hyper_parameters": {"kl_beta": 0.5, "fp_weight": 10.0},
            "encoder_config": {"tokenizer_path": "artifacts/tokenizer/smiles_new.json"},
        }
    )
    assert kwargs["ckpt_path"] is None
    assert kwargs["tokenizer_path"] == "artifacts/tokenizer/smiles_new.json"
    assert kwargs["kl_beta"] == 0.5
    assert kwargs["fp_weight"] == 10.0

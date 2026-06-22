"""Checkpoint loading: full Lightning ``.ckpt`` only (no legacy partial bundles).

Every stage saves the whole module, so loading is a single prefix split:
``encoder.*`` for the encoder (backbone + adapter + time) and ``head.*`` for the
energy head. The hook layer range and dims travel with the ckpt as
``encoder_config``, so a single file is enough to rebuild the encoder.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lattice_lab.models.builders import (
    adapter_run_id,
    parse_adapter_state,
    parse_head_checkpoint,
    resolve_adapter_ckpt,
    zm_store_path,
)


def _full_ckpt(tmp_path: Path, *, t0: float = 0.25, name: str = "model.ckpt") -> Path:
    """A whole-model Lightning checkpoint: frozen backbone + adapter + head."""
    ckpt_path = tmp_path / name
    torch.save(
        {
            "state_dict": {
                "encoder.backbone.block.0.weight": torch.randn(4, 4),
                "encoder.adapter.input_proj.weight": torch.randn(4, 8),
                "encoder.time_logit": torch.logit(torch.tensor(t0)),
                "head.film.weight": torch.randn(2, 4),
            },
            "encoder_config": {"backbone_layer_start": 8, "backbone_layer_end": 11},
        },
        ckpt_path,
    )
    return ckpt_path


def test_parse_adapter_state_from_full_ckpt(tmp_path: Path) -> None:
    raw = torch.load(_full_ckpt(tmp_path), weights_only=False)
    adapter_state = parse_adapter_state(raw)
    assert set(adapter_state) == {"input_proj.weight"}  # only encoder.adapter.*


def test_parse_head_checkpoint_from_full_ckpt(tmp_path: Path) -> None:
    raw = torch.load(_full_ckpt(tmp_path), weights_only=False)
    head_state = parse_head_checkpoint(raw)
    assert set(head_state) == {"film.weight"}  # only head.*


def test_resolve_adapter_ckpt_from_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "abc123"
    run_dir.mkdir()
    _full_ckpt(run_dir, name="last.ckpt")
    resolved = resolve_adapter_ckpt(run_dir)
    assert resolved.name == "last.ckpt"


def test_adapter_run_id_and_zm_store_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "xt4v2kk8"
    run_dir.mkdir()
    ckpt = _full_ckpt(run_dir, name="last.ckpt")
    assert adapter_run_id(ckpt) == "xt4v2kk8"
    assert zm_store_path(ckpt, "decoy_zm") == Path("artifacts/decoys/xt4v2kk8/decoy_zm")
    assert zm_store_path(ckpt, "binder_zm") == Path("artifacts/binders/xt4v2kk8/binder_zm")


def test_parse_adapter_state_rejects_no_adapter() -> None:
    raw = {"state_dict": {"head.film.weight": torch.randn(2, 4)}}
    with pytest.raises(ValueError, match="no adapter weights"):
        parse_adapter_state(raw)


def test_parse_head_checkpoint_rejects_no_head() -> None:
    raw = {"state_dict": {"encoder.adapter.input_proj.weight": torch.randn(4, 8)}}
    with pytest.raises(ValueError, match="no energy-head weights"):
        parse_head_checkpoint(raw)


def test_checkpoint_without_state_dict_rejected() -> None:
    with pytest.raises(ValueError, match="no 'state_dict'"):
        parse_head_checkpoint({})

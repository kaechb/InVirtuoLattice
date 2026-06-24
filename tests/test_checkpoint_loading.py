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
    ebm_run_id,
    infer_energy_head_dims,
    load_energy_head,
    parse_adapter_state,
    parse_head_checkpoint,
    resolve_adapter_ckpt,
    resolve_ebm_ckpt,
    resolve_ssl_best_ckpt,
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


def test_parse_encoder_state_from_full_ckpt(tmp_path: Path) -> None:
    from lattice_lab.models.builders import parse_encoder_state

    raw = torch.load(_full_ckpt(tmp_path), weights_only=False)
    enc_state = parse_encoder_state(raw)
    assert set(enc_state) == {
        "backbone.block.0.weight",
        "adapter.input_proj.weight",
        "time_logit",
    }


def test_load_encoder_from_ckpt_requires_ckpt() -> None:
    from lattice_lab.models.builders import load_encoder_from_ckpt

    with pytest.raises(TypeError, match="requires ckpt"):
        load_encoder_from_ckpt()


def test_adapter_fingerprint_includes_backbone(tmp_path: Path) -> None:
    from lattice_lab.models.builders import adapter_fingerprint

    ckpt_a = _full_ckpt(tmp_path, name="a.ckpt")
    ckpt_b = _full_ckpt(tmp_path, name="b.ckpt")
    raw = torch.load(ckpt_b, weights_only=False)
    raw["state_dict"]["encoder.backbone.block.0.weight"] = torch.randn(4, 4)
    torch.save(raw, ckpt_b)
    assert adapter_fingerprint(ckpt_a) != adapter_fingerprint(ckpt_b)


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


def test_adapter_run_id_from_best_r2mean(tmp_path: Path) -> None:
    run_dir = tmp_path / "abc123"
    run_dir.mkdir()
    best = _full_ckpt(run_dir, name="best-r2mean-epoch=01-step=1000.ckpt")
    assert adapter_run_id(best) == "abc123"
    assert resolve_ssl_best_ckpt(run_dir) == best


def test_resolve_ebm_ckpt_prefers_ebm_prefix(tmp_path: Path) -> None:
    run_dir = tmp_path / "ebmrun"
    run_dir.mkdir()
    _full_ckpt(run_dir, name="last.ckpt")
    best = _full_ckpt(run_dir, name="ebm-epoch=02-step=2000.ckpt")
    assert resolve_ebm_ckpt(run_dir) == best
    assert ebm_run_id(best) == "ebmrun"


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


def _ebm_head_ckpt(tmp_path: Path, *, d_m: int = 256, d_p: int = 1280) -> Path:
    d_hidden = 512
    ckpt_path = tmp_path / "ebm.ckpt"
    torch.save(
        {
            "state_dict": {
                "head.mol_proj.weight": torch.randn(d_hidden, d_m),
                "head.mol_proj.bias": torch.randn(d_hidden),
                "head.protein_proj.0.weight": torch.randn(d_hidden, d_p),
                "head.protein_proj.0.bias": torch.randn(d_hidden),
            },
            "hyper_parameters": {"d_adapter": d_m, "d_protein": d_p},
        },
        ckpt_path,
    )
    return ckpt_path


def test_infer_energy_head_dims_from_hyperparams(tmp_path: Path) -> None:
    raw = torch.load(_ebm_head_ckpt(tmp_path), weights_only=False)
    assert infer_energy_head_dims(raw) == (256, 1280)


def test_infer_energy_head_dims_from_weights_without_hyperparams(
    tmp_path: Path,
) -> None:
    raw = torch.load(_ebm_head_ckpt(tmp_path), weights_only=False)
    del raw["hyper_parameters"]
    assert infer_energy_head_dims(raw) == (256, 1280)


def test_load_energy_head_infers_dims(tmp_path: Path) -> None:
    from lattice_lab.models.builders import build_energy_head

    head = build_energy_head(d_adapter=256, d_protein=1280)
    ckpt_path = tmp_path / "ebm.ckpt"
    torch.save(
        {
            "state_dict": {f"head.{k}": v for k, v in head.state_dict().items()},
            "hyper_parameters": {"d_adapter": 256, "d_protein": 1280},
        },
        ckpt_path,
    )
    loaded = load_energy_head(ckpt_path)
    assert loaded.d_m == 256
    assert loaded.d_p == 1280

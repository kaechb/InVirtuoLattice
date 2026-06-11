"""Exact-parity tests: the re-homed kernels must produce bit-identical results
to the original ``lattice`` package.

The new package's kernels are copies of the originals with only import paths
rewritten, so on identical inputs + weights every forward pass must match
exactly (``torch.equal``). These tests guard against accidental divergence
(e.g. a stale copy after concurrent edits to ``lattice/``).

Skipped automatically when the original ``lattice`` package or the real adapter
checkpoint isn't importable/present, so they're a no-op in a bare checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("lattice", reason="original lattice package not importable")

REPO = Path(__file__).resolve().parents[1]
ADAPTER_CKPT = REPO / "artifacts/adapter/checkpoints_ssl2/adapter_v1.pt"
TRAIN_PARQUET = REPO / "artifacts/processed/bindingdb/threshold_90/train.parquet"

D_ADAPTER, D_PROTEIN, N_LAYERS = 512, 1280, 4


def _sample_views(n: int = 6, seed: int = 123) -> list[str]:
    """Deterministic FragMol views from real binder SMILES (read-only)."""
    import pandas as pd

    from lattice.preprocessing.molecules import smiles_to_fragmol_views

    smis = pd.read_parquet(TRAIN_PARQUET, columns=["smiles"]).head(64)["smiles"].tolist()
    views: list[str] = []
    for s in smis:
        if len(views) >= n:
            break
        v = smiles_to_fragmol_views(s, n_views=1, seed=seed)
        views.append(v[0] if v else "C")
    return views


requires_ckpt = pytest.mark.skipif(
    not (ADAPTER_CKPT.exists() and TRAIN_PARQUET.exists()),
    reason="real adapter checkpoint / train parquet not present on this node",
)


@requires_ckpt
def test_encoder_zm_bit_identical() -> None:
    from lattice.training.train_ebm import EBMTrainConfig
    from lattice.training.train_ebm import build_encoder as old_build

    from lattice_lab.models.builders import build_ebm_encoder as new_build

    old_enc = old_build(EBMTrainConfig(
        adapter_ckpt=ADAPTER_CKPT, n_fragmol_layers=N_LAYERS, d_adapter=D_ADAPTER,
    ))
    new_enc = new_build(adapter_ckpt=ADAPTER_CKPT, n_fragmol_layers=N_LAYERS, d_adapter=D_ADAPTER)

    views = _sample_views()
    with torch.no_grad():
        z_old = old_enc.encode_views(views, device="cpu")
        z_new = new_enc.encode_views(views, device="cpu")
    assert torch.equal(z_old, z_new), (z_old - z_new).abs().max().item()


def test_energy_head_bit_identical() -> None:
    from lattice.ebm.head import EnergyHead as OldHead
    from lattice.ebm.head import EnergyHeadConfig as OldCfg

    from lattice_lab.ebm.head import EnergyHead as NewHead
    from lattice_lab.ebm.head import EnergyHeadConfig as NewCfg

    torch.manual_seed(0)
    new_head = NewHead(NewCfg(d_m=D_ADAPTER, d_p=D_PROTEIN, arch="cross_attn"))
    old_head = OldHead(OldCfg(d_m=D_ADAPTER, d_p=D_PROTEIN, arch="cross_attn"))
    old_head.load_state_dict(new_head.state_dict())  # force identical weights
    old_head.eval()   # disable dropout so the forward is deterministic
    new_head.eval()

    torch.manual_seed(1)
    z_m = torch.randn(6, D_ADAPTER)
    z_p = torch.randn(6, D_PROTEIN)
    with torch.no_grad():
        e_old = old_head(z_m, z_p)
        e_new = new_head(z_m, z_p)
    assert torch.equal(e_old, e_new)


def test_losses_bit_identical() -> None:
    from lattice.ebm.losses import InfoNCEEnergyLoss as OldInfo
    from lattice.ebm.losses import cross_target_margin_loss as old_ct
    from lattice.ebm.losses import sinkhorn_divergence_1d as old_sink

    from lattice_lab.ebm.losses import InfoNCEEnergyLoss as NewInfo
    from lattice_lab.ebm.losses import cross_target_margin_loss as new_ct
    from lattice_lab.ebm.losses import sinkhorn_divergence_1d as new_sink

    torch.manual_seed(2)
    e_pos = torch.randn(8)
    e_dec = torch.randn(8, 16)
    e_wrong = torch.randn(8)

    assert torch.equal(OldInfo(0.1)(e_pos, e_dec)[0], NewInfo(0.1)(e_pos, e_dec)[0])
    assert torch.equal(old_ct(e_pos, e_wrong, margin=2.0)[0], new_ct(e_pos, e_wrong, margin=2.0)[0])

    x = torch.cat([e_pos.unsqueeze(1), e_dec], dim=1)
    y = torch.randn(8, 17)
    assert torch.equal(old_sink(x, y), new_sink(x, y))

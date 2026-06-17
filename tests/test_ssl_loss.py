"""Unit tests for SSL losses (NT-Xent + LeJEPA/SIGReg)."""

from __future__ import annotations

import torch

from lattice_lab.training.ssl_loss import (
    LeJEPALoss,
    NTXentLoss,
    SIGReg,
    lejepa_retrieval_acc1,
    top1_paired_accuracy,
)
from lattice_lab.training.ssl_val_probes import embedding_covariance_rank


def test_sigreg_forward_shape() -> None:
    proj = torch.randn(2, 8, 16)
    loss = SIGReg(num_projections=32, knots=9)(proj)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_lejepa_inv_lower_when_locals_near_global_mean() -> None:
    loss_fn = LeJEPALoss(lejepa_lambda=0.05, sigreg_num_projections=16, sigreg_knots=9)
    global_z = torch.randn(8, 2, 64)
    close_local = global_z.mean(dim=1, keepdim=True) + torch.randn(8, 2, 64) * 0.05
    far_local = torch.randn(8, 2, 64)
    z_all_close = torch.cat([global_z, close_local], dim=1)
    z_all_far = torch.cat([global_z, far_local], dim=1)
    assert loss_fn(global_z, z_all_close).inv.item() < loss_fn(global_z, z_all_far).inv.item()


def test_lejepa_inv_zero_when_all_views_match_global_center() -> None:
    center = torch.randn(8, 64)
    z_g = center.unsqueeze(1).expand(8, 2, 64)
    z_all = center.unsqueeze(1).expand(8, 4, 64)
    terms = LeJEPALoss(lejepa_lambda=0.05, sigreg_num_projections=16, sigreg_knots=9)(z_g, z_all)
    assert terms.inv.item() == 0.0


def test_lejepa_unnormalized_views_have_gradients() -> None:
    z_g = torch.randn(8, 2, 128, requires_grad=True)
    z_l = z_g + torch.randn(8, 2, 128) * 0.1
    z_all = torch.cat([z_g, z_l], dim=1)
    terms = LeJEPALoss(lejepa_lambda=0.05, sigreg_num_projections=32, sigreg_knots=9)(z_g, z_all)
    terms.total.backward()
    assert z_g.grad is not None
    assert z_g.grad.norm().item() > 1e-4
    assert terms.inv.item() > 1e-3


def test_ntxent_perfect_pairs_lower_than_shuffled() -> None:
    z = torch.randn(4, 32)
    z = torch.nn.functional.normalize(z, dim=-1)
    loss_fn = NTXentLoss(temperature=0.1)
    perfect = loss_fn(z, z)
    shuffled = loss_fn(z, z.roll(1, dims=0))
    assert perfect.item() < shuffled.item()


def test_top1_paired_accuracy() -> None:
    z = torch.eye(4)
    assert top1_paired_accuracy(z, z) == 1.0


def test_embedding_covariance_rank_collapsed_vs_full() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    full = rng.standard_normal((128, 32))
    collapsed = rng.standard_normal((128, 1)) @ rng.standard_normal((1, 32))
    full_eff, full_num = embedding_covariance_rank(full)
    collapsed_eff, collapsed_num = embedding_covariance_rank(collapsed)
    assert full_eff > collapsed_eff
    assert full_num > collapsed_num


def test_lejepa_retrieval_acc1_global_views() -> None:
    c = torch.randn(8, 16)
    z_g = torch.stack([c, c], dim=1)
    z_all = torch.cat([z_g, torch.randn(8, 2, 16)], dim=1)
    assert lejepa_retrieval_acc1(z_g, z_g) == 1.0
    assert lejepa_retrieval_acc1(z_g, z_all) is not None

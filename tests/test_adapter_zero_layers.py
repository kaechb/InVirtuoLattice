"""Adapter with n_layers=0: linear proj + pool only (no token transformer)."""

from __future__ import annotations

import pytest
import torch

from lattice_lab.backbone.adapter import Adapter


def test_adapter_zero_layers_forward() -> None:
    a = Adapter(d_backbone=768, n_backbone_layers=4, d_adapter=512, n_layers=0)
    assert a.encoder is None
    x = torch.randn(2, 10, 4 * 768)
    m = torch.ones(2, 10)
    z = a(x, m)
    assert z.shape == (2, 512)
    z_m, tok = a(x, m, return_tokens=True, normalize=False)
    assert z_m.shape == (2, 512)
    assert tok.shape == (2, 10, 512)


def test_dual_attn_pool_requires_attn() -> None:
    with pytest.raises(ValueError, match="dual_attn_pool requires pool='attn'"):
        Adapter(d_backbone=64, n_backbone_layers=1, d_adapter=16, n_layers=0, dual_attn_pool=True)


def test_dual_attn_pool_halves_and_concats() -> None:
    """Each pool is half-width; z_m is their concatenation back to d_adapter."""
    a = Adapter(
        d_backbone=64, n_backbone_layers=1, d_adapter=16, n_layers=0,
        pool="attn", dual_attn_pool=True,
    )
    assert a.d_pool == 8
    assert a.pool_query.shape[-1] == 8 and a.proj_pool_query.shape[-1] == 8
    x = torch.randn(2, 8, 64)
    m = torch.ones(2, 8)
    z_m = a(x, m, normalize=False)
    assert z_m.shape == (2, 16)  # concat(8, 8)


def test_dual_attn_pool_isolates_contrastive_from_regression_half() -> None:
    """The contrastive projection reads only the projection half, so its gradient
    never reaches the regression (main) pool. z_m (concat) trains both."""
    a = Adapter(
        d_backbone=64, n_backbone_layers=1, d_adapter=16, n_layers=0,
        pool="attn", dual_attn_pool=True,
    )
    x = torch.randn(2, 8, 64)
    m = torch.ones(2, 8)

    # Contrastive gradient touches only the projection pool, never the main pool.
    _, z_p = a(x, m, return_projection=True, normalize=False)
    z_p.sum().backward()
    assert a.proj_pool_query.grad is not None
    assert a.pool_query.grad is None

    a.zero_grad(set_to_none=True)

    # z_m (concat → regression / downstream) trains the main pool.
    z_m = a(x, m, normalize=False)
    z_m.sum().backward()
    assert a.pool_query.grad is not None

"""Adapter with n_layers=0: linear proj + pool only (no token transformer)."""

from __future__ import annotations

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

"""Conditional energy head ``E_θ(z_m | z_p)``.

The head takes a molecule latent ``z_m ∈ R^{d_m}`` (output of the frozen
backbone+adapter, default ``d_m=512``) and a protein latent ``z_p ∈ R^{d_p}``
(mean-pooled ESM-2 embedding, default ``d_p=1280``) and returns a scalar
energy. By convention, lower energy = more likely binder, so
``p(bind | z_m, z_p) ∝ exp(-E_θ)``.

Architecture:
- Project ``z_p`` 1280 → ``2 * d_hidden`` via a 2-layer MLP that produces the
  FiLM parameters ``γ, β``.
- Condition the molecule latent multiplicatively: ``h = h_m * (1 + γ) + β``.
  Because both representations are single (mean-pooled) vectors, FiLM is the
  natural conditioning operator — there is no sequence to attend over — and it
  cannot drop ``z_p`` from the computation the way an additive residual could.
- LayerNorm, then a 3-layer MLP that outputs a scalar energy.

The final layer of ``protein_proj`` is zero-initialised so γ=β=0 at init and
the head starts as the identity (``h = h_m``); the cross-target gradient then
progressively opens the gate rather than fighting random ``z_p`` perturbations.

The forward pass accepts an extra leading "molecules-per-target" dimension so
the InfoNCE loss can score one binder + N decoys against the same target in a
single call without an explicit Python loop.
"""

from __future__ import annotations

import torch
from torch import nn


class EnergyHead(nn.Module):
    """``E_θ(z_m, z_p) → scalar``. Supports arbitrary leading batch shapes."""

    def __init__(
        self,
        *,
        d_m: int = 512,
        d_p: int = 1280,
        d_hidden: int = 512,
        mlp_hidden: int = 512,
        mlp_out: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_m = d_m
        self.d_p = d_p
        self.d_hidden = d_hidden

        if d_m != d_hidden:
            self.mol_proj: nn.Module = nn.Linear(d_m, d_hidden)
        else:
            self.mol_proj = nn.Identity()

        self.protein_proj = nn.Sequential(
            nn.Linear(d_p, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 2 * d_hidden),
        )

        self.post_norm = nn.LayerNorm(d_hidden)
        self.energy_mlp = nn.Sequential(
            nn.Linear(d_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_out, 1),
        )

        self._init_weights()
        self._init_film_to_identity()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_film_to_identity(self) -> None:
        """Zero the final layer of ``protein_proj`` so γ=β=0 at init.

        FiLM then starts as the identity (``h = h_m * 1 + 0``), which lets the
        cross-target gradient progressively open the gate rather than fighting
        random ``z_p`` perturbations.
        """
        last = self.protein_proj[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, z_m: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """Compute energies for paired ``(z_m, z_p)``.

        Args:
            z_m: ``[..., d_m]`` molecule latents.
            z_p: ``[..., d_p]`` protein latents. Must broadcast against ``z_m``
                on every dimension except the last.

        Returns:
            ``[...]`` scalar energies (no trailing feature axis).
        """
        if z_m.shape[-1] != self.d_m:
            raise ValueError(
                f"z_m last dim {z_m.shape[-1]} != d_m {self.d_m}"
            )
        if z_p.shape[-1] != self.d_p:
            raise ValueError(
                f"z_p last dim {z_p.shape[-1]} != d_p {self.d_p}"
            )

        h_m = self.mol_proj(z_m)
        h_p = self.protein_proj(z_p)  # [..., 2 * d_hidden]

        # FiLM: h = h_m * (1 + γ) + β, broadcasting z_p over molecule axes.
        # h_p typically arrives already aligned (e.g. [K, 1, 2D] vs h_m [K, M, D]).
        gamma, beta = h_p.chunk(2, dim=-1)
        h = h_m * (1.0 + gamma) + beta

        h = self.post_norm(h)
        e = self.energy_mlp(h).squeeze(-1)
        return e


class CosineMatchHead(nn.Module):
    """Contrastive matching baseline: ``E = -cos(proj_m(z_m), proj_p(z_p))``.

    Linear projections into a shared space, L2-normalize, score by negative
    cosine. Same ``(z_m, z_p) → scalar`` contract as :class:`EnergyHead` so
    Stage-5/6 wiring is unchanged. Use via ``model.head_type=cosine``.
    """

    def __init__(self, *, d_m: int = 512, d_p: int = 1280, d_hidden: int = 512) -> None:
        super().__init__()
        self.d_m = d_m
        self.d_p = d_p
        self.d_hidden = d_hidden
        self.mol = nn.Linear(d_m, d_hidden, bias=False)
        self.prot = nn.Linear(d_p, d_hidden, bias=False)

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, z_m: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        if z_m.shape[-1] != self.d_m:
            raise ValueError(f"z_m last dim {z_m.shape[-1]} != d_m {self.d_m}")
        if z_p.shape[-1] != self.d_p:
            raise ValueError(f"z_p last dim {z_p.shape[-1]} != d_p {self.d_p}")
        a = nn.functional.normalize(self.mol(z_m), dim=-1)
        b = nn.functional.normalize(self.prot(z_p), dim=-1)
        return -(a * b).sum(dim=-1)


if __name__ == "__main__":
    # Binder and protein that align after proj must score lower energy than a decoy.
    torch.manual_seed(0)
    h = CosineMatchHead(d_m=8, d_p=16, d_hidden=8)
    with torch.no_grad():
        h.mol.weight.copy_(torch.eye(8))
        h.prot.weight.zero_()
        h.prot.weight[:, :8] = torch.eye(8)
    z_m_pos = torch.randn(1, 8)
    z_m_neg = -z_m_pos
    z_p = torch.cat([z_m_pos, torch.zeros(1, 8)], dim=-1)
    e_pos = h(z_m_pos, z_p)
    e_neg = h(z_m_neg, z_p)
    assert e_pos.item() < e_neg.item(), (e_pos.item(), e_neg.item())
    print("CosineMatchHead ok")

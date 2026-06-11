"""Conditional energy head ``E_θ(z_m | z_p)``.

The head takes a molecule latent ``z_m ∈ R^{d_m}`` (output of the frozen
backbone+adapter, default ``d_m=512``) and a protein latent ``z_p ∈ R^{d_p}``
(mean-pooled ESM-2 embedding, default ``d_p=1280``) and returns a scalar
energy. By convention, lower energy = more likely binder, so
``p(bind | z_m, z_p) ∝ exp(-E_θ)``.

Architecture (README §Stage 4):
- Project ``z_p`` 1280 → 512 via a 2-layer MLP.
- Single-layer cross-attention block where the molecule latent is the query
  and the (projected) protein latent is the key/value. Because both
  representations are single vectors, the attention is run on a length-1
  sequence; the multi-head softmax over a single key still provides a learnt
  per-head gate. This keeps the README spec while remaining well-defined for
  mean-pooled inputs.
- Residual + LayerNorm, then a 3-layer MLP that outputs a scalar energy.

The forward pass accepts an extra leading "molecules-per-target" dimension so
the InfoNCE loss can score one binder + N decoys against the same target in a
single call without an explicit Python loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

# Hard-negative mining scores ``batch × (n_decoys × mult)`` pairs in one call.
# Flattening that into a single ``MultiheadAttention`` batch (e.g. 64×1800=115k)
# exceeds CUDA SDPA launch limits on consumer GPUs; chunk instead.
_CROSS_ATTN_CHUNK = 8192


@dataclass(frozen=True)
class EnergyHeadConfig:
    """Hyperparameters for :class:`EnergyHead`.

    ``arch="cross_attn"`` (default) is the original residual cross-attention
    block. ``arch="film"`` swaps the residual for FiLM conditioning
    (``h = h_m * (1+γ) + β`` with ``γ, β = MLP(z_p)``) so the energy cannot
    drop ``z_p`` from the computation — the cross-attention residual can be
    silenced by driving ``attn_out → 0``, but FiLM has no such shortcut.
    """

    d_m: int = 512        # molecule latent dim (adapter output)
    d_p: int = 1280       # protein latent dim (ESM-2 650M mean-pool)
    d_hidden: int = 512   # shared hidden width
    n_heads: int = 8
    mlp_hidden: int = 512
    mlp_out: int = 256
    dropout: float = 0.1
    arch: str = "cross_attn"


class EnergyHead(nn.Module):
    """``E_θ(z_m, z_p) → scalar``. Supports arbitrary leading batch shapes."""

    def __init__(self, cfg: EnergyHeadConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or EnergyHeadConfig()
        c = self.cfg
        if c.d_hidden % c.n_heads != 0:
            raise ValueError(
                f"d_hidden={c.d_hidden} must be divisible by n_heads={c.n_heads}"
            )

        if c.arch not in ("cross_attn", "film"):
            raise ValueError(f"unknown arch={c.arch!r}; expected 'cross_attn' or 'film'")

        if c.d_m != c.d_hidden:
            self.mol_proj: nn.Module = nn.Linear(c.d_m, c.d_hidden)
        else:
            self.mol_proj = nn.Identity()

        if c.arch == "cross_attn":
            self.protein_proj = nn.Sequential(
                nn.Linear(c.d_p, c.d_hidden),
                nn.GELU(),
                nn.Linear(c.d_hidden, c.d_hidden),
            )
            self.q_norm = nn.LayerNorm(c.d_hidden)
            self.kv_norm = nn.LayerNorm(c.d_hidden)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=c.d_hidden,
                num_heads=c.n_heads,
                dropout=c.dropout,
                batch_first=True,
            )
        else:  # film
            self.protein_proj = nn.Sequential(
                nn.Linear(c.d_p, c.d_hidden),
                nn.GELU(),
                nn.Linear(c.d_hidden, 2 * c.d_hidden),
            )

        self.post_norm = nn.LayerNorm(c.d_hidden)
        self.energy_mlp = nn.Sequential(
            nn.Linear(c.d_hidden, c.mlp_hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.mlp_hidden, c.mlp_out),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.mlp_out, 1),
        )

        self._init_weights()
        if c.arch == "film":
            self._init_film_to_identity()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_film_to_identity(self) -> None:
        """Zero the final layer of ``protein_proj`` so γ=β=0 at init.

        FiLM then starts as the identity (``h = h_m * 1 + 0``), which keeps the
        early training dynamics close to the cross-attention variant and lets
        the cross-target gradient progressively open the gate rather than
        fighting random ``z_p`` perturbations.
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
        if z_m.shape[-1] != self.cfg.d_m:
            raise ValueError(
                f"z_m last dim {z_m.shape[-1]} != d_m {self.cfg.d_m}"
            )
        if z_p.shape[-1] != self.cfg.d_p:
            raise ValueError(
                f"z_p last dim {z_p.shape[-1]} != d_p {self.cfg.d_p}"
            )

        h_m = self.mol_proj(z_m)
        h_p = self.protein_proj(z_p)  # [..., d_hidden] or [..., 2*d_hidden]

        if self.cfg.arch == "cross_attn":
            h_p = h_p.expand_as(h_m)
            lead_shape = h_m.shape[:-1]
            flat = math.prod(lead_shape) if lead_shape else 1
            h_m_flat = h_m.reshape(flat, self.cfg.d_hidden)
            h_p_flat = h_p.reshape(flat, self.cfg.d_hidden)
            attn_chunks: list[torch.Tensor] = []
            for start in range(0, flat, _CROSS_ATTN_CHUNK):
                end = min(start + _CROSS_ATTN_CHUNK, flat)
                q = self.q_norm(h_m_flat[start:end]).unsqueeze(1)
                kv = self.kv_norm(h_p_flat[start:end]).unsqueeze(1)
                out, _ = self.cross_attn(q, kv, kv, need_weights=False)
                attn_chunks.append(out.squeeze(1))
            attn_out = torch.cat(attn_chunks, dim=0).reshape(*lead_shape, self.cfg.d_hidden)
            h = h_m + attn_out
        else:  # film: h = h_m * (1 + γ) + β, broadcasting z_p over molecule axes
            # h_p has shape [..., 2 * d_hidden] from z_p's leading dims; broadcast
            # against h_m on every leading axis except the feature one.
            gamma, beta = h_p.chunk(2, dim=-1)
            # Broadcasting: rely on torch's usual rules — z_p typically arrives
            # already aligned (e.g. [K, 1, 2D] vs h_m [K, M, D]).
            h = h_m * (1.0 + gamma) + beta

        h = self.post_norm(h)
        e = self.energy_mlp(h).squeeze(-1)
        return e

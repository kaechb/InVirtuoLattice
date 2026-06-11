"""Stage 2 adapter module.

Takes hidden states from the last ``L`` FragMol decoder layers, projects them to
``d_adapter``, runs a bidirectional Transformer encoder, mean-pools over real
tokens, and returns ``z_m ∈ R^{d_adapter}``. A small projection head (MLP) is
provided for SimCLR contrastive training; it is discarded after SSL.

Total parameter target: ~10M (4-layer, 8-head, d=512 encoder is the dominant
chunk; linear projection contributes ~1.5M for 4×768 → 512).
"""

#TODO: this the input_layer makes 0 sense - just take the normal layers of the pretrained transformers so we dont add a shitton of parameters for nothing

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterConfig:
    d_fragmol: int = 768
    n_fragmol_layers: int = 4  # how many of FragMol's last layers to consume
    d_adapter: int = 512
    n_heads: int = 8
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    proj_dim: int = 128  # SimCLR projection-head output dim
    proj_hidden: int = 512


class Adapter(nn.Module):
    """Concat → linear → bidirectional encoder → masked mean-pool → ``z_m``.

    The projection head (``proj_head``) is exposed separately so callers can
    forward through it during SSL but ignore it after training is frozen.
    """

    def __init__(self, cfg: AdapterConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or AdapterConfig()
        in_dim = self.cfg.d_fragmol * self.cfg.n_fragmol_layers

        self.input_proj = nn.Linear(in_dim, self.cfg.d_adapter)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_adapter,
            nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.d_adapter * self.cfg.ff_mult,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.n_layers)
        self.norm = nn.LayerNorm(self.cfg.d_adapter)

        self.proj_head = nn.Sequential(
            nn.Linear(self.cfg.d_adapter, self.cfg.proj_hidden),
            nn.GELU(),
            nn.Linear(self.cfg.proj_hidden, self.cfg.proj_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for m in self.proj_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over the time axis ignoring positions where ``mask`` is 0.

        ``x``:    ``[B, T, D]``. ``mask``: ``[B, T]`` with values in {0, 1}.
        Returns ``[B, D]``. Falls back to a uniform mean if a row's mask is all-zero,
        which should never happen on real data but guards against NaNs.
        """
        m = mask.unsqueeze(-1).to(x.dtype)
        s = (x * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1e-6)
        return s / denom

    def forward(
        self,
        hidden_states_concat: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_projection: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute ``z_m`` (and optionally the SimCLR projection ``z_p``).

        Args:
            hidden_states_concat: ``[B, T, L*d_fragmol]`` — concatenated hidden
                states from the last L FragMol layers.
            attention_mask: ``[B, T]`` with 1 at real-token positions, 0 at pads.
            return_projection: if True, also return the projection-head output.

        Returns:
            ``z_m`` ``[B, d_adapter]`` (L2-normalized), or
            ``(z_m, z_p)`` with ``z_p`` ``[B, proj_dim]`` also L2-normalized.
        """
        x = self.input_proj(hidden_states_concat)
        # nn.TransformerEncoder expects ``src_key_padding_mask`` where True = ignore.
        key_padding_mask = attention_mask <= 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        # Exclude BOS (idx 0) and EOS (last real token) from pooling, per README:
        # "Mean pooling over token positions (excluding special tokens)". We assume
        # callers pass a mask where BOS and EOS positions are already zeroed; the
        # ``stack_views`` helper does that. If they aren't, masked_mean still works.
        pooled = self.masked_mean(x, attention_mask)
        z_m = torch.nn.functional.normalize(pooled, dim=-1)
        if not return_projection:
            return z_m
        z_p = self.proj_head(pooled)
        z_p = torch.nn.functional.normalize(z_p, dim=-1)
        return z_m, z_p

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

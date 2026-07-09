"""Stage 2 adapter module.

Takes hidden states from ``L`` backbone blocks, projects them to
``d_adapter``, runs a bidirectional Transformer encoder, mean-pools over real
tokens, and returns ``z_m ∈ R^{d_adapter}``. A small projection head (MLP) is
provided for SimCLR contrastive training; it is discarded after SSL.

Total parameter target: ~10M (4-layer, 8-head, d=512 encoder is the dominant
chunk; linear projection contributes ~1.5M for 4×768 → 512).
"""

from __future__ import annotations

import torch
from torch import nn


class Adapter(nn.Module):
    """Concat → linear → bidirectional encoder → masked mean-pool → ``z_m``.

    The projection head (``proj_head``) is exposed separately so callers can
    forward through it during SSL but ignore it after training is frozen.

    With ``dual_attn_pool`` (requires ``pool='attn'``) there are two independent
    attention pools, each of half width (``d_adapter // 2``). The kept representation
    ``z_m`` is their concatenation (back to ``d_adapter``), so t-SNE / linear probes
    see both. The contrastive projection head reads *only* the second (projection)
    half, so its invariance gradient never touches the first (regression) half —
    decoupling the contrastive invariance pressure from the latent-regression
    detail pressure while the shared backbone/token features serve both.
    """

    def __init__(
        self,
        *,
        d_backbone: int = 768,
        n_backbone_layers: int = 4,
        d_adapter: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        proj_dim: int = 128,
        proj_hidden: int = 512,
        pool: str = "mean",
        dual_attn_pool: bool = False,
    ) -> None:
        super().__init__()
        if pool not in ("mean", "attn"):
            raise ValueError(f"pool must be 'mean' or 'attn', got {pool!r}")
        if dual_attn_pool and pool != "attn":
            raise ValueError(
                f"dual_attn_pool requires pool='attn' (mean pooling is "
                f"parameter-free, so a second mean pool is identical), got pool={pool!r}"
            )
        self.d_backbone = d_backbone
        self.n_backbone_layers = n_backbone_layers
        self.d_adapter = d_adapter
        self.pool = pool
        # When set, two independent half-width attention pools; ``z_m`` is their
        # concatenation (still ``d_adapter``), and the contrastive projection head
        # reads only the second (projection) half so its invariance gradient never
        # touches the first (regression) half. Decouples the contrastive invariance
        # pressure from the latent-regression detail pressure while both share the
        # backbone/token features.
        self.dual_attn_pool = bool(dual_attn_pool)
        # Per-pool width: halved when dual so the concatenation stays ``d_adapter``.
        self.d_pool = d_adapter // 2 if self.dual_attn_pool else d_adapter
        if self.dual_attn_pool:
            if d_adapter % 2 != 0:
                raise ValueError(
                    f"dual_attn_pool concatenates two half-width pools, so d_adapter "
                    f"must be even, got {d_adapter}"
                )
            if self.d_pool % n_heads != 0:
                raise ValueError(
                    f"dual_attn_pool: per-pool width d_adapter//2 ({self.d_pool}) must "
                    f"be divisible by n_heads ({n_heads})"
                )
        in_dim = d_backbone * n_backbone_layers

        self.input_proj = nn.Linear(in_dim, d_adapter)
        self.n_layers = n_layers
        if n_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_adapter,
                nhead=n_heads,
                dim_feedforward=d_adapter * ff_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            # ponytail: linear proj + pool only (no token mixing); PyTorch's
            # TransformerEncoder(num_layers=0) still indexes layers[0] in forward.
            self.encoder = None
        self.norm = nn.LayerNorm(d_adapter)

        # Attention pooling: a single learnable query attends over the valid
        # tokens to produce z_m, replacing the parameter-free masked mean. Left
        # unbuilt in "mean" mode so existing checkpoints keep their exact keys.
        if pool == "attn":
            # Queries live in the (possibly halved) pool width; keys/values are the
            # full-width token features, so kdim/vdim are set explicitly when dual.
            kv = (
                dict(kdim=d_adapter, vdim=d_adapter) if self.dual_attn_pool else {}
            )
            self.pool_query = nn.Parameter(torch.zeros(1, 1, self.d_pool))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(
                self.d_pool, num_heads=n_heads, dropout=dropout, batch_first=True, **kv
            )
            # Second, independent half-width pool feeding the projection head. Left
            # unbuilt unless requested so existing checkpoints keep their keys.
            if self.dual_attn_pool:
                self.proj_pool_query = nn.Parameter(torch.zeros(1, 1, self.d_pool))
                nn.init.trunc_normal_(self.proj_pool_query, std=0.02)
                self.proj_pool_attn = nn.MultiheadAttention(
                    self.d_pool, num_heads=n_heads, dropout=dropout, batch_first=True,
                    kdim=d_adapter, vdim=d_adapter,
                )

        self.proj_head = nn.Sequential(
            nn.Linear(self.d_pool, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
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

    def _attn_pool(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        query: torch.Tensor,
        attn: nn.MultiheadAttention,
    ) -> torch.Tensor:
        b = x.size(0)
        q = query.expand(b, -1, -1)            # [B, 1, D]
        key_padding_mask = mask <= 0           # True = ignore (pad/BOS/EOS)
        pooled, _ = attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return pooled.squeeze(1)               # [B, D]

    def masked_attn_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Learned-query attention pool over valid tokens.

        ``x``: ``[B, T, D]``. ``mask``: ``[B, T]`` in {0, 1} (1 = attend). A single
        learnable query attends over the valid (mask==1) tokens; returns ``[B, D]``.
        Like :meth:`masked_mean` this assumes each row has >= 1 valid token (real
        molecules always do — BOS/EOS are masked out but body tokens remain).
        """
        return self._attn_pool(x, mask, self.pool_query, self.pool_attn)

    def forward(
        self,
        hidden_states_concat: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_projection: bool = False,
        normalize: bool = True,
        return_tokens: bool = False,
        hole_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute ``z_m`` (and optionally the SimCLR projection ``z_p``).

        Args:
            hidden_states_concat: ``[B, T, L*d_backbone]`` — concatenated hidden
                states from L backbone blocks.
            attention_mask: ``[B, T]`` with 1 at real-token positions, 0 at pads.
            return_projection: if True, also return the projection-head output.
            normalize: L2-normalize outputs. LeJEPA needs raw pooled latents
                (``normalize=False``); NT-Xent uses normalized projections.
            return_tokens: if True, return ``(z_m, x)`` where ``x`` is the
                per-token (pre-pool, post-norm) representation ``[B, T, D]`` — for
                position-level objectives (I-JEPA) that pool specific token spans
                themselves. Mutually exclusive with ``return_projection`` (tokens
                win); ``z_m`` still honors ``normalize`` but ``x`` is always raw.

        Returns:
            ``z_m`` ``[B, d_adapter]`` (L2-normalized when ``normalize=True``), or
            ``(z_m, z_p)`` with ``z_p`` ``[B, proj_dim]`` (also normalized when enabled),
            or ``(z_m, x)`` when ``return_tokens``.
        """
        x = self.input_proj(hidden_states_concat)
        if self.encoder is not None:
            # nn.TransformerEncoder expects ``src_key_padding_mask`` where True = ignore.
            key_padding_mask = attention_mask <= 0
            if hole_mask is not None:
                key_padding_mask = key_padding_mask | hole_mask.bool()
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        # Exclude BOS (idx 0) and EOS (last real token) from pooling, per README:
        # "Mean pooling over token positions (excluding special tokens)". We assume
        # callers pass a mask where BOS and EOS positions are already zeroed; the
        # ``stack_views`` helper does that. If they aren't, masked_mean still works.
        proj_pooled = None
        if self.pool == "attn":
            pooled = self.masked_attn_pool(x, attention_mask)  # [B, d_pool]
            if self.dual_attn_pool:
                proj_pooled = self._attn_pool(
                    x, attention_mask, self.proj_pool_query, self.proj_pool_attn
                )
                # z_m concatenates the two half-width pools back to d_adapter, so
                # t-SNE / linear probes see both the regression and projection halves.
                pooled = torch.cat([pooled, proj_pooled], dim=-1)
        else:
            pooled = self.masked_mean(x, attention_mask)
        z_m = (
            torch.nn.functional.normalize(pooled, dim=-1)
            if normalize
            else pooled
        )
        if return_tokens:
            return z_m, x
        if not return_projection:
            return z_m
        # Contrastive head reads only the projection half when dual pooling is on,
        # so its invariance gradient never touches z_m's regression half (see __init__).
        proj_in = proj_pooled if self.dual_attn_pool else pooled
        z_p = self.proj_head(proj_in)
        if normalize:
            z_p = torch.nn.functional.normalize(z_p, dim=-1)
        return z_m, z_p

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""Conditional denoising-JEPA training kernels (encoder, denoiser, loss)."""

from __future__ import annotations

import numbers
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lattice_lab.backbone.discrete_flow import (
    DiscreteFlowBundle,
    _sample_timesteps,
    load_ddit,
    load_discrete_flow,
)

__all__ = [
    "AttentionPool",
    "VAEHead",
    "JEPAEncoder",
    "DenoisingJEPAModel",
    "masked_mean",
    "effective_rank",
    "corrupt_tokens",
    "reconstruction_loss",
    "denoise_logits",
    "encode_pooled_latent",
    "denoising_loss",
    "condition_bypass_gap",
    "assert_condition_active",
    "train_step",
    "build_jepa",
    "build_denoising_jepa",
]


# --------------------------------------------------------------------------- #
# New component 1: attention pooling head (clean tokens → z_s)
# --------------------------------------------------------------------------- #
class AttentionPool(nn.Module):
    """Learned-query attention pool: ``(B, L, D) -> (B, D)``."""

    def __init__(self, dim: int, *, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.query = nn.Parameter(torch.zeros(1, 1, self.dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.dim)

    def forward(
        self, tokens: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Pool ``tokens`` ``[B, L, D]`` to ``[B, D]``.

        ``key_padding_mask`` follows the torch convention: ``[B, L]`` with
        ``True`` at positions to **ignore** (pad). ``None`` attends everything.
        """
        b = tokens.size(0)
        query = self.query.expand(b, -1, -1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(torch.bool)
        attn_out, _ = self.attn(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(attn_out.squeeze(1))


# --------------------------------------------------------------------------- #
# New component 2: VAE reparameterization head (optional latent regularizer)
# --------------------------------------------------------------------------- #
class VAEHead(nn.Module):
    """``z_s [B, D] → mu, log_var``; sample when ``training=True``, else ``mu``."""

    def __init__(self, dim: int, *, free_bits: float = 0.0) -> None:
        super().__init__()
        self.mu = nn.Linear(dim, dim)
        self.log_var = nn.Linear(dim, dim)
        self.free_bits = float(free_bits)

    def reparameterize(self, z: Tensor, *, training: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(z_out, kl, log_var)``; ``log_var`` is ``[B, D]``."""
        mu = self.mu(z)
        log_var = self.log_var(z).clamp(-10.0, 10.0)
        if training:
            z_out = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
        else:
            z_out = mu
        kl_dim = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())  # [B, D]
        kl_dim = kl_dim.mean(dim=0)  # [D] expected KL per latent dimension
        if self.free_bits > 0.0:
            kl_dim = kl_dim.clamp_min(self.free_bits)
        return z_out, kl_dim.mean(), log_var


# --------------------------------------------------------------------------- #
# Shared low-level helpers (reuse backbone conventions)
# --------------------------------------------------------------------------- #
def _additive_key_mask(mask_bool: Tensor) -> Tensor:
    """Build DDiT's additive ``[B, 1, L, L]`` mask from a ``[B, L]`` validity mask.

    ``mask_bool`` is ``True`` at real tokens; the returned mask is ``-inf`` on
    columns whose *key* is padding (matching ``DiscreteFlowEncoder._attn_mask``).
    """
    b, length = mask_bool.shape
    key_invalid = ~mask_bool  # True where the key is pad
    block = key_invalid[:, None, :].expand(b, length, length)
    add = torch.zeros((b, length, length), dtype=torch.float32, device=mask_bool.device)
    add = add.masked_fill(block, float("-inf"))
    return add.unsqueeze(1)


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``x`` ``[B, L, D]`` over real positions (``mask`` ``[B, L]``) → ``[B, D]``."""
    m = mask.to(x.dtype).unsqueeze(-1)
    s = (x * m).sum(dim=1)
    denom = m.sum(dim=1).clamp_min(1e-6)
    return s / denom


# --------------------------------------------------------------------------- #
# Encoder wrapper (reused backbone → pooled molecule latent z_s)
# --------------------------------------------------------------------------- #
class JEPAEncoder(nn.Module):
    """``backbone → [B, L, D] hiddens → AttentionPool → z_s [B, D]``."""

    def __init__(
        self,
        backbone: nn.Module,
        pool: AttentionPool,
        *,
        encode_time: float = 1.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.register_buffer("encode_time", torch.tensor(float(encode_time)))

    def _time_vec(self, b: int, device: torch.device, t) -> Tensor:
        if t is None:
            return self.encode_time.to(device).expand(b)
        if torch.is_tensor(t):
            return t.to(device).float()
        return torch.full((b,), float(t), device=device)

    def latents(self, ids: Tensor, mask: Tensor, t=None) -> Tensor:
        """Per-position contextual latents ``[B, L, D]`` (clean, ``t = 1`` default)."""
        ids = ids.long()
        mask_bool = mask.to(torch.bool)
        attn = _additive_key_mask(mask_bool)
        t_vec = self._time_vec(ids.size(0), ids.device, t)
        out = self.backbone(ids, t_vec, attn_mask=attn, conds=None, return_hidden=True)
        # return_hidden=True → (logits, hidden_states[B, L, D]).
        return out[1] if isinstance(out, tuple) else out

    def forward(self, ids: Tensor, mask: Tensor, t=None) -> Tensor:
        """Pooled molecule latent ``z_s`` ``[B, D]``."""
        h = self.latents(ids, mask, t=t)
        return self.pool(h, key_padding_mask=~mask.to(torch.bool))


class _DenoisingCore(Protocol):
    encoder: JEPAEncoder
    denoiser: nn.Module
    pad_id: int
    vocab_size: int
    token_id_min: int
    vae_head: VAEHead | None
    training: bool


def encode_pooled_latent(
    model: _DenoisingCore,
    ids: Tensor,
    mask: Tensor,
    *,
    training: bool | None = None,
) -> Tensor:
    """Encoder pool output, then VAE μ (+ noise when ``training``)."""
    z = model.encoder(ids, mask)
    if model.vae_head is not None:
        if training is None:
            training = model.training
        z, _, _ = model.vae_head.reparameterize(z, training=training)
    return z


def _encode_z_with_vae(
    model: _DenoisingCore,
    ids: Tensor,
    mask: Tensor,
    *,
    training: bool | None = None,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Pooled latent plus optional ``(kl, log_var)`` when a VAE head is attached."""
    z = model.encoder(ids, mask)
    kl: Tensor | None = None
    log_var: Tensor | None = None
    if model.vae_head is not None:
        if training is None:
            training = model.training
        z, kl, log_var = model.vae_head.reparameterize(z, training=training)
    return z, kl, log_var


def _logvar_metrics(log_var: Tensor) -> dict[str, float]:
    with torch.no_grad():
        return {
            "logvar_mean": float(log_var.mean()),
            "logvar_std": float(log_var.std(unbiased=False)),
            "logvar_exp_mean": float(log_var.exp().mean()),
        }


def denoise_forward(
    model: _DenoisingCore,
    x_t: Tensor,
    mask: Tensor,
    t: Tensor,
    conds: Tensor,
    *,
    return_hidden: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    x_t = x_t.long()
    attn = _additive_key_mask(mask.to(torch.bool))
    out = model.denoiser(
        x_t, t.float(), attn_mask=attn, conds=conds, return_hidden=return_hidden
    )
    if return_hidden:
        return out[0], out[1]
    return out[0] if isinstance(out, tuple) else out


def denoise_logits(
    model: _DenoisingCore, x_t: Tensor, mask: Tensor, t: Tensor, conds: Tensor
) -> Tensor:
    """Conditional denoiser logits ``[B, L, vocab]`` for corrupted ``x_t``."""
    return denoise_forward(model, x_t, mask, t, conds, return_hidden=False)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Tiny fresh-build model (unit tests only)
# --------------------------------------------------------------------------- #
class DenoisingJEPAModel(nn.Module):
    """Encoder + conditional denoiser + token metadata (tests / smoke builds)."""

    def __init__(
        self,
        encoder: JEPAEncoder,
        denoiser: nn.Module,
        *,
        pad_id: int,
        vocab_size: int,
        token_id_min: int,
        bundle: DiscreteFlowBundle | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.denoiser = denoiser
        self.pad_id = int(pad_id)
        self.vocab_size = int(vocab_size)
        self.token_id_min = int(token_id_min)
        self.bundle = bundle
        self.build_config: dict | None = None
        self.vae_head: VAEHead | None = None

    def denoise_logits(
        self, x_t: Tensor, mask: Tensor, t: Tensor, conds: Tensor
    ) -> Tensor:
        return denoise_logits(self, x_t, mask, t, conds)


# --------------------------------------------------------------------------- #
# effective_rank
# --------------------------------------------------------------------------- #
def effective_rank(z: Tensor, *, eps: float = 1e-12) -> float:
    """Entropy-based effective rank of ``z`` ``[B, D]`` covariance."""
    z = z.detach().to(torch.float64)
    if z.ndim != 2 or z.shape[0] < 2:
        return float("nan")
    n = z.shape[0]
    centered = z - z.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    eig = (singular**2) / max(n - 1, 1)
    total = float(eig.sum())
    if total <= eps:
        return 0.0
    probs = eig / total
    active = probs > eps
    return float(torch.exp(-(probs[active] * probs[active].log()).sum()))


# --------------------------------------------------------------------------- #
# Corruption (reuse the discrete-flow noise()) — returns the noised-position mask
# --------------------------------------------------------------------------- #
def corrupt_tokens(
    ids: Tensor,
    mask: Tensor,
    *,
    vocab_size: int,
    token_id_min: int,
    pad_id: int,
    t: Optional[Tensor | float | tuple[float, float]] = None,
    t_cap: float = 1e-3,
    path_power: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Uniform-source corruption via the discrete-flow path; return
    ``(x_t, t, noised_mask)``.

    ``x0 ~ U[token_id_min, vocab)`` (uniform source). Each *real* position is
    replaced by its source token with probability ``sigma_t = 1 - t**path_power``
    (so ``t → 0`` ≈ all-noise, ``t = 1`` = clean). ``noised_mask`` ``[B, L]`` is
    the boolean mask of positions actually replaced (real positions only) — the
    reconstruction loss is taken only there. ``t`` may be ``None`` (discrete-flow
    sampling), a scalar (fixed), a ``(lo, hi)`` pair (uniform per-sample), or a
    tensor.
    """
    b = ids.size(0)
    device = ids.device
    if t is None:
        t = _sample_timesteps(b, device, t_cap=t_cap)
    elif torch.is_tensor(t):
        t = t.to(device)
    elif isinstance(t, numbers.Number):
        t = torch.full((b,), float(t), device=device)
    else:
        seq = list(t)
        if len(seq) != 2:
            raise ValueError(f"corrupt time range must be a (lo, hi) pair, got {seq!r}")
        lo, hi = float(seq[0]), float(seq[1])
        t = torch.rand(b, device=device) * (hi - lo) + lo

    valid = mask.to(torch.bool)
    ids = ids.long()
    x0 = torch.randint(int(token_id_min), int(vocab_size), ids.shape, device=device)
    # Bernoulli(sigma_t) draw of which positions take the source token.
    sigma_t = 1.0 - t#.pow(float(path_power))
    src = torch.rand(ids.shape, device=device) < sigma_t.unsqueeze(-1)
    src = src & valid
    x_t = torch.where(src, x0, ids).masked_fill(~valid, int(pad_id))
    return x_t, t, src


# --------------------------------------------------------------------------- #
# Reconstruction loss (pure generative CE at the noised positions)
# --------------------------------------------------------------------------- #
def reconstruction_loss(
    logits: Tensor,
    clean_ids: Tensor,
    noised_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Token CE at noised positions only; ``0`` when nothing was noised."""
    targets = clean_ids.long()
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")  # [B, L]
    w = noised_mask.to(ce.dtype)
    denom = w.sum().clamp_min(1.0)
    loss = (ce * w).sum() / denom
    return loss, ce


# --------------------------------------------------------------------------- #
# denoising_loss
# --------------------------------------------------------------------------- #
def denoising_loss(
    model: _DenoisingCore,
    batch: tuple[Tensor, Tensor],
    *,
    batch_b: tuple[Tensor, Tensor] | None = None,
    align_lambda: float = 0.0,
    corrupt_t: Optional[Tensor | float | tuple[float, float]] = (0.1, 0.6),
    t_cap: float = 1e-3,
    path_power: float = 1.0,
    compute_rank: bool = True,
    return_outputs: bool = False,
    cond_noise_scale: float = 0.0,
    latent_consistency_lambda: float = 0.0,
):
    """Return ``(loss, metrics)``; optional ``(z_s, logits, noised, ids, kl), terms``."""
    ids, mask = batch
    ids = ids.long()

    z_s, kl, log_var = _encode_z_with_vae(model, ids, mask)

    x_corrupt, t, noised = corrupt_tokens(
        ids,
        mask,
        vocab_size=model.vocab_size,
        token_id_min=model.token_id_min,
        pad_id=model.pad_id,
        t=corrupt_t,
        t_cap=t_cap,
        path_power=path_power,
    )

    z_s_cond = z_s
    if cond_noise_scale > 0.0 and model.training:
        t_c = t.unsqueeze(-1)
        z_s_cond = (1.0 - t_c) * z_s + t_c * torch.randn_like(z_s)
    need_latent = float(latent_consistency_lambda) > 0.0
    if need_latent:
        logits, hidden = denoise_forward(
            model, x_corrupt, mask, t, z_s_cond, return_hidden=True
        )
    else:
        logits = denoise_forward(model, x_corrupt, mask, t, z_s_cond)

    recon, _ = reconstruction_loss(logits, ids, noised)
    loss = recon

    align: Optional[Tensor] = None
    if batch_b is not None and align_lambda > 0.0:
        ids_b, mask_b = batch_b
        ids_b = ids_b.long()
        z_b, _, _ = _encode_z_with_vae(model, ids_b, mask_b)
        align = (1.0 - F.cosine_similarity(z_s, z_b, dim=-1)).mean()
        loss = recon + float(align_lambda) * align

    latent: Optional[Tensor] = None
    if need_latent and noised.any():
        valid = mask.to(torch.bool)
        pool_kpm = ~(noised & valid)
        z_hat = model.encoder.pool(hidden, key_padding_mask=pool_kpm)
        latent = (1.0 - F.cosine_similarity(z_hat, z_s.detach(), dim=-1)).mean()
        loss = loss + float(latent_consistency_lambda) * latent

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        n_noised = noised.float().sum().clamp_min(1.0)
        recon_acc = float(((pred == ids) & noised).float().sum() / n_noised)
        mask_bool = mask.to(torch.bool)
    metrics = {
        "loss": float(loss.detach()),
        "recon": float(recon.detach()),
        "recon_acc": recon_acc,
        "noised_frac": float(noised.float().sum() / mask_bool.float().sum().clamp_min(1.0)),
        "rank_s": effective_rank(z_s) if compute_rank else float("nan"),
    }
    if align is not None:
        metrics["align"] = float(align.detach())
        metrics["align_cos"] = float(1.0 - align.detach())
    if latent is not None:
        metrics["latent"] = float(latent.detach())
        metrics["latent_cos"] = float(1.0 - latent.detach())
    if log_var is not None:
        metrics.update(_logvar_metrics(log_var))
    if return_outputs:
        terms: dict[str, Tensor] = {"recon": recon}
        if align is not None:
            terms["align"] = float(align_lambda) * align
        if latent is not None:
            terms["latent"] = float(latent_consistency_lambda) * latent
        return loss, metrics, (z_s, logits, noised, ids, kl), terms
    return loss, metrics


# --------------------------------------------------------------------------- #
# condition_bypass_gap
# --------------------------------------------------------------------------- #
@torch.no_grad()
def condition_bypass_gap(
    model: _DenoisingCore,
    batch: tuple[Tensor, Tensor],
    *,
    corrupt_t: Optional[Tensor | float | tuple[float, float]] = 0.1,
    t_cap: float = 1e-3,
    path_power: float = 1.0,
) -> dict[str, float]:
    """Return ``{recon_real, recon_zeroed, gap}`` with ``gap = recon_zeroed - recon_real``."""
    model.eval()
    ids, mask = batch
    ids = ids.long()
    z_s = encode_pooled_latent(model, ids, mask, training=False)
    x_corrupt, t, noised = corrupt_tokens(
        ids,
        mask,
        vocab_size=model.vocab_size,
        token_id_min=model.token_id_min,
        pad_id=model.pad_id,
        t=corrupt_t,
        t_cap=t_cap,
        path_power=path_power,
    )
    logits_real = denoise_logits(model, x_corrupt, mask, t, conds=z_s)
    logits_zero = denoise_logits(model, x_corrupt, mask, t, conds=torch.zeros_like(z_s))
    recon_real, _ = reconstruction_loss(logits_real, ids, noised)
    recon_zero, _ = reconstruction_loss(logits_zero, ids, noised)
    return {
        "recon_real": float(recon_real),
        "recon_zeroed": float(recon_zero),
        "gap": float(recon_zero - recon_real),
    }


def assert_condition_active(
    model: _DenoisingCore,
    batch: tuple[Tensor, Tensor],
    *,
    margin: float = 0.1,
    corrupt_t: Optional[Tensor | float | tuple[float, float]] = 0.1,
    hard_fail: bool = True,
) -> dict[str, float]:
    stats = condition_bypass_gap(model, batch, corrupt_t=corrupt_t)
    if stats["gap"] < float(margin) and hard_fail:
        raise RuntimeError(
            f"condition-bypass: gap={stats['gap']:.4f} < margin={margin} "
            f"(recon_real={stats['recon_real']:.4f}, "
            f"recon_zeroed={stats['recon_zeroed']:.4f})"
        )
    return stats


# --------------------------------------------------------------------------- #
# train_step
# --------------------------------------------------------------------------- #
def train_step(
    model: _DenoisingCore,
    opt: torch.optim.Optimizer,
    batch: tuple[Tensor, Tensor],
    *,
    batch_b: tuple[Tensor, Tensor] | None = None,
    align_lambda: float = 0.0,
    corrupt_t: Optional[Tensor | float | tuple[float, float]] = (0.1, 0.6),
    t_cap: float = 1e-3,
    path_power: float = 1.0,
) -> dict[str, float]:
    """One optimization step of the conditional-denoising objective."""
    model.train()
    loss, metrics = denoising_loss(
        model,
        batch,
        batch_b=batch_b,
        align_lambda=align_lambda,
        corrupt_t=corrupt_t,
        t_cap=t_cap,
        path_power=path_power,
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return metrics


def _wire_denoising_jepa(
    parent: nn.Module,
    *,
    encoder: JEPAEncoder,
    denoiser: nn.Module,
    bundle: DiscreteFlowBundle,
    token_id_min: int,
    build_config: dict[str, Any],
) -> None:
    """Register encoder/denoiser/token metadata on ``parent`` (module or model)."""
    parent.encoder = encoder
    parent.denoiser = denoiser
    parent.pad_id = int(bundle.pad_id)
    parent.vocab_size = int(bundle.vocab_size)
    parent.token_id_min = int(token_id_min)
    parent.bundle = bundle
    parent.build_config = dict(build_config)
    if not hasattr(parent, "vae_head"):
        parent.vae_head = None


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_jepa(
    *,
    vocab_size: int,
    hidden_size: int = 768,
    n_heads: int = 12,
    n_layer: int = 12,
    pool_heads: int = 8,
    pool_dropout: float = 0.0,
    dropout: float = 0.1,
    encode_time: float = 1.0,
    pad_id: int = 3,
    token_id_min: int = 4,
    ckpt_path: Optional[str] = None,
    device: str | torch.device = "cpu",
) -> DenoisingJEPAModel:
    """Wire a conditional-denoising model from fresh (or warm-started) DDiTs."""
    backbone, _ = load_ddit(
        ckpt_path=ckpt_path,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_heads,
        n_embd=hidden_size,
        dropout=dropout,
        n_conds=0,
    )
    denoiser, _ = load_ddit(
        ckpt_path=ckpt_path,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_heads,
        n_embd=hidden_size,
        dropout=dropout,
        n_conds=hidden_size,
    )
    pool = AttentionPool(hidden_size, num_heads=pool_heads, dropout=pool_dropout)
    encoder = JEPAEncoder(backbone, pool, encode_time=encode_time)
    model = DenoisingJEPAModel(
        encoder,
        denoiser,
        pad_id=pad_id,
        vocab_size=vocab_size,
        token_id_min=token_id_min,
    )
    model.to(device)
    return model


def build_denoising_jepa(
    *,
    ckpt_path: Optional[str],
    tokenizer_path: str,
    pool_heads: int = 8,
    pool_dropout: float = 0.0,
    encode_time: float = 1.0,
    freeze_backbone: bool = True,
    token_id_min: int = 4,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 768,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
    parent: nn.Module | None = None,
) -> DenoisingJEPAModel | None:
    """Build encoder + denoiser; register on ``parent`` or return a test model."""
    bundle = load_discrete_flow(
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        freeze_backbone=freeze_backbone,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        n_conds=0,
        device=device,
    )
    hidden = int(bundle.n_embd)
    denoiser, _ = load_ddit(
        ckpt_path=ckpt_path,
        vocab_size=bundle.vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        n_conds=hidden,
        force_n_conds=True,
    )
    denoiser.to(device)
    pool = AttentionPool(hidden, num_heads=pool_heads, dropout=pool_dropout)
    encoder = JEPAEncoder(bundle.model, pool, encode_time=encode_time)
    build_config = {
        "tokenizer_path": tokenizer_path,
        "pool_heads": int(pool_heads),
        "encode_time": float(encode_time),
        "token_id_min": int(token_id_min),
        "n_layer": int(bundle.n_layer),
        "n_head": int(n_head),
        "n_embd": hidden,
        "dropout": float(dropout),
        "freeze_backbone": bool(freeze_backbone),
    }
    if parent is not None:
        _wire_denoising_jepa(
            parent,
            encoder=encoder,
            denoiser=denoiser,
            bundle=bundle,
            token_id_min=token_id_min,
            build_config=build_config,
        )
        parent.to(device)
        return None
    model = DenoisingJEPAModel(
        encoder,
        denoiser,
        pad_id=bundle.pad_id,
        vocab_size=bundle.vocab_size,
        token_id_min=token_id_min,
        bundle=bundle,
    )
    model.build_config = build_config
    model.to(device)
    return model

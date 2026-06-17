"""Conditional denoising-JEPA for the SMILES (DDiT) encoder.

An LLM-JEPA-style scheme where a learned, *pooled* molecule latent conditions a
denoiser that reconstructs a corrupted copy of the string. The objective is a
**pure generative (token) loss** — cross-entropy to the ground-truth tokens at
the corrupted positions — so there is no representation matching, and therefore
**no EMA teacher / no stop-gradient is needed** (the targets are real tokens,
not a learned representation, so there is nothing to collapse onto):

    z_s = encoder(clean,    t=1)               # pooled molecule latent (grad)
    x_t = corrupt(clean,    t=corrupt_t)        # uniform-source substitution
    logits = denoiser(x_t,  t=corrupt_t | z_s)  # z_s is the conditioning input
    loss = CE(logits, clean)  over the NOISED positions only

The encoder is trained *through* the reconstruction: ``z_s`` must carry enough
about the molecule for the denoiser to fill the noised positions back in. The
remaining failure mode is therefore not representation collapse but an **inert
anchor** (the denoiser reconstructs from the visible context and ignores
``z_s``). :func:`condition_bypass_gap` / :func:`assert_condition_active` guard
against that, and :func:`effective_rank` (``z_s``) is logged as a secondary
tripwire.

Two components are new relative to the reused discrete-flow stack:
:class:`AttentionPool` (clean tokens → ``z_s``) and a separate conditional
:class:`~lattice_lab.backbone.ddit.model_ddit.DDiT` denoiser (``n_conds = D``).
Everything else is reused: the backbone, the tokenizer
(:func:`~lattice_lab.backbone.discrete_flow.load_discrete_flow`), and the
corruption math (:func:`~lattice_lab.backbone.discrete_flow._sample_path` /
``_sample_timesteps``). The encoder backbone (``n_conds = 0``) and the denoiser
(``n_conds = D``) are necessarily distinct DDiTs.
"""

from __future__ import annotations

import numbers
from typing import Any, Optional

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
    "JEPAStudent",
    "masked_mean",
    "effective_rank",
    "corrupt_tokens",
    "reconstruction_loss",
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
    """Learned-query attention pool: ``(B, L, D) -> (B, D)``.

    A single learned query attends over the token axis with
    :class:`torch.nn.MultiheadAttention`, honoring a key-padding mask, and the
    pooled vector is LayerNorm'd. The pool itself carries **no positional
    component**, so it is permutation-invariant in the token axis — order is
    already baked into the backbone's contextual hidden states it consumes.

    This is the encoder head that produces the molecule latent ``z_s`` used as
    the denoiser's conditioning input (it is trained through reconstruction).
    """

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
    """Beta-VAE reparameterization head: ``z_s [B, D] → (mu, log_var) [B, D]``.

    During training samples ``z ~ N(mu, sigma)`` (reparameterization trick) so
    the KL penalty can flow gradients back through the encoder. During eval
    returns ``mu`` deterministically. Attach to :class:`JEPAStudent` as
    ``student.vae_head``; :func:`denoising_loss` applies it automatically when
    the attribute is set.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(dim, dim)
        self.log_var = nn.Linear(dim, dim)

    def reparameterize(self, z: Tensor, *, training: bool) -> tuple[Tensor, Tensor]:
        """Return ``(z_out, kl)`` where ``kl = -0.5 * mean(1 + lv - mu² - exp(lv))``."""
        mu = self.mu(z)
        log_var = self.log_var(z).clamp(-10.0, 10.0)
        if training:
            z_out = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
        else:
            z_out = mu
        kl = -0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp()).mean()
        return z_out, kl


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
    """``backbone (DDiT) → contextual hiddens [B, L, D] → AttentionPool → z_s``.

    :meth:`forward` returns the pooled molecule latent ``z_s`` ``[B, D]`` used as
    the denoiser's conditioning input. The encoder always runs on the **clean**
    string at the clean endpoint ``encode_time`` (``t = 1``); :meth:`latents`
    exposes the per-position hiddens if needed for diagnostics.
    """

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


# --------------------------------------------------------------------------- #
# Student container (encoder + conditional denoiser)
# --------------------------------------------------------------------------- #
class JEPAStudent(nn.Module):
    """Holds the trainable encoder, the conditional denoiser, and token metadata.

    ``JEPAStudent.parameters()`` is exactly what the optimizer is built over.
    There is no teacher: the loss is generative (CE to real tokens).
    """

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
        self.vae_head: VAEHead | None = None  # set by DenoisingJEPAModule when kl_beta > 0

    def denoise_logits(
        self, x_t: Tensor, mask: Tensor, t: Tensor, conds: Tensor
    ) -> Tensor:
        """Conditional denoiser logits ``[B, L, vocab]`` for corrupted ``x_t``."""
        x_t = x_t.long()
        attn = _additive_key_mask(mask.to(torch.bool))
        out = self.denoiser(x_t, t.float(), attn_mask=attn, conds=conds)
        return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------- #
# Collapse / informativeness tripwire
# --------------------------------------------------------------------------- #
def effective_rank(z: Tensor, *, eps: float = 1e-12) -> float:
    """Entropy-based effective rank ``exp(-Σ p_i log p_i)`` of ``z``'s covariance.

    ``z`` is ``[B, D]``. Mirrors
    :func:`lattice_lab.training.ssl_val_probes.embedding_covariance_rank` (the
    effective component) but stays in torch so it can be logged every step. A
    value collapsing toward ``1`` means ``z_s`` carries almost no per-molecule
    information (an inert / degenerate anchor).
    """
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
    sigma_t = 1.0 - t.pow(float(path_power))
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
    """Token CE to the clean ids, averaged over the noised positions only.

    Returns ``(loss, per_position_ce)``. Clean (un-noised) positions are visible
    in the denoiser input, so scoring them is trivial and would dilute the
    signal that forces ``z_s`` to be used — the loss is restricted to the
    positions that were actually corrupted. ``loss`` is ``0`` when nothing was
    noised.
    """
    targets = clean_ids.long()
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")  # [B, L]
    w = noised_mask.to(ce.dtype)
    denom = w.sum().clamp_min(1.0)
    loss = (ce * w).sum() / denom
    return loss, ce


# --------------------------------------------------------------------------- #
# Conditional denoising objective (no teacher / no EMA / no stop-grad)
# --------------------------------------------------------------------------- #
def denoising_loss(
    student: JEPAStudent,
    batch: tuple[Tensor, Tensor],
    *,
    batch_b: tuple[Tensor, Tensor] | None = None,
    align_lambda: float = 0.0,
    align_corrupt_t: Optional[Tensor | float | tuple[float, float]] = None,
    corrupt_t: Optional[Tensor | float | tuple[float, float]] = (0.1, 0.6),
    t_cap: float = 1e-3,
    path_power: float = 1.0,
    compute_rank: bool = True,
    return_outputs: bool = False,
    cond_noise_scale: float = 0.0,
):
    """Forward the conditional-denoising objective; return ``(loss, metrics)``.

    Stages: (1) encoder pools the **clean** string at ``t = 1`` → ``z_s``;
    (2) corrupt the clean ids (uniform source) → ``x_t`` at ``corrupt_t``;
    (3) denoiser reconstructs, conditioned on ``z_s`` → ``logits``;
    (4) CE to the clean tokens at the noised positions. No stop-gradient is
    needed (targets are ground-truth tokens), so gradient flows into both the
    encoder (via ``z_s``) and the denoiser.

    View-invariance regularizer (optional): when ``batch_b`` (a second
    augmentation — e.g. a different fragment shuffle — of the *same* molecules)
    is given and ``align_lambda > 0``, add ``align_lambda * mean(1 - cos(z_s,
    z_s_b))``. The generative term keeps ``z_s`` informative (so the alignment
    cannot collapse to a constant), while the alignment pulls the two views'
    codes together so ``z_s`` encodes order-invariant molecule identity rather
    than reconstruction-specific token detail (keeps the embedding semantic).

    ``align_corrupt_t`` makes the positive pair *hard*: the second view is
    corrupted (uniform source) at that flow time and encoded at it before being
    aligned to the clean anchor ``z_s``. Fragment-shuffle alone is nearly free
    for a permutation-invariant pool (``align_cos`` saturates at 1 fast and stops
    regularizing); requiring invariance to genuine token corruption forces the
    encoder to capture molecule-level structure. ``None`` keeps the clean
    (``t = 1``) second view.

    ``compute_rank`` toggles the per-call ``effective_rank(z_s)`` SVD.
    ``return_outputs`` also returns ``(z_s, logits, noised_mask, clean_ids)`` for
    accuracy / retrieval diagnostics.
    """
    ids, mask = batch
    ids = ids.long()

    # Stage 1 — encoder pools the clean string at the clean endpoint (t = 1).
    z_s = student.encoder(ids, mask)

    # Optional VAE reparameterization (beta-VAE regularizer on z_s).
    kl: Optional[Tensor] = None
    if student.vae_head is not None:
        z_s, kl = student.vae_head.reparameterize(z_s, training=student.training)

    # Stage 2 — uniform-source corruption (+ which positions were noised).
    x_corrupt, t, noised = corrupt_tokens(
        ids,
        mask,
        vocab_size=student.vocab_size,
        token_id_min=student.token_id_min,
        pad_id=student.pad_id,
        t=corrupt_t,
        t_cap=t_cap,
        path_power=path_power,
    )

    # Stage 3 — conditional denoiser reconstructs, conditioned on z_s.
    # t-scaled noise on the conditioning signal: at t≈0 (heavy corruption) z_s
    # is clean and load-bearing; at t≈1 (easy task) z_s is noisy so the denoiser
    # cannot lean on it, preventing easy-task gradients from eroding the pool.
    # FP distillation / SIGReg always see the clean z_s above.
    z_s_cond = z_s
    if cond_noise_scale > 0.0 and student.training:
        z_s_cond = z_s + t.unsqueeze(-1) * cond_noise_scale * torch.randn_like(z_s)
    logits = student.denoise_logits(x_corrupt, mask, t, conds=z_s_cond)

    # Stage 4 — pure generative CE at the noised positions.
    recon, _ = reconstruction_loss(logits, ids, noised)
    loss = recon

    align: Optional[Tensor] = None
    if batch_b is not None and align_lambda > 0.0:
        ids_b, mask_b = batch_b
        ids_b = ids_b.long()
        if align_corrupt_t is not None:
            # Hard positive: corrupt the second view and encode it at its noise
            # time, so invariance demands molecule-level (not surface) features.
            x_b, t_b, _ = corrupt_tokens(
                ids_b,
                mask_b,
                vocab_size=student.vocab_size,
                token_id_min=student.token_id_min,
                pad_id=student.pad_id,
                t=align_corrupt_t,
                t_cap=t_cap,
                path_power=path_power,
            )
            z_b = student.encoder(x_b, mask_b, t=t_b)
        else:
            z_b = student.encoder(ids_b, mask_b)
        align = (1.0 - F.cosine_similarity(z_s, z_b, dim=-1)).mean()
        loss = recon + float(align_lambda) * align

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
        metrics["align"] = float(align.detach())          # 1 - cos (lower better)
        metrics["align_cos"] = float(1.0 - align.detach())  # cos(z_s_a, z_s_b)
    if return_outputs:
        return loss, metrics, (z_s, logits, noised, ids, kl)
    return loss, metrics


# --------------------------------------------------------------------------- #
# Condition-bypass diagnostic (is z_s actually used, or inert?)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def condition_bypass_gap(
    student: JEPAStudent,
    batch: tuple[Tensor, Tensor],
    *,
    corrupt_t: Optional[Tensor | float | tuple[float, float]] = 0.1,
    t_cap: float = 1e-3,
    path_power: float = 1.0,
) -> dict[str, float]:
    """Reconstruct once with the real ``z_s`` and once with ``z_s`` zeroed.

    Returns ``{recon_real, recon_zeroed, gap}`` where ``gap = recon_zeroed -
    recon_real``. A positive gap means the denoiser genuinely uses the anchor; a
    gap near zero means ``z_s`` is **inert** (the corruption is too weak, or the
    context alone is enough) and the encoder is learning nothing. Run at a fixed,
    strong corruption (``corrupt_t`` small) so the noised positions are hard.
    """
    student.eval()
    ids, mask = batch
    ids = ids.long()
    z_s = student.encoder(ids, mask)
    x_corrupt, t, noised = corrupt_tokens(
        ids,
        mask,
        vocab_size=student.vocab_size,
        token_id_min=student.token_id_min,
        pad_id=student.pad_id,
        t=corrupt_t,
        t_cap=t_cap,
        path_power=path_power,
    )
    logits_real = student.denoise_logits(x_corrupt, mask, t, conds=z_s)
    logits_zero = student.denoise_logits(x_corrupt, mask, t, conds=torch.zeros_like(z_s))
    recon_real, _ = reconstruction_loss(logits_real, ids, noised)
    recon_zero, _ = reconstruction_loss(logits_zero, ids, noised)
    return {
        "recon_real": float(recon_real),
        "recon_zeroed": float(recon_zero),
        "gap": float(recon_zero - recon_real),
    }


def assert_condition_active(
    student: JEPAStudent,
    batch: tuple[Tensor, Tensor],
    *,
    margin: float = 0.1,
    corrupt_t: Optional[Tensor | float | tuple[float, float]] = 0.1,
    hard_fail: bool = True,
) -> dict[str, float]:
    """Assert the conditioning gap exceeds ``margin`` (loud if not).

    Raises :class:`RuntimeError` when ``hard_fail`` and the gap is below
    ``margin`` (the anchor is inert — do not let a run proceed silently). When
    ``hard_fail`` is ``False`` it returns the gap dict so a caller can warn.
    """
    stats = condition_bypass_gap(student, batch, corrupt_t=corrupt_t)
    if stats["gap"] < float(margin) and hard_fail:
        raise RuntimeError(
            f"condition-bypass: gap={stats['gap']:.4f} < margin={margin} "
            f"(recon_real={stats['recon_real']:.4f}, "
            f"recon_zeroed={stats['recon_zeroed']:.4f}). z_s is inert — increase "
            f"corruption (lower corrupt_t) so reconstruction must rely on z_s."
        )
    return stats


# --------------------------------------------------------------------------- #
# One optimization step (no teacher update)
# --------------------------------------------------------------------------- #
def train_step(
    student: JEPAStudent,
    opt: torch.optim.Optimizer,
    batch: tuple[Tensor, Tensor],
    *,
    batch_b: tuple[Tensor, Tensor] | None = None,
    align_lambda: float = 0.0,
    align_corrupt_t: Optional[Tensor | float | tuple[float, float]] = None,
    corrupt_t: Optional[Tensor | float | tuple[float, float]] = (0.1, 0.6),
    t_cap: float = 1e-3,
    path_power: float = 1.0,
) -> dict[str, float]:
    """One optimization step of the conditional-denoising objective."""
    student.train()
    loss, metrics = denoising_loss(
        student,
        batch,
        batch_b=batch_b,
        align_lambda=align_lambda,
        align_corrupt_t=align_corrupt_t,
        corrupt_t=corrupt_t,
        t_cap=t_cap,
        path_power=path_power,
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return metrics


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
) -> JEPAStudent:
    """Wire a conditional-denoising :class:`JEPAStudent` from fresh (or
    warm-started) DDiTs.

    Two DDiTs: the encoder backbone (``n_conds = 0``) and the conditional
    denoiser (``n_conds = hidden_size`` so it accepts ``z_s``). ``encode_time =
    1`` is the clean endpoint the encoder runs at.
    """
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
    student = JEPAStudent(
        encoder,
        denoiser,
        pad_id=pad_id,
        vocab_size=vocab_size,
        token_id_min=token_id_min,
    )
    student.to(device)
    return student


def build_denoising_jepa(
    *,
    ckpt_path: Optional[str],
    tokenizer_path: str,
    pool_heads: int = 8,
    pool_dropout: float = 0.0,
    # The encoder runs on the *clean* string, so its flow time is the clean
    # endpoint t = 1 (sigma = 1 - t**power = 0).
    encode_time: float = 1.0,
    # The encoder backbone may be frozen (adapter-style: only the pool + denoiser
    # train); the denoiser is always trainable.
    freeze_backbone: bool = True,
    token_id_min: int = 4,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 768,
    dropout: float = 0.1,
    device: str | torch.device = "cpu",
) -> JEPAStudent:
    """Hydra entrypoint: build a conditional-denoising :class:`JEPAStudent` from
    the reused discrete-flow parts (tokenizer + two ``DDiT`` backbones).

    The encoder backbone is loaded via :func:`load_discrete_flow` (so the student
    carries a tokenizer :class:`DiscreteFlowBundle`) and may be frozen. The
    conditional denoiser is a second ``DDiT`` warm-started from the same
    checkpoint with ``n_conds = hidden`` (its fresh conditioning projection is
    the only un-restored part); it is always trainable.
    """
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
    # Force n_conds=hidden even when warm-starting from an unconditional (n_conds
    # = 0) pretrained DDiT: the conditioning projection stays freshly init'd.
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
    student = JEPAStudent(
        encoder,
        denoiser,
        pad_id=bundle.pad_id,
        vocab_size=bundle.vocab_size,
        token_id_min=token_id_min,
        bundle=bundle,
    )
    student.to(device)
    student.build_config = {
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
    return student

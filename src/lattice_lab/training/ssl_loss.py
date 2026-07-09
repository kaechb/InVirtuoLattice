"""SSL losses for paired molecule views.

* **NT-Xent** — SimCLR-style contrastive loss (default).
* **LeJEPA** — invariance loss + SIGReg isotropy regularizer:
  ``L = (1 - lambda) * L_inv + lambda * L_sigreg`` (convex combination, see
  ``galilai-group/lejepa`` and ``LeJEPALoss``'s docstring). Every view of a
  molecule (intact shuffles + masked) is pulled directly toward that molecule's
  intact center (no predictor); SIGReg keeps the target batch isotropic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


class NTXentLoss(nn.Module):
    """Symmetric NT-Xent (SimCLR) loss with cosine similarity.

    Args:
        temperature: softmax temperature. SimCLR uses 0.1–0.5; default 0.1.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute the symmetric NT-Xent loss.

        Both inputs must be L2-normalized along the last dim.
        """
        if z_a.shape != z_b.shape:
            raise ValueError(f"z_a/z_b shape mismatch: {z_a.shape} vs {z_b.shape}")
        b = z_a.shape[0]
        device = z_a.device

        # 2B x D stack, with positive pairs at offset B.
        z = torch.cat([z_a, z_b], dim=0)
        sim = z @ z.t() / self.temperature  # [2B, 2B]

        # Mask self-similarity.
        eye = torch.eye(2 * b, dtype=torch.bool, device=device)
        sim.masked_fill_(eye, float("-inf"))

        # Targets: row i has positive at i+B (mod 2B).
        targets = torch.arange(2 * b, device=device)
        targets = (targets + b) % (2 * b)

        return torch.nn.functional.cross_entropy(sim, targets)


class SigLIPLoss(nn.Module):
    """Sigmoid contrastive loss (SigLIP, arXiv:2303.15343).

    Drop-in alternative to :class:`NTXentLoss` for a batch of paired views. Each
    of the ``B*B`` view-a/view-b pairs becomes an independent binary decision on
    the logit ``t * <z_a_i, z_b_j> + b``: matched views (the diagonal) are
    positives (label ``+1``), all others negatives (``-1``). Because the loss is a
    plain sum of per-pair sigmoids — no softmax over the batch — it does not lean
    on large batches / many negatives the way NT-Xent does, and it learns its own
    scale/threshold via a scalar temperature ``t = exp(logit_scale)`` and bias
    ``logit_bias`` (paper inits ``logit_scale=log(10)``, ``logit_bias=-10``).

    Inputs must be L2-normalized along the last dim.
    """

    def __init__(
        self,
        *,
        init_logit_scale: float = math.log(10.0),
        init_logit_bias: float = -10.0,
    ) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(init_logit_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(float(init_logit_bias)))

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if z_a.shape != z_b.shape:
            raise ValueError(f"z_a/z_b shape mismatch: {z_a.shape} vs {z_b.shape}")
        b = z_a.shape[0]
        logits = z_a @ z_b.t() * self.logit_scale.exp() + self.logit_bias  # [B, B]
        # +1 on the diagonal (matched views), -1 off-diagonal.
        labels = 2.0 * torch.eye(b, device=z_a.device, dtype=logits.dtype) - 1.0
        # SigLIP: -(1/B) * sum_i sum_j log sigmoid(label_ij * logit_ij).
        return -torch.nn.functional.logsigmoid(labels * logits).sum(dim=1).mean()

    @torch.no_grad()
    def diagnostics(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict[str, float]:
        """Alignment/calibration stats to tell margin-pushing from real gain.

        Reported in interpretable units so the two effects are separable:

        * ``cos_pos_mean`` / ``cos_neg_mean`` / ``cos_margin`` — cosine of matched
          vs mismatched views. Pure *feature* geometry, independent of the
          learnable scale/bias. A matched cosine that jumps to ~1 almost
          immediately is the "encoders already contained the solution" shortcut
          signature; ``cos_margin`` (pos − neg) is the real separation.
        * ``temperature`` / ``logit_bias`` — the learnable calibration scalars. If
          the loss is mostly calibrating a decision boundary rather than the
          encoder learning, ``cos_*`` plateau early while these keep sliding.
        * ``pos_prob_mean`` / ``neg_prob_mean`` — the model's own predicted match
          probability ``sigmoid(t·cos + b)``. Folds in the scale/bias, so it shows
          whether the classifier is actually calibrated (pos→1, neg→0) rather than
          the raw cosine separation.
        """
        b = z_a.shape[0]
        cos = z_a @ z_b.t()  # [B, B]; rows are L2-normalized so dot = cosine.
        diag = torch.eye(b, device=z_a.device, dtype=torch.bool)
        cos_pos, cos_neg = cos[diag], cos[~diag]
        t = self.logit_scale.exp()
        prob = torch.sigmoid(cos * t + self.logit_bias)
        return {
            "siglip/diagnostics_temperature": float(t),
            "siglip/diagnostics_logit_bias": float(self.logit_bias),
            "siglip/diagnostics_cos_pos_mean": float(cos_pos.mean()),
            "siglip/diagnostics_cos_neg_mean": float(cos_neg.mean()),
            "siglip/diagnostics_cos_margin": float(cos_pos.mean() - cos_neg.mean()),
            "siglip/diagnostics_pos_prob_mean": float(prob[diag].mean()),
            "siglip/diagnostics_neg_prob_mean": float(prob[~diag].mean()),
        }


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (Epps–Pulley, LeJEPA MINIMAL.md).

    Expects ``proj`` shaped ``[V, B, D]`` (views × batch × dim).
    """

    def __init__(
        self,
        *,
        num_projections: int = 256,
        knots: int = 17,
        t_max: float = 3.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_projections < 1:
            raise ValueError(f"num_projections must be >= 1, got {num_projections}")
        if knots < 2:
            raise ValueError(f"knots must be >= 2, got {knots}")
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.num_projections = int(num_projections)
        self.eps = float(eps)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.dim() != 3:
            raise ValueError(f"SIGReg expects [V, B, D], got {tuple(proj.shape)}")
        device, dtype = proj.device, proj.dtype
        d = proj.size(-1)
        a = torch.randn(d, self.num_projections, device=device, dtype=dtype)
        a = a / a.norm(p=2, dim=0, keepdim=True).clamp_min(self.eps)
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class VISReg(nn.Module):
    """VISReg: Variance-Invariance-Sketching regularizer (arXiv:2606.02572).

    Drop-in anti-collapse alternative to :class:`SIGReg` on
    ``[N, D]`` rows. VISReg keeps VICReg's variance term but replaces the
    off-diagonal *covariance* penalty with a **sliced-Wasserstein sketching**
    term: every random 1D projection of the (centered, scale-normalized) batch
    is matched to a standard-Gaussian reference by comparing sorted projected
    values to Gaussian quantiles. This enforces the full marginal distributional
    *shape* rather than just second-order decorrelation, so it keeps producing
    useful gradients even under rank collapse (where a bare covariance penalty
    goes flat once dims are already uncorrelated but low-rank).

    Three sub-terms, all plain means so they stay ``O(1)`` at any batch size:

    * **scale**  ``mean_j (gamma - std_j)^2`` — pull every per-dim std to ``gamma``
      (a two-sided ``L2`` anchor on the *raw* std, so ``gamma=1`` pins the absolute
      output scale; this is what stops MSE invariance terms collapsing by scale).
    * **center** ``mean_j mu_j^2`` — pull the batch mean to 0 (Gaussian reference
      is centered).
    * **shape**  sliced-Wasserstein^2 to ``N(0,1)`` on the scale-normalized batch:
      ``mean_k mean_n (sort_n(z_norm @ w_k) - q_n)^2`` with ``q`` the standard-normal
      quantiles at ``n/(N+1)``. ``z_norm`` divides by ``std.detach()`` so the shape
      gradient does not fight the scale term over the output magnitude.

    :meth:`components` returns ``(scale, center + shape)`` — grouping ``center``
    with the distributional term so the two match VICReg's ``(var, cov)`` split
    for logging. :meth:`forward` returns ``scale + shape_coeff * (center + shape)``;
    the paper weights all three equally (``shape_coeff=1``). The caller scales the
    result the same way it scaled VICReg/SIGReg.

    Note (deviation from the paper): the paper's ``L_shape`` and ``L_center`` sum
    over dimensions/samples (``||.||_2^2``); we average instead, keeping every term
    ``O(1)`` so ``lejepa_lambda`` stays batch-size- and dim-independent, matching
    this file's convention for the other regularizers.
    """

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        shape_coeff: float = 1.0,
        num_projections: int = 4096,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if shape_coeff < 0.0:
            raise ValueError(f"shape_coeff must be >= 0, got {shape_coeff}")
        if num_projections < 1:
            raise ValueError(f"num_projections must be >= 1, got {num_projections}")
        self.gamma = float(gamma)
        self.shape_coeff = float(shape_coeff)
        self.num_projections = int(num_projections)
        self.eps = float(eps)

    @staticmethod
    def _gaussian_quantiles(
        n: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Standard-normal quantiles at plotting positions ``i/(N+1)``, ``i=1..N``."""
        u = torch.arange(1, n + 1, device=device, dtype=torch.float32) / (n + 1)
        q = torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)  # N(0,1) inverse CDF
        return q.to(dtype)

    def components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(scale_term, center + shape_term)`` for diagnostics/logging."""
        if x.dim() != 2:
            raise ValueError(f"VISReg expects [N, D], got {tuple(x.shape)}")
        n, d = x.shape
        if n < 2:
            return x.new_zeros(()), x.new_zeros(())
        mu = x.mean(dim=0, keepdim=True)                        # [1, D]
        x_cent = x - mu
        std = torch.sqrt(x_cent.var(dim=0) + self.eps)          # [D], unbiased
        scale = (self.gamma - std).square().mean()
        center = mu.squeeze(0).square().mean()
        # Sliced-Wasserstein "sketch": match each random 1D marginal to N(0,1).
        # std is detached so the shape term shapes the distribution without
        # tugging on the overall scale (that is the scale term's job).
        z_norm = x_cent / std.detach().clamp_min(self.eps)      # [N, D]
        w = torch.randn(d, self.num_projections, device=x.device, dtype=x.dtype)
        w = w / w.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
        proj = z_norm @ w                                       # [N, K]
        proj_sorted, _ = torch.sort(proj, dim=0)                # sort over samples
        q = self._gaussian_quantiles(n, x.device, x.dtype).unsqueeze(1)  # [N, 1]
        shape = (proj_sorted - q).square().mean()
        return scale, center + shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale, dist = self.components(x)
        return scale + self.shape_coeff * dist


@dataclass
class LeJEPALossTerms:
    """``inv``/``sigreg`` are the raw (unweighted) sub-losses, graph attached
    (not detached) so callers can take per-term gradients for diagnostics;
    detach before logging as a scalar.

    ``inv_rel`` is a detached diagnostic: invariance MSE divided by the
    between-molecule variance of per-molecule centers."""

    total: torch.Tensor
    inv: torch.Tensor
    sigreg: torch.Tensor
    inv_rel: torch.Tensor
    # VISReg split (None when SIGReg is used); ``sigreg == reg_scale + shape_coeff*reg_shape``.
    reg_scale: torch.Tensor | None = None
    reg_shape: torch.Tensor | None = None


class LeJEPALoss(nn.Module):
    """Minimal LeJEPA loss: invariance term + anti-collapse regularizer.

    ``L = (1 - lambda) * L_inv + lambda * L_reg``

    * **Invariance**: ``mean ||z_all - center||^2`` where ``center`` is the mean
      of intact (target) views per molecule. The center is *not* detached — this
      term is fully symmetric (no EMA/stop-grad teacher), so ``L_reg`` is the only
      thing preventing collapse.
    * **Regularizer** (``use_visreg``): SIGReg (Epps-Pulley Gaussianity CF test) by
      default, or VISReg (variance + sliced-Wasserstein sketching) — both on the
      batch of intact target embeddings. Reported as ``sigreg`` either way.

    Latents should be unnormalized pooled adapter outputs.

    SIGReg's raw statistic scales with batch size; divide by ``B`` here so
    ``lejepa_lambda`` is batch-size-independent (see ``forward``). VISReg is
    already ``O(1)`` (all means) and needs no rescaling.
    """

    def __init__(
        self,
        *,
        lejepa_lambda: float = 0.05,
        sigreg_num_projections: int = 256,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_eps: float = 1e-8,
        use_visreg: bool = False,
        visreg_gamma: float = 1.0,
        visreg_shape_coeff: float = 1.0,
        visreg_num_projections: int = 4096,
    ) -> None:
        super().__init__()
        if not (0.0 <= lejepa_lambda <= 1.0):
            raise ValueError(f"lejepa_lambda must be in [0, 1], got {lejepa_lambda}")
        self.lejepa_lambda = float(lejepa_lambda)
        # Anti-collapse regularizer on the intact target embeddings: SIGReg
        # (Gaussianity CF test) or VISReg (variance + sliced-Wasserstein sketching).
        # Either one is the *only* thing preventing collapse here — the invariance
        # term is symmetric (no EMA/stop-grad teacher), so this is a clean test of
        # whether the regularizer alone holds the representation up.
        self.use_visreg = bool(use_visreg)
        if self.use_visreg:
            self.visreg = VISReg(
                gamma=visreg_gamma,
                shape_coeff=visreg_shape_coeff,
                num_projections=visreg_num_projections,
            )
            self.sigreg = None
        else:
            self.visreg = None
            self.sigreg = SIGReg(
                num_projections=sigreg_num_projections,
                knots=sigreg_knots,
                t_max=sigreg_t_max,
                eps=sigreg_eps,
            )

    def forward(
        self,
        z_global: torch.Tensor,
        z_all: torch.Tensor,
    ) -> LeJEPALossTerms:
        """Compute LeJEPA loss.

        ``z_global``: ``[B, Vg, D]`` intact (fragment-shuffle) target views.
        ``z_all``: ``[B, Vg+Vl, D]`` all views (intact + masked-fragment).
        """
        if z_global.dim() != 3 or z_all.dim() != 3:
            raise ValueError(
                f"LeJEPALoss expects [B, V, D] tensors; got "
                f"global={tuple(z_global.shape)} all={tuple(z_all.shape)}"
            )
        if z_global.size(0) != z_all.size(0) or z_global.size(2) != z_all.size(2):
            raise ValueError("z_global and z_all batch/dim mismatch")
        if z_global.size(1) < 1:
            raise ValueError("LeJEPALoss needs >= 1 global (target) view")
        center = z_global.mean(dim=1, keepdim=True)             # [B, 1, D]
        inv = (z_all - center).square().mean()
        reg_scale = reg_shape = None
        if self.visreg is not None:
            # VISReg regularizes the batch of intact target embeddings as [N, D]
            # rows (all global views flattened together). Already O(1) (all means).
            reg_scale, reg_shape = self.visreg.components(
                z_global.reshape(-1, z_global.size(-1))
            )
            sigreg = reg_scale + self.visreg.shape_coeff * reg_shape
        else:
            # Per-sample SIGReg: the raw statistic carries an explicit *B factor
            # (Epps-Pulley N-scaling); divide it back out so the term is O(1) and
            # lambda is a clean relative weight. See class docstring.
            sigreg = self.sigreg(z_global.transpose(0, 1)) / z_global.size(0)  # [Vg, B, D]
        lam = self.lejepa_lambda
        total = (1.0 - lam) * inv + lam * sigreg
        # Diagnostic: invariance MSE relative to the between-molecule spread of
        # the per-molecule centers. See LeJEPALossTerms.
        with torch.no_grad():
            c = center.squeeze(1)                                # [B, D]
            baseline = (c - c.mean(dim=0, keepdim=True)).square().mean()
            inv_rel = inv.detach() / baseline.clamp_min(1e-12)
        return LeJEPALossTerms(
            total=total, inv=inv, sigreg=sigreg, inv_rel=inv_rel,
            reg_scale=reg_scale, reg_shape=reg_shape,
        )


@dataclass
class IJEPALossTerms:
    """Raw (unweighted) sub-losses with graph attached."""

    total: torch.Tensor
    predict: torch.Tensor
    glob: torch.Tensor
    inv: torch.Tensor
    sigreg: torch.Tensor
    gram: torch.Tensor
    # Noise-invariance: whole-molecule mean-pool of the noise-corrupted view vs the
    # stop-grad EMA-teacher clean global. Zero when ``noise_inv_weight == 0``.
    noise_inv: torch.Tensor | None = None
    # VISReg split (None when SIGReg is used); ``sigreg == reg_scale + shape_coeff*reg_shape``.
    reg_scale: torch.Tensor | None = None
    reg_shape: torch.Tensor | None = None


def gram_anchoring_loss(
    tok_online: torch.Tensor,
    tok_target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Gram anchoring (DINOv3): match the student's patch-patch similarity matrix
    to a stop-grad teacher's, per molecule.

    Each entry of the per-molecule Gram matrix is the cosine similarity between two
    patches (tokens). Matching Gram matrices constrains the *second-order statistics
    across patches* — the structure of dense features — without forcing the feature
    values themselves to match, which is what stabilizes dense representations over
    long training schedules.

    ``tok_online``/``tok_target`` are ``[B,T,D]`` per-token reps (the target is
    detached here); ``valid`` is ``[B,T]`` bool. Returns a scalar averaged over
    molecules, each normalized by its patch count squared so it is length-invariant.

    ponytail: single-scale only. DINOv3 also anchors across image resolutions, but a
    token sequence has no resolution axis, so only the patch (token) axis applies.
    """
    if tok_online.shape != tok_target.shape:
        raise ValueError(
            f"gram online/target shape mismatch: "
            f"{tuple(tok_online.shape)} vs {tuple(tok_target.shape)}"
        )
    if tok_online.dim() != 3 or valid.shape != tok_online.shape[:2]:
        raise ValueError(
            f"gram expects tok [B,T,D] and valid [B,T]; "
            f"got {tuple(tok_online.shape)} valid={tuple(valid.shape)}"
        )
    on = torch.nn.functional.normalize(tok_online, dim=-1)
    tg = torch.nn.functional.normalize(tok_target, dim=-1)
    gram_on = torch.bmm(on, on.transpose(1, 2))                 # [B,T,T] cos(patch_i, patch_j)
    gram_tg = torch.bmm(tg, tg.transpose(1, 2)).detach()
    pair = (valid.unsqueeze(2) & valid.unsqueeze(1)).to(gram_on.dtype)
    sq = (gram_on - gram_tg).square() * pair                    # zero out pad rows/cols
    p = valid.sum(dim=1).clamp_min(1).to(gram_on.dtype)         # patches per molecule
    return (sq.flatten(1).sum(dim=1) / (p * p)).mean()


class _IJEPAPredictor(nn.Module):
    """Small transformer: visible encoder reps + mask-token queries (with position embed)."""

    def __init__(
        self,
        dim: int,
        *,
        max_positions: int = 512,
        n_layers: int = 1,
        n_heads: int = 2,
        ff_mult: int = 2,
    ) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.pos_embed = nn.Embedding(int(max_positions), dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * int(ff_mult),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tok: torch.Tensor,
        hole: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``tok`` ``[B,T,D]``, ``hole`` ``[B,T]`` → hole predictions ``[N,D]``."""
        if tok.dim() != 3 or hole.shape != tok.shape[:2]:
            raise ValueError(
                f"predictor expects tok [B,T,D] and hole [B,T]; "
                f"got {tuple(tok.shape)} vs {tuple(hole.shape)}"
            )
        b, t_len, dim = tok.shape
        device = tok.device
        if valid is None:
            valid = torch.ones(b, t_len, dtype=torch.bool, device=device)
        visible = valid & ~hole

        n_vis = visible.sum(dim=1)
        n_hole = hole.sum(dim=1)
        active = n_hole > 0
        if not bool(active.any()):
            return tok.new_zeros(0, dim)

        tok_a = tok[active]
        visible_a = visible[active]
        hole_a = hole[active]
        n_vis_a = n_vis[active]
        b_a = tok_a.size(0)

        vis_cum = torch.where(
            visible_a, visible_a.long().cumsum(dim=1) - 1, 0
        )
        hole_cum = torch.where(hole_a, hole_a.long().cumsum(dim=1) - 1, 0)

        vis_b, vis_t = visible_a.nonzero(as_tuple=True)
        hole_b, hole_t = hole_a.nonzero(as_tuple=True)
        max_pos = 0
        if vis_t.numel():
            max_pos = max(max_pos, int(vis_t.max()))
        if hole_t.numel():
            max_pos = max(max_pos, int(hole_t.max()))
        if max_pos >= self.pos_embed.num_embeddings:
            raise ValueError(
                f"token position {max_pos} >= max_positions "
                f"{self.pos_embed.num_embeddings}"
            )

        seq_lens = n_vis_a + n_hole[active]
        max_len = int(seq_lens.max())
        seq = tok.new_zeros(b_a, max_len, dim)
        arange = torch.arange(max_len, device=device)
        pad = arange.unsqueeze(0) >= seq_lens.unsqueeze(1)

        if vis_b.numel():
            seq[vis_b, vis_cum[vis_b, vis_t]] = (
                tok_a[vis_b, vis_t] + self.pos_embed(vis_t)
            )
        if hole_b.numel():
            seq[hole_b, n_vis_a[hole_b] + hole_cum[hole_b, hole_t]] = (
                self.mask_token + self.pos_embed(hole_t)
            )

        out = self.norm(self.transformer(seq, src_key_padding_mask=pad))
        return out[hole_b, n_vis_a[hole_b] + hole_cum[hole_b, hole_t]]


def _mean_visible_pool(
    tok: torch.Tensor, hole: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Mean-pool visible (non-hole, valid) token reps per row: ``[B,T,D]`` → ``[B,D]``."""
    vis = (valid & ~hole).to(tok.dtype).unsqueeze(-1)
    return (tok * vis).sum(dim=1) / vis.sum(dim=1).clamp_min(1.0)


class _GlobalReadout(nn.Module):
    """Maps a pooled embedding to a predicted global target embedding.

    ``out_dim`` defaults to ``dim`` (square MLP); set it when the input pool width
    differs from the target (e.g. cross-modal prediction off a half-width pool).
    """

    def __init__(self, dim: int, out_dim: int | None = None) -> None:
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, out_dim),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx)


# Same MLP; public alias for DINO pool → predict → prototype path.
PooledEmbeddingPredictor = _GlobalReadout


def _zero_visible_tokens(
    tok: torch.Tensor, hole: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    out = tok.clone()
    out[(valid & ~hole).bool()] = 0.0
    return out


def _shuffle_visible_tokens(
    tok: torch.Tensor, hole: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    out = tok.clone()
    for i in range(tok.size(0)):
        vis = (valid[i] & ~hole[i]).nonzero(as_tuple=True)[0]
        if vis.numel() > 1:
            out[i, vis] = out[i, vis[torch.randperm(vis.numel(), device=tok.device)]]
    return out


class IJEPALoss(nn.Module):
    """Per-token I-JEPA plus optional global readout, shuffle- and noise-invariance.

    ``L = (1 - lambda) * (w_pred*L_predict + w_glob*L_glob + w_inv*L_inv
                          + w_noise_inv*L_noise_inv + w_gram*L_gram) + lambda*L_reg``

    ``L_glob``: cosine(readout(mean_visible(tok)), stopgrad(z_teacher)). ``L_inv``:
    MSE between the online intact pooled embedding and the stop-grad EMA-teacher
    pooled embedding of a different fragment shuffle (asymmetric → collapse-safe).
    ``L_noise_inv``: MSE between the whole-molecule mean-pool of a corrupted local
    view and the stop-grad EMA-teacher clean global (alignment-free, so the local
    may be an independent fragment shuffle). Setting ``predict_weight=0`` (with
    ``glob``/``gram`` off) drops the token-level objective entirely — the locals
    then match the teacher purely via the pooled ``L_noise_inv``.
    """

    def __init__(
        self,
        *,
        dim: int,
        lejepa_lambda: float = 0.05,
        predict_weight: float = 1.0,
        glob_weight: float = 1.0,
        inv_weight: float = 0.1,
        noise_inv_weight: float = 0.0,
        gram_weight: float = 0.0,
        sigreg_num_projections: int = 256,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_eps: float = 1e-8,
        use_visreg: bool = False,
        visreg_gamma: float = 1.0,
        visreg_shape_coeff: float = 1.0,
        visreg_num_projections: int = 4096,
        ijepa_max_positions: int = 512,
        ijepa_predictor_layers: int = 1,
        ijepa_predictor_heads: int = 2,
        ijepa_predictor_ff_mult: int = 2,
    ) -> None:
        super().__init__()
        if not (0.0 <= lejepa_lambda <= 1.0):
            raise ValueError(f"lejepa_lambda must be in [0, 1], got {lejepa_lambda}")
        if min(predict_weight, glob_weight, inv_weight, noise_inv_weight, gram_weight) < 0.0:
            raise ValueError(
                f"predict/glob/inv/noise_inv/gram weights must be >= 0, got "
                f"{predict_weight}, {glob_weight}, {inv_weight}, {noise_inv_weight}, {gram_weight}"
            )
        self.lejepa_lambda = float(lejepa_lambda)
        self.predict_weight = float(predict_weight)
        self.glob_weight = float(glob_weight)
        self.inv_weight = float(inv_weight)
        self.noise_inv_weight = float(noise_inv_weight)
        self.gram_weight = float(gram_weight)
        self.predictor = _IJEPAPredictor(
            dim,
            max_positions=int(ijepa_max_positions),
            n_layers=int(ijepa_predictor_layers),
            n_heads=int(ijepa_predictor_heads),
            ff_mult=int(ijepa_predictor_ff_mult),
        )
        self.glob_readout = _GlobalReadout(dim)
        self.use_visreg = bool(use_visreg)
        if self.use_visreg:
            self.visreg = VISReg(
                gamma=visreg_gamma,
                shape_coeff=visreg_shape_coeff,
                num_projections=visreg_num_projections,
            )
            self.sigreg = None
        else:
            self.visreg = None
            self.sigreg = SIGReg(
                num_projections=sigreg_num_projections,
                knots=sigreg_knots,
                t_max=sigreg_t_max,
                eps=sigreg_eps,
            )

    def _reg(self, rows: torch.Tensor) -> torch.Tensor:
        """Anti-collapse regularizer on ``[N, D]`` rows (SIGReg or VISReg).

        SIGReg's raw statistic carries an explicit ``*N`` batch factor (see
        :class:`SIGReg`), divided back out here so the term is ``O(1)`` and
        ``lejepa_lambda`` stays batch-size-independent. VISReg is already a mean,
        so it needs no rescaling.
        """
        if self.visreg is not None:
            return self.visreg(rows)
        return self.sigreg(rows.unsqueeze(0)) / rows.size(0)

    def _predict(
        self,
        tok: torch.Tensor,
        hole: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid is None:
            valid = torch.ones_like(hole, dtype=torch.bool)
        return self.predictor(tok, hole, valid=valid)

    def predict_loss(
        self,
        tok: torch.Tensor,
        hole: torch.Tensor,
        target: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cosine regression: ``1 - cos(predictor(visible, mask_tok), stopgrad(target))``."""
        pred = self._predict(tok, hole, valid=valid)
        if pred.shape != target.shape:
            raise ValueError(
                f"predict/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        return (
            1.0 - torch.nn.functional.cosine_similarity(pred, target.detach(), dim=-1)
        ).mean()

    def predict_global_embedding(
        self,
        tok: torch.Tensor,
        hole: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict clean-global ``z_m`` from visible (non-hole) token reps.

        ``glob_readout(mean_visible(tok))`` — used by ``glob_loss`` only.
        """
        if valid is None:
            valid = torch.ones_like(hole, dtype=torch.bool)
        return self.glob_readout(_mean_visible_pool(tok, hole, valid))

    def glob_loss(
        self,
        tok: torch.Tensor,
        hole: torch.Tensor,
        z_teacher: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``1 - cos(readout(mean_visible(tok)), stopgrad(z_teacher))`` per row."""
        if valid is None:
            valid = torch.ones_like(hole, dtype=torch.bool)
        if z_teacher.shape != (tok.size(0), tok.size(-1)):
            raise ValueError(
                f"z_teacher must be [B,D] matching tok batch; "
                f"got {tuple(z_teacher.shape)} vs tok {tuple(tok.shape)}"
            )
        pred = self.predict_global_embedding(tok, hole, valid=valid)
        return (
            1.0 - torch.nn.functional.cosine_similarity(pred, z_teacher.detach(), dim=-1)
        ).mean()

    @staticmethod
    def inv_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """MSE between an online pooled embedding and a stop-grad EMA-teacher target.

        Both are pooled embeddings of the same molecules under *different* fragment
        shuffles. ``z_b`` is the stop-grad teacher (caller detaches): this asymmetry
        is the anti-collapse mechanism, so no VICReg/SIGReg is needed for ``inv``.
        """
        if z_a.shape != z_b.shape:
            raise ValueError(f"z_a/z_b shape mismatch: {tuple(z_a.shape)} vs {tuple(z_b.shape)}")
        return torch.nn.functional.mse_loss(z_a, z_b)

    @torch.no_grad()
    def condition_bypass_gap(
        self,
        tok: torch.Tensor,
        hole: torch.Tensor,
        target: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Does the predictor use visible context, or ignore it?"""
        if valid is None:
            valid = torch.ones_like(hole, dtype=torch.bool)
        was_training = self.training
        self.eval()

        def _loss(t: torch.Tensor) -> torch.Tensor:
            return self.predict_loss(t, hole, target, valid=valid)

        l_true = _loss(tok)
        l_zero = _loss(_zero_visible_tokens(tok, hole, valid))
        l_shuf = _loss(_shuffle_visible_tokens(tok, hole, valid))
        if was_training:
            self.train()
        return {
            "predict_true": float(l_true),
            "predict_shuf": float(l_shuf),
            "predict_zero": float(l_zero),
            "gap_zero": float(l_zero - l_true),
            "gap_shuf": float(l_shuf - l_true),
        }

    def forward(
        self,
        tok: torch.Tensor,
        hole: torch.Tensor,
        target: torch.Tensor,
        z_pooled: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
        z_teacher_rows: torch.Tensor | None = None,
        z_inv_target: torch.Tensor | None = None,
        z_noised_pooled: torch.Tensor | None = None,
        gram_online: torch.Tensor | None = None,
        gram_target: torch.Tensor | None = None,
        gram_valid: torch.Tensor | None = None,
    ) -> IJEPALossTerms:
        """``tok`` ``[B,T,D]``; ``z_pooled`` ``[B_mol,D]`` for ``L_reg``.

        When ``gram_weight > 0``, ``gram_online``/``gram_target`` are the intact-view
        per-token reps ``[B_mol,T,D]`` from the online and EMA encoders (with
        ``gram_valid`` ``[B_mol,T]``) and a Gram-anchoring term is added.
        """
        if tok.dim() != 3 or hole.dim() != 2:
            raise ValueError(
                f"IJEPALoss expects tok [B,T,D] and hole [B,T]; "
                f"got {tuple(tok.shape)} hole={tuple(hole.shape)}"
            )
        if target.dim() != 2:
            raise ValueError(f"IJEPALoss expects target [N,D], got {tuple(target.shape)}")
        if z_pooled.dim() != 2:
            raise ValueError(f"IJEPALoss expects z_pooled [B,D], got {tuple(z_pooled.shape)}")
        if valid is None:
            valid = torch.ones_like(hole, dtype=torch.bool)
        if hole.sum() < 1:
            raise ValueError("IJEPALoss needs >= 1 hole token")
        if self.predict_weight > 0.0:
            predict = self.predict_loss(tok, hole, target, valid=valid)
        else:
            predict = tok.new_zeros(())
        if self.glob_weight > 0.0:
            if z_teacher_rows is None:
                raise ValueError("z_teacher_rows required when glob_weight > 0")
            glob = self.glob_loss(tok, hole, z_teacher_rows, valid=valid)
        else:
            glob = tok.new_zeros(())
        if self.inv_weight > 0.0:
            if z_inv_target is None:
                raise ValueError("z_inv_target required when inv_weight > 0")
            inv = self.inv_loss(z_pooled, z_inv_target.detach())
        else:
            inv = tok.new_zeros(())
        if self.noise_inv_weight > 0.0:
            if z_noised_pooled is None or z_teacher_rows is None:
                raise ValueError(
                    "z_noised_pooled and z_teacher_rows required when noise_inv_weight > 0"
                )
            # Whole-molecule pool of the noise-corrupted view pulled to the
            # stop-grad EMA clean global — collapse-safe like inv (asymmetric target).
            noise_inv = self.inv_loss(z_noised_pooled, z_teacher_rows.detach())
        else:
            noise_inv = tok.new_zeros(())
        if self.gram_weight > 0.0:
            if gram_online is None or gram_target is None or gram_valid is None:
                raise ValueError(
                    "gram_online, gram_target, gram_valid required when gram_weight > 0"
                )
            gram = gram_anchoring_loss(gram_online, gram_target, gram_valid)
        else:
            gram = tok.new_zeros(())
        lam = self.lejepa_lambda
        if lam > 0.0:
            if self.visreg is not None:
                reg_scale, reg_shape = self.visreg.components(z_pooled)
                sigreg = reg_scale + self.visreg.shape_coeff * reg_shape
            else:
                reg_scale = reg_shape = None
                sigreg = self._reg(z_pooled)
        else:
            reg_scale = reg_shape = None
            sigreg = z_pooled.new_zeros(())
        main = (
            self.predict_weight * predict
            + self.glob_weight * glob
            + self.inv_weight * inv
            + self.noise_inv_weight * noise_inv
            + self.gram_weight * gram
        )
        total = (1.0 - lam) * main + lam * sigreg
        return IJEPALossTerms(
            total=total, predict=predict, glob=glob, inv=inv, sigreg=sigreg, gram=gram,
            noise_inv=noise_inv, reg_scale=reg_scale, reg_shape=reg_shape,
        )


class DINOHead(nn.Module):
    """MLP → L2-normalized bottleneck → weight-normed prototype logits.

    Maps pooled ``z_m`` ``[N, dim]`` to ``[N, out_dim]``. Nonlinearity before the
    prototype layer matters: a bare linear map on L2-normalized 512-d inputs failed
    to learn (CE stuck at log K, rank → 2). Head is loss-only; downstream keeps ``z_m``.
    """

    def __init__(
        self, dim: int, out_dim: int, *, hidden: int = 2048, bottleneck: int = 256
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last.weight_g.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(self.mlp(x), dim=-1)
        return self.last(x)


class DINOLoss(nn.Module):
    """DINO cross-entropy: student matches a centered, sharpened stop-grad teacher.

    ``student_logits``/``teacher_logits`` are paired rows ``[N, K]`` (student =
    online crops, teacher = EMA global repeated to align). The teacher is centered
    (running EMA mean subtracted) then sharpened (low temperature); together these
    are DINO's collapse guard. Call :meth:`update_center` once per train step.
    """

    def __init__(
        self,
        out_dim: int,
        *,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        if not (teacher_temp > 0.0 and student_temp > 0.0):
            raise ValueError(
                f"temps must be > 0, got teacher={teacher_temp} student={student_temp}"
            )
        if not (0.0 <= center_momentum < 1.0):
            raise ValueError(f"center_momentum must be in [0, 1), got {center_momentum}")
        self.teacher_temp = float(teacher_temp)
        self.student_temp = float(student_temp)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, int(out_dim)))

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"dino student/teacher shape mismatch: "
                f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
            )
        student = torch.nn.functional.log_softmax(
            student_logits / self.student_temp, dim=-1
        )
        teacher = torch.nn.functional.softmax(
            (teacher_logits.detach() - self.center) / self.teacher_temp, dim=-1
        )
        return -(teacher * student).sum(dim=-1).mean()

    @torch.no_grad()
    def teacher_probs(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Centered, sharpened stop-grad teacher assignment ``[N, K]``."""
        return torch.nn.functional.softmax(
            (teacher_logits - self.center) / self.teacher_temp, dim=-1
        )

    @torch.no_grad()
    def utilization(self, teacher_logits: torch.Tensor) -> tuple[float, float]:
        """Prototype usage from the batch-mean teacher assignment.

        Returns ``(entropy_nats, n_active)``. ``entropy_nats == log(K)`` when all
        prototypes share mass evenly; ``~0`` when one prototype dominates. ``n_active``
        counts prototypes with mean batch prob ``>= 1/K``.
        """
        mean_p = self.teacher_probs(teacher_logits).mean(dim=0)
        k = int(mean_p.numel())
        eps = 1e-12
        entropy = float(-(mean_p * (mean_p + eps).log()).sum())
        active = float((mean_p >= 1.0 / k).sum())
        return entropy, active

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        """EMA-update the center toward the batch mean of teacher logits.

        ponytail: single-GPU batch mean (this pipeline runs 1 GPU/node). Multi-GPU
        DINO all-reduces this mean — add a dist.all_reduce here if that changes.
        """
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum
        )


class _FingerprintCache:
    """Caches Morgan bit vectors (as float32 numpy rows) keyed by SMILES.

    Morgan fingerprints are deterministic per molecule, so we compute each
    SMILES once across the whole run. Unparseable SMILES map to an all-zero
    row (they contribute Tanimoto 0 to every partner, which is harmless).
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        self.radius = radius
        self.n_bits = n_bits
        self._cache: dict[str, "np.ndarray"] = {}

    def bits(self, smiles: list[str]):
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        out = np.zeros((len(smiles), self.n_bits), dtype=np.float32)
        for i, s in enumerate(smiles):
            row = self._cache.get(s)
            if row is None:
                row = np.zeros(self.n_bits, dtype=np.float32)
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(
                        mol, self.radius, nBits=self.n_bits
                    )
                    DataStructs.ConvertToNumpyArray(fp, row)
                self._cache[s] = row
            out[i] = row
        return out


def tanimoto_target_matrix(
    bits: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Pairwise Tanimoto similarity ``[B, B]`` from binary fingerprint rows.

    ``bits`` is ``[B, n_bits]`` with values in {0, 1}. For binary vectors the
    Tanimoto (Jaccard) similarity is ``|A∩B| / |A∪B|`` =
    ``inter / (|A| + |B| − inter)``, computed here in closed form on the GPU.
    """
    inter = bits @ bits.t()                       # [B, B]
    counts = bits.sum(dim=1)                       # [B]
    union = counts.unsqueeze(0) + counts.unsqueeze(1) - inter
    return inter / union.clamp_min(eps)


def similarity_distillation_loss(
    z_m: torch.Tensor, target_sim: torch.Tensor
) -> torch.Tensor:
    """MSE between off-diagonal cosine similarities of ``z_m`` and ``target_sim``.

    ``z_m`` must be L2-normalized so ``z_m @ z_m.T`` is cosine similarity.
    """
    b = z_m.shape[0]
    cos = z_m @ z_m.t()                            # [B, B]
    mask = ~torch.eye(b, dtype=torch.bool, device=z_m.device)
    return torch.nn.functional.mse_loss(cos[mask], target_sim[mask])


def _top1_retrieval_one_way(z_query: torch.Tensor, z_key: torch.Tensor) -> float:
    """Fraction of queries whose max-dot-product key is the paired index."""
    if z_query.shape[0] != z_key.shape[0] or z_query.shape[0] == 0:
        raise ValueError(f"bad shapes for top-1: {z_query.shape}, {z_key.shape}")
    sim = z_query @ z_key.t()
    pred = sim.argmax(dim=1)
    target = torch.arange(z_query.shape[0], device=z_query.device)
    return (pred == target).float().mean().item()


def top1_paired_accuracy(
    z_a: torch.Tensor, z_b: torch.Tensor, *, symmetric: bool = False,
) -> float:
    """Top-1 retrieval of paired views within a batch (sanity metric).

    Uses max dot product (works for unnormalized LeJEPA latents). When
    ``symmetric=True``, averages query→key and key→query directions.
    """
    acc = _top1_retrieval_one_way(z_a, z_b)
    if symmetric:
        acc = 0.5 * (acc + _top1_retrieval_one_way(z_b, z_a))
    return acc


def lejepa_retrieval_acc1(z_global: torch.Tensor, z_all: torch.Tensor) -> float | None:
    """LeJEPA retrieval acc@1: global↔global if possible, else global↔local."""
    if z_global.dim() != 3 or z_all.dim() != 3:
        raise ValueError("expected [B, V, D] tensors")
    n_g = z_global.size(1)
    if n_g >= 2:
        return top1_paired_accuracy(z_global[:, 0], z_global[:, 1], symmetric=True)
    n_l = z_all.size(1) - n_g
    if n_g >= 1 and n_l >= 1:
        return top1_paired_accuracy(z_global[:, 0], z_all[:, n_g], symmetric=True)
    return None

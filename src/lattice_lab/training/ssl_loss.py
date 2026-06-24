"""SSL losses for paired molecule views.

* **NT-Xent** — SimCLR-style contrastive loss (default).
* **LeJEPA** — invariance loss + SIGReg isotropy regularizer:
  ``L = (1 - lambda) * L_inv + lambda * L_sigreg`` (convex combination, see
  ``galilai-group/lejepa`` and ``LeJEPALoss``'s docstring). Every view of a
  molecule (intact shuffles + masked) is pulled directly toward that molecule's
  intact center (no predictor); SIGReg keeps the target batch isotropic.
"""

from __future__ import annotations

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


class VICReg(nn.Module):
    """VICReg variance + covariance regularizer on ``[N, D]`` rows.

    Drop-in anti-collapse alternative to :class:`SIGReg`: a per-dimension
    variance hinge (push each std up toward ``gamma``) plus an off-diagonal
    covariance penalty (decorrelate dims). Unlike SIGReg it makes no
    Gaussianity assumption and both terms are plain means, so it stays
    well-behaved when the regularized batch ``N`` is small.

    Returns ``var_term + cov_coeff * cov_term`` as a single scalar; the caller
    scales it the same way it scaled SIGReg.
    """

    def __init__(
        self, *, gamma: float = 1.0, cov_coeff: float = 1.0, eps: float = 1e-4
    ) -> None:
        super().__init__()
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if cov_coeff < 0.0:
            raise ValueError(f"cov_coeff must be >= 0, got {cov_coeff}")
        self.gamma = float(gamma)
        self.cov_coeff = float(cov_coeff)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"VICReg expects [N, D], got {tuple(x.shape)}")
        n, d = x.shape
        if n < 2:
            return x.new_zeros(())
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + self.eps)  # unbiased (n-1), matches cov
        var_term = torch.relu(self.gamma - std).mean()
        cov = (x.T @ x) / (n - 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        cov_term = off_diag.square().sum() / d
        return var_term + self.cov_coeff * cov_term


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


class LeJEPALoss(nn.Module):
    """Minimal LeJEPA loss: invariance term + SIGReg isotropy regularizer.

    ``L = (1 - lambda) * L_inv + lambda * L_sigreg``

    * **Invariance**: ``mean ||z_all - center||^2`` where ``center`` is the mean
      of intact (target) views per molecule.
    * **SIGReg**: Epps-Pulley statistic on the batch of intact target embeddings.

    Latents should be unnormalized pooled adapter outputs.

    SIGReg's raw statistic scales with batch size; divide by ``B`` here so
    ``lejepa_lambda`` is batch-size-independent (see ``forward``).
    """

    def __init__(
        self,
        *,
        lejepa_lambda: float = 0.05,
        sigreg_num_projections: int = 256,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if not (0.0 <= lejepa_lambda <= 1.0):
            raise ValueError(f"lejepa_lambda must be in [0, 1], got {lejepa_lambda}")
        self.lejepa_lambda = float(lejepa_lambda)
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
        return LeJEPALossTerms(total=total, inv=inv, sigreg=sigreg, inv_rel=inv_rel)


@dataclass
class IJEPALossTerms:
    """Raw (unweighted) sub-losses with graph attached."""

    total: torch.Tensor
    predict: torch.Tensor
    glob: torch.Tensor
    inv: torch.Tensor
    sigreg: torch.Tensor
    gram: torch.Tensor


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
    """Maps pooled visible context to a global embedding prediction."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx)


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
    """Per-token I-JEPA plus optional global readout and shuffle-invariance terms.

    ``L = (1 - lambda) * (L_predict + w_glob * L_glob + w_inv * L_inv) + lambda * L_reg``

    ``L_glob``: cosine(readout(mean_visible(tok)), stopgrad(z_teacher)). ``L_inv``:
    MSE between the online intact pooled embedding and the stop-grad EMA-teacher
    pooled embedding of a different fragment shuffle (asymmetric → collapse-safe).
    """

    def __init__(
        self,
        *,
        dim: int,
        lejepa_lambda: float = 0.05,
        glob_weight: float = 1.0,
        inv_weight: float = 0.1,
        gram_weight: float = 0.0,
        sigreg_num_projections: int = 256,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_eps: float = 1e-8,
        use_vicreg: bool = False,
        vicreg_gamma: float = 1.0,
        vicreg_cov_coeff: float = 1.0,
        ijepa_max_positions: int = 512,
        ijepa_predictor_layers: int = 1,
        ijepa_predictor_heads: int = 2,
        ijepa_predictor_ff_mult: int = 2,
    ) -> None:
        super().__init__()
        if not (0.0 <= lejepa_lambda <= 1.0):
            raise ValueError(f"lejepa_lambda must be in [0, 1], got {lejepa_lambda}")
        if glob_weight < 0.0 or inv_weight < 0.0 or gram_weight < 0.0:
            raise ValueError(
                f"glob_weight, inv_weight, gram_weight must be >= 0, "
                f"got {glob_weight}, {inv_weight}, {gram_weight}"
            )
        self.lejepa_lambda = float(lejepa_lambda)
        self.glob_weight = float(glob_weight)
        self.inv_weight = float(inv_weight)
        self.gram_weight = float(gram_weight)
        self.predictor = _IJEPAPredictor(
            dim,
            max_positions=int(ijepa_max_positions),
            n_layers=int(ijepa_predictor_layers),
            n_heads=int(ijepa_predictor_heads),
            ff_mult=int(ijepa_predictor_ff_mult),
        )
        self.glob_readout = _GlobalReadout(dim)
        self.use_vicreg = bool(use_vicreg)
        if self.use_vicreg:
            self.vicreg = VICReg(gamma=vicreg_gamma, cov_coeff=vicreg_cov_coeff)
            self.sigreg = None
        else:
            self.vicreg = None
            self.sigreg = SIGReg(
                num_projections=sigreg_num_projections,
                knots=sigreg_knots,
                t_max=sigreg_t_max,
                eps=sigreg_eps,
            )

    def _reg(self, rows: torch.Tensor) -> torch.Tensor:
        """Anti-collapse regularizer on ``[N, D]`` rows (SIGReg or VICReg).

        SIGReg's raw statistic carries an explicit ``*N`` batch factor (see
        :class:`SIGReg`), divided back out here so the term is ``O(1)`` and
        ``lejepa_lambda`` stays batch-size-independent. VICReg is already a mean,
        so it needs no rescaling.
        """
        if self.vicreg is not None:
            return self.vicreg(rows)
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
        ctx = _mean_visible_pool(tok, hole, valid)
        pred = self.glob_readout(ctx)
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
        predict = self.predict_loss(tok, hole, target, valid=valid)
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
        if self.gram_weight > 0.0:
            if gram_online is None or gram_target is None or gram_valid is None:
                raise ValueError(
                    "gram_online, gram_target, gram_valid required when gram_weight > 0"
                )
            gram = gram_anchoring_loss(gram_online, gram_target, gram_valid)
        else:
            gram = tok.new_zeros(())
        sigreg = self._reg(z_pooled)
        lam = self.lejepa_lambda
        main = (
            predict
            + self.glob_weight * glob
            + self.inv_weight * inv
            + self.gram_weight * gram
        )
        total = (1.0 - lam) * main + lam * sigreg
        return IJEPALossTerms(
            total=total, predict=predict, glob=glob, inv=inv, sigreg=sigreg, gram=gram,
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

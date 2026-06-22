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

    ``inv_rel`` is a detached diagnostic: the invariance MSE divided by the
    between-molecule variance of the per-molecule centers. ~1 means a molecule's
    views are spread as widely as molecules are from each other (collapse /
    uninformative); «1 means views of a molecule cluster much tighter than the
    batch spreads (discriminative). The raw ``inv`` MSE is on unnormalized
    latents, so its absolute value is uninterpretable on its own — this ratio is
    what tells you whether the representation is actually discriminative."""

    total: torch.Tensor
    inv: torch.Tensor
    sigreg: torch.Tensor
    inv_rel: torch.Tensor


class LeJEPALoss(nn.Module):
    """Minimal LeJEPA loss: invariance term + SIGReg isotropy regularizer.

    ``L = (1 - lambda) * L_inv + lambda * L_sigreg``

    A convex combination, so ``lejepa_lambda in [0, 1]`` is a genuine *relative*
    trade-off between the two terms.

    * **Invariance**: every view of a molecule (intact fragment-shuffles +
      masked-fragment views) is pulled toward that molecule's center — the mean
      of its *intact* (target) views: ``mean ||z_all - center||^2``. No
      predictor: the views are aligned directly. A predictor only earns its keep
      when context↔target are information-asymmetric *and* paired with a
      stop-grad/EMA target (I-JEPA); here the same encoder produces symmetric
      views, so direct alignment is the honest objective and SIGReg is what
      prevents collapse.
    * **SIGReg**: runs on the batch of intact (target) embeddings only,
      forcing the chemical map toward an isotropic Gaussian.

    Latents should be **unnormalized** pooled adapter outputs.

    SIGReg's ``* proj.size(-2)`` (the formal Epps-Pulley statistic's batch-size
    scaling) makes the raw statistic grow ~linearly with batch size, unlike the
    invariance MSE's plain ``.mean()``. We divide that ``B`` back out here so the
    SIGReg term is ``O(1)`` and ``lambda`` is a clean, batch-size-independent
    relative weight (rather than the old trick of dividing ``lambda`` by ``B``).
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
    """``predict``/``sigreg`` are the raw (unweighted) sub-losses, graph attached.
    ``predict`` is visible+mask-token predictor → target cosine regression;
    ``sigreg`` is the anti-collapse regularizer (SIGReg or VICReg) on the
    **mean-pooled** intact ``z_m`` batch — the level at which collapse is observed."""

    total: torch.Tensor
    predict: torch.Tensor
    sigreg: torch.Tensor


class _IJEPAPredictor(nn.Module):
    """I-JEPA-style predictor: visible encoder reps + learnable mask tokens (pos embed).

    Kept intentionally small (default 1 layer) so semantics stay in the encoder.
    """

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
        b, _, dim = tok.shape
        if valid is None:
            valid = torch.ones(b, hole.size(1), dtype=torch.bool, device=tok.device)
        visible = valid & ~hole

        ctx_parts: list[torch.Tensor] = []
        query_parts: list[torch.Tensor] = []
        seq_lens: list[int] = []
        for i in range(b):
            hole_idx = hole[i].nonzero(as_tuple=True)[0]
            if hole_idx.numel() == 0:
                continue
            vis_idx = visible[i].nonzero(as_tuple=True)[0]
            max_pos = max(int(hole_idx.max()), int(vis_idx.max()) if vis_idx.numel() else 0)
            if max_pos >= self.pos_embed.num_embeddings:
                raise ValueError(
                    f"token position {max_pos} >= max_positions "
                    f"{self.pos_embed.num_embeddings}"
                )
            # Visible reps carry their position (same table as the mask-token
            # queries) so the predictor reasons about where each visible token sits
            # relative to the holes, instead of pooling a positionless bag.
            ctx_parts.append(tok[i, vis_idx] + self.pos_embed(vis_idx))
            query_parts.append(
                self.mask_token.expand(hole_idx.numel(), dim) + self.pos_embed(hole_idx)
            )
            seq_lens.append(int(vis_idx.numel() + hole_idx.numel()))
        if not ctx_parts:
            return tok.new_zeros(0, dim)

        max_len = max(seq_lens)
        n_seq = len(ctx_parts)
        seq = tok.new_zeros(n_seq, max_len, dim)
        pad = torch.ones(n_seq, max_len, dtype=torch.bool, device=tok.device)
        for j, (ctx, queries) in enumerate(zip(ctx_parts, query_parts)):
            n = ctx.size(0) + queries.size(0)
            seq[j, : ctx.size(0)] = ctx
            seq[j, ctx.size(0) : n] = queries
            pad[j, :n] = False
        out = self.norm(self.transformer(seq, src_key_padding_mask=pad))
        preds: list[torch.Tensor] = []
        for j, (ctx, queries) in enumerate(zip(ctx_parts, query_parts)):
            n_vis = ctx.size(0)
            preds.append(out[j, n_vis : n_vis + queries.size(0)])
        return torch.cat(preds, dim=0)


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
    """Per-token I-JEPA: visible reps + mask-token predictor → intact EMA targets.

    ``L = (1 - lambda) * L_predict + lambda * L_reg``

    One encoder pass on the masked sequence; **visible** adapter outputs (not ``<UNK>``
    rows) feed the predictor together with learnable mask tokens carrying hole position
    embeddings. Optional ``ijepa_block_hole_attn`` blocks hole keys in DDiT + adapter so
    visible reps are context-only. ``target`` is the EMA encoder on the intact sequence
    at the same indices.

    ``L_predict`` is the per-token prediction cosine (unnormalized adapter token
    outputs). ``L_reg`` is the anti-collapse regularizer (SIGReg or VICReg), applied
    to the **mean-pooled** intact ``z_m`` batch ``[B, D]`` — the level at which the
    pooled-representation rank actually collapses (token-level isotropy does not
    constrain the per-molecule mean).
    """

    def __init__(
        self,
        *,
        dim: int,
        lejepa_lambda: float = 0.05,
        sigreg_num_projections: int = 256,
        sigreg_knots: int = 17,
        sigreg_t_max: float = 3.0,
        sigreg_eps: float = 1e-8,
        # When true, VICReg (variance hinge + covariance penalty) replaces SIGReg
        # as the pooled-z_m anti-collapse regularizer.
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
        self.lejepa_lambda = float(lejepa_lambda)
        self.predictor = _IJEPAPredictor(
            dim,
            max_positions=int(ijepa_max_positions),
            n_layers=int(ijepa_predictor_layers),
            n_heads=int(ijepa_predictor_heads),
            ff_mult=int(ijepa_predictor_ff_mult),
        )
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
    ) -> IJEPALossTerms:
        """``tok`` ``[B,T,D]`` masked-view reps; ``hole`` ``[B,T]``; ``target`` ``[N,D]``;
        ``z_pooled`` ``[B_mol,D]`` online intact mean-pooled latents (graph-attached)
        — the anti-collapse regularizer acts on this batch, not on token rows."""
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
        sigreg = self._reg(z_pooled)
        lam = self.lejepa_lambda
        total = (1.0 - lam) * predict + lam * sigreg
        return IJEPALossTerms(total=total, predict=predict, sigreg=sigreg)


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
    """MSE between the cosine-similarity geometry of ``z_m`` and a target
    similarity matrix (e.g. Tanimoto), over off-diagonal pairs only.

    ``z_m`` must be L2-normalized (the adapter does this), so ``z_m @ z_m.T``
    is cosine similarity. Aligning it to the Morgan-FP Tanimoto matrix pulls
    chemically similar molecules together and dissimilar ones apart — the
    structure plain instance-discrimination SSL never learns.
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

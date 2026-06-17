"""SSL losses for paired molecule views.

* **NT-Xent** — SimCLR-style contrastive loss (default).
* **LeJEPA** — invariance (views match their batch mean) + SIGReg isotropy
  regularizer: ``(1 - eff_lambda) * inv + eff_lambda * sigreg`` where
  ``eff_lambda = lambda / batch_size`` (see ``galilai-group/lejepa`` and
  ``LeJEPALoss``'s docstring for why).
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


@dataclass
class LeJEPALossTerms:
    """``inv``/``sigreg`` are the raw (unweighted) sub-losses, graph attached
    (not detached) so callers can take per-term gradients for diagnostics;
    detach before logging as a scalar."""

    total: torch.Tensor
    inv: torch.Tensor
    sigreg: torch.Tensor


class LeJEPALoss(nn.Module):
    """LeJEPA loss with separate global and local (masked) views.

    Invariance pulls every view in ``z_all`` toward the mean of **global**
    views only (``z_global``). SIGReg runs on all views. Latents should be
    **unnormalized** pooled adapter outputs.

    SIGReg's ``* proj.size(-2)`` (the formal Epps-Pulley statistic's batch-size
    scaling) makes its gradient grow roughly linearly with batch size, unlike
    ``inv``'s plain ``.mean()`` -- measured ~10000x raw gradient-norm gap at
    batch_size=128. To keep ``lejepa_lambda`` meaningful across batch sizes
    without altering SIGReg itself, it's divided by batch size at combination
    time (``effective_lambda = lejepa_lambda / batch_size``).
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

        ``z_global``: ``[B, Vg, D]`` global (fragment-shuffle) views.
        ``z_all``: ``[B, Vg+Vl, D]`` global + local (masked-fragment) views.
        """
        if z_global.dim() != 3 or z_all.dim() != 3:
            raise ValueError(
                f"LeJEPALoss expects [B, V, D] tensors; got "
                f"global={tuple(z_global.shape)} all={tuple(z_all.shape)}"
            )
        if z_global.size(0) != z_all.size(0) or z_global.size(2) != z_all.size(2):
            raise ValueError("z_global and z_all batch/dim mismatch")
        if z_global.size(1) < 1:
            raise ValueError("LeJEPALoss needs at least one global view")
        centers = z_global.mean(dim=1, keepdim=True)
        inv = (centers - z_all).square().mean()
        sigreg = self.sigreg(z_all.transpose(0, 1))
        # SIGReg's statistic scales ~linearly with batch size (it's the literal
        # N-scaled Epps-Pulley statistic); divide lambda by batch size so it
        # keeps a stable, batch-size-independent meaning. See class docstring.
        effective_lambda = self.lejepa_lambda / z_global.size(0)
        total = (1.0 - effective_lambda) * inv + effective_lambda * sigreg
        return LeJEPALossTerms(total=total, inv=inv, sigreg=sigreg)


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

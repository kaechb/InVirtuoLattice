"""SimCLR-style NT-Xent loss for paired molecule views.

Given two L2-normalized projection batches ``z_a, z_b`` of shape ``[B, D]``, the
loss treats the corresponding pairs (i, i) as positives and all other pairs as
negatives. The symmetric variant averages over both anchor directions.

Implementation notes:
- Cosine similarity assumes inputs are already unit-norm; the adapter does this.
- Self-similarity (diagonal of the same-batch similarity matrix) is masked out
  by setting it to ``-inf`` before the softmax.
"""

from __future__ import annotations

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


def top1_paired_accuracy(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """Top-1 retrieval of paired views within a batch (sanity metric).

    For each ``z_a[i]`` the nearest ``z_b[j]`` should be at ``j == i``. Returns
    the fraction of rows for which this holds.
    """
    if z_a.shape != z_b.shape or z_a.shape[0] == 0:
        raise ValueError(f"bad shapes for top-1: {z_a.shape}, {z_b.shape}")
    sim = z_a @ z_b.t()
    pred = sim.argmax(dim=1)
    target = torch.arange(z_a.shape[0], device=z_a.device)
    return (pred == target).float().mean().item()

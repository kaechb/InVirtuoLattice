"""Sanity check #2 — nearest-neighbor retrieval on a curated bioisostere set.

For each pair ``(A, B)`` in the curated CSV, both molecules should embed close.
We compute the metric per *unique molecule*: build the set of valid partners
across all pairs (bioisosterism is many-to-many — benzoic acid is partnered with
benzenesulfonamide AND with cyclohexanecarboxylic acid AND with thiophene-2-
carboxylic acid in the curated set), encode every unique molecule, and check
whether at least one valid partner appears in the top-K nearest neighbors.

README target: ``recall@10 > 0.7``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDLogger

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder
from lattice_lab.eval.encode_utils import encode_smiles_batched

RDLogger.DisableLog("rdApp.*")

DEFAULT_BIOISOSTERE_CSV: Path = Path(__file__).resolve().parent / "data" / "bioisosteres.csv"


@dataclass(frozen=True)
class BioisostereResult:
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    n_molecules: int
    n_pairs: int
    threshold: float
    passed: bool

    def as_metrics(self) -> dict[str, float | int | bool]:
        return {
            "bioiso/recall@1": self.recall_at_1,
            "bioiso/recall@5": self.recall_at_5,
            "bioiso/recall@10": self.recall_at_10,
            "bioiso/n_molecules": self.n_molecules,
            "bioiso/n_pairs": self.n_pairs,
            "bioiso/threshold": self.threshold,
            "bioiso/pass": bool(self.passed),
        }


def _build_partner_index(df: pd.DataFrame) -> tuple[list[str], dict[int, set[int]]]:
    """Return ``(unique_smiles, partner_idx_set_per_mol)``.

    Each SMILES is canonicalized once. ``partner_idx_set_per_mol[i]`` is the set
    of indices ``j`` such that some pair links ``unique_smiles[i]`` to ``unique_smiles[j]``.
    """
    pos: dict[str, int] = {}

    def _get_idx(smi: str) -> int:
        c = Chem.CanonSmiles(smi)
        if c not in pos:
            pos[c] = len(pos)
        return pos[c]

    partners: dict[int, set[int]] = defaultdict(set)
    for _, row in df.iterrows():
        i = _get_idx(row["smiles_a"])
        j = _get_idx(row["smiles_b"])
        if i == j:
            continue  # degenerate pair, drop
        partners[i].add(j)
        partners[j].add(i)

    unique_smiles = [None] * len(pos)
    for smi, idx in pos.items():
        unique_smiles[idx] = smi
    return unique_smiles, dict(partners)  # type: ignore[return-value]


@torch.no_grad()
def evaluate_bioisostere_retrieval(
    encoder: DiscreteFlowEncoder,
    csv_path: Path | str = DEFAULT_BIOISOSTERE_CSV,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    threshold: float = 0.7,
    seed: int = 0,
    n_jobs: int | None = None,
) -> BioisostereResult:
    """Compute recall@K on the bioisostere set; pass-condition is ``recall@10 ≥ threshold``."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return BioisostereResult(0.0, 0.0, 0.0, 0, 0, threshold, False)

    unique_smiles, partners = _build_partner_index(df)
    z, valid_idx = encode_smiles_batched(
        encoder, unique_smiles, batch_size=batch_size, device=device, seed=seed,
        desc="bioiso encode", n_jobs=n_jobs,
    )
    # Drop molecules whose embedding failed (very rare after Stage-1 standardization).
    valid_set = set(valid_idx)
    keep_mask = torch.tensor([i in valid_set for i in range(len(unique_smiles))])
    keep_idx_in_orig = [i for i, k in enumerate(keep_mask.tolist()) if k]
    orig_to_enc = {orig: enc for enc, orig in enumerate(valid_idx)}

    # Pairwise cosine similarity (z is L2-normalized).
    sim = z @ z.t()
    sim.fill_diagonal_(float("-inf"))  # don't retrieve self

    n_eval = 0
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    for orig_i in keep_idx_in_orig:
        enc_i = orig_to_enc[orig_i]
        partner_origs = partners.get(orig_i, set())
        # Map partner originals to encoded positions; drop ones that failed to embed.
        partner_encs = {orig_to_enc[p] for p in partner_origs if p in orig_to_enc}
        if not partner_encs:
            continue
        n_eval += 1
        k = min(10, sim.shape[1])
        _, topk_idx = sim[enc_i].topk(k)
        topk_set = set(topk_idx.tolist())
        if topk_set & partner_encs:
            hits_10 += 1
        if set(topk_idx[:5].tolist()) & partner_encs:
            hits_5 += 1
        if int(topk_idx[0].item()) in partner_encs:
            hits_1 += 1

    if n_eval == 0:
        return BioisostereResult(0.0, 0.0, 0.0, 0, len(df), threshold, False)

    recall_10 = hits_10 / n_eval
    return BioisostereResult(
        recall_at_1=hits_1 / n_eval,
        recall_at_5=hits_5 / n_eval,
        recall_at_10=recall_10,
        n_molecules=n_eval,
        n_pairs=len(df),
        threshold=threshold,
        passed=recall_10 >= threshold,
    )

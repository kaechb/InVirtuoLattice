"""Classical baselines + literature references for Stage 2 sanity comparisons.

Two kinds of context for each adapter metric:

1. **Computed baselines** run on the *same* data as the adapter check (Morgan FP
   + Ridge for QM9, Morgan FP + Tanimoto for bioisostere retrieval). These are
   apples-to-apples — they tell you whether the adapter is doing anything beyond
   what a classical chemical fingerprint already does.

2. **Literature references** are paper-reported values for SMILES/2D-only
   methods on QM9 and similar benchmarks. They give a rough sense of where the
   adapter sits in the field. We do NOT aim to beat 3D-GNN SOTA here — the
   reference range is wide on purpose.

Both are logged to W&B under the ``baseline/`` and ``reference/`` namespaces
respectively so they sit next to the live adapter metrics on the same charts.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from lattice_lab.eval.bioisostere_retrieval import (
    BioisostereResult,
    _build_partner_index,
)
from lattice_lab.eval.qm9_probe import Qm9ProbeResult

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


# Citation tuples: (label, value, source).
# Values are *rough* targets for SMILES-only methods reported in the literature.
# QM9 HOMO/LUMO R² depends heavily on subsample size and split; treat as ranges.
LITERATURE_REFERENCES: dict[str, list[tuple[str, float, str]]] = {
    "qm9/r2_homo": [
        ("Morgan FP + Ridge (this run)", float("nan"), "computed baseline"),
        ("ChemBERTa linear probe", 0.65,
         "Chithrananda et al., NeurIPS-W 2020"),
        ("MolBERT linear probe", 0.72,
         "Fabian et al., arXiv 2011.13230 (2020)"),
        ("D-MPNN supervised", 0.86,
         "Yang et al., JCIM 2019 (2D graph, supervised on QM9)"),
        ("SchNet (3D coords)", 0.97,
         "Schütt et al., NeurIPS 2017"),
    ],
    "qm9/r2_lumo": [
        ("Morgan FP + Ridge (this run)", float("nan"), "computed baseline"),
        ("ChemBERTa linear probe", 0.67,
         "Chithrananda et al., NeurIPS-W 2020"),
        ("MolBERT linear probe", 0.74,
         "Fabian et al., arXiv 2011.13230 (2020)"),
        ("D-MPNN supervised", 0.88,
         "Yang et al., JCIM 2019"),
        ("SchNet (3D coords)", 0.97,
         "Schütt et al., NeurIPS 2017"),
    ],
    "bioiso/recall@10": [
        ("Morgan FP + Tanimoto (this run)", float("nan"), "computed baseline"),
        ("MolCLR (contrastive, 2D graph)", 0.78,
         "Wang et al., Nat. Mach. Intell. 2022 (qualitative)"),
        ("ChemBERTa", 0.72, "Chithrananda et al., NeurIPS-W 2020 (qualitative)"),
    ],
    "val/top1_acc": [
        # Self-retrieval over paired contrastive views is a SimCLR-style setup;
        # MolCLR reports >0.9 paired-view top-1 after pretraining.
        ("MolCLR paired-view top-1", 0.93,
         "Wang et al., Nat. Mach. Intell. 2022"),
        ("Random baseline (1/N)", float("nan"),
         "computed as 1 / val_set_size at log time"),
    ],
}


def _morgan_fp_array(
    smiles_list: Sequence[str], *, radius: int = 2, n_bits: int = 2048
) -> tuple[np.ndarray, list[int]]:
    """Return ``([N, n_bits] float32 array, valid_idx)``; invalid rows are zero.

    ``valid_idx[i]`` is the position in the input of the molecule whose FP is
    ``arr[i]``. Failed parses are skipped (not zero-padded), so callers can
    align targets without a separate mask.
    """
    rows: list[np.ndarray] = []
    valid: list[int] = []
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        v = np.zeros(n_bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, v)
        rows.append(v.astype(np.float32))
        valid.append(i)
    if not rows:
        return np.zeros((0, n_bits), dtype=np.float32), []
    return np.stack(rows), valid


def morgan_qm9_baseline(
    qm9_csv: Path | str,
    *,
    targets: Sequence[str] = ("homo", "lumo"),
    test_size: float = 0.2,
    n_subset: int | None = 5000,
    seed: int = 0,
    ridge_alpha: float = 1.0,
) -> Qm9ProbeResult:
    """Ridge regression on Morgan(r=2, 2048) → HOMO/LUMO; same split as adapter probe."""
    df = pd.read_csv(qm9_csv)
    for t in targets:
        if t not in df.columns:
            raise ValueError(f"target {t!r} not in {list(df.columns)}")
    smiles = df["smiles"].astype(str).tolist()
    y_all = df[list(targets)].to_numpy(dtype=np.float32)

    if n_subset is not None and n_subset < len(smiles):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(smiles), size=n_subset, replace=False)
        smiles = [smiles[i] for i in idx]
        y_all = y_all[idx]

    fps, valid = _morgan_fp_array(smiles)
    if not valid:
        return Qm9ProbeResult({t: 0.0 for t in targets}, 0.0, 0, 0, 0.0, False)
    y_kept = y_all[valid]
    x_tr, x_te, y_tr, y_te = train_test_split(
        fps, y_kept, test_size=test_size, random_state=seed
    )
    r2: dict[str, float] = {}
    for j, t in enumerate(targets):
        m = Ridge(alpha=ridge_alpha)
        m.fit(x_tr, y_tr[:, j])
        r2[t] = float(r2_score(y_te[:, j], m.predict(x_te)))
    mean_r2 = float(np.mean(list(r2.values())))
    return Qm9ProbeResult(
        r2_by_target=r2,
        mean_r2=mean_r2,
        n_train=len(x_tr),
        n_test=len(x_te),
        threshold=0.0,  # baseline has no threshold; it's the reference.
        passed=True,
    )


def morgan_bioisostere_baseline(
    csv_path: Path | str, *, threshold: float = 0.7
) -> BioisostereResult:
    """Tanimoto similarity over Morgan FPs as the retrieval signal."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return BioisostereResult(0.0, 0.0, 0.0, 0, 0, threshold, False)

    unique, partners = _build_partner_index(df)
    fps_arr, valid = _morgan_fp_array(unique)
    if not valid:
        return BioisostereResult(0.0, 0.0, 0.0, 0, len(df), threshold, False)

    fps = fps_arr.astype(np.float32)
    # Tanimoto = |A ∩ B| / |A ∪ B| on bit vectors.
    intersect = fps @ fps.T
    pop = fps.sum(axis=1, keepdims=True)
    union = pop + pop.T - intersect
    sim = np.where(union > 0, intersect / np.maximum(union, 1e-9), 0.0)
    np.fill_diagonal(sim, -1.0)  # don't retrieve self

    valid_set = set(valid)
    orig_to_enc = {orig: enc for enc, orig in enumerate(valid)}
    hits = {1: 0, 5: 0, 10: 0}
    n_eval = 0
    for orig_i in valid:
        enc_i = orig_to_enc[orig_i]
        partner_origs = partners.get(orig_i, set())
        partner_encs = {orig_to_enc[p] for p in partner_origs if p in orig_to_enc}
        if not partner_encs:
            continue
        n_eval += 1
        order = np.argsort(-sim[enc_i])
        for k in (1, 5, 10):
            topk = set(order[:k].tolist())
            if topk & partner_encs:
                hits[k] += 1
    if n_eval == 0:
        return BioisostereResult(0.0, 0.0, 0.0, 0, len(df), threshold, False)
    recall_10 = hits[10] / n_eval
    return BioisostereResult(
        recall_at_1=hits[1] / n_eval,
        recall_at_5=hits[5] / n_eval,
        recall_at_10=recall_10,
        n_molecules=n_eval,
        n_pairs=len(df),
        threshold=threshold,
        passed=recall_10 >= threshold,
    )


def format_reference_table(
    metric_key: str,
    adapter_value: float | None,
    baseline_value: float | None = None,
) -> str:
    """Return a fixed-width text table comparing adapter vs baseline vs literature."""
    lines: list[str] = [
        f"== {metric_key} ==",
        f"  {'method':<48} {'value':>8}  source",
        f"  {'-' * 48:<48} {'-' * 8:>8}  {'-' * 40}",
    ]
    if adapter_value is not None:
        lines.append(f"  {'LATTICE adapter (this run)':<48} {adapter_value:>8.3f}  current run")
    if baseline_value is not None:
        lines.append(
            f"  {'Morgan FP baseline (this run)':<48} {baseline_value:>8.3f}  computed"
        )
    for label, value, source in LITERATURE_REFERENCES.get(metric_key, []):
        if "this run" in label.lower():
            continue
        if not np.isfinite(value):
            continue
        lines.append(f"  {label:<48} {value:>8.3f}  {source}")
    return "\n".join(lines)

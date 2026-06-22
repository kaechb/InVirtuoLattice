"""Molecular preprocessing primitives.

Order of operations (each step is pure and unit-tested):

1. ``standardize_smiles`` — largest fragment, neutralize, canonical tautomer.
2. ``passes_property_filter`` — MW/logP/atom whitelist gate.
3. ``inchikey_of`` — canonical key for deduplication.
4. ``smiles_to_fragment_views`` — produce K augmented space-separated fragment views.
5. ``morgan_fingerprint`` — Morgan FP for retrieval / sanity (NOT used for training).

These functions are intentionally side-effect free; the orchestrator in
``run_preprocessing.py`` handles I/O and parallelism.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

# Suppress RDKit chatter once at import time.
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

ATOM_WHITELIST: frozenset[str] = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})


@dataclass(frozen=True)
class PropertyFilter:
    """Drug-like property gate. Defaults match README Stage 1 spec."""

    mw_min: float = 150.0
    mw_max: float = 700.0
    logp_min: float = -2.0
    logp_max: float = 6.0
    atom_whitelist: frozenset[str] = ATOM_WHITELIST


def _make_standardizer() -> tuple[rdMolStandardize.LargestFragmentChooser,
                                  rdMolStandardize.Uncharger,
                                  rdMolStandardize.TautomerEnumerator]:
    """Instantiate the three RDKit MolStandardize objects.

    Returned together because they are stateless once built and we want one set
    per worker process (creation is non-trivial).
    """
    params = rdMolStandardize.CleanupParameters()
    return (
        rdMolStandardize.LargestFragmentChooser(),
        rdMolStandardize.Uncharger(),
        rdMolStandardize.TautomerEnumerator(params),
    )


def standardize_smiles(smiles: str) -> str | None:
    """Return a canonical, standardized SMILES, or ``None`` if input is invalid.

    Steps: parse → largest organic fragment → uncharge → canonical tautomer →
    canonical SMILES. Idempotent: ``standardize_smiles(standardize_smiles(x)) == standardize_smiles(x)``.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        largest, uncharger, taut = _make_standardizer()
        mol = largest.choose(mol)
        mol = uncharger.uncharge(mol)
        mol = taut.Canonicalize(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as exc:  # RDKit errors are non-standard; capture broadly.
        logger.debug("standardize failed for %s: %s", smiles, exc)
        return None


def _atoms_in_whitelist(mol: Chem.Mol, whitelist: frozenset[str]) -> bool:
    return all(a.GetSymbol() in whitelist for a in mol.GetAtoms())


def passes_property_filter(smiles: str, pf: PropertyFilter | None = None) -> bool:
    """Return True iff the molecule passes the MW / logP / atom-set gate."""
    pf = pf or PropertyFilter()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    if not (pf.mw_min <= mw <= pf.mw_max):
        return False
    logp = Crippen.MolLogP(mol)
    if not (pf.logp_min <= logp <= pf.logp_max):
        return False
    return _atoms_in_whitelist(mol, pf.atom_whitelist)


def inchikey_of(smiles: str) -> str | None:
    """Return the InChIKey (used for dedup), or ``None`` on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def _brics_fragment_smiles(mol: Chem.Mol) -> list[str]:
    """Return canonical SMILES for each BRICS fragment of ``mol``.

    Falls back to an empty list when RDKit BRICS fails on pathological
    structures (some cluster RDKit builds raise internally on certain scaffolds).
    Callers treat ``[]`` as "use the whole molecule".
    """
    from rdkit.Chem import BRICS

    try:
        pieces = BRICS.BRICSDecompose(mol)
    except Exception as exc:
        logger.debug("BRICS decompose failed: %s", exc)
        return []

    out: list[str] = []
    for piece in pieces:
        if isinstance(piece, str):
            smi = piece
        elif isinstance(piece, Chem.Mol):
            smi = Chem.MolToSmiles(piece)
        else:
            logger.debug("BRICS returned unexpected piece type: %r", type(piece))
            continue
        frag = Chem.MolFromSmiles(smi)
        if frag is not None:
            out.append(Chem.MolToSmiles(frag))
    return out


def smiles_to_fragment_views(
    smiles: str,
    n_views: int = 3,
    *,
    max_fragments: int = 7,
    seed: int | None = None,
    max_attempts: int = 64,
) -> list[str]:
    """Generate ``n_views`` distinct space-separated fragment views for ``smiles``.

    Uses RDKit BRICS decomposition (no external fragmentation dependency). Each
    view samples a random number of fragments in ``[1, min(max_fragments, 7)]``,
    shuffles them, and joins with spaces — the format expected by the
    discrete-flow tokenizer.
    """
    rng = random.Random(seed)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    canon = Chem.MolToSmiles(mol)
    frags = _brics_fragment_smiles(mol)
    if len(frags) <= 1:
        return [canon]

    max_frags = min(max_fragments, 7, len(frags))
    views: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(views) < n_views and attempts < max_attempts:
        attempts += 1
        k = rng.randint(1, max_frags)
        if k == 1:
            if canon not in seen:
                seen.add(canon)
                views.append(canon)
            continue
        chosen = rng.sample(frags, min(k, len(frags)))
        rng.shuffle(chosen)
        view = " ".join(chosen)
        if view not in seen:
            seen.add(view)
            views.append(view)
    return views


def seeded_views(smiles: str, k: int) -> list[str]:
    """Up to ``k`` deterministic seeded fragment views (per-molecule seed derived
    from the SMILES, so runs are reproducible). Falls back to the canonical SMILES
    for un-fragmentable molecules and ``[]`` for RDKit-unparseable ones.

    Defined here (a torch-free module) so ``joblib``/``loky`` workers that call it
    don't cold-import torch at spawn — the slow part of multi-view cache builds.
    """
    import hashlib

    seed = int.from_bytes(hashlib.sha1(smiles.encode()).digest()[:4], "little")
    views = smiles_to_fragment_views(smiles, n_views=k, seed=seed)
    if views:
        return views
    return [Chem.CanonSmiles(smiles)] if Chem.MolFromSmiles(smiles) else []


def fragment_view_for_smiles(smiles: str) -> str | None:
    """One fragment view for adapter encode; canonical SMILES fallback; else None."""
    from rdkit import Chem

    try:
        views = smiles_to_fragment_views(smiles, n_views=1)
        if views:
            return views[0]
    except Exception as exc:
        logger.debug("fragment view failed for %r: %s", smiles, exc)
    if Chem.MolFromSmiles(smiles) is None:
        return None
    return Chem.CanonSmiles(smiles)


def build_smiles_fragment_views(
    smiles_list: Sequence[str],
    *,
    n_jobs: int = 1,
) -> dict[str, str]:
    """Map unique SMILES → one fragment view; omits unfragmentable ligands."""
    unique = list(dict.fromkeys(smiles_list))
    if not unique:
        return {}

    def _pair(smi: str) -> tuple[str, str | None]:
        return smi, fragment_view_for_smiles(smi)

    if n_jobs in (0, 1):
        pairs = [_pair(s) for s in unique]
    else:
        from joblib import Parallel, delayed

        pairs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_pair)(s) for s in unique
        )
    return {s: v for s, v in pairs if v is not None}


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    """Morgan fingerprint as a uint8 bit vector. ``None`` for invalid SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def process_smiles_record(
    smiles: str,
    *,
    n_views: int = 3,
    pf: PropertyFilter | None = None,
    seed: int | None = None,
) -> dict[str, object] | None:
    """End-to-end per-molecule pipeline.

    Returns a record dict (or ``None`` if the molecule is rejected) with keys:
    ``smiles`` (canonical), ``inchikey``, ``views`` (list of fragment strings).

    This is the unit dispatched to worker processes by the orchestrator.
    """
    std = standardize_smiles(smiles)
    if std is None:
        return None
    if not passes_property_filter(std, pf):
        return None
    key = inchikey_of(std)
    if key is None:
        return None
    views = smiles_to_fragment_views(std, n_views=n_views, seed=seed)
    if not views:
        return None
    return {"smiles": std, "inchikey": key, "views": views}


def molecule_probe_props(smiles: str) -> tuple[float, float, float] | None:
    """Return ``(QED, MolWt, logP)`` for a SMILES string, or ``None`` if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return (
        float(QED.qed(mol)),
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
    )


def molecule_qed_molwt(smiles: str) -> tuple[float, float] | None:
    """Return ``(QED, MolWt)`` for a SMILES string, or ``None`` if invalid."""
    row = molecule_probe_props(smiles)
    return None if row is None else (row[0], row[1])


def fragment_view_column(columns) -> str:
    """Return the fragment-view column name (supports legacy ``fragmol_view``).

    Accepts a DataFrame or any iterable of column names (e.g. parquet schema).
    """
    names = columns.columns if hasattr(columns, "columns") else columns
    for col in ("fragment_view", "fragmol_view"):
        if col in names:
            return col
    raise KeyError("expected fragment_view or fragmol_view column in parquet")


def fragment_view_column_for_parquet(path) -> str:
    """Resolve the view column from a parquet file schema (no row I/O)."""
    import pyarrow.parquet as pq

    return fragment_view_column(pq.read_schema(path).names)


def shards_have_body_ids(shards: list) -> bool:
    """True when the first shard was written with a ``body_ids`` column."""
    if not shards:
        return False
    import pyarrow.parquet as pq

    return "body_ids" in pq.read_schema(shards[0]).names


def load_smiles_tokenizer(path: str | Path):
    """Discrete-flow SMILES tokenizer (``PreTrainedTokenizerFast`` json)."""
    from transformers import PreTrainedTokenizerFast

    p = Path(path)
    if p.is_file():
        return PreTrainedTokenizerFast(tokenizer_file=str(p))
    return PreTrainedTokenizerFast.from_pretrained(str(p))


def tokenize_fragment_view(view: str, tokenizer) -> list[int]:
    """Tokenize a space-separated fragment view to body ids (no special tokens)."""
    return tokenizer.encode(view, add_special_tokens=False)


def dedup_records(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Deduplicate by ``inchikey``, keeping first occurrence; preserves order."""
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for r in records:
        if r is None:
            continue
        k = r.get("inchikey")
        if not isinstance(k, str) or k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def flatten_views_to_rows(
    records: Sequence[dict[str, object]],
    *,
    tokenizer=None,
) -> list[dict[str, object]]:
    """Explode each record's ``views`` list into one row per (molecule, view_idx).

    When ``tokenizer`` is set, also stores ``body_ids`` (pretokenized fragment view).
    """
    rows: list[dict[str, object]] = []
    for r in records:
        views = r["views"]
        assert isinstance(views, list)
        for i, v in enumerate(views):
            row: dict[str, object] = {
                "smiles": r["smiles"],
                "inchikey": r["inchikey"],
                "view_idx": i,
                "fragment_view": v,
            }
            if tokenizer is not None:
                row["body_ids"] = tokenize_fragment_view(str(v), tokenizer)
            rows.append(row)
    return rows

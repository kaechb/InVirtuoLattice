"""Molecular preprocessing primitives.

Order of operations (each step is pure and unit-tested):

1. ``standardize_smiles`` — largest fragment, neutralize, canonical tautomer.
2. ``passes_property_filter`` — MW/logP/atom whitelist gate.
3. ``inchikey_of`` — canonical key for deduplication.
4. ``smiles_to_fragmol_views`` — produce K augmented FragMol-notation views per molecule.
5. ``morgan_fingerprint`` — Morgan FP for retrieval / sanity (NOT used for training).

These functions are intentionally side-effect free; the orchestrator in
``run_preprocessing.py`` handles I/O and parallelism.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize

from lattice_lab.paths import ensure_fragmol_on_path

ensure_fragmol_on_path()

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


def smiles_to_fragmol_views(
    smiles: str,
    n_views: int = 3,
    *,
    max_fragments: int = 7,
    seed: int | None = None,
    max_attempts: int = 64,
) -> list[str]:
    """Generate ``n_views`` distinct FragMol-notation strings for ``smiles``.

    Each view cuts the molecule into a **random** number of rBRICS fragments,
    drawn per view in ``[1, min(max_fragments, 7, n_possible)]``, then shuffles
    and renumbers attachment points (``order_fragments_by_attachment_points``).
    Varying the fragment count across views — not just the fragment order —
    makes the SSL contrastive task harder: the two paired views of a molecule
    can differ in granularity, so ``acc@1`` no longer saturates instantly.

    A draw of 1 fragment yields the plain canonical SMILES (no cut), matching
    FragMol's own ``genfrags.py`` handling of single-fragment molecules. The
    7-fragment ceiling also matches ``genfrags.py``
    (``num_fragments = min(num_frags(smiles), 7)``).

    Fragmented views are validity-checked by round-trip canonicalization: only
    views that reconstruct to the input's canonical SMILES are returned.

    Returns at most ``n_views`` items, possibly fewer if rBRICS fails to produce
    distinct valid views within ``max_attempts``.
    """
    # Local import: FragMol modules require sys.path entry that ``paths`` provides.
    from utils.fragments import (
        bridge_smiles_fragments,
        num_frags,
        order_fragments_by_attachment_points,
        smiles2frags,
    )

    rng = random.Random(seed)
    # RDKit raises an (uncatchable) Boost.Python.ArgumentError if CanonSmiles is
    # handed a SMILES that MolFromSmiles can't sanitize (e.g. hypervalent P that
    # slipped past Stage-1 curation). Guard it so a single bad row can't kill a run.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    canon = Chem.MolToSmiles(mol)
    n_possible = num_frags(canon)

    # No rBRICS cut points: the canonical SMILES is the only possible view.
    if n_possible <= 1:
        return [canon]

    # Per-view fragment count is sampled from [1, max_frags]; the 7 cap matches
    # FragMol's genfrags.py training distribution.
    max_frags = min(max_fragments, 7, n_possible)
    views: list[str] = []
    seen: set[str] = set()

    # Reseed FragMol's global random so each call is reproducible if seed != None.
    saved_state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        attempts = 0
        while len(views) < n_views and attempts < max_attempts:
            attempts += 1
            num_fragments = rng.randint(1, max_frags)

            # A draw of 1 means "no cut": the view is the whole molecule's
            # canonical SMILES (same as genfrags.py for single-fragment mols).
            if num_fragments == 1:
                if canon not in seen:
                    seen.add(canon)
                    views.append(canon)
                continue

            frags = smiles2frags(canon, num_fragments)
            if not frags:
                continue
            # FragMol convention: random shuffle then order by attachment points.
            rng.shuffle(frags)
            frags = order_fragments_by_attachment_points(frags)
            try:
                rebuilt = bridge_smiles_fragments(frags)
                rebuilt_canon = Chem.CanonSmiles(rebuilt) if rebuilt else None
            except Exception:
                rebuilt_canon = None
            if rebuilt_canon != canon:
                continue
            view = " ".join(frags)
            if view in seen:
                continue
            seen.add(view)
            views.append(view)
    finally:
        random.setstate(saved_state)
    return views


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
    ``smiles`` (canonical), ``inchikey``, ``views`` (list of FragMol strings).

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
    views = smiles_to_fragmol_views(std, n_views=n_views, seed=seed)
    if not views:
        return None
    return {"smiles": std, "inchikey": key, "views": views}


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


def flatten_views_to_rows(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Explode each record's ``views`` list into one row per (molecule, view_idx)."""
    rows: list[dict[str, object]] = []
    for r in records:
        views = r["views"]
        assert isinstance(views, list)
        for i, v in enumerate(views):
            rows.append(
                {
                    "smiles": r["smiles"],
                    "inchikey": r["inchikey"],
                    "view_idx": i,
                    "fragmol_view": v,
                }
            )
    return rows

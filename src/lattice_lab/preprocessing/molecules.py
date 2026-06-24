"""Molecular preprocessing primitives.

Order of operations (each step is pure and unit-tested):

1. ``standardize_smiles`` — largest fragment, neutralize, canonical tautomer.
2. ``passes_property_filter`` — MW/logP/atom whitelist gate.
3. ``inchikey_of`` — canonical key for deduplication.
4. ``fragment_view`` (faithful, full-coverage) / ``augment_fragment_views`` (SSL aug).
5. ``morgan_fingerprint`` — Morgan FP for retrieval / sanity (NOT used for training).

These functions are intentionally side-effect free; the orchestrator in
``run_preprocessing.py`` handles I/O and parallelism.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    GraphDescriptors,
    QED,
    inchi,
    rdMolDescriptors,
)
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


def _brics_partition(mol: Chem.Mol, bonds_to_cut) -> list[str] | None:
    """Cut ``mol`` at the given BRICS bonds and return the resulting pieces as
    canonical SMILES.

    A true **partition**: every heavy atom lands in exactly one piece, so summed
    over pieces the heavy-atom count is conserved — unlike
    :func:`_brics_fragment_smiles` (``BRICSDecompose``), which returns the
    *deduplicated set of building blocks* and therefore loses repeated fragments.

    Attachment points use **unique same-number pairs**: the ``i``-th cut bond
    becomes ``[i*]`` on *both* resulting stubs, so a view's dummy labels form
    matched pairs that identify which two stubs were bonded (and the view is
    reassemblable). This is why we cut with :func:`Chem.FragmentOnBonds` +
    explicit ``dummyLabels`` rather than :func:`BRICS.BreakBRICSBonds`, whose
    labels encode the BRICS *environment type* (``[12*]`` pairs with ``[5*]``,
    never another ``[12*]``) and so don't form same-number matched pairs.

    ``bonds_to_cut`` is a subset of ``BRICS.FindBRICSBonds(mol)``; cutting fewer
    bonds yields fewer, larger pieces (coarser granularity). Returns ``None`` if
    cutting fails (caller falls back to the whole-molecule SMILES).
    """
    bonds_to_cut = list(bonds_to_cut)
    if not bonds_to_cut:
        smi = Chem.MolToSmiles(mol)
        return [smi] if Chem.MolFromSmiles(smi) is not None else None

    bond_indices: list[int] = []
    for (a1, a2), _labels in bonds_to_cut:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is None:
            return None
        bond_indices.append(bond.GetIdx())
    # Cut i → isotope i+1 on both stubs → matched same-number pair [i+1*]…[i+1*].
    dummy_labels = [(i + 1, i + 1) for i in range(len(bond_indices))]

    try:
        broken = Chem.FragmentOnBonds(
            mol, bond_indices, addDummies=True, dummyLabels=dummy_labels
        )
        pieces = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    except Exception as exc:  # RDKit raises non-standard errors on odd scaffolds.
        logger.debug("BRICS partition failed: %s", exc)
        return None
    out: list[str] = []
    for piece in pieces:
        smi = Chem.MolToSmiles(piece)
        if Chem.MolFromSmiles(smi) is None:
            return None
        out.append(smi)
    return out


def fragment_view(
    smiles: str,
    *,
    merge: bool = False,
    max_pieces: int = 7,
    seed: int | None = None,
) -> str | None:
    """One **faithful**, full-coverage fragment view of the *whole* molecule.

    This is the representation every downstream consumer must use when the
    molecule's ``z_m`` has to faithfully encode the molecule — decoy/binder
    precompute, EBM ranking, eval. It never drops atoms (heavy-atom count is
    conserved across the joined fragments).

    - ``merge=False``: cut *all* BRICS bonds → finest partition, joined in
      canonical (sorted) order — deterministic, no ``seed`` needed.
    - ``merge=True``: cut a random subset of BRICS bonds → a coarser partition
      (``2..max_pieces`` pieces) that still tiles the whole molecule. ``seed``
      makes it reproducible per molecule.

    Returns canonical SMILES for unfragmentable molecules, ``None`` if RDKit
    can't parse ``smiles``.
    """
    from rdkit.Chem import BRICS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)
    bonds = list(BRICS.FindBRICSBonds(mol))
    if not bonds:
        return canon
    if merge:
        rng = random.Random(seed)
        n_pieces = rng.randint(2, min(max_pieces, len(bonds) + 1))
        cut = rng.sample(bonds, min(n_pieces - 1, len(bonds)))
        frags = _brics_partition(mol, cut)
        # A single piece still carrying dummy stubs (only possible if a ring bond
        # was cut, e.g. rBRICS) is a lone unpaired fragment — emit the clean whole
        # molecule instead, matching the merge=False branch below.
        if not frags or len(frags) <= 1:
            return canon
        return " ".join(frags)
    frags = _brics_partition(mol, bonds)
    if not frags or len(frags) <= 1:
        return canon
    return " ".join(sorted(frags))


def augment_fragment_views(
    smiles: str,
    n_views: int = 3,
    *,
    merge: bool = True,
    max_fragments: int = 7,
    seed: int | None = None,
    max_attempts: int = 64,
) -> list[str]:
    """Generate ``n_views`` **full-coverage** fragment views for SSL.

    Every view tiles the *whole* molecule (no atoms dropped) — two composable ops:

    - ``merge``: vary granularity by cutting a random subset of BRICS bonds
      (coarser, still full coverage).
    - order shuffle (always): the discrete-flow SSL datamodule also reshuffles
      online; this just seeds extra distinct stored views.

    Fragment dropping (I-JEPA-style atom masking) is intentionally *not* offered:
    it orphans attachment stubs (e.g. a lone ``[1*]c1ccccc1``) and breaks the
    invertible matched-pair labelling. The SSL datamodule applies any masking
    online instead. For a single faithful view, call :func:`fragment_view`.
    """
    from rdkit.Chem import BRICS

    rng = random.Random(seed)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    canon = Chem.MolToSmiles(mol)
    bonds = list(BRICS.FindBRICSBonds(mol))
    finest = _brics_partition(mol, bonds) if bonds else None
    if not finest or len(finest) <= 1:
        return [canon]

    views: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(views) < n_views and attempts < max_attempts:
        attempts += 1
        frags: list[str] | None = None
        if merge and rng.random() < 0.5:
            n_pieces = rng.randint(2, min(max_fragments, len(bonds) + 1))
            frags = _brics_partition(mol, rng.sample(bonds, min(n_pieces - 1, len(bonds))))
        # A single piece still carrying dummy stubs (only from a ring cut, e.g.
        # rBRICS) is a lone unpaired fragment; fall back to the finest partition.
        if not frags or len(frags) <= 1:
            frags = list(finest)
        rng.shuffle(frags)
        view = " ".join(frags) if frags else canon
        if view not in seen:
            seen.add(view)
            views.append(view)
    return views


def seeded_views(smiles: str, k: int) -> list[str]:
    """Up to ``k`` deterministic, **faithful** multi-granularity views (per-molecule
    seed derived from the SMILES, so runs are reproducible). Used to ensemble a
    molecule's ``z_m`` over fragmentations at eval — every view is full coverage
    (varying coarseness via :func:`fragment_view`), never a lossy subset.

    Defined here (a torch-free module) so ``joblib``/``loky`` workers that call it
    don't cold-import torch at spawn — the slow part of multi-view cache builds.
    """
    import hashlib

    if Chem.MolFromSmiles(smiles) is None:
        return []
    base = int.from_bytes(hashlib.sha1(smiles.encode()).digest()[:4], "little")
    views: list[str] = []
    seen: set[str] = set()
    # View 0 is the deterministic finest decomposition; the rest vary granularity.
    for i in range(max(1, k)):
        v = fragment_view(smiles, merge=(i > 0), seed=base + i)
        if v and v not in seen:
            seen.add(v)
            views.append(v)
    return views or [Chem.CanonSmiles(smiles)]


def fragment_view_for_smiles(smiles: str, *, merge: bool = False) -> str | None:
    """One faithful, full-coverage fragment view for adapter encode; canonical
    SMILES fallback; ``None`` if RDKit can't parse it.

    ``merge=False`` (default): finest BRICS partition. ``merge=True``: coarser
    partition, seeded deterministically from ``smiles``.
    """
    import hashlib

    try:
        seed = None
        if merge:
            seed = int.from_bytes(hashlib.sha1(smiles.encode()).digest()[:4], "little")
        v = fragment_view(smiles, merge=merge, seed=seed)
        if v is not None:
            return v
    except Exception as exc:
        logger.debug("fragment view failed for %r: %s", smiles, exc)
    return Chem.CanonSmiles(smiles) if Chem.MolFromSmiles(smiles) is not None else None


def build_smiles_fragment_views(
    smiles_list: Sequence[str],
    *,
    n_jobs: int = 1,
    merge: bool = False,
) -> dict[str, str]:
    """Map unique SMILES → one fragment view; omits unfragmentable ligands."""
    unique = list(dict.fromkeys(smiles_list))
    if not unique:
        return {}

    def _pair(smi: str) -> tuple[str, str | None]:
        return smi, fragment_view_for_smiles(smi, merge=merge)

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
    merge: bool = False,
) -> dict[str, object] | None:
    """End-to-end per-molecule pipeline for SSL preprocessing.

    Returns a record dict (or ``None`` if the molecule is rejected) with keys:
    ``smiles`` (canonical), ``inchikey``, ``views``. Views are always faithful,
    full-coverage fragmentations (``view_idx==0`` is a lossless molecule for the
    decoy pool); the SSL datamodule applies masking/shuffle online. ``merge``
    bakes coarser multi-granularity partitions into the shards. This is the unit
    dispatched to worker processes by the orchestrator.
    """
    std = standardize_smiles(smiles)
    if std is None:
        return None
    if not passes_property_filter(std, pf):
        return None
    key = inchikey_of(std)
    if key is None:
        return None
    views = augment_fragment_views(std, n_views=n_views, merge=merge, seed=seed)
    if not views:
        return None
    return {"smiles": std, "inchikey": key, "views": views}


# Ordered registry of cheap (2D, no-conformer) RDKit probe descriptors.
#
# ``qed`` and ``molwt`` MUST stay first: ``molecule_qed_molwt`` and the sum-pool
# probe index by position. The split below is the point of the set: the
# "additive" descriptors are ~linear in Morgan-fingerprint bits (weighted
# substructure counts), so a fingerprint-aligned embedding predicts them
# trivially with a *linear* probe — high R² there is near-circular. The
# "structural" descriptors are non-linear functions of the molecular graph
# (saturation / complexity / topology), which a linear probe cannot read off a
# fingerprint-aligned space; a high R² on them is real evidence the encoder
# captured global structure rather than just distilling substructure presence.
_PROBE_DESCRIPTORS: tuple[tuple[str, Callable[[Chem.Mol], float]], ...] = (
    ("qed", QED.qed),
    ("molwt", Descriptors.MolWt),
    ("logp", Crippen.MolLogP),
    ("fraction_csp3", rdMolDescriptors.CalcFractionCSP3),
    ("bertz_ct", GraphDescriptors.BertzCT),
    ("balaban_j", GraphDescriptors.BalabanJ),
)

PROBE_DESCRIPTOR_NAMES: tuple[str, ...] = tuple(name for name, _ in _PROBE_DESCRIPTORS)
PROBE_ADDITIVE_NAMES: tuple[str, ...] = ("qed", "molwt", "logp")
PROBE_STRUCTURAL_NAMES: tuple[str, ...] = ("fraction_csp3", "bertz_ct", "balaban_j")


def molecule_probe_props(smiles: str) -> tuple[float, ...] | None:
    """Return the probe descriptors for ``smiles`` in :data:`PROBE_DESCRIPTOR_NAMES`
    order, or ``None`` if the SMILES is invalid or any descriptor is non-finite
    (e.g. ``BalabanJ`` on a degenerate graph)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vals: list[float] = []
    for _, fn in _PROBE_DESCRIPTORS:
        v = float(fn(mol))
        if not math.isfinite(v):
            return None
        vals.append(v)
    return tuple(vals)


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

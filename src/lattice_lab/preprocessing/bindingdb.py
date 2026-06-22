"""BindingDB TSV parser + curation pipeline.

Reference
---------
Gilson, M. K. et al. *BindingDB in 2024: a FAIR knowledgebase of protein-small
molecule binding data.* Nucleic Acids Res. 53, D1633-D1639 (2025).
https://academic.oup.com/nar/article/53/D1/D1633/7906836

The official ``BindingDB_All.tsv`` already incorporates the in-database
curation flags described in the 2025 paper (BDBC). This module then applies
project-specific cleaning so the rows are usable for LATTICE's EBM:

1.  Restrict to single-chain targets (``Number of Protein Chains in Target == 1``)
    and rows with a UniProt-SwissProt primary id (cleanest target identifier).
2.  Standardise ligand SMILES (largest fragment, neutralise, canonical tautomer)
    and gate on the same MW/logP/atom whitelist used in Stage 1.
3.  Length-filter the target chain sequence to 50..1500 residues (matches
    ``lattice_lab.preprocessing.proteins.filter_length``).
4.  Parse all four affinity columns (Ki/Kd/IC50/EC50), stripping qualifiers
    such as ``>``, ``<``, ``~``. Keep the *minimum* numeric nM value as the
    representative affinity and remember which assay produced it.
5.  De-duplicate by ``(InChIKey, UniProt accession)``; on collision keep the
    row with the lowest affinity (strongest binder).
6.  Compute an ``is_binder_10uM`` boolean (any of Ki/Kd/IC50 ≤ 10000 nM,
    matching DrugCLIP's BioLiP2 fine-tuning policy). All rows are kept so the
    EBM can use annotated non-binders as hard negatives in ablations; the
    column lets default training subset to binders only.

The output is a list of ``BindingDbRow`` dataclasses; the orchestrator in
``lattice_lab.preprocessing.run_bindingdb`` materialises them into parquet shards.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from tqdm.auto import tqdm

from lattice_lab.preprocessing.molecules import (
    PropertyFilter,
    inchikey_of,
    passes_property_filter,
    standardize_smiles,
)
from lattice_lab.preprocessing.proteins import AA_ALPHABET

logger = logging.getLogger(__name__)


BINDER_THRESHOLD_NM = 10_000.0  # 10 µM — DrugCLIP / BioLiP2 fine-tuning policy.

# Column names in BindingDB_All.tsv. Stored as a contract so a header change
# (BindingDB occasionally renames columns) breaks loudly at parse time rather
# than silently producing empty fields.
#
# Target columns are suffixed " 1", " 2", ... per protein chain in the complex.
# We restrict to single-chain targets, so we only ever read the chain-1 columns.
_N_CHAINS_COL = "Number of Protein Chains in Target (>1 implies a multichain complex)"
_SEQUENCE_COL = "BindingDB Target Chain Sequence 1"
_UNIPROT_COL = "UniProt (SwissProt) Primary ID of Target Chain 1"

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Ligand SMILES",
    "BindingDB MonomerID",
    "Target Name",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    _N_CHAINS_COL,
    _SEQUENCE_COL,
    _UNIPROT_COL,
)

_AFFINITY_COLS: tuple[tuple[str, str], ...] = (
    ("Ki (nM)", "Ki"),
    ("Kd (nM)", "Kd"),
    ("IC50 (nM)", "IC50"),
    ("EC50 (nM)", "EC50"),
)


@dataclass(frozen=True)
class BindingDbRow:
    """One curated BindingDB record consumed by Stage 1/2.

    Field order matches the parquet schema written by the orchestrator.
    """

    monomer_id: str
    target_name: str
    uniprot: str
    sequence: str
    smiles: str           # canonical, standardized
    inchikey: str
    ki_nm: float | None
    kd_nm: float | None
    ic50_nm: float | None
    ec50_nm: float | None
    best_nm: float | None   # min(Ki, Kd, IC50, EC50), None if all missing
    best_assay: str         # "Ki" / "Kd" / "IC50" / "EC50" / ""
    is_binder_10uM: bool


def _first_uniprot(raw: str | None) -> str:
    """Normalise the BindingDB UniProt column to a single accession.

    A small fraction of rows list multiple UniProt accessions in the Chain-1
    Primary ID column, separated by whitespace, comma, or semicolon (e.g.
    ``"Q96CA5 P98170"``). FASTA headers terminate at the first whitespace, so
    leaving the raw string in place causes downstream cluster lookups to fail
    when the FASTA-parsed pid (first token) does not match the stored row
    field. We canonicalise to the first non-empty token. Returns ``""`` for
    empty / missing input.
    """
    if not raw:
        return ""
    for sep in (",", ";"):
        raw = raw.replace(sep, " ")
    for tok in raw.split():
        tok = tok.strip()
        if tok:
            return tok
    return ""


def _parse_affinity(raw: str) -> float | None:
    """Parse a single BindingDB affinity cell.

    Cells can be empty, a bare float, or qualifier-prefixed (``>10000``,
    ``<0.1``, ``~5``). We strip qualifiers and return the numeric value;
    callers can choose how to interpret ``>`` later (currently treated as
    upper-bound information, kept as-is).
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    # Strip leading qualifier.
    if s[0] in "><~=":
        s = s[1:].strip()
    # Some rows are ranges "10-100" — keep the lower bound.
    if "-" in s and not s.startswith("-"):
        s = s.split("-")[0].strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v > 0 else None


def _is_canonical_sequence(seq: str, min_len: int = 50, max_len: int = 1500) -> bool:
    if not seq or not (min_len <= len(seq) <= max_len):
        return False
    return set(seq.upper()).issubset(AA_ALPHABET)


def iter_tsv_rows(tsv_path: str | Path) -> Iterator[dict[str, str]]:
    """Yield raw BindingDB rows as dicts. Validates the header once."""
    path = Path(tsv_path)
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty file")
        missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{path}: missing required columns {missing}. "
                "BindingDB may have renamed them — update _REQUIRED_COLUMNS."
            )
        for row in reader:
            yield row


def _curate_row(
    raw: dict[str, str], pf: PropertyFilter | None = None
) -> BindingDbRow | None:
    """Apply per-row curation; return ``None`` if the row is rejected."""
    n_chains = raw.get(_N_CHAINS_COL, "").strip()
    if n_chains != "1":
        return None

    uniprot = _first_uniprot(raw.get(_UNIPROT_COL, ""))
    if not uniprot:
        return None

    sequence = raw.get(_SEQUENCE_COL, "").strip().upper()
    if not _is_canonical_sequence(sequence):
        return None

    smiles_raw = raw.get("Ligand SMILES", "").strip()
    if not smiles_raw:
        return None
    smiles = standardize_smiles(smiles_raw)
    if smiles is None or not passes_property_filter(smiles, pf):
        return None

    key = inchikey_of(smiles)
    if key is None:
        return None

    ki = _parse_affinity(raw.get("Ki (nM)", ""))
    kd = _parse_affinity(raw.get("Kd (nM)", ""))
    ic50 = _parse_affinity(raw.get("IC50 (nM)", ""))
    ec50 = _parse_affinity(raw.get("EC50 (nM)", ""))

    best: tuple[float, str] | None = None
    for value, assay in ((ki, "Ki"), (kd, "Kd"), (ic50, "IC50"), (ec50, "EC50")):
        if value is None:
            continue
        if best is None or value < best[0]:
            best = (value, assay)
    best_nm = best[0] if best else None
    best_assay = best[1] if best else ""

    # DrugCLIP definition: binder iff any of Ki/Kd/IC50 ≤ 10 µM. EC50 is
    # functional (not strictly binding) so excluded from the binder flag.
    is_binder = any(v is not None and v <= BINDER_THRESHOLD_NM for v in (ki, kd, ic50))

    return BindingDbRow(
        monomer_id=raw.get("BindingDB MonomerID", "").strip(),
        target_name=raw.get("Target Name", "").strip(),
        uniprot=uniprot,
        sequence=sequence,
        smiles=smiles,
        inchikey=key,
        ki_nm=ki,
        kd_nm=kd,
        ic50_nm=ic50,
        ec50_nm=ec50,
        best_nm=best_nm,
        best_assay=best_assay,
        is_binder_10uM=is_binder,
    )


def _chunks(it: Iterator[dict[str, str]], size: int) -> Iterator[list[dict[str, str]]]:
    """Group an iterator into fixed-size lists (the last chunk may be shorter)."""
    buf: list[dict[str, str]] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def _curate_chunk(
    chunk: list[dict[str, str]], pf: PropertyFilter | None = None
) -> list[BindingDbRow]:
    """Module-level chunk worker; pickleable so it works under ``spawn``."""
    return [r for r in (_curate_row(x, pf) for x in chunk) if r is not None]


def _estimate_total_rows(tsv_path: Path) -> int | None:
    """Best-effort row count for the progress bar.

    A real ``wc -l`` on a 9 GB file takes ~30 s; instead we sample the first
    1 MB to estimate the average byte-per-row and extrapolate to file size.
    The estimate is only used for the tqdm ETA — it is not load-bearing.
    Returns ``None`` if the file is tiny or sampling fails.
    """
    try:
        size = tsv_path.stat().st_size
        if size < 1 << 20:
            return None
        sample_bytes = 1 << 20
        with open(tsv_path, "rb") as fh:
            sample = fh.read(sample_bytes)
        lines_in_sample = sample.count(b"\n")
        if lines_in_sample < 2:
            return None
        # First sample line is the header; the rest are data rows.
        bytes_per_row = sample_bytes / (lines_in_sample - 1)
        return max(0, int(size / bytes_per_row) - 1)
    except OSError:
        return None


def curate_tsv(
    tsv_path: str | Path,
    *,
    pf: PropertyFilter | None = None,
    n_jobs: int = 1,
    chunk_size: int = 4096,
    limit: int | None = None,
    show_progress: bool = True,
) -> list[BindingDbRow]:
    """Stream the BindingDB TSV through ``_curate_row``, optionally parallel.

    ``n_jobs == 1`` (default) keeps everything in-process — simpler, and SMILES
    standardisation is the bulk of the work so multiprocessing is the main lever.
    ``limit`` is a debugging knob that caps the number of *raw* rows read.
    Progress is reported via a ``tqdm`` bar (set ``show_progress=False`` for
    quiet runs, e.g. in tests).
    """
    path = Path(tsv_path)
    total = limit if limit is not None else _estimate_total_rows(path)

    if n_jobs <= 1:
        kept: list[BindingDbRow] = []
        bar = tqdm(
            iter_tsv_rows(path),
            total=total,
            desc="curate_tsv",
            unit="row",
            unit_scale=True,
            disable=not show_progress,
            mininterval=1.0,
        )
        for i, raw in enumerate(bar):
            if limit is not None and i >= limit:
                break
            row = _curate_row(raw, pf)
            if row is not None:
                kept.append(row)
            if (i + 1) % 50_000 == 0:
                bar.set_postfix(kept=len(kept), refresh=False)
        bar.close()
        return kept

    # Parallel path: dispatch fixed-size chunks of raw dicts to workers. We
    # use a process pool so RDKit's GIL-bound work runs truly in parallel.
    # Progress: wrap the per-row input iterator with tqdm so the bar advances
    # as ``imap_unordered`` pulls rows in the main process — this gives
    # accurate progress even before any worker has finished its first chunk.
    import multiprocessing as mp
    from functools import partial
    from itertools import islice

    stream = iter_tsv_rows(path)
    if limit is not None:
        stream = islice(stream, limit)

    bar = tqdm(
        stream,
        total=total,
        desc=f"curate_tsv (n_jobs={n_jobs})",
        unit="row",
        unit_scale=True,
        disable=not show_progress,
        mininterval=1.0,
    )
    out: list[BindingDbRow] = []
    with mp.get_context("spawn").Pool(processes=n_jobs) as pool:
        for batch_out in pool.imap_unordered(
            partial(_curate_chunk, pf=pf), _chunks(bar, chunk_size)
        ):
            out.extend(batch_out)
            bar.set_postfix(kept=len(out), refresh=False)
    bar.close()
    return out


def dedup_rows(rows: list[BindingDbRow]) -> list[BindingDbRow]:
    """De-duplicate by ``(inchikey, uniprot)``; keep the strongest affinity."""
    best: dict[tuple[str, str], BindingDbRow] = {}
    for r in rows:
        key = (r.inchikey, r.uniprot)
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        # Lower nM = stronger binding. None loses to a numeric value.
        if r.best_nm is not None and (cur.best_nm is None or r.best_nm < cur.best_nm):
            best[key] = r
    return list(best.values())


def row_from_record(rec: dict[str, object]) -> BindingDbRow:
    """Build a row from a parquet dict, ignoring extra columns (``fragment_view``, etc.)."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(BindingDbRow)}
    return BindingDbRow(**{k: v for k, v in rec.items() if k in fields})


def rows_to_records(rows: list[BindingDbRow]) -> list[dict[str, object]]:
    """Flatten rows into plain dicts (parquet-friendly)."""
    return [asdict(r) for r in rows]


def ligand_view_records(
    rows: list[BindingDbRow],
    *,
    views: dict[str, str],
    body_ids: dict[str, list[int]] | None = None,
) -> list[dict[str, object]]:
    """Attach ``fragment_view`` / optional ``body_ids`` per ligand SMILES."""
    out: list[dict[str, object]] = []
    for r in rows:
        d = asdict(r)
        d["fragment_view"] = views.get(r.smiles)
        if body_ids is not None:
            d["body_ids"] = body_ids.get(r.smiles)
        out.append(d)
    return out

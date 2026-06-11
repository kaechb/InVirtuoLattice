"""Stage 1 orchestrator.

Reads a CSV/TXT of SMILES, runs the per-molecule pipeline in parallel, deduplicates
by InChIKey, and writes parquet shards of ~``rows_per_shard`` rows each. Shard
filenames are deterministic (``shard_0000.parquet``) so re-running with the same
output directory is idempotent: existing shards are skipped.

Usage::

    python -m lattice.preprocessing.run_preprocessing \\
        --input data/raw.smi --output artifacts/processed/moses/ \\
        --n-views 3 --n-jobs 16
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from lattice_lab.preprocessing.molecules import (
    PropertyFilter,
    dedup_records,
    flatten_views_to_rows,
    process_smiles_record,
)

logger = logging.getLogger(__name__)


def _read_smiles(path: Path) -> list[str]:
    """Read SMILES from .smi / .csv / .txt. Assumes one SMILES per line, optional header."""
    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
        return df.iloc[:, 0].astype(str).tolist()
    out: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tok = line.split()[0]
            if tok.lower() == "smiles":  # skip header
                continue
            out.append(tok)
    return out


def _batched(iterable: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def run(
    input_path: Path,
    output_dir: Path,
    *,
    n_views: int = 3,
    rows_per_shard: int = 1_000_000,
    n_jobs: int | None = None,
    chunk_size: int = 50_000,
    seed: int = 0,
    filter_overrides: dict[str, float] | None = None,
) -> dict[str, int]:
    """Run the preprocessing pipeline. Returns a summary dict with row counts.

    Idempotency: if ``output_dir`` already contains ``shard_NNNN.parquet`` files,
    they are kept and the next available index is used for new shards. Pipeline
    output is fully determined by inputs + ``seed`` per molecule.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)

    pf = PropertyFilter()
    if filter_overrides:
        pf = PropertyFilter(**{**pf.__dict__, **filter_overrides})

    smiles_list = _read_smiles(input_path)
    logger.info("read %d SMILES from %s", len(smiles_list), input_path)

    fn = partial(process_smiles_record, n_views=n_views, pf=pf)

    accumulated_records: list[dict[str, object]] = []
    for batch in tqdm(list(_batched(smiles_list, chunk_size)), desc="batches"):
        seeded = [(smi, seed + i) for i, smi in enumerate(batch)]
        # Each worker reseeds RDKit's process-local random via ``seed`` kwarg below.
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(fn)(smi, seed=s) for smi, s in seeded
        )
        for r in results:
            if r is not None:
                accumulated_records.append(r)

    records = dedup_records(accumulated_records)
    rows = flatten_views_to_rows(records)
    logger.info("kept %d molecules → %d view rows", len(records), len(rows))

    # Detect next shard index for idempotent re-runs.
    existing = sorted(output_dir.glob("shard_*.parquet"))
    next_idx = len(existing)

    for shard_i, start in enumerate(range(0, len(rows), rows_per_shard)):
        df = pd.DataFrame(rows[start : start + rows_per_shard])
        out_path = output_dir / f"shard_{next_idx + shard_i:04d}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("wrote %s (%d rows)", out_path.name, len(df))

    return {
        "input": len(smiles_list),
        "kept_molecules": len(records),
        "rows": len(rows),
        "shards_written": (len(rows) + rows_per_shard - 1) // rows_per_shard,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="SMILES file (.smi/.csv/.txt)")
    parser.add_argument("--output", required=True, type=Path, help="output directory for parquet shards")
    parser.add_argument("--n-views", type=int, default=3, help="FragMol augmentations per molecule")
    parser.add_argument("--rows-per-shard", type=int, default=1_000_000)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    mp.set_start_method("spawn", force=True)
    summary = run(
        input_path=args.input,
        output_dir=args.output,
        n_views=args.n_views,
        rows_per_shard=args.rows_per_shard,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    logger.info("summary: %s", summary)


if __name__ == "__main__":
    main()

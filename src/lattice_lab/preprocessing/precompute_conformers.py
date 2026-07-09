"""Stage: precompute + cache one 3D conformer per molecule.

Reads the MOSES fragment-view parquet shards (``shard_*.parquet``), deduplicates
molecules by InChIKey (``view_idx == 0``), generates a single RDKit ETKDGv3
heavy-atom conformer per SMILES (in parallel), drops failures, and writes a
``conformers.parquet`` cache with columns ``[inchikey, atoms, coords]``:

* ``atoms``  — space-joined heavy-atom element symbols (e.g. ``"C C O"``)
* ``coords`` — row-major flattened ``float32`` xyz (length ``3 * n_atoms``)

Load it back with :func:`lattice_lab.data.conformers.load_conformer_cache`.

Usage::

    python -m lattice_lab.preprocessing.precompute_conformers \\
        --shard-dir artifacts/preprocessing/processed/moses/ \\
        --output artifacts/preprocessing/processed/moses/conformers.parquet \\
        --n-jobs 16
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from lattice_lab.data.conformers import generate_conformer, remove_hydrogens, normalize_coordinates

logger = logging.getLogger(__name__)


def _load_unique_molecules(shard_dir: Path) -> pd.DataFrame:
    """Unique ``(inchikey, smiles)`` rows (``view_idx == 0``) across all shards."""
    shards = sorted(shard_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {shard_dir}")
    frames = [pd.read_parquet(s, columns=["inchikey", "view_idx", "smiles"]) for s in shards]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["view_idx"] == 0].drop_duplicates("inchikey").reset_index(drop=True)
    return df[["inchikey", "smiles"]]


def _one(key: str, smiles: str, seed: int) -> dict | None:
    ac = generate_conformer(smiles, seed=seed)
    if ac is None:
        return None
    atoms, coords = remove_hydrogens(*ac)
    if len(atoms) == 0:
        return None
    coords = normalize_coordinates(coords)
    return {
        "key": key,
        "atoms": " ".join(map(str, atoms.tolist())),
        "coords": coords.reshape(-1).astype(np.float32).tolist(),
    }


def _generate(
    pairs: list[tuple[str, str]], key_col: str, *, n_jobs: int, seed: int
) -> pd.DataFrame:
    """Generate one conformer per ``(key, smiles)`` pair → ``[key_col, atoms, coords]``."""
    records = Parallel(n_jobs=n_jobs)(
        delayed(_one)(k, s, seed)
        for k, s in tqdm(pairs, total=len(pairs), desc="conformers")
    )
    ok = [r for r in records if r is not None]
    n_fail = len(records) - len(ok)
    logger.info("conformers: %d ok, %d failed (%.1f%%)", len(ok), n_fail,
                100.0 * n_fail / max(len(records), 1))
    df = pd.DataFrame(ok, columns=["key", "atoms", "coords"])
    return df.rename(columns={"key": key_col})


def _load_unique_from_parquet(
    parquets: list[Path], key_col: str, smiles_col: str
) -> list[tuple[str, str]]:
    """Unique ``(key, smiles)`` rows across parquets, deduped on ``key_col``."""
    seen: dict[str, str] = {}
    for p in parquets:
        df = pd.read_parquet(p, columns=[key_col, smiles_col])
        for k, s in zip(df[key_col].astype(str), df[smiles_col].astype(str)):
            if k not in seen:
                seen[k] = s
    return list(seen.items())


def run(
    shard_dir: str | Path,
    output: str | Path,
    *,
    n_jobs: int = 8,
    seed: int = 42,
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    shard_dir = Path(shard_dir)
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace")

    df = _load_unique_molecules(shard_dir)
    if limit is not None:
        df = df.head(limit)
    logger.info("generating conformers for %d unique molecules (n_jobs=%d)", len(df), n_jobs)
    pairs = list(zip(df["inchikey"].astype(str), df["smiles"].astype(str)))
    out_df = _generate(pairs, "inchikey", n_jobs=n_jobs, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)
    logger.info("wrote %d conformers -> %s", len(out_df), output)
    return output


def run_from_parquet(
    parquets: list[str | Path],
    output: str | Path,
    *,
    key_col: str = "inchikey",
    smiles_col: str = "smiles",
    n_jobs: int = 8,
    seed: int = 42,
    limit: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Conformer cache keyed by ``key_col`` from arbitrary parquet(s).

    Used for the Stage-4 hard-negative (BindingDB, keyed by ``inchikey``) and
    binder (keyed by ``smiles``) pools, whose ligands have no MOSES-shard
    conformer cache. Output columns: ``[key_col, atoms, coords]`` — readable via
    :func:`lattice_lab.data.conformers.load_conformer_cache` with the same
    ``key_col``.
    """
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace")
    pairs = _load_unique_from_parquet([Path(p) for p in parquets], key_col, smiles_col)
    if limit is not None:
        pairs = pairs[:limit]
    logger.info(
        "generating conformers for %d unique %s (n_jobs=%d)", len(pairs), key_col, n_jobs
    )
    out_df = _generate(pairs, key_col, n_jobs=n_jobs, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output, index=False)
    logger.info("wrote %d conformers -> %s", len(out_df), output)
    return output


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--shard-dir", help="dir with MOSES shard_*.parquet (inchikey-keyed)")
    src.add_argument("--parquet", action="append", default=[],
                     help="source parquet(s); repeatable. Keyed by --key-col")
    ap.add_argument("--key-col", default="inchikey",
                    help="key column for --parquet mode (inchikey | smiles)")
    ap.add_argument("--smiles-col", default="smiles", help="SMILES column for --parquet mode")
    ap.add_argument("--output", required=True, help="output conformers.parquet path")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="cap #molecules (debug)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.shard_dir:
        run(
            args.shard_dir, args.output,
            n_jobs=args.n_jobs, seed=args.seed, limit=args.limit, overwrite=args.overwrite,
        )
    else:
        run_from_parquet(
            args.parquet, args.output,
            key_col=args.key_col, smiles_col=args.smiles_col,
            n_jobs=args.n_jobs, seed=args.seed, limit=args.limit, overwrite=args.overwrite,
        )


if __name__ == "__main__":
    _main()

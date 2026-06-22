"""Stage 1 — BindingDB curation + homology-filtered split (DrugCLIP-style).

The split is filtered against a **held-out reference benchmark** — either
LIT-PCBA (``--lit-pcba-dir``, the default) or DUD-E (``--dude-dir``). DrugCLIP
reports a *separate* single model per benchmark, each trained on data with the
targets of *that* benchmark removed at 90 % identity, so the two references
produce two different splits; pass exactly one.

End-to-end pipeline:

    1.  Stream the BindingDB-All TSV through ``bindingdb.curate_tsv``
        (standardise SMILES, length-filter sequence, parse Ki/Kd/IC50/EC50).
    2.  De-duplicate by (InChIKey, UniProt) keeping the strongest affinity.
    3.  Build the reference target sequences (LIT-PCBA or DUD-E) for the
        cross-FASTA homology filter.
    4.  Run ``mmseqs easy-search`` once at the *lowest* requested identity
        cutoff (default 30 %), then derive exclusion sets at 30 / 60 / 90 %
        from the same hit table — matches DrugCLIP's reported thresholds.
    5.  For each kept-threshold, drop BindingDB rows whose UniProt's sequence
        is too similar to any reference target, then split the remainder
        into train/val (90/10) at the **UniProt-cluster** level (all rows
        sharing a cluster go to one side — no target leakage).
    6.  Write parquet shards + a manifest per threshold. DUD-E outputs carry a
        ``_dude`` suffix so both references can share one ``--output-dir`` (and
        reuse the expensive ``bindingdb_curated.parquet``) without colliding:

            <out>/threshold_90{suffix}/{train.parquet, val.parquet}   # suffix="" for LIT-PCBA, "_dude" for DUD-E
            <out>/threshold_60{suffix}/...
            <out>/threshold_30{suffix}/...
            <out>/bindingdb_targets.fasta                              # shared (reference-independent)
            <out>/manifest{suffix}.json
            <out>/test_lit_pcba.parquet + lit_pcba_targets.fasta       # LIT-PCBA only

    The DUD-E **test** parquet + FASTA are produced separately by
    ``lattice_lab.preprocessing.run_dude`` (→ ``artifacts/preprocessing/processed/moses_dude/``);
    this script only needs the DUD-E target *sequences* to filter against.

The script is **idempotent**: parquet files that already exist are skipped
unless ``--overwrite`` is set.

Examples::

    # 90 % split held out against LIT-PCBA (the released default).
    python -m lattice_lab.preprocessing.run_bindingdb \\
        --bindingdb-tsv artifacts/preprocessing/raw/bindingdb/BindingDB_All.tsv \\
        --lit-pcba-dir  artifacts/preprocessing/raw/lit_pcba \\
        --output-dir    artifacts/preprocessing/processed/bindingdb \\
        --identity      all --n-jobs 16

    # 90 % split held out against DUD-E (reuses the cached curated parquet).
    python -m lattice_lab.preprocessing.run_bindingdb \\
        --bindingdb-tsv artifacts/preprocessing/raw/bindingdb/BindingDB_All.tsv \\
        --dude-dir      artifacts/preprocessing/raw/dude \\
        --output-dir    artifacts/preprocessing/processed/bindingdb \\
        --identity      90 --n-jobs 16
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from lattice_lab.preprocessing import bindingdb, dude, lit_pcba
from lattice_lab.preprocessing.homology import (
    DRUGCLIP_THRESHOLDS,
    excluded_at,
    max_identity_per_query,
    mmseqs_easy_search,
)
from lattice_lab.preprocessing.molecules import (
    build_smiles_fragment_views,
    load_smiles_tokenizer,
    tokenize_fragment_view,
)
from lattice_lab.preprocessing.proteins import ProteinRecord, cluster_proteins
from lattice_lab.preprocessing.splits import cluster_split

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_thresholds(arg: str) -> list[float]:
    if arg.lower() == "all":
        return list(DRUGCLIP_THRESHOLDS)
    out: list[float] = []
    for tok in arg.split(","):
        v = float(tok.strip())
        if v > 1.0:
            v /= 100.0
        out.append(v)
    return out


def _normalise_uniprot_in_rows(
    rows: list[bindingdb.BindingDbRow],
) -> tuple[list[bindingdb.BindingDbRow], int]:
    """Replace multi-token UniProt strings with the first accession.

    Returns the (possibly new) row list and the number of rows that needed
    fixing. We rebuild rows immutably because ``BindingDbRow`` is a frozen
    dataclass.
    """
    fixed: list[bindingdb.BindingDbRow] = []
    n_fixed = 0
    for r in rows:
        canon = bindingdb._first_uniprot(r.uniprot)
        if canon and canon != r.uniprot:
            fixed.append(replace(r, uniprot=canon))
            n_fixed += 1
        else:
            fixed.append(r)
    return fixed, n_fixed


def _write_targets_fasta(rows: list[bindingdb.BindingDbRow], out_path: Path) -> dict[str, str]:
    """One FASTA entry per (unique) UniProt id in the curated BindingDB rows."""
    uniq: dict[str, str] = {}
    for r in rows:
        uniq.setdefault(r.uniprot, r.sequence)
    with open(out_path, "w") as fh:
        for pid, seq in uniq.items():
            fh.write(f">{pid}\n{seq}\n")
    return uniq


def _write_parquet(records: list[dict[str, object]], out_path: Path) -> None:
    if not records:
        # An empty parquet is fine; write a zero-row frame with the schema.
        pd.DataFrame(records).to_parquet(out_path, index=False)
        return
    pd.DataFrame.from_records(records).to_parquet(out_path, index=False)


def _parquet_has_columns(path: Path, *cols: str) -> bool:
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    return all(c in names for c in cols)


def _needs_ligand_view_enrich(path: Path, *, want_body_ids: bool) -> bool:
    if not path.is_file():
        return True
    need = ["fragment_view"]
    if want_body_ids:
        need.append("body_ids")
    return not _parquet_has_columns(path, *need)


def _views_from_parquet(df: pd.DataFrame) -> dict[str, str]:
    if "fragment_view" not in df.columns:
        return {}
    sub = df.drop_duplicates("smiles")
    return {
        str(s): str(v)
        for s, v in zip(sub["smiles"], sub["fragment_view"])
        if pd.notna(v)
    }


def _body_ids_from_parquet(df: pd.DataFrame) -> dict[str, list[int]]:
    if "body_ids" not in df.columns:
        return {}
    out: dict[str, list[int]] = {}
    sub = df.drop_duplicates("smiles")
    for s, b in zip(sub["smiles"], sub["body_ids"]):
        if b is None or (isinstance(b, float) and np.isnan(b)):
            continue
        out[str(s)] = list(b) if not isinstance(b, list) else b
    return out


def _ligand_view_maps(
    rows: list[bindingdb.BindingDbRow],
    curated_parquet: Path,
    *,
    overwrite: bool,
    n_jobs: int,
    tokenizer_path: Path | None,
) -> tuple[dict[str, str], dict[str, list[int]] | None]:
    """Load or compute SMILES → fragment_view (+ optional body_ids)."""
    want_body = tokenizer_path is not None
    views: dict[str, str] = {}
    body_ids: dict[str, list[int]] | None = None

    if curated_parquet.is_file() and not overwrite:
        df = pd.read_parquet(curated_parquet)
        views = _views_from_parquet(df)
        if want_body and _parquet_has_columns(curated_parquet, "body_ids"):
            body_ids = _body_ids_from_parquet(df)

    if not views:
        unique = list(dict.fromkeys(r.smiles for r in rows))
        logger.info("fragmentizing %d unique BindingDB ligands (n_jobs=%d)", len(unique), n_jobs)
        views = build_smiles_fragment_views(unique, n_jobs=n_jobs)
        logger.info("fragment views: %d / %d unique ligands", len(views), len(unique))

    if want_body and body_ids is None:
        tok = load_smiles_tokenizer(tokenizer_path)
        logger.info("pretokenizing fragment views with %s", tokenizer_path)
        body_ids = {s: tokenize_fragment_view(v, tok) for s, v in views.items()}

    return views, body_ids if want_body else None


def _enrich_split_parquet(
    path: Path,
    *,
    views: dict[str, str],
    body_ids: dict[str, list[int]] | None,
) -> None:
    if not _needs_ligand_view_enrich(path, want_body_ids=body_ids is not None):
        return
    df = pd.read_parquet(path)
    df["fragment_view"] = df["smiles"].map(views)
    if body_ids is not None:
        df["body_ids"] = df["smiles"].map(body_ids)
    df.to_parquet(path, index=False)
    logger.info("enriched %s with ligand views", path)


def _split_by_uniprot_cluster(
    rows: list[bindingdb.BindingDbRow],
    seq_by_uniprot: dict[str, str],
    *,
    train: float,
    val: float,
    cluster_identity: float,
    seed: int,
    workdir: Path,
) -> tuple[list[bindingdb.BindingDbRow], list[bindingdb.BindingDbRow]]:
    """Cluster targets, then bucket rows into train/val cluster-disjointly.

    A 0-size test bucket is requested from ``cluster_split`` (LIT-PCBA is the
    held-out test set, not these rows).
    """
    proteins = [ProteinRecord(pid=p, sequence=s) for p, s in seq_by_uniprot.items()]
    pid_to_cluster = cluster_proteins(proteins, min_identity=cluster_identity, workdir=workdir)
    cluster_ids = [pid_to_cluster[r.uniprot] for r in rows]
    split = cluster_split(cluster_ids, train=train, val=val, test=0.0, seed=seed)
    # ``cluster_split`` rounds independently, so train+val can be < len(rows);
    # the remainder sits in split["test"]. We have no test split (LIT-PCBA is
    # test), so fold it back into train to avoid silently dropping rows.
    train_idx = list(split["train"]) + list(split["test"])
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in split["val"]]
    return train_rows, val_rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bindingdb-tsv", type=Path, required=True,
                   help="Path to BindingDB_All.tsv (produced by scripts/download_bindingdb.sh).")
    p.add_argument("--lit-pcba-dir", type=Path, default=None,
                   help="Directory with one subfolder per LIT-PCBA target (artifacts/preprocessing/raw/lit_pcba). "
                        "Held-out reference for the homology filter; mutually exclusive with --dude-dir.")
    p.add_argument("--dude-dir", type=Path, default=None,
                   help="Directory with one subfolder per DUD-E target (artifacts/preprocessing/raw/dude). "
                        "Build the split held out against DUD-E instead of LIT-PCBA; "
                        "mutually exclusive with --lit-pcba-dir.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--identity", type=str, default="all",
                   help='Either "all" (30,60,90 — default) or a comma-separated list, e.g. "90" or "30,90".')
    p.add_argument("--cluster-identity", type=float, default=0.4,
                   help="MMseqs2 identity for protein clustering used by the train/val splitter.")
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument(
        "--tokenizer-path", type=Path, default=None,
        help="optional SMILES tokenizer json; adds body_ids per ligand fragment_view",
    )
    p.add_argument("--row-limit", type=int, default=None,
                   help="Cap the number of raw TSV rows (debugging only).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run all stages even if output files already exist.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    _setup_logging(args.verbose)

    if abs(args.train_frac + args.val_frac - 1.0) > 1e-6:
        raise SystemExit("--train-frac + --val-frac must sum to 1.0 (test = the held-out benchmark, kept separately).")

    if (args.lit_pcba_dir is None) == (args.dude_dir is None):
        raise SystemExit("provide exactly one held-out reference benchmark: --lit-pcba-dir or --dude-dir.")

    thresholds = sorted(_parse_thresholds(args.identity))
    logger.info("identity thresholds: %s", thresholds)

    out_root: Path = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Held-out reference targets (for the homology filter) ------
    # DUD-E outputs are suffixed so a DUD-E split can live beside the LIT-PCBA
    # one in the same --output-dir (sharing bindingdb_curated.parquet).
    if args.dude_dir is not None:
        bench, suffix = "dude", "_dude"
        logger.info("loading DUD-E reference targets from %s", args.dude_dir)
        ref_targets = dude.load_all(args.dude_dir)
        if not ref_targets:
            raise SystemExit(f"no DUD-E targets found in {args.dude_dir}")
        # DUD-E's test parquet + FASTA are written by lattice_lab.preprocessing.run_dude
        # (-> artifacts/preprocessing/processed/moses_dude/); here we only need the sequences.
    else:
        bench, suffix = "lit_pcba", ""
        ref_fasta = out_root / "lit_pcba_targets.fasta"
        ref_parquet = out_root / "test_lit_pcba.parquet"
        if not ref_fasta.exists() or not ref_parquet.exists() or args.overwrite:
            logger.info("loading LIT-PCBA from %s", args.lit_pcba_dir)
            ref_targets = lit_pcba.load_all(args.lit_pcba_dir)
            if not ref_targets:
                raise SystemExit(f"no LIT-PCBA targets found in {args.lit_pcba_dir}")
            lit_pcba.write_fasta(ref_targets, ref_fasta)
            _write_parquet(lit_pcba.to_records(ref_targets), ref_parquet)
            logger.info("LIT-PCBA: %d targets, %d ligand rows",
                        len(ref_targets), sum(len(t.ligands) for t in ref_targets))
        else:
            logger.info("LIT-PCBA outputs already present — re-using.")
            ref_targets = lit_pcba.load_all(args.lit_pcba_dir)

    ref_seqs = {t.name: t.sequence for t in ref_targets}
    logger.info("%s reference targets: %d", bench, len(ref_seqs))

    # ---------- 2. BindingDB curation ---------------------------------------
    curated_parquet = out_root / "bindingdb_curated.parquet"
    want_body = args.tokenizer_path is not None
    rows: list[bindingdb.BindingDbRow]
    rewrite_curated = args.overwrite or not curated_parquet.exists()

    if curated_parquet.exists() and not args.overwrite:
        logger.info("re-using cached %s", curated_parquet)
        cur_df = pd.read_parquet(curated_parquet)
        rows = [bindingdb.row_from_record(rec) for rec in cur_df.to_dict("records")]
        # Salvage path: earlier curation runs stored multi-token UniProt strings
        # ("Q96CA5 P98170"). Normalise to the first accession + re-dedup; rewrite
        # the cache so subsequent runs see clean rows.
        n_before = len(rows)
        rows, n_fixed = _normalise_uniprot_in_rows(rows)
        if n_fixed:
            rows = bindingdb.dedup_rows(rows)
            logger.info(
                "patched %d cached rows with multi-token UniProt; deduped %d -> %d, rewriting cache",
                n_fixed, n_before, len(rows),
            )
            rewrite_curated = True
        elif _needs_ligand_view_enrich(curated_parquet, want_body_ids=want_body):
            rewrite_curated = True
    else:
        logger.info("curating BindingDB TSV %s (n_jobs=%d)", args.bindingdb_tsv, args.n_jobs)
        rows = bindingdb.curate_tsv(
            args.bindingdb_tsv, n_jobs=args.n_jobs, limit=args.row_limit
        )
        rows = bindingdb.dedup_rows(rows)
        logger.info("kept %d curated, deduplicated BindingDB rows", len(rows))

    if not rows:
        raise SystemExit("no rows survived BindingDB curation — check the input TSV.")

    views, body_ids = _ligand_view_maps(
        rows,
        curated_parquet,
        overwrite=args.overwrite,
        n_jobs=args.n_jobs,
        tokenizer_path=args.tokenizer_path,
    )
    if rewrite_curated:
        _write_parquet(
            bindingdb.ligand_view_records(rows, views=views, body_ids=body_ids),
            curated_parquet,
        )
        logger.info(
            "wrote %s (%d rows; %d unique ligands with fragment_view)",
            curated_parquet, len(rows), len(views),
        )

    # ---------- 3. Cross-FASTA homology search ------------------------------
    bdb_fasta = out_root / "bindingdb_targets.fasta"
    if bdb_fasta.exists() and not args.overwrite:
        bdb_seqs: dict[str, str] = {}
        for r in rows:
            bdb_seqs.setdefault(r.uniprot, r.sequence)
    else:
        bdb_seqs = _write_targets_fasta(rows, bdb_fasta)
    logger.info("BindingDB unique targets: %d", len(bdb_seqs))

    min_threshold = min(thresholds)
    hits_workdir = out_root / f"_mmseqs_search{suffix}"
    hits_workdir.mkdir(parents=True, exist_ok=True)
    hits = mmseqs_easy_search(
        bdb_seqs, ref_seqs, min_identity=min_threshold, workdir=hits_workdir
    )
    max_id = max_identity_per_query(hits)
    logger.info("BindingDB targets with any %s hit (>= %.0f%%): %d",
                bench, min_threshold * 100, len(max_id))

    # ---------- 4. Per-threshold split + parquet writes ---------------------
    manifest: dict[str, object] = {
        "bindingdb_tsv": str(args.bindingdb_tsv),
        "reference_benchmark": bench,
        "reference_dir": str(args.dude_dir if args.dude_dir is not None else args.lit_pcba_dir),
        "reference_n_targets": len(ref_seqs),
        "n_curated_rows": len(rows),
        "n_curated_targets": len(bdb_seqs),
        "thresholds": thresholds,
        "cluster_identity": args.cluster_identity,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path else None,
        "n_ligands_with_fragment_view": len(views),
        "per_threshold": {},
    }

    for th in thresholds:
        sub_dir = out_root / f"threshold_{int(th * 100):02d}{suffix}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        train_parquet = sub_dir / "train.parquet"
        val_parquet = sub_dir / "val.parquet"

        excluded_uniprots = excluded_at(max_id, th)
        kept_rows = [r for r in rows if r.uniprot not in excluded_uniprots]
        kept_seqs = {p: s for p, s in bdb_seqs.items() if p not in excluded_uniprots}
        logger.info(
            "threshold %.0f%%: excluded %d / %d targets -> %d rows / %d unique targets",
            th * 100, len(excluded_uniprots), len(bdb_seqs),
            len(kept_rows), len(kept_seqs),
        )

        if train_parquet.exists() and val_parquet.exists() and not args.overwrite:
            _enrich_split_parquet(train_parquet, views=views, body_ids=body_ids)
            _enrich_split_parquet(val_parquet, views=views, body_ids=body_ids)
            logger.info("re-using %s", sub_dir)
            train_df = pd.read_parquet(train_parquet)
            val_df = pd.read_parquet(val_parquet)
        else:
            train_rows, val_rows = _split_by_uniprot_cluster(
                kept_rows,
                kept_seqs,
                train=args.train_frac,
                val=args.val_frac,
                cluster_identity=args.cluster_identity,
                seed=args.seed,
                workdir=sub_dir / "_mmseqs_cluster",
            )
            _write_parquet(
                bindingdb.ligand_view_records(train_rows, views=views, body_ids=body_ids),
                train_parquet,
            )
            _write_parquet(
                bindingdb.ligand_view_records(val_rows, views=views, body_ids=body_ids),
                val_parquet,
            )
            train_df = pd.read_parquet(train_parquet)
            val_df = pd.read_parquet(val_parquet)

        manifest["per_threshold"][f"{int(th * 100):02d}"] = {
            "n_excluded_targets": len(excluded_uniprots),
            "excluded_uniprots": sorted(excluded_uniprots),
            "n_train_rows": int(len(train_df)),
            "n_val_rows": int(len(val_df)),
            "n_train_targets": int(train_df["uniprot"].nunique()) if len(train_df) else 0,
            "n_val_targets": int(val_df["uniprot"].nunique()) if len(val_df) else 0,
            "n_train_binders": int(train_df["is_binder_10uM"].sum()) if len(train_df) else 0,
            "n_val_binders": int(val_df["is_binder_10uM"].sum()) if len(val_df) else 0,
        }

    manifest_path = out_root / f"manifest{suffix}.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    logger.info("wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()

"""Precompute frozen-adapter ``z_m`` for every BindingDB ligand.

Companion to :mod:`lattice.ebm.precompute_decoys`. The MOSES decoy pool gives
us drug-likeness-style random negatives, but the InfoNCE shortcut we saw in
the first training run (`cross_target_viol ≈ 0.9`) shows that random
negatives don't force the head to use the protein latent at all.

The fix is to mix in **experimental** decoys per the hard-negative recipe
used by BigBind / DrugCLIP-DUDE:

- **Other-target binders**: drug-like molecules that bind *some other*
  protein. The only way to score them lower than the true binder for the
  current target is to actually use ``z_p``.
- **Annotated non-binders**: BindingDB rows with ``is_binder_10uM=False``
  (Ki/IC50/Kd > 10 µM in the experimental assay). Real molecules tested
  against the same protein and shown not to bind.

This script writes a second :class:`EmbeddingStore` at the chosen path
(default ``artifacts/decoys/bdb_zm/``) plus a sidecar parquet
(``index.parquet``) with one row per unique InChIKey carrying::

    inchikey, is_binder_any_target

The collator at training time joins the two: pool row index → InChIKey →
``is_binder_any_target`` flag selects which sub-pool to draw from. We do
*not* track per-UniProt binder lists here — the collision rate between a
random "other-target binder" draw and the current target's own binders is
<0.1 % at 1.2 M binders / 7 K targets, well below the false-negative rate
of MOSES decoys.

Idempotent on InChIKey: re-running adds only the missing rows.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder
from lattice_lab.eval.encode_utils import encode_views_inference
from lattice_lab.models.builders import (
    adapter_run_id,
    build_eval_encoder,
    merge_from_ckpt,
    zm_store_path,
)
from lattice_lab.preprocessing.molecules import fragment_view_for_smiles
from lattice_lab.protein.store import EmbeddingStore

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

INDEX_FILE = "index.parquet"


def _clear_store(store_path: Path) -> None:
    """Delete an existing store's files so ``create`` rebuilds it from scratch."""
    import shutil

    removed = False
    for name in (EmbeddingStore.MANIFEST, EmbeddingStore.PIDS, EmbeddingStore.MEAN, INDEX_FILE):
        f = store_path / name
        if f.exists():
            f.unlink()
            removed = True
    perres = store_path / EmbeddingStore.PERRES_DIR
    if perres.is_dir():
        shutil.rmtree(perres)
        removed = True
    if removed:
        logger.warning("--force: cleared existing bdb_zm store at %s", store_path)


# --------------------------------------------------------------------------
# Helpers (shared shape with lit_pcba.evaluate)
# --------------------------------------------------------------------------


def _inchikey_or_none(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol) or None


def _fragment_view_or_canon(smiles: str) -> str | None:
    return fragment_view_for_smiles(smiles)


def _views_for_todo(
    todo_inchikeys: list[str],
    todo_smiles: list[str],
    fv_by_ik: pd.Series | None,
    *,
    n_jobs: int,
) -> list[str | None]:
    """Resolve fragment views; use parquet column when present, else BRICS."""
    views: list[str | None] = [None] * len(todo_smiles)
    need: list[int] = []
    if fv_by_ik is not None:
        for i, ik in enumerate(todo_inchikeys):
            v = fv_by_ik.get(ik)
            if v is not None and pd.notna(v):
                views[i] = str(v)
            else:
                need.append(i)
    else:
        need = list(range(len(todo_smiles)))

    if not need:
        return views

    smiles_need = [todo_smiles[i] for i in need]
    if n_jobs in (0, 1):
        computed = [_fragment_view_or_canon(s) for s in smiles_need]
    else:
        from joblib import Parallel, delayed

        computed = list(
            Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_fragment_view_or_canon)(s) for s in smiles_need
            )
        )
    for idx, v in zip(need, computed):
        views[idx] = v
    return views


def _build_encoder(args: argparse.Namespace) -> DiscreteFlowEncoder:
    return build_eval_encoder(args.adapter_ckpt, device=args.device)


# --------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, int]:
    with EmbeddingStore.exclusive_lock(args.store_path):
        return _run_locked(args)


def _run_locked(args: argparse.Namespace) -> dict[str, int]:
    args.store_path.mkdir(parents=True, exist_ok=True)
    if args.force:
        _clear_store(args.store_path)

    encoder = _build_encoder(args)
    d_adapter = encoder.adapter.d_adapter

    logger.info("loading BindingDB curated parquet: %s", args.bdb_parquet)
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(args.bdb_parquet).names)
    cols = ["smiles", "inchikey", "is_binder_10uM"]
    if "fragment_view" in schema_names:
        cols.append("fragment_view")
    df = pd.read_parquet(args.bdb_parquet, columns=cols)
    if args.limit:
        df = df.head(args.limit)
    logger.info("loaded %d rows, %d unique InChIKeys",
                len(df), df["inchikey"].nunique())

    # An InChIKey is treated as a binder if *any* of its rows is a binder
    # (i.e. it binds at least one target). This drives the cross-target
    # hard-negative sampling at training time.
    grp = df.groupby("inchikey", sort=False)["is_binder_10uM"].any().rename(
        "is_binder_any_target"
    )
    smiles_map = df.drop_duplicates("inchikey").set_index("inchikey")["smiles"]

    store = EmbeddingStore.create(
        args.store_path,
        embedding_dim=d_adapter,
        model_name="lattice-adapter-v1",
        dtype="float16",
        per_residue=False,
        extra={
            "source_parquet": str(args.bdb_parquet),
            "adapter_ckpt": str(args.adapter_ckpt),
            "adapter_run_id": adapter_run_id(args.adapter_ckpt),
        },
    )
    already = set(store.pid_to_row)
    logger.info("bdb_zm store at %s has %d existing rows", args.store_path, len(already))

    # Dedupe + filter to missing InChIKeys.
    todo_inchikeys: list[str] = []
    todo_smiles: list[str] = []
    for ik, smi in zip(grp.index, smiles_map.reindex(grp.index).tolist()):
        if ik in already:
            continue
        todo_inchikeys.append(ik)
        todo_smiles.append(smi)
    logger.info("need to encode %d new ligands (n_jobs=%d for fragmentize)",
                len(todo_inchikeys), args.n_jobs)

    fv_by_ik = None
    if "fragment_view" in df.columns:
        fv_by_ik = df.drop_duplicates("inchikey").set_index("inchikey")["fragment_view"]
        n_pre = sum(
            1 for ik in todo_inchikeys
            if ik in fv_by_ik.index and pd.notna(fv_by_ik.loc[ik])
        )
        logger.info("using precomputed fragment_view for %d / %d ligands", n_pre, len(todo_inchikeys))

    views = _views_for_todo(
        todo_inchikeys, todo_smiles, fv_by_ik, n_jobs=args.n_jobs,
    )

    # Drop unfragmentable rows.
    keep_ids: list[str] = []
    keep_views: list[str] = []
    n_skipped = 0
    for ik, v in zip(todo_inchikeys, views):
        if v is None:
            n_skipped += 1
            continue
        keep_ids.append(ik)
        keep_views.append(v)
    logger.info("fragmentize: %d kept, %d rdkit-rejected", len(keep_ids), n_skipped)

    # GPU-encode.
    n_written = 0
    pbar = tqdm(range(0, len(keep_ids), args.batch_size),
                desc="encode z_m", unit="batch", dynamic_ncols=True)
    for i in pbar:
        ids = keep_ids[i : i + args.batch_size]
        v = keep_views[i : i + args.batch_size]
        z_m = encode_views_inference(encoder, v, device=args.device)
        arr = z_m.detach().cpu().to(torch.float16).numpy()
        n_written += store.append_mean(ids, arr)
    pbar.close()

    # Always (re)write the index parquet — cheap and keeps it consistent with
    # the current store contents.
    index_df = pd.DataFrame({
        "inchikey": grp.index,
        "is_binder_any_target": grp.values.astype(bool),
    })
    # Keep only InChIKeys that are now in the store (skipped/unfragmentable
    # ones drop out).
    index_df = index_df[index_df["inchikey"].isin(store.pid_to_row)].reset_index(drop=True)
    index_df["row_idx"] = index_df["inchikey"].map(store.pid_to_row).astype(np.int64)
    out_path = args.store_path / INDEX_FILE
    index_df.to_parquet(out_path, index=False)
    logger.info(
        "wrote index: %s (%d rows; binders=%d non-binders=%d)",
        out_path, len(index_df),
        int(index_df["is_binder_any_target"].sum()),
        int((~index_df["is_binder_any_target"]).sum()),
    )

    logger.info(
        "done: wrote %d new, skipped_existing=%d, skipped_rdkit=%d, total_in_store=%d",
        n_written, len(already), n_skipped, store.manifest.count,
    )
    return {
        "written": n_written,
        "skipped_existing": len(already),
        "skipped_rdkit": n_skipped,
        "total_in_store": store.manifest.count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdb-parquet", type=Path,
                        default=Path("artifacts/preprocessing/processed/bindingdb/bindingdb_curated.parquet"))
    parser.add_argument("--adapter-ckpt", type=Path, required=True,
                        help="Stage-2 Lightning .ckpt or run directory")
    parser.add_argument("--store", dest="store_path", type=Path, default=None,
                        help="default: artifacts/decoys/<adapter_run_id>/bdb_zm")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers for the CPU fragmentize step")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Cap on number of unique ligands (default -1 = all)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the store: delete any existing rows and re-encode from scratch",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.limit = None if args.limit < 0 else args.limit
    if args.store_path is None:
        # Store variant follows the adapter (recorded in its ckpt) — never the env.
        args.store_path = zm_store_path(
            args.adapter_ckpt, "bdb_zm", merge=merge_from_ckpt(args.adapter_ckpt)
        )
    run(args)


if __name__ == "__main__":
    main()

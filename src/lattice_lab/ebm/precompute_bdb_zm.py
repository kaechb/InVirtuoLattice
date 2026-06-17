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
from lattice_lab.models.builders import build_eval_encoder
from lattice_lab.preprocessing.molecules import smiles_to_fragment_views
from lattice_lab.protein.store import EmbeddingStore
from lattice_lab.training.run_logger import RunLogger

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
    try:
        v = smiles_to_fragment_views(smiles, n_views=1)
        if v:
            return v[0]
    except Exception as exc:
        logger.debug("fragment view failed for %r: %s", smiles, exc)
    if Chem.MolFromSmiles(smiles) is None:
        return None
    return Chem.CanonSmiles(smiles)


def _build_encoder(args: argparse.Namespace) -> DiscreteFlowEncoder:
    return build_eval_encoder(args.adapter_ckpt, device=args.device)


# --------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict[str, int]:
    args.store_path.mkdir(parents=True, exist_ok=True)
    if args.force:
        _clear_store(args.store_path)

    encoder = _build_encoder(args)
    d_adapter = encoder.adapter.d_adapter

    logger.info("loading BindingDB curated parquet: %s", args.bdb_parquet)
    df = pd.read_parquet(args.bdb_parquet, columns=["smiles", "inchikey", "is_binder_10uM"])
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

    # Fragmentize in parallel.
    if args.n_jobs in (0, 1):
        views: list[str | None] = []
        for s in tqdm(todo_smiles, desc="fragmentize", unit="mol", dynamic_ncols=True):
            views.append(_fragment_view_or_canon(s))
    else:
        from joblib import Parallel, delayed

        views = list(
            tqdm(
                Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator")(
                    delayed(_fragment_view_or_canon)(s) for s in todo_smiles
                ),
                total=len(todo_smiles),
                desc="fragmentize", unit="mol", dynamic_ncols=True,
            )
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
    with RunLogger(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
        tags=["stage4", "precompute", "bdb_zm"],
    ) as run_logger:
        pbar = tqdm(range(0, len(keep_ids), args.batch_size),
                    desc="encode z_m", unit="batch", dynamic_ncols=True)
        for i in pbar:
            ids = keep_ids[i : i + args.batch_size]
            v = keep_views[i : i + args.batch_size]
            with torch.no_grad():
                z_m = encoder.encode_views(v, device=args.device)
            arr = z_m.detach().cpu().to(torch.float16).numpy()
            n_written += store.append_mean(ids, arr)
            run_logger.log(
                {"bdb_zm/n_written": n_written, "bdb_zm/n_total": store.manifest.count},
                step=n_written, pbar=pbar,
            )

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

    return {
        "written": n_written,
        "skipped_existing": len(already),
        "skipped_rdkit": n_skipped,
        "total_in_store": store.manifest.count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdb-parquet", type=Path,
                        default=Path("artifacts/processed/bindingdb/bindingdb_curated.parquet"))
    parser.add_argument("--adapter-ckpt", type=Path, required=True,
                        help="Stage-2 Lightning .ckpt or run directory")
    parser.add_argument("--store", dest="store_path", type=Path,
                        default=Path("artifacts/decoys/bdb_zm/"))
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
    parser.add_argument("--wandb-project", default="lattice")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args.limit = None if args.limit < 0 else args.limit
    run(args)


if __name__ == "__main__":
    main()

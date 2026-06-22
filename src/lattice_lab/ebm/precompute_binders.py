"""Precompute frozen-adapter ``z_m`` latents for the BindingDB binders.

Stage-5 binders are a fixed set (the ``smiles`` column of the train/val
parquets), so — exactly like the decoy pool in :mod:`precompute_decoys` — their
``z_m`` can be encoded once and looked up at train time instead of being
re-tokenised + pushed through the DDiT backbone every epoch. This removes the
warm-up cost entirely and amortises it across the 3-seed Stage-5 array.

The store is keyed by the raw SMILES string (the same key the data module looks
up), uses the shared :class:`EmbeddingStore` layout, and must match the frozen
Stage-2 adapter used at EBM train time.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from lattice_lab.models.builders import adapter_run_id, build_eval_encoder, zm_store_path
from lattice_lab.models.encode import encode_binders
from lattice_lab.protein.store import EmbeddingStore

logger = logging.getLogger(__name__)


def _unique_smiles_and_views(parquets: list[Path]) -> tuple[list[str], dict[str, str]]:
    """Collect unique ``smiles`` and optional precomputed ``fragment_view``."""
    import pyarrow.parquet as pq

    seen: dict[str, None] = {}
    views: dict[str, str] = {}
    for p in parquets:
        names = set(pq.read_schema(p).names)
        cols = ["smiles"]
        if "fragment_view" in names:
            cols.append("fragment_view")
        df = pd.read_parquet(p, columns=cols)
        for _, row in df.iterrows():
            s = str(row["smiles"])
            if s in seen:
                continue
            seen[s] = None
            if "fragment_view" in df.columns and pd.notna(row.get("fragment_view")):
                views[s] = str(row["fragment_view"])
    return list(seen), views


def _clear_store(store_path: Path) -> None:
    """Delete an existing store's files so ``create`` rebuilds it from scratch."""
    import shutil

    removed = False
    for name in (EmbeddingStore.MANIFEST, EmbeddingStore.PIDS, EmbeddingStore.MEAN):
        f = store_path / name
        if f.exists():
            f.unlink()
            removed = True
    perres = store_path / EmbeddingStore.PERRES_DIR
    if perres.is_dir():
        shutil.rmtree(perres)
        removed = True
    if removed:
        logger.warning("--force: cleared existing binder store at %s", store_path)


def run(args: argparse.Namespace) -> dict[str, int]:
    with EmbeddingStore.exclusive_lock(args.store_path):
        return _run_locked(args)


def _run_locked(args: argparse.Namespace) -> dict[str, int]:
    args.store_path.mkdir(parents=True, exist_ok=True)
    if args.force:
        _clear_store(args.store_path)

    parquets = [p for p in (args.train_parquet, args.val_parquet) if p is not None]
    if not parquets:
        raise ValueError("need at least one of --train-parquet / --val-parquet")
    for p in parquets:
        if not Path(p).is_file():
            raise FileNotFoundError(f"missing parquet: {p}")

    encoder = build_eval_encoder(args.adapter_ckpt, device=args.device)
    encoder.adapter.to(args.device).eval()
    d_adapter = encoder.adapter.d_adapter

    store = EmbeddingStore.create(
        args.store_path,
        embedding_dim=d_adapter,
        model_name="lattice-adapter-v1",
        dtype="float16",
        per_residue=False,
        extra={
            "source_train_parquet": str(args.train_parquet),
            "source_val_parquet": str(args.val_parquet),
            "adapter_ckpt": str(args.adapter_ckpt),
            "adapter_run_id": adapter_run_id(args.adapter_ckpt),
            "backbone_layer_start": str(encoder.backbone_layer_start),
            "backbone_layer_end": str(encoder.backbone_layer_end),
        },
    )
    already = set(store.pid_to_row)
    logger.info("binder store at %s has %d existing rows", args.store_path, len(already))

    smiles, precomputed_views = _unique_smiles_and_views(parquets)
    if precomputed_views:
        logger.info(
            "using precomputed fragment_view for %d / %d unique binders",
            len(precomputed_views), len(smiles),
        )
    if args.limit:
        smiles = smiles[: args.limit]
    todo = [s for s in smiles if s not in already]
    n_skipped = len(smiles) - len(todo)
    logger.info(
        "need to encode %d new binders (skipped %d already-present)",
        len(todo), n_skipped,
    )

    n_written = 0
    pbar = tqdm(total=len(todo), desc="encode binders", unit="mol", dynamic_ncols=True)
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start : start + args.batch_size]
        batch_views = [precomputed_views.get(s) for s in batch] if precomputed_views else None
        with torch.no_grad():
            z_m = encode_binders(encoder, batch, args.device, grad=False, views=batch_views)
        arr = z_m.detach().cpu().to(torch.float16).numpy()
        n_written += store.append_mean(batch, arr)
        pbar.update(len(batch))
    pbar.close()

    logger.info(
        "done: wrote %d new, skipped %d already-present, total=%d",
        n_written, n_skipped, store.manifest.count,
    )
    return {"written": n_written, "skipped": n_skipped, "total": store.manifest.count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-parquet", type=Path,
        default=Path("artifacts/preprocessing/processed/bindingdb/threshold_90/train.parquet"),
    )
    parser.add_argument(
        "--val-parquet", type=Path,
        default=Path("artifacts/preprocessing/processed/bindingdb/threshold_90/val.parquet"),
    )
    parser.add_argument("--adapter-ckpt", type=Path, required=True,
                        help="Stage-2 Lightning .ckpt or run directory")
    parser.add_argument("--store", dest="store_path", type=Path, default=None,
                        help="default: artifacts/binders/<adapter_run_id>/binder_zm")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=-1, help="-1 = all")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the store: delete any existing rows and re-encode from scratch",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.limit = None if args.limit < 0 else args.limit
    if args.store_path is None:
        args.store_path = zm_store_path(args.adapter_ckpt, "binder_zm")
    run(args)


if __name__ == "__main__":
    main()

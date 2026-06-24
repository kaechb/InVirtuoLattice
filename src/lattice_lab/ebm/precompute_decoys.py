"""Precompute frozen-adapter ``z_m`` latents for the decoy pool."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from lattice_lab.eval.encode_utils import encode_views_inference
from lattice_lab.models.builders import (
    adapter_run_id,
    build_eval_encoder,
    merge_from_ckpt,
    zm_store_path,
)
from lattice_lab.preprocessing.molecules import fragment_view_column_for_parquet
from lattice_lab.protein.store import EmbeddingStore

logger = logging.getLogger(__name__)


def _iter_unique_views(shard_paths: list[Path], limit: int | None):
    seen: set[str] = set()
    n = 0
    view_col = fragment_view_column_for_parquet(shard_paths[0])
    for p in shard_paths:
        df = pd.read_parquet(p, columns=["inchikey", "view_idx", view_col])
        df = df[df["view_idx"] == 0]
        for ik, view in zip(df["inchikey"], df[view_col]):
            if ik in seen:
                continue
            seen.add(ik)
            yield ik, str(view)
            n += 1
            if limit and n >= limit:
                return


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
        logger.warning("--force: cleared existing decoy store at %s", store_path)


def run(args: argparse.Namespace) -> dict[str, int]:
    with EmbeddingStore.exclusive_lock(args.store_path):
        return _run_locked(args)


def _run_locked(args: argparse.Namespace) -> dict[str, int]:
    args.store_path.mkdir(parents=True, exist_ok=True)
    if args.force:
        _clear_store(args.store_path)
    shards = sorted(Path(args.shard_dir).glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {args.shard_dir}")

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
            "source_shard_dir": str(args.shard_dir),
            "adapter_ckpt": str(args.adapter_ckpt),
            "adapter_run_id": adapter_run_id(args.adapter_ckpt),
            "backbone_layer_start": str(encoder.backbone_layer_start),
            "backbone_layer_end": str(encoder.backbone_layer_end),
        },
    )
    already = set(store.pid_to_row)
    logger.info("decoy store at %s has %d existing rows", args.store_path, len(already))

    todo_ids: list[str] = []
    todo_views: list[str] = []
    n_skipped = 0
    for ik, view in _iter_unique_views(shards, args.limit):
        if ik in already:
            n_skipped += 1
            continue
        todo_ids.append(ik)
        todo_views.append(view)
    logger.info(
        "need to encode %d new decoys (skipped %d already-present)",
        len(todo_ids), n_skipped,
    )

    n_written = 0
    pbar = tqdm(total=len(todo_views), desc="encode decoys", unit="mol", dynamic_ncols=True)
    for start in range(0, len(todo_views), args.batch_size):
        ids_batch = todo_ids[start : start + args.batch_size]
        views_batch = todo_views[start : start + args.batch_size]
        z_m = encode_views_inference(encoder, views_batch, device=args.device)
        arr = z_m.detach().cpu().to(torch.float16).numpy()
        n_written += store.append_mean(ids_batch, arr)
        pbar.update(len(views_batch))
    pbar.close()

    logger.info(
        "done: wrote %d new, skipped %d already-present, total=%d",
        n_written, n_skipped, store.manifest.count,
    )
    return {"written": n_written, "skipped": n_skipped, "total": store.manifest.count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=Path("artifacts/preprocessing/processed/moses"))
    parser.add_argument("--adapter-ckpt", type=Path, required=True,
                        help="Stage-2 Lightning .ckpt or run directory")
    parser.add_argument("--store", dest="store_path", type=Path, default=None,
                        help="default: artifacts/decoys/<adapter_run_id>/decoy_zm")
    parser.add_argument("--batch-size", type=int, default=64)
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
        # Store variant follows the adapter (recorded in its ckpt) — never the env.
        args.store_path = zm_store_path(
            args.adapter_ckpt, "decoy_zm", merge=merge_from_ckpt(args.adapter_ckpt)
        )
    run(args)


if __name__ == "__main__":
    main()

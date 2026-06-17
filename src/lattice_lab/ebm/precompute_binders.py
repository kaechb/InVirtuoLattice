"""Precompute frozen-adapter ``z_m`` latents for the BindingDB binders.

Stage-5 binders are a fixed set (the ``smiles`` column of the train/val
parquets), so — exactly like the decoy pool in :mod:`precompute_decoys` — their
``z_m`` can be encoded once and looked up at train time instead of being
re-tokenised + pushed through the DDiT backbone every epoch. This removes the
warm-up cost entirely and amortises it across the 3-seed Stage-5 array.

The store is keyed by the raw SMILES string (the same key the data module looks
up), uses the shared :class:`EmbeddingStore` layout, and is only valid for a
*frozen* adapter (skip it / set ``finetune_adapter=true`` to encode live).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from lattice_lab.models.builders import build_eval_encoder
from lattice_lab.models.encode import encode_binders
from lattice_lab.protein.store import EmbeddingStore
from lattice_lab.training.run_logger import RunLogger

logger = logging.getLogger(__name__)


def _unique_smiles(parquets: list[Path]) -> list[str]:
    """Collect unique ``smiles`` across the train/val parquets (order-stable)."""
    seen: dict[str, None] = {}
    for p in parquets:
        df = pd.read_parquet(p, columns=["smiles"])
        for s in df["smiles"]:
            if s is not None and s not in seen:
                seen[str(s)] = None
    return list(seen)


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
            "backbone_layer_start": str(encoder.backbone_layer_start),
            "backbone_layer_end": str(encoder.backbone_layer_end),
        },
    )
    already = set(store.pid_to_row)
    logger.info("binder store at %s has %d existing rows", args.store_path, len(already))

    smiles = _unique_smiles(parquets)
    if args.limit:
        smiles = smiles[: args.limit]
    todo = [s for s in smiles if s not in already]
    n_skipped = len(smiles) - len(todo)
    logger.info(
        "need to encode %d new binders (skipped %d already-present)",
        len(todo), n_skipped,
    )

    n_written = 0
    with RunLogger(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=vars(args),
        tags=["stage4", "precompute", "binders"],
    ) as run_logger:
        pbar = tqdm(total=len(todo), desc="encode binders", unit="mol", dynamic_ncols=True)
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            with torch.no_grad():
                z_m = encode_binders(encoder, batch, args.device, grad=False)
            arr = z_m.detach().cpu().to(torch.float16).numpy()
            n_written += store.append_mean(batch, arr)
            run_logger.log({"binders/n_written": n_written}, step=n_written, pbar=pbar)
            pbar.update(len(batch))
        pbar.close()

    logger.info("done: wrote %d new, skipped %d already-present", n_written, n_skipped)
    return {"written": n_written, "skipped": n_skipped, "total": store.manifest.count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-parquet", type=Path,
        default=Path("artifacts/processed/bindingdb/threshold_90/train.parquet"),
    )
    parser.add_argument(
        "--val-parquet", type=Path,
        default=Path("artifacts/processed/bindingdb/threshold_90/val.parquet"),
    )
    parser.add_argument("--adapter-ckpt", type=Path, required=True,
                        help="Stage-2 Lightning .ckpt or run directory")
    parser.add_argument("--store", dest="store_path", type=Path,
                        default=Path("artifacts/binders/binder_zm/"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=-1, help="-1 = all")
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
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.limit = None if args.limit < 0 else args.limit
    run(args)


if __name__ == "__main__":
    main()

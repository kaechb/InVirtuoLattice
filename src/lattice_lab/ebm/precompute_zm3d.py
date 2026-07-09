"""Stage-4 (Uni-Mol variant) — precompute ``z_m`` pools with the 3D encoder.

Drop-in alternative to :mod:`precompute_decoys` / :mod:`precompute_bdb_zm` /
:mod:`precompute_binders`: instead of encoding fragment-view SMILES with the 2D
DDiT+adapter, it encodes each ligand's **3D conformer** with the frozen Uni-Mol
point-cloud tower baked into a VIEW3D Stage-2 checkpoint (``encoder_3d.*``,
loaded via :func:`load_encoder_3d_from_ckpt`).

The output stores use the **same keys and layout** as the 2D ones so Stage 5
consumes them unchanged (only the store paths differ):

* ``decoy``  — keyed by InChIKey, from the MOSES ``conformers.parquet``.
* ``bdb``    — keyed by InChIKey, plus the ``index.parquet`` sidecar
  (``is_binder_any_target``) the hard-negative collator needs.
* ``binder`` — keyed by raw SMILES. Because a binder missing a conformer would
  otherwise fall back to *2D* live encoding at train time (mixing latent spaces),
  this pool also writes **filtered** train/val parquets containing only the
  binders it covers, which Stage 5 must consume (``data.{train,val}_parquet``).

The conformer caches are built first by
:mod:`lattice_lab.preprocessing.precompute_conformers` (``--parquet`` mode).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from lattice_lab.data.conformers import (
    DEFAULT_DICT_PATH,
    Dictionary,
    collate_conformers,
    featurize_conformer,
    load_conformer_cache,
)
from lattice_lab.models.builders import (
    adapter_run_id,
    load_encoder_3d_from_ckpt,
)
from lattice_lab.protein.store import EmbeddingStore

logger = logging.getLogger(__name__)

INDEX_FILE = "index.parquet"
# ponytail: match featurize_conformer's default and the >heavy-atom ligand sizes
# we see; ligands are far smaller than this so truncation never bites.
MAX_ATOMS = 256


def _resolve_dictionary(enc3d) -> Dictionary:
    """Load the atom vocab the encoder was trained with (from its build_config)."""
    dict_path = None
    cfg = getattr(enc3d, "build_config", None)
    if isinstance(cfg, dict):
        dict_path = cfg.get("dict_path")
    if not dict_path or not Path(dict_path).is_file():
        logger.warning(
            "encoder_3d dict_path=%r not found; falling back to packaged %s",
            dict_path, DEFAULT_DICT_PATH,
        )
        dict_path = DEFAULT_DICT_PATH
    return Dictionary.load(str(dict_path))


def _encode_batch(enc3d, dictionary, items, device) -> np.ndarray:
    """Featurize + collate + encode a batch of ``(atoms, coords)`` → ``[B, d]`` fp16."""
    feats = [featurize_conformer(a, c, dictionary, MAX_ATOMS) for a, c in items]
    batch = collate_conformers(feats, key_prefix=enc3d.key_prefix)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        z = enc3d(batch)
    return z.detach().cpu().to(torch.float16).numpy()


def _clear_store(store_path: Path) -> None:
    import shutil

    for name in (EmbeddingStore.MANIFEST, EmbeddingStore.PIDS, EmbeddingStore.MEAN, INDEX_FILE):
        f = store_path / name
        if f.exists():
            f.unlink()
    perres = store_path / EmbeddingStore.PERRES_DIR
    if perres.is_dir():
        shutil.rmtree(perres)


def run(args: argparse.Namespace) -> dict[str, int]:
    with EmbeddingStore.exclusive_lock(args.store_path):
        return _run_locked(args)


def _run_locked(args: argparse.Namespace) -> dict[str, int]:
    args.store_path.mkdir(parents=True, exist_ok=True)
    if args.force:
        _clear_store(args.store_path)

    enc3d = load_encoder_3d_from_ckpt(args.adapter_ckpt, device=args.device)
    dictionary = _resolve_dictionary(enc3d)
    d_out = int(getattr(enc3d, "output_dim"))

    key_col = "smiles" if args.pool == "binder" else "inchikey"
    logger.info("loading conformer cache %s (key=%s)", args.conformer_cache, key_col)
    cache = load_conformer_cache(str(args.conformer_cache), key_col=key_col)
    keys = list(cache)
    if args.limit:
        keys = keys[: args.limit]
    logger.info("conformer cache has %d %s entries", len(keys), key_col)

    store = EmbeddingStore.create(
        args.store_path,
        embedding_dim=d_out,
        model_name="lattice-unimol3d-v1",
        dtype="float16",
        per_residue=False,
        extra={
            "encoder": "unimol3d",
            "conformer_cache": str(args.conformer_cache),
            "adapter_ckpt": str(args.adapter_ckpt),
            "adapter_run_id": adapter_run_id(args.adapter_ckpt),
        },
    )
    already = set(store.pid_to_row)
    todo = [k for k in keys if k not in already]
    logger.info("need to encode %d new (skipped %d already-present)", len(todo), len(keys) - len(todo))

    n_written = 0
    pbar = tqdm(range(0, len(todo), args.batch_size), desc=f"encode {args.pool} z_m3d",
                unit="batch", dynamic_ncols=True)
    for i in pbar:
        ids = todo[i : i + args.batch_size]
        items = [cache[k] for k in ids]
        arr = _encode_batch(enc3d, dictionary, items, args.device)
        n_written += store.append_mean(ids, arr)
        pbar.set_postfix(written=n_written)
    pbar.close()

    if args.pool == "bdb":
        _write_bdb_index(args, store)
    if args.pool == "binder":
        _write_filtered_binder_parquets(args, store)

    logger.info("done: wrote %d new, total=%d", n_written, store.manifest.count)
    return {"written": n_written, "total": store.manifest.count}


def _write_bdb_index(args: argparse.Namespace, store: EmbeddingStore) -> None:
    """Rebuild the hard-negative ``index.parquet`` for the InChIKeys now in the store."""
    df = pd.read_parquet(args.bdb_parquet, columns=["inchikey", "is_binder_10uM"])
    grp = df.groupby("inchikey", sort=False)["is_binder_10uM"].any()
    index_df = pd.DataFrame({
        "inchikey": grp.index,
        "is_binder_any_target": grp.values.astype(bool),
    })
    index_df = index_df[index_df["inchikey"].isin(store.pid_to_row)].reset_index(drop=True)
    index_df["row_idx"] = index_df["inchikey"].map(store.pid_to_row).astype(np.int64)
    index_df.to_parquet(args.store_path / INDEX_FILE, index=False)
    logger.info(
        "wrote index: %d rows (binders=%d non-binders=%d)",
        len(index_df), int(index_df["is_binder_any_target"].sum()),
        int((~index_df["is_binder_any_target"]).sum()),
    )


def _write_filtered_binder_parquets(args: argparse.Namespace, store: EmbeddingStore) -> None:
    """Write train/val parquets restricted to binders present in the 3D store.

    Guarantees Stage 5 never hits the (2D) live-encode fallback for a binder that
    lacks a 3D conformer — which would silently mix latent spaces.
    """
    covered = set(store.pid_to_row)
    for src in (args.train_parquet, args.val_parquet):
        if src is None:
            continue
        src = Path(src)
        df = pd.read_parquet(src)
        kept = df[df["smiles"].astype(str).isin(covered)].reset_index(drop=True)
        out = args.store_path / f"{src.stem}_3d.parquet"
        kept.to_parquet(out, index=False)
        logger.info("filtered %s: %d/%d rows covered -> %s", src.name, len(kept), len(df), out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, choices=["decoy", "bdb", "binder"])
    parser.add_argument("--adapter-ckpt", type=Path, required=True,
                        help="Stage-2 VIEW3D .ckpt carrying encoder_3d.* / encoder_3d_config")
    parser.add_argument("--conformer-cache", type=Path, required=True,
                        help="conformers.parquet (inchikey-keyed for decoy/bdb; smiles-keyed for binder)")
    parser.add_argument("--store", dest="store_path", type=Path, required=True)
    parser.add_argument("--bdb-parquet", type=Path, default=None,
                        help="curated BindingDB parquet (pool=bdb: for index.parquet)")
    parser.add_argument("--train-parquet", type=Path, default=None,
                        help="binder train parquet (pool=binder: filtered copy written)")
    parser.add_argument("--val-parquet", type=Path, default=None,
                        help="binder val parquet (pool=binder: filtered copy written)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=-1, help="-1 = all")
    parser.add_argument("--force", action="store_true", help="rebuild the store from scratch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.limit = None if args.limit < 0 else args.limit
    if args.pool == "bdb" and args.bdb_parquet is None:
        parser.error("--bdb-parquet is required for --pool bdb")
    if args.pool == "binder" and args.train_parquet is None and args.val_parquet is None:
        parser.error("--pool binder needs --train-parquet and/or --val-parquet for filtering")
    run(args)


if __name__ == "__main__":
    main()

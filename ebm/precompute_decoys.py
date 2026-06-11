"""Precompute frozen-adapter ``z_m`` latents for the decoy pool.

The EBM training loop draws ``N=500`` decoy ``z_m`` vectors per binder per
target every batch. Re-encoding them with FragMol+adapter at every step is
prohibitive; instead we compute them once here and write a memory-mapped
store (same layout as the protein store) that the training loop opens
read-only.

Source pool: any Stage-1 SMILES parquet. By default we use the MOSES shards
produced by ``lattice.preprocessing.run_preprocessing`` (we adopted MOSES as
the SSL corpus; the README intent was ZINC, but the role is identical).
One row per molecule (we keep ``view_idx==0`` only — repeated views would
inflate the pool with near-duplicates).

Resumable: re-running the script with the same ``--store`` will skip
molecules already encoded (keyed on InChIKey). Idempotent w.r.t. interruption
because :class:`EmbeddingStore.append_mean` does one atomic resize per call.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from lattice_lab.backbone.adapter import Adapter, AdapterConfig
from lattice_lab.backbone.encoder import EncoderConfig, MoleculeEncoder
from lattice_lab.backbone.fragmol_loader import (
    custom_tokenize,
    encode_view,
    load_fragmol,
    pad_batch,
)
from lattice_lab.protein.store import EmbeddingStore
from lattice_lab.training.run_logger import RunLogger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Parallel tokenization workers
# --------------------------------------------------------------------------
# The per-molecule cost here is FragMol tokenization (``custom_tokenize`` does
# longest-prefix matching over the whole vocab at every position), run for
# ~1.9M decoys. It is pure-CPU and independent per molecule, so we farm it out
# to a process pool and let the GPU encode the pre-tokenized batches. Workers
# only need plain picklable data (the vocab), never the FragMol GPU model — and
# we use a "spawn" pool so the parent's CUDA context is not forked.

_WORKER_TOK: dict = {}


def _init_tokenizer_worker(
    sorted_vocab: tuple[str, ...], vocab: dict[str, int],
    bos_id: int, eos_id: int, unk_id: int,
) -> None:
    _WORKER_TOK.update(
        sorted_vocab=sorted_vocab, vocab=vocab,
        bos_id=bos_id, eos_id=eos_id, unk_id=unk_id,
    )


def _tokenize_view_to_ids(view: str) -> list[int]:
    """Replicate ``fragmol_loader.encode_view`` using the worker-local vocab."""
    w = _WORKER_TOK
    vocab, unk = w["vocab"], w["unk_id"]
    body = [vocab.get(t, unk) for t in custom_tokenize(view, w["sorted_vocab"])]
    return [w["bos_id"], *body, w["eos_id"]]


@dataclass
class DecoyPrecomputeConfig:
    shard_dir: Path = Path("artifacts/processed/moses")
    adapter_ckpt: Path = Path("artifacts/adapter/checkpoints/adapter_v1.pt")
    store_path: Path = Path("artifacts/decoys/decoy_zm/")
    batch_size: int = 64
    n_jobs: int = 1
    n_fragmol_layers: int = 4
    d_adapter: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    limit: int | None = None
    wandb_project: str = "lattice"
    wandb_run_name: str | None = None


def _iter_unique_views(shard_paths: list[Path], limit: int | None) -> Iterator[tuple[str, str]]:
    """Stream ``(inchikey, fragmol_view)`` keeping only one view per molecule."""
    seen: set[str] = set()
    n = 0
    for p in shard_paths:
        df = pd.read_parquet(p, columns=["inchikey", "view_idx", "fragmol_view"])
        df = df[df["view_idx"] == 0]
        for ik, view in zip(df["inchikey"], df["fragmol_view"]):
            if ik in seen:
                continue
            seen.add(ik)
            yield ik, view
            n += 1
            if limit and n >= limit:
                return


def _load_adapter(ckpt_path: Path, d_fragmol: int, cfg: DecoyPrecomputeConfig) -> Adapter:
    # ``adapter_v1.pt`` produced by older revisions of ``train_adapter`` left
    # ``pathlib.PosixPath`` instances in the cfg block. PyTorch ≥ 2.6 defaults
    # to ``weights_only=True`` and refuses them. The weights themselves are
    # plain tensors, so allowlisting Path is safe.
    from pathlib import PosixPath, WindowsPath
    with torch.serialization.safe_globals([PosixPath, WindowsPath]):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    adapter = Adapter(
        AdapterConfig(
            d_fragmol=d_fragmol,
            n_fragmol_layers=cfg.n_fragmol_layers,
            d_adapter=cfg.d_adapter,
        )
    )
    adapter.load_state_dict(state["adapter_state_dict"])
    adapter.eval()
    return adapter


def run(cfg: DecoyPrecomputeConfig) -> dict[str, int]:
    cfg.store_path.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(cfg.shard_dir).glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {cfg.shard_dir}")

    bundle = load_fragmol(device=cfg.device)
    adapter = _load_adapter(cfg.adapter_ckpt, bundle.n_embd, cfg)
    encoder = MoleculeEncoder(
        fragmol=bundle,
        adapter=adapter,
        config=EncoderConfig(n_fragmol_layers=cfg.n_fragmol_layers),
    )
    encoder.adapter.to(cfg.device)
    encoder.adapter.eval()

    store = EmbeddingStore.create(
        cfg.store_path,
        embedding_dim=cfg.d_adapter,
        model_name="lattice-adapter-v1",
        dtype="float16",
        per_residue=False,
        extra={
            "source_shard_dir": str(cfg.shard_dir),
            "adapter_ckpt": str(cfg.adapter_ckpt),
            "n_fragmol_layers": str(cfg.n_fragmol_layers),
        },
    )
    already = set(store.pid_to_row)
    logger.info("decoy store at %s has %d existing rows", cfg.store_path, len(already))

    # Collect the molecules still to encode (dedup + resume skip up front), so
    # tokenization can be parallelized over the whole remaining set.
    todo_ids: list[str] = []
    todo_views: list[str] = []
    n_skipped = 0
    for ik, view in _iter_unique_views(shards, cfg.limit):
        if ik in already:
            n_skipped += 1
            continue
        todo_ids.append(ik)
        todo_views.append(view)
    logger.info(
        "need to encode %d new decoys (skipped %d already-present); n_jobs=%d",
        len(todo_ids), n_skipped, cfg.n_jobs,
    )

    n_written = 0
    cap = encoder.cfg.fragmol_max_len

    def _encode_token_batch(ids_batch: list[str], tok_batch: list[list[int]]) -> int:
        seqs = [t[:cap] for t in tok_batch]
        ids, mask = pad_batch(seqs, pad_id=bundle.pad_id, max_len=encoder.cfg.max_len)
        ids = ids.to(cfg.device)
        mask = mask.to(cfg.device)
        with torch.no_grad():
            z_m = encoder.encode_token_ids(ids, mask)
        arr = z_m.detach().cpu().to(torch.float16).numpy()
        return store.append_mean(ids_batch, arr)

    with RunLogger(
        project=cfg.wandb_project,
        run_name=cfg.wandb_run_name,
        config=vars(cfg),
        tags=["stage4", "precompute", "decoys"],
    ) as run_logger:

        def _gpu_consume(id_iter, tokids_iter, pbar) -> None:
            """Batch the (id, token_ids) stream and GPU-encode each batch."""
            nonlocal n_written
            buf_ids: list[str] = []
            buf_tok: list[list[int]] = []
            for ik, tokids in zip(id_iter, tokids_iter):
                buf_ids.append(ik)
                buf_tok.append(tokids)
                if len(buf_tok) >= cfg.batch_size:
                    n_written += _encode_token_batch(buf_ids, buf_tok)
                    buf_ids, buf_tok = [], []
                    run_logger.log({"decoys/n_written": n_written}, step=n_written, pbar=pbar)
                pbar.update(1)
            if buf_tok:
                n_written += _encode_token_batch(buf_ids, buf_tok)

        pbar = tqdm(total=len(todo_ids), desc="encode decoys",
                    dynamic_ncols=True, unit="mol")
        if cfg.n_jobs and cfg.n_jobs > 1 and todo_views:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            vocab = dict(bundle.tokenizer.get_vocab())
            unk_id = bundle.tokenizer.token_to_id("[UNK]")
            # Spawn (not fork) so workers don't inherit the parent's CUDA state.
            with ProcessPoolExecutor(
                max_workers=cfg.n_jobs,
                mp_context=mp.get_context("spawn"),
                initializer=_init_tokenizer_worker,
                initargs=(bundle.sorted_vocab, vocab, bundle.bos_id, bundle.eos_id, unk_id),
            ) as ex:
                # Process in chunks so outstanding tokenized lists stay bounded
                # and the GPU starts working without waiting for all 1.9M.
                chunk = max(cfg.batch_size * 20, 4096)
                for start in range(0, len(todo_views), chunk):
                    id_slice = todo_ids[start : start + chunk]
                    view_slice = todo_views[start : start + chunk]
                    tok_slice = ex.map(_tokenize_view_to_ids, view_slice, chunksize=256)
                    _gpu_consume(id_slice, tok_slice, pbar)
        else:
            # Serial fallback: tokenize on the main process (one view at a time).
            tok_iter = (encode_view(bundle, v) for v in todo_views)
            _gpu_consume(todo_ids, tok_iter, pbar)
        pbar.close()

    logger.info("done: wrote %d new, skipped %d already-present", n_written, n_skipped)
    return {"written": n_written, "skipped": n_skipped, "total": store.manifest.count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=Path("artifacts/processed/moses"))
    parser.add_argument(
        "--adapter-ckpt", type=Path,
        default=Path("artifacts/adapter/checkpoints/adapter_v1.pt"),
    )
    parser.add_argument(
        "--store", dest="store_path", type=Path,
        default=Path("artifacts/decoys/decoy_zm/"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel worker processes for FragMol tokenization "
                             "(the CPU bottleneck). 1 = serial (default).")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Cap on number of unique decoys (default -1 = all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--wandb-project", default="lattice")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    run(DecoyPrecomputeConfig(
        shard_dir=args.shard_dir,
        adapter_ckpt=args.adapter_ckpt,
        store_path=args.store_path,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        limit=None if args.limit < 0 else args.limit,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    ))


if __name__ == "__main__":
    main()

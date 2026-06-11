"""Stage 3 orchestrator — FASTA → frozen ESM-2 → memmap store.

Pipeline (mirrors ``lattice/preprocessing/run_preprocessing.py``):

1. Parse FASTA via ``lattice.preprocessing.proteins`` (already covers length /
   alphabet filters).
2. Skip pids already present in the store (idempotent re-runs).
3. Sort remaining pids by sequence length so per-batch padding is minimal
   (length-bucketed dynamic batching, single pass).
4. Stream through ``ProteinEncoder.embed_batch`` in batches sized to fit memory.
5. ``append_mean`` into the store after each batch; optionally write per-residue
   tensors as ``.npy`` files.
6. Log throughput + counts to W&B under ``protein/*``.

Run as::

    python -m lattice.protein.precompute \\
        --fasta 00_data/raw/targets.fasta \\
        --store 03_protein_encoder/embeddings/esm2_650M/ \\
        --batch-size 8 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from lattice_lab.preprocessing.proteins import (
    ProteinRecord,
    filter_length,
    filter_valid_residues,
    parse_fasta,
)
from lattice_lab.protein.encoder import (
    ESM2_DEFAULT_DIM,
    ESM2_DEFAULT_MODEL,
    ProteinEncoder,
    ProteinEncoderConfig,
)
from lattice_lab.protein.store import EmbeddingStore, iter_missing_pids
from lattice_lab.training.run_logger import RunLogger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrecomputeSummary:
    n_input: int
    n_after_filter: int
    n_already_in_store: int
    n_embedded: int
    seconds: float


def _filter_records(
    records: Sequence[ProteinRecord],
    *,
    min_len: int,
    max_len: int,
    require_canonical: bool,
) -> list[ProteinRecord]:
    kept = filter_length(records, min_len=min_len, max_len=max_len)
    if require_canonical:
        kept = filter_valid_residues(kept)
    return list(kept)


def precompute_embeddings(
    fasta_path: Path | str,
    store_dir: Path | str,
    *,
    encoder: ProteinEncoder | None = None,
    model_name: str = ESM2_DEFAULT_MODEL,
    embedding_dim: int = ESM2_DEFAULT_DIM,
    device: str = "cpu",
    dtype: str = "float32",
    max_length: int = 1024,
    batch_size: int = 8,
    min_len: int = 50,
    max_len: int = 1500,
    require_canonical: bool = True,
    per_residue: bool = False,
    overwrite: bool = False,
    sort_by_length: bool = True,
    run_logger: RunLogger | None = None,
) -> PrecomputeSummary:
    """Embed every protein in ``fasta_path`` and persist to ``store_dir``.

    Pass an existing ``encoder`` to avoid re-loading the 2.5 GB checkpoint
    across calls (and to allow tests to inject a stub).
    """
    fasta_path = Path(fasta_path)
    store_dir = Path(store_dir)
    records = parse_fasta(fasta_path)
    logger.info("parsed %d records from %s", len(records), fasta_path)

    kept = _filter_records(
        records,
        min_len=min_len,
        max_len=max_len,
        require_canonical=require_canonical,
    )
    logger.info("kept %d records after length + alphabet filters", len(kept))

    store = EmbeddingStore.create(
        store_dir,
        embedding_dim=embedding_dim,
        model_name=model_name,
        dtype=dtype,
        per_residue=per_residue,
        extra={"max_length": str(max_length)},
        exist_ok=True,
    )

    if overwrite:
        pending = kept
    else:
        already = {r.pid for r in kept if store.contains(r.pid)}
        pending = [r for r in kept if r.pid not in already]
    n_skipped = len(kept) - len(pending)

    if not pending:
        logger.info("nothing to do — all %d records already in store", len(kept))
        return PrecomputeSummary(
            n_input=len(records),
            n_after_filter=len(kept),
            n_already_in_store=n_skipped,
            n_embedded=0,
            seconds=0.0,
        )

    if sort_by_length:
        pending = sorted(pending, key=lambda r: len(r.sequence))

    if encoder is None:
        encoder = ProteinEncoder(
            ProteinEncoderConfig(
                model_name=model_name,
                embedding_dim=embedding_dim,
                max_length=max_length,
                dtype=dtype,
                device=device,
                per_residue=per_residue,
            )
        )

    t0 = time.perf_counter()
    n_done = 0
    iterator = tqdm(
        range(0, len(pending), batch_size),
        desc="esm2 embed",
        dynamic_ncols=True,
        leave=False,
    )
    for start in iterator:
        batch = pending[start : start + batch_size]
        seqs = [r.sequence for r in batch]
        with torch.no_grad():
            z = encoder.embed_batch(seqs).numpy()
        store.append_mean([r.pid for r in batch], z, overwrite=overwrite)
        if per_residue:
            for r in batch:
                pr = encoder.embed_per_residue(r.sequence).numpy()
                store.append_per_residue(r.pid, pr, overwrite=overwrite)
        n_done += len(batch)
        if run_logger is not None:
            run_logger.log(
                {
                    "protein/n_embedded": n_done,
                    "protein/total": len(pending),
                    "protein/seq_per_sec": n_done / max(time.perf_counter() - t0, 1e-9),
                },
                pbar=iterator,
            )

    store.save()
    elapsed = time.perf_counter() - t0
    summary = PrecomputeSummary(
        n_input=len(records),
        n_after_filter=len(kept),
        n_already_in_store=n_skipped,
        n_embedded=n_done,
        seconds=elapsed,
    )
    logger.info("summary: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True, type=Path, help="input FASTA")
    parser.add_argument("--store", required=True, type=Path, help="output store directory")
    parser.add_argument("--model-name", default=ESM2_DEFAULT_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=ESM2_DEFAULT_DIM)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=1500)
    parser.add_argument(
        "--no-canonical-filter",
        action="store_true",
        help="keep sequences with non-canonical residues (X, U, B, Z, …)",
    )
    parser.add_argument(
        "--per-residue",
        action="store_true",
        help="also write per-residue arrays under per_residue/<pid>.npy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-embed pids already present in the store",
    )
    parser.add_argument("--wandb-project", default="lattice")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with RunLogger(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        tags=["stage-3", "esm2"],
    ) as rl:
        summary = precompute_embeddings(
            fasta_path=args.fasta,
            store_dir=args.store,
            model_name=args.model_name,
            embedding_dim=args.embedding_dim,
            device=args.device,
            dtype=args.dtype,
            max_length=args.max_length,
            batch_size=args.batch_size,
            min_len=args.min_len,
            max_len=args.max_len,
            require_canonical=not args.no_canonical_filter,
            per_residue=args.per_residue,
            overwrite=args.overwrite,
            run_logger=rl,
        )
        rl.log({
            "protein/summary/n_input": summary.n_input,
            "protein/summary/n_after_filter": summary.n_after_filter,
            "protein/summary/n_already_in_store": summary.n_already_in_store,
            "protein/summary/n_embedded": summary.n_embedded,
            "protein/summary/seconds": summary.seconds,
        })

    logger.info("done: %s", summary)


# Surface a numpy import in this module's symbol table without lint noise.
_ = np

if __name__ == "__main__":
    main()

"""Stage 3 orchestrator — FASTA → frozen protein encoder → memmap store.

Two interchangeable backends (``--backend``): ``esm2`` (HuggingFace
``facebook/esm2_t33_650M_UR50D``, d=1280, the default) and ``esmc`` (ESM C
Cambrian 600M via the EvolutionaryScale ``esm`` SDK, d=1152). The store records
the model name + embedding dim, and downstream ``d_protein`` must match the
backend you embed with (1280 for esm2, 1152 for esmc).

Pipeline (mirrors ``lattice/preprocessing/run_preprocessing.py``):

1. Parse FASTA via ``lattice_lab.preprocessing.proteins`` (already covers length /
   alphabet filters).
2. Skip pids already present in the store (idempotent re-runs).
3. Sort remaining pids by sequence length so per-batch padding is minimal
   (length-bucketed dynamic batching, single pass).
4. Stream through ``encoder.embed_batch`` in batches sized to fit memory.
5. ``append_mean`` into the store after each batch; optionally write per-residue
   tensors as ``.npy`` files.
6. Log throughput + counts to W&B under ``protein/*``.

Run as::

    # ESM-2 (default)
    python -m lattice_lab.protein.precompute \\
        --fasta artifacts/preprocessing/raw/targets.fasta \\
        --store artifacts/protein_store/embeddings/esm2_650M/ \\
        --batch-size 8 --device cuda

    # ESM C 600M
    python -m lattice_lab.protein.precompute --backend esmc \\
        --fasta artifacts/preprocessing/raw/targets.fasta \\
        --store artifacts/protein_store/embeddings/esmc_600m/ \\
        --batch-size 8 --device cuda
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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
    ESMC_DEFAULT_DIM,
    ESMC_DEFAULT_MODEL,
    ESMCEncoder,
    ProteinEncoder,
    build_protein_encoder,
)

# Per-backend (default model, default embedding dim) used to fill CLI defaults.
_BACKEND_DEFAULTS: dict[str, tuple[str, int]] = {
    "esm2": (ESM2_DEFAULT_MODEL, ESM2_DEFAULT_DIM),
    "esmc": (ESMC_DEFAULT_MODEL, ESMC_DEFAULT_DIM),
}
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


def _resolve_device(device: str) -> str:
    """Validate ``--device cuda`` before loading multi-GB ESM-2 weights."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested --device cuda but PyTorch sees no GPU "
            "(torch.cuda.is_available() is False). "
            "On LUMI: (1) use `srun --partition=small-g --gpus-per-node=1 ... "
            "--pty bash` — not bare `salloc`; (2) `module load "
            "PyTorch/2.7.1-rocm-6.2.4-python-3.12-singularity-20250827`; "
            "(3) verify with `python -c \"import torch; print(torch.cuda.is_available())\"`. "
            "A non-ROCm conda env fails here even on a GPU node."
        )
    return device


def precompute_embeddings(
    fasta_path: Path | str,
    store_dir: Path | str,
    *,
    encoder: ProteinEncoder | ESMCEncoder | None = None,
    backend: str = "esm2",
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
    device = _resolve_device(device)
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
    logger.info("store %s currently holds %d embeddings", store_dir, len(store))

    if overwrite:
        pending = kept
        logger.info("--overwrite set — re-embedding all %d records", len(kept))
    else:
        already = {r.pid for r in kept if store.contains(r.pid)}
        pending = [r for r in kept if r.pid not in already]
    n_skipped = len(kept) - len(pending)
    logger.info(
        "%d/%d records already in store, %d to embed",
        n_skipped,
        len(kept),
        len(pending),
    )

    if not pending:
        logger.info(
            "nothing to do — all %d records already in store (pass --force to re-embed)",
            len(kept),
        )
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
        encoder = build_protein_encoder(
            backend,
            model_name=model_name,
            embedding_dim=embedding_dim,
            max_length=max_length,
            dtype=dtype,
            device=device,
            per_residue=per_residue,
        )

    t0 = time.perf_counter()
    n_done = 0
    iterator = tqdm(
        range(0, len(pending), batch_size),
        desc=f"{backend} embed",
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
    parser.add_argument(
        "--backend",
        default="esm2",
        choices=["esm2", "esmc"],
        help="protein encoder: esm2 (transformers, d=1280) or esmc (ESM C 600M, d=1152)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="override the backend's default checkpoint (esm2: %s, esmc: %s)"
        % (ESM2_DEFAULT_MODEL, ESMC_DEFAULT_MODEL),
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="override the backend's default hidden size (esm2: %d, esmc: %d)"
        % (ESM2_DEFAULT_DIM, ESMC_DEFAULT_DIM),
    )
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
        "--force",
        action="store_true",
        dest="overwrite",
        help="re-embed pids already present in the store (alias: --force)",
    )
    parser.add_argument("--wandb-project", default="lattice")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Fill model-name / embedding-dim from the chosen backend's defaults unless
    # the user overrode them, so `--backend esmc` alone gives a correct store.
    default_model, default_dim = _BACKEND_DEFAULTS[args.backend]
    if args.model_name is None:
        args.model_name = default_model
    if args.embedding_dim is None:
        args.embedding_dim = default_dim

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    _resolve_device(args.device)

    embed_kwargs = dict(
        fasta_path=args.fasta,
        store_dir=args.store,
        backend=args.backend,
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
    )
    if os.environ.get("WANDB_MODE", "online") == "disabled":
        summary = precompute_embeddings(**embed_kwargs, run_logger=None)
    else:
        with RunLogger(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            tags=["stage-3", args.backend],
        ) as rl:
            summary = precompute_embeddings(**embed_kwargs, run_logger=rl)
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

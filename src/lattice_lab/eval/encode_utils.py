"""Batched SMILES → ``z_m`` encoding shared by every evaluator.

Two stages, each with its own progress bar:

1. **Fragmentation (CPU-bound, parallel).** ``smiles_to_single_view`` calls
   rBRICS via FragMol's utilities for each molecule (~5–10 ms each). For 100K+
   inputs this dominates wall-clock if run sequentially — we use a
   ``multiprocessing.Pool`` of ``cpu_count() - 1`` workers by default.
2. **Encoding (GPU-bound, batched).** ``encode_views_batched`` runs the frozen
   FragMol + adapter on the batched views; the bottleneck shifts to the GPU.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Sequence

import torch
from tqdm.auto import tqdm

from lattice_lab.backbone.encoder import MoleculeEncoder
from lattice_lab.preprocessing.molecules import smiles_to_fragmol_views

logger = logging.getLogger(__name__)


def _frag_worker(args: tuple[int, str, int]) -> tuple[int, str] | None:
    """Pickle-friendly worker: ``(orig_idx, smiles, seed) → (orig_idx, view) | None``.

    Module-level so ``multiprocessing.Pool`` can ship it to child processes.
    """
    i, smi, seed = args
    vs = smiles_to_fragmol_views(smi, n_views=1, seed=seed)
    return (i, vs[0]) if vs else None


def smiles_to_single_view(
    smiles_list: Sequence[str],
    *,
    seed: int = 0,
    n_jobs: int | None = None,
    show_progress: bool = True,
    desc: str = "fragmolize",
) -> tuple[list[str], list[int]]:
    """Convert SMILES to one FragMol view per molecule, in parallel.

    Returns ``(views, valid_idx)``. ``valid_idx[i]`` is the position in the input
    of the molecule whose view is ``views[i]``. SMILES that cannot be fragmented
    are skipped. Order is preserved: ``valid_idx`` is monotonically increasing.

    Args:
        n_jobs: worker count. ``None`` (default) → ``cpu_count() - 1``. Pass
            ``1`` to run in the parent process (useful for debugging or tiny
            inputs where pool startup dominates).
        show_progress: if True, render a tqdm bar covering all input molecules.
    """
    n_jobs = n_jobs if n_jobs is not None else max(1, (os.cpu_count() or 2) - 1)
    args_iter = [(i, smi, seed + i) for i, smi in enumerate(smiles_list)]
    if not args_iter:
        return [], []

    if n_jobs <= 1:
        iterator = tqdm(args_iter, desc=desc, dynamic_ncols=True, leave=False) \
            if show_progress else args_iter
        results = [_frag_worker(a) for a in iterator]
    else:
        # ``imap_unordered`` lets tqdm tick as results stream in, regardless of
        # which worker finishes first; we resort by orig index at the end.
        with mp.Pool(n_jobs) as pool:
            stream = pool.imap_unordered(_frag_worker, args_iter, chunksize=200)
            if show_progress:
                stream = tqdm(stream, total=len(args_iter), desc=desc,
                              dynamic_ncols=True, leave=False)
            results = list(stream)

    pairs = sorted((r for r in results if r is not None), key=lambda x: x[0])
    valid = [i for i, _ in pairs]
    views = [v for _, v in pairs]
    return views, valid


@torch.no_grad()
def encode_views_batched(
    encoder: MoleculeEncoder,
    views: Sequence[str],
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    desc: str | None = None,
) -> torch.Tensor:
    """Encode pre-tokenized FragMol view strings, returning ``[N, D]`` on CPU."""
    encoder.adapter.eval()
    out: list[torch.Tensor] = []
    iterator = range(0, len(views), batch_size)
    if desc is not None:
        iterator = tqdm(
            iterator,
            desc=desc,
            total=(len(views) + batch_size - 1) // batch_size,
            dynamic_ncols=True,
            leave=False,
        )
    for start in iterator:
        batch = list(views[start : start + batch_size])
        z = encoder.encode_views(batch, device=device)
        out.append(z.detach().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_smiles_batched(
    encoder: MoleculeEncoder,
    smiles_list: Sequence[str],
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    seed: int = 0,
    desc: str | None = None,
    n_jobs: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Encode raw SMILES, returning ``(z, valid_idx)``.

    Some SMILES may not survive FragMol's fragmentation pipeline; the caller
    can use ``valid_idx`` to align embeddings with the original list.
    """
    frag_desc = f"{desc} fragmolize" if desc else "fragmolize"
    views, valid = smiles_to_single_view(
        smiles_list, seed=seed, n_jobs=n_jobs, desc=frag_desc
    )
    z = encode_views_batched(
        encoder, views, batch_size=batch_size, device=device, desc=desc
    )
    return z, valid

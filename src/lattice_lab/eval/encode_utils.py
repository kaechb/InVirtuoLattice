"""Batched SMILES → ``z_m`` encoding shared by every evaluator."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Sequence

import torch
from tqdm.auto import tqdm

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder
from lattice_lab.preprocessing.molecules import fragment_view

logger = logging.getLogger(__name__)


def encode_views_inference(
    encoder: DiscreteFlowEncoder,
    views: Sequence[str],
    device: str | torch.device = "cpu",
    **kwargs: object,
) -> torch.Tensor:
    """``encode_views`` under ``no_grad`` + fp16 autocast on GPU (precompute/eval default)."""
    dev = torch.device(device)
    with torch.no_grad(), torch.autocast(
        device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"
    ):
        return encoder.encode_views(views, device=dev, **kwargs)
    """Pickle-friendly worker: ``(orig_idx, smiles, seed) → (orig_idx, view) | None``."""
    i, smi, _seed = args
    v = fragment_view(smi, merge=False)  # faithful, full-coverage, deterministic
    return (i, v) if v is not None else None


def smiles_to_single_view(
    smiles_list: Sequence[str],
    *,
    seed: int = 0,
    n_jobs: int | None = None,
    show_progress: bool = True,
    desc: str = "fragmentize",
) -> tuple[list[str], list[int]]:
    """Convert SMILES to one fragment view per molecule, in parallel."""
    n_jobs = n_jobs if n_jobs is not None else max(1, (os.cpu_count() or 2) - 1)
    args_iter = [(i, smi, seed + i) for i, smi in enumerate(smiles_list)]
    if not args_iter:
        return [], []

    if n_jobs <= 1:
        iterator = tqdm(args_iter, desc=desc, dynamic_ncols=True, leave=False) \
            if show_progress else args_iter
        results = [_frag_worker(a) for a in iterator]
    else:
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
    encoder: DiscreteFlowEncoder,
    views: Sequence[str],
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    desc: str | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode fragment-view strings, returning ``[N, D]`` on CPU."""
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
        z = encode_views_inference(encoder, batch, device=device, normalize=normalize)
        out.append(z.detach().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_views_sum_pooled_batched(
    encoder: DiscreteFlowEncoder,
    views: Sequence[str],
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    desc: str | None = None,
) -> torch.Tensor:
    """Sum-pool adapter token reps (not mean) — diagnostic for size-sensitive probes."""
    from lattice_lab.backbone.discrete_flow import encode_smiles, pad_batch, prepare_backbone_tokens

    encoder.adapter.eval()
    bundle = encoder.bundle
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
        seqs = [encode_smiles(bundle, v) for v in batch]
        ids, mask = pad_batch(seqs, pad_id=bundle.pad_id)
        dev = torch.device(device)
        ids = ids.to(dev)
        mask = mask.to(dev)
        _, attn = prepare_backbone_tokens(
            ids, mask,
            bos_id=bundle.bos_id, eos_id=bundle.eos_id, pad_id=bundle.pad_id,
        )
        with torch.autocast(
            device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"
        ):
            _, tok = encoder.encode_token_ids(ids, mask, normalize=False, return_tokens=True)
        m = attn.unsqueeze(-1).to(tok.dtype)
        out.append((tok * m).sum(dim=1).detach().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def encode_smiles_batched(
    encoder: DiscreteFlowEncoder,
    smiles_list: Sequence[str],
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    seed: int = 0,
    desc: str | None = None,
    n_jobs: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Encode raw SMILES, returning ``(z, valid_idx)``."""
    frag_desc = f"{desc} fragmentize" if desc else "fragmentize"
    views, valid = smiles_to_single_view(
        smiles_list, seed=seed, n_jobs=n_jobs, desc=frag_desc
    )
    z = encode_views_batched(
        encoder, views, batch_size=batch_size, device=device, desc=desc
    )
    return z, valid

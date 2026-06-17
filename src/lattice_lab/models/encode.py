"""Binder SMILES → z_m encoding and the per-target EF metric."""

from __future__ import annotations

import logging

import numpy as np
import torch

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder

logger = logging.getLogger(__name__)

_fragmentize_fallback_count = 0
_unparseable_binder_count = 0


def encode_binders(
    encoder: DiscreteFlowEncoder,
    smiles_list: list[str],
    device: torch.device | str,
    *,
    grad: bool = False,
) -> torch.Tensor:
    """Encode a list of SMILES (one view per molecule) → ``[B, d_m]``.

    ``grad=True`` keeps the adapter forward in the autograd graph (used with
    adapter fine-tuning). The DDiT backbone stays frozen. Molecules that cannot
    be fragmented fall back to canonical SMILES; truly unparseable rows get a
    benign placeholder so the batch stays aligned with proteins/decoys.
    """
    global _fragmentize_fallback_count, _unparseable_binder_count
    from rdkit import Chem

    from lattice_lab.preprocessing.molecules import smiles_to_fragment_views

    views: list[str] = []
    n_fallback = 0
    for s in smiles_list:
        v = smiles_to_fragment_views(s, n_views=1)
        if v:
            views.append(v[0])
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            _unparseable_binder_count += 1
            if _unparseable_binder_count <= 16:
                logger.warning(
                    "unparseable binder SMILES %r — substituting placeholder "
                    "(running total: %d)",
                    s, _unparseable_binder_count,
                )
            views.append("C")
            n_fallback += 1
            continue
        views.append(Chem.MolToSmiles(mol))
        n_fallback += 1
    if n_fallback:
        _fragmentize_fallback_count += n_fallback
        if _fragmentize_fallback_count <= 32 or _fragmentize_fallback_count % 100 == 0:
            logger.info(
                "fragmentize fallback used for %d/%d binders this batch "
                "(running total: %d)",
                n_fallback, len(smiles_list), _fragmentize_fallback_count,
            )
    if grad:
        return encoder.encode_views(views, device=device)
    with torch.no_grad():
        return encoder.encode_views(views, device=device)


def ef_at(percent: float, scores: np.ndarray, labels: np.ndarray) -> float:
    """Enrichment factor at a fraction of the ranked list (higher score = binder)."""
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("scores and labels must be 1D and equal length")
    n = scores.shape[0]
    if n == 0:
        return 0.0
    pos_rate = float(labels.mean())
    if pos_rate == 0.0:
        return 0.0
    k = max(1, int(round(n * percent / 100.0)))
    top_k = np.argsort(-scores)[:k]
    return float(labels[top_k].mean()) / pos_rate

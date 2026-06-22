"""Sanity check #1 — contrastive alignment on a held-out val split.

For each molecule with ≥2 views, encode two distinct views and check that
``z_b`` is the nearest neighbor of ``z_a`` over the full val set. README target:
top-1 retrieval ``> 0.9``.

Returns metrics under the ``val/`` namespace so they sit beside training metrics
in W&B.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from lattice_lab.backbone.discrete_flow import DiscreteFlowEncoder
from lattice_lab.eval.encode_utils import encode_views_batched
from lattice_lab.training.ssl_dataset import PairedViewDataset


@dataclass(frozen=True)
class ValAlignmentResult:
    top1_acc: float
    top5_acc: float
    n_pairs: int
    threshold: float
    passed: bool

    def as_metrics(self) -> dict[str, float | int | bool]:
        return {
            "val/top1_acc": self.top1_acc,
            "val/top5_acc": self.top5_acc,
            "val/n_pairs": self.n_pairs,
            "val/threshold": self.threshold,
            "val/pass": bool(self.passed),
        }


@torch.no_grad()
def evaluate_val_alignment(
    encoder: DiscreteFlowEncoder,
    val_dataset: PairedViewDataset,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    threshold: float = 0.9,
    max_pairs: int | None = None,
) -> ValAlignmentResult:
    """Compute top-1 / top-5 retrieval of view_b given view_a over the val set.

    ``max_pairs`` caps the eval size — useful during smoke tests or if val is huge
    (full pairwise similarity is O(N²) memory).
    """
    n_total = len(val_dataset)
    n = n_total if max_pairs is None else min(n_total, max_pairs)
    if n == 0:
        return ValAlignmentResult(0.0, 0.0, 0, threshold, False)

    views_a: list[str] = []
    views_b: list[str] = []
    for i in range(n):
        a, b = val_dataset[i]
        views_a.append(a)
        views_b.append(b)

    z_a = encode_views_batched(encoder, views_a, batch_size=batch_size, device=device,
                               desc="val z_a")
    z_b = encode_views_batched(encoder, views_b, batch_size=batch_size, device=device,
                               desc="val z_b")

    # Cosine similarity; both already L2-normalized by the adapter.
    sim = z_a @ z_b.t()  # [N, N]
    target = torch.arange(n)

    top1 = (sim.argmax(dim=1) == target).float().mean().item()
    k = min(5, n)
    _, topk_idx = sim.topk(k, dim=1)
    top5 = (topk_idx == target.unsqueeze(1)).any(dim=1).float().mean().item()

    return ValAlignmentResult(
        top1_acc=float(top1),
        top5_acc=float(top5),
        n_pairs=n,
        threshold=threshold,
        passed=top1 >= threshold,
    )

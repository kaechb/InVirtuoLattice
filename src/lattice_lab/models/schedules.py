"""LR + loss-weight schedules (extracted verbatim from the original trainers)."""

from __future__ import annotations

import math


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    """Linear warmup → cosine decay to 0.1x peak."""
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def lambda_sink_schedule(step: int, ramp: int, ceiling: float) -> float:
    """Ramp the Sinkhorn weight linearly from 0 → ``ceiling`` over ``ramp`` steps."""
    if ramp <= 0:
        return ceiling
    return ceiling * min(1.0, step / ramp)

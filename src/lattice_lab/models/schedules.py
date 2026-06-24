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


def cosine_ema_decay(step: int, base: float, total: int) -> float:
    """Cosine ramp of EMA decay from ``base`` at step 0 → 1.0 at ``total`` (DINO-style)."""
    if total <= 0 or base >= 1.0:
        return base
    progress = min(1.0, step / total)
    return 1.0 - (1.0 - base) * 0.5 * (1.0 + math.cos(math.pi * progress))


if __name__ == "__main__":
    assert abs(cosine_ema_decay(0, 0.996, 1000) - 0.996) < 1e-12
    assert abs(cosine_ema_decay(1000, 0.996, 1000) - 1.0) < 1e-12
    mid = cosine_ema_decay(500, 0.996, 1000)
    assert 0.996 < mid < 1.0
    print("schedules self-check ok")

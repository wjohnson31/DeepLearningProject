from __future__ import annotations

import math


def balancing_coefficient(
    step: int,
    total_steps: int,
    policy: str,
    start: float = 0.01,
    end: float = 0.001,
) -> float:
    if policy == "zero":
        return 0.0
    if policy == "fixed":
        return start
    if policy != "scheduled":
        raise ValueError(f"unknown balancing policy: {policy}")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = min(max(2.0 * step / total_steps, 0.0), 1.0)
    return start + (end - start) * progress


def cosine_learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    minimum: float,
) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(warmup_steps, 1)
    if total_steps <= warmup_steps:
        return minimum
    progress = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))

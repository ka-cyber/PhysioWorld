"""Shared validation at communication boundaries."""

from __future__ import annotations

import math


def integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value: float, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or (minimum is not None and value < minimum):
        raise ValueError(
            f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else "")
        )
    return float(value)


def probability(value: float, name: str) -> float:
    result = finite(value, name, 0.0)
    if result > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result

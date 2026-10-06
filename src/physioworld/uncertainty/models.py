"""Uncertainty is typed by source and carried through reports."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence


class UncertaintyKind(str, Enum):
    ALEATORIC = "aleatoric"
    EPISTEMIC = "epistemic"
    CALIBRATION = "calibration"
    DISTRIBUTION_SHIFT = "distribution_shift"
    PARAMETER = "parameter"
    CHANNEL = "channel"
    RESOURCE = "resource"


@dataclass(frozen=True)
class Uncertainty:
    kind: UncertaintyKind
    estimate: float
    interval: tuple[float, float] | None = None
    method: str = "unspecified"
    assumptions: tuple[str, ...] = ()


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = lambda x: sum(x) / len(x),
    *,
    repetitions: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """IID-row percentile interval; use cluster_bootstrap for subject/window inference."""
    if (
        not values
        or repetitions < 2
        or not 0 < alpha < 1
        or any(not math.isfinite(v) for v in values)
    ):
        raise ValueError("invalid bootstrap request")
    rng = random.Random(seed)
    samples = [
        statistic([values[rng.randrange(len(values))] for _ in values]) for _ in range(repetitions)
    ]
    if any(not math.isfinite(v) for v in samples):
        raise ValueError("bootstrap statistic must be finite")
    samples.sort()
    low = samples[max(0, int(alpha / 2 * repetitions))]
    high = samples[min(repetitions - 1, int((1 - alpha / 2) * repetitions))]
    return low, high


@dataclass(frozen=True)
class RobustObjective:
    """Explicitly distinguishes sup_pi inf_K from inf_K sup_pi."""

    kernel_values: tuple[float, ...] | tuple[tuple[float, ...], ...]

    def _matrix(self) -> tuple[tuple[float, ...], ...]:
        raw = self.kernel_values
        if not raw:
            raise ValueError("nonempty policy by kernel value matrix required")
        if isinstance(raw[0], (int, float)) and not isinstance(raw[0], bool):
            vector_values: list[float] = []
            for value in raw:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError("kernel values must be a vector or a rectangular matrix")
                vector_values.append(float(value))
            vector = tuple(vector_values)
            rows: tuple[tuple[float, ...], ...] = (vector,)
        else:
            matrix: list[tuple[float, ...]] = []
            for raw_row in raw:
                if not isinstance(raw_row, (tuple, list)):
                    raise ValueError("kernel values must be a vector or a rectangular matrix")
                row: list[float] = []
                for value in raw_row:
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        raise ValueError("kernel values must be finite real numbers")
                    row.append(float(value))
                matrix.append(tuple(row))
            rows = tuple(matrix)
        if (
            not rows[0]
            or any(len(row) != len(rows[0]) for row in rows)
            or any(not math.isfinite(v) for row in rows for v in row)
        ):
            raise ValueError("finite rectangular policy by kernel matrix required")
        return rows

    def sup_pi_inf_kernel(self) -> float:
        return max(min(row) for row in self._matrix())

    def inf_kernel_sup_pi(self) -> float:
        return min(max(column) for column in zip(*self._matrix()))

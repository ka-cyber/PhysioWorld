"""Declared sensor calibration; no calibration inferred from acquired signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from physioworld.instrumentation.units import unit


@dataclass(frozen=True)
class Calibration:
    input_unit: str
    output_unit: str
    coefficients: tuple[float, ...]  # ascending powers: c0 + c1*x + ...
    method: str
    valid_range: tuple[float, float]
    reference_points: tuple[tuple[float, float], ...] = ()
    reference_id: str | None = None

    def __post_init__(self) -> None:
        unit(self.input_unit)
        unit(self.output_unit)
        if (
            not self.coefficients
            or not all(map(math.isfinite, self.coefficients))
            or not all(map(math.isfinite, self.valid_range))
            or self.valid_range[0] >= self.valid_range[1]
        ):
            raise ValueError("finite coefficients and increasing calibration range required")
        try:
            low, high = self.output_bounds
        except OverflowError as exc:
            raise ValueError("calibration bounds overflow floating-point range") from exc
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("calibration must declare a nonconstant finite output range")

    def apply(self, value: float) -> float:
        if not math.isfinite(value) or not self.valid_range[0] <= value <= self.valid_range[1]:
            raise ValueError("sample outside declared calibration range")
        result = 0.0
        for coefficient in reversed(self.coefficients):
            result = result * value + coefficient
        if not math.isfinite(result):
            raise ValueError("calibration produced a nonfinite value")
        return result

    @property
    def output_bounds(self) -> tuple[float, float]:
        """Conservative interval bound, including nonlinear interior extrema."""
        low, high = self.valid_range
        out_low = out_high = 0.0
        for power, coefficient in enumerate(self.coefficients):
            candidates = [low**power, high**power]
            if power > 0 and power % 2 == 0 and low <= 0 <= high:
                candidates.append(0.0)
            terms = [coefficient * v for v in candidates]
            out_low += min(terms)
            out_high += max(terms)
        return out_low, out_high

    @classmethod
    def linear(
        cls,
        points: Sequence[tuple[float, float]],
        input_unit: str,
        output_unit: str,
        reference_id: str | None = None,
    ) -> Calibration:
        points = tuple((float(x), float(y)) for x, y in points)
        if len(points) < 2 or not all(math.isfinite(v) for point in points for v in point):
            raise ValueError("linear calibration needs at least two finite reference points")
        mx = sum(x for x, _ in points) / len(points)
        my = sum(y for _, y in points) / len(points)
        denominator = sum((x - mx) ** 2 for x, _ in points)
        if denominator == 0:
            raise ValueError("calibration reference inputs must differ")
        slope = sum((x - mx) * (y - my) for x, y in points) / denominator
        if slope == 0:
            raise ValueError("sensor calibration slope cannot be zero")
        return cls(
            input_unit,
            output_unit,
            (my - slope * mx, slope),
            "linear_two_point" if len(points) == 2 else "linear_least_squares",
            (min(x for x, _ in points), max(x for x, _ in points)),
            points,
            reference_id,
        )

    @classmethod
    def polynomial(
        cls,
        coefficients: Sequence[float],
        input_unit: str,
        output_unit: str,
        valid_range: tuple[float, float],
        reference_id: str | None = None,
    ) -> Calibration:
        """Nonlinear coefficients supplied by the caller; no unstable implicit fit."""
        return cls(
            input_unit,
            output_unit,
            tuple(coefficients),
            "declared_polynomial",
            valid_range,
            reference_id=reference_id,
        )

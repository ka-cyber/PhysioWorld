"""Actual numerical convergence, conservation, and finite-parameter sensitivity checks."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class Tolerance:
    absolute: float
    relative: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.absolute)
            or not math.isfinite(self.relative)
            or self.absolute < 0
            or self.relative < 0
        ):
            raise ValueError("finite nonnegative tolerances required")

    def accepts(self, observed: float, reference: float) -> bool:
        return (
            math.isfinite(observed)
            and math.isfinite(reference)
            and abs(observed - reference) <= self.absolute + self.relative * abs(reference)
        )


@dataclass(frozen=True)
class ConvergenceStudy:
    step_sizes: tuple[float, ...]
    solutions: tuple[tuple[float, ...], ...]
    reference: tuple[float, ...]
    errors: tuple[float, ...]
    observed_orders: tuple[float | None, ...]
    finest_within_tolerance: bool
    converging: bool
    order_requirement_met: bool
    reference_method: str
    status: str


def convergence_study(
    solver: Callable[[float], Sequence[float]],
    independent_reference: Callable[[], Sequence[float]],
    step_sizes: Sequence[float],
    *,
    tolerance: Tolerance,
    reference_method: str,
    minimum_order: float | None = None,
) -> ConvergenceStudy:
    if (
        len(step_sizes) < 3
        or not reference_method
        or any(not math.isfinite(h) or h <= 0 for h in step_sizes)
        or any(a <= b for a, b in zip(step_sizes, step_sizes[1:]))
        or (minimum_order is not None and (not math.isfinite(minimum_order) or minimum_order < 0))
    ):
        raise ValueError("three decreasing positive steps and independent reference required")
    reference = tuple(independent_reference())
    solutions = tuple(tuple(solver(h)) for h in step_sizes)
    if (
        not reference
        or any(len(s) != len(reference) for s in solutions)
        or any(not math.isfinite(v) for v in reference)
        or any(not math.isfinite(v) for s in solutions for v in s)
    ):
        raise ValueError("finite matching numerical solutions required")
    errors = tuple(max(abs(v - r) for v, r in zip(s, reference)) for s in solutions)
    orders = tuple(
        math.log(a / b) / math.log(h1 / h2) if a > 0 and b > 0 else None
        for a, b, h1, h2 in zip(errors, errors[1:], step_sizes, step_sizes[1:])
    )
    finest = all(tolerance.accepts(v, r) for v, r in zip(solutions[-1], reference))
    converging = all(b < a or a == b == 0 for a, b in zip(errors, errors[1:]))
    order_met = minimum_order is None or all(
        (order is not None and order >= minimum_order) or a == b == 0
        for order, a, b in zip(orders, errors, errors[1:])
    )
    passed = finest and converging and order_met
    return ConvergenceStudy(
        tuple(step_sizes),
        solutions,
        reference,
        errors,
        orders,
        finest,
        converging,
        order_met,
        reference_method,
        "NUMERICALLY_VERIFIED_AT_TESTED_STEPS" if passed else "FAILED",
    )


def rk4_integrate(
    rhs: Callable[[float, tuple[float, ...]], Sequence[float]],
    initial: Sequence[float],
    *,
    duration: float,
    steps: int,
) -> tuple[float, ...]:
    """Fixed-step classical RK4; independent analytic references belong in the caller."""
    if (
        not initial
        or any(not math.isfinite(v) for v in initial)
        or not math.isfinite(duration)
        or duration <= 0
        or steps < 1
    ):
        raise ValueError("finite initial state, positive duration, and steps required")
    state = tuple(initial)
    h = duration / steps

    def derivative(time: float, values: tuple[float, ...]) -> tuple[float, ...]:
        result = tuple(rhs(time, values))
        if len(result) != len(state) or any(not math.isfinite(v) for v in result):
            raise ValueError("invalid differential kernel output")
        return result

    for i in range(steps):
        time = i * h
        k1 = derivative(time, state)
        k2 = derivative(time + h / 2, tuple(y + h * k / 2 for y, k in zip(state, k1)))
        k3 = derivative(time + h / 2, tuple(y + h * k / 2 for y, k in zip(state, k2)))
        k4 = derivative(time + h, tuple(y + h * k for y, k in zip(state, k3)))
        state = tuple(
            y + h * (a + 2 * b + 2 * c + d) / 6 for y, a, b, c, d in zip(state, k1, k2, k3, k4)
        )
        if any(not math.isfinite(v) for v in state):
            raise ValueError("nonfinite integration state")
    return state


@dataclass(frozen=True)
class ConservationCheck:
    initial_total: float
    final_total: float
    integrated_net_inflow: float
    residual: float
    passed: bool
    quantity: str
    unit: str


def conservation_check(
    initial_total: float,
    final_total: float,
    integrated_net_inflow: float,
    *,
    tolerance: Tolerance,
    quantity: str,
    unit: str,
) -> ConservationCheck:
    if (
        any(not math.isfinite(v) for v in (initial_total, final_total, integrated_net_inflow))
        or not quantity
        or not unit
    ):
        raise ValueError("finite conservation quantities and units required")
    expected = initial_total + integrated_net_inflow
    return ConservationCheck(
        initial_total,
        final_total,
        integrated_net_inflow,
        final_total - expected,
        tolerance.accepts(final_total, expected),
        quantity,
        unit,
    )


@dataclass(frozen=True)
class SensitivityStudy:
    nominal_value: float
    gradients: tuple[float, ...]
    perturbations: tuple[float, ...]
    corner_values: tuple[tuple[tuple[float, ...], float], ...]
    minimum: float
    maximum: float
    status: str = "FINITE_DIFFERENCE_AND_CORNER_RANGE_NOT_STATISTICAL_CI"


def parameter_sensitivity(
    model: Callable[[tuple[float, ...]], float],
    nominal: Sequence[float],
    perturbations: Sequence[float],
    *,
    bounds: Sequence[tuple[float, float]],
) -> SensitivityStudy:
    if (
        not nominal
        or not len(nominal) == len(perturbations) == len(bounds)
        or any(not math.isfinite(v) for v in (*nominal, *perturbations))
        or any(h <= 0 for h in perturbations)
        or any(
            not math.isfinite(lo)
            or not math.isfinite(hi)
            or lo > hi
            or not lo <= x - h <= x + h <= hi
            for x, h, (lo, hi) in zip(nominal, perturbations, bounds)
        )
    ):
        raise ValueError("finite bounded nominal parameters and feasible perturbations required")
    center = tuple(nominal)

    def evaluate(parameters: tuple[float, ...]) -> float:
        value = model(parameters)
        if not math.isfinite(value):
            raise ValueError("sensitivity objective must be finite")
        return value

    gradients = []
    for i, h in enumerate(perturbations):
        high, low = list(center), list(center)
        high[i] += h
        low[i] -= h
        gradients.append((evaluate(tuple(high)) - evaluate(tuple(low))) / (2 * h))
    corners = tuple(
        (tuple(corner), evaluate(tuple(corner))) for corner in itertools.product(*bounds)
    )
    return SensitivityStudy(
        evaluate(center),
        tuple(gradients),
        tuple(perturbations),
        corners,
        min(v for _, v in corners),
        max(v for _, v in corners),
    )

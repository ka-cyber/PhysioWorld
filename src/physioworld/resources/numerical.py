"""Independent low-order integrator used only for numerical cross-checks."""

from __future__ import annotations

import math


def euler_capacitor_voltage(
    v0: float,
    current: float,
    capacitance: float,
    duration: float,
    steps: int = 10000,
    charging_current: float = 0.0,
) -> float:
    if (
        not isinstance(steps, int)
        or steps < 1
        or capacitance <= 0
        or duration < 0
        or not all(math.isfinite(v) for v in (v0, current, capacitance, duration, charging_current))
    ):
        raise ValueError("invalid numerical integration request")
    dt = duration / steps
    voltage = v0
    for _ in range(steps):
        voltage += (charging_current - current) * dt / capacitance
    return voltage


def euler_foster_state(
    v0: float,
    currents: tuple[float, ...],
    capacitance: float,
    taus: tuple[float, ...],
    resistances: tuple[float, ...],
    duration: float,
    steps: int = 10000,
) -> tuple[float, tuple[float, ...]]:
    if (
        len(taus) != len(resistances)
        or len(currents) != steps
        or not isinstance(steps, int)
        or steps < 1
        or capacitance <= 0
        or duration < 0
        or not all(
            math.isfinite(v) for v in (v0, capacitance, duration, *taus, *resistances, *currents)
        )
        or any(t <= 0 for t in taus)
        or any(r <= 0 for r in resistances)
    ):
        raise ValueError("Foster integrator dimensions are invalid")
    dt = duration / steps
    voltage = v0
    branches = [0.0] * len(taus)
    for current in currents:
        voltage += -current * dt / capacitance
        branches = [
            branch + (resistance * current - branch) / tau * dt
            for branch, tau, resistance in zip(branches, taus, resistances)
        ]
    return voltage, tuple(branches)

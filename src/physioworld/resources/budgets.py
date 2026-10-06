"""Simultaneous resource budgets and causal discrete harvest schedules."""

import math
from dataclasses import dataclass, field
from typing import Mapping


def _valid(values):
    return all(name and math.isfinite(v) and v >= 0 for name, v in values.items())


@dataclass(frozen=True)
class MultiResourceBudget:
    available: Mapping[str, float]
    units: Mapping[str, str]
    reserved: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not _valid(self.available)
            or not _valid(self.reserved)
            or set(self.available) != set(self.units)
            or not all(self.units.values())
            or not set(self.reserved) <= set(self.available)
            or any(v > self.available[k] for k, v in self.reserved.items())
        ):
            raise ValueError("invalid typed multi-resource budget/reservation")

    def permits(self, cost):
        if not _valid(cost) or not set(cost) <= set(self.available):
            raise ValueError("cost must be finite, nonnegative, with declared resource units")
        return all(
            v <= self.available[k] - self.reserved.get(k, 0.0) + 1e-12 for k, v in cost.items()
        )

    def consume(self, cost):
        if not self.permits(cost):
            raise ValueError("multi-resource budget exceeded")
        return MultiResourceBudget(
            {k: v - cost.get(k, 0.0) for k, v in self.available.items()}, self.units, self.reserved
        )


@dataclass(frozen=True)
class ScheduleResult:
    balances: tuple[float, ...]
    served: tuple[bool, ...]
    total_value: float
    spilled: float


def evaluate_harvest_schedule(
    harvest, demands, rewards, *, initial=0.0, capacity=1e6, retention=1.0
):
    """Harvest arrives before same-slot demand; future harvest is unavailable.

    Declared scalar fungible storage submodel only. Failed demand consumes
    nothing. Capacity overflow and inter-slot leakage are explicit.
    """
    if not len(harvest) == len(demands) == len(rewards) or not harvest:
        raise ValueError("schedule arrays must be aligned and nonempty")
    if (
        not all(math.isfinite(v) and v >= 0 for v in (*harvest, *demands, initial, capacity))
        or not all(math.isfinite(v) for v in rewards)
        or not 0 <= retention <= 1
        or initial > capacity
    ):
        raise ValueError("invalid storage schedule")
    balance, value, spilled = initial, 0.0, 0.0
    balances, served = [balance], []
    for supply, demand, reward in zip(harvest, demands, rewards):
        supply_balance = balance + supply
        spilled += max(0.0, supply_balance - capacity)
        balance = min(capacity, supply_balance)
        success = balance + 1e-12 >= demand
        if success:
            balance -= demand
            value += reward
        served.append(success)
        balance *= retention
        balances.append(balance)
    return ScheduleResult(tuple(balances), tuple(served), value, spilled)


def schedule_dominates(a, b):
    """Prefix-supply criterion for lossless, unbounded, fungible storage.

    Not valid for finite-capacity overflow, leakage, power limits, or signed
    supplies. Under the stated assumptions all feasible b-demand schedules
    can be emulated by a (unused surplus can be discarded).
    """
    if len(a) != len(b) or not a or not all(math.isfinite(v) and v >= 0 for v in (*a, *b)):
        raise ValueError("invalid harvest sequences")
    sa = sb = 0.0
    for x, y in zip(a, b):
        sa, sb = sa + x, sb + y
        if sa + 1e-12 < sb:
            return False
    return True


@dataclass(frozen=True)
class ApproximationBudget:
    epsilon_phys: float
    epsilon_grid: float
    epsilon_transition: float
    epsilon_stat: float
    sources: Mapping[str, str]
    target_estimand: str
    unit: str
    composition_assumptions: tuple[str, ...]

    def __post_init__(self):
        names = ("phys", "grid", "transition", "stat")
        if not all(math.isfinite(v) and v >= 0 for v in self.terms):
            raise ValueError("approximation terms must be finite and nonnegative")
        if set(self.sources) != set(names) or not all(self.sources.values()):
            raise ValueError("every approximation term needs its declared source")
        if (
            not self.target_estimand
            or not self.unit
            or not self.composition_assumptions
            or not all(self.composition_assumptions)
        ):
            raise ValueError(
                "additive budget needs a common estimand/unit and declared composition assumptions"
            )

    @property
    def terms(self):
        return (self.epsilon_phys, self.epsilon_grid, self.epsilon_transition, self.epsilon_stat)

    @property
    def upper_bound(self):
        """Conditional sum: inputs must already be transferred into the same metric.

        This is not permission to add voltage errors directly to kernel TV or
        to a statistical confidence interval. Such transfers are caller-declared.
        """
        return math.fsum(self.terms)

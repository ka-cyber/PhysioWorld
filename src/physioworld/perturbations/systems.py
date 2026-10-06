"""System-level perturbation declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeSwitching:
    regimes: tuple[str, ...]
    switch_points: tuple[int, ...]

    def regime_at(self, step: int) -> str:
        if not self.regimes:
            raise ValueError("at least one regime is required")
        index = sum(step >= point for point in self.switch_points)
        return self.regimes[min(index, len(self.regimes) - 1)]


@dataclass(frozen=True)
class BandwidthLimitation:
    bytes_per_second: float


@dataclass(frozen=True)
class ResourceStarvation:
    available_energy_j: float


@dataclass(frozen=True)
class CPUContention:
    capacity_fraction: float

    def __post_init__(self) -> None:
        if not 0 < self.capacity_fraction <= 1:
            raise ValueError("capacity_fraction must be in (0, 1]")


@dataclass(frozen=True)
class MemoryLimitation:
    bytes_available: int


@dataclass(frozen=True)
class ModelUncertainty:
    standard_deviation: float


@dataclass(frozen=True)
class ConceptDrift:
    onset_step: int
    description: str


@dataclass(frozen=True)
class DriftScenario:
    """Controlled non-stationarity with an explicit observable/delay model."""

    mode: str
    change_points: tuple[int, ...] = ()
    drift_per_step: float = 0.0
    observation_delay_steps: int = 0
    label_delay_steps: int = 0
    adaptation_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {
            "stationary",
            "switching",
            "slow_drift",
            "fast_drift",
            "observable_switching",
            "partially_observable",
        }:
            raise ValueError("unsupported drift scenario")
        if (
            min(self.observation_delay_steps, self.label_delay_steps) < 0
            or self.adaptation_cost < 0
        ):
            raise ValueError("delays and adaptation cost must be non-negative")

    def parameter_at(self, step: int, baseline: float = 0.0) -> float:
        if self.mode == "stationary":
            return baseline
        if self.mode in {"switching", "observable_switching", "partially_observable"}:
            return baseline + float(sum(step >= point for point in self.change_points))
        return baseline + self.drift_per_step * step


def adaptation_value(fixed_value: float, adaptive_value: float, adaptation_cost: float) -> float:
    if adaptation_cost < 0:
        raise ValueError("adaptation cost must be non-negative")
    return adaptive_value - fixed_value - adaptation_cost

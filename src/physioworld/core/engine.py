"""Joint cross-layer simulator contract.

The engine is intentionally small: each layer is injected, and the returned
record retains layer outcomes so downstream utility is the objective rather
than packet delivery or signal quality alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .models import EnergyStatus, EnergyValue, WorldState


@dataclass(frozen=True)
class CrossLayerConstraints:
    energy_max_j: float | None = None
    latency_limit_s: float | None = None
    bandwidth_max_bytes: int | None = None
    compute_max_s: float | None = None
    safety: Callable[[WorldState, Any], bool] | None = None


@dataclass(frozen=True)
class CrossLayerAction:
    sample: bool = True
    modality: str | None = None
    representation: str | None = None
    compression: str | None = None
    coding: str | None = None
    route: str | None = None
    inference_node: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class CrossLayerStep:
    state: WorldState
    action: CrossLayerAction
    observation: Any
    downstream_value: float
    energy: EnergyValue
    latency_s: float
    bandwidth_bytes: int
    compute_s: float
    feasible: bool
    violations: tuple[str, ...] = ()

    @property
    def energy_j(self) -> float:
        return self.energy.value_joules


@dataclass
class JointWorld:
    state: WorldState
    constraints: CrossLayerConstraints = field(default_factory=CrossLayerConstraints)
    steps: list[CrossLayerStep] = field(default_factory=list)

    def step(
        self,
        action: CrossLayerAction,
        *,
        observation: Any,
        downstream_value: float,
        energy_j: float = 0.0,
        latency_s: float = 0.0,
        bandwidth_bytes: int = 0,
        compute_s: float = 0.0,
    ) -> CrossLayerStep:
        violations = []
        if self.constraints.energy_max_j is not None and energy_j > self.constraints.energy_max_j:
            violations.append("energy")
        if (
            self.constraints.latency_limit_s is not None
            and latency_s > self.constraints.latency_limit_s
        ):
            violations.append("latency")
        if (
            self.constraints.bandwidth_max_bytes is not None
            and bandwidth_bytes > self.constraints.bandwidth_max_bytes
        ):
            violations.append("bandwidth")
        if (
            self.constraints.compute_max_s is not None
            and compute_s > self.constraints.compute_max_s
        ):
            violations.append("compute")
        if self.constraints.safety is not None and not self.constraints.safety(self.state, action):
            violations.append("safety")
        record = CrossLayerStep(
            self.state.snapshot(),
            action,
            observation,
            downstream_value,
            EnergyValue(energy_j, EnergyStatus.SIMULATED, "joint_world"),
            latency_s,
            bandwidth_bytes,
            compute_s,
            not violations,
            tuple(violations),
        )
        self.steps.append(record)
        return record

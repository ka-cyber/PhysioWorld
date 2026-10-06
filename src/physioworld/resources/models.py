"""Resource-state contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class ResourceState:
    """All variables required for future feasibility and transition."""

    voltage: float
    internal_states: tuple[float, ...] = ()
    charge_coulombs: float | None = None
    time: float = 0.0
    load_current: float = 0.0
    charging_current: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    converter_ready: bool = True

    def __post_init__(self) -> None:
        values = (
            self.voltage,
            self.time,
            self.load_current,
            self.charging_current,
            *self.internal_states,
        )
        if self.charge_coulombs is not None:
            values += (self.charge_coulombs,)
        if not all(math.isfinite(v) for v in values) or self.time < 0:
            raise ValueError("resource state must be finite with nonnegative time")
        if not isinstance(self.converter_ready, bool):
            raise ValueError("converter readiness must be an explicit boolean")


@dataclass(frozen=True)
class ResourceEvent:
    kind: str
    time: float
    value: float
    threshold: float
    time_error_s: float = 0.0


@dataclass(frozen=True)
class EnergyBalance:
    stored_change_j: float
    terminal_work_j: float
    esr_loss_j: float
    leakage_loss_j: float
    polarization_loss_j: float = 0.0
    integration_error_j: float | None = 0.0

    @property
    def residual_j(self) -> float:
        """Signed work is positive for discharge, negative for charging."""
        return (
            self.stored_change_j
            + self.terminal_work_j
            + self.esr_loss_j
            + self.leakage_loss_j
            + self.polarization_loss_j
        )


@dataclass(frozen=True)
class ResourceTransition:
    """Electrical request evaluation.

    Base LTI models return the requested horizon even if it is unsafe; that
    next state is diagnostic and must not be committed as an admissible step.
    Events bracket the first safety violation. Lifecycle transitions stop at
    their terminal event and report their actually integrated duration.
    """

    previous: ResourceState
    next: ResourceState
    duration: float
    terminal_voltage: float
    feasible: bool
    reason: str | None = None
    events: tuple[ResourceEvent, ...] = ()
    energy_balance: EnergyBalance | None = None
    method: str = "analytic"
    load_work_j: float | None = None
    charging_work_j: float | None = None
    converter_output_energy_j: float | None = None
    converter_loss_j: float | None = None


class ResourceModel(Protocol):
    def step(
        self, state: ResourceState, current: float, duration: float, **kwargs: Any
    ) -> ResourceTransition: ...

    def terminal_voltage(self, state: ResourceState, current: float) -> float: ...

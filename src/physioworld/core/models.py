"""Small dependency-free domain models shared by all engines."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

from .contracts import finite


class EnergyStatus(str, Enum):
    """Epistemic origin of a numerical energy/resource quantity."""

    MEASURED = "MEASURED"
    MODEL_DERIVED = "MODEL_DERIVED"
    SIMULATED = "SIMULATED"
    THEORETICAL = "THEORETICAL"


@dataclass(frozen=True)
class EnergyValue:
    value_joules: float
    status: EnergyStatus
    model: str | None = None
    units: str = "J"

    def __post_init__(self) -> None:
        finite(self.value_joules, "energy")
        object.__setattr__(self, "status", EnergyStatus(self.status))
        if self.value_joules < 0:
            raise ValueError("energy cannot be negative")


@dataclass(frozen=True)
class Provenance:
    """Reproducibility metadata attached to generated objects."""

    source: str
    operation: str
    seed: int | None = None
    parent_ids: tuple[str, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    """Separated physiological, history, resource, wireless, and compute state."""

    physiological: Mapping[str, Any] = field(default_factory=dict)
    history: Sequence[Any] = field(default_factory=tuple)
    resource: Any = None
    wireless: Mapping[str, Any] = field(default_factory=dict)
    compute: Mapping[str, Any] = field(default_factory=dict)
    time: float = 0.0

    def snapshot(self) -> "WorldState":
        # Copy the whole object in one operation to retain internal aliasing while
        # isolating nested resource states, history arrays, and mutable metadata.
        return deepcopy(self)


@dataclass(frozen=True)
class Environment:
    """Declared environment contract E=(S,R,H,A,Y,K,Pi,C,I)."""

    state: Any
    resource: Any
    history: Any
    actions: tuple[Any, ...] = ()
    observations: tuple[Any, ...] = ()
    transition_kernel: Any = None
    policy_space: Any = None
    constraints: Any = None
    intervention_semantics: Any = None

    def validate(self) -> None:
        if self.state is None:
            raise ValueError("environment requires a physiological state")
        if self.history is None:
            raise ValueError("environment requires an information/history state")
        if self.transition_kernel is None:
            raise ValueError("environment requires an explicit transition kernel")

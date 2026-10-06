"""Hard safety constraints, safe fallback, and emergency-stop semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence

from .models import ResourceState


class ActionSafety(str, Enum):
    ADMISSIBLE = "admissible"
    INADMISSIBLE = "inadmissible"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SafetyEnvelope:
    min_voltage: float | None = None
    max_voltage: float | None = None
    max_current: float | None = None
    checker: Callable[[ResourceState, Any], bool] | None = None
    resource_model: Any | None = None

    def __post_init__(self):
        values = (self.min_voltage, self.max_voltage, self.max_current)
        if any(v is not None and (not math.isfinite(v) or v < 0) for v in values):
            raise ValueError("safety limits must be finite and nonnegative")
        if (
            self.min_voltage is not None
            and self.max_voltage is not None
            and self.min_voltage > self.max_voltage
        ):
            raise ValueError("invalid voltage safety interval")

    def assess(
        self,
        state: ResourceState,
        action: Any,
        current: float = 0.0,
        *,
        duration_s: float | None = None,
    ) -> ActionSafety:
        if not math.isfinite(current):
            raise ValueError("safety current must be finite")
        if self.min_voltage is not None and state.voltage < self.min_voltage:
            return ActionSafety.UNSAFE
        if self.max_voltage is not None and state.voltage > self.max_voltage:
            return ActionSafety.UNSAFE
        if self.max_current is not None and abs(current) > self.max_current:
            return ActionSafety.UNSAFE
        if self.checker is None:
            if self.resource_model is None:
                return ActionSafety.ADMISSIBLE
            if duration_s is None:
                duration_s = getattr(action, "duration_s", None)
            if duration_s is None or not math.isfinite(duration_s) or duration_s < 0:
                return ActionSafety.UNKNOWN
            try:
                transition = self.resource_model.step(state, current, duration_s)
            except Exception:
                return ActionSafety.UNKNOWN
            return ActionSafety.ADMISSIBLE if transition.feasible else ActionSafety.UNSAFE
        try:
            return ActionSafety.ADMISSIBLE if self.checker(state, action) else ActionSafety.UNSAFE
        except Exception:
            return ActionSafety.UNKNOWN


@dataclass(frozen=True)
class ConstraintCheck:
    status: ActionSafety
    action: Any
    reason: str | None = None


class ConstraintChecker:
    def __init__(self, envelope: SafetyEnvelope):
        self.envelope = envelope

    def check(
        self,
        state: ResourceState,
        action: Any,
        *,
        current: float = 0.0,
        duration_s: float | None = None,
    ) -> ConstraintCheck:
        status = self.envelope.assess(state, action, current, duration_s=duration_s)
        return ConstraintCheck(
            status,
            action,
            None if status == ActionSafety.ADMISSIBLE else f"safety status: {status.value}",
        )

    def filter(
        self,
        state: ResourceState,
        actions: Sequence[Any],
        *,
        current: float = 0.0,
        duration_s: float | None = None,
    ) -> tuple[Any, ...]:
        checks = tuple(
            self.check(state, action, current=current, duration_s=duration_s) for action in actions
        )
        return tuple(check.action for check in checks if check.status == ActionSafety.ADMISSIBLE)


@dataclass(frozen=True)
class SafeFallbackPolicy:
    fallback_action: Any

    def choose(
        self, state: ResourceState, actions: Sequence[Any], checker: ConstraintChecker
    ) -> Any:
        admissible = checker.filter(state, actions)
        if self.fallback_action in admissible:
            return self.fallback_action
        raise RuntimeError("no safe fallback action is admissible")


class EmergencyStop:
    def __init__(self) -> None:
        self.triggered = False
        self.reason: str | None = None

    def trigger(self, reason: str) -> None:
        self.triggered = True
        self.reason = reason

    def require_running(self) -> None:
        if self.triggered:
            raise RuntimeError(f"emergency stop is active: {self.reason}")

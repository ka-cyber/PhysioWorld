"""Explicit scalar ODE, hybrid events, solver and heterogeneous clock contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from physioworld.instrumentation.units import unit


@dataclass(frozen=True)
class Variable:
    name: str
    value_reference: int
    unit: str
    causality: str
    start: float = 0.0
    variability: str = "continuous"

    def __post_init__(self) -> None:
        unit(self.unit)
        if (
            not self.name
            or not 0 <= self.value_reference < 2**32
            or not math.isfinite(self.start)
            or self.causality not in {"input", "output", "parameter", "local", "independent"}
        ):
            raise ValueError("invalid scalar variable declaration")


@dataclass(frozen=True)
class ScheduledAction:
    time: float
    updates: dict[str, float] = field(default_factory=dict)
    impulse: float = 0.0  # state-unit increment, not an undocumented delta input
    label: str = "scheduled_action"

    def __post_init__(self) -> None:
        if not all(map(math.isfinite, (self.time, self.impulse, *self.updates.values()))):
            raise ValueError("events must have finite times/values")


@dataclass(frozen=True)
class ScheduledClock:
    name: str
    period_s: float
    phase_s: float = 0.0
    updates: dict[str, float] = field(default_factory=dict)
    impulse: float = 0.0

    def __post_init__(self) -> None:
        if (
            not self.name
            or not all(
                map(
                    math.isfinite,
                    (self.period_s, self.phase_s, self.impulse, *self.updates.values()),
                )
            )
            or self.period_s <= 0
            or self.phase_s < 0
            or self.phase_s + self.period_s <= self.phase_s
        ):
            raise ValueError("clock requires a name and finite positive period")


@dataclass(frozen=True)
class ModelContract:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    continuous_state: tuple[str, ...]
    discrete_state: tuple[str, ...]
    direct_feedthrough: bool
    initialization: str
    algebraic_constraints: tuple[str, ...]
    event_behavior: str
    solver_needs: str
    clocks: tuple[ScheduledClock, ...]


@dataclass(frozen=True)
class ModelEvent:
    time: float
    label: str
    before: dict[str, float]
    after: dict[str, float]


@dataclass(frozen=True)
class StepResult:
    last_successful_time: float
    event_occurred: bool
    events: tuple[ModelEvent, ...]


class ScalarSolver(Protocol):
    def __call__(
        self, derivative: Callable[[float, float], float], time: float, state: float, step: float
    ) -> float: ...


def rk4(
    derivative: Callable[[float, float], float], time: float, state: float, step: float
) -> float:
    """Classical fourth-order step; caller controls stability and error tolerance."""
    k1 = derivative(time, state)
    k2 = derivative(time + step / 2, state + step * k1 / 2)
    k3 = derivative(time + step / 2, state + step * k2 / 2)
    k4 = derivative(time + step, state + step * k3)
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6


class FirstOrderModel:
    """dy/dt = (gain*u - y)/tau, or dy/dt=0 when mode=0.

    Inputs are zero-order held per communication interval. Exact integration is
    available for this model only. Events split integration at their actual times.
    """

    kind = "physioworld.first_order/1"

    def __init__(
        self,
        name: str = "FirstOrder",
        *,
        output_unit: str = "1",
        input_unit: str | None = None,
        tau: float = 1.0,
        gain: float = 1.0,
        initial: float = 0.0,
        input_start: float = 0.0,
        actions: tuple[ScheduledAction, ...] = (),
        clocks: tuple[ScheduledClock, ...] = (),
    ) -> None:
        input_unit = input_unit or output_unit
        if not all(map(math.isfinite, (tau, gain, initial, input_start))) or tau <= 0:
            raise ValueError("positive time constant and finite model values required")
        if len({c.name for c in clocks}) != len(clocks):
            raise ValueError("clock names must be unique")
        self.name, self.output_unit, self.input_unit = name, output_unit, input_unit
        self.variables = (
            Variable("time", 0, "s", "independent"),
            Variable("u", 1, input_unit, "input", input_start),
            Variable("y", 2, output_unit, "output", initial),
            Variable("tau", 3, "s", "parameter", tau, "fixed"),
            Variable(
                "gain", 4, str(unit(output_unit) / unit(input_unit)), "parameter", gain, "fixed"
            ),
            Variable("mode", 5, "1", "output", 1.0, "discrete"),
        )
        if set(c.name for c in clocks) & {v.name for v in self.variables}:
            raise ValueError("clock names must not collide with scalar variables")
        self.actions, self.clocks = tuple(sorted(actions, key=lambda e: e.time)), tuple(clocks)
        for action in actions:
            self._validate_updates(action.updates)
        for clock in clocks:
            self._validate_updates(clock.updates)
        self.contract = ModelContract(
            ("u",),
            ("y", "mode"),
            ("y",),
            ("mode",),
            False,
            "declared start values then events at start",
            (),
            "scheduled updates/impulses split integration",
            "exact scalar exponential or explicit external scalar solver",
            self.clocks,
        )
        self.values = {v.name: v.start for v in self.variables}
        self.initialized = False
        self.event_history: list[ModelEvent] = []
        self._action_index = 0
        self._clock_ticks: dict[str, int] = {}

    def _validate_updates(self, updates: dict[str, float]) -> None:
        if not set(updates) <= {"u", "y", "mode"}:
            raise ValueError("scheduled updates support u, y and mode; parameters are fixed")
        if "mode" in updates and updates["mode"] not in {0.0, 1.0}:
            raise ValueError("mode must be 0 (hold) or 1 (active)")

    @property
    def time(self) -> float:
        return self.values["time"]

    def initialize(self, start_time: float = 0.0) -> None:
        if not math.isfinite(start_time):
            raise ValueError("finite model start required")
        if any(a.time < start_time for a in self.actions):
            raise ValueError("events before start require an explicitly initialized history")
        self.values = {v.name: v.start for v in self.variables}
        self.values["time"] = start_time
        self._action_index = 0
        self._clock_ticks = {
            c.name: max(0, math.ceil((start_time - c.phase_s) / c.period_s - 1e-12))
            for c in self.clocks
        }
        for clock in self.clocks:
            if self._next_clock(clock) < start_time:
                self._clock_ticks[clock.name] += 1
            if self._next_clock(clock) + clock.period_s <= self._next_clock(clock):
                raise ValueError("clock period is below timestamp resolution at model start")
        self.event_history.clear()
        self.initialized = True
        self._apply_due()

    def set(self, name: str, value: float, declared_unit: str) -> None:
        variable = next((v for v in self.variables if v.name == name), None)
        if variable is None:
            raise KeyError(name)
        if variable.causality != "input" or not self.initialized:
            raise ValueError("runtime set permits initialized input variables only")
        if unit(declared_unit) != unit(variable.unit) or not math.isfinite(value):
            raise ValueError(
                "finite value in identical declared units required; convert explicitly"
            )
        self.values[name] = float(value)

    def get(self, name: str) -> float:
        if not self.initialized:
            raise RuntimeError("model not initialized")
        return self.values[name]

    def derivative(self, time: float, state: float) -> float:
        return (
            (self.values["gain"] * self.values["u"] - state) / self.values["tau"]
            if self.values["mode"]
            else 0.0
        )

    def _next_clock(self, clock: ScheduledClock) -> float:
        return clock.phase_s + self._clock_ticks[clock.name] * clock.period_s

    def _apply_due(self) -> None:
        events = []
        while (
            self._action_index < len(self.actions)
            and self.actions[self._action_index].time <= self.time
        ):
            events.append(self.actions[self._action_index])
            self._action_index += 1
        for clock in self.clocks:
            if self._next_clock(clock) <= self.time:
                events.append(
                    ScheduledAction(self.time, clock.updates, clock.impulse, f"clock:{clock.name}")
                )
                self._clock_ticks[clock.name] += 1
        # Same-time actions use declaration order, then declaration-order clocks.
        for event in events:
            before = dict(self.values)
            self.values.update(event.updates)
            self.values["y"] += event.impulse
            self.event_history.append(ModelEvent(self.time, event.label, before, dict(self.values)))

    def do_step(
        self, current_time: float, step_size: float, solver: ScalarSolver | None = None
    ) -> StepResult:
        if (
            not self.initialized
            or not math.isfinite(step_size)
            or step_size <= 0
            or not math.isclose(current_time, self.time, abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError(
                "initialized matching communication point and positive finite step required"
            )
        stop, start_events = current_time + step_size, len(self.event_history)
        if not math.isfinite(stop):
            raise ValueError("step endpoint must be finite")
        while self.time < stop:
            if len(self.event_history) - start_events > 100000:
                raise RuntimeError(
                    "more than 100000 clock/event activations in one step; shorten the step"
                )
            boundaries = [stop]
            if self._action_index < len(self.actions):
                boundaries.append(self.actions[self._action_index].time)
            boundaries.extend(self._next_clock(c) for c in self.clocks)
            target = min(boundaries)
            if target < self.time:
                raise RuntimeError("clock/event scheduler moved backward")
            dt, state = target - self.time, self.values["y"]
            if dt:
                if solver is None:
                    equilibrium = self.values["gain"] * self.values["u"]
                    state = (
                        equilibrium + (state - equilibrium) * math.exp(-dt / self.values["tau"])
                        if self.values["mode"]
                        else state
                    )
                else:
                    state = solver(self.derivative, self.time, state, dt)
                if not math.isfinite(state):
                    raise ArithmeticError("solver produced nonfinite state")
                self.values["y"] = state
                self.values["time"] = target
            self._apply_due()
        events = tuple(self.event_history[start_events:])
        return StepResult(self.time, bool(events), events)

    def specification(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "output_unit": self.output_unit,
            "input_unit": self.input_unit,
            "tau": self.variables[3].start,
            "gain": self.variables[4].start,
            "initial": self.variables[2].start,
            "input_start": self.variables[1].start,
            "actions": [asdict(a) for a in self.actions],
            "clocks": [asdict(c) for c in self.clocks],
        }

    @classmethod
    def from_specification(cls, specification: dict) -> FirstOrderModel:
        spec = dict(specification)
        if spec.pop("kind") != cls.kind:
            raise ValueError("unsupported Python model kind")
        spec["actions"] = tuple(ScheduledAction(**a) for a in spec.get("actions", ()))
        spec["clocks"] = tuple(ScheduledClock(**c) for c in spec.get("clocks", ()))
        return cls(**spec)


class PythonModelExchange:
    """Python solver-facing boundary; does not claim the FMI native ME C ABI."""

    def __init__(self, model: FirstOrderModel, solver: ScalarSolver = rk4) -> None:
        self.model, self.solver = model, solver

    def continuous_states(self) -> tuple[float, ...]:
        return (self.model.get("y"),)

    def derivatives(self) -> tuple[float, ...]:
        return (self.model.derivative(self.model.time, self.model.get("y")),)

    def advance(self, step_size: float) -> StepResult:
        return self.model.do_step(self.model.time, step_size, self.solver)

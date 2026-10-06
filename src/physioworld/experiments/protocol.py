"""Typed, declarative experiment protocols with an explicit simulation clock.

This module is deliberately a small execution kernel rather than a scripting
language.  A protocol is data made from typed nodes; control flow is limited to
the node types below and expressions are restricted to named references and
comparisons.  Python callables are accepted only at the injected handler
boundary.  Protocol data is never passed to ``eval`` or executed as code.

Handlers are the integration boundary for devices, recordings, physiology
models, computation engines, and exports.  The kernel owns ordering, bounds,
unit checks for protocol values, seeded scheduling, and an auditable trace.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
import typing
from dataclasses import dataclass, field, replace
from importlib.resources import files
from typing import (
    Any,
    Callable,
    ClassVar,
    Mapping,
    Sequence,
    cast,
)

from physioworld.instrumentation.units import unit

PROTOCOL_SCHEMA_VERSION = "1.0.0"
PROTOCOL_SCHEMA_NAME = "protocol.schema.json"


class ProtocolError(ValueError):
    """Base error for invalid declarations or execution contracts."""


class ProtocolValidationError(ProtocolError):
    """A protocol or typed node is not a valid declaration."""


class ProtocolLimitError(ProtocolError):
    """Execution would exceed an explicitly declared bound."""


class ProtocolExecutionError(ProtocolError):
    """An injected handler failed; ``result`` contains the emergency trace."""

    def __init__(self, message: str, result: "ExecutionResult | None" = None) -> None:
        super().__init__(message)
        self.result = result


class ProtocolAbort(ProtocolError):
    """A handler requested an abort; the emergency sequence is run."""


def _finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProtocolValidationError(f"{label} must be a finite number")
    result = float(value)
    if nonnegative and result < 0:
        raise ProtocolValidationError(f"{label} must be non-negative")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProtocolValidationError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class Quantity:
    """A finite scalar with an explicit unit.

    Numeric protocol parameters should use ``Quantity`` or be accompanied by
    a node ``units`` mapping.  Time remains a separate protocol coordinate in
    seconds and is never silently interpreted as a quantity.
    """

    value: float
    unit: str

    def __post_init__(self) -> None:
        _finite(self.value, "quantity.value")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ProtocolValidationError("quantity.unit must be a non-empty string")
        try:
            unit(self.unit)
        except ValueError as exc:
            raise ProtocolValidationError(f"invalid quantity unit {self.unit!r}") from exc
        object.__setattr__(self, "value", float(self.value))

    def to(self, target_unit: str) -> "Quantity":
        try:
            converted = self.value * unit(self.unit).factor_to(unit(target_unit))
        except ValueError as exc:
            raise ProtocolError(f"cannot convert {self.unit!r} to {target_unit!r}") from exc
        return Quantity(converted, target_unit)

    def to_dict(self) -> dict[str, Any]:
        return {"$quantity": {"value": self.value, "unit": self.unit}}


@dataclass(frozen=True)
class Reference:
    """A safe lookup into prior execution state, never a Python expression."""

    path: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not self.path.strip()
            or "(" in self.path
            or ")" in self.path
        ):
            raise ProtocolValidationError("reference path must be a non-empty data path")

    def to_dict(self) -> dict[str, str]:
        return {"$ref": self.path}


@dataclass(frozen=True)
class ParameterRef:
    """A named parameter supplied by a ``ParameterSweep`` case."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProtocolValidationError("parameter reference name must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"$parameter": self.name}


Operand = Reference | Quantity | ParameterRef | str | int | float | bool | None


@dataclass(frozen=True)
class Condition:
    """A unit-aware comparison over literal values or prior state references."""

    left: Operand
    operator: str
    right: Operand

    OPERATORS: ClassVar[frozenset[str]] = frozenset({"==", "!=", "<", "<=", ">", ">="})

    def __post_init__(self) -> None:
        if self.operator not in self.OPERATORS:
            raise ProtocolValidationError(f"unsupported condition operator {self.operator!r}")
        if isinstance(self.left, (dict, list)) or isinstance(self.right, (dict, list)):
            raise ProtocolValidationError(
                "conditions accept literals, quantities, or references only"
            )
        _validate_data(self.left, "condition.left")
        _validate_data(self.right, "condition.right")


def _copy_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ProtocolValidationError(f"{label} must be an object with string keys")
    try:
        result = copy.deepcopy(dict(value))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ProtocolValidationError(f"{label} must contain copyable data") from exc
    _validate_data(result, label)
    return result


def _validate_data(value: Any, label: str) -> None:
    """Reject opaque objects and non-finite values in protocol-owned data."""
    if isinstance(value, (Quantity, Reference, ParameterRef)) or value is None:
        return
    if isinstance(value, bool) or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _finite(value, label)
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ProtocolValidationError(f"{label} object keys must be strings")
        for key, item in value.items():
            _validate_data(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_data(item, f"{label}[{index}]")
        return
    raise ProtocolValidationError(f"{label} contains unsupported data {type(value).__name__}")


def _validate_units(parameters: Mapping[str, Any], units: Mapping[str, str], label: str) -> None:
    parameter_names = set(parameters)
    if set(units) - parameter_names:
        raise ProtocolValidationError(
            f"{label}.units contains parameters not present in parameters"
        )
    for name, value in parameters.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not isinstance(value, Quantity)
        ):
            if name not in units:
                raise ProtocolValidationError(f"{label}.{name} requires an explicit unit")
    for name, declared in units.items():
        try:
            unit(declared)
        except ValueError as exc:
            raise ProtocolValidationError(f"{label}.units[{name!r}] is invalid") from exc


@dataclass(frozen=True)
class Node:
    """Base class for a typed protocol node."""

    node_id: str
    op: ClassVar[str] = "node"

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ProtocolValidationError("every protocol node needs a non-empty id")

    @property
    def id(self) -> str:
        return self.node_id


@dataclass(frozen=True)
class Setup(Node):
    parameters: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    handler: str = "setup"
    at_s: float | None = None
    op: ClassVar[str] = "setup"

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_units(self.parameters, self.units, self.op)
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(self, "parameters", _copy_mapping(self.parameters, "setup.parameters"))
        object.__setattr__(self, "units", dict(self.units))


@dataclass(frozen=True)
class Calibration(Node):
    channel: str
    input_unit: str
    output_unit: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    handler: str = "calibration"
    at_s: float | None = None
    op: ClassVar[str] = "calibration"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.channel, "calibration.channel")
        _check_unit(self.input_unit, "calibration.input_unit")
        _check_unit(self.output_unit, "calibration.output_unit")
        _validate_units(self.parameters, self.units, self.op)
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(
            self, "parameters", _copy_mapping(self.parameters, "calibration.parameters")
        )
        object.__setattr__(self, "units", dict(self.units))


@dataclass(frozen=True)
class Baseline(Node):
    duration_s: float
    channels: tuple[str, ...] = ()
    handler: str = "baseline"
    at_s: float | None = None
    op: ClassVar[str] = "baseline"

    def __post_init__(self) -> None:
        super().__post_init__()
        _duration(self.duration_s, "baseline.duration_s", positive=True)
        if any(not isinstance(c, str) or not c.strip() for c in self.channels):
            raise ProtocolValidationError("baseline.channels must contain non-empty names")
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(self, "channels", tuple(self.channels))


@dataclass(frozen=True)
class Stimulus(Node):
    actuator: str
    value: float | Quantity | ParameterRef
    unit: str
    duration_s: float
    handler: str = "stimulus"
    at_s: float | None = None
    op: ClassVar[str] = "stimulus"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.actuator, "stimulus.actuator")
        _check_unit(self.unit, "stimulus.unit")
        if isinstance(self.value, bool) or not isinstance(
            self.value, (int, float, Quantity, ParameterRef)
        ):
            raise ProtocolValidationError("stimulus.value must be a finite scalar or Quantity")
        if isinstance(self.value, Quantity) and self.value.unit != self.unit:
            raise ProtocolValidationError("stimulus.value unit must equal stimulus.unit")
        if isinstance(self.value, (int, float)):
            _finite(self.value, "stimulus.value")
        _duration(self.duration_s, "stimulus.duration_s", positive=True)
        _validate_handler(self.handler)
        _validate_at(self.at_s)


@dataclass(frozen=True)
class Acquire(Node):
    channel: str
    sample_rate_hz: float
    duration_s: float
    parameters: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    handler: str = "acquire"
    at_s: float | None = None
    op: ClassVar[str] = "acquire"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.channel, "acquire.channel")
        _duration(self.sample_rate_hz, "acquire.sample_rate_hz", positive=True)
        _duration(self.duration_s, "acquire.duration_s", positive=True)
        _validate_units(self.parameters, self.units, self.op)
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(self, "parameters", _copy_mapping(self.parameters, "acquire.parameters"))
        object.__setattr__(self, "units", dict(self.units))


@dataclass(frozen=True)
class Event(Node):
    label: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    handler: str = "event"
    at_s: float | None = None
    op: ClassVar[str] = "event"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.label, "event.label")
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(self, "attributes", _copy_mapping(self.attributes, "event.attributes"))


@dataclass(frozen=True)
class Wait(Node):
    duration_s: float
    handler: str = "wait"
    at_s: float | None = None
    op: ClassVar[str] = "wait"

    def __post_init__(self) -> None:
        super().__post_init__()
        _duration(self.duration_s, "wait.duration_s", positive=True)
        _validate_handler(self.handler)
        _validate_at(self.at_s)


@dataclass(frozen=True)
class Computation(Node):
    output: str
    output_unit: str
    inputs: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    handler: str = "computation"
    at_s: float | None = None
    op: ClassVar[str] = "computation"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.output, "computation.output")
        _check_unit(self.output_unit, "computation.output_unit")
        if any(not isinstance(value, str) or not value.strip() for value in self.inputs):
            raise ProtocolValidationError("computation.inputs must contain names")
        _validate_units(self.parameters, self.units, self.op)
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(
            self, "parameters", _copy_mapping(self.parameters, "computation.parameters")
        )
        object.__setattr__(self, "units", dict(self.units))


@dataclass(frozen=True)
class Decision(Node):
    condition: Condition
    then_nodes: tuple[Node, ...] = ()
    else_nodes: tuple[Node, ...] = ()
    handler: str = "builtin"
    at_s: float | None = None
    op: ClassVar[str] = "decision"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.condition, Condition):
            raise ProtocolValidationError("decision.condition must be a Condition")
        _validate_children(self.then_nodes, "decision.then")
        _validate_children(self.else_nodes, "decision.else")
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(self, "then_nodes", tuple(self.then_nodes))
        object.__setattr__(self, "else_nodes", tuple(self.else_nodes))


@dataclass(frozen=True)
class Intervention(Node):
    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    handler: str = "intervention"
    at_s: float | None = None
    op: ClassVar[str] = "intervention"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.kind, "intervention.kind")
        _validate_units(self.parameters, self.units, self.op)
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(
            self, "parameters", _copy_mapping(self.parameters, "intervention.parameters")
        )
        object.__setattr__(self, "units", dict(self.units))


@dataclass(frozen=True)
class Resource(Node):
    name: str
    value: float | Quantity | ParameterRef | str | bool
    unit: str
    handler: str = "resource"
    at_s: float | None = None
    op: ClassVar[str] = "resource"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.name, "resource.name")
        _check_unit(self.unit, "resource.unit")
        if isinstance(self.value, Quantity) and self.value.unit != self.unit:
            raise ProtocolValidationError("resource.value unit must equal resource.unit")
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            _finite(self.value, "resource.value")
        _validate_handler(self.handler)
        _validate_at(self.at_s)


@dataclass(frozen=True)
class Measurement(Node):
    name: str
    source: str
    unit: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    handler: str = "measurement"
    at_s: float | None = None
    op: ClassVar[str] = "measurement"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.name, "measurement.name")
        _nonempty(self.source, "measurement.source")
        _check_unit(self.unit, "measurement.unit")
        _validate_units(self.parameters, self.units, self.op)
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(
            self, "parameters", _copy_mapping(self.parameters, "measurement.parameters")
        )
        object.__setattr__(self, "units", dict(self.units))


@dataclass(frozen=True)
class Stop(Node):
    reason: str
    condition: Condition | None = None
    emergency: bool = False
    handler: str = "stop"
    at_s: float | None = None
    op: ClassVar[str] = "stop"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.reason, "stop.reason")
        if self.condition is not None and not isinstance(self.condition, Condition):
            raise ProtocolValidationError("stop.condition must be a Condition")
        if not isinstance(self.emergency, bool):
            raise ProtocolValidationError("stop.emergency must be boolean")
        _validate_handler(self.handler)
        _validate_at(self.at_s)


@dataclass(frozen=True)
class Export(Node):
    target: str
    format: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    handler: str = "export"
    at_s: float | None = None
    op: ClassVar[str] = "export"

    def __post_init__(self) -> None:
        super().__post_init__()
        _nonempty(self.target, "export.target")
        _nonempty(self.format, "export.format")
        _validate_handler(self.handler)
        _validate_at(self.at_s)
        object.__setattr__(self, "parameters", _copy_mapping(self.parameters, "export.parameters"))


@dataclass(frozen=True)
class Loop(Node):
    count: int
    body: tuple[Node, ...]
    at_s: float | None = None
    op: ClassVar[str] = "loop"

    def __post_init__(self) -> None:
        super().__post_init__()
        _positive_int(self.count, "loop.count")
        _validate_children(self.body, "loop.body")
        _validate_at(self.at_s)
        object.__setattr__(self, "body", tuple(self.body))


@dataclass(frozen=True)
class If(Node):
    condition: Condition
    then_nodes: tuple[Node, ...] = ()
    else_nodes: tuple[Node, ...] = ()
    at_s: float | None = None
    op: ClassVar[str] = "if"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.condition, Condition):
            raise ProtocolValidationError("if.condition must be a Condition")
        _validate_children(self.then_nodes, "if.then")
        _validate_children(self.else_nodes, "if.else")
        _validate_at(self.at_s)
        object.__setattr__(self, "then_nodes", tuple(self.then_nodes))
        object.__setattr__(self, "else_nodes", tuple(self.else_nodes))


@dataclass(frozen=True)
class ScheduleCase:
    case_id: str
    nodes: tuple[Node, ...]

    def __post_init__(self) -> None:
        _nonempty(self.case_id, "schedule.case.id")
        _validate_children(self.nodes, "schedule.case.nodes")
        object.__setattr__(self, "nodes", tuple(self.nodes))


@dataclass(frozen=True)
class Schedule(Node):
    cases: tuple[ScheduleCase, ...]
    mode: str = "randomized"
    repetitions: int = 1
    at_s: float | None = None
    op: ClassVar[str] = "schedule"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mode not in {"sequential", "randomized", "counterbalanced"}:
            raise ProtocolValidationError(
                "schedule.mode must be sequential, randomized, or counterbalanced"
            )
        _positive_int(self.repetitions, "schedule.repetitions")
        if not self.cases:
            raise ProtocolValidationError("schedule requires at least one case")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ProtocolValidationError("schedule case IDs must be unique")
        object.__setattr__(self, "cases", tuple(self.cases))
        _validate_at(self.at_s)


@dataclass(frozen=True)
class ParameterSweep(Node):
    parameters: Mapping[str, tuple[Any, ...]]
    body: tuple[Node, ...]
    at_s: float | None = None
    op: ClassVar[str] = "sweep"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.parameters or any(
            not isinstance(k, str) or not k.strip() for k in self.parameters
        ):
            raise ProtocolValidationError("sweep.parameters must contain named values")
        if any(not values for values in self.parameters.values()):
            raise ProtocolValidationError("sweep parameter value lists cannot be empty")
        for name, values in self.parameters.items():
            _validate_data(values, f"sweep.parameters.{name}")
        _validate_children(self.body, "sweep.body")
        object.__setattr__(self, "parameters", {k: tuple(v) for k, v in self.parameters.items()})
        object.__setattr__(self, "body", tuple(self.body))
        _validate_at(self.at_s)


ProtocolNode = (
    Setup
    | Calibration
    | Baseline
    | Stimulus
    | Acquire
    | Event
    | Wait
    | Computation
    | Decision
    | Intervention
    | Resource
    | Measurement
    | Stop
    | Export
    | Loop
    | If
    | Schedule
    | ParameterSweep
)


def _nonempty(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{label} must be a non-empty string")


def _check_unit(value: Any, label: str) -> None:
    _nonempty(value, label)
    try:
        unit(value)
    except ValueError as exc:
        raise ProtocolValidationError(f"{label} is not a supported unit expression") from exc


def _validate_handler(value: Any) -> None:
    _nonempty(value, "node.handler")
    if any(character in value for character in ("(", ")", "\n", "\r")):
        raise ProtocolValidationError("handler is an identifier, not executable source")


def _validate_at(value: Any) -> None:
    if value is not None:
        _finite(value, "node.at_s", nonnegative=True)


def _duration(value: Any, label: str, *, positive: bool = False) -> None:
    result = _finite(value, label, nonnegative=not positive)
    if positive and result <= 0:
        raise ProtocolValidationError(f"{label} must be positive")


def _validate_children(nodes: Sequence[Node], label: str) -> None:
    if not isinstance(nodes, (tuple, list)) or any(not isinstance(node, Node) for node in nodes):
        raise ProtocolValidationError(f"{label} must contain typed protocol nodes")
    ids = [node.node_id for node in nodes]
    if len(ids) != len(set(ids)):
        raise ProtocolValidationError(f"{label} contains duplicate node IDs")


def _collect_node_ids(nodes: Sequence[Node]) -> list[str]:
    ids: list[str] = []
    for node in nodes:
        ids.append(node.node_id)
        if isinstance(node, (Loop, ParameterSweep)):
            ids.extend(_collect_node_ids(node.body))
        elif isinstance(node, (If, Decision)):
            ids.extend(_collect_node_ids((*node.then_nodes, *node.else_nodes)))
        elif isinstance(node, Schedule):
            for case in node.cases:
                ids.extend(_collect_node_ids(case.nodes))
    return ids


@dataclass(frozen=True)
class ExecutionLimits:
    """Finite resource limits for both ordinary and emergency execution."""

    max_commands: int = 10000
    max_loop_iterations: int = 1000
    max_sweep_cases: int = 1000
    max_branch_depth: int = 32
    max_emergency_commands: int = 32

    def __post_init__(self) -> None:
        for name in (
            "max_commands",
            "max_loop_iterations",
            "max_sweep_cases",
            "max_branch_depth",
            "max_emergency_commands",
        ):
            _positive_int(getattr(self, name), f"limits.{name}")


@dataclass(frozen=True)
class Protocol:
    """A versioned machine-readable protocol declaration."""

    protocol_id: str
    horizon_s: float
    nodes: tuple[Node, ...]
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    emergency: tuple[Node, ...] = ()
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty(self.protocol_id, "protocol_id")
        _duration(self.horizon_s, "clock.horizon_s", positive=True)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ProtocolValidationError("protocol seed must be an integer")
        if self.schema_version != PROTOCOL_SCHEMA_VERSION:
            raise ProtocolValidationError(
                f"unsupported protocol schema version {self.schema_version!r}"
            )
        _validate_children(self.nodes, "protocol.nodes")
        _validate_children(self.emergency, "protocol.emergency")
        declared_ids = _collect_node_ids((*self.nodes, *self.emergency))
        if len(declared_ids) != len(set(declared_ids)):
            raise ProtocolValidationError(
                "protocol node IDs must be unique across normal and emergency paths"
            )
        if not isinstance(self.limits, ExecutionLimits):
            raise ProtocolValidationError("protocol.limits must be ExecutionLimits")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "emergency", tuple(self.emergency))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "protocol.metadata"))

    @property
    def clock_unit(self) -> str:
        return "s"

    def to_dict(self) -> dict[str, Any]:
        return {
            "$schema": PROTOCOL_SCHEMA_NAME,
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "seed": self.seed,
            "clock": {"time_unit": "s", "horizon_s": self.horizon_s},
            "limits": {
                "max_commands": self.limits.max_commands,
                "max_loop_iterations": self.limits.max_loop_iterations,
                "max_sweep_cases": self.limits.max_sweep_cases,
                "max_branch_depth": self.limits.max_branch_depth,
                "max_emergency_commands": self.limits.max_emergency_commands,
            },
            "metadata": _encode_value(self.metadata),
            "nodes": [_node_to_dict(node) for node in self.nodes],
            "emergency": [_node_to_dict(node) for node in self.emergency],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Protocol":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("protocol declaration must be an object")
        allowed = {
            "$schema",
            "schema_version",
            "protocol_id",
            "seed",
            "clock",
            "limits",
            "metadata",
            "nodes",
            "emergency",
        }
        _reject_unknown(payload, allowed, "protocol")
        required = allowed
        missing = required - set(payload)
        if missing:
            raise ProtocolValidationError(f"protocol missing required fields: {sorted(missing)}")
        declaration = payload.get("$schema", PROTOCOL_SCHEMA_NAME)
        if (
            not isinstance(declaration, str)
            or declaration.rsplit("/", 1)[-1] != PROTOCOL_SCHEMA_NAME
        ):
            raise ProtocolValidationError("protocol.$schema must identify protocol.schema.json")
        if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
            raise ProtocolValidationError("protocol schema_version must be exactly 1.0.0")
        clock = payload.get("clock")
        if not isinstance(clock, Mapping):
            raise ProtocolValidationError("protocol.clock is required")
        _reject_unknown(clock, {"time_unit", "horizon_s"}, "protocol.clock")
        if clock.get("time_unit") != "s":
            raise ProtocolValidationError("protocol.clock.time_unit must be 's'")
        limits_raw = payload.get("limits")
        if not isinstance(limits_raw, Mapping):
            raise ProtocolValidationError("protocol.limits must be an object")
        limit_fields = set(vars(ExecutionLimits()))
        _reject_unknown(limits_raw, limit_fields, "protocol.limits")
        if limit_fields - set(limits_raw):
            raise ProtocolValidationError("protocol.limits is missing a required bound")
        limits = ExecutionLimits(**dict(limits_raw))
        nodes = payload.get("nodes")
        if not isinstance(nodes, list):
            raise ProtocolValidationError("protocol.nodes must be an array")
        emergency = payload.get("emergency", [])
        if not isinstance(emergency, list):
            raise ProtocolValidationError("protocol.emergency must be an array")
        protocol_id = payload.get("protocol_id", "")
        schema_version = payload.get("schema_version")
        horizon_s = clock.get("horizon_s")
        seed = payload.get("seed", 0)
        if (
            not isinstance(protocol_id, str)
            or not isinstance(schema_version, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ProtocolValidationError(
                "protocol identity, schema version, and seed have invalid types"
            )
        if isinstance(horizon_s, bool) or not isinstance(horizon_s, (int, float)):
            raise ProtocolValidationError("protocol.clock.horizon_s must be numeric")
        return cls(
            protocol_id=protocol_id,
            horizon_s=float(horizon_s),
            nodes=tuple(_node_from_dict(node) for node in nodes),
            seed=payload.get("seed", 0),
            metadata=_decode_value(payload.get("metadata", {})),
            limits=limits,
            emergency=tuple(_node_from_dict(node) for node in emergency),
            schema_version=schema_version,
        )


@dataclass(frozen=True)
class PhysioScript(Protocol):
    """Named spelling for the public declarative scientific DSL."""


class ProtocolBuilder:
    """Small Python builder that produces the same typed declaration as JSON."""

    def __init__(
        self,
        protocol_id: str,
        *,
        horizon_s: float,
        seed: int = 0,
        metadata: Mapping[str, Any] | None = None,
        limits: ExecutionLimits | None = None,
    ) -> None:
        self.protocol_id, self.horizon_s, self.seed = protocol_id, horizon_s, seed
        self.metadata = dict(metadata or {})
        self.limits = limits or ExecutionLimits()
        self._nodes: list[Node] = []
        self._emergency: list[Node] = []

    def add(self, node: Node, *, emergency: bool = False) -> "ProtocolBuilder":
        if not isinstance(node, Node):
            raise ProtocolValidationError("builder.add accepts typed protocol nodes")
        (self._emergency if emergency else self._nodes).append(node)
        return self

    def setup(
        self, node_id: str, parameters: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> "ProtocolBuilder":
        return self.add(Setup(node_id, parameters or {}, **kwargs))

    def calibration(
        self,
        node_id: str,
        channel: str,
        input_unit: str,
        output_unit: str,
        parameters: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> "ProtocolBuilder":
        return self.add(
            Calibration(node_id, channel, input_unit, output_unit, parameters or {}, **kwargs)
        )

    def baseline(self, node_id: str, duration_s: float, **kwargs: Any) -> "ProtocolBuilder":
        return self.add(Baseline(node_id, duration_s, **kwargs))

    def stimulus(
        self,
        node_id: str,
        actuator: str,
        value: float | Quantity,
        unit: str,
        duration_s: float,
        **kwargs: Any,
    ) -> "ProtocolBuilder":
        return self.add(Stimulus(node_id, actuator, value, unit, duration_s, **kwargs))

    def acquire(
        self, node_id: str, channel: str, sample_rate_hz: float, duration_s: float, **kwargs: Any
    ) -> "ProtocolBuilder":
        return self.add(Acquire(node_id, channel, sample_rate_hz, duration_s, **kwargs))

    def event(self, node_id: str, label: str, **kwargs: Any) -> "ProtocolBuilder":
        return self.add(Event(node_id, label, **kwargs))

    def wait(self, node_id: str, duration_s: float, **kwargs: Any) -> "ProtocolBuilder":
        return self.add(Wait(node_id, duration_s, **kwargs))

    def computation(
        self, node_id: str, output: str, output_unit: str, **kwargs: Any
    ) -> "ProtocolBuilder":
        return self.add(Computation(node_id, output, output_unit, **kwargs))

    def decision(self, node_id: str, condition: Condition, **kwargs: Any) -> "ProtocolBuilder":
        return self.add(Decision(node_id, condition, **kwargs))

    def if_branch(
        self,
        node_id: str,
        condition: Condition,
        then_nodes: Sequence[Node] = (),
        else_nodes: Sequence[Node] = (),
        **kwargs: Any,
    ) -> "ProtocolBuilder":
        return self.add(If(node_id, condition, tuple(then_nodes), tuple(else_nodes), **kwargs))

    def loop(
        self, node_id: str, count: int, body: Sequence[Node], **kwargs: Any
    ) -> "ProtocolBuilder":
        return self.add(Loop(node_id, count, tuple(body), **kwargs))

    def schedule(
        self, node_id: str, cases: Sequence[ScheduleCase], **kwargs: Any
    ) -> "ProtocolBuilder":
        return self.add(Schedule(node_id, tuple(cases), **kwargs))

    def sweep(
        self,
        node_id: str,
        parameters: Mapping[str, Sequence[Any]],
        body: Sequence[Node],
        **kwargs: Any,
    ) -> "ProtocolBuilder":
        return self.add(
            ParameterSweep(
                node_id,
                {key: tuple(values) for key, values in parameters.items()},
                tuple(body),
                **kwargs,
            )
        )

    def intervention(self, node_id: str, kind: str, **kwargs: Any) -> "ProtocolBuilder":
        return self.add(Intervention(node_id, kind, **kwargs))

    def resource(
        self,
        node_id: str,
        name: str,
        value: float | Quantity | str | bool,
        unit: str,
        **kwargs: Any,
    ) -> "ProtocolBuilder":
        return self.add(Resource(node_id, name, value, unit, **kwargs))

    def measurement(
        self, node_id: str, name: str, source: str, unit: str, **kwargs: Any
    ) -> "ProtocolBuilder":
        return self.add(Measurement(node_id, name, source, unit, **kwargs))

    def stop(self, node_id: str, reason: str, **kwargs: Any) -> "ProtocolBuilder":
        return self.add(Stop(node_id, reason, **kwargs))

    def export(self, node_id: str, target: str, format: str, **kwargs: Any) -> "ProtocolBuilder":
        return self.add(Export(node_id, target, format, **kwargs))

    def emergency(self, node: Node) -> "ProtocolBuilder":
        return self.add(node, emergency=True)

    def build(self, *, physioscript: bool = False) -> Protocol:
        cls = PhysioScript if physioscript else Protocol
        return cls(
            self.protocol_id,
            self.horizon_s,
            tuple(self._nodes),
            self.seed,
            self.metadata,
            self.limits,
            tuple(self._emergency),
        )


@dataclass
class SimulationClock:
    """Monotone protocol time in canonical seconds; no wall-clock behavior."""

    horizon_s: float
    now_s: float = 0.0

    def __post_init__(self) -> None:
        _duration(self.horizon_s, "clock.horizon_s", positive=True)
        _duration(self.now_s, "clock.now_s")
        if self.now_s > self.horizon_s:
            raise ProtocolValidationError("clock.now_s cannot exceed clock.horizon_s")

    def advance_to(self, target_s: float) -> None:
        _duration(target_s, "clock.target_s")
        if target_s < self.now_s:
            raise ProtocolError("protocol clock cannot move into the past")
        if target_s > self.horizon_s:
            raise ProtocolLimitError("protocol action exceeds the declared horizon")
        self.now_s = float(target_s)

    def advance(self, duration_s: float) -> None:
        _duration(duration_s, "clock.duration_s")
        self.advance_to(self.now_s + float(duration_s))


@dataclass
class ExecutionContext:
    """State exposed to injected handlers and retained in the execution result."""

    clock: SimulationClock
    seed: int
    variables: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Quantity | str | bool] = field(default_factory=dict)
    measurements: dict[str, Quantity] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[dict[str, Any]] = field(default_factory=list)
    chronology: list[dict[str, Any]] = field(default_factory=list)
    abort_requested: bool = False
    abort_reason: str = ""
    stop_requested: bool = False
    stop_reason: str = ""

    def resolve(self, path: str) -> Any:
        if not isinstance(path, str) or not path.strip():
            raise ProtocolError("empty state reference")
        clean = path[1:] if path.startswith("$") else path
        if clean.startswith("measurement."):
            key = clean.split(".", 1)[1]
            if key not in self.measurements:
                raise ProtocolError(f"reference is not available yet: {path}")
            return self.measurements[key]
        if clean.startswith("resource."):
            key = clean.split(".", 1)[1]
            if key not in self.resources:
                raise ProtocolError(f"reference is not available yet: {path}")
            return self.resources[key]
        if clean in self.measurements:
            return self.measurements[clean]
        if clean in self.resources:
            return self.resources[clean]
        if clean in self.variables:
            return self.variables[clean]
        if clean == "time_s":
            return Quantity(self.clock.now_s, "s")
        raise ProtocolError(f"reference is not available yet: {path}")

    def abort(self, reason: str) -> None:
        _nonempty(reason, "abort.reason")
        self.abort_requested, self.abort_reason = True, reason


Handler = Callable[[Node, ExecutionContext], Any]


class ProtocolHandler(typing.Protocol):
    """Native injected handler signature; handlers do not come from protocol data."""

    def __call__(self, node: Node, context: ExecutionContext) -> Any: ...


@dataclass(frozen=True)
class HandlerContract:
    operation: str
    required_parameters: tuple[str, ...] = ()
    allowed_parameters: tuple[str, ...] | None = None
    output_unit: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.operation, "handler contract.operation")
        if self.allowed_parameters is not None and not set(self.required_parameters) <= set(
            self.allowed_parameters
        ):
            raise ProtocolValidationError("required handler parameters must be allowed")
        if self.output_unit is not None:
            _check_unit(self.output_unit, "handler contract.output_unit")


class HandlerRegistry:
    """Operation handlers plus optional contracts, with no dynamic code lookup."""

    def __init__(
        self,
        handlers: Mapping[str, ProtocolHandler] | Any | None = None,
        *,
        contracts: Mapping[str, HandlerContract] | None = None,
        **named: ProtocolHandler,
    ) -> None:
        values: dict[str, ProtocolHandler] = {}
        if isinstance(handlers, Mapping):
            values.update(handlers)
        elif handlers is not None:
            for operation in _COMMAND_OPS:
                candidate = getattr(handlers, operation, None)
                if callable(candidate):
                    values[operation] = candidate
        values.update(named)
        if any(
            not isinstance(name, str) or not name.strip() or not callable(handler)
            for name, handler in values.items()
        ):
            raise ProtocolValidationError("handlers must map names to callables")
        self._handlers = dict(values)
        self.contracts = dict(contracts or {})

    def resolve(self, operation: str, name: str) -> ProtocolHandler:
        candidates = (f"{operation}.{name}", name if name != operation else "", operation)
        for candidate in candidates:
            if candidate and candidate in self._handlers:
                return self._handlers[candidate]
        raise ProtocolExecutionError(f"no injected handler for {operation!r} ({name!r})")

    def validate(self, node: Node) -> None:
        contract = self.contracts.get(f"{node.op}.{getattr(node, 'handler', node.op)}")
        contract = contract or self.contracts.get(getattr(node, "handler", node.op))
        if contract is None:
            return
        if contract.operation != node.op:
            raise ProtocolExecutionError(f"handler contract operation mismatch for {node.node_id}")
        parameters = getattr(node, "parameters", {})
        missing = set(contract.required_parameters) - set(parameters)
        if missing:
            raise ProtocolExecutionError(
                f"{node.node_id}: handler parameters missing {sorted(missing)}"
            )
        if contract.allowed_parameters is not None and not set(parameters) <= set(
            contract.allowed_parameters
        ):
            raise ProtocolExecutionError(
                f"{node.node_id}: handler parameters contain unknown names"
            )


@dataclass(frozen=True)
class ExecutionResult:
    protocol_id: str
    status: str
    time_s: float
    seed: int
    measurements: Mapping[str, Quantity]
    resources: Mapping[str, Quantity | str | bool]
    events: tuple[Mapping[str, Any], ...]
    actions: tuple[Mapping[str, Any], ...]
    chronology: tuple[Mapping[str, Any], ...]
    provenance: tuple[Mapping[str, Any], ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "physioworld.protocol.execution/1",
            "protocol_id": self.protocol_id,
            "status": self.status,
            "time_s": self.time_s,
            "seed": self.seed,
            "measurements": {
                name: _encode_value(value) for name, value in self.measurements.items()
            },
            "resources": {name: _encode_value(value) for name, value in self.resources.items()},
            "events": [_encode_value(value) for value in self.events],
            "actions": [_encode_value(value) for value in self.actions],
            "chronology": [_encode_value(value) for value in self.chronology],
            "provenance": [_encode_value(value) for value in self.provenance],
            "error": self.error,
        }


class ProtocolRuntime:
    """Execute a validated protocol against explicitly injected native handlers."""

    def __init__(
        self,
        handlers: HandlerRegistry | Mapping[str, ProtocolHandler] | Any,
        *,
        contracts: Mapping[str, HandlerContract] | None = None,
    ) -> None:
        self.handlers = (
            handlers
            if isinstance(handlers, HandlerRegistry)
            else HandlerRegistry(handlers, contracts=contracts)
        )

    def run(
        self,
        protocol: Protocol,
        *,
        initial_variables: Mapping[str, Any] | None = None,
        initial_resources: Mapping[str, Quantity | str | bool] | None = None,
    ) -> ExecutionResult:
        if not isinstance(protocol, Protocol):
            raise ProtocolValidationError("runtime requires a Protocol or PhysioScript")
        context = ExecutionContext(
            SimulationClock(protocol.horizon_s),
            protocol.seed,
            variables=dict(initial_variables or {}),
            resources=dict(initial_resources or {}),
        )
        declaration = json.dumps(
            protocol.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        context.provenance.append(
            {
                "kind": "protocol",
                "protocol_id": protocol.protocol_id,
                "schema_version": protocol.schema_version,
                "sha256": hashlib.sha256(declaration).hexdigest(),
                "seed": protocol.seed,
                "time_unit": "s",
                "evidence_class": "DECLARATION",
            }
        )
        status = "completed"
        error: str | None = None
        try:
            self._execute_nodes(protocol.nodes, protocol, context, {}, 0, False)
            if context.abort_requested:
                status = "aborted"
                self._run_emergency(protocol, context)
            elif context.stop_requested:
                status = "stopped"
        except ProtocolAbort as exc:
            status, error = "aborted", str(exc)
            context.abort_requested, context.abort_reason = True, str(exc)
            self._run_emergency(protocol, context)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._run_emergency(protocol, context)
            result = self._result(protocol, context, "failed", error)
            raise ProtocolExecutionError(error, result) from exc
        return self._result(protocol, context, status, error)

    def _result(
        self, protocol: Protocol, context: ExecutionContext, status: str, error: str | None
    ) -> ExecutionResult:
        return ExecutionResult(
            protocol.protocol_id,
            status,
            context.clock.now_s,
            protocol.seed,
            dict(context.measurements),
            dict(context.resources),
            tuple(dict(event) for event in context.events),
            tuple(dict(action) for action in context.actions),
            tuple(dict(item) for item in context.chronology),
            tuple(dict(item) for item in context.provenance),
            error,
        )

    def _run_emergency(self, protocol: Protocol, context: ExecutionContext) -> None:
        if not protocol.emergency:
            context.provenance.append(
                {"kind": "emergency", "status": "not_declared", "time_s": context.clock.now_s}
            )
            return
        context.provenance.append(
            {"kind": "emergency", "status": "started", "time_s": context.clock.now_s}
        )
        before = len(context.chronology)
        try:
            self._execute_nodes(protocol.emergency, protocol, context, {}, 0, True)
            used = len(context.chronology) - before
            if used > protocol.limits.max_emergency_commands:
                raise ProtocolLimitError("emergency sequence exceeded max_emergency_commands")
            context.provenance.append(
                {
                    "kind": "emergency",
                    "status": "completed",
                    "time_s": context.clock.now_s,
                    "commands": used,
                }
            )
        except Exception as exc:
            context.provenance.append(
                {
                    "kind": "emergency",
                    "status": "failed",
                    "time_s": context.clock.now_s,
                    "error": str(exc),
                }
            )

    def _execute_nodes(
        self,
        nodes: Sequence[Node],
        protocol: Protocol,
        context: ExecutionContext,
        parameters: Mapping[str, Any],
        depth: int,
        emergency: bool,
    ) -> None:
        if depth > protocol.limits.max_branch_depth:
            raise ProtocolLimitError("maximum protocol branch depth exceeded")
        for original in nodes:
            if context.stop_requested and not emergency:
                return
            if (
                emergency
                and len([item for item in context.chronology if item.get("emergency")])
                >= protocol.limits.max_emergency_commands
            ):
                raise ProtocolLimitError("maximum emergency commands exceeded")
            node = _materialize_node(original, parameters)
            self._position(node, context)
            if isinstance(node, Loop):
                total = node.count
                if total > protocol.limits.max_loop_iterations:
                    raise ProtocolLimitError(f"loop {node.node_id} exceeds max_loop_iterations")
                for iteration in range(total):
                    context.provenance.append(
                        {
                            "kind": "loop",
                            "node_id": node.node_id,
                            "iteration": iteration,
                            "time_s": context.clock.now_s,
                            "emergency": emergency,
                        }
                    )
                    self._execute_nodes(
                        node.body, protocol, context, parameters, depth + 1, emergency
                    )
                    if context.stop_requested or context.abort_requested:
                        return
                continue
            if isinstance(node, If):
                selected = self._decision(
                    node.condition,
                    node.handler if hasattr(node, "handler") else "builtin",
                    node,
                    protocol,
                    context,
                )
                context.provenance.append(
                    {
                        "kind": "branch",
                        "node_id": node.node_id,
                        "selected": "then" if selected else "else",
                        "time_s": context.clock.now_s,
                        "emergency": emergency,
                    }
                )
                self._execute_nodes(
                    node.then_nodes if selected else node.else_nodes,
                    protocol,
                    context,
                    parameters,
                    depth + 1,
                    emergency,
                )
                continue
            if isinstance(node, Decision):
                selected = self._decision(node.condition, node.handler, node, protocol, context)
                context.variables[f"decision.{node.node_id}"] = selected
                context.provenance.append(
                    {
                        "kind": "decision",
                        "node_id": node.node_id,
                        "selected": selected,
                        "time_s": context.clock.now_s,
                        "emergency": emergency,
                    }
                )
                self._execute_nodes(
                    node.then_nodes if selected else node.else_nodes,
                    protocol,
                    context,
                    parameters,
                    depth + 1,
                    emergency,
                )
                continue
            if isinstance(node, Schedule):
                self._execute_schedule(node, protocol, context, parameters, depth, emergency)
                continue
            if isinstance(node, ParameterSweep):
                self._execute_sweep(node, protocol, context, parameters, depth, emergency)
                continue
            self._dispatch(node, protocol, context, emergency)
            if context.abort_requested and not emergency:
                return

    def _position(self, node: Node, context: ExecutionContext) -> None:
        at_s = getattr(node, "at_s", None)
        if at_s is not None:
            context.clock.advance_to(at_s)
        context.chronology.append(
            {
                "kind": "position",
                "node_id": node.node_id,
                "op": node.op,
                "time_s": context.clock.now_s,
            }
        )

    def _execute_schedule(
        self,
        node: Schedule,
        protocol: Protocol,
        context: ExecutionContext,
        parameters: Mapping[str, Any],
        depth: int,
        emergency: bool,
    ) -> None:
        cases = list(node.cases)
        if node.mode == "randomized":
            digest = hashlib.sha256(f"{protocol.seed}:{node.node_id}".encode()).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
        else:
            rng = None
        for repetition in range(node.repetitions):
            if node.mode == "randomized":
                order = list(cases)
                assert rng is not None
                rng.shuffle(order)
            elif node.mode == "counterbalanced":
                start = (protocol.seed + repetition) % len(cases)
                order = cases[start:] + cases[:start]
            else:
                order = cases
            context.provenance.append(
                {
                    "kind": "schedule",
                    "node_id": node.node_id,
                    "mode": node.mode,
                    "repetition": repetition,
                    "order": [case.case_id for case in order],
                    "time_s": context.clock.now_s,
                    "emergency": emergency,
                }
            )
            for case in order:
                context.variables[f"schedule.{node.node_id}.case"] = case.case_id
                self._execute_nodes(case.nodes, protocol, context, parameters, depth + 1, emergency)
                if context.stop_requested or context.abort_requested:
                    return

    def _execute_sweep(
        self,
        node: ParameterSweep,
        protocol: Protocol,
        context: ExecutionContext,
        parameters: Mapping[str, Any],
        depth: int,
        emergency: bool,
    ) -> None:
        keys = tuple(sorted(node.parameters))
        values = [node.parameters[key] for key in keys]
        count = math.prod(len(value) for value in values)
        if count > protocol.limits.max_sweep_cases:
            raise ProtocolLimitError(f"sweep {node.node_id} exceeds max_sweep_cases")
        for index, combination in enumerate(itertools.product(*values)):
            local = dict(parameters)
            local.update(zip(keys, combination))
            context.provenance.append(
                {
                    "kind": "sweep",
                    "node_id": node.node_id,
                    "case": index,
                    "parameters": _encode_value(dict(zip(keys, combination))),
                    "time_s": context.clock.now_s,
                    "emergency": emergency,
                }
            )
            self._execute_nodes(node.body, protocol, context, local, depth + 1, emergency)
            if context.stop_requested or context.abort_requested:
                return

    def _decision(
        self,
        condition: Condition,
        handler_name: str,
        node: Node,
        protocol: Protocol,
        context: ExecutionContext,
    ) -> bool:
        if handler_name != "builtin":
            handler = self.handlers.resolve("decision", handler_name)
            result = handler(node, context)
            if not isinstance(result, bool):
                raise ProtocolExecutionError(
                    f"decision handler for {node.node_id} must return bool"
                )
            return result
        left, right = (
            _resolve_operand(condition.left, context),
            _resolve_operand(condition.right, context),
        )
        if isinstance(left, Quantity) or isinstance(right, Quantity):
            if not isinstance(left, Quantity) or not isinstance(right, Quantity):
                raise ProtocolError("quantity comparisons require units on both operands")
            right = right.to(left.unit)
            left_value, right_value = left.value, right.value
        else:
            left_value, right_value = left, right
        try:
            return {
                "==": lambda: left_value == right_value,
                "!=": lambda: left_value != right_value,
                "<": lambda: left_value < right_value,
                "<=": lambda: left_value <= right_value,
                ">": lambda: left_value > right_value,
                ">=": lambda: left_value >= right_value,
            }[condition.operator]()
        except (KeyError, TypeError) as exc:
            raise ProtocolError("condition operands are not comparable") from exc

    def _dispatch(
        self, node: Node, protocol: Protocol, context: ExecutionContext, emergency: bool
    ) -> None:
        command_count = sum(
            1
            for item in context.chronology
            if item.get("kind") == "dispatch" and not item.get("emergency")
        )
        if not emergency and command_count >= protocol.limits.max_commands:
            raise ProtocolLimitError("maximum protocol commands exceeded")
        self.handlers.validate(node)
        handler = self.handlers.resolve(node.op, getattr(node, "handler", node.op))
        start = context.clock.now_s
        context.provenance.append(
            {
                "kind": "dispatch",
                "node_id": node.node_id,
                "op": node.op,
                "handler": getattr(node, "handler", node.op),
                "time_s": start,
                "emergency": emergency,
                "evidence_class": "EXECUTION_TRACE",
            }
        )
        try:
            result = handler(node, context)
        except ProtocolAbort:
            raise
        except Exception as exc:
            raise ProtocolExecutionError(
                f"handler {node.op}/{getattr(node, 'handler', node.op)} failed: {exc}"
            ) from exc
        self._record_result(node, result, context)
        duration = getattr(node, "duration_s", 0.0)
        if duration:
            context.clock.advance(duration)
        entry = {
            "kind": "dispatch",
            "node_id": node.node_id,
            "op": node.op,
            "time_s": start,
            "end_time_s": context.clock.now_s,
            "emergency": emergency,
        }
        context.chronology.append(entry)
        if isinstance(node, Event):
            context.events.append(
                {
                    "time_s": start,
                    "label": node.label,
                    "attributes": _encode_value(node.attributes),
                    "node_id": node.node_id,
                }
            )
        if isinstance(node, Intervention):
            context.actions.append(
                {
                    "time_s": start,
                    "kind": node.kind,
                    "parameters": _encode_value(node.parameters),
                    "node_id": node.node_id,
                    "evidence_class": "DECLARED_ACTION",
                }
            )
        if isinstance(node, Stop):
            should_stop = node.condition is None or self._decision(
                node.condition, "builtin", node, protocol, context
            )
            if should_stop:
                context.stop_requested, context.stop_reason = True, node.reason
                if node.emergency:
                    context.abort_requested, context.abort_reason = True, node.reason

    def _record_result(self, node: Node, result: Any, context: ExecutionContext) -> None:
        if isinstance(node, (Measurement, Computation)):
            quantity = _result_quantity(
                result, getattr(node, "output_unit", getattr(node, "unit", "1")), node.node_id
            )
            name = node.name if isinstance(node, Measurement) else node.output
            context.measurements[name] = quantity
            context.variables[name] = quantity
        elif isinstance(node, Resource):
            value = result if result is not None else node.value
            context.resources[node.name] = _result_resource(value, node.unit, node.node_id)
            context.variables[f"resource.{node.name}"] = context.resources[node.name]
        elif result is not None:
            context.variables[f"result.{node.node_id}"] = result
        context.provenance.append(
            {
                "kind": "result",
                "node_id": node.node_id,
                "result": _encode_value(result),
                "time_s": context.clock.now_s,
                "evidence_class": "HANDLER_DECLARED",
            }
        )


_COMMAND_OPS = frozenset(
    {
        "setup",
        "calibration",
        "baseline",
        "stimulus",
        "acquire",
        "event",
        "wait",
        "computation",
        "decision",
        "intervention",
        "resource",
        "measurement",
        "stop",
        "export",
    }
)


def _resolve_operand(value: Operand, context: ExecutionContext) -> Any:
    if isinstance(value, Reference):
        return context.resolve(value.path)
    if isinstance(value, ParameterRef):
        raise ProtocolError(f"unresolved parameter reference {value.name!r}")
    return value


def _result_quantity(result: Any, declared_unit: str, node_id: str) -> Quantity:
    if isinstance(result, Quantity):
        try:
            return result.to(declared_unit)
        except ProtocolError as exc:
            raise ProtocolExecutionError(f"{node_id}: handler result unit mismatch") from exc
    if isinstance(result, Mapping) and set(result) == {"value", "unit"}:
        return Quantity(result["value"], result["unit"]).to(declared_unit)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ProtocolExecutionError(f"{node_id}: handler must return a scalar or Quantity")
    return Quantity(result, declared_unit)


def _result_resource(result: Any, declared_unit: str, node_id: str) -> Quantity | str | bool:
    if isinstance(result, Quantity):
        return result.to(declared_unit)
    if isinstance(result, Mapping) and set(result) == {"value", "unit"}:
        return Quantity(result["value"], result["unit"]).to(declared_unit)
    if isinstance(result, (str, bool)):
        return result
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        return Quantity(result, declared_unit)
    raise ProtocolExecutionError(f"{node_id}: resource handler returned an unsupported value")


def _encode_value(value: Any) -> Any:
    if isinstance(value, Quantity):
        return value.to_dict()
    if isinstance(value, Reference):
        return value.to_dict()
    if isinstance(value, ParameterRef):
        return value.to_dict()
    if isinstance(value, Condition):
        return {
            "left": _encode_value(value.left),
            "operator": value.operator,
            "right": _encode_value(value.right),
        }
    if isinstance(value, Mapping):
        return {str(k): _encode_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode_value(v) for v in value]
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode_value(item) for item in value)
    if isinstance(value, Mapping):
        if set(value) == {"$quantity"}:
            raw = value["$quantity"]
            if not isinstance(raw, Mapping) or set(raw) != {"value", "unit"}:
                raise ProtocolValidationError("malformed quantity marker")
            return Quantity(raw["value"], raw["unit"])
        if set(value) == {"$ref"}:
            return Reference(value["$ref"])
        if set(value) == {"$parameter"}:
            return ParameterRef(value["$parameter"])
        return {key: _decode_value(item) for key, item in value.items()}
    return value


def _materialize_value(value: Any, parameters: Mapping[str, Any]) -> Any:
    if isinstance(value, ParameterRef):
        if value.name not in parameters:
            raise ProtocolError(f"parameter {value.name!r} is not available in this sweep case")
        return copy.deepcopy(parameters[value.name])
    if isinstance(value, Mapping):
        return {key: _materialize_value(item, parameters) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_materialize_value(item, parameters) for item in value)
    if isinstance(value, Condition):
        return Condition(
            _materialize_value(value.left, parameters),
            value.operator,
            _materialize_value(value.right, parameters),
        )
    return value


def _materialize_node(node: Node, parameters: Mapping[str, Any]) -> Node:
    updates: dict[str, Any] = {}
    for name in ("parameters", "attributes", "value", "condition"):
        if hasattr(node, name):
            updates[name] = _materialize_value(getattr(node, name), parameters)
    if isinstance(node, ParameterSweep):
        return node
    return replace(node, **updates) if updates else node


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ProtocolValidationError(f"{label} contains unknown fields: {sorted(unknown)}")


def _node_to_dict(node: Node) -> dict[str, Any]:
    result: dict[str, Any] = {"op": node.op, "id": node.node_id}
    if isinstance(
        node,
        (
            Setup,
            Calibration,
            Acquire,
            Event,
            Wait,
            Computation,
            Intervention,
            Resource,
            Measurement,
            Stop,
            Export,
            Stimulus,
            Baseline,
            Decision,
        ),
    ):
        if getattr(node, "at_s", None) is not None:
            result["at_s"] = node.at_s
    if isinstance(node, Setup):
        result.update(
            parameters=_encode_value(node.parameters), units=dict(node.units), handler=node.handler
        )
    elif isinstance(node, Calibration):
        result.update(
            channel=node.channel,
            input_unit=node.input_unit,
            output_unit=node.output_unit,
            parameters=_encode_value(node.parameters),
            units=dict(node.units),
            handler=node.handler,
        )
    elif isinstance(node, Baseline):
        result.update(
            duration_s=node.duration_s, channels=list(node.channels), handler=node.handler
        )
    elif isinstance(node, Stimulus):
        result.update(
            actuator=node.actuator,
            value=_encode_value(node.value),
            unit=node.unit,
            duration_s=node.duration_s,
            handler=node.handler,
        )
    elif isinstance(node, Acquire):
        result.update(
            channel=node.channel,
            sample_rate_hz=node.sample_rate_hz,
            duration_s=node.duration_s,
            parameters=_encode_value(node.parameters),
            units=dict(node.units),
            handler=node.handler,
        )
    elif isinstance(node, Event):
        result.update(
            label=node.label, attributes=_encode_value(node.attributes), handler=node.handler
        )
    elif isinstance(node, Wait):
        result.update(duration_s=node.duration_s, handler=node.handler)
    elif isinstance(node, Computation):
        result.update(
            output=node.output,
            output_unit=node.output_unit,
            inputs=list(node.inputs),
            parameters=_encode_value(node.parameters),
            units=dict(node.units),
            handler=node.handler,
        )
    elif isinstance(node, Decision):
        result.update(
            condition=_encode_value(node.condition),
            then=[_node_to_dict(n) for n in node.then_nodes],
            else_=[_node_to_dict(n) for n in node.else_nodes],
            handler=node.handler,
        )
        result["else"] = result.pop("else_")
    elif isinstance(node, Intervention):
        result.update(
            kind=node.kind,
            parameters=_encode_value(node.parameters),
            units=dict(node.units),
            handler=node.handler,
        )
    elif isinstance(node, Resource):
        result.update(
            name=node.name, value=_encode_value(node.value), unit=node.unit, handler=node.handler
        )
    elif isinstance(node, Measurement):
        result.update(
            name=node.name,
            source=node.source,
            unit=node.unit,
            parameters=_encode_value(node.parameters),
            units=dict(node.units),
            handler=node.handler,
        )
    elif isinstance(node, Stop):
        result.update(
            reason=node.reason,
            condition=_encode_value(node.condition) if node.condition else None,
            emergency=node.emergency,
            handler=node.handler,
        )
    elif isinstance(node, Export):
        result.update(
            target=node.target,
            format=node.format,
            parameters=_encode_value(node.parameters),
            handler=node.handler,
        )
    elif isinstance(node, Loop):
        result.update(count=node.count, body=[_node_to_dict(n) for n in node.body])
        if node.at_s is not None:
            result["at_s"] = node.at_s
    elif isinstance(node, If):
        result.update(
            condition=_encode_value(node.condition),
            then=[_node_to_dict(n) for n in node.then_nodes],
            else_=[_node_to_dict(n) for n in node.else_nodes],
        )
        result["else"] = result.pop("else_")
        if node.at_s is not None:
            result["at_s"] = node.at_s
    elif isinstance(node, Schedule):
        result.update(
            mode=node.mode,
            repetitions=node.repetitions,
            cases=[
                {"id": case.case_id, "nodes": [_node_to_dict(n) for n in case.nodes]}
                for case in node.cases
            ],
        )
        if node.at_s is not None:
            result["at_s"] = node.at_s
    elif isinstance(node, ParameterSweep):
        result.update(
            parameters=_encode_value(node.parameters), body=[_node_to_dict(n) for n in node.body]
        )
        if node.at_s is not None:
            result["at_s"] = node.at_s
    return result


def _condition_from_dict(raw: Any) -> Condition:
    if not isinstance(raw, Mapping):
        raise ProtocolValidationError("condition must be an object")
    _reject_unknown(raw, {"left", "operator", "right"}, "condition")
    operator = raw.get("operator")
    if not isinstance(operator, str):
        raise ProtocolValidationError("condition.operator must be a string")
    return Condition(_decode_value(raw.get("left")), operator, _decode_value(raw.get("right")))


def _node_from_dict(raw: Any) -> Node:
    if not isinstance(raw, Mapping):
        raise ProtocolValidationError("protocol node must be an object")
    op = raw.get("op")
    if op not in _ALL_OPS:
        raise ProtocolValidationError(f"unknown protocol operation {op!r}")
    node_id = raw.get("id")
    if not isinstance(node_id, str):
        raise ProtocolValidationError("protocol node id must be a string")
    op = cast(str, op)
    common_at = raw.get("at_s", None)
    required_by_op = {
        "setup": {"op", "id", "parameters", "units", "handler"},
        "calibration": {
            "op",
            "id",
            "channel",
            "input_unit",
            "output_unit",
            "parameters",
            "units",
            "handler",
        },
        "baseline": {"op", "id", "duration_s", "channels", "handler"},
        "stimulus": {"op", "id", "actuator", "value", "unit", "duration_s", "handler"},
        "acquire": {
            "op",
            "id",
            "channel",
            "sample_rate_hz",
            "duration_s",
            "parameters",
            "units",
            "handler",
        },
        "event": {"op", "id", "label", "attributes", "handler"},
        "wait": {"op", "id", "duration_s", "handler"},
        "computation": {
            "op",
            "id",
            "output",
            "output_unit",
            "inputs",
            "parameters",
            "units",
            "handler",
        },
        "decision": {"op", "id", "condition", "then", "else", "handler"},
        "if": {"op", "id", "condition", "then", "else"},
        "intervention": {"op", "id", "kind", "parameters", "units", "handler"},
        "resource": {"op", "id", "name", "value", "unit", "handler"},
        "measurement": {"op", "id", "name", "source", "unit", "parameters", "units", "handler"},
        "stop": {"op", "id", "reason", "condition", "emergency", "handler"},
        "export": {"op", "id", "target", "format", "parameters", "handler"},
        "loop": {"op", "id", "count", "body"},
        "schedule": {"op", "id", "mode", "repetitions", "cases"},
        "sweep": {"op", "id", "parameters", "body"},
    }
    missing = required_by_op[op] - set(raw)
    if missing:
        raise ProtocolValidationError(f"{op} node missing required fields: {sorted(missing)}")
    if op == "setup":
        _reject_unknown(raw, {"op", "id", "parameters", "units", "handler", "at_s"}, op)
        return Setup(
            node_id,
            _decode_value(raw.get("parameters", {})),
            raw.get("units", {}),
            cast(str, raw.get("handler", "setup")),
            common_at,
        )
    if op == "calibration":
        _reject_unknown(
            raw,
            {
                "op",
                "id",
                "channel",
                "input_unit",
                "output_unit",
                "parameters",
                "units",
                "handler",
                "at_s",
            },
            op,
        )
        return Calibration(
            node_id,
            cast(str, raw.get("channel")),
            cast(str, raw.get("input_unit")),
            cast(str, raw.get("output_unit")),
            _decode_value(raw.get("parameters", {})),
            raw.get("units", {}),
            cast(str, raw.get("handler", "calibration")),
            common_at,
        )
    if op == "baseline":
        _reject_unknown(raw, {"op", "id", "duration_s", "channels", "handler", "at_s"}, op)
        return Baseline(
            node_id,
            cast(float, raw.get("duration_s")),
            tuple(raw.get("channels", ())),
            cast(str, raw.get("handler", "baseline")),
            common_at,
        )
    if op == "stimulus":
        _reject_unknown(
            raw, {"op", "id", "actuator", "value", "unit", "duration_s", "handler", "at_s"}, op
        )
        return Stimulus(
            node_id,
            cast(str, raw.get("actuator")),
            _decode_value(raw.get("value")),
            cast(str, raw.get("unit")),
            cast(float, raw.get("duration_s")),
            cast(str, raw.get("handler", "stimulus")),
            common_at,
        )
    if op == "acquire":
        _reject_unknown(
            raw,
            {
                "op",
                "id",
                "channel",
                "sample_rate_hz",
                "duration_s",
                "parameters",
                "units",
                "handler",
                "at_s",
            },
            op,
        )
        return Acquire(
            node_id,
            cast(str, raw.get("channel")),
            cast(float, raw.get("sample_rate_hz")),
            cast(float, raw.get("duration_s")),
            _decode_value(raw.get("parameters", {})),
            raw.get("units", {}),
            cast(str, raw.get("handler", "acquire")),
            common_at,
        )
    if op == "event":
        _reject_unknown(raw, {"op", "id", "label", "attributes", "handler", "at_s"}, op)
        return Event(
            node_id,
            cast(str, raw.get("label")),
            _decode_value(raw.get("attributes", {})),
            cast(str, raw.get("handler", "event")),
            common_at,
        )
    if op == "wait":
        _reject_unknown(raw, {"op", "id", "duration_s", "handler", "at_s"}, op)
        return Wait(
            node_id,
            cast(float, raw.get("duration_s")),
            cast(str, raw.get("handler", "wait")),
            common_at,
        )
    if op == "computation":
        _reject_unknown(
            raw,
            {
                "op",
                "id",
                "output",
                "output_unit",
                "inputs",
                "parameters",
                "units",
                "handler",
                "at_s",
            },
            op,
        )
        return Computation(
            node_id,
            cast(str, raw.get("output")),
            cast(str, raw.get("output_unit")),
            tuple(raw.get("inputs", ())),
            _decode_value(raw.get("parameters", {})),
            raw.get("units", {}),
            cast(str, raw.get("handler", "computation")),
            common_at,
        )
    if op in {"decision", "if"}:
        _reject_unknown(
            raw,
            {"op", "id", "condition", "then", "else", "handler", "at_s"}
            if op == "decision"
            else {"op", "id", "condition", "then", "else", "at_s"},
            op,
        )
        condition = _condition_from_dict(raw.get("condition"))
        then = tuple(_node_from_dict(item) for item in raw.get("then", []))
        otherwise = tuple(_node_from_dict(item) for item in raw.get("else", []))
        return (
            Decision(
                node_id,
                condition,
                then,
                otherwise,
                cast(str, raw.get("handler", "builtin")),
                common_at,
            )
            if op == "decision"
            else If(node_id, condition, then, otherwise, common_at)
        )
    if op == "intervention":
        _reject_unknown(raw, {"op", "id", "kind", "parameters", "units", "handler", "at_s"}, op)
        return Intervention(
            node_id,
            cast(str, raw.get("kind")),
            _decode_value(raw.get("parameters", {})),
            raw.get("units", {}),
            cast(str, raw.get("handler", "intervention")),
            common_at,
        )
    if op == "resource":
        _reject_unknown(raw, {"op", "id", "name", "value", "unit", "handler", "at_s"}, op)
        return Resource(
            node_id,
            cast(str, raw.get("name")),
            _decode_value(raw.get("value")),
            cast(str, raw.get("unit")),
            cast(str, raw.get("handler", "resource")),
            common_at,
        )
    if op == "measurement":
        _reject_unknown(
            raw,
            {"op", "id", "name", "source", "unit", "parameters", "units", "handler", "at_s"},
            op,
        )
        return Measurement(
            node_id,
            cast(str, raw.get("name")),
            cast(str, raw.get("source")),
            cast(str, raw.get("unit")),
            _decode_value(raw.get("parameters", {})),
            raw.get("units", {}),
            cast(str, raw.get("handler", "measurement")),
            common_at,
        )
    if op == "stop":
        _reject_unknown(
            raw, {"op", "id", "reason", "condition", "emergency", "handler", "at_s"}, op
        )
        return Stop(
            node_id,
            cast(str, raw.get("reason")),
            _condition_from_dict(raw["condition"]) if raw.get("condition") is not None else None,
            cast(bool, raw.get("emergency", False)),
            cast(str, raw.get("handler", "stop")),
            common_at,
        )
    if op == "export":
        _reject_unknown(raw, {"op", "id", "target", "format", "parameters", "handler", "at_s"}, op)
        return Export(
            node_id,
            cast(str, raw.get("target")),
            cast(str, raw.get("format")),
            _decode_value(raw.get("parameters", {})),
            cast(str, raw.get("handler", "export")),
            common_at,
        )
    if op == "loop":
        _reject_unknown(raw, {"op", "id", "count", "body", "at_s"}, op)
        return Loop(
            node_id,
            cast(int, raw.get("count")),
            tuple(_node_from_dict(item) for item in raw.get("body", [])),
            common_at,
        )
    if op == "schedule":
        _reject_unknown(raw, {"op", "id", "mode", "repetitions", "cases", "at_s"}, op)
        cases = []
        for case in raw.get("cases", []):
            if not isinstance(case, Mapping):
                raise ProtocolValidationError("schedule cases must be objects")
            _reject_unknown(case, {"id", "nodes"}, "schedule.case")
            cases.append(
                ScheduleCase(
                    cast(str, case.get("id")),
                    tuple(_node_from_dict(item) for item in case.get("nodes", [])),
                )
            )
        return Schedule(
            node_id,
            tuple(cases),
            cast(str, raw.get("mode", "randomized")),
            cast(int, raw.get("repetitions", 1)),
            common_at,
        )
    _reject_unknown(raw, {"op", "id", "parameters", "body", "at_s"}, op)
    params = _decode_value(raw.get("parameters", {}))
    if not isinstance(params, Mapping):
        raise ProtocolValidationError("sweep.parameters must be an object")
    return ParameterSweep(
        node_id,
        {key: tuple(value) for key, value in params.items()},
        tuple(_node_from_dict(item) for item in raw.get("body", [])),
        common_at,
    )


_ALL_OPS = _COMMAND_OPS | {"loop", "if", "schedule", "sweep"}


def load_protocol_schema() -> dict[str, Any]:
    """Load the bundled strict protocol schema from an installed package."""
    resource = files("physioworld").joinpath("schemas").joinpath(PROTOCOL_SCHEMA_NAME)
    return json.loads(resource.read_text(encoding="utf-8"))


__all__ = [
    "Acquire",
    "Baseline",
    "Calibration",
    "Computation",
    "Condition",
    "Decision",
    "Event",
    "ExecutionContext",
    "ExecutionLimits",
    "ExecutionResult",
    "Export",
    "HandlerContract",
    "HandlerRegistry",
    "If",
    "Intervention",
    "Loop",
    "Measurement",
    "Node",
    "ParameterRef",
    "ParameterSweep",
    "PhysioScript",
    "Protocol",
    "ProtocolAbort",
    "ProtocolBuilder",
    "ProtocolError",
    "ProtocolExecutionError",
    "ProtocolHandler",
    "ProtocolLimitError",
    "ProtocolRuntime",
    "ProtocolValidationError",
    "Quantity",
    "Reference",
    "Resource",
    "Schedule",
    "ScheduleCase",
    "Setup",
    "SimulationClock",
    "Stop",
    "Stimulus",
    "Wait",
    "load_protocol_schema",
]

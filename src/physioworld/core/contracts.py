"""Versioned scientific contracts and strict estimand comparison.

The JSON format contains a record type and schema version at every record boundary.
Decoding rejects unknown fields/versions rather than silently reinterpreting a result.
"""

from __future__ import annotations

import json
import math
import types
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, fields
from enum import Enum
from numbers import Integral, Real
from typing import (
    Any,
    ClassVar,
    Mapping,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

RecordT = TypeVar("RecordT", bound="VersionedRecord")


def finite(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _encode(value: Any) -> Any:
    if isinstance(value, VersionedRecord):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, MappingABC):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON record mappings require string keys")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        finite(value, "serialized value")
        return int(value) if isinstance(value, Integral) else float(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"not a serializable scientific value: {type(value).__name__}")


def _decode(value: Any, annotation: Any) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        if isinstance(value, list):
            return tuple(_decode(item, Any) for item in value)
        if isinstance(value, dict):
            return {key: _decode(item, Any) for key, item in value.items()}
        return value
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _decode(value, option)
            except (ValueError, TypeError, KeyError):
                pass
        raise ValueError(f"value does not match {annotation}")
    if annotation is type(None):
        if value is not None:
            raise ValueError("expected null")
        return None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and issubclass(annotation, VersionedRecord):
        return annotation.from_dict(value)
    if origin in (tuple, list, SequenceABC):
        if not isinstance(value, (list, tuple)):
            raise ValueError("expected an array")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(value) != len(args):
                raise ValueError("tuple has wrong length")
            return tuple(_decode(item, typ) for item, typ in zip(value, args))
        items = [_decode(item, args[0] if args else Any) for item in value]
        return items if origin is list else tuple(items)
    if origin in (dict, MappingABC):
        if not isinstance(value, dict):
            raise ValueError("expected a mapping")
        return {_decode(key, args[0]): _decode(item, args[1]) for key, item in value.items()}
    if annotation is float:
        finite(value, "decoded value")
        return float(value)
    if annotation in (str, int, bool):
        if type(value) is not annotation:
            raise ValueError(f"expected {annotation.__name__}")
        return value
    raise TypeError(f"unsupported schema annotation: {annotation}")


class VersionedRecord:
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": type(self).__name__,
            **{item.name: _encode(getattr(self, item.name)) for item in fields(cast(Any, self))},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls: type[RecordT], payload: Mapping[str, Any]) -> RecordT:
        if not isinstance(payload, MappingABC):
            raise ValueError("record must be a mapping")
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported {cls.__name__} schema version")
        if payload.get("record_type") != cls.__name__:
            raise ValueError(f"expected record_type {cls.__name__}")
        allowed = {item.name for item in fields(cast(Any, cls))}
        unknown = set(payload) - allowed - {"schema_version", "record_type"}
        if unknown:
            raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
        hints = get_type_hints(cls)
        return cls(
            **{key: _decode(value, hints[key]) for key, value in payload.items() if key in allowed}
        )

    @classmethod
    def from_json(cls: type[RecordT], payload: str) -> RecordT:
        return cls.from_dict(json.loads(payload))


class EvidenceClass(str, Enum):
    THEOREM = "THEOREM"
    DEFINITION = "DEFINITION"
    ASSUMPTION = "ASSUMPTION"
    MODEL = "MODEL"
    SIMULATION = "SIMULATION"
    NUMERICAL_VERIFICATION = "NUMERICAL_VERIFICATION"
    EMPIRICAL_ESTIMATE = "EMPIRICAL_ESTIMATE"
    EXTERNAL_VALIDATION = "EXTERNAL_VALIDATION"
    CAUSAL_IDENTIFICATION = "CAUSAL_IDENTIFICATION"
    CLINICAL_EVIDENCE = "CLINICAL_EVIDENCE"
    HYPOTHESIS = "HYPOTHESIS"
    INTERPRETATION = "INTERPRETATION"


class CausalStatus(str, Enum):
    PREDICTIVE = "predictive"
    MODEL_INTERVENTIONAL = "model_interventional"
    IDENTIFIED = "identified_causal"
    UNIDENTIFIED = "unidentified_causal"
    NOT_APPLICABLE = "not_applicable"


class TaskKind(str, Enum):
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    FORECASTING = "forecasting"
    ESTIMATION = "estimation"
    RANKING = "ranking"
    EVENT_TIME = "event_time"
    DECISION = "decision"
    SEQUENTIAL_DECISION = "sequential_decision"
    POLICY_EVALUATION = "policy_evaluation"
    INTERVENTIONAL_OUTCOME = "interventional_outcome"
    SEQUENTIAL_INTERVENTION = "sequential_intervention"


@dataclass(frozen=True)
class Task(VersionedRecord):
    """Task semantics; the first fields retain the legacy TaskDefinition signature."""

    task_id: str
    kind: TaskKind
    horizon: float
    target: str
    loss_or_utility: str
    constraints: tuple[str, ...] = ()
    evaluation_timing: str = "end_of_horizon"
    acceptable_uncertainty: float | None = None
    clinical_relevance: str | None = None
    aggregation_rule: str = "mean"
    threshold: float | str | None = None
    utility: str | None = None
    clinical_meaning: str | None = None
    uncertainty_method: str | None = None
    prediction_time: str = "time_zero"
    evaluation_population: str = "unspecified"
    action_semantics: str | None = None
    intervention_semantics: str | None = None
    endpoint_definition: str | None = None
    resource_regime: str = "unspecified"
    reference_condition: str = "unspecified"
    time_unit: str = "s"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TaskKind(self.kind))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        self.validate()

    def validate(self) -> None:
        finite(self.horizon, "task horizon")
        if self.horizon < 0:
            raise ValueError("task horizon cannot be negative")
        for name in (
            "task_id",
            "target",
            "loss_or_utility",
            "evaluation_timing",
            "aggregation_rule",
            "prediction_time",
            "time_unit",
            "version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"task definition missing {name}")
        if isinstance(self.threshold, (int, float)):
            finite(self.threshold, "task threshold")
        if self.acceptable_uncertainty is not None:
            finite(self.acceptable_uncertainty, "acceptable_uncertainty")
            if self.acceptable_uncertainty < 0:
                raise ValueError("uncertainty cannot be negative")


@dataclass(frozen=True)
class MetricContext(VersionedRecord):
    estimand_name: str
    data_domain: str
    evaluation_regime: str
    conditioning: tuple[str, ...]
    policy_class: str
    reference_state: str
    time_horizon: float
    task_family: tuple[str, ...]
    resource_state: str
    uncertainty_treatment: str
    metric_type: str = "task_value"
    utility: str = "unspecified"
    population: str = "unspecified"
    unit: str = "1"
    causal_status: CausalStatus = CausalStatus.PREDICTIVE
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION
    communication_regime: str = "none"
    information_condition: str = "unspecified"
    estimator: str = "unspecified"
    time_unit: str = "s"

    def __post_init__(self) -> None:
        object.__setattr__(self, "causal_status", CausalStatus(self.causal_status))
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "conditioning", tuple(self.conditioning))
        object.__setattr__(self, "task_family", tuple(self.task_family))
        finite(self.time_horizon, "time horizon")
        if self.time_horizon < 0 or not self.task_family:
            raise ValueError("context needs a nonnegative horizon and a task family")
        if any(
            not isinstance(name, str) or not name
            for name in (*self.conditioning, *self.task_family)
        ):
            raise ValueError("context conditioning/task-family labels must be nonempty strings")
        if len(set(self.task_family)) != len(self.task_family):
            raise ValueError("context task-family IDs must be unique")
        for item in fields(self):
            if item.name in {
                "time_horizon",
                "conditioning",
                "task_family",
                "causal_status",
                "evidence_class",
            }:
                continue
            value = getattr(self, item.name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"context missing {item.name}")


class EstimandMismatch(ValueError):
    pass


class EstimandFirewall:
    """Exact context comparisons; no implicit estimand or evidence conversion."""

    @staticmethod
    def differences(left: MetricContext, right: MetricContext) -> tuple[str, ...]:
        return tuple(
            item.name
            for item in fields(MetricContext)
            if getattr(left, item.name) != getattr(right, item.name)
        )

    @classmethod
    def compare(cls, left: MetricContext, right: MetricContext) -> None:
        differences = cls.differences(left, right)
        if differences:
            raise EstimandMismatch(f"incompatible metric contexts: {', '.join(differences)}")

    @classmethod
    def common_reference(
        cls, left: MetricContext, right: MetricContext, *, allow_information_change: bool = False
    ) -> None:
        """Only an explicitly declared information intervention may vary here.

        Policy-class comparability must be established by the caller's design;
        differing utilities/resources/references cannot be waived by this method.
        """
        cls.require_complete_reference(left)
        cls.require_complete_reference(right)
        allowed = {"information_condition"} if allow_information_change else set()
        differences = set(cls.differences(left, right)) - allowed
        if differences:
            raise EstimandMismatch(f"common-reference failure: {', '.join(sorted(differences))}")

    @staticmethod
    def require_complete_reference(context: MetricContext) -> None:
        required = (
            "utility",
            "population",
            "reference_state",
            "policy_class",
            "resource_state",
            "communication_regime",
            "estimator",
            "uncertainty_treatment",
        )
        unspecified = [name for name in required if getattr(context, name) == "unspecified"]
        if unspecified:
            raise EstimandMismatch(
                f"common reference has unspecified fields: {', '.join(unspecified)}"
            )

    @staticmethod
    def require_metric(context: MetricContext, metric_type: str, estimand_name: str) -> None:
        if context.metric_type != metric_type or context.estimand_name != estimand_name:
            raise EstimandMismatch(
                f"expected {metric_type}/{estimand_name}, got "
                f"{context.metric_type}/{context.estimand_name}"
            )


@dataclass(frozen=True)
class Metric(VersionedRecord):
    value: float
    context: MetricContext
    source_artifact: str

    def __post_init__(self) -> None:
        finite(self.value, "metric value")
        if not self.source_artifact:
            raise ValueError("metric needs a source artifact")


def compile_task(
    task: Task,
    *,
    metric_type: str,
    estimand_name: str,
    data_domain: str,
    evaluation_regime: str,
    policy_class: str,
    conditioning: tuple[str, ...] = (),
    unit: str = "1",
    evidence_class: EvidenceClass = EvidenceClass.EMPIRICAL_ESTIMATE,
    causal_status: CausalStatus = CausalStatus.PREDICTIVE,
    estimator: str = "unspecified",
) -> MetricContext:
    """Compile declared task semantics, refusing a falsely predictive causal endpoint."""
    task.validate()
    causal_kinds = {TaskKind.INTERVENTIONAL_OUTCOME, TaskKind.SEQUENTIAL_INTERVENTION}
    if task.kind in causal_kinds and causal_status in {
        CausalStatus.PREDICTIVE,
        CausalStatus.NOT_APPLICABLE,
    }:
        raise EstimandMismatch("intervention task requires explicit causal status")
    value_kinds = causal_kinds | {
        TaskKind.DECISION,
        TaskKind.SEQUENTIAL_DECISION,
        TaskKind.POLICY_EVALUATION,
    }
    if task.kind in value_kinds and metric_type in {"auc", "correlation", "packet_delivery"}:
        raise EstimandMismatch("a predictive/transport score cannot replace task utility")
    return MetricContext(
        estimand_name,
        data_domain,
        evaluation_regime,
        conditioning,
        policy_class,
        task.reference_condition,
        task.horizon,
        (task.task_id,),
        task.resource_regime,
        task.uncertainty_method or "none_declared",
        metric_type,
        task.utility or task.loss_or_utility,
        task.evaluation_population,
        unit,
        causal_status,
        evidence_class,
        estimator=estimator,
        time_unit=task.time_unit,
    )

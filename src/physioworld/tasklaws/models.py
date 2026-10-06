"""Task entities and empirical conditional-law comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from physioworld.core.contracts import (
    EvidenceClass,
    MetricContext,
    Task,
    VersionedRecord,
    finite,
)
from physioworld.core.contracts import TaskKind as TaskKind
from physioworld.evidence_origin import require_verified_artifact_metadata

TaskDefinition = Task  # compatible constructor; one authoritative task schema


@dataclass(frozen=True)
class DiscreteDistribution(VersionedRecord):
    """A probability law, not just its conditional expectation."""

    support: tuple[float, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "support", tuple(self.support))
        object.__setattr__(self, "probabilities", tuple(self.probabilities))
        if not self.support or len(self.support) != len(self.probabilities):
            raise ValueError("law needs equal nonempty support and probability arrays")
        if len(set(self.support)) != len(self.support):
            raise ValueError("law support must be unique")
        for value in (*self.support, *self.probabilities):
            finite(value, "distribution value")
        if (
            any(p < 0 or p > 1 for p in self.probabilities)
            or abs(sum(float(p) for p in self.probabilities) - 1) > 1e-12
        ):
            raise ValueError("probability law must be normalized and nonnegative")

    def expectation(self, function: Callable[[float], float] | None = None) -> float:
        function = function or (lambda value: value)
        values = tuple(function(value) for value in self.support)
        for value in values:
            finite(value, "task function value")
        return sum(value * p for value, p in zip(values, self.probabilities))

    def cdf(self, value: float) -> float:
        finite(value, "CDF threshold")
        return sum(p for item, p in zip(self.support, self.probabilities) if item <= value)

    def quantile(self, probability: float) -> float:
        finite(probability, "quantile probability")
        if not 0 <= probability <= 1:
            raise ValueError("quantile probability must lie in [0, 1]")
        cumulative = 0.0
        ordered = sorted((value, p) for value, p in zip(self.support, self.probabilities) if p > 0)
        for value, p in ordered:
            cumulative += p
            if cumulative >= probability:
                return value
        return ordered[-1][0]


@dataclass(frozen=True)
class ConditionalCell(VersionedRecord):
    group: Any
    distribution: DiscreteDistribution
    count: int

    def __post_init__(self) -> None:
        hash(self.group)
        if self.count < 1:
            raise ValueError("conditional cell count must be positive")


@dataclass(frozen=True)
class TaskLaw(VersionedRecord):
    task: TaskDefinition
    full_information_values: tuple[float, ...]
    representation_values: tuple[float, ...]
    conditioning_labels: tuple[Any, ...] = ()
    assumptions: tuple[str, ...] = ()
    conditional_distributions: tuple[ConditionalCell, ...] = ()
    history_distributions: tuple[ConditionalCell, ...] = ()
    task_function: str = "identity"
    estimator: str = "declared_conditional_expectations"
    context: MetricContext | None = None
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION
    evidence_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "full_information_values", tuple(self.full_information_values))
        object.__setattr__(self, "representation_values", tuple(self.representation_values))
        object.__setattr__(self, "conditioning_labels", tuple(self.conditioning_labels))
        if not isinstance(self.evidence_metadata, Mapping):
            raise ValueError("task-law evidence_metadata must be a mapping")
        object.__setattr__(self, "evidence_metadata", dict(self.evidence_metadata))
        require_verified_artifact_metadata(self.evidence_class, self.evidence_metadata)
        if (
            len(self.full_information_values) != len(self.representation_values)
            or not self.full_information_values
        ):
            raise ValueError("task-law arrays must have equal nonzero length")
        for value in (*self.full_information_values, *self.representation_values):
            finite(value, "conditional task expectation")

    def distribution(self, group: object) -> DiscreteDistribution:
        for cell in self.conditional_distributions:
            if cell.group == group:
                return cell.distribution
        raise KeyError(group)

    def discrepancy(self) -> float:
        if len(self.full_information_values) != len(self.representation_values):
            raise ValueError("task-law arrays must have equal length")
        if not self.full_information_values:
            return 0.0
        return sum(
            abs(a - b) for a, b in zip(self.full_information_values, self.representation_values)
        ) / len(self.full_information_values)


@dataclass(frozen=True)
class TaskLawComparison(VersionedRecord):
    discrepancies: Mapping[str, float]
    aggregation: str
    per_observation_losses: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    estimand: str = "sup_task_mean_absolute_conditional_expectation_difference"

    def __post_init__(self) -> None:
        if not self.discrepancies:
            raise ValueError("task family cannot be empty")
        for value in self.discrepancies.values():
            finite(value, "task discrepancy")
            if value < 0:
                raise ValueError("task discrepancy cannot be negative")

    @property
    def worst_task(self) -> float:
        return max(self.discrepancies.values(), default=0.0)

    @property
    def average_task(self) -> float:
        values = tuple(self.discrepancies.values())
        return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class BootstrapResult(VersionedRecord):
    point_estimate: float
    bootstrap_mean: float
    lower: float
    upper: float
    confidence: float
    replicates: tuple[float, ...]
    method: str
    resampling_unit: str
    seed: int | None
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION
    evidence_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        if not isinstance(self.evidence_metadata, Mapping):
            raise ValueError("bootstrap evidence_metadata must be a mapping")
        object.__setattr__(self, "evidence_metadata", dict(self.evidence_metadata))
        require_verified_artifact_metadata(self.evidence_class, self.evidence_metadata)
        for value in (
            self.point_estimate,
            self.bootstrap_mean,
            self.lower,
            self.upper,
            *self.replicates,
        ):
            finite(value, "bootstrap value")
        if not 0 < self.confidence < 1 or self.lower > self.upper or not self.replicates:
            raise ValueError("invalid bootstrap uncertainty output")
        if abs(self.bootstrap_mean - sum(self.replicates) / len(self.replicates)) > 1e-12 * max(
            1, abs(self.bootstrap_mean)
        ):
            raise ValueError("bootstrap mean does not match the supplied replicate estimates")

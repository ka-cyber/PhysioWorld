"""Event-time drift diagnostics with explicit scientific boundaries.

The diagnostics in this module are deliberately small and dependency free.  A
covariate diagnostic compares observed feature distributions.  A concept
diagnostic compares binned conditional label means.  A performance diagnostic
compares task values carrying an identical :class:`MetricContext`.  These are
different objects and none of them is a causal effect estimate.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from physioworld.core.contracts import (
    CausalStatus,
    EstimandFirewall,
    EvidenceClass,
    MetricContext,
    VersionedRecord,
    finite,
)


class MonitoringError(ValueError):
    """Base error for invalid monitoring data or monitoring configuration."""


class FutureDataError(MonitoringError):
    """Raised when a diagnostic would use data unavailable at its declared time."""


class ChronologyError(MonitoringError):
    """Raised when an online chronological operation receives time-traveling data."""


class DriftKind(str, Enum):
    COVARIATE = "covariate"
    CONCEPT = "concept"
    PERFORMANCE = "performance"


class DriftMethod(str, Enum):
    TWO_SAMPLE_KS = "two_sample_ks"
    BINNED_CONDITIONAL_MEAN = "binned_conditional_label_mean"
    TASK_VALUE_DELTA = "task_value_delta"


class DriftDomain(str, Enum):
    """Declared source/domain of a covariate change."""

    PHYSIOLOGY = "physiology"
    DEVICE = "device"
    SITE = "site"
    PROTOCOL = "protocol"
    POPULATION = "population"
    CALIBRATION = "calibration"


@dataclass(frozen=True)
class EventObservation(VersionedRecord):
    """One event-time observation and its explicit availability metadata.

    ``event_time`` is the time at which the observation belongs in a window;
    ``input_available_s`` and ``label_available_s`` describe when it could be
    used.  Missing availability is allowed for unsupervised covariate scans but
    supervised training/evaluation guards require it to be declared.
    """

    record_id: str
    subject_id: str
    event_time: float
    features: Mapping[str, float]
    prediction: float | None = None
    label: float | None = None
    input_available_s: float | None = None
    label_available_s: float | None = None
    prediction_time_s: float | None = None
    clock_basis: str = "event_time"
    source_record_id: str | None = None
    label_used_as_feature: bool = False

    def __post_init__(self) -> None:
        finite(self.event_time, "event_time")
        if not self.record_id or not self.subject_id or not self.clock_basis:
            raise ValueError("event observations need record, subject, and clock identities")
        if not self.features:
            raise ValueError("event observations require at least one feature")
        object.__setattr__(self, "features", dict(self.features))
        for name, value in self.features.items():
            if not isinstance(name, str) or not name:
                raise ValueError("feature names must be nonempty strings")
            finite(value, f"feature {name}")
        for name in (
            "prediction",
            "label",
            "input_available_s",
            "label_available_s",
            "prediction_time_s",
        ):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
        if self.prediction_time_s is not None and self.input_available_s is not None:
            if self.input_available_s > self.prediction_time_s:
                raise FutureDataError(f"{self.record_id}: input is available after prediction time")

    @property
    def effective_prediction_time_s(self) -> float:
        return self.event_time if self.prediction_time_s is None else self.prediction_time_s

    @property
    def effective_input_available_s(self) -> float:
        return self.event_time if self.input_available_s is None else self.input_available_s


@dataclass(frozen=True)
class EventTimeWindow(VersionedRecord):
    """A half-open event-time interval ``[start_s, end_s)``."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        finite(self.start_s, "window start")
        finite(self.end_s, "window end")
        if self.start_s >= self.end_s:
            raise ValueError("event-time window must have positive width")

    def contains(self, event_time: float) -> bool:
        return self.start_s <= event_time < self.end_s


@dataclass(frozen=True)
class DriftMonitorConfig(VersionedRecord):
    """Configuration for an event-time monitor.

    The cutoff is mandatory: a reference set is a training/baseline artifact,
    and a current event at or before that cutoff is rejected rather than
    silently being treated as prospective data.
    """

    training_cutoff_s: float
    clock_basis: str
    window_size_s: float
    step_s: float
    alpha: float = 0.05
    minimum_reference_n: int = 2
    minimum_current_n: int = 2
    performance_threshold: float = 0.0
    evaluation_as_of_s: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "training_cutoff_s",
            "window_size_s",
            "step_s",
            "alpha",
            "performance_threshold",
        ):
            finite(getattr(self, name), name)
        if not self.clock_basis:
            raise ValueError("monitor configuration requires a clock basis")
        if self.window_size_s <= 0 or self.step_s <= 0:
            raise ValueError("window size and step must be positive")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if self.minimum_reference_n < 1 or self.minimum_current_n < 1:
            raise ValueError("minimum sample sizes must be positive")
        if self.performance_threshold < 0:
            raise ValueError("performance threshold cannot be negative")
        if self.evaluation_as_of_s is not None:
            finite(self.evaluation_as_of_s, "evaluation_as_of_s")
            if self.evaluation_as_of_s <= self.training_cutoff_s:
                raise ValueError("evaluation_as_of_s must follow the training cutoff")

    def window_for(self, start_s: float) -> EventTimeWindow:
        return EventTimeWindow(start_s, start_s + self.window_size_s)


@dataclass(frozen=True)
class DriftDiagnostic(VersionedRecord):
    """A distribution drift result for one feature and one event-time window."""

    drift_kind: DriftKind
    method: DriftMethod
    feature: str
    window_start_s: float
    window_end_s: float
    reference_n: int
    current_n: int
    statistic: float
    p_value: float
    threshold: float
    drift_detected: bool
    evaluated: bool
    status: str
    reference_time_max_s: float
    current_time_min_s: float
    interpretation: str = "distributional evidence only; no causal interpretation"
    domain: DriftDomain = DriftDomain.POPULATION
    evidence_class: EvidenceClass = EvidenceClass.EMPIRICAL_ESTIMATE
    causal_status: CausalStatus = CausalStatus.PREDICTIVE

    def __post_init__(self) -> None:
        object.__setattr__(self, "drift_kind", DriftKind(self.drift_kind))
        object.__setattr__(self, "method", DriftMethod(self.method))
        object.__setattr__(self, "domain", DriftDomain(self.domain))
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "causal_status", CausalStatus(self.causal_status))
        for name in (
            "window_start_s",
            "window_end_s",
            "statistic",
            "p_value",
            "threshold",
            "reference_time_max_s",
            "current_time_min_s",
        ):
            finite(getattr(self, name), name)
        if self.reference_n < 0 or self.current_n < 0:
            raise ValueError("diagnostic sample counts cannot be negative")
        if not 0 <= self.p_value <= 1:
            raise ValueError("p_value must be in [0, 1]")
        if self.threshold < 0:
            raise ValueError("diagnostic threshold cannot be negative")


@dataclass(frozen=True)
class ConceptDriftDiagnostic(VersionedRecord):
    """A non-causal, binned comparison of the conditional label relationship."""

    feature: str
    bins: tuple[float, ...]
    reference_counts: tuple[int, ...]
    current_counts: tuple[int, ...]
    cell_absolute_gaps: tuple[float, ...]
    weighted_absolute_gap: float
    threshold: float
    drift_detected: bool
    reference_n: int
    current_n: int
    as_of_s: float
    interpretation: str = (
        "conditional label relationship changed in observed bins; this is not a causal effect"
    )
    evidence_class: EvidenceClass = EvidenceClass.EMPIRICAL_ESTIMATE
    causal_status: CausalStatus = CausalStatus.PREDICTIVE

    def __post_init__(self) -> None:
        object.__setattr__(self, "bins", tuple(self.bins))
        object.__setattr__(self, "reference_counts", tuple(self.reference_counts))
        object.__setattr__(self, "current_counts", tuple(self.current_counts))
        object.__setattr__(self, "cell_absolute_gaps", tuple(self.cell_absolute_gaps))
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "causal_status", CausalStatus(self.causal_status))
        if len(self.bins) < 2 or any(not math.isfinite(value) for value in self.bins):
            raise ValueError("concept drift needs at least two finite bin edges")
        if tuple(sorted(self.bins)) != self.bins or len(set(self.bins)) != len(self.bins):
            raise ValueError("concept drift bins must be strictly increasing")
        expected = len(self.bins) - 1
        if not all(
            len(values) == expected
            for values in (self.reference_counts, self.current_counts, self.cell_absolute_gaps)
        ):
            raise ValueError("concept drift bin arrays have inconsistent lengths")
        finite(self.weighted_absolute_gap, "weighted_absolute_gap")
        finite(self.threshold, "threshold")
        finite(self.as_of_s, "as_of_s")
        if self.weighted_absolute_gap < 0 or self.threshold < 0:
            raise ValueError("concept drift gaps/threshold cannot be negative")


@dataclass(frozen=True)
class TaskValuePoint(VersionedRecord):
    """A chronological task value with its exact metric estimand context."""

    event_time: float
    value: float
    context: MetricContext
    source_artifact: str
    available_s: float | None = None

    def __post_init__(self) -> None:
        finite(self.event_time, "task value event_time")
        finite(self.value, "task value")
        if not self.source_artifact:
            raise ValueError("task value requires a source artifact")
        if self.available_s is not None:
            finite(self.available_s, "task value availability")


@dataclass(frozen=True)
class CapabilityValuePoint(VersionedRecord):
    """A chronological capability value under one declared capability context."""

    event_time: float
    value: float
    context: MetricContext
    source_artifact: str
    available_s: float | None = None

    def __post_init__(self) -> None:
        finite(self.event_time, "capability value event_time")
        finite(self.value, "capability value")
        if not self.source_artifact:
            raise ValueError("capability value requires a source artifact")
        if self.available_s is not None:
            finite(self.available_s, "capability value availability")


@dataclass(frozen=True)
class PerformanceDriftDiagnostic(VersionedRecord):
    """Chronological task-value drift; exact context equality is mandatory."""

    window_start_s: float
    window_end_s: float
    reference_n: int
    current_n: int
    reference_mean: float
    current_mean: float
    delta: float
    threshold: float
    drift_detected: bool
    evaluated: bool
    status: str
    context: MetricContext
    interpretation: str = "task-value change under the declared estimand; not a causal effect"
    evidence_class: EvidenceClass = EvidenceClass.EMPIRICAL_ESTIMATE
    causal_status: CausalStatus = CausalStatus.PREDICTIVE

    def __post_init__(self) -> None:
        for name in (
            "window_start_s",
            "window_end_s",
            "reference_mean",
            "current_mean",
            "delta",
            "threshold",
        ):
            finite(getattr(self, name), name)
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "causal_status", CausalStatus(self.causal_status))
        if self.reference_n < 0 or self.current_n < 0 or self.threshold < 0:
            raise ValueError("invalid performance diagnostic count or threshold")


def _ordered(events: Sequence[EventObservation]) -> tuple[EventObservation, ...]:
    return tuple(sorted(events, key=lambda event: (event.event_time, event.record_id)))


def _validate_event_partition(
    reference: Sequence[EventObservation],
    current: Sequence[EventObservation],
    config: DriftMonitorConfig,
) -> tuple[tuple[EventObservation, ...], tuple[EventObservation, ...]]:
    ref = _ordered(reference)
    cur = _ordered(current)
    all_events = ref + cur
    ids = [event.record_id for event in all_events]
    if len(ids) != len(set(ids)):
        raise MonitoringError("reference/current event IDs must be unique")
    if any(event.clock_basis != config.clock_basis for event in all_events):
        raise MonitoringError("all events must use the monitor clock basis")
    bad_reference = [
        event.record_id for event in ref if event.event_time > config.training_cutoff_s
    ]
    bad_current = [event.record_id for event in cur if event.event_time <= config.training_cutoff_s]
    if bad_reference:
        raise FutureDataError(
            "reference contains events after training cutoff: " + ", ".join(bad_reference)
        )
    if bad_current:
        raise FutureDataError(
            "current contains events at/before training cutoff: " + ", ".join(bad_current)
        )
    for event in all_events:
        if event.effective_input_available_s > event.effective_prediction_time_s:
            raise FutureDataError(f"{event.record_id}: input is unavailable at prediction time")
    return ref, cur


def _window_starts(
    events: Sequence[EventObservation], config: DriftMonitorConfig
) -> tuple[float, ...]:
    if not events:
        return ()
    first = events[0].event_time
    last = events[-1].event_time
    start = math.floor((first - config.training_cutoff_s) / config.step_s) * config.step_s
    start += config.training_cutoff_s
    starts: list[float] = []
    while start <= last:
        starts.append(start)
        start += config.step_s
    return tuple(starts)


def ks_statistic(reference: Sequence[float], current: Sequence[float]) -> float:
    """Return the two-sample KS distance, evaluating ties at unique values."""

    if not reference or not current:
        raise ValueError("KS requires two nonempty samples")
    left = sorted(float(value) for value in reference)
    right = sorted(float(value) for value in current)
    if any(not math.isfinite(value) for value in (*left, *right)):
        raise ValueError("KS samples must be finite")
    values = sorted(set(left + right))
    i = j = 0
    distance = 0.0
    for value in values:
        while i < len(left) and left[i] <= value:
            i += 1
        while j < len(right) and right[j] <= value:
            j += 1
        distance = max(distance, abs(i / len(left) - j / len(right)))
    return distance


def _ks_p_value(statistic: float, n_reference: int, n_current: int) -> float:
    """Asymptotic two-sided KS p-value; ties affect D, not a hidden tie-breaking order."""

    if statistic == 0:
        return 1.0
    effective = math.sqrt(n_reference * n_current / (n_reference + n_current))
    lam = (effective + 0.12 + 0.11 / effective) * statistic
    total = 0.0
    for index in range(1, 1000):
        term = 2.0 * ((-1.0) ** (index - 1)) * math.exp(-2.0 * (lam * index) ** 2)
        total += term
        if abs(term) < 1e-12:
            break
    return min(1.0, max(0.0, total))


class WindowedDriftMonitor:
    """Compute deterministic event-time covariate drift windows."""

    def __init__(self, reference: Sequence[EventObservation], config: DriftMonitorConfig):
        self.config = config
        self.reference = _ordered(reference)
        if not self.reference:
            raise ValueError("a drift monitor requires reference events")
        if any(event.clock_basis != config.clock_basis for event in self.reference):
            raise MonitoringError("reference events use an incompatible clock basis")
        if any(event.event_time > config.training_cutoff_s for event in self.reference):
            raise FutureDataError("reference event occurs after training cutoff")
        if len({event.record_id for event in self.reference}) != len(self.reference):
            raise MonitoringError("reference event IDs must be unique")

    def scan_covariate(
        self,
        current: Sequence[EventObservation],
        feature: str,
        *,
        domain: DriftDomain = DriftDomain.POPULATION,
    ) -> tuple[DriftDiagnostic, ...]:
        reference, current_events = _validate_event_partition(self.reference, current, self.config)
        if not all(feature in event.features for event in (*reference, *current_events)):
            raise KeyError(f"feature not present in every event: {feature}")
        reference_values = [event.features[feature] for event in reference]
        reference_max = max(event.event_time for event in reference)
        diagnostics: list[DriftDiagnostic] = []
        for start in _window_starts(current_events, self.config):
            window = self.config.window_for(start)
            events = [event for event in current_events if window.contains(event.event_time)]
            values = [event.features[feature] for event in events]
            evaluated = (
                len(reference_values) >= self.config.minimum_reference_n
                and len(values) >= self.config.minimum_current_n
            )
            statistic = ks_statistic(reference_values, values) if evaluated else 0.0
            p_value = (
                _ks_p_value(statistic, len(reference_values), len(values)) if evaluated else 1.0
            )
            diagnostics.append(
                DriftDiagnostic(
                    DriftKind.COVARIATE,
                    DriftMethod.TWO_SAMPLE_KS,
                    feature,
                    window.start_s,
                    window.end_s,
                    len(reference_values),
                    len(values),
                    statistic,
                    p_value,
                    self.config.alpha,
                    evaluated and p_value < self.config.alpha,
                    evaluated,
                    "evaluated" if evaluated else "insufficient_current_samples",
                    reference_max,
                    min(event.event_time for event in events) if events else window.start_s,
                    domain=domain,
                )
            )
        return tuple(diagnostics)

    def scan_covariates(
        self,
        current: Sequence[EventObservation],
        features: Sequence[str],
        *,
        domain: DriftDomain = DriftDomain.POPULATION,
    ) -> tuple[DriftDiagnostic, ...]:
        return tuple(
            result
            for feature in features
            for result in self.scan_covariate(current, feature, domain=domain)
        )

    def scan_concept(
        self,
        current: Sequence[EventObservation],
        feature: str,
        bins: Sequence[float],
        *,
        as_of_s: float | None = None,
        threshold: float | None = None,
    ) -> ConceptDriftDiagnostic:
        """Compare observed ``E[label | feature-bin]`` without causal language."""

        reference, current_events = _validate_event_partition(self.reference, current, self.config)
        as_of = self.config.evaluation_as_of_s if as_of_s is None else as_of_s
        if as_of is None:
            raise FutureDataError("concept drift requires an explicit label availability as_of_s")
        finite(as_of, "concept drift as_of_s")
        if as_of <= self.config.training_cutoff_s:
            raise FutureDataError("concept drift as_of_s must follow training cutoff")
        edges = tuple(float(value) for value in bins)
        if len(edges) < 2 or any(not math.isfinite(value) for value in edges):
            raise ValueError("concept drift needs finite bin edges")
        if tuple(sorted(edges)) != edges or len(set(edges)) != len(edges):
            raise ValueError("concept drift bins must be strictly increasing")

        def usable(event: EventObservation) -> bool:
            if feature not in event.features or event.label is None:
                return False
            if event.label_available_s is None:
                raise FutureDataError(f"{event.record_id}: label availability is undeclared")
            return event.label_available_s <= (
                self.config.training_cutoff_s if event in reference else as_of
            )

        ref = [event for event in reference if usable(event)]
        cur = [event for event in current_events if usable(event)]
        if not ref or not cur:
            raise MonitoringError("concept drift requires available labels in both partitions")
        ref_values: list[list[float]] = [[] for _ in range(len(edges) - 1)]
        cur_values: list[list[float]] = [[] for _ in range(len(edges) - 1)]

        def add(target: list[list[float]], event: EventObservation) -> None:
            index = bisect.bisect_right(edges, event.features[feature]) - 1
            if 0 <= index < len(target):
                target[index].append(event.label)  # type: ignore[arg-type]

        for event in ref:
            add(ref_values, event)
        for event in cur:
            add(cur_values, event)
        ref_counts = tuple(len(values) for values in ref_values)
        cur_counts = tuple(len(values) for values in cur_values)
        gaps = []
        weights = []
        for left, right in zip(ref_values, cur_values):
            if left and right:
                gap = abs(sum(left) / len(left) - sum(right) / len(right))
                gaps.append(gap)
                weights.append((len(left) + len(right), gap))
            else:
                gaps.append(0.0)
        total_weight = sum(weight for weight, _ in weights)
        weighted_gap = (
            sum(weight * gap for weight, gap in weights) / total_weight if total_weight else 0.0
        )
        selected_threshold = self.config.performance_threshold if threshold is None else threshold
        finite(selected_threshold, "concept drift threshold")
        return ConceptDriftDiagnostic(
            feature,
            edges,
            ref_counts,
            cur_counts,
            tuple(gaps),
            weighted_gap,
            selected_threshold,
            weighted_gap > selected_threshold,
            len(ref),
            len(cur),
            as_of,
        )


ValuePoint = TaskValuePoint | CapabilityValuePoint


def _validate_task_points(
    reference: Sequence[ValuePoint],
    current: Sequence[ValuePoint],
    context: MetricContext,
    config: DriftMonitorConfig,
    as_of_s: float,
) -> tuple[tuple[ValuePoint, ...], tuple[ValuePoint, ...]]:
    for point in (*reference, *current):
        EstimandFirewall.compare(context, point.context)
    ref = tuple(sorted(reference, key=lambda point: (point.event_time, point.source_artifact)))
    cur = tuple(sorted(current, key=lambda point: (point.event_time, point.source_artifact)))
    if not ref or not cur:
        raise ValueError("performance drift requires reference and current task values")
    if any(point.event_time > config.training_cutoff_s for point in ref):
        raise FutureDataError("reference task value occurs after training cutoff")
    if any(point.event_time <= config.training_cutoff_s for point in cur):
        raise FutureDataError("current task value occurs at/before training cutoff")
    for point in (*ref, *cur):
        if point.available_s is not None and point.available_s > as_of_s:
            raise FutureDataError(f"{point.source_artifact}: task value unavailable at as_of_s")
    return ref, cur


def track_task_value(
    points: Sequence[TaskValuePoint], context: MetricContext
) -> tuple[TaskValuePoint, ...]:
    """Validate an online task-value sequence without changing its order."""

    EstimandFirewall.require_metric(context, "task_value", context.estimand_name)
    ordered: list[TaskValuePoint] = []
    last_time = -math.inf
    for point in points:
        EstimandFirewall.compare(context, point.context)
        if point.event_time < last_time:
            raise ChronologyError("task-value stream is not event-time ordered")
        ordered.append(point)
        last_time = point.event_time
    return tuple(ordered)


def track_capability_value(
    points: Sequence[CapabilityValuePoint],
    context: MetricContext,
) -> tuple[CapabilityValuePoint, ...]:
    """Validate an online capability series under one exact context."""

    EstimandFirewall.require_metric(context, "capability", context.estimand_name)
    ordered: list[CapabilityValuePoint] = []
    last_time = -math.inf
    for point in points:
        EstimandFirewall.compare(context, point.context)
        if point.event_time < last_time:
            raise ChronologyError("capability-value stream is not event-time ordered")
        ordered.append(point)
        last_time = point.event_time
    return tuple(ordered)


def scan_performance_drift(
    reference: Sequence[ValuePoint],
    current: Sequence[ValuePoint],
    context: MetricContext,
    config: DriftMonitorConfig,
    *,
    as_of_s: float | None = None,
) -> tuple[PerformanceDriftDiagnostic, ...]:
    """Track chronological task value under one exact estimand context."""

    as_of = config.evaluation_as_of_s if as_of_s is None else as_of_s
    if as_of is None:
        raise FutureDataError("performance drift requires explicit as_of_s")
    finite(as_of, "performance drift as_of_s")
    ref, cur = _validate_task_points(reference, current, context, config, as_of)
    starts = _window_starts(
        tuple(
            EventObservation(
                point.source_artifact,
                point.source_artifact,
                point.event_time,
                {"task_value": point.value},
            )
            for point in cur
        ),
        config,
    )
    results: list[PerformanceDriftDiagnostic] = []
    reference_mean = sum(point.value for point in ref) / len(ref)
    for start in starts:
        window = config.window_for(start)
        values = [point for point in cur if window.contains(point.event_time)]
        evaluated = bool(values)
        current_mean = (
            sum(point.value for point in values) / len(values) if values else reference_mean
        )
        delta = current_mean - reference_mean
        results.append(
            PerformanceDriftDiagnostic(
                window.start_s,
                window.end_s,
                len(ref),
                len(values),
                reference_mean,
                current_mean,
                delta,
                config.performance_threshold,
                evaluated and abs(delta) > config.performance_threshold,
                evaluated,
                "evaluated" if evaluated else "empty_window",
                context,
            )
        )
    return tuple(results)


def scan_capability_drift(
    reference: Sequence[CapabilityValuePoint],
    current: Sequence[CapabilityValuePoint],
    context: MetricContext,
    config: DriftMonitorConfig,
    *,
    as_of_s: float | None = None,
) -> tuple[PerformanceDriftDiagnostic, ...]:
    """Apply the same chronological delta diagnostic to capability values."""

    EstimandFirewall.require_metric(context, "capability", context.estimand_name)
    return scan_performance_drift(reference, current, context, config, as_of_s=as_of_s)


__all__ = [
    "CapabilityValuePoint",
    "ChronologyError",
    "ConceptDriftDiagnostic",
    "DriftDiagnostic",
    "DriftDomain",
    "DriftKind",
    "DriftMethod",
    "DriftMonitorConfig",
    "EventObservation",
    "EventTimeWindow",
    "FutureDataError",
    "MonitoringError",
    "PerformanceDriftDiagnostic",
    "TaskValuePoint",
    "WindowedDriftMonitor",
    "ks_statistic",
    "scan_capability_drift",
    "scan_performance_drift",
    "track_capability_value",
    "track_task_value",
]

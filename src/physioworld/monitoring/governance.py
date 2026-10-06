"""Training cutoffs, adaptation policy, and leakage guards.

The guard translates event observations into the repository's ``SplitItem``
contract and delegates chronology, input availability, overlap, and label
checks to ``SplitCompiler``.  It therefore shares the same inclusive-boundary
semantics as ordinary dataset splitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from physioworld.core.contracts import VersionedRecord, finite
from physioworld.datasets.splits import (
    CompiledSplit,
    LeakageIssue,
    PreprocessingArtifact,
    SplitCompiler,
    SplitItem,
    SplitLeakageError,
    SplitSpec,
)

from .drift import EventObservation, FutureDataError


class AdaptationMode(str, Enum):
    FROZEN = "frozen"
    PERIODIC = "periodic"
    DRIFT_TRIGGERED = "drift_triggered"


@dataclass(frozen=True)
class AdaptationPolicy(VersionedRecord):
    """Explicit model training cutoff and permissible adaptation schedule."""

    policy_id: str
    training_cutoff_s: float
    clock_basis: str
    mode: AdaptationMode = AdaptationMode.FROZEN
    adaptation_interval_s: float | None = None
    embargo_s: float = 0.0
    labels_required: bool = True
    input_deadline_is_event_time: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", AdaptationMode(self.mode))
        if not self.policy_id or not self.clock_basis:
            raise ValueError("adaptation policy needs an ID and clock basis")
        finite(self.training_cutoff_s, "training_cutoff_s")
        finite(self.embargo_s, "embargo_s")
        if self.embargo_s < 0:
            raise ValueError("embargo cannot be negative")
        if self.mode == AdaptationMode.PERIODIC:
            if self.adaptation_interval_s is None:
                raise ValueError("periodic adaptation requires adaptation_interval_s")
            finite(self.adaptation_interval_s, "adaptation_interval_s")
            if self.adaptation_interval_s <= 0:
                raise ValueError("adaptation interval must be positive")
        elif self.adaptation_interval_s is not None:
            finite(self.adaptation_interval_s, "adaptation_interval_s")

    def validate_schedule(self, cutoffs: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in cutoffs)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("adaptation cutoffs must be finite")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("adaptation cutoffs must be strictly increasing")
        if any(value <= self.training_cutoff_s for value in values):
            raise FutureDataError("adaptation cannot occur at or before training cutoff")
        if self.mode == AdaptationMode.FROZEN and values:
            raise ValueError("frozen policy does not permit adaptation cutoffs")
        if self.mode == AdaptationMode.PERIODIC and self.adaptation_interval_s is not None:
            expected = self.training_cutoff_s + self.adaptation_interval_s
            for value in values:
                if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("periodic adaptation cutoff does not match declared interval")
                expected += self.adaptation_interval_s
        return values


def _split_item(event: EventObservation) -> SplitItem:
    return SplitItem(
        record_id=event.record_id,
        subject_id=event.subject_id,
        source_record_id=event.source_record_id,
        start_s=event.event_time,
        end_s=event.event_time,
        prediction_time_s=event.effective_prediction_time_s,
        input_available_s=event.effective_input_available_s,
        label_available_s=event.label_available_s,
        clock_basis=event.clock_basis,
        labels_used_as_features=event.label_used_as_feature,
    )


def compile_adaptation_split(
    events: Sequence[EventObservation],
    policy: AdaptationPolicy,
    *,
    assignments: Mapping[str, str] | None = None,
    preprocessing: Sequence[PreprocessingArtifact] = (),
) -> CompiledSplit:
    """Compile a leakage-safe prospective split at the policy cutoff."""

    if not events:
        raise ValueError("adaptation split requires events")
    items = tuple(_split_item(event) for event in events)
    if any(event.clock_basis != policy.clock_basis for event in events):
        raise SplitLeakageError(
            (
                LeakageIssue(
                    "incomparable_clocks",
                    (),
                    "event and adaptation policy clocks differ",
                ),
            )
        )
    spec = SplitSpec(
        "record",
        strategy="prospective",
        cutoff_s=policy.training_cutoff_s,
        embargo_s=policy.embargo_s,
    )
    compiled = SplitCompiler().compile(
        items, spec, assignments=assignments, preprocessing=preprocessing
    )
    if policy.labels_required:
        by_id = {event.record_id: event for event in events}
        training = tuple(
            record_id for record_id, part in compiled.assignments.items() if part == "train"
        )
        missing = tuple(record_id for record_id in training if by_id[record_id].label is None)
        missing_time = tuple(
            record_id for record_id in training if by_id[record_id].label_available_s is None
        )
        if missing:
            raise SplitLeakageError(
                (
                    LeakageIssue(
                        "missing_label",
                        missing,
                        "supervised adaptation requires an explicit training label",
                    ),
                )
            )
        if missing_time:
            raise SplitLeakageError(
                (
                    LeakageIssue(
                        "missing_label_availability",
                        missing_time,
                        "supervised adaptation requires training label availability timestamps",
                    ),
                )
            )
    return compiled


def eligible_fit_events(
    events: Sequence[EventObservation],
    policy: AdaptationPolicy,
    *,
    as_of_s: float | None = None,
) -> tuple[EventObservation, ...]:
    """Return only rows whose inputs and (when required) labels exist at ``as_of_s``.

    ``SplitCompiler.validate`` is intentionally invoked even though this helper
    returns only training rows.  The compiler remains the source of truth for
    future-input, overlap, label, and clock semantics; the helper suppresses
    only its expected empty-evaluation-partition warning.
    """

    cutoff = policy.training_cutoff_s if as_of_s is None else float(as_of_s)
    finite(cutoff, "fit as_of_s")
    if cutoff < policy.training_cutoff_s:
        raise FutureDataError("fit as_of_s cannot precede the training cutoff")
    if not events:
        return ()
    items = tuple(_split_item(event) for event in events)
    assignments = {
        event.record_id: "train" if event.event_time <= cutoff else "test" for event in events
    }
    spec = SplitSpec("record", strategy="prospective", cutoff_s=cutoff, embargo_s=policy.embargo_s)
    issues = SplitCompiler().validate(items, spec, assignments, preprocessing=())
    meaningful = tuple(issue for issue in issues if issue.code != "empty_partition")
    if meaningful:
        raise SplitLeakageError(meaningful)
    eligible = []
    for event in sorted(events, key=lambda item: (item.event_time, item.record_id)):
        if assignments[event.record_id] != "train":
            continue
        if policy.labels_required:
            if event.label is None or event.label_available_s is None:
                raise SplitLeakageError(
                    (
                        LeakageIssue(
                            "missing_label_availability",
                            (event.record_id,),
                            "label availability is required before fitting",
                        ),
                    )
                )
            if event.label_available_s > cutoff:
                raise FutureDataError(f"{event.record_id}: label is unavailable at fit cutoff")
        if event.effective_input_available_s > cutoff:
            raise FutureDataError(f"{event.record_id}: input is unavailable at fit cutoff")
        eligible.append(event)
    return tuple(eligible)


__all__ = [
    "AdaptationMode",
    "AdaptationPolicy",
    "compile_adaptation_split",
    "eligible_fit_events",
]

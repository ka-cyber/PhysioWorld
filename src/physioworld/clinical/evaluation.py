"""Protocol-locked independent-cohort evaluation of caller-supplied observations.

Hashes identify supplied artifacts; they do not attest to their contents or establish
clinical benefit. This module does not download or generate clinical data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Sequence

from physioworld.uncertainty import (
    BootstrapResult,
    CalibrationBin,
    brier_score,
    calibration_bins,
    cluster_bootstrap,
)

from .endpoints import binary_confusion
from .protocol import _require_explicit_text


@dataclass(frozen=True)
class BinaryUtility:
    true_positive: float
    true_negative: float
    false_positive: float
    false_negative: float
    alert_cost: float = 0.0

    def value(self, label: int, alert: int) -> float:
        if label not in (0, 1) or alert not in (0, 1):
            raise ValueError("binary endpoint and decision required")
        value = (
            (self.true_negative, self.false_positive),
            (self.false_negative, self.true_positive),
        )[label][alert]
        return value - self.alert_cost * alert

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) for v in asdict(self).values()) or self.alert_cost < 0:
            raise ValueError("finite utilities and nonnegative alert cost required")


@dataclass(frozen=True)
class AlarmWorkload:
    workload_id: str
    user: str
    endpoint: str
    horizon: float
    threshold: float
    utility: BinaryUtility
    resource_regime: str
    safety_constraints: tuple[str, ...]
    maximum_alerts_per_hour: float
    time_unit: str = "s"

    def __post_init__(self) -> None:
        for name in ("workload_id", "user", "endpoint", "resource_regime"):
            _require_explicit_text(getattr(self, name), name)
        for constraint in self.safety_constraints:
            _require_explicit_text(constraint, "safety_constraints")
        if (
            self.time_unit != "s"
            or not self.safety_constraints
            or not math.isfinite(self.horizon)
            or self.horizon <= 0
            or not math.isfinite(self.threshold)
            or not 0 <= self.threshold <= 1
            or not math.isfinite(self.maximum_alerts_per_hour)
            or self.maximum_alerts_per_hour < 0
        ):
            raise ValueError("explicit workload, timing, threshold, resources, and safety required")


@dataclass(frozen=True)
class FrozenArtifacts:
    preprocessing_hash: str
    representation_hash: str
    model_hash: str
    task_hash: str
    resource_hash: str

    def __post_init__(self) -> None:
        if any(
            len(s) != 64 or any(c not in "0123456789abcdef" for c in s)
            for s in asdict(self).values()
        ):
            raise ValueError("artifacts require lowercase SHA-256 content hashes")


@dataclass(frozen=True)
class LockedClinicalProtocol:
    protocol_id: str
    workload: AlarmWorkload
    artifacts: FrozenArtifacts
    population: str
    eligibility: str
    reference_standard: str
    missing_data_rule: str
    development_subject_ids: tuple[str, ...]
    development_cohort_ids: tuple[str, ...]
    locked_at: float
    fitting_complete_at: float
    prospective: bool = False
    primary_metric: str = "mean_utility"
    bootstrap_repetitions: int = 1000
    alpha: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "population",
            "eligibility",
            "reference_standard",
            "missing_data_rule",
        ):
            _require_explicit_text(getattr(self, name), name)
        for subject in self.development_subject_ids:
            _require_explicit_text(subject, "development_subject_ids")
        for cohort in self.development_cohort_ids:
            _require_explicit_text(cohort, "development_cohort_ids")
        if (
            not self.development_subject_ids
            or not self.development_cohort_ids
            or not math.isfinite(self.locked_at)
            or not math.isfinite(self.fitting_complete_at)
            or self.fitting_complete_at > self.locked_at
            or self.primary_metric not in ("mean_utility", "brier", "false_alarms_per_hour")
            or self.bootstrap_repetitions < 2
            or not 0 < self.alpha < 1
        ):
            raise ValueError("invalid locked protocol or fitting/lock chronology")

    @property
    def frozen_hash(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class CohortOrigin(str, Enum):
    SYNTHETIC = "SYNTHETIC"
    HUMAN_OBSERVATIONAL = "HUMAN_OBSERVATIONAL"
    CONTROLLED_HUMAN = "CONTROLLED_HUMAN"


@dataclass(frozen=True)
class EvaluationCohort:
    cohort_id: str
    origin: CohortOrigin
    source: str
    dataset_version: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("cohort_id", "source", "dataset_version"):
            _require_explicit_text(getattr(self, name), name)
        if not isinstance(self.origin, CohortOrigin) or not self.evidence_refs:
            raise ValueError("cohort origin and explicit evidence references are required")
        for reference in self.evidence_refs:
            _require_explicit_text(reference, "evidence_refs")


@dataclass(frozen=True)
class EndpointObservation:
    subject_id: str
    label: int
    probability: float
    prediction_time: float
    feature_available_at: float
    outcome_ascertained_at: float
    exposure_start: float
    exposure_end: float
    enrolled_at: float
    model_hash: str
    subgroup: str | None = None


@dataclass(frozen=True)
class CohortEvaluation:
    protocol_hash: str
    cohort: EvaluationCohort
    metrics: dict[str, float | None]
    primary_interval: BootstrapResult
    calibration: tuple[CalibrationBin, ...]
    subgroup_metrics: dict[str, dict[str, float | None]]
    status: str
    limitations: tuple[str, ...]
    clinical_validation_status: str = field(default="NOT_ESTABLISHED", init=False)

    @property
    def evaluation_scope(self) -> str:
        return (
            "SIMULATION_EVALUATION"
            if self.cohort.origin == CohortOrigin.SYNTHETIC
            else "EXTERNAL_PREDICTIVE_EVALUATION"
        )

    @property
    def external_predictive_evaluation_status(self) -> str:
        return "NOT_APPLICABLE" if self.cohort.origin == CohortOrigin.SYNTHETIC else "COMPLETED"


def average_precision(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    """Threshold-grouped average precision (step AUPRC), including score ties."""
    brier_score(labels, probabilities)  # common finite/binary validation
    positives = sum(labels)
    if not positives:
        return None
    pairs = sorted(zip(probabilities, labels), reverse=True)
    tp = total = 0
    result = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        added = 0
        while index < len(pairs) and pairs[index][0] == score:
            added += pairs[index][1]
            total += 1
            index += 1
        tp += added
        result += (tp / total) * added / positives
    return result


def _metrics(
    rows: Sequence[EndpointObservation], workload: AlarmWorkload
) -> dict[str, float | None]:
    labels = [r.label for r in rows]
    probabilities = [r.probability for r in rows]
    alerts = [int(p >= workload.threshold) for p in probabilities]
    counts = binary_confusion(labels, alerts)
    tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))
    hours = sum(r.exposure_end - r.exposure_start for r in rows) / 3600
    return {
        "prevalence": sum(labels) / len(rows),
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "average_precision": average_precision(labels, probabilities),
        "false_alarms_per_hour": fp / hours,
        "alerts_per_hour": sum(alerts) / hours,
        "mean_utility": sum(workload.utility.value(y, a) for y, a in zip(labels, alerts))
        / len(rows),
        "brier": brier_score(labels, probabilities),
        "subjects": float(len({r.subject_id for r in rows})),
        "windows": float(len(rows)),
    }


def _required_metric(metrics: dict[str, float | None], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        raise ValueError(f"metric {name!r} is undefined for this cohort")
    return value


def evaluate_independent_cohort(
    protocol: LockedClinicalProtocol,
    cohort: EvaluationCohort,
    observations: Sequence[EndpointObservation],
    *,
    expected_protocol_hash: str,
    applied_artifacts: FrozenArtifacts,
) -> CohortEvaluation:
    if expected_protocol_hash != protocol.frozen_hash or applied_artifacts != protocol.artifacts:
        raise ValueError("protocol/artifact lock mismatch")
    if not observations:
        raise ValueError("cohort observations are required")
    if cohort.cohort_id in protocol.development_cohort_ids:
        raise ValueError("independent cohort overlaps development cohorts")
    exposure_by_subject: dict[str, list[tuple[float, float]]] = {}
    for row in observations:
        _require_explicit_text(row.subject_id, "subject_id")
        if row.subject_id in protocol.development_subject_ids:
            raise ValueError("development/evaluation subject leakage")
        times = (
            row.prediction_time,
            row.feature_available_at,
            row.outcome_ascertained_at,
            row.exposure_start,
            row.exposure_end,
            row.enrolled_at,
        )
        if (
            any(not math.isfinite(t) for t in times)
            or row.feature_available_at > row.prediction_time
            or row.outcome_ascertained_at < row.prediction_time + protocol.workload.horizon
            or row.enrolled_at > row.prediction_time
            or row.exposure_end <= row.exposure_start
            or not row.exposure_start <= row.prediction_time < row.exposure_end
            or row.model_hash != protocol.artifacts.model_hash
        ):
            raise ValueError(
                "time leakage, incomplete endpoint follow-up, or prediction artifact mismatch"
            )
        if protocol.prospective and (
            row.enrolled_at < protocol.locked_at or row.prediction_time < protocol.locked_at
        ):
            raise ValueError("prospective subjects and predictions must follow protocol lock")
        exposure_by_subject.setdefault(row.subject_id, []).append(
            (row.exposure_start, row.exposure_end)
        )
    for windows in exposure_by_subject.values():
        ordered = sorted(windows)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("overlapping exposure windows would double-count alarm burden")
    metrics = _metrics(observations, protocol.workload)
    interval = cluster_bootstrap(
        observations,
        [r.subject_id for r in observations],
        lambda rows: _required_metric(_metrics(rows, protocol.workload), protocol.primary_metric),
        repetitions=protocol.bootstrap_repetitions,
        alpha=protocol.alpha,
        seed=protocol.seed,
    )
    subgroup_rows: dict[str, list[EndpointObservation]] = {}
    for row in observations:
        if row.subgroup is not None:
            subgroup_rows.setdefault(row.subgroup, []).append(row)
    subgroups = {group: _metrics(rows, protocol.workload) for group, rows in subgroup_rows.items()}
    status = (
        "SYNTHETIC_PROTOCOL_EVALUATION"
        if cohort.origin == CohortOrigin.SYNTHETIC
        else "EXTERNAL_PREDICTIVE_EVALUATION"
    )
    limitations = [
        "predictive/workload evaluation does not establish causal benefit or clinical safety",
        "hashes and timestamps are caller-provided provenance, not third-party attestation",
        "subgroups are descriptive; no multiplicity-adjusted generalization claim",
        "cluster bootstrap assumes independent subjects",
        "clinical validation is not established by this predictive endpoint estimate",
    ]
    if _required_metric(metrics, "alerts_per_hour") > protocol.workload.maximum_alerts_per_hour:
        limitations.append("declared maximum alert burden exceeded")
    return CohortEvaluation(
        protocol.frozen_hash,
        cohort,
        metrics,
        interval,
        calibration_bins([r.label for r in observations], [r.probability for r in observations]),
        subgroups,
        status,
        tuple(limitations),
    )


def temporal_performance(
    observations: Sequence[EndpointObservation],
    workload: AlarmWorkload,
    boundaries: Sequence[float],
) -> tuple[dict[str, float | None] | None, ...]:
    """Frozen-model temporal slices; no automatic retraining on future outcomes."""
    if (
        len(boundaries) < 2
        or any(not math.isfinite(v) for v in boundaries)
        or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("strictly increasing finite time boundaries required")
    if any(
        r.feature_available_at > r.prediction_time
        or r.outcome_ascertained_at < r.prediction_time + workload.horizon
        or r.exposure_end <= r.exposure_start
        for r in observations
    ):
        raise ValueError("temporal evaluation requires nonleaking complete endpoint windows")
    result = []
    for low, high in zip(boundaries, boundaries[1:]):
        rows = [r for r in observations if low <= r.prediction_time < high]
        result.append(_metrics(rows, workload) if rows else None)
    return tuple(result)

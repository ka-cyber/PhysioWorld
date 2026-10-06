"""Caller-supplied metric summaries, distinct from protocol-locked cohort evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ClinicalValidationResult:
    workload_id: str
    clinical_metric: Mapping[str, float]
    task_law_metric: Mapping[str, float]
    capability: Mapping[str, Any]
    pir: Mapping[str, Any]
    energy_j: float
    latency_s: float
    bandwidth_bytes: int
    compute_s: float
    failure_rate: float
    uncertainty: Mapping[str, Any]
    limitations: tuple[str, ...]
    evidence_status: str = "UNVALIDATED_CALLER_SUPPLIED_SUMMARY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clinical_validation_mode(
    workload_id: str,
    *,
    clinical_metric: Mapping[str, float],
    task_law_metric: Mapping[str, float],
    capability: Mapping[str, Any],
    pir: Mapping[str, Any],
    energy_j: float,
    latency_s: float,
    bandwidth_bytes: int,
    compute_s: float,
    failure_rate: float,
    uncertainty: Mapping[str, Any],
    limitations: tuple[str, ...] = (),
) -> ClinicalValidationResult:
    if not 0 <= failure_rate <= 1:
        raise ValueError("failure_rate must be in [0, 1]")
    return ClinicalValidationResult(
        workload_id,
        clinical_metric,
        task_law_metric,
        capability,
        pir,
        energy_j,
        latency_s,
        bandwidth_bytes,
        compute_s,
        failure_rate,
        uncertainty,
        limitations,
    )

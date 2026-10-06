"""Independent data/model comparisons with declared endpoint tolerances and provenance."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

from physioworld.uncertainty import BootstrapResult, cluster_bootstrap

from .numerics import Tolerance


@dataclass(frozen=True)
class ValidationComparison:
    source: str
    evidence_class: str
    model_artifact_hash: str
    software_version: str
    data_hash: str
    unit: str
    subject_ids: tuple[str, ...]
    residuals: tuple[float, ...]
    mean_error: float
    rmse: float
    maximum_error: float
    mean_absolute_error_interval: BootstrapResult
    tolerance: Tolerance
    passed: bool
    status: str


def compare_validation_data(
    predictions: Sequence[float],
    observations: Sequence[float],
    subject_ids: Sequence[str],
    *,
    training_subject_ids: Sequence[str],
    source: str,
    evidence_class: str,
    model_artifact_hash: str,
    software_version: str,
    unit: str,
    tolerance: Tolerance,
    repetitions: int = 1000,
    seed: int = 0,
) -> ValidationComparison:
    """Compute actual held-out residuals; pass requires every endpoint within tolerance.

    The MAE interval resamples subjects, with the fitted model held frozen. Evidence
    origin is a caller declaration; a synthetic comparison is never clinical evidence.
    """
    if (
        not observations
        or not len(predictions) == len(observations) == len(subject_ids)
        or any(not math.isfinite(v) for v in (*predictions, *observations))
        or any(not s for s in subject_ids)
        or not training_subject_ids
        or not source
        or not software_version
        or not unit
        or len(model_artifact_hash) != 64
        or any(c not in "0123456789abcdef" for c in model_artifact_hash)
        or evidence_class not in ("SIMULATION", "EMPIRICAL_ESTIMATE", "EXTERNAL_VALIDATION")
    ):
        raise ValueError("finite explicit held-out validation data and provenance required")
    if set(subject_ids) & set(training_subject_ids):
        raise ValueError("model validation training/evaluation subject leakage")
    residuals = tuple(p - y for p, y in zip(predictions, observations))
    absolute = tuple(abs(r) for r in residuals)
    interval = cluster_bootstrap(
        absolute,
        subject_ids,
        lambda rows: sum(rows) / len(rows),
        repetitions=repetitions,
        seed=seed,
    )
    passed = all(tolerance.accepts(p, y) for p, y in zip(predictions, observations))
    data_hash = hashlib.sha256(
        json.dumps(
            (tuple(predictions), tuple(observations), tuple(subject_ids), source, unit),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return ValidationComparison(
        source,
        evidence_class,
        model_artifact_hash,
        software_version,
        data_hash,
        unit,
        tuple(subject_ids),
        residuals,
        sum(residuals) / len(residuals),
        math.sqrt(sum(r * r for r in residuals) / len(residuals)),
        max(absolute),
        interval,
        tolerance,
        passed,
        "WITHIN_DECLARED_ENDPOINT_TOLERANCE" if passed else "FAILED_TOLERANCE",
    )

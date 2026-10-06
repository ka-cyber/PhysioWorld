"""Declared grid calibration and local excitation diagnostics.

No structural identifiability theorem or statistical confidence interval is
implied by a finite grid fit or a finite-difference sensitivity rank.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from .information import DataRequest
from .models import MODEL_VERSION, VARIABLE_UNITS, PatientParameters
from .scenario import PhysiologicalScenario


@dataclass(frozen=True)
class CalibrationArtifact:
    dataset_id: str
    input_sha256: str
    fitting_method: str
    model_version: str
    parameters: Mapping[str, float]
    objective: float
    uncertainty: Mapping[str, Any]
    operating_conditions: Mapping[str, Any]
    evidence_class: str
    data_origin: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _predictions(
    scenario: PhysiologicalScenario,
    times: Sequence[float],
    names: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    engine = replace(scenario, requests=DataRequest(names), observation_noise={}).create_engine()
    result = []
    for time in times:
        observation = engine.advance_to(time)
        result.append(tuple(observation.scalar(n) for n in names))
    return tuple(result)


def calibrate_patient(
    scenario: PhysiologicalScenario,
    observations: Sequence[Mapping[str, Any]],
    *,
    parameter_grid: Mapping[str, Sequence[float]],
    units: Mapping[str, str],
    scales: Mapping[str, float],
    dataset_id: str,
    data_origin: str,
    operating_conditions: Mapping[str, Any],
    objective_tolerance: float = 0.0,
) -> CalibrationArtifact:
    """Fit patient parameters to timestamped ``{time_s, values}`` rows by grid SSE.

    ``scales`` are declared residual normalizers in each variable's native unit,
    not fitted noise. The retained near-optimal grid region is a finite search
    uncertainty set, not a statistical confidence interval. No resampling.
    """
    if not observations or not parameter_grid or not dataset_id:
        raise ValueError("calibration needs data, parameter grid and dataset identity")
    if data_origin not in {"SYNTHETIC", "EMPIRICAL"}:
        raise ValueError("data_origin must be SYNTHETIC or EMPIRICAL")
    if not math.isfinite(objective_tolerance) or objective_tolerance < 0:
        raise ValueError("objective tolerance must be finite nonnegative")
    names = tuple(sorted(observations[0]["values"]))
    if any(set(row["values"]) != set(names) for row in observations):
        raise ValueError("calibration rows must have identical declared variables")
    for name in names:
        if name not in VARIABLE_UNITS or units.get(name) != VARIABLE_UNITS[name]:
            raise ValueError(f"calibration unit mismatch: {name}")
        if not math.isfinite(scales.get(name, 0)) or scales.get(name, 0) <= 0:
            raise ValueError(f"declare positive residual scale: {name}")
    times = tuple(float(row["time_s"]) for row in observations)
    if any(not math.isfinite(t) or not 0 <= t <= scenario.duration_s for t in times):
        raise ValueError("calibration times outside scenario")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("calibration times must strictly increase")
    if any(not math.isfinite(row["values"][n]) for row in observations for n in names):
        raise ValueError(
            "calibration values must be finite; missingness requires an explicit protocol"
        )
    params = tuple(sorted(parameter_grid))
    if not set(params) <= set(scenario.patient.ranges()):
        raise ValueError("unknown numerical patient parameter")
    fits = []
    for values in itertools.product(*(parameter_grid[n] for n in params)):
        candidate = dict(zip(params, values))
        patient_values: dict[str, Any] = asdict(scenario.patient)
        patient_values.update(candidate)
        patient_values["parameter_origin"] = "FITTED"
        patient = PatientParameters(**patient_values)
        predicted = _predictions(replace(scenario, patient=patient), times, names)
        score = sum(
            ((row["values"][n] - output[i]) / scales[n]) ** 2
            for row, output in zip(observations, predicted)
            for i, n in enumerate(names)
        )
        fits.append((score, candidate))
    if not fits:
        raise ValueError("empty parameter grid")
    fits.sort(key=lambda item: (item[0], tuple(item[1].values())))
    score, best = fits[0]
    region = [c for s, c in fits if s <= score + objective_tolerance]
    canonical_input = {
        "observations": list(observations),
        "units": dict(units),
        "scales": dict(scales),
        "scenario": scenario.to_dict(),
        "parameter_grid": dict(parameter_grid),
    }
    digest = hashlib.sha256(
        json.dumps(canonical_input, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return CalibrationArtifact(
        dataset_id,
        digest,
        "declared_grid_normalized_SSE",
        MODEL_VERSION,
        best,
        score,
        {
            "kind": "NEAR_OPTIMAL_FINITE_GRID_SET",
            "objective_tolerance": objective_tolerance,
            "parameter_sets": region,
            "is_confidence_interval": False,
        },
        dict(operating_conditions),
        "NUMERICAL_VERIFICATION" if data_origin == "SYNTHETIC" else "EMPIRICAL_ESTIMATE",
        data_origin,
    )


def _rank(matrix: Sequence[Sequence[float]], tolerance: float) -> int:
    rows = [list(row) for row in matrix]
    rank = 0
    for col in range(len(rows[0]) if rows else 0):
        pivot = max(range(rank, len(rows)), key=lambda i: abs(rows[i][col]), default=None)
        if pivot is None or abs(rows[pivot][col]) <= tolerance:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][col]
        rows[rank] = [v / divisor for v in rows[rank]]
        for i in range(rank + 1, len(rows)):
            factor = rows[i][col]
            rows[i] = [v - factor * w for v, w in zip(rows[i], rows[rank])]
        rank += 1
        if rank == len(rows):
            break
    return rank


def local_identifiability(
    scenario: PhysiologicalScenario,
    parameter_names: Sequence[str],
    *,
    variables: Sequence[str],
    times_s: Sequence[float],
    scales: Mapping[str, float],
    relative_step: float = 0.01,
    rank_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Central-difference, output-normalized local sensitivity/excitation report.

    Columns represent a unit relative parameter perturbation. Cosine similarity
    measures sensitivity collinearity, not posterior parameter correlation.
    """
    if not 0 < relative_step < 0.1 or not rank_tolerance > 0:
        raise ValueError("invalid difference step/rank tolerance")
    names = tuple(variables)
    if not names or not parameter_names or not times_s:
        raise ValueError("parameters, variables and times required")
    if not set(parameter_names) <= set(scenario.patient.ranges()):
        raise ValueError("unknown numerical patient parameter")
    if any(not math.isfinite(scales.get(n, 0)) or scales.get(n, 0) <= 0 for n in names):
        raise ValueError("positive output scales required")
    columns = []
    for parameter in parameter_names:
        value = getattr(scenario.patient, parameter)
        delta = relative_step * max(abs(value), 1e-3)
        plus = replace(scenario, patient=replace(scenario.patient, **{parameter: value + delta}))
        minus = replace(scenario, patient=replace(scenario.patient, **{parameter: value - delta}))
        a, b = _predictions(plus, times_s, names), _predictions(minus, times_s, names)
        columns.append(
            tuple(
                (x[i] - y[i]) / (2 * relative_step * scales[n])
                for x, y in zip(a, b)
                for i, n in enumerate(names)
            )
        )
    norms = [math.sqrt(sum(x * x for x in c)) for c in columns]
    cosines = [
        [
            sum(x * y for x, y in zip(a, b)) / (norms[i] * norms[j])
            if norms[i] * norms[j] > 0
            else None
            for j, b in enumerate(columns)
        ]
        for i, a in enumerate(columns)
    ]
    jacobian = tuple(zip(*columns))
    return {
        "evidence_class": "NUMERICAL_VERIFICATION",
        "method": "central_difference_local_excitation",
        "parameters": list(parameter_names),
        "variables": list(names),
        "times_s": list(times_s),
        "jacobian": [list(row) for row in jacobian],
        "sensitivity_norms": norms,
        "sensitivity_cosine_similarity": cosines,
        "local_numerical_rank": _rank(jacobian, rank_tolerance),
        "relative_step": relative_step,
        "rank_tolerance": rank_tolerance,
        "structural_identifiability": "NOT_ANALYZED",
        "practical_identifiability": "EXCITATION_DIAGNOSTIC_ONLY",
        "is_posterior_uncertainty": False,
    }

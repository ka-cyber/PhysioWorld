"""Physical-parameter provenance and measured-data calibration adapter."""

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ParameterStatus(str, Enum):
    MEASURED = "measured"
    FITTED = "fitted"
    ASSUMED = "assumed"
    BOUNDED = "bounded"
    SYNTHETIC = "synthetic"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PhysicalParameter:
    name: str
    value: float | None
    unit: str
    status: ParameterStatus
    source: str
    bounds: tuple[float, float] | None = None

    def __post_init__(self):
        if not self.name or not self.unit or not self.source:
            raise ValueError("parameter needs name, SI unit and provenance")
        if not isinstance(self.status, ParameterStatus):
            raise ValueError("parameter status must be an explicit ParameterStatus")
        if self.status == ParameterStatus.UNAVAILABLE:
            if self.value is not None:
                raise ValueError("unavailable parameter cannot carry a fabricated value")
        elif self.value is None or not math.isfinite(self.value):
            raise ValueError("available parameter must have a finite value")
        if self.bounds is not None:
            lo, hi = self.bounds
            if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
                raise ValueError("invalid parameter bounds")
            if self.value is not None and not lo <= self.value <= hi:
                raise ValueError("value outside declared parameter bounds")


@dataclass(frozen=True)
class CalibrationArtifact:
    parameters: tuple[PhysicalParameter, ...]
    input_checksum_sha256: str
    fitting_method: str
    version: str
    operating_conditions: Mapping[str, str]
    residual_rms: float
    local_jacobian_rank: int
    singular_values: tuple[float, ...]
    condition_number: float | None
    data_origin: str
    observation_count: int
    sensitivity_column_cosines: tuple[tuple[float, ...], ...]
    uncertainty_status: str = "no statistical confidence interval estimated"
    identifiability_scope: str = (
        "local numerical sensitivity only; not structural mechanism identification"
    )


def calibrate_resource(
    observations,
    predict,
    initial_parameters,
    bounds,
    *,
    units,
    source,
    operating_conditions,
    data_origin,
    version="1",
):
    """Bounded least squares against actual supplied voltage observations.

    observations: JSON-serializable records with `voltage_v` and explicit
    experimental inputs consumed by predict(parameters, record). No synthetic
    fallback or uncertainty interpretation is inferred. SciPy is optional.
    """
    if data_origin not in ("hardware_measurement", "synthetic"):
        raise ValueError("declare hardware_measurement or synthetic input origin")
    if not observations or not source or not operating_conditions:
        raise ValueError("calibration requires data, source, and operating conditions")
    names = tuple(initial_parameters)
    if not names or set(names) != set(bounds) or set(names) != set(units):
        raise ValueError("calibration parameter declarations must match")
    for name in names:
        PhysicalParameter(
            name,
            initial_parameters[name],
            units[name],
            ParameterStatus.ASSUMED,
            source,
            bounds[name],
        )
    if not all(math.isfinite(record["voltage_v"]) for record in observations):
        raise ValueError("observed voltages must be finite")
    import numpy as np
    from scipy.optimize import least_squares

    def residual(vector):
        parameters = dict(zip(names, vector))
        values = [predict(parameters, record) - record["voltage_v"] for record in observations]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("calibration predictor returned nonfinite voltage")
        return values

    fit = least_squares(
        residual,
        [initial_parameters[k] for k in names],
        bounds=([bounds[k][0] for k in names], [bounds[k][1] for k in names]),
    )
    if not fit.success:
        raise ArithmeticError(fit.message)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(fit.jac))
    condition = float(singular[0] / singular[-1]) if rank == len(names) else None
    if condition is not None and not math.isfinite(condition):
        condition = None
    norms = np.linalg.norm(fit.jac, axis=0)
    denominator = np.outer(norms, norms)
    cosines = np.divide(
        fit.jac.T @ fit.jac,
        denominator,
        out=np.zeros((len(names), len(names))),
        where=denominator > 0,
    )
    checksum = hashlib.sha256(
        json.dumps(observations, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()
    status = (
        ParameterStatus.FITTED
        if data_origin == "hardware_measurement"
        else ParameterStatus.SYNTHETIC
    )
    parameters = tuple(
        PhysicalParameter(k, float(v), units[k], status, source, bounds[k])
        for k, v in zip(names, fit.x)
    )
    return CalibrationArtifact(
        parameters,
        checksum,
        "bounded_least_squares",
        version,
        operating_conditions,
        float(np.sqrt(np.mean(fit.fun**2))),
        rank,
        tuple(map(float, singular)),
        condition,
        data_origin,
        len(observations),
        tuple(tuple(map(float, row)) for row in cosines),
    )


def canonical_fixture_status():
    """Availability, not a fitted reconstruction of incomplete quoted results."""
    from .capacitor import canonical_current_limit

    return {
        "declared_no_leak_esr_baseline": {
            "status": "available",
            "evidence": "MODEL_DERIVED",
            "parameters": {
                "v0_v": 2.65,
                "vmin_v": 1.85,
                "capacitance_f": 0.2,
                "esr_ohm": 8.0,
                "duration_s": 10.0,
                "leakage_siemens": 0.0,
            },
            "current_a": canonical_current_limit(2.65, 1.85, 0.2, 8.0, 10.0),
            "scope": "prompt's P0 label includes ESR; strictly ideal P0 has zero ESR",
        },
        "p1_quoted_current": {
            "status": "unavailable",
            "missing": ["leakage parameterization", "original circuit provenance"],
        },
        "p2_nominal_and_128_corners": {
            "status": "unavailable",
            "missing": [
                "complete branch/capacitance/leakage parameters",
                "seven uncertain coordinates",
                "initial internal states",
                "original fixture provenance",
            ],
        },
        "tailored_uniform_margin_energy": {
            "status": "unavailable",
            "missing": [
                "original physical parameterization",
                "ready-state design objective and constraints",
                "initial-state architecture",
                "original fixture provenance",
            ],
        },
    }

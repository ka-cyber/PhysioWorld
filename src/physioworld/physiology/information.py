"""Medical information state and guarded information-access contracts.

PIV is implemented canonically by :mod:`physioworld.capability.values`.  The
names remain available here for compatibility with the physiology namespace,
but the canonical estimator requires an explicit paired information-
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from physioworld.capability.values import PIVDesign, PIVEstimate, estimate_piv

__all__ = [
    "MedicalInformationState",
    "PIVDesign",
    "PIVEstimate",
    "estimate_piv",
    "validate_piv_against_utility",
    "RequestAccessError",
    "DataRequest",
    "ASSESSMENT_VARIABLES",
    "Assessment",
    "Observation",
]


@dataclass(frozen=True)
class MedicalInformationState:
    sqi: Mapping[str, float]
    modality_availability: Mapping[str, bool]
    physiological_importance: Mapping[str, float]
    event_urgency: float
    uncertainty: float

    def vector(self) -> tuple[float, ...]:
        modalities = tuple(sorted(self.modality_availability))
        return (
            tuple(self.sqi.get(m, 0.0) for m in modalities)
            + tuple(float(self.modality_availability[m]) for m in modalities)
            + tuple(self.physiological_importance.get(m, 0.0) for m in modalities)
            + (self.event_urgency, self.uncertainty)
        )


def validate_piv_against_utility(
    estimates: Sequence[PIVEstimate], observed_utility_gains: Mapping[str, float]
) -> float:
    """Return mean absolute estimation error against downstream utility gains."""
    comparable = [
        (estimate.incremental_value, observed_utility_gains[estimate.modality])
        for estimate in estimates
        if estimate.modality in observed_utility_gains
    ]
    if not comparable:
        raise ValueError("no PIV estimates have observed downstream utility")
    return sum(abs(a - b) for a, b in comparable) / len(comparable)


class RequestAccessError(ValueError):
    """An observation attempted to access an undeclared variable/assessment."""


@dataclass(frozen=True)
class DataRequest:
    """Only named variables are exposed, with native units (no implicit conversion).

    Compartment variables use ``compartment.<state_field>`` and caller-managed
    resource variables use ``resource.<key>``. Assessments explicitly declare
    their complete output bundle. This is an information boundary, not a sandbox
    for hostile Python code with direct access to an engine checkpoint.
    """

    variables: tuple[str, ...] = (
        "heart_rate_bpm",
        "mean_pressure_mmhg",
        "oxygen_saturation",
        "respiratory_rate_bpm",
        "lactate_mmol_l",
    )
    assessments: tuple[str, ...] = ()
    events: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "assessments", tuple(self.assessments))
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("duplicate requested variables")
        if len(set(self.assessments)) != len(self.assessments):
            raise ValueError("duplicate requested assessments")


ASSESSMENT_VARIABLES = {
    "blood_gas": ("arterial_po2_mmhg", "arterial_pco2_mmhg", "oxygen_saturation"),
    "blood_count": ("hemoglobin_g_dl", "red_cell_volume_l", "blood_volume_l"),
    "metabolic": (
        "oxygen_delivery_ml_min",
        "oxygen_consumption_ml_min",
        "oxygen_demand_ml_min",
        "lactate_mmol_l",
        "sodium_mmol_l",
    ),
    "renal": ("renal_filtration_ml_min", "urine_flow_ml_min", "urine_l", "adh"),
}


@dataclass(frozen=True)
class Assessment:
    name: str
    time_s: float
    values: Mapping[str, float]
    units: Mapping[str, str]
    evidence_class: str = "SIMULATION"

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "time_s": self.time_s,
            "values": dict(self.values),
            "units": dict(self.units),
            "evidence_class": self.evidence_class,
        }


@dataclass(frozen=True)
class Observation:
    time_s: float
    subject_id: str
    values: Mapping[str, float | None]
    units: Mapping[str, str]
    assessments: tuple[Assessment, ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    evidence_class: str = "SIMULATION"

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "units", MappingProxyType(dict(self.units)))
        object.__setattr__(self, "events", tuple(MappingProxyType(dict(e)) for e in self.events))

    def __getitem__(self, name: str) -> float | None:
        if name not in self.values:
            raise RequestAccessError(f"undeclared observation variable: {name}")
        return self.values[name]

    def scalar(self, name: str) -> float:
        """Return an available finite scalar or fail explicitly on missing data."""
        value = self[name]
        if value is None:
            raise ValueError(f"observation variable is unavailable: {name}")
        if not math.isfinite(value):
            raise ValueError(f"observation variable is non-finite: {name}")
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "subject_id": self.subject_id,
            "values": dict(self.values),
            "units": dict(self.units),
            "assessments": [a.to_dict() for a in self.assessments],
            "events": [dict(e) for e in self.events],
            "evidence_class": self.evidence_class,
        }

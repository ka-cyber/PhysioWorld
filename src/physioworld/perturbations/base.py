"""Perturbation contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from physioworld.core import Provenance


@dataclass(frozen=True)
class PerturbationResult:
    value: Any
    severity: float
    ground_truth: Any
    provenance: Provenance
    parameters: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class Perturbation:
    """Timed signal intervention; the active interval is [onset_s, end_s)."""

    onset_s: float = 0.0
    end_s: float | None = None
    cause: str = "controlled synthetic sensor degradation"
    evidence_basis: str = "MODEL_ASSUMPTION"
    safety_status: str = "SIMULATION_ONLY"
    name = "perturbation"

    def __post_init__(self) -> None:
        self.validate_timing()

    def validate_timing(self) -> None:
        if not math.isfinite(self.onset_s) or self.onset_s < 0:
            raise ValueError("perturbation onset must be finite nonnegative seconds")
        if self.end_s is not None and (not math.isfinite(self.end_s) or self.end_s < self.onset_s):
            raise ValueError("perturbation end precedes onset")
        if not self.cause or not self.evidence_basis or not self.safety_status:
            raise ValueError("perturbation metadata required")

    def active(self, time_s: float) -> bool:
        return self.onset_s <= time_s and (self.end_s is None or time_s < self.end_s)

    def timing_mask(self, timestamps: Any) -> tuple[bool, ...]:
        self.validate_timing()
        timestamps = tuple(timestamps)
        if any(not math.isfinite(t) for t in timestamps):
            raise ValueError("perturbation timestamps must be finite seconds")
        return tuple(self.active(t) for t in timestamps)

    def provenance(self, **parameters: Any) -> Provenance:
        return Provenance(
            "physioworld",
            self.name,
            getattr(self, "seed", None),
            notes={
                **parameters,
                "cause": self.cause,
                "timing": {
                    "onset_s": self.onset_s,
                    "end_s": self.end_s,
                    "interval": "half_open_seconds",
                },
                "mechanism": self.name,
                "parameterization": parameters,
                "expected_state_effect": getattr(
                    self,
                    "expected_state_effect",
                    "signal/availability changes only; latent physiology unchanged",
                ),
                "evidence_basis": self.evidence_basis,
                "evidence_class": "SIMULATION",
                "safety_status": self.safety_status,
            },
        )

    def apply(self, stream: Any) -> Any:
        raise NotImplementedError

    def result(self, stream: Any) -> PerturbationResult:
        value = self.apply(stream)
        return PerturbationResult(
            value=value,
            severity=1.0,
            ground_truth=stream,
            provenance=self.provenance(**dict(self.__dict__)),
            parameters=dict(self.__dict__),
        )

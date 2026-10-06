"""Timed mechanistic challenges applied to the coupled world, not signal offsets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from physioworld.physiology.engine import PhysiologicalAction, PhysiologyEngine


@dataclass(frozen=True)
class PhysiologicalChallenge:
    kind: str
    onset_s: float
    parameters: Mapping[str, Any]
    duration_s: float | None = None
    recovery_parameters: Mapping[str, Any] = field(default_factory=dict)
    cause: str = "controlled synthetic physiological challenge"
    evidence_basis: str = "MODEL_ASSUMPTION"
    safety_status: str = "SIMULATION_ONLY"

    def actions(self) -> tuple[PhysiologicalAction, ...]:
        first = PhysiologicalAction(
            self.onset_s,
            self.kind,
            self.parameters,
            self.cause,
            self.evidence_basis,
            self.safety_status,
        )
        if self.duration_s is None:
            return (first,)
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("challenge duration must be finite nonnegative")
        if not self.recovery_parameters:
            raise ValueError("finite challenge requires explicit recovery_parameters")
        return (
            first,
            PhysiologicalAction(
                self.onset_s + self.duration_s,
                self.kind,
                self.recovery_parameters,
                f"recovery from {self.cause}",
                self.evidence_basis,
                self.safety_status,
            ),
        )

    def schedule(self, world: PhysiologyEngine) -> tuple[int, ...]:
        actions = self.actions()
        for a in actions:
            if a.time_s < world.time_s:
                raise ValueError("challenge action precedes current simulation time")
            world._validate_action(a)
        return tuple(world.schedule(a) for a in actions)

    def metadata(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "timing": {"onset_s": self.onset_s, "duration_s": self.duration_s},
            "mechanism": PhysiologyEngine._mechanism(self.kind),
            "parameterization": dict(self.parameters),
            "expected_state_effect": PhysiologyEngine._mechanism(self.kind),
            "evidence_basis": self.evidence_basis,
            "evidence_class": "SIMULATION",
            "safety_status": self.safety_status,
        }


def orthostatic_transition(
    onset_s: float, *, duration_s: float | None = None, fraction: float = 1
) -> PhysiologicalChallenge:
    return PhysiologicalChallenge(
        "orthostasis", onset_s, {"orthostasis": fraction}, duration_s, {"orthostasis": 0}
    )


def exercise_challenge(
    onset_s: float, *, mets: float = 3, duration_s: float | None = None
) -> PhysiologicalChallenge:
    return PhysiologicalChallenge(
        "exercise", onset_s, {"exercise_mets": mets}, duration_s, {"exercise_mets": 1}
    )


def hemorrhage_challenge(
    onset_s: float, *, rate_ml_min: float = 100, duration_s: float | None = None
) -> PhysiologicalChallenge:
    return PhysiologicalChallenge(
        "hemorrhage",
        onset_s,
        {"hemorrhage_ml_min": rate_ml_min},
        duration_s,
        {"hemorrhage_ml_min": 0},
    )


def lbnp_challenge(
    onset_s: float, *, pressure_mmhg: float = 30, duration_s: float | None = None
) -> PhysiologicalChallenge:
    return PhysiologicalChallenge(
        "lbnp", onset_s, {"lbnp_mmhg": pressure_mmhg}, duration_s, {"lbnp_mmhg": 0}
    )

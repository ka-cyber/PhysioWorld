"""Executable identification boundary; empirical diagnostics never prove exchangeability."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class AssumptionStatus(str, Enum):
    VERIFIED = "VERIFIED"
    ASSUMED = "ASSUMED"
    TESTED = "TESTED"
    PLAUSIBLE = "PLAUSIBLE"
    UNCHECKABLE = "UNCHECKABLE"
    VIOLATED = "VIOLATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


MANDATORY_ASSUMPTIONS = (
    "consistency",
    "sequential_exchangeability",
    "positivity",
    "temporality",
    "action_support",
    "safety_constraints",
    "intervention_variation",
    "outcome_measurement",
    "no_interference",
    "censoring",
)


@dataclass(frozen=True)
class CausalAssumption:
    name: str
    status: AssumptionStatus
    justification: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, AssumptionStatus) or not self.justification.strip():
            raise ValueError("assumptions require a typed status and justification")
        if self.status == AssumptionStatus.VERIFIED and not self.evidence_refs:
            raise ValueError("verified assumptions require evidence references")


@dataclass(frozen=True)
class TargetTrial:
    time_zero: str
    eligibility: str
    actions: tuple[str, ...]
    treatment_strategies: tuple[str, ...]
    outcome: str
    follow_up: float
    censoring: str
    time_unit: str = "s"

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (
                self.time_zero,
                self.eligibility,
                self.outcome,
                self.censoring,
                self.time_unit,
            )
        ):
            raise ValueError("target trial fields must be explicit")
        if (
            not self.actions
            or len(set(self.actions)) != len(self.actions)
            or not self.treatment_strategies
            or not math.isfinite(self.follow_up)
            or self.follow_up <= 0
        ):
            raise ValueError("invalid actions, strategies, or follow-up")


@dataclass(frozen=True)
class SupportRecord:
    context: str
    action: str
    probability: float
    observations: int
    source: str

    def __post_init__(self) -> None:
        if (
            not self.context
            or not self.source
            or not math.isfinite(self.probability)
            or not 0 <= self.probability <= 1
            or self.observations < 0
        ):
            raise ValueError("invalid support record")


class IdentificationError(ValueError):
    """The requested causal estimand is not supported by its identification record."""


@dataclass(frozen=True)
class IdentificationRecord:
    estimand: str
    treatment: str
    trial: TargetTrial
    assumptions: tuple[CausalAssumption, ...]
    observed_support: tuple[SupportRecord, ...]
    estimator: str
    data_source: str
    model_id: str
    uncertainty: str
    sensitivity: str
    limitations: tuple[str, ...]
    resource_regime: str = "not_declared"

    @property
    def blockers(self) -> tuple[str, ...]:
        issues = []
        if any(
            not value.strip()
            for value in (
                self.estimand,
                self.treatment,
                self.estimator,
                self.data_source,
                self.model_id,
                self.uncertainty,
                self.sensitivity,
            )
        ):
            issues.append("identification metadata is incomplete")
        names = [a.name for a in self.assumptions]
        if len(names) != len(set(names)):
            issues.append("duplicate assumptions")
        by_name = {a.name: a for a in self.assumptions}
        for name in MANDATORY_ASSUMPTIONS:
            assumption = by_name.get(name)
            if assumption is None:
                issues.append(f"missing mandatory assumption: {name}")
            elif assumption.status not in (AssumptionStatus.VERIFIED, AssumptionStatus.ASSUMED):
                issues.append(f"{name}: {assumption.status.value}")
        for assumption in self.assumptions:
            if assumption.status in (AssumptionStatus.VIOLATED, AssumptionStatus.UNCHECKABLE):
                issues.append(f"{assumption.name}: {assumption.status.value}")
        for action in self.trial.actions:
            if not any(
                s.action == action and s.probability > 0 and s.observations > 0
                for s in self.observed_support
            ):
                issues.append(f"no declared observed support for action: {action}")
        return tuple(issues)

    @property
    def status(self) -> str:
        if self.blockers:
            return "NOT_IDENTIFIED"
        if any(a.status == AssumptionStatus.ASSUMED for a in self.assumptions):
            return "IDENTIFIED_UNDER_EXPLICIT_ASSUMPTIONS"
        return "MODEL_IDENTIFIED"

    def require_identified(self, estimator: str | None = None) -> None:
        if self.blockers:
            raise IdentificationError("; ".join(self.blockers))
        if estimator is not None and self.estimator != estimator:
            raise IdentificationError(f"record estimator {self.estimator!r} != {estimator!r}")

    def explain(self) -> dict[str, object]:
        return {
            "estimand": self.estimand,
            "data_source": self.data_source,
            "model_id": self.model_id,
            "target_trial": self.trial,
            "assumptions": self.assumptions,
            "status": self.status,
            "blockers": self.blockers,
            "uncertainty": self.uncertainty,
            "sensitivity": self.sensitivity,
            "limitations": self.limitations,
            "resource_regime": self.resource_regime,
        }

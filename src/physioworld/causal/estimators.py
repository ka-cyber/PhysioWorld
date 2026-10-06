"""Legacy numerical diagnostics. These floats carry no causal identification claim.

Use IdentificationRecord and evaluate_policy for guarded interventional estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class InterventionKernel:
    """Declared observational/interventional transition semantics."""

    action_space: tuple[object, ...]
    transition: Callable[[object, object], object]
    semantics: str = "do(action)"

    def intervene(self, state: object, action: object) -> object:
        if action not in self.action_space:
            raise ValueError("action is outside the declared intervention space")
        return self.transition(state, action)


@dataclass(frozen=True)
class CausalEstimand:
    estimand_id: str
    treatment: str
    outcome: str
    intervention: str
    time_horizon: float
    assumptions: tuple[str, ...]
    target_trial: str | None = None


@dataclass(frozen=True)
class PositivityDiagnostic:
    minimum_propensity: float
    maximum_propensity: float
    violations: int
    threshold: float
    valid: bool


def positivity_diagnostic(
    propensities: Sequence[float], threshold: float = 0.01
) -> PositivityDiagnostic:
    if not 0 < threshold < 0.5:
        raise ValueError("threshold must be in (0, 0.5)")
    if not propensities or any(not math.isfinite(p) or p <= 0 or p >= 1 for p in propensities):
        invalid = sum(not math.isfinite(p) or p <= 0 or p >= 1 for p in propensities)
        return PositivityDiagnostic(
            min(propensities, default=0.0),
            max(propensities, default=0.0),
            invalid,
            threshold,
            False,
        )
    violations = sum(min(p, 1 - p) < threshold for p in propensities)
    return PositivityDiagnostic(
        min(propensities), max(propensities), violations, threshold, violations == 0
    )


def longitudinal_g_formula(
    baseline_states: Sequence[object],
    actions: Sequence[object],
    transition: Callable[[object, object], object],
    outcome: Callable[[object], float],
) -> float:
    """Legacy one-transition model average, not a longitudinal identified g-formula."""
    if len(baseline_states) != len(actions) or not baseline_states:
        raise ValueError("states and actions must have equal non-zero length")
    return sum(
        outcome(transition(state, action)) for state, action in zip(baseline_states, actions)
    ) / len(baseline_states)


def inverse_probability_weighted_value(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    observed_actions: Sequence[int],
    policy_action: int,
) -> float:
    """Self-normalized diagnostic; p is P(observed action | history), including action 0."""
    if not (len(outcomes) == len(propensities) == len(observed_actions)) or not outcomes:
        raise ValueError("arrays must have equal non-zero length")
    if any(not math.isfinite(p) or not 0 < p <= 1 for p in propensities):
        raise ValueError("observed-action propensities must be in (0, 1]")
    if any(not math.isfinite(y) for y in outcomes):
        raise ValueError("outcomes must be finite")
    weights = [
        1.0 / p if action == policy_action else 0.0
        for p, action in zip(propensities, observed_actions)
    ]
    denominator = sum(weights)
    if denominator == 0:
        raise ValueError("policy has no supported observations")
    return sum(w * y for w, y in zip(weights, outcomes)) / denominator


def doubly_robust_value(
    outcomes: Sequence[float],
    propensities: Sequence[float],
    observed_actions: Sequence[int],
    policy_action: int,
    outcome_model: Sequence[float],
) -> float:
    """One-action augmented numeric diagnostic, with observed-action propensities."""
    if (
        not (len(outcomes) == len(propensities) == len(observed_actions) == len(outcome_model))
        or not outcomes
    ):
        raise ValueError("arrays must have equal non-zero length")
    if any(not math.isfinite(p) or not 0 < p <= 1 for p in propensities):
        raise ValueError("observed-action propensities must be in (0, 1]")
    if any(not math.isfinite(v) for v in (*outcomes, *outcome_model)):
        raise ValueError("outcomes and predictions must be finite")
    estimates = []
    for y, p, action, model in zip(outcomes, propensities, observed_actions, outcome_model):
        correction = (y - model) / p if action == policy_action else 0.0
        estimates.append(model + correction)
    return sum(estimates) / len(estimates)

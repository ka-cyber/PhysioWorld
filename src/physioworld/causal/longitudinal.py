"""Finite logged-trajectory g-computation, per-decision IPW, and sequential DR.

Propensities are full distributions P(A_t | H_t), not an ambiguous binary scalar.
Only complete, uncensored, equal-horizon trajectories are accepted. Nuisance models
must be external or fitted on disjoint subjects. No weight clipping is performed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .identification import IdentificationError, IdentificationRecord


@dataclass(frozen=True)
class TimedCovariate:
    name: str
    value: float
    available_at: float


@dataclass(frozen=True)
class HistoryView:
    index: int
    state: str
    covariates: tuple[TimedCovariate, ...]
    past_actions: tuple[str, ...]
    past_rewards: tuple[float, ...]
    decision_time: float


@dataclass(frozen=True)
class LoggedStep:
    state: str
    covariates: tuple[TimedCovariate, ...]
    action: str
    behavior_probabilities: tuple[tuple[str, float], ...]
    reward: float
    decision_time: float
    outcome_time: float
    safe_actions: tuple[str, ...]


@dataclass(frozen=True)
class LoggedTrajectory:
    subject_id: str
    time_zero: float
    eligible_at: float
    steps: tuple[LoggedStep, ...]
    eligible: bool = True
    censored: bool = False

    def history(self, index: int) -> HistoryView:
        step = self.steps[index]
        return HistoryView(
            index,
            step.state,
            step.covariates,
            tuple(s.action for s in self.steps[:index]),
            tuple(s.reward for s in self.steps[:index]),
            step.decision_time,
        )


@dataclass(frozen=True)
class EvaluationPolicy:
    strategy_id: str
    probabilities: Callable[[HistoryView], Mapping[str, float]]


@dataclass(frozen=True)
class OutcomeNuisance:
    q_value: Callable[[HistoryView, str], float]
    source: str
    fit_subject_ids: tuple[str, ...] = ()
    external: bool = False


@dataclass(frozen=True)
class PolicyValueEstimate:
    value: float
    method: str
    identification: IdentificationRecord
    policy_id: str
    subject_ids: tuple[str, ...]
    subject_contributions: tuple[float, ...]
    effective_sample_size: float | None
    minimum_behavior_probability: float
    maximum_cumulative_weight: float
    limitations: tuple[str, ...]
    discount: float

    @property
    def claim_status(self) -> str:
        return self.identification.status


def _distribution(
    entries: Sequence[tuple[str, float]], actions: tuple[str, ...]
) -> dict[str, float]:
    probabilities = dict(entries)
    if len(probabilities) != len(entries) or set(probabilities) != set(actions):
        raise ValueError("probability distribution must declare every action exactly once")
    if any(
        not math.isfinite(p) or p < 0 or p > 1 for p in probabilities.values()
    ) or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-10):
        raise ValueError("action probabilities must be finite, nonnegative, and normalized")
    return probabilities


def _policy(
    policy: EvaluationPolicy,
    history: HistoryView,
    actions: tuple[str, ...],
    safe_actions: tuple[str, ...],
) -> dict[str, float]:
    probabilities = _distribution(tuple(policy.probabilities(history).items()), actions)
    if any(p > 0 and action not in safe_actions for action, p in probabilities.items()):
        raise IdentificationError("target policy requests an unsafe action")
    return probabilities


def _validate(
    trajectories: Sequence[LoggedTrajectory],
    policy: EvaluationPolicy,
    record: IdentificationRecord,
    method: str,
    discount: float,
) -> None:
    record.require_identified(method)
    if not math.isfinite(discount) or not 0 <= discount <= 1 or not trajectories:
        raise ValueError("nonempty trajectories and discount in [0, 1] are required")
    if policy.strategy_id not in record.trial.treatment_strategies:
        raise IdentificationError("policy is not a declared target-trial strategy")
    horizon = len(trajectories[0].steps)
    if not horizon or len({t.subject_id for t in trajectories}) != len(trajectories):
        raise ValueError("one nonempty trajectory per independent subject is required")
    actions = record.trial.actions
    for trajectory in trajectories:
        if (
            not trajectory.subject_id
            or not trajectory.eligible
            or trajectory.censored
            or len(trajectory.steps) != horizon
        ):
            raise IdentificationError(
                "requires eligible complete uncensored equal-horizon subjects"
            )
        if (
            not math.isfinite(trajectory.time_zero)
            or not math.isfinite(trajectory.eligible_at)
            or trajectory.eligible_at > trajectory.time_zero
        ):
            raise IdentificationError("eligibility must be established at or before time zero")
        previous_outcome = trajectory.time_zero
        for index, step in enumerate(trajectory.steps):
            if (
                not step.state
                or any(
                    not math.isfinite(v)
                    for v in (step.reward, step.decision_time, step.outcome_time)
                )
                or step.decision_time < previous_outcome
                or step.outcome_time <= step.decision_time
                or step.outcome_time > trajectory.time_zero + record.trial.follow_up
            ):
                raise IdentificationError("invalid action/outcome chronology or follow-up")
            if len({c.name for c in step.covariates}) != len(step.covariates) or any(
                not c.name
                or not math.isfinite(c.value)
                or not math.isfinite(c.available_at)
                or c.available_at > step.decision_time
                for c in step.covariates
            ):
                raise IdentificationError("future or invalid covariate in decision history")
            if not set(step.safe_actions) <= set(actions) or step.action not in step.safe_actions:
                raise IdentificationError("observed action violates declared action/safety support")
            behavior = _distribution(step.behavior_probabilities, actions)
            target = _policy(policy, trajectory.history(index), actions, step.safe_actions)
            if behavior.get(step.action, 0) <= 0:
                raise IdentificationError("observed action has zero logging propensity")
            if any(target[a] > 0 and behavior[a] <= 0 for a in actions):
                raise IdentificationError("target policy violates conditional positivity")
            previous_outcome = step.outcome_time
    observed_actions = {step.action for trajectory in trajectories for step in trajectory.steps}
    if not set(actions) <= observed_actions:
        raise IdentificationError("logged data lack declared intervention variation/action support")


def _nuisance_guard(nuisance: OutcomeNuisance, trajectories: Sequence[LoggedTrajectory]) -> None:
    if not nuisance.source or (not nuisance.external and not nuisance.fit_subject_ids):
        raise ValueError("declare external nuisance source or independent fitting subjects")
    if set(nuisance.fit_subject_ids) & {t.subject_id for t in trajectories}:
        raise IdentificationError("nuisance fitting/evaluation subject leakage")


def _v(
    nuisance: OutcomeNuisance, history: HistoryView, probabilities: Mapping[str, float]
) -> float:
    values = [
        probability * nuisance.q_value(history, action)
        for action, probability in probabilities.items()
        if probability > 0
    ]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("nuisance predictions must be finite")
    return sum(values)


def evaluate_policy(
    trajectories: Sequence[LoggedTrajectory],
    policy: EvaluationPolicy,
    identification: IdentificationRecord,
    *,
    method: str = "ipw",
    nuisance: OutcomeNuisance | None = None,
    discount: float = 1.0,
) -> PolicyValueEstimate:
    """Estimate E_pi[sum gamma**t R_t] using IPW, sequential DR, or Q g-computation.

    DR uses V_0 + sum gamma**t rho_0:t (R_t + gamma V_{t+1} - Q_t).
    G-computation requires Q_t to be the *target-policy return*, not a one-step mean.
    Nuisance uncertainty is not included in the returned per-subject contributions.
    """
    if method not in ("ipw", "dr", "g_computation", "aipw"):
        raise ValueError("unknown estimator")
    _validate(trajectories, policy, identification, method, discount)
    if method == "aipw" and len(trajectories[0].steps) != 1:
        raise ValueError("AIPW is the one-decision estimator; use dr for longitudinal data")
    if method != "ipw":
        if nuisance is None:
            raise ValueError("this estimator requires an outcome nuisance model")
        _nuisance_guard(nuisance, trajectories)
    contributions, final_weights = [], []
    minimum_probability, maximum_weight = 1.0, 0.0
    for trajectory in trajectories:
        targets = [
            _policy(policy, trajectory.history(i), identification.trial.actions, s.safe_actions)
            for i, s in enumerate(trajectory.steps)
        ]
        values = (
            [_v(nuisance, trajectory.history(i), p) for i, p in enumerate(targets)]
            if nuisance is not None
            else []
        )
        if method != "ipw" and nuisance is None:
            raise ValueError("this estimator requires an outcome nuisance model")
        contribution = values[0] if method != "ipw" else 0.0
        weight = 1.0
        for i, step in enumerate(trajectory.steps):
            behavior = dict(step.behavior_probabilities)
            minimum_probability = min(
                minimum_probability, *(behavior[a] for a, p in targets[i].items() if p > 0)
            )
            weight *= targets[i][step.action] / behavior[step.action]
            if not math.isfinite(weight):
                raise ValueError("importance weight overflow; overlap is inadequate")
            maximum_weight = max(maximum_weight, weight)
            if method == "ipw":
                contribution += discount**i * weight * step.reward
            elif method in ("dr", "aipw"):
                assert nuisance is not None
                q = nuisance.q_value(trajectory.history(i), step.action)
                if not math.isfinite(q):
                    raise ValueError("nuisance predictions must be finite")
                following = values[i + 1] if i + 1 < len(values) else 0.0
                contribution += discount**i * weight * (step.reward + discount * following - q)
        if not math.isfinite(contribution):
            raise ValueError("nonfinite policy contribution")
        contributions.append(contribution)
        final_weights.append(weight)
    squared = sum(w * w for w in final_weights)
    ess = sum(final_weights) ** 2 / squared if squared else 0.0
    if method != "g_computation" and not squared:
        raise IdentificationError("no observed complete trajectory supports this policy")
    return PolicyValueEstimate(
        sum(contributions) / len(contributions),
        method,
        identification,
        policy.strategy_id,
        tuple(t.subject_id for t in trajectories),
        tuple(contributions),
        None if method == "g_computation" else ess,
        minimum_probability,
        maximum_weight,
        identification.limitations
        + (
            "finite-sample estimate; not clinical evidence",
            "nuisance fitting uncertainty requires refitting bootstrap",
        ),
        discount,
    )


def empirical_g_formula(
    trajectories: Sequence[LoggedTrajectory],
    policy: EvaluationPolicy,
    identification: IdentificationRecord,
    *,
    discount: float = 1.0,
) -> PolicyValueEstimate:
    """Saturated discrete-history g-formula, fitted from finite logged trajectories.

    Each history/action cell estimates its empirical joint reward/next-history law.
    Unsupported target branches raise; there is no smoothing or extrapolation.
    Histories include past rewards and acquisition times, so continuous data usually
    require an independently fitted conditional model through g_computation instead.
    """
    _validate(trajectories, policy, identification, "empirical_g_formula", discount)
    cells: dict[tuple[HistoryView, str], list[tuple[float, HistoryView | None]]] = {}
    safe: dict[HistoryView, tuple[str, ...]] = {}
    for trajectory in trajectories:
        for i, step in enumerate(trajectory.steps):
            history = trajectory.history(i)
            if history in safe and set(safe[history]) != set(step.safe_actions):
                raise IdentificationError("inconsistent safe actions for identical history")
            safe[history] = step.safe_actions
            following = trajectory.history(i + 1) if i + 1 < len(trajectory.steps) else None
            cells.setdefault((history, step.action), []).append((step.reward, following))
    cache: dict[HistoryView, float] = {}

    def value(history: HistoryView) -> float:
        if history not in cache:
            probabilities = _policy(policy, history, identification.trial.actions, safe[history])
            total = 0.0
            for action, probability in probabilities.items():
                if probability == 0:
                    continue
                rows = cells.get((history, action))
                if not rows:
                    raise IdentificationError(
                        "g-formula target history/action cell has no observations"
                    )
                total += (
                    probability
                    * sum(
                        reward + (discount * value(following) if following is not None else 0)
                        for reward, following in rows
                    )
                    / len(rows)
                )
            cache[history] = total
        return cache[history]

    contributions = tuple(value(t.history(0)) for t in trajectories)
    return PolicyValueEstimate(
        sum(contributions) / len(contributions),
        "empirical_g_formula",
        identification,
        policy.strategy_id,
        tuple(t.subject_id for t in trajectories),
        contributions,
        None,
        min(p for t in trajectories for s in t.steps for _, p in s.behavior_probabilities if p > 0),
        0.0,
        identification.limitations
        + (
            "saturated finite-history empirical model",
            "refit the conditional law for bootstrap uncertainty",
        ),
        discount,
    )


@dataclass(frozen=True)
class SensitivityBounds:
    lower: float
    upper: float
    gamma: float
    model: str = "binary per-history propensity odds bounded by Gamma"
    status: str = "SENSITIVITY_BOUND_NOT_IDENTIFICATION"


def hidden_confounding_bounds(
    trajectories: Sequence[LoggedTrajectory],
    policy: EvaluationPolicy,
    identification: IdentificationRecord,
    *,
    gamma: float,
    discount: float = 1.0,
) -> SensitivityBounds:
    """Conservative unnormalized IPW bounds; not sharp joint-law bounds."""
    if not math.isfinite(gamma) or gamma < 1 or len(identification.trial.actions) != 2:
        raise ValueError("binary actions and finite Gamma >= 1 are required")
    _validate(trajectories, policy, identification, identification.estimator, discount)
    lower = upper = 0.0
    for trajectory in trajectories:
        weight_low = weight_high = 1.0
        for i, step in enumerate(trajectory.steps):
            target = _policy(
                policy, trajectory.history(i), identification.trial.actions, step.safe_actions
            )[step.action]
            p = dict(step.behavior_probabilities)[step.action]
            p_low = p / (p + gamma * (1 - p))
            p_high = gamma * p / (1 - p + gamma * p)
            weight_low *= target / p_high
            weight_high *= target / p_low
            endpoints = (
                discount**i * step.reward * weight_low,
                discount**i * step.reward * weight_high,
            )
            lower += min(endpoints)
            upper += max(endpoints)
    return SensitivityBounds(lower / len(trajectories), upper / len(trajectories), gamma)


def causal_placebo(
    trajectories: Sequence[LoggedTrajectory],
    policy: EvaluationPolicy,
    identification: IdentificationRecord,
    pre_treatment_outcomes: Sequence[tuple[float, float]],
) -> float:
    """Weighted baseline-placebo contrast; nonzero values can falsify an analysis.

    A zero contrast cannot establish exchangeability. Pairs are (value, available_at).
    """
    _validate(trajectories, policy, identification, identification.estimator, 1.0)
    if len(pre_treatment_outcomes) != len(trajectories):
        raise ValueError("one timestamped placebo outcome per subject is required")
    contrasts = []
    for trajectory, (outcome, available_at) in zip(trajectories, pre_treatment_outcomes):
        if (
            not math.isfinite(outcome)
            or not math.isfinite(available_at)
            or available_at > trajectory.time_zero
        ):
            raise IdentificationError("placebo outcome must precede treatment assignment")
        step = trajectory.steps[0]
        probability = _policy(
            policy, trajectory.history(0), identification.trial.actions, step.safe_actions
        )[step.action]
        weight = probability / dict(step.behavior_probabilities)[step.action]
        contrasts.append((weight - 1) * outcome)
    return sum(contrasts) / len(contrasts)

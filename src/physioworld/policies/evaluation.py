"""Policy comparison under one shared objective and constraint declaration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PolicyObjective:
    objective_id: str
    utility: str
    constraint_id: str
    horizon: int

    def validate(self) -> None:
        if not self.objective_id or not self.utility or not self.constraint_id or self.horizon < 1:
            raise ValueError("policy objective requires utility, constraints, and positive horizon")


@dataclass(frozen=True)
class PolicyComparison:
    objective: PolicyObjective
    values: Mapping[str, float]
    violations: Mapping[str, int]

    def regret_to(self, oracle_name: str) -> dict[str, float]:
        if oracle_name not in self.values:
            raise KeyError(oracle_name)
        oracle_value = self.values[oracle_name]
        return {name: oracle_value - value for name, value in self.values.items()}

    def feasible(self, policy_name: str) -> bool:
        return self.violations.get(policy_name, 0) == 0


@dataclass(frozen=True)
class FinitePolicyEvaluation:
    values: Mapping[tuple[int, object], float]
    safety_violation_probabilities: Mapping[tuple[int, object], float]
    decision_problem_token: int
    evidence: str = "MODEL_DERIVED"


def evaluate_finite_policy(mdp, action_law):
    """Exact Markov policy evaluation, including probability of ever leaving safe states.

    action_law(time,state) supplies probabilities over declared actions.
    Resource/time admissibility is mandatory. Safety risk is reported rather
    than conflated with value or silently filtering the policy.
    """
    mdp.validate()
    values, risks = {}, {}
    safe = set(mdp.states) if mdp.safe_states is None else set(mdp.safe_states)
    for state in mdp.states:
        values[(mdp.horizon, state)] = (
            0.0 if mdp.terminal_reward is None else mdp.terminal_reward(state)
        )
        risks[(mdp.horizon, state)] = float(state not in safe)
    for time in range(mdp.horizon - 1, -1, -1):
        for state in mdp.states:
            law = action_law(time, state)
            if (
                not law
                or not set(law) <= set(mdp.actions[state])
                or not all(math.isfinite(p) and p >= 0 for p in law.values())
                or abs(math.fsum(law.values()) - 1) > 1e-12
            ):
                raise ValueError("policy law must be stochastic on declared actions")
            value, risk = 0.0, 0.0
            for action, probability in law.items():
                if probability == 0:
                    continue
                if mdp.admissible is not None and not mdp.admissible(state, action, time):
                    raise ValueError("policy violates explicit resource/time admissibility")
                kernel = mdp.kernel(time, state, action)
                value += probability * (
                    mdp.immediate(time, state, action)
                    + mdp.gamma * math.fsum(p * values[(time + 1, n)] for n, p in kernel.items())
                )
                risk += probability * math.fsum(p * risks[(time + 1, n)] for n, p in kernel.items())
            values[(time, state)] = value
            risks[(time, state)] = 1.0 if state not in safe else risk
    return FinitePolicyEvaluation(values, risks, id(mdp))


def select_chance_constrained_policy(evaluations, initial_state, maximum_risk, *, time=0):
    """Select among exact evaluations sharing one objective and model declaration."""
    if not math.isfinite(maximum_risk) or not 0 <= maximum_risk <= 1:
        raise ValueError("maximum safety risk must lie in [0,1]")
    if not evaluations or len({e.decision_problem_token for e in evaluations.values()}) != 1:
        raise ValueError(
            "chance selection requires evaluations of the same decision-problem instance"
        )
    candidates = [
        (evaluation.values[(time, initial_state)], policy)
        for policy, evaluation in evaluations.items()
        if evaluation.safety_violation_probabilities[(time, initial_state)] <= maximum_risk
    ]
    if not candidates:
        raise ValueError("no policy meets the declared chance constraint")
    value, policy = max(candidates, key=lambda pair: pair[0])
    return policy, value

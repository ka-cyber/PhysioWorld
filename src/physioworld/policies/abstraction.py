"""Finite Markov quotient verification and controlled reachable state sets."""

import math
from dataclasses import dataclass
from typing import Any

from .dp import FiniteMDP


@dataclass(frozen=True)
class QuotientCheck:
    valid: bool
    violations: tuple[str, ...]
    scope: str = "exact supplied finite model up to declared floating-point tolerance"


def check_markov_quotient(mdp, partition, *, tolerance=1e-12):
    mdp.validate()
    if set(partition) != set(mdp.states) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("partition must cover every state and tolerance must be finite")
    classes: dict[Any, list[Any]] = {}
    for state in mdp.states:
        classes.setdefault(partition[state], []).append(state)
    safe = set(mdp.states) if mdp.safe_states is None else set(mdp.safe_states)
    violations: list[str] = []
    for label, states in classes.items():
        representative = states[0]
        for state in states[1:]:
            if set(mdp.actions[state]) != set(mdp.actions[representative]):
                violations.append(f"action mismatch in class {label!r}")
                continue
            if (state in safe) != (representative in safe):
                violations.append(f"safety mismatch in class {label!r}")
            if (
                mdp.terminal_reward is not None
                and abs(mdp.terminal_reward(state) - mdp.terminal_reward(representative))
                > tolerance
            ):
                violations.append(f"terminal reward mismatch in class {label!r}")
            for time in range(mdp.horizon):
                for action in mdp.actions[state]:
                    if (
                        abs(
                            mdp.immediate(time, state, action)
                            - mdp.immediate(time, representative, action)
                        )
                        > tolerance
                    ):
                        violations.append(f"reward mismatch in class {label!r}")
                    if mdp.admissible is not None and mdp.admissible(
                        state, action, time
                    ) != mdp.admissible(representative, action, time):
                        violations.append(f"admissibility mismatch in class {label!r}")
                    left: dict[Any, float] = {}
                    right: dict[Any, float] = {}
                    for s, p in mdp.kernel(time, state, action).items():
                        left[partition[s]] = left.get(partition[s], 0.0) + p
                    for s, p in mdp.kernel(time, representative, action).items():
                        right[partition[s]] = right.get(partition[s], 0.0) + p
                    if any(abs(left.get(k, 0) - right.get(k, 0)) > tolerance for k in classes):
                        violations.append(f"pushforward kernel mismatch in class {label!r}")
    return QuotientCheck(not violations, tuple(dict.fromkeys(violations)))


def quotient_mdp(mdp, partition, *, tolerance=1e-12):
    check = check_markov_quotient(mdp, partition, tolerance=tolerance)
    if not check.valid:
        raise ValueError("; ".join(check.violations))
    representatives: dict[Any, Any] = {}
    for state in mdp.states:
        representatives.setdefault(partition[state], state)
    safe = None if mdp.safe_states is None else frozenset(partition[s] for s in mdp.safe_states)

    def transition(time, label, action):
        result: dict[Any, float] = {}
        for state, p in mdp.kernel(time, representatives[label], action).items():
            result[partition[state]] = result.get(partition[state], 0.0) + p
        return result

    return FiniteMDP(
        tuple(representatives),
        {k: mdp.actions[s] for k, s in representatives.items()},
        lambda label, a: transition(0, label, a),
        lambda label, a: mdp.immediate(0, representatives[label], a),
        mdp.horizon,
        mdp.gamma,
        safe,
        None
        if mdp.admissible is None
        else lambda label, a, t: mdp.admissible(representatives[label], a, t),
        None
        if mdp.terminal_reward is None
        else lambda label: mdp.terminal_reward(representatives[label]),
        None if mdp.time_transition is None else transition,
        None
        if mdp.time_reward is None
        else lambda t, label, a: mdp.immediate(t, representatives[label], a),
    )


def reachable_sets(mdp, initial_states, *, safe_only=False):
    """Possible states per time under admissible actions, not a viability claim."""
    mdp.validate()
    if not initial_states or not set(initial_states) <= set(mdp.states):
        raise ValueError("initial states must belong to declared state space")
    safe = set(mdp.states) if mdp.safe_states is None else set(mdp.safe_states)
    current, sets = set(initial_states), [frozenset(initial_states)]
    for time in range(mdp.horizon):
        next_states: set[Any] = set()
        for state in current:
            if safe_only and state not in safe:
                continue
            for action in mdp.actions[state]:
                if mdp.admissible is not None and not mdp.admissible(state, action, time):
                    continue
                next_states.update(
                    n
                    for n, p in mdp.kernel(time, state, action).items()
                    if p > 0 and (not safe_only or n in safe)
                )
        current = next_states
        sets.append(frozenset(current))
    return tuple(sets)

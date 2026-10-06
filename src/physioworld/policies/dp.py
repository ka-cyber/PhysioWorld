"""Validated finite/infinite Markov control, hard viability, and true lookahead."""

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .base import Policy


@dataclass(frozen=True)
class FiniteMDP:
    states: tuple[Any, ...]
    actions: Mapping[Any, tuple[Any, ...]]
    transition: Callable[[Any, Any], Mapping[Any, float]]
    reward: Callable[[Any, Any], float]
    horizon: int
    gamma: float = 1.0
    safe_states: frozenset[Any] | None = None
    admissible: Callable[[Any, Any, int], bool] | None = None
    terminal_reward: Callable[[Any], float] | None = None
    time_transition: Callable[[int, Any, Any], Mapping[Any, float]] | None = None
    time_reward: Callable[[int, Any, Any], float] | None = None

    def validate(self):
        if (
            not isinstance(self.horizon, int)
            or self.horizon < 1
            or not self.states
            or len(set(self.states)) != len(self.states)
            or not math.isfinite(self.gamma)
            or not 0 <= self.gamma <= 1
        ):
            raise ValueError("finite MDP needs unique states, positive horizon and valid discount")
        if self.safe_states is not None and not self.safe_states <= set(self.states):
            raise ValueError("safe states must belong to declared state space")
        for state in self.states:
            if (
                state not in self.actions
                or not self.actions[state]
                or len(set(self.actions[state])) != len(self.actions[state])
            ):
                raise ValueError(f"state {state!r} needs unique nonempty actions")
            terminal = 0.0 if self.terminal_reward is None else self.terminal_reward(state)
            if not math.isfinite(terminal):
                raise ValueError("terminal rewards must be finite")
            for time in range(self.horizon):
                for action in self.actions[state]:
                    _validate_transition(self.kernel(time, state, action), set(self.states))
                    if not math.isfinite(self.immediate(time, state, action)):
                        raise ValueError("rewards must be finite")

    def kernel(self, time, state, action):
        return (
            self.transition(state, action)
            if self.time_transition is None
            else self.time_transition(time, state, action)
        )

    def immediate(self, time, state, action):
        return (
            self.reward(state, action)
            if self.time_reward is None
            else self.time_reward(time, state, action)
        )


def _validate_transition(kernel, states):
    if (
        not kernel
        or not set(kernel) <= states
        or not all(math.isfinite(p) and p >= 0 for p in kernel.values())
        or abs(math.fsum(kernel.values()) - 1.0) > 1e-12
    ):
        raise ValueError("transition must be a stochastic kernel on declared states")


class DynamicProgrammingOracle(Policy):
    name = "dynamic_programming_oracle"

    def __init__(self, mdp):
        mdp.validate()
        self.mdp = mdp
        self._values, self._actions = {}, {}
        # Snapshot the validated model: later mutable callbacks cannot change
        # the problem midway through an optimization.
        self._kernels = {
            (t, s, a): dict(mdp.kernel(t, s, a))
            for t in range(mdp.horizon)
            for s in mdp.states
            for a in mdp.actions[s]
        }
        self._rewards = {(t, s, a): mdp.immediate(t, s, a) for t, s, a in self._kernels}
        self._allowed = {
            (t, s, a): mdp.admissible is None or mdp.admissible(s, a, t)
            for t, s, a in self._kernels
        }
        for key, kernel in self._kernels.items():
            _validate_transition(kernel, set(mdp.states))
            if not math.isfinite(self._rewards[key]):
                raise ValueError("snapshot rewards must remain finite")
        safe = set(mdp.states) if mdp.safe_states is None else set(mdp.safe_states)
        viable: list[set[Any]] = [set() for _ in range(mdp.horizon + 1)]
        viable[-1] = safe
        for state in mdp.states:
            self._values[(mdp.horizon, state)] = (
                (0.0 if mdp.terminal_reward is None else mdp.terminal_reward(state))
                if state in safe
                else -math.inf
            )
        for time in range(mdp.horizon - 1, -1, -1):
            for state in mdp.states:
                candidates = []
                if state in safe:
                    for action in mdp.actions[state]:
                        key = (time, state, action)
                        kernel = self._kernels[key]
                        if self._allowed[key] and all(
                            n in viable[time + 1] for n, p in kernel.items() if p > 0
                        ):
                            expectation = math.fsum(
                                p * self._values[(time + 1, n)] for n, p in kernel.items() if p > 0
                            )
                            candidates.append(
                                (self._rewards[key] + mdp.gamma * expectation, action)
                            )
                if candidates:
                    value, selected = max(candidates, key=lambda pair: pair[0])
                    self._values[(time, state)], self._actions[(time, state)] = value, selected
                    viable[time].add(state)
                else:
                    self._values[(time, state)] = -math.inf
        self.viability_sets = tuple(frozenset(v) for v in viable)

    def action_at(self, time, state):
        if (time, state) not in self._actions:
            raise ValueError("state/time has no viable action or lies outside the decision horizon")
        return self._actions[(time, state)]

    def action(self, state, available_actions):
        selected = self.action_at(0, state)
        if selected not in available_actions:
            raise ValueError("oracle action is not currently admissible")
        return selected

    def value(self, time, state):
        return self._values[(time, state)]


def viability_kernel(mdp):
    """Greatest almost-sure invariant subset for a time-homogeneous MDP."""
    mdp.validate()
    if mdp.time_transition is not None or mdp.admissible is not None:
        raise ValueError("invariant viability requires time-homogeneous explicit action sets")
    viable = set(mdp.states) if mdp.safe_states is None else set(mdp.safe_states)
    while True:
        updated = {
            s
            for s in viable
            if any(
                all(n in viable for n, p in mdp.transition(s, a).items() if p > 0)
                for a in mdp.actions[s]
            )
        }
        if updated == viable:
            return frozenset(viable)
        viable = updated


@dataclass(frozen=True)
class InfiniteHorizonResult:
    values: Mapping[Any, float]
    actions: Mapping[Any, Any]
    iterations: int
    bellman_residual: float
    value_error_bound: float
    converged: bool
    viable_states: frozenset[Any]


def discounted_value_iteration(mdp, *, gamma=None, tolerance=1e-10, max_iterations=100000):
    mdp.validate()
    gamma = mdp.gamma if gamma is None else gamma
    if (
        not math.isfinite(gamma)
        or not 0 <= gamma < 1
        or not math.isfinite(tolerance)
        or tolerance <= 0
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("infinite discounted control requires gamma<1 and valid stopping criteria")
    if mdp.time_transition is not None or mdp.time_reward is not None or mdp.admissible is not None:
        raise ValueError("infinite-horizon kernel, reward and admissibility must be stationary")
    viable = viability_kernel(mdp)
    values = {s: 0.0 for s in viable}
    if not viable:
        return InfiniteHorizonResult({}, {}, 0, 0.0, 0.0, True, viable)
    kernels = {
        (s, a): dict(mdp.transition(s, a))
        for s in viable
        for a in mdp.actions[s]
        if all(n in viable for n, p in mdp.transition(s, a).items() if p > 0)
    }
    rewards = {(s, a): mdp.reward(s, a) for s, a in kernels}

    def bellman(v):
        updated, selected = {}, {}
        for s in viable:
            pairs = [
                (rewards[(s, a)] + gamma * math.fsum(p * v[n] for n, p in k.items() if p > 0), a)
                for (state, a), k in kernels.items()
                if state == s
            ]
            updated[s], selected[s] = max(pairs, key=lambda pair: pair[0])
        return updated, selected

    for iteration in range(1, max_iterations + 1):
        values, _ = bellman(values)
        next_values, actions = bellman(values)
        residual = max(abs(next_values[s] - values[s]) for s in viable)
        if residual / (1 - gamma) <= tolerance:
            return InfiniteHorizonResult(
                values, actions, iteration, residual, residual / (1 - gamma), True, viable
            )
    return InfiniteHorizonResult(
        values, actions, iteration, residual, residual / (1 - gamma), False, viable
    )


@dataclass(frozen=True)
class MPCPolicy(Policy):
    objective: Callable[[Any, Any], float]
    horizon: int = 1
    name: str = "mpc"
    transition: Callable[[Any, Any], Mapping[Any, float]] | None = None
    future_actions: Callable[[Any], Sequence[Any]] | None = None
    gamma: float = 1.0

    def action(self, state, available_actions):
        if (
            not isinstance(self.horizon, int)
            or self.horizon < 1
            or not available_actions
            or not math.isfinite(self.gamma)
            or not 0 <= self.gamma <= 1
        ):
            raise ValueError("invalid MPC request")
        if self.horizon > 1 and (self.transition is None or self.future_actions is None):
            raise ValueError("multi-step MPC requires dynamics and future admissible actions")

        transition = self.transition
        future_actions = self.future_actions

        def score(s, action, depth):
            reward = self.objective(s, action)
            if not math.isfinite(reward):
                raise ValueError("MPC rewards must be finite")
            if depth == 1:
                return reward
            # The guard above establishes both callbacks for a multi-step plan.
            # Keep local bindings so that narrowing remains valid in recursion.
            if transition is None or future_actions is None:
                raise RuntimeError("MPC dynamics callbacks are unavailable")
            kernel = transition(s, action)
            _validate_transition(kernel, set(kernel))
            continuation = 0.0
            for n, p in kernel.items():
                if p == 0:
                    continue
                actions = future_actions(n)
                if not actions:
                    return -math.inf
                continuation += p * max(score(n, a, depth - 1) for a in actions)
            return reward + self.gamma * continuation

        scores = [(score(state, a, self.horizon), a) for a in available_actions]
        value, action = max(scores, key=lambda pair: pair[0])
        if value == -math.inf:
            raise ValueError("no viable MPC plan")
        return action

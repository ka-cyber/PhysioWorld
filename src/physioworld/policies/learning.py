"""Seeded tabular bandit, TD, and empirical model-based learning algorithms."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence

from .base import Policy


@dataclass
class ContextualBandit(Policy):
    actions: tuple[Any, ...]
    epsilon: float = 0.1
    seed: int = 0
    name: str = "contextual_bandit"
    values: dict[tuple[Hashable, Any], float] = field(default_factory=dict)
    counts: dict[tuple[Hashable, Any], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.actions
            or len(set(self.actions)) != len(self.actions)
            or not 0 <= self.epsilon <= 1
        ):
            raise ValueError("bandit requires unique actions and epsilon in [0,1]")
        self._rng = random.Random(self.seed)

    def action(self, state: Hashable, available_actions: Sequence[Any]) -> Any:
        if not available_actions or not set(available_actions) <= set(self.actions):
            raise ValueError("bandit needs a nonempty subset of its declared actions")
        if self._rng.random() < self.epsilon:
            return available_actions[self._rng.randrange(len(available_actions))]
        return max(available_actions, key=lambda action: self.values.get((state, action), 0.0))

    def update(self, state: Hashable, action: Any, reward: float) -> None:
        if action not in self.actions or not math.isfinite(reward):
            raise ValueError("invalid bandit feedback")
        key = (state, action)
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        old = self.values.get(key, 0.0)
        self.values[key] = old + (reward - old) / count


@dataclass
class QLearningPolicy(Policy):
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    seed: int = 0
    name: str = "q_learning"
    q: dict[tuple[Hashable, Any], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1 or not 0 <= self.gamma <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("invalid learning rates/discount/exploration")
        self._rng = random.Random(self.seed)

    def action(self, state: Hashable, available_actions: Sequence[Any]) -> Any:
        if not available_actions:
            raise ValueError("no available actions")
        if self._rng.random() < self.epsilon:
            return available_actions[self._rng.randrange(len(available_actions))]
        return max(available_actions, key=lambda action: self.q.get((state, action), 0.0))

    def update(
        self,
        state: Hashable,
        action: Any,
        reward: float,
        next_state: Hashable,
        next_actions: Sequence[Any],
        *,
        terminal=False,
    ) -> None:
        if not math.isfinite(reward):
            raise ValueError("learning rewards must be finite")
        if not terminal and not next_actions:
            raise ValueError("nonterminal Q-learning update requires next admissible actions")
        next_max = (
            0.0
            if terminal
            else max((self.q.get((next_state, a), 0.0) for a in next_actions), default=0.0)
        )
        key = (state, action)
        self.q[key] = self.q.get(key, 0.0) + self.alpha * (
            reward + self.gamma * next_max - self.q.get(key, 0.0)
        )


@dataclass
class SARSAPolicy(QLearningPolicy):
    name: str = "sarsa"

    def update_sarsa(
        self,
        state: Hashable,
        action: Any,
        reward: float,
        next_state: Hashable,
        next_action: Any,
        *,
        terminal=False,
    ) -> None:
        if not math.isfinite(reward):
            raise ValueError("learning rewards must be finite")
        key = (state, action)
        target = reward + (
            0.0 if terminal else self.gamma * self.q.get((next_state, next_action), 0.0)
        )
        self.q[key] = self.q.get(key, 0.0) + self.alpha * (target - self.q.get(key, 0.0))


class TabularModelBasedPolicy(Policy):
    """Empirical transition/reward learning followed by finite-horizon planning.

    Offline mode admits only actions supported by min_count observations.
    Online unseen actions use an explicitly declared zero-reward self-loop
    prior and epsilon exploration. Neither mode invents clinical dynamics.
    """

    name = "tabular_model_based"

    def __init__(
        self,
        states,
        actions,
        horizon,
        *,
        gamma=1.0,
        epsilon=0.1,
        seed=0,
        offline=False,
        min_count=1,
        safe_states=None,
    ):
        if not 0 <= epsilon <= 1 or min_count < 1:
            raise ValueError("invalid exploration/support threshold")
        self.states, self.actions, self.horizon = tuple(states), dict(actions), horizon
        self.gamma, self.epsilon, self.offline = gamma, epsilon, offline
        self.min_count, self.safe_states = min_count, safe_states
        self._rng = random.Random(seed)
        self.counts, self.transitions, self.reward_sums = {}, {}, {}
        self._validate_prior()

    def _validate_prior(self):
        from .dp import FiniteMDP

        FiniteMDP(
            self.states,
            self.actions,
            lambda s, a: {s: 1.0},
            lambda s, a: 0.0,
            self.horizon,
            self.gamma,
            self.safe_states,
        ).validate()

    def observe(self, state, action, reward, next_state):
        if (
            state not in self.actions
            or action not in self.actions[state]
            or next_state not in self.states
            or not math.isfinite(reward)
        ):
            raise ValueError("observation outside declared model or nonfinite reward")
        key = (state, action)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.reward_sums[key] = self.reward_sums.get(key, 0.0) + reward
        row = self.transitions.setdefault(key, {})
        row[next_state] = row.get(next_state, 0) + 1

    def fit_offline(self, records):
        if not self.offline:
            raise ValueError("offline fitting requires offline=True")
        for state, action, reward, next_state in records:
            self.observe(state, action, reward, next_state)
        return self

    def plan(self):
        from .dp import DynamicProgrammingOracle, FiniteMDP

        def kernel(s, a):
            key = (s, a)
            count = self.counts.get(key, 0)
            return {n: c / count for n, c in self.transitions[key].items()} if count else {s: 1.0}

        def reward(s, a):
            key = (s, a)
            return self.reward_sums.get(key, 0.0) / max(1, self.counts.get(key, 0))

        return DynamicProgrammingOracle(
            FiniteMDP(
                self.states,
                self.actions,
                kernel,
                reward,
                self.horizon,
                self.gamma,
                self.safe_states,
                admissible=lambda s, a, t: (
                    not self.offline or self.counts.get((s, a), 0) >= self.min_count
                ),
            )
        )

    def action(self, state, available_actions):
        if state not in self.actions or not set(available_actions) <= set(self.actions[state]):
            raise ValueError("action request outside learned model")
        oracle = self.plan()
        safe = tuple(
            a
            for a in available_actions
            if oracle._allowed[(0, state, a)]
            and all(
                n in oracle.viability_sets[1]
                for n, p in oracle._kernels[(0, state, a)].items()
                if p > 0
            )
        )
        if not safe:
            raise ValueError("no supported viable action")
        if not self.offline and self._rng.random() < self.epsilon:
            return self._rng.choice(safe)
        # Replan the first action with current availability, preserving all
        # continuation policies from the same empirical model.
        return max(
            safe,
            key=lambda a: (
                oracle._rewards[(0, state, a)]
                + self.gamma
                * sum(
                    p * oracle.value(1, n)
                    for n, p in oracle._kernels[(0, state, a)].items()
                    if p > 0
                )
            ),
        )

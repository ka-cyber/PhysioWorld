"""Policy API with explicit action-time information."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence


class Policy:
    name = "policy"

    def action(self, state: Any, available_actions: Sequence[Any]) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class FixedPolicy(Policy):
    fixed_action: Any
    name: str = "fixed"

    def action(self, state: Any, available_actions: Sequence[Any]) -> Any:
        if self.fixed_action not in available_actions:
            raise ValueError("fixed action is not admissible")
        return self.fixed_action


@dataclass(frozen=True)
class RulePolicy(Policy):
    rule: Callable[[Any, Sequence[Any]], Any]
    name: str = "rule_based"

    def action(self, state: Any, available_actions: Sequence[Any]) -> Any:
        action = self.rule(state, available_actions)
        if action not in available_actions:
            raise ValueError("rule selected an inadmissible action")
        return action


@dataclass(frozen=True)
class GreedyPolicy(Policy):
    score: Callable[[Any, Any], float]
    name: str = "greedy"

    def action(self, state: Any, available_actions: Sequence[Any]) -> Any:
        if not available_actions:
            raise ValueError("no admissible actions")
        return max(available_actions, key=lambda action: self.score(state, action))


@dataclass(frozen=True)
class TimedInformation:
    key: str
    value: Any
    acquired_at: float
    available_at: float

    def __post_init__(self):
        if (
            not self.key
            or not math.isfinite(self.acquired_at)
            or not math.isfinite(self.available_at)
            or self.acquired_at > self.available_at
        ):
            raise ValueError("invalid information timestamps")


@dataclass(frozen=True)
class DecisionContext:
    time: float
    information: tuple[TimedInformation, ...]
    allowed_keys: frozenset[str]

    def visible(self):
        if not math.isfinite(self.time) or len({i.key for i in self.information}) != len(
            self.information
        ):
            raise ValueError("invalid decision context")
        return {
            item.key: item.value
            for item in self.information
            if item.key in self.allowed_keys and item.available_at <= self.time
        }


@dataclass(frozen=True)
class NonanticipativePolicy(Policy):
    policy: Policy
    name: str = "nonanticipative"

    def action(self, state: DecisionContext, available_actions):
        if not isinstance(state, DecisionContext):
            raise TypeError("nonanticipative policy requires explicit timed information")
        selected = self.policy.action(state.visible(), available_actions)
        if selected not in available_actions:
            raise ValueError("nonanticipative policy selected an inadmissible action")
        return selected


class StochasticPolicy(Policy):
    name = "stochastic"

    def __init__(self, probabilities, seed=0):
        self.probabilities, self._rng = probabilities, random.Random(seed)

    def action(self, state, available_actions):
        probabilities = dict(self.probabilities(state, tuple(available_actions)))
        if (
            not probabilities
            or not set(probabilities) <= set(available_actions)
            or not all(math.isfinite(p) and p >= 0 for p in probabilities.values())
            or abs(math.fsum(probabilities.values()) - 1) > 1e-12
        ):
            raise ValueError("stochastic policy must assign a valid law to admissible actions")
        draw, cumulative = self._rng.random(), 0.0
        for action, probability in probabilities.items():
            cumulative += probability
            if draw < cumulative:
                return action
        return next(reversed(probabilities))


@dataclass(frozen=True)
class SafePolicyFilter(Policy):
    policy: Policy
    is_safe: Callable[[Any, Any], bool]
    fallback_action: Any
    name: str = "safe_policy_filter"

    def action(self, state, available_actions):
        safe = tuple(a for a in available_actions if self.is_safe(state, a))
        if not safe:
            raise RuntimeError("no safe action, including the configured fallback")
        proposed = self.policy.action(state, available_actions)
        if proposed in safe:
            return proposed
        if self.fallback_action not in safe:
            raise RuntimeError("configured fallback is not safe/admissible")
        return self.fallback_action

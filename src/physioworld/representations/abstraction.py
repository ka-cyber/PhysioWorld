"""Finite task-law quotients and controlled Markov state abstraction."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

from ._numeric import matrix, probability
from .base import BaseRepresentation, RepresentationResult
from .information import discrete_key
from .standard import _spec


class QuotientRepresentation(BaseRepresentation[Any]):
    """Explicit finite equivalence classes, encoded as one-hot class identity."""

    def __init__(self, classes: Mapping[Any, Any]):
        if not classes:
            raise ValueError("quotient classes must be nonempty")
        converted = {discrete_key(k): discrete_key(v) for k, v in classes.items()}
        labels = tuple(dict.fromkeys(converted.values()))
        index = {v: i for i, v in enumerate(labels)}
        self.classes_ = tuple(
            tuple(k for k, v in converted.items() if v == label) for label in labels
        )
        self._class_index = {k: index[v] for k, v in converted.items()}
        self.dimension = len(labels)
        self.spec = _spec("quotient", self.dimension, training="declared_equivalence_relation")

    def class_index(self, history: Any) -> int:
        key = discrete_key(history)
        if key not in self._class_index:
            raise ValueError("state outside declared quotient domain")
        return self._class_index[key]

    def transform(self, history: Any) -> RepresentationResult:
        index = self.class_index(history)
        return RepresentationResult(
            tuple(float(i == index) for i in range(self.dimension)),
            self.spec,
            {"class_index": index, "evidence_class": "DEFINITION"},
        )

    def inverse_class(self, latent: int | RepresentationResult) -> tuple:
        if isinstance(latent, RepresentationResult):
            if (
                len(latent.values) != self.dimension
                or latent.values.count(1.0) != 1
                or any(v not in (0.0, 1.0) for v in latent.values)
            ):
                raise ValueError("quotient inverse requires a one-hot class")
            index = latent.values.index(1.0)
        else:
            index = latent
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.dimension:
            raise ValueError("invalid quotient class index")
        return self.classes_[index]


class MinimalSufficientRepresentation(QuotientRepresentation):
    """Coarsest deterministic quotient preserving supplied finite task laws.

    Equality is exact (no tolerance chaining). This is minimal cardinality on
    the declared finite domain and task family, not an unrestricted search
    over arbitrary latent stochastic encoders or a causal-sufficiency claim.
    Each state's signature concatenates its conditional distributions for all
    tasks. Different distributions with the same mean remain distinct.
    """

    def __init__(self, task_laws: Mapping[Any, Mapping[str, Sequence[float]]]):
        if not task_laws:
            raise ValueError("task laws must be nonempty")
        tasks = tuple(next(iter(task_laws.values())))
        if not tasks:
            raise ValueError("at least one task law is required")
        signatures: dict[Hashable, tuple[tuple[float, ...], ...]] = {}
        widths: dict[str, int] = {}
        for state, laws in task_laws.items():
            if set(laws) != set(tasks):
                raise ValueError("all states must share the declared task family")
            signature = []
            for task in tasks:
                law = probability(laws[task])
                if task in widths and widths[task] != len(law):
                    raise ValueError("task law supports must align across states")
                widths[task] = len(law)
                signature.append(law)
            signatures[discrete_key(state)] = tuple(signature)
        super().__init__(signatures)
        self.task_ids_ = tasks
        self.preserved_laws_ = tuple(signatures[group[0]] for group in self.classes_)
        self.spec = _spec(
            "minimal_sufficient_task_law_quotient",
            self.dimension,
            training="supplied_finite_conditional_laws",
            task=str(tasks),
        )

    def task_law(self, history: Any, task_id: str) -> tuple[float, ...]:
        if task_id not in self.task_ids_:
            raise ValueError("unknown task")
        return self.preserved_laws_[self.class_index(history)][self.task_ids_.index(task_id)]

    def certificate(self) -> dict:
        return {
            "scope": "finite_domain_deterministic_quotients_for_declared_task_laws",
            "minimal_classes": self.dimension,
            "domain_size": len(self._class_index),
            "preserved_tasks": self.task_ids_,
            "equality": "exact_distribution_equality",
            "evidence_class": "DEFINITION",
            "causal_sufficiency": False,
        }


def minimal_sufficient_representation(
    task_laws: Mapping[Any, Mapping[str, Sequence[float]]],
) -> MinimalSufficientRepresentation:
    return MinimalSufficientRepresentation(task_laws)


class StateAbstractionRepresentation(QuotientRepresentation):
    """Coarsest exact reward-respecting probabilistic bisimulation of a finite MDP.

    Partition refinement preserves each action's reward and class transition
    probabilities. Unlike arbitrary feature selection, the resulting quotient
    is a controlled Markov model with explicit transitions and rewards.
    Floating-point equality is exact; numerical near-equivalence is not silently
    merged. Its guarantee is relative to the supplied model and reward family.
    """

    def __init__(
        self,
        transitions: Mapping[Any, Sequence[Sequence[float]]],
        rewards: Mapping[Any, Sequence[float]],
        *,
        states: Sequence[Any] | None = None,
    ):
        if not transitions or set(transitions) != set(rewards):
            raise ValueError("matching action transitions and rewards are required")
        kernels = {a: matrix(rows) for a, rows in transitions.items()}
        n = len(next(iter(kernels.values())))
        actions = tuple(kernels)
        for action in actions:
            if len(kernels[action]) != n or len(kernels[action][0]) != n:
                raise ValueError("all action kernels must be square with equal dimensions")
            kernels[action] = tuple(probability(row, n) for row in kernels[action])
        from ._numeric import vector

        r = {a: vector(rewards[a], n) for a in actions}
        labels = tuple(range(n)) if states is None else tuple(discrete_key(s) for s in states)
        if len(labels) != n or len(set(labels)) != n:
            raise ValueError("states must be distinct and match kernels")

        def partition(signatures: Sequence[Any]) -> tuple[int, ...]:
            keys = {key: i for i, key in enumerate(dict.fromkeys(signatures))}
            return tuple(keys[key] for key in signatures)

        blocks = partition(tuple(tuple(r[a][i] for a in actions) for i in range(n)))
        iterations = 0
        while True:
            count = max(blocks) + 1
            signatures = tuple(
                (
                    blocks[i],
                    tuple(
                        tuple(
                            math.fsum(kernels[a][i][j] for j in range(n) if blocks[j] == b)
                            for b in range(count)
                        )
                        for a in actions
                    ),
                )
                for i in range(n)
            )
            refined = partition(signatures)
            iterations += 1
            if refined == blocks:
                break
            blocks = refined
        super().__init__({s: blocks[i] for i, s in enumerate(labels)})
        representatives = tuple(blocks.index(b) for b in range(self.dimension))
        self.transitions_ = {
            a: tuple(
                tuple(
                    math.fsum(kernels[a][i][j] for j in range(n) if blocks[j] == b)
                    for b in range(self.dimension)
                )
                for i in representatives
            )
            for a in actions
        }
        self.rewards_ = {a: tuple(r[a][i] for i in representatives) for a in actions}
        self.refinement_iterations_ = iterations
        self.spec = _spec(
            "markov_bisimulation_state_abstraction",
            self.dimension,
            training="declared_finite_controlled_markov_model",
            task="declared_rewards",
        )

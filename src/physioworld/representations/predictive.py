"""Observable-test predictive states realized by a finite controlled HMM."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence

from ._numeric import dot, matrix, probability, symmetric_eigen, vector
from .base import BaseRepresentation, RepresentationResult
from .standard import _spec

ObservableTest = tuple[tuple[Hashable, Hashable], ...]


class PredictiveStateRepresentation(BaseRepresentation[Sequence[tuple[Hashable, Hashable]]]):
    """A sufficient observable-test state for a declared finite HMM model.

    Timing: action a first transitions the hidden state by T_a, then emits
    observation o by O_a. A test is a sequence of (action, observation) pairs;
    its probability conditions on that fixed action sequence. The state stores
    only future test probabilities. A full-column-rank test matrix augmented
    with the constant test permits recovery of a model belief and an exact
    Bayes update, followed by projection back to the observable tests.

    This is a model-based PSR realization, not a data-learned PSR and not causal
    identification from observational data. Rank-deficient test families are
    rejected instead of claiming predictive sufficiency.
    """

    def __init__(
        self,
        transitions: Mapping[Hashable, Sequence[Sequence[float]]],
        emissions: Mapping[Hashable, Sequence[Sequence[float]]],
        initial: Sequence[float],
        *,
        observations: Sequence[Hashable] | None = None,
        tests: Sequence[Sequence[tuple[Hashable, Hashable]]] | None = None,
        rank_tolerance: float = 1e-11,
    ):
        self.initial = probability(initial)
        n = len(self.initial)
        if not transitions or set(transitions) != set(emissions):
            raise ValueError("matching nonempty action kernels are required")
        if not math.isfinite(rank_tolerance) or rank_tolerance <= 0:
            raise ValueError("rank_tolerance must be finite and positive")
        self.transitions = {}
        self.emissions = {}
        obs_width = None
        for action in transitions:
            t, e = matrix(transitions[action]), matrix(emissions[action])
            if len(t) != n or len(t[0]) != n or len(e) != n:
                raise ValueError("kernel dimensions must match initial belief")
            self.transitions[action] = tuple(probability(row, n) for row in t)
            if obs_width is None:
                obs_width = len(e[0])
            self.emissions[action] = tuple(probability(row, obs_width) for row in e)
        if obs_width is None:
            raise ValueError("emission kernels must have a nonempty observation alphabet")
        self.observations = tuple(range(obs_width)) if observations is None else tuple(observations)
        if len(self.observations) != obs_width or len(set(self.observations)) != obs_width:
            raise ValueError("distinct observation symbols must match emission width")
        self._observation_index = {o: i for i, o in enumerate(self.observations)}
        if tests is None:
            self.tests: tuple[ObservableTest, ...] = tuple(
                ((action, observation),)
                for action in self.transitions
                for observation in self.observations
            )
        else:
            normalised_tests: list[ObservableTest] = []
            for test in tests:
                normalised_test: list[tuple[Hashable, Hashable]] = []
                for pair in test:
                    if len(pair) != 2:
                        raise ValueError("each test step is an (action, observation) pair")
                    normalised_test.append((pair[0], pair[1]))
                normalised_tests.append(tuple(normalised_test))
            self.tests = tuple(normalised_tests)
        if (
            not self.tests
            or len(set(self.tests)) != len(self.tests)
            or any(not t for t in self.tests)
        ):
            raise ValueError("future observable tests must be distinct and nonempty")
        self.test_vectors_ = tuple(self._test_vector(t) for t in self.tests)
        augmented = ((1.0,) * n,) + self.test_vectors_
        gram = tuple(
            tuple(math.fsum(row[i] * row[j] for row in augmented) for j in range(n))
            for i in range(n)
        )
        eig, basis = symmetric_eigen(gram)
        self.observability_rank_ = sum(v > eig[0] * rank_tolerance for v in eig)
        if self.observability_rank_ < n:
            raise ValueError(
                "future tests do not identify a sufficient predictive state; add tests"
            )
        self._left_inverse = tuple(
            tuple(
                math.fsum(basis[k][i] * dot(basis[k], row) / eig[k] for k in range(n))
                for row in augmented
            )
            for i in range(n)
        )
        self._augmented = augmented
        self.spec = _spec(
            "predictive_state",
            len(self.tests),
            "action_observation_history",
            "declared_controlled_hmm",
            "future_observable_tests",
        )

    def _test_vector(self, test: Sequence[tuple[Hashable, Hashable]]) -> tuple[float, ...]:
        values = (1.0,) * len(self.initial)
        for pair in reversed(tuple(test)):
            if len(pair) != 2:
                raise ValueError("each test step is an (action, observation) pair")
            a, o = pair
            if a not in self.transitions or o not in self._observation_index:
                raise ValueError("unknown action or observation")
            index = self._observation_index[o]
            weighted = tuple(self.emissions[a][j][index] * values[j] for j in range(len(values)))
            values = tuple(dot(row, weighted) for row in self.transitions[a])
        return values

    def from_belief(self, belief: Sequence[float]) -> RepresentationResult:
        b = probability(belief, len(self.initial))
        return RepresentationResult(
            tuple(dot(row, b) for row in self.test_vectors_),
            self.spec,
            {
                "tests": self.tests,
                "evidence_class": "MODEL",
                "model": "finite_controlled_hmm",
                "predictively_sufficient_under_model": True,
                "causal_identification": False,
            },
        )

    def belief_from_state(self, state: Sequence[float] | RepresentationResult) -> tuple[float, ...]:
        values = vector(
            state.values if isinstance(state, RepresentationResult) else state, len(self.tests)
        )
        if min(values) < -1e-10 or max(values) > 1 + 1e-10:
            raise ValueError("test probabilities must lie in [0, 1]")
        augmented = (1.0,) + values
        b = tuple(dot(row, augmented) for row in self._left_inverse)
        residual = max(abs(dot(row, b) - v) for row, v in zip(self._augmented, augmented))
        if min(b) < -1e-9 or residual > 1e-9 or abs(math.fsum(b) - 1.0) > 1e-9:
            raise ValueError("state is inconsistent with the declared predictive model")
        clipped = tuple(max(v, 0.0) for v in b)
        return tuple(v / math.fsum(clipped) for v in clipped)

    def initial_state(self) -> RepresentationResult:
        return self.from_belief(self.initial)

    def update(
        self, state: Sequence[float] | RepresentationResult, action: Hashable, observation: Hashable
    ) -> RepresentationResult:
        if action not in self.transitions or observation not in self._observation_index:
            raise ValueError("unknown action or observation")
        b = self.belief_from_state(state)
        n, index = len(b), self._observation_index[observation]
        prior = tuple(
            math.fsum(b[i] * self.transitions[action][i][j] for i in range(n)) for j in range(n)
        )
        mass = tuple(prior[j] * self.emissions[action][j][index] for j in range(n))
        likelihood = math.fsum(mass)
        if likelihood <= 0:
            raise ValueError("observation is impossible under the current predictive state")
        result = self.from_belief(tuple(v / likelihood for v in mass))
        return RepresentationResult(
            result.values, result.spec, {**result.metadata, "observation_likelihood": likelihood}
        )

    def predict_test(
        self,
        state: Sequence[float] | RepresentationResult,
        test: Sequence[tuple[Hashable, Hashable]],
    ) -> float:
        return dot(self._test_vector(test), self.belief_from_state(state))

    def predict_observation(
        self, state: Sequence[float] | RepresentationResult, action: Hashable
    ) -> tuple[float, ...]:
        return tuple(self.predict_test(state, ((action, o),)) for o in self.observations)

    def transform(self, history: Sequence[tuple[Hashable, Hashable]]) -> RepresentationResult:
        state = self.initial_state()
        for pair in history:
            if len(pair) != 2:
                raise ValueError("history must contain (action, observation) pairs")
            state = self.update(state, *pair)
        return state

    def predictive_sufficiency(self) -> dict:
        """Numerical observability certificate under the supplied model assumptions."""
        n = len(self.initial)
        residual = max(
            abs(
                math.fsum(
                    self._left_inverse[i][k] * self._augmented[k][j]
                    for k in range(len(self._augmented))
                )
                - float(i == j)
            )
            for i in range(n)
            for j in range(n)
        )
        return {
            "sufficient_under_model": True,
            "observability_rank": self.observability_rank_,
            "hidden_dimension": n,
            "left_inverse_residual": residual,
            "evidence_class": "NUMERICAL_VERIFICATION",
            "assumptions": (
                "correct finite controlled HMM",
                "declared action/observation timing",
                "test observability",
                "history available by prediction time",
            ),
            "causal_identification": False,
        }

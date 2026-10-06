"""Finite adversarial search, negative controls, and competing-mechanism design."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Counterexample:
    counterexample_id: str
    claim_id: str
    requirement_ids: tuple[str, ...]
    reference_test_ids: tuple[str, ...]
    parameters: tuple[float, ...]
    observed: float
    expected_upper_bound: float
    explanation: str
    evidence_class: str = "NUMERICAL_VERIFICATION"


@dataclass(frozen=True)
class FalsificationResult:
    template_id: str
    claim_id: str
    evaluations: tuple[tuple[tuple[float, ...], float], ...]
    counterexamples: tuple[Counterexample, ...]
    status: str
    domain: str


@dataclass(frozen=True)
class FalsificationTemplate:
    template_id: str
    claim_id: str
    expected_result: str
    alternative: str
    failure_mode: str
    adversarial_test: str
    domain: str
    parameters: tuple[tuple[float, ...], ...]
    statistic: Callable[[tuple[float, ...]], float]
    expected_upper_bound: float
    requirement_ids: tuple[str, ...]
    reference_test_ids: tuple[str, ...]

    def run(self) -> FalsificationResult:
        if (
            not self.parameters
            or not math.isfinite(self.expected_upper_bound)
            or not self.requirement_ids
            or not self.reference_test_ids
            or any(
                not s
                for s in (
                    self.template_id,
                    self.claim_id,
                    self.expected_result,
                    self.alternative,
                    self.failure_mode,
                    self.adversarial_test,
                    self.domain,
                )
            )
        ):
            raise ValueError("complete falsification template and finite search required")
        evaluations, counterexamples = [], []
        for i, parameters in enumerate(self.parameters):
            if any(not math.isfinite(v) for v in parameters):
                raise ValueError("finite search parameters required")
            observed = self.statistic(parameters)
            if not math.isfinite(observed):
                raise ValueError("nonfinite falsification statistic")
            evaluations.append((parameters, observed))
            if observed > self.expected_upper_bound:
                counterexamples.append(
                    Counterexample(
                        f"{self.template_id}:{i}",
                        self.claim_id,
                        self.requirement_ids,
                        self.reference_test_ids,
                        parameters,
                        observed,
                        self.expected_upper_bound,
                        self.failure_mode,
                    )
                )
        return FalsificationResult(
            self.template_id,
            self.claim_id,
            tuple(evaluations),
            tuple(counterexamples),
            "FALSIFIED" if counterexamples else "SURVIVED_FINITE_TESTED_DOMAIN",
            self.domain,
        )


class CounterexampleRegistry:
    def __init__(self) -> None:
        self.templates: dict[str, FalsificationTemplate] = {}

    def register(self, template: FalsificationTemplate) -> None:
        if template.template_id in self.templates:
            raise ValueError("duplicate falsification template ID")
        self.templates[template.template_id] = template

    def run(self) -> tuple[FalsificationResult, ...]:
        return tuple(template.run() for template in self.templates.values())


def permutation_control(values: Sequence[T], *, seed: int = 0) -> tuple[T, ...]:
    """Seeded label/semantic/temporal permutation; chance identity is not suppressed."""
    if not values:
        raise ValueError("nonempty negative-control data required")
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return tuple(shuffled)


@dataclass(frozen=True)
class NegativeControlResult:
    kind: str
    original_statistic: float
    control_statistic: float
    change: float
    seed: int
    status: str = "DIAGNOSTIC_NOT_IDENTIFICATION"


def permutation_diagnostic(
    values: Sequence[T], statistic: Callable[[Sequence[T]], float], *, kind: str, seed: int = 0
) -> NegativeControlResult:
    if kind not in (
        "semantic_permutation",
        "label_permutation",
        "temporal_shuffle",
        "channel_mismatch",
    ):
        raise ValueError("unknown permutation control")
    original = statistic(values)
    control = statistic(permutation_control(values, seed=seed))
    if not math.isfinite(original) or not math.isfinite(control):
        raise ValueError("finite negative-control statistics required")
    return NegativeControlResult(kind, original, control, control - original, seed)


def predictive_causal_counterexample() -> FalsificationTemplate:
    """Two SCMs: A=U,Y=A versus A=U,Y=U, with identical observational laws."""

    def discrepancy(parameters: tuple[float, ...]) -> float:
        prevalence = parameters[0]
        observational_a = ((0, 0, 1 - prevalence), (1, 1, prevalence))
        observational_b = ((0, 0, 1 - prevalence), (1, 1, prevalence))
        if observational_a != observational_b:
            raise AssertionError("counterexample observational laws differ")
        do_a_one = 1.0
        do_b_one = prevalence
        do_a_zero = 0.0
        do_b_zero = prevalence
        return abs((do_a_one - do_a_zero) - (do_b_one - do_b_zero))

    return FalsificationTemplate(
        "predictive-not-causal",
        "observational-prediction-identifies-intervention",
        "equal observational laws imply equal interventional effects",
        "hidden confounding produces distinct intervention laws",
        "perfect observational prediction can coexist with zero intervention effect",
        "compare confounded SCMs under do(A)",
        "binary U with tested prevalences",
        ((0.2,), (0.5,), (0.8,)),
        discrepancy,
        0.0,
        ("causal-boundary",),
        ("synthetic-scm-identification",),
    )


@dataclass(frozen=True)
class DesignCandidate:
    candidate_id: str
    kind: str
    parameters: tuple[tuple[str, float], ...]
    cost: float
    safe: bool


@dataclass(frozen=True)
class Mechanism:
    mechanism_id: str
    prediction: Callable[[DesignCandidate], Sequence[float]]
    explanation: str


def _probabilities(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(values)
    if (
        not result
        or any(not math.isfinite(p) or p < 0 or p > 1 for p in result)
        or not math.isclose(sum(result), 1.0, abs_tol=1e-10)
    ):
        raise ValueError("normalized finite mechanism probabilities required")
    return result


def _prediction_matrix(
    mechanisms: Sequence[Mechanism], candidate: DesignCandidate
) -> tuple[tuple[float, ...], ...]:
    if (
        not mechanisms
        or len({m.mechanism_id for m in mechanisms}) != len(mechanisms)
        or any(not m.mechanism_id or not m.explanation for m in mechanisms)
    ):
        raise ValueError("unique declared mechanisms required")
    matrix = tuple(_probabilities(m.prediction(candidate)) for m in mechanisms)
    if any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("mechanisms must predict the same finite outcome space")
    return matrix


def expected_information_gain(
    mechanisms: Sequence[Mechanism], prior: Sequence[float], candidate: DesignCandidate
) -> float:
    """Exact finite mutual information I(mechanism; observation | candidate), in nats."""
    weights = _probabilities(prior)
    matrix = _prediction_matrix(mechanisms, candidate)
    if len(weights) != len(matrix):
        raise ValueError("one prior probability per mechanism required")
    mixture = [sum(w * row[j] for w, row in zip(weights, matrix)) for j in range(len(matrix[0]))]
    return sum(
        w * p * math.log(p / mixture[j])
        for w, row in zip(weights, matrix)
        for j, p in enumerate(row)
        if w > 0 and p > 0
    )


@dataclass(frozen=True)
class DesignSearchResult:
    selected: DesignCandidate
    objective: str
    scores: tuple[tuple[str, float], ...]
    rejected_candidates: tuple[str, ...]
    status: str = "OPTIMAL_OVER_ENUMERATED_FEASIBLE_CANDIDATES"


def finite_design_search(
    candidates: Sequence[DesignCandidate],
    *,
    budget: float,
    objective: Callable[[DesignCandidate], float],
    objective_name: str,
    feasible: Callable[[DesignCandidate], bool] = lambda candidate: True,
) -> DesignSearchResult:
    """Evaluate the actual objective at every finite feasible candidate, then maximize."""
    if (
        not candidates
        or not objective_name
        or not math.isfinite(budget)
        or budget < 0
        or len({c.candidate_id for c in candidates}) != len(candidates)
    ):
        raise ValueError("unique finite candidate list, objective, and budget required")
    scores, rejected, accepted = [], [], []
    for candidate in candidates:
        if (
            not candidate.candidate_id
            or not candidate.kind
            or not math.isfinite(candidate.cost)
            or candidate.cost < 0
            or any(not name or not math.isfinite(value) for name, value in candidate.parameters)
        ):
            raise ValueError("invalid design candidate")
        if not candidate.safe or candidate.cost > budget or not feasible(candidate):
            rejected.append(candidate.candidate_id)
            continue
        score = objective(candidate)
        if not math.isfinite(score):
            raise ValueError("design objective must be finite")
        accepted.append(candidate)
        scores.append((candidate.candidate_id, score))
    if not accepted:
        raise ValueError("no feasible design candidate")
    index = max(range(len(scores)), key=lambda i: scores[i][1])
    return DesignSearchResult(accepted[index], objective_name, tuple(scores), tuple(rejected))


@dataclass(frozen=True)
class MechanismComparison:
    mechanism_ids: tuple[str, ...]
    log_likelihoods: tuple[float, ...]
    posterior_probabilities: tuple[float, ...]
    status: str = "FINITE_MODEL_COMPARISON_NOT_MECHANISM_PROOF"


def compare_mechanisms(
    mechanisms: Sequence[Mechanism],
    prior: Sequence[float],
    observations: Sequence[tuple[DesignCandidate, int]],
) -> MechanismComparison:
    weights = _probabilities(prior)
    if not observations or len(weights) != len(mechanisms):
        raise ValueError("observations and aligned mechanism prior required")
    log_likelihoods = [0.0] * len(mechanisms)
    for candidate, outcome in observations:
        matrix = _prediction_matrix(mechanisms, candidate)
        if not isinstance(outcome, int) or not 0 <= outcome < len(matrix[0]):
            raise ValueError("outcome outside declared outcome space")
        for i, row in enumerate(matrix):
            log_likelihoods[i] += math.log(row[outcome]) if row[outcome] > 0 else -math.inf
    log_posteriors = [
        likelihood + (math.log(w) if w > 0 else -math.inf)
        for likelihood, w in zip(log_likelihoods, weights)
    ]
    maximum = max(log_posteriors)
    if maximum == -math.inf:
        raise ValueError("all mechanisms assign zero probability to observed evidence")
    unnormalized = [math.exp(v - maximum) for v in log_posteriors]
    total = sum(unnormalized)
    return MechanismComparison(
        tuple(m.mechanism_id for m in mechanisms),
        tuple(log_likelihoods),
        tuple(v / total for v in unnormalized),
    )

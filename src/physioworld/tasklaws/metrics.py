"""Metrics with explicit estimand-sensitive behavior."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Mapping, Sequence

from physioworld.core.contracts import EstimandFirewall, EvidenceClass, MetricContext, finite
from physioworld.core.evidence import AssumptionSet
from physioworld.evidence_origin import require_verified_artifact_metadata

from .models import (
    BootstrapResult,
    ConditionalCell,
    DiscreteDistribution,
    TaskDefinition,
    TaskKind,
    TaskLaw,
    TaskLawComparison,
)


def empirical_conditional_mean(
    values: Sequence[float], groups: Sequence[object]
) -> dict[object, float]:
    """Estimate `E[value | group]` without pretending groups are continuous laws."""
    if len(values) != len(groups) or len(values) == 0:
        raise ValueError("values and groups must have equal non-zero length")
    grouped: dict[object, list[float]] = {}
    for value, group in zip(values, groups):
        finite(value, "conditional mean observation")
        grouped.setdefault(group, []).append(float(value))
    return {group: sum(items) / len(items) for group, items in grouped.items()}


def empirical_conditional_distribution(
    values: Sequence[float], groups: Sequence[object]
) -> tuple[ConditionalCell, ...]:
    """Finite empirical laws with all atoms/masses retained for each observed stratum."""
    if len(values) != len(groups) or len(values) == 0:
        raise ValueError("values and groups must have equal nonzero length")
    grouped: dict[object, list[float]] = {}
    for value, group in zip(values, groups):
        finite(value, "conditional law observation")
        grouped.setdefault(group, []).append(float(value))
    cells = []
    for group, observations in grouped.items():
        support = tuple(sorted(set(observations)))
        probabilities = tuple(observations.count(value) / len(observations) for value in support)
        cells.append(
            ConditionalCell(group, DiscreteDistribution(support, probabilities), len(observations))
        )
    return tuple(cells)


def conditional_task_law(
    full_values: Sequence[float],
    representation_groups: Sequence[object],
    task: TaskDefinition,
    *,
    history_groups: Sequence[object] | None = None,
    task_function: Callable[[float], float] | None = None,
    task_function_name: str = "identity",
    context: MetricContext | None = None,
    evidence_class: EvidenceClass | None = None,
    evidence_metadata: Mapping[str, Any] | None = None,
) -> TaskLaw:
    """Estimate empirical Q(Y|Z), Q(Y|H) and their declared task expectations.

    Without history_groups, each row is an empirical history atom (a degenerate
    in-sample reference law). This is not an out-of-sample sufficiency claim.
    Grouping continuous covariates must be declared externally; no hidden bins.
    """
    task.validate()
    if context is not None:
        EstimandFirewall.require_metric(
            context,
            "task_law_discrepancy",
            "sup_task_mean_absolute_conditional_expectation_difference",
        )
        if task.task_id not in context.task_family or context.time_horizon != task.horizon:
            raise ValueError("task-law context does not match the declared task/horizon")
    resolved_evidence = EvidenceClass(
        evidence_class
        if evidence_class is not None
        else context.evidence_class
        if context is not None
        else EvidenceClass.NUMERICAL_VERIFICATION
    )
    if context is not None and resolved_evidence != context.evidence_class:
        raise ValueError("task-law evidence class must propagate from its metric context")
    require_verified_artifact_metadata(resolved_evidence, evidence_metadata)
    if task_function is not None and task_function_name == "identity":
        raise ValueError("a nonidentity task function requires a declared task_function_name")
    cells = empirical_conditional_distribution(full_values, representation_groups)
    histories = tuple(range(len(full_values))) if history_groups is None else tuple(history_groups)
    full_cells = empirical_conditional_distribution(full_values, histories)
    conditional = {cell.group: cell.distribution.expectation(task_function) for cell in cells}
    full = {cell.group: cell.distribution.expectation(task_function) for cell in full_cells}
    return TaskLaw(
        task,
        tuple(full[group] for group in histories),
        tuple(conditional[group] for group in representation_groups),
        tuple(representation_groups),
        (
            "observed_strata_empirical_law",
            "in_sample_plugin_estimate",
            "row_history_atoms" if history_groups is None else "declared_history_strata",
        ),
        cells,
        full_cells,
        task_function_name,
        "empirical_conditional_distribution",
        context,
        resolved_evidence,
        dict(evidence_metadata or {}),
    )


def task_function_discrepancy(
    full_laws: Sequence[DiscreteDistribution],
    representation_laws: Sequence[DiscreteDistribution],
    task_functions: Mapping[str, Callable[[float], float]],
) -> TaskLawComparison:
    """sup_g E |E[g(Y)|H] - E[g(Y)|Z]| for a finite declared function family."""
    if len(full_laws) != len(representation_laws) or len(full_laws) == 0 or not task_functions:
        raise ValueError("paired laws and task family must be nonempty")
    losses = {
        name: tuple(
            abs(full.expectation(function) - reduced.expectation(function))
            for full, reduced in zip(full_laws, representation_laws)
        )
        for name, function in task_functions.items()
    }
    return TaskLawComparison(
        {name: sum(items) / len(items) for name, items in losses.items()}, "max", losses
    )


def kernel_conditional_distribution(
    outcomes: Sequence[float],
    covariates: Sequence[Sequence[float]],
    queries: Sequence[Sequence[float]],
    *,
    bandwidth: float,
) -> tuple[DiscreteDistribution, ...]:
    """Gaussian-kernel conditional empirical laws on observed outcome atoms.

    The bandwidth and Euclidean covariate scale are caller declarations. This
    nonparametric estimator is not a calibrated forecast without evaluation.
    """
    finite(bandwidth, "kernel bandwidth")
    if (
        bandwidth <= 0
        or len(outcomes) == 0
        or len(outcomes) != len(covariates)
        or len(queries) == 0
    ):
        raise ValueError("conditional kernel needs observations/queries and positive bandwidth")
    dimension = len(covariates[0])
    if dimension == 0 or any(len(row) != dimension for row in (*covariates, *queries)):
        raise ValueError("conditional covariates/queries must share a nonempty dimension")
    for value in outcomes:
        finite(value, "kernel outcome")
    for row in (*covariates, *queries):
        for value in row:
            finite(value, "kernel covariate")
    covariates = tuple(tuple(float(value) for value in row) for row in covariates)
    queries = tuple(tuple(float(value) for value in row) for row in queries)
    outcomes = tuple(float(value) for value in outcomes)
    support = tuple(sorted(set(outcomes)))
    laws = []
    for query in queries:
        log_weights = tuple(
            -0.5 * sum(((left - right) / bandwidth) ** 2 for left, right in zip(query, row))
            for row in covariates
        )
        peak = max(log_weights)
        if not math.isfinite(peak):
            raise ValueError("kernel distances overflow; rescale covariates or bandwidth")
        weights = tuple(math.exp(value - peak) for value in log_weights)
        total = sum(weights)
        probabilities = tuple(
            sum(weight for outcome, weight in zip(outcomes, weights) if outcome == atom) / total
            for atom in support
        )
        laws.append(DiscreteDistribution(support, probabilities))
    return tuple(laws)


def clustered_bootstrap(
    sample_count: int,
    estimator: Callable[[tuple[int, ...]], float],
    *,
    clusters: Sequence[object] | None = None,
    repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION,
    evidence_metadata: Mapping[str, Any] | None = None,
) -> BootstrapResult:
    """Resample whole subjects/clusters and re-run the caller's estimator.

    The point estimate and bootstrap mean are separate estimands. A callback
    allows refitting conditional laws and independently optimizing policies in
    every replicate rather than bootstrapping a stale optimized point value.
    """
    if sample_count < 1 or repetitions < 2 or not 0 < confidence < 1:
        raise ValueError("bootstrap needs observations, >=2 replicates, and confidence in (0,1)")
    resolved_evidence = EvidenceClass(evidence_class)
    require_verified_artifact_metadata(resolved_evidence, evidence_metadata)
    if clusters is not None and len(clusters) != sample_count:
        raise ValueError("cluster labels must match the observation count")
    grouped: dict[object, list[int]] = {}
    for index, cluster in enumerate(range(sample_count) if clusters is None else clusters):
        grouped.setdefault(cluster, []).append(index)
    units = tuple(tuple(indices) for indices in grouped.values())
    rng = random.Random(seed)

    def estimate(indices: tuple[int, ...]) -> float:
        value = estimator(indices)
        finite(value, "bootstrap estimate")
        return float(value)

    point = estimate(tuple(range(sample_count)))
    replicates = tuple(
        estimate(tuple(index for _ in units for index in rng.choice(units)))
        for _ in range(repetitions)
    )
    ordered = sorted(replicates)

    def quantile(p: float) -> float:
        location = (len(ordered) - 1) * p
        low, high = math.floor(location), math.ceil(location)
        return ordered[low] + (ordered[high] - ordered[low]) * (location - low)

    tail = (1 - confidence) / 2
    return BootstrapResult(
        point,
        sum(replicates) / len(replicates),
        quantile(tail),
        quantile(1 - tail),
        confidence,
        replicates,
        "percentile_refit",
        "cluster" if clusters is not None else "observation",
        seed,
        resolved_evidence,
        dict(evidence_metadata or {}),
    )


def binary_optimal_accuracy(probabilities: Sequence[float]) -> float:
    """Bayes accuracy `1/2 + E|P(Y=1|I)-1/2|`."""
    if len(probabilities) == 0:
        raise ValueError("probabilities cannot be empty")
    for p in probabilities:
        finite(p, "binary probability")
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("probabilities must lie in [0, 1]")
    return 0.5 + sum(abs(float(p) - 0.5) for p in probabilities) / len(probabilities)


def binary_forecasting_certificate(
    full_probabilities: Sequence[float],
    representation_probabilities: Sequence[float],
    *,
    task: TaskDefinition | None = None,
    assumptions: Sequence[str] | AssumptionSet = (),
) -> dict[str, float | bool | str | tuple[str, ...]]:
    """Return the forecasting certificate only when assumptions are declared.

    This function intentionally rejects intervention and policy tasks because
    the predictive information-loss inequality does not imply a causal or
    control bound.
    """
    if len(full_probabilities) != len(representation_probabilities) or len(full_probabilities) == 0:
        raise ValueError("probability arrays must be non-empty and equal length")
    if task and task.kind in {
        TaskKind.INTERVENTIONAL_OUTCOME,
        TaskKind.POLICY_EVALUATION,
        TaskKind.DECISION,
        TaskKind.SEQUENTIAL_DECISION,
        TaskKind.SEQUENTIAL_INTERVENTION,
    }:
        raise ValueError(
            "binary forecasting certificate is not an intervention/control certificate"
        )
    required = {"binary_outcome", "same_target", "same_population", "representation_is_measurable"}
    if isinstance(assumptions, AssumptionSet):
        required.add("correct_conditional_laws")
        assumptions.require(tuple(sorted(required)))
        names = tuple(assumption.assumption_id for assumption in assumptions.assumptions)
    else:
        names = tuple(assumptions)
    missing = required.difference(names)
    if missing:
        raise ValueError(f"missing certificate assumptions: {sorted(missing)}")
    ah = binary_optimal_accuracy(full_probabilities)
    az = binary_optimal_accuracy(representation_probabilities)
    discrepancy = sum(
        abs(float(a) - float(b)) for a, b in zip(full_probabilities, representation_probabilities)
    ) / len(full_probabilities)
    gap = ah - az
    return {
        "A_H_star": ah,
        "A_Z_star": az,
        "gap": gap,
        "D_tau": discrepancy,
        "valid": 0 <= gap <= discrepancy + 1e-12,
        "assumptions": names,
        "evidence_class": "NUMERICAL_VERIFICATION",
        "assumption_status": "guarded"
        if isinstance(assumptions, AssumptionSet)
        else "declared_only",
        "high_confidence": isinstance(assumptions, AssumptionSet),
        "scope": "binary predictive Bayes-accuracy inequality; not causal validation",
    }


def total_task_discrepancy(tasks: Sequence[TaskLaw]) -> TaskLawComparison:
    if len({law.task.task_id for law in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique")
    values = {law.task.task_id: law.discrepancy() for law in tasks}
    return TaskLawComparison(values, aggregation="max")


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q) or len(p) == 0:
        raise ValueError("distributions must have equal non-zero length")
    _validate_probabilities(p)
    _validate_probabilities(q)
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(p, q))


def _validate_probabilities(probabilities: Sequence[float]) -> None:
    for p in probabilities:
        finite(p, "probability")
    if (
        len(probabilities) == 0
        or any(p < 0 or p > 1 for p in probabilities)
        or abs(sum(float(p) for p in probabilities) - 1) > 1e-12
    ):
        raise ValueError("probabilities must be normalized and nonnegative")


def wasserstein_1d(
    p: Sequence[float], q: Sequence[float], values: Sequence[float] | None = None
) -> float:
    if len(p) != len(q) or len(p) == 0:
        raise ValueError("distributions must have equal non-zero length")
    if values is None:
        values = tuple(float(i) for i in range(len(p)))
    if len(values) != len(p):
        raise ValueError("support and distribution sizes differ")
    _validate_probabilities(p)
    _validate_probabilities(q)
    for value in values:
        finite(value, "Wasserstein support")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("Wasserstein support must be strictly increasing")
    cumulative = 0.0
    distance = 0.0
    for i in range(len(p) - 1):
        cumulative += float(p[i]) - float(q[i])
        distance += abs(cumulative) * (float(values[i + 1]) - float(values[i]))
    return distance


def expected_calibration_error(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> float:
    if len(probabilities) != len(outcomes) or len(probabilities) == 0:
        raise ValueError("probabilities and outcomes must have equal non-zero length")
    if bins < 1:
        raise ValueError("bins must be positive")
    binary_optimal_accuracy(probabilities)  # validate finite probabilities
    if any(outcome not in (0, 1) for outcome in outcomes):
        raise ValueError("calibration outcomes must be binary")
    total = 0.0
    for bucket in range(bins):
        members = [i for i, p in enumerate(probabilities) if min(bins - 1, int(p * bins)) == bucket]
        if members:
            confidence = sum(float(probabilities[i]) for i in members) / len(members)
            accuracy = sum(float(outcomes[i]) for i in members) / len(members)
            total += len(members) / len(probabilities) * abs(confidence - accuracy)
    return total


def decision_value(
    utilities: Sequence[float], actions: Sequence[object], *, aggregation: str = "mean"
) -> float:
    if len(utilities) != len(actions) or len(utilities) == 0:
        raise ValueError("utilities and actions must have equal non-zero length")
    for value in utilities:
        finite(value, "utility")
    utilities = tuple(float(value) for value in utilities)
    if aggregation == "mean":
        return sum(utilities) / len(utilities)
    if aggregation == "sum":
        return sum(utilities)
    if aggregation == "minimum":
        return min(utilities)
    raise ValueError("unsupported utility aggregation")


def forecasting_capability(
    task_values: Sequence[float], *, threshold: float | None = None
) -> dict[str, float]:
    if len(task_values) == 0:
        raise ValueError("task_values cannot be empty")
    for value in task_values:
        finite(value, "task value")
    task_values = tuple(float(value) for value in task_values)
    if threshold is not None:
        finite(threshold, "capability threshold")
    result = {"mean": sum(task_values) / len(task_values), "worst": min(task_values)}
    if threshold is not None:
        result["fraction_above_threshold"] = sum(value >= threshold for value in task_values) / len(
            task_values
        )
    return result


def conditional_coverage(
    lower: Sequence[float],
    upper: Sequence[float],
    outcomes: Sequence[float],
    groups: Sequence[object],
) -> dict[object, float]:
    if len(outcomes) == 0 or not (len(lower) == len(upper) == len(outcomes) == len(groups)):
        raise ValueError("coverage arrays must have equal nonzero length")
    covered = []
    for low, high, outcome in zip(lower, upper, outcomes):
        for value in (low, high, outcome):
            finite(value, "coverage value")
        if low > high:
            raise ValueError("coverage interval is reversed")
        covered.append(float(low <= outcome <= high))
    return empirical_conditional_mean(covered, groups)

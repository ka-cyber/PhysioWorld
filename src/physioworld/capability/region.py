"""Capability and PIR accounting objects."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations, product
from typing import Iterable, Mapping, Sequence

from physioworld.core.contracts import EstimandFirewall, MetricContext, VersionedRecord, finite


def _vector(values: Sequence[float], dimension: int, name: str) -> tuple[float, ...]:
    if len(values) != dimension:
        raise ValueError(f"{name} has wrong dimension")
    for value in values:
        finite(value, name)
    return tuple(float(value) for value in values)


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> tuple[float, ...] | None:
    """Small pivoted linear solve used by finite convex-feasibility enumeration."""
    size = len(rhs)
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda i: abs(rows[i][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for index in range(size):
            if index != column:
                factor = rows[index][column]
                rows[index] = [
                    left - factor * right for left, right in zip(rows[index], rows[column])
                ]
    return tuple(row[-1] for row in rows)


@dataclass(frozen=True)
class AttainabilityResult(VersionedRecord):
    attainable: bool
    requirement: tuple[float, ...]
    policy_weights: Mapping[str, float]
    attained_value: tuple[float, ...] | None
    randomized: bool
    downward_closure: bool
    tolerance: float


@dataclass(frozen=True)
class CapabilityRegion(VersionedRecord):
    """Finite policy image `{V(pi)}` with policy IDs retained for audit."""

    policy_values: Mapping[str, tuple[float, ...]]
    representation: str
    randomized_policies: bool = False
    task_labels: tuple[str, ...] = ()
    context: MetricContext | None = None
    declared_dimension: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_values", {key: tuple(value) for key, value in self.policy_values.items()}
        )
        dimensions = {len(v) for v in self.policy_values.values()}
        if len(dimensions) > 1:
            raise ValueError("capability vectors must have equal dimension")
        if self.declared_dimension is not None:
            if (
                isinstance(self.declared_dimension, bool)
                or not isinstance(self.declared_dimension, int)
                or self.declared_dimension < 0
            ):
                raise ValueError("declared capability dimension must be a nonnegative integer")
            if dimensions and next(iter(dimensions)) != self.declared_dimension:
                raise ValueError("declared dimension disagrees with capability vectors")
        if self.policy_values and (0 in dimensions or any(not key for key in self.policy_values)):
            raise ValueError("capability policies need IDs and nonempty vectors")
        for vector in self.policy_values.values():
            for value in vector:
                finite(value, "capability value")
        object.__setattr__(
            self,
            "policy_values",
            {
                key: tuple(float(value) for value in vector)
                for key, vector in self.policy_values.items()
            },
        )
        if self.task_labels and (
            len(self.task_labels) != self.dimension or len(set(self.task_labels)) != self.dimension
        ):
            raise ValueError("task labels must be unique and match capability dimension")

    @property
    def dimension(self) -> int:
        if self.policy_values:
            return len(next(iter(self.policy_values.values())))
        if self.declared_dimension is not None:
            return self.declared_dimension
        return len(self.task_labels) or (len(self.context.task_family) if self.context else 0)

    def support(self, direction: Sequence[float]) -> tuple[float, str | None]:
        direction = _vector(direction, self.dimension, "support direction")
        if not self.policy_values:
            return (float("-inf"), None)
        best_id, best_value = max(
            self.policy_values.items(),
            key=lambda item: sum(q * v for q, v in zip(direction, item[1])),
        )
        return sum(q * v for q, v in zip(direction, best_value)), best_id

    def coordinatewise_max(self) -> tuple[float, ...]:
        return tuple(
            max(values[i] for values in self.policy_values.values()) for i in range(self.dimension)
        )

    def average_policy_value(self) -> tuple[float, ...]:
        if not self.policy_values:
            return ()
        return tuple(
            sum(v[i] for v in self.policy_values.values()) / len(self.policy_values)
            for i in range(self.dimension)
        )

    def pareto_front(self) -> "CapabilityRegion":
        def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
            return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

        def convex_dominated(policy: str, value: tuple[float, ...]) -> bool:
            other = CapabilityRegion(
                {key: vector for key, vector in self.policy_values.items() if key != policy},
                self.representation,
                randomized_policies=True,
            )
            if not other.policy_values:
                return False
            # Numerical strict Pareto dominance at the documented 1e-9 margin.
            return any(
                other.attainability(
                    tuple(
                        v + (1e-9 if index == coordinate else 0.0) for index, v in enumerate(value)
                    ),
                    tolerance=1e-11,
                ).attainable
                for coordinate in range(self.dimension)
            )

        policies = {
            policy: value
            for policy, value in self.policy_values.items()
            if not any(dominates(other, value) for other in self.policy_values.values())
            and not (self.randomized_policies and convex_dominated(policy, value))
        }
        return CapabilityRegion(
            policies,
            self.representation,
            self.randomized_policies,
            self.task_labels,
            self.context,
            self.dimension,
        )

    def attainability(
        self,
        requirement: Sequence[float],
        *,
        downward_closure: bool = True,
        tolerance: float = 1e-10,
    ) -> AttainabilityResult:
        """Finite policy feasibility, optionally convexified by ex ante randomization.

        Randomized policies concern expected utility coordinates. They do not
        imply that simultaneous per-realization safety constraints are met.
        The dependency-free active-set enumeration is intended for small finite
        policy images; no coordinatewise-optimum shortcut is used.
        """
        target = _vector(requirement, self.dimension, "requirement")
        finite(tolerance, "attainability tolerance")
        if tolerance < 0:
            raise ValueError("attainability tolerance must be nonnegative")

        def meets(value: Sequence[float]) -> bool:
            return all(
                v >= r - tolerance if downward_closure else abs(v - r) <= tolerance
                for v, r in zip(value, target)
            )

        for policy, value in self.policy_values.items():
            if meets(value):
                return AttainabilityResult(
                    True, target, {policy: 1.0}, value, False, downward_closure, tolerance
                )
        if self.randomized_policies:
            # A feasible polytope vertex has k positive weights and at least
            # k-1 active coordinate constraints, with k <= dimension+1.
            items = tuple(self.policy_values.items())
            for count in range(2, min(len(items), self.dimension + 1) + 1):
                for selected in combinations(items, count):
                    for active in combinations(range(self.dimension), count - 1):
                        matrix = [[1.0] * count] + [
                            [value[i] for _, value in selected] for i in active
                        ]
                        weights = _solve(matrix, [1.0] + [target[i] for i in active])
                        if weights is None or any(weight < -tolerance for weight in weights):
                            continue
                        # Remove numerical negative weights and renormalize.
                        positive = tuple(max(0.0, weight) for weight in weights)
                        total = sum(positive)
                        normalized = tuple(weight / total for weight in positive)
                        value = tuple(
                            sum(
                                weight * vector[i]
                                for weight, (_, vector) in zip(normalized, selected)
                            )
                            for i in range(self.dimension)
                        )
                        if meets(value):
                            return AttainabilityResult(
                                True,
                                target,
                                {
                                    policy: weight
                                    for weight, (policy, _) in zip(normalized, selected)
                                    if weight > 0
                                },
                                value,
                                True,
                                downward_closure,
                                tolerance,
                            )
        return AttainabilityResult(
            False, target, {}, None, self.randomized_policies, downward_closure, tolerance
        )

    def jointly_attainable(
        self, requirement: Sequence[float], *, downward_closure: bool = True
    ) -> bool:
        return self.attainability(requirement, downward_closure=downward_closure).attainable

    def requirement_region(self) -> "RequirementRegion":
        return RequirementRegion(self)

    def worst_task(self, policy: str) -> float:
        return min(self.policy_values[policy])

    def dominates(self, other: "CapabilityRegion", *, tolerance: float = 1e-10) -> bool:
        """Numerical inclusion of downward requirement regions, not resource coordinates.

        Convex-to-nonconvex inclusion partitions coordinate space into cells at
        the deterministic upper-orthant boundaries and checks convex feasibility
        in uncovered cells. This avoids the incorrect vertex-only inclusion test.
        Like attainability enumeration, this is intended for small finite images.
        """
        finite(tolerance, "dominance tolerance")
        if tolerance <= 0:
            raise ValueError("dominance tolerance must be positive")
        if self.dimension != other.dimension or self.task_labels != other.task_labels:
            raise ValueError("capability dominance needs identical task coordinates")
        if self.context is not None or other.context is not None:
            if self.context is None or other.context is None:
                raise ValueError(
                    "both capability contexts are required for a contextual comparison"
                )
            EstimandFirewall.common_reference(
                self.context, other.context, allow_information_change=True
            )
        if other.randomized_policies and not self.randomized_policies:
            if not other.policy_values:
                return True
            if not self.policy_values:
                return False
            if self.attainability(other.coordinatewise_max(), tolerance=tolerance).attainable:
                return True
            # Each coordinate cell is a singleton at the minimum or (low, high].
            # Within a cell, membership in every deterministic lower orthant is
            # constant. Only cells not covered by any one policy need an LP test.
            axes = []
            for coordinate in range(self.dimension):
                low = min(vector[coordinate] for vector in other.policy_values.values())
                high = max(vector[coordinate] for vector in other.policy_values.values())
                cuts = sorted(
                    {
                        low,
                        high,
                        *(
                            vector[coordinate]
                            for vector in self.policy_values.values()
                            if low < vector[coordinate] < high
                        ),
                    }
                )
                axes.append(
                    (
                        (low, low, False),
                        *tuple((left, right, True) for left, right in zip(cuts, cuts[1:])),
                    )
                )
            bounded_hull = CapabilityRegion(
                {
                    policy: (*vector, *(-value for value in vector))
                    for policy, vector in other.policy_values.items()
                },
                other.representation,
                randomized_policies=True,
            )
            for cell in product(*axes):
                upper = tuple(interval[1] for interval in cell)
                if any(
                    all(value >= limit - tolerance for value, limit in zip(vector, upper))
                    for vector in self.policy_values.values()
                ):
                    continue
                # Strict lower boundaries use the declared numerical tolerance.
                lower = tuple(
                    low + (tolerance if open_lower else 0.0) for low, _, open_lower in cell
                )
                target = (*lower, *(-value for value in upper))
                if bounded_hull.attainability(target, tolerance=tolerance / 10).attainable:
                    return False
            return True
        return all(
            self.attainability(value, tolerance=tolerance).attainable
            for value in other.policy_values.values()
        )

    def direction_sweep(
        self, directions: Iterable[Sequence[float]]
    ) -> tuple[dict[str, float | str | tuple[float, ...] | None], ...]:
        return tuple(
            {
                "direction": tuple(direction),
                "support": self.support(direction)[0],
                "policy": self.support(direction)[1],
            }
            for direction in directions
        )


@dataclass(frozen=True)
class RobustCapabilityRegion(VersionedRecord):
    """Policy values indexed by uncertainty kernel.

    Support uses `sup_pi inf_K`, while `optimistic_support` exposes the
    distinct `inf_K sup_pi` order for comparison.
    """

    policy_kernel_values: Mapping[str, tuple[tuple[float, ...], ...]]
    representation: str
    kernel_ids: tuple[str, ...] = ()
    task_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_kernel_values",
            {
                policy: tuple(tuple(vector) for vector in kernels)
                for policy, kernels in self.policy_kernel_values.items()
            },
        )
        counts = {len(kernels) for kernels in self.policy_kernel_values.values()}
        dimensions = {
            len(vector) for kernels in self.policy_kernel_values.values() for vector in kernels
        }
        if (
            len(counts) > 1
            or (counts and 0 in counts)
            or len(dimensions) > 1
            or (dimensions and 0 in dimensions)
        ):
            raise ValueError(
                "robust policies must share a nonempty, aligned kernel set and vector dimension"
            )
        if self.kernel_ids and (
            len(self.kernel_ids) != next(iter(counts), 0)
            or len(set(self.kernel_ids)) != len(self.kernel_ids)
        ):
            raise ValueError("robust kernel IDs must be unique and align with each policy")
        if self.task_labels and len(self.task_labels) != self.dimension:
            raise ValueError("robust task labels have wrong dimension")
        for kernels in self.policy_kernel_values.values():
            for vector in kernels:
                _vector(vector, self.dimension, "robust capability")
        object.__setattr__(
            self,
            "policy_kernel_values",
            {
                policy: tuple(
                    _vector(vector, self.dimension, "robust capability") for vector in kernels
                )
                for policy, kernels in self.policy_kernel_values.items()
            },
        )

    @property
    def dimension(self) -> int:
        return (
            len(next(iter(self.policy_kernel_values.values()))[0])
            if self.policy_kernel_values
            else 0
        )

    def support(self, direction: Sequence[float]) -> tuple[float, str | None]:
        direction = _vector(direction, self.dimension, "robust direction")
        scored = {
            policy: min(sum(q * value for q, value in zip(direction, vector)) for vector in kernels)
            for policy, kernels in self.policy_kernel_values.items()
            if kernels
        }
        if not scored:
            return float("-inf"), None
        policy, value = max(scored.items(), key=lambda item: item[1])
        return value, policy

    def optimistic_support(self, direction: Sequence[float]) -> float:
        """Oracle upper bound inf_K sup_pi, with aligned uncertainty scenarios."""
        direction = _vector(direction, self.dimension, "oracle direction")
        if not self.policy_kernel_values:
            return float("-inf")
        count = len(next(iter(self.policy_kernel_values.values())))
        return min(
            max(
                sum(q * value for q, value in zip(direction, kernels[index]))
                for kernels in self.policy_kernel_values.values()
            )
            for index in range(count)
        )

    def requirement_region(self) -> "RequirementRegion":
        values = {
            policy: tuple(min(vector[i] for vector in kernels) for i in range(self.dimension))
            for policy, kernels in self.policy_kernel_values.items()
        }
        return CapabilityRegion(
            values, self.representation, task_labels=self.task_labels
        ).requirement_region()


@dataclass(frozen=True)
class CapabilityConfidenceRegion(VersionedRecord):
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    method: str

    def __post_init__(self) -> None:
        for value in (*self.lower, *self.upper):
            finite(value, "confidence bound")
        if len(self.lower) != len(self.upper) or any(
            low > high for low, high in zip(self.lower, self.upper)
        ):
            raise ValueError("confidence bounds are invalid")


@dataclass(frozen=True)
class ResourceConditionedCapability(VersionedRecord):
    regions: Mapping[str, CapabilityRegion]
    resource_model: str

    def at(self, resource_condition: str) -> CapabilityRegion:
        if resource_condition not in self.regions:
            raise KeyError(resource_condition)
        return self.regions[resource_condition]


def condition_on_resource(
    region: CapabilityRegion, feasible_policies: Sequence[str], resource_condition: str
) -> ResourceConditionedCapability:
    allowed = set(feasible_policies)
    if allowed - set(region.policy_values):
        raise ValueError("resource feasibility refers to unknown policies")
    context = (
        replace(
            region.context,
            resource_state=resource_condition,
            evaluation_regime="resource_conditioned",
        )
        if region.context
        else None
    )
    conditioned = CapabilityRegion(
        {policy: value for policy, value in region.policy_values.items() if policy in allowed},
        region.representation,
        region.randomized_policies,
        region.task_labels,
        context,
        region.dimension,
    )
    return ResourceConditionedCapability(
        {resource_condition: conditioned}, "declared_resource_constraint"
    )


def support_gap(
    full: CapabilityRegion, reduced: CapabilityRegion, direction: Sequence[float]
) -> dict[str, float | str | None]:
    """Compare independently optimized support values.

    The selected policy in `full` is intentionally not reused in `reduced`.
    """
    if not full.policy_values or not reduced.policy_values:
        raise ValueError("support gap requires two nonempty feasible policy sets")
    if full.dimension != reduced.dimension or full.task_labels != reduced.task_labels:
        raise ValueError("support gap requires identical task coordinates")
    if full.context is not None or reduced.context is not None:
        if full.context is None or reduced.context is None:
            raise ValueError("both support comparison contexts must be declared")
        EstimandFirewall.common_reference(
            full.context, reduced.context, allow_information_change=True
        )
    full_value, full_policy = full.support(direction)
    reduced_value, reduced_policy = reduced.support(direction)
    return {
        "full_support": full_value,
        "reduced_support": reduced_value,
        "gap": full_value - reduced_value,
        "full_policy": full_policy,
        "reduced_policy": reduced_policy,
        "estimand": "independent_policy_support_gap",
    }


@dataclass(frozen=True)
class PIRContext(VersionedRecord):
    """Reference conditions required to interpret PIR."""

    task_id: str
    horizon: float
    utility: str
    policy: str
    resource: str
    communication: str
    reference: str
    estimator: str = "declared_values"
    uncertainty: str = "none_declared"
    population: str = "unspecified"

    def __post_init__(self) -> None:
        finite(self.horizon, "PIR horizon")
        if self.horizon < 0 or not all(
            (
                self.task_id,
                self.utility,
                self.policy,
                self.resource,
                self.communication,
                self.reference,
                self.estimator,
                self.uncertainty,
            )
        ):
            raise ValueError("PIR context must declare common scientific references")


@dataclass(frozen=True)
class PIRResult(VersionedRecord):
    ideal_value: float
    actual_optimal_value: float
    actual_policy_value: float
    minimum_value: float
    information_loss: float
    policy_loss: float
    total_loss: float
    pir: float | None
    references: Mapping[str, str] = field(default_factory=dict)
    context: PIRContext | None = None
    metric_context: MetricContext | None = None
    reference_checked: bool = False
    limitations: tuple[str, ...] = ()
    value_contexts: Mapping[str, MetricContext] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "ideal_value",
            "actual_optimal_value",
            "actual_policy_value",
            "minimum_value",
            "information_loss",
            "policy_loss",
            "total_loss",
        ):
            value = getattr(self, name)
            finite(value, "PIR result value")
            object.__setattr__(self, name, float(value))
        expected = (
            self.ideal_value - self.actual_optimal_value,
            self.actual_optimal_value - self.actual_policy_value,
            self.ideal_value - self.actual_policy_value,
        )
        for actual, correct in zip(
            (self.information_loss, self.policy_loss, self.total_loss), expected
        ):
            if abs(actual - correct) > 1e-12 * max(1, abs(correct)):
                raise ValueError("PIR result has an inconsistent loss decomposition")
        denominator = self.ideal_value - self.minimum_value
        if denominator == 0:
            if self.pir is not None:
                raise ValueError("zero-reference PIR denominator must produce an undefined ratio")
        else:
            if self.pir is None:
                raise ValueError("nonzero-reference PIR denominator requires a ratio")
            finite(self.pir, "PIR ratio")
            correct = (self.actual_optimal_value - self.minimum_value) / denominator
            if abs(self.pir - correct) > 1e-12 * max(1, abs(correct)):
                raise ValueError("PIR result ratio disagrees with its reference values")
        if self.reference_checked:
            if set(self.value_contexts) != {"ideal", "actual_optimal", "actual_policy", "minimum"}:
                raise ValueError("reference-checked PIR must retain all four scientific contexts")
            ideal_ctx, actual_ctx = (
                self.value_contexts["ideal"],
                self.value_contexts["actual_optimal"],
            )
            for ctx in self.value_contexts.values():
                EstimandFirewall.require_metric(ctx, "task_value", "policy_value")
                EstimandFirewall.common_reference(ideal_ctx, ctx, allow_information_change=True)
            if (
                ideal_ctx.information_condition == actual_ctx.information_condition
                or self.value_contexts["actual_policy"].information_condition
                != actual_ctx.information_condition
            ):
                raise ValueError("PIR result has inconsistent ideal/actual information references")
            if self.metric_context != replace(
                actual_ctx,
                metric_type="pir",
                unit="1",
                estimand_name="normalized_optimal_information_value",
            ):
                raise ValueError("PIR result metric context disagrees with its value contexts")


def compute_pir(
    ideal_value: float,
    actual_optimal_value: float,
    actual_policy_value: float,
    minimum_value: float,
    *,
    references: Mapping[str, str] | None = None,
    context: PIRContext | None = None,
    value_contexts: Mapping[str, MetricContext] | None = None,
    information_conditions: tuple[str, str] | None = None,
) -> PIRResult:
    for value in (ideal_value, actual_optimal_value, actual_policy_value, minimum_value):
        finite(value, "PIR value")
    ideal_value, actual_optimal_value, actual_policy_value, minimum_value = (
        float(value)
        for value in (ideal_value, actual_optimal_value, actual_policy_value, minimum_value)
    )
    if not references and context is None and value_contexts is None:
        raise ValueError("PIR requires declared common references")
    checked_context = None
    if value_contexts is not None:
        expected = {"ideal", "actual_optimal", "actual_policy", "minimum"}
        if set(value_contexts) != expected or information_conditions is None:
            raise ValueError(
                "PIR requires all four value contexts and ideal/actual information conditions"
            )
        ideal_context = value_contexts["ideal"]
        actual_context = value_contexts["actual_optimal"]
        for value_context in value_contexts.values():
            EstimandFirewall.require_metric(value_context, "task_value", "policy_value")
            EstimandFirewall.common_reference(
                ideal_context, value_context, allow_information_change=True
            )
        if (
            ideal_context.information_condition,
            actual_context.information_condition,
        ) != information_conditions:
            raise ValueError(
                "PIR information conditions do not match declared ideal/actual references"
            )
        if (
            information_conditions[0] == information_conditions[1]
            or value_contexts["actual_policy"].information_condition != information_conditions[1]
        ):
            raise ValueError(
                "PIR needs distinct ideal/actual information and a controller under actual information"
            )
        checked_context = replace(
            actual_context,
            metric_type="pir",
            unit="1",
            estimand_name="normalized_optimal_information_value",
        )
    if minimum_value > min(ideal_value, actual_optimal_value, actual_policy_value) + 1e-12:
        raise ValueError("minimum reference exceeds a task value")
    if actual_policy_value > actual_optimal_value + 1e-12:
        raise ValueError("deployed policy value exceeds the declared actual optimum")
    denominator = ideal_value - minimum_value
    pir = None if denominator == 0 else (actual_optimal_value - minimum_value) / denominator
    return PIRResult(
        ideal_value=ideal_value,
        actual_optimal_value=actual_optimal_value,
        actual_policy_value=actual_policy_value,
        minimum_value=minimum_value,
        information_loss=ideal_value - actual_optimal_value,
        policy_loss=actual_optimal_value - actual_policy_value,
        total_loss=ideal_value - actual_policy_value,
        pir=pir,
        references=dict(references or {}),
        context=context,
        metric_context=checked_context,
        reference_checked=value_contexts is not None,
        limitations=()
        if value_contexts is not None
        else (
            "Common references declared jointly; per-value information/policy contexts not independently checked.",
        ),
        value_contexts=dict(value_contexts or {}),
    )


@dataclass(frozen=True)
class RequirementRegion(VersionedRecord):
    """Downward closure of a declared deterministic or convex policy image."""

    capability: CapabilityRegion

    def contains(self, requirement: Sequence[float]) -> bool:
        return self.capability.jointly_attainable(requirement)

    def support(self, direction: Sequence[float]) -> tuple[float, str | None]:
        _vector(direction, self.capability.dimension, "requirement direction")
        if not self.capability.policy_values:
            return self.capability.support(direction)
        if any(value < 0 for value in direction):
            raise ValueError("downward-closure support is unbounded for negative directions")
        return self.capability.support(direction)

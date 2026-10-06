"""Common-policy risk, declared resource relations, and experiment comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from physioworld.core.contracts import EstimandFirewall, VersionedRecord, finite
from physioworld.core.evidence import AssumptionSet
from physioworld.core.flow import FiniteKernel

from .region import CapabilityRegion, RobustCapabilityRegion, _vector


def robust_value(
    policy_kernel_values: Mapping[str, Sequence[float]], *, order: str = "sup_inf"
) -> float:
    """Return the common-policy robust value ``sup_pi inf_K J(pi,K)``.

    The optimistic oracle ``inf_K sup_pi`` is deliberately not accepted under
    this name because it allows the policy to change after the uncertainty
    state is revealed. Use :func:`optimistic_value` when that distinct upper
    bound is scientifically intended.
    """
    if not isinstance(policy_kernel_values, Mapping) or not policy_kernel_values:
        raise ValueError(
            "robust_value needs a policy-by-kernel mapping, not preoptimized scenario values"
        )
    region = RobustCapabilityRegion(
        {
            policy: tuple((value,) for value in values)
            for policy, values in policy_kernel_values.items()
        },
        "declared_common_policy_problem",
    )
    if order != "sup_inf":
        raise ValueError("robust_value is common-policy sup_inf; use optimistic_value for inf_sup")
    return region.support((1.0,))[0]


def optimistic_value(policy_kernel_values: Mapping[str, Sequence[float]]) -> float:
    """Return the distinct oracle upper bound ``inf_K sup_pi J(pi,K)``."""
    if not isinstance(policy_kernel_values, Mapping) or not policy_kernel_values:
        raise ValueError("optimistic_value needs a policy-by-kernel mapping")
    region = RobustCapabilityRegion(
        {
            policy: tuple((value,) for value in values)
            for policy, values in policy_kernel_values.items()
        },
        "declared_oracle_upper_bound",
    )
    return region.optimistic_support((1.0,))


class PolicyRelationKind(str, Enum):
    POLICY_EMULATION = "policy_emulation"
    CAPABILITY_INCLUSION = "capability_inclusion"
    SIMULATION = "simulation_relation"


@dataclass(frozen=True)
class PolicyRelation(VersionedRecord):
    dominating_condition: str
    dominated_condition: str
    kind: PolicyRelationKind
    policy_map: Mapping[str, str]
    assumptions: AssumptionSet
    source_artifact: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PolicyRelationKind(self.kind))
        if not all((self.dominating_condition, self.dominated_condition, self.source_artifact)):
            raise ValueError("policy relation must declare conditions and a source artifact")


def resource_dominates(
    a: CapabilityRegion, b: CapabilityRegion, *, relation: PolicyRelation, strict: bool = False
) -> bool:
    """Check a declared policy emulation/capability relation, never resource coordinates.

    A finite value check validates only the specified task coordinates. Universal
    dynamics, safety and trajectory preservation remain explicit assumptions of
    a simulation/emulation witness; they are not inferred from these values.
    """
    if not isinstance(a, CapabilityRegion) or not isinstance(b, CapabilityRegion):
        raise ValueError("resource dominance requires declared policy capability images")
    if (a.representation, b.representation) != (
        relation.dominating_condition,
        relation.dominated_condition,
    ):
        raise ValueError("resource relation does not match compared conditions")
    relation.assumptions.require()
    if a.dimension != b.dimension or a.task_labels != b.task_labels:
        raise ValueError("resource dominance requires common task coordinates")
    if a.context is not None or b.context is not None:
        if a.context is None or b.context is None:
            raise ValueError("resource dominance needs both scientific contexts")
        EstimandFirewall.require_complete_reference(a.context)
        EstimandFirewall.require_complete_reference(b.context)
        differences = set(EstimandFirewall.differences(a.context, b.context))
        # The relation explicitly authorizes resource/info/policy feasibility changes.
        differences -= {"resource_state", "information_condition", "policy_class"}
        if differences:
            raise ValueError(
                f"resource dominance has incompatible references: {sorted(differences)}"
            )
    if relation.kind == PolicyRelationKind.CAPABILITY_INCLUSION:
        # Context resource changes have already been checked through the witness.
        bare_a = CapabilityRegion(
            a.policy_values, a.representation, a.randomized_policies, a.task_labels
        )
        bare_b = CapabilityRegion(
            b.policy_values, b.representation, b.randomized_policies, b.task_labels
        )
        included = bare_a.dominates(bare_b)
        return included and (not strict or not bare_b.dominates(bare_a))
    if set(relation.policy_map) != set(b.policy_values):
        raise ValueError("policy emulation must cover every dominated policy")
    improved = False
    for weak_policy, strong_policy in relation.policy_map.items():
        if strong_policy not in a.policy_values:
            raise ValueError("policy relation refers to an unknown emulating policy")
        weak, strong = b.policy_values[weak_policy], a.policy_values[strong_policy]
        if any(left < right for left, right in zip(strong, weak)):
            return False
        improved |= any(left > right for left, right in zip(strong, weak))
    return not strict or improved


def lower_tail_cvar(
    values: Sequence[float], probabilities: Sequence[float], *, tail_probability: float = 0.05
) -> float:
    """Lower-tail expected utility with fractional mass at the discrete quantile.

    This is utility CVaR (larger is better), not upper-tail loss CVaR.
    """
    finite(tail_probability, "CVaR tail probability")
    if not 0 < tail_probability <= 1 or len(values) == 0 or len(values) != len(probabilities):
        raise ValueError("CVaR needs paired values/probabilities and tail probability in (0,1]")
    for value in (*values, *probabilities):
        finite(value, "CVaR input")
    values = tuple(float(value) for value in values)
    probabilities = tuple(float(value) for value in probabilities)
    if any(p < 0 or p > 1 for p in probabilities) or abs(sum(probabilities) - 1) > 1e-12:
        raise ValueError("CVaR probabilities must be normalized and nonnegative")
    remaining, total = tail_probability, 0.0
    for value, probability in sorted(zip(values, probabilities)):
        mass = min(remaining, probability)
        total += mass * value
        remaining -= mass
        if remaining <= 1e-15:
            break
    return total / tail_probability


@dataclass(frozen=True)
class ChanceCapabilityResult(VersionedRecord):
    requirement: tuple[float, ...]
    minimum_probability: float
    policy_probabilities: Mapping[str, float]
    feasible_policies: tuple[str, ...]
    scenario_ids: tuple[str, ...]


@dataclass(frozen=True)
class StochasticCapabilityRegion(VersionedRecord):
    policy_scenario_values: Mapping[str, tuple[tuple[float, ...], ...]]
    scenario_probabilities: tuple[float, ...]
    representation: str
    scenario_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        aligned = RobustCapabilityRegion(
            self.policy_scenario_values, self.representation, self.scenario_ids
        )
        object.__setattr__(self, "policy_scenario_values", aligned.policy_kernel_values)
        object.__setattr__(self, "scenario_probabilities", tuple(self.scenario_probabilities))
        count = (
            len(next(iter(self.policy_scenario_values.values())))
            if self.policy_scenario_values
            else 0
        )
        if len(self.scenario_probabilities) != count or count == 0:
            raise ValueError("stochastic capabilities need aligned scenario probabilities")
        lower_tail_cvar(
            tuple(0.0 for _ in range(count)), self.scenario_probabilities, tail_probability=1
        )

    @property
    def dimension(self) -> int:
        return len(next(iter(self.policy_scenario_values.values()))[0])

    def cvar_support(
        self, direction: Sequence[float], *, tail_probability: float = 0.05
    ) -> tuple[float, str]:
        _vector(direction, self.dimension, "CVaR utility direction")
        scores = {
            policy: lower_tail_cvar(
                tuple(
                    sum(q * value for q, value in zip(direction, vector)) for vector in scenarios
                ),
                self.scenario_probabilities,
                tail_probability=tail_probability,
            )
            for policy, scenarios in self.policy_scenario_values.items()
        }
        policy = max(scores, key=scores.__getitem__)
        return scores[policy], policy

    def chance_capability(
        self, requirement: Sequence[float], *, minimum_probability: float
    ) -> ChanceCapabilityResult:
        target = _vector(requirement, self.dimension, "joint chance requirement")
        finite(minimum_probability, "minimum chance probability")
        if not 0 <= minimum_probability <= 1:
            raise ValueError("minimum probability must lie in [0,1]")
        probabilities = {
            policy: sum(
                p
                for p, vector in zip(self.scenario_probabilities, scenarios)
                if all(value >= threshold for value, threshold in zip(vector, target))
            )
            for policy, scenarios in self.policy_scenario_values.items()
        }
        return ChanceCapabilityResult(
            target,
            minimum_probability,
            probabilities,
            tuple(
                policy
                for policy, probability in probabilities.items()
                if probability >= minimum_probability
            ),
            self.scenario_ids,
        )


def blackwell_dominates(
    informative: FiniteKernel,
    reduced: FiniteKernel,
    garbling: FiniteKernel,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Verify an explicit stochastic garbling witness Q = P G on finite experiments."""
    finite(tolerance, "Blackwell tolerance")
    if tolerance < 0:
        raise ValueError("Blackwell tolerance must be nonnegative")
    if (
        informative.input_space != reduced.input_space
        or informative.input_states != reduced.input_states
        or informative.output_space != garbling.input_space
        or informative.output_states != garbling.input_states
        or reduced.output_space != garbling.output_space
        or reduced.output_states != garbling.output_states
    ):
        raise ValueError("Blackwell witness requires aligned world/observation spaces")
    return all(
        abs(
            sum(p * garbling.probabilities[k][j] for k, p in enumerate(row))
            - reduced.probabilities[i][j]
        )
        <= tolerance
        for i, row in enumerate(informative.probabilities)
        for j in range(len(reduced.output_states))
    )


@dataclass(frozen=True)
class StrategicMeasure(VersionedRecord):
    policy_id: str
    trajectories: tuple[tuple[str, ...], ...]
    probabilities: tuple[float, ...]
    information_set: str
    action_support: tuple[str, ...]
    reference: str

    def __post_init__(self) -> None:
        if len(set(self.trajectories)) != len(self.trajectories):
            raise ValueError("strategic-measure trajectories must be unique")
        if not all((self.policy_id, self.information_set, self.action_support, self.reference)):
            raise ValueError(
                "strategic measure needs policy, information, action support and reference"
            )
        lower_tail_cvar(
            tuple(0.0 for _ in self.trajectories), self.probabilities, tail_probability=1
        )

    def value(self, trajectory_utilities: Mapping[tuple[str, ...], float]) -> float:
        if set(trajectory_utilities) != set(self.trajectories):
            raise ValueError("utility must be defined on exactly the strategic trajectory support")
        values = tuple(trajectory_utilities[trajectory] for trajectory in self.trajectories)
        for value in values:
            finite(value, "trajectory utility")
        return sum(p * value for p, value in zip(self.probabilities, values))

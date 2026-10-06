"""Paired information-availability PIV and contextual value evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Sequence

from physioworld.core.contracts import (
    CausalStatus,
    EstimandFirewall,
    EvidenceClass,
    MetricContext,
    VersionedRecord,
    finite,
)
from physioworld.core.evidence import AssumptionSet, IdentificationRecord
from physioworld.tasklaws import BootstrapResult, clustered_bootstrap


class InformationDesignKind(str, Enum):
    PAIRED_MODEL_REPLAY = "paired_model_replay"
    RANDOMIZED_CROSSOVER = "randomized_crossover"
    IDENTIFIED_PAIRED_COUNTERFACTUAL = "identified_paired_counterfactual"


PIV_REQUIREMENTS = (
    "paired_units",
    "same_initial_reference",
    "information_availability_intervention",
    "common_utility",
    "comparable_policy_classes",
    "no_future_information",
)


@dataclass(frozen=True)
class InformationAvailabilityDesign(VersionedRecord):
    design_id: str
    modality: str
    kind: InformationDesignKind
    baseline_context: MetricContext
    augmented_context: MetricContext
    pair_ids: tuple[str, ...]
    assumptions: AssumptionSet
    source_artifact: str
    identification: IdentificationRecord | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", InformationDesignKind(self.kind))
        object.__setattr__(self, "pair_ids", tuple(self.pair_ids))
        self.validate()

    def validate(self) -> None:
        if not all((self.design_id, self.modality, self.pair_ids, self.source_artifact)):
            raise ValueError("PIV design must declare paired units, modality and references")
        if len(set(self.pair_ids)) != len(self.pair_ids):
            raise ValueError("PIV pair IDs must be unique")
        for context in (self.baseline_context, self.augmented_context):
            EstimandFirewall.require_metric(context, "task_value", "policy_value")
        EstimandFirewall.common_reference(
            self.baseline_context, self.augmented_context, allow_information_change=True
        )
        if (
            self.baseline_context.information_condition
            == self.augmented_context.information_condition
        ):
            raise ValueError(
                "PIV requires distinct baseline and augmented information availability"
            )
        self.assumptions.require(PIV_REQUIREMENTS)
        if self.kind == InformationDesignKind.PAIRED_MODEL_REPLAY:
            if self.baseline_context.evidence_class not in {
                EvidenceClass.MODEL,
                EvidenceClass.SIMULATION,
            }:
                raise ValueError("model replay PIV must be labeled MODEL or SIMULATION")
            if self.baseline_context.causal_status not in {
                CausalStatus.MODEL_INTERVENTIONAL,
                CausalStatus.NOT_APPLICABLE,
            }:
                raise ValueError("model replay cannot identify an empirical causal effect")
        else:
            if self.identification is None or self.identification.status != CausalStatus.IDENTIFIED:
                raise ValueError(
                    "empirical counterfactual PIV needs a guarded identification record"
                )
            self.identification.require_identified()
            if self.baseline_context.causal_status != CausalStatus.IDENTIFIED:
                raise ValueError("identified PIV requires identified causal context")


PIVDesign = InformationAvailabilityDesign


def _piv_context(
    design: InformationAvailabilityDesign, uncertainty: BootstrapResult | None
) -> MetricContext:
    return replace(
        design.augmented_context,
        metric_type="piv",
        estimand_name="paired_information_availability_value",
        information_condition=(
            f"{design.augmented_context.information_condition} minus "
            f"{design.baseline_context.information_condition}"
        ),
        estimator="paired_mean_utility_difference",
        uncertainty_treatment="clustered_percentile_bootstrap"
        if uncertainty and uncertainty.resampling_unit == "cluster"
        else "paired_percentile_bootstrap"
        if uncertainty
        else "none_declared",
    )


@dataclass(frozen=True)
class PIVEstimate(VersionedRecord):
    modality: str
    incremental_value: float
    paired_differences: tuple[float, ...]
    pair_ids: tuple[str, ...]
    context: MetricContext
    design_id: str
    estimator: str
    evidence_class: EvidenceClass
    design: InformationAvailabilityDesign
    uncertainty: BootstrapResult | None = None

    def __post_init__(self) -> None:
        self.design.validate()
        finite(self.incremental_value, "PIV result value")
        if self.pair_ids != self.design.pair_ids or len(self.paired_differences) != len(
            self.pair_ids
        ):
            raise ValueError("PIV result does not retain its paired design")
        for value in self.paired_differences:
            finite(value, "paired PIV difference")
        expected = sum(self.paired_differences) / len(self.paired_differences)
        if abs(self.incremental_value - expected) > 1e-12 * max(1, abs(expected)):
            raise ValueError("PIV result does not match its paired utility differences")
        if self.modality != self.design.modality or self.design_id != self.design.design_id:
            raise ValueError("PIV result modality/design ID does not match its design")
        if self.evidence_class != self.design.augmented_context.evidence_class:
            raise ValueError("PIV result cannot upgrade the design's evidence class")
        EstimandFirewall.require_metric(
            self.context, "piv", "paired_information_availability_value"
        )
        if (
            self.context != _piv_context(self.design, self.uncertainty)
            or self.estimator != "paired_mean_utility_difference"
        ):
            raise ValueError(
                "PIV result context/estimator disagrees with the information-availability design"
            )
        if self.uncertainty is not None and abs(self.uncertainty.point_estimate - expected) > 1e-12:
            raise ValueError("PIV uncertainty is attached to a different point estimand")


def estimate_piv(
    baseline_values: Sequence[float],
    augmented_values: Sequence[float],
    *,
    design: InformationAvailabilityDesign,
    baseline_pair_ids: Sequence[str],
    augmented_pair_ids: Sequence[str],
    clusters: Sequence[object] | None = None,
    bootstrap_repetitions: int = 0,
    seed: int | None = None,
) -> PIVEstimate:
    """E[V(G,M)-V(G)] using an explicit paired information intervention.

    Values are paired downstream utilities, not SQI, confidence, or predictive
    scores. Pair alignment is checked rather than inferred from array lengths.
    """
    design.validate()
    if tuple(baseline_pair_ids) != design.pair_ids or tuple(augmented_pair_ids) != design.pair_ids:
        raise ValueError("PIV values are not aligned to the declared paired design")
    if len(baseline_values) != len(design.pair_ids) or len(augmented_values) != len(
        design.pair_ids
    ):
        raise ValueError("PIV value counts do not match the paired design")
    for value in (*baseline_values, *augmented_values):
        finite(value, "paired utility")
    differences = tuple(
        float(augmented) - float(baseline)
        for baseline, augmented in zip(baseline_values, augmented_values)
    )
    uncertainty = None
    if bootstrap_repetitions:
        uncertainty = clustered_bootstrap(
            len(differences),
            lambda indices: sum(differences[i] for i in indices) / len(indices),
            clusters=clusters,
            repetitions=bootstrap_repetitions,
            seed=seed,
        )
    elif clusters is not None and len(clusters) != len(differences):
        raise ValueError("PIV cluster labels have wrong length")
    result_context = _piv_context(design, uncertainty)
    return PIVEstimate(
        design.modality,
        sum(differences) / len(differences),
        differences,
        design.pair_ids,
        result_context,
        design.design_id,
        "paired_mean_utility_difference",
        design.augmented_context.evidence_class,
        design,
        uncertainty,
    )


@dataclass(frozen=True)
class OptimalValue(VersionedRecord):
    value: float
    policy_id: str
    feasible_policies: tuple[str, ...]
    context: MetricContext

    def __post_init__(self) -> None:
        finite(self.value, "optimal task value")
        object.__setattr__(self, "value", float(self.value))
        EstimandFirewall.require_metric(self.context, "task_value", "policy_value")
        EstimandFirewall.require_complete_reference(self.context)
        if self.policy_id not in self.feasible_policies or len(set(self.feasible_policies)) != len(
            self.feasible_policies
        ):
            raise ValueError(
                "optimal value needs a unique feasible policy set containing the selected policy"
            )


def optimal_task_value(
    policy_values: Mapping[str, float],
    *,
    context: MetricContext,
    feasible_policies: Sequence[str] | None = None,
) -> OptimalValue:
    EstimandFirewall.require_metric(context, "task_value", "policy_value")
    EstimandFirewall.require_complete_reference(context)
    allowed = tuple(policy_values) if feasible_policies is None else tuple(feasible_policies)
    if not allowed or set(allowed) - set(policy_values) or len(set(allowed)) != len(allowed):
        raise ValueError("task value needs a nonempty declared feasible policy set")
    for value in policy_values.values():
        finite(value, "policy value")
    policy = max(allowed, key=policy_values.__getitem__)
    return OptimalValue(policy_values[policy], policy, allowed, context)


@dataclass(frozen=True)
class RepresentationDeficiency(VersionedRecord):
    value: float
    full: OptimalValue
    representation: OptimalValue
    context: MetricContext

    def __post_init__(self) -> None:
        finite(self.value, "representation deficiency")
        EstimandFirewall.common_reference(
            self.full.context, self.representation.context, allow_information_change=True
        )
        if (
            self.full.context.information_condition
            == self.representation.context.information_condition
        ):
            raise ValueError(
                "representation deficiency requires declared different information conditions"
            )
        correct = self.full.value - self.representation.value
        if abs(self.value - correct) > 1e-12 * max(1, abs(correct)):
            raise ValueError(
                "representation deficiency disagrees with its optimal value references"
            )
        expected_context = replace(
            self.representation.context,
            metric_type="representation_deficiency",
            estimand_name="optimal_information_value_difference",
        )
        if self.context != expected_context:
            raise ValueError("representation deficiency has an incompatible metric context")


def representation_deficiency(
    full: OptimalValue, representation: OptimalValue
) -> RepresentationDeficiency:
    EstimandFirewall.common_reference(
        full.context, representation.context, allow_information_change=True
    )
    if full.context.information_condition == representation.context.information_condition:
        raise ValueError(
            "representation deficiency requires declared different information conditions"
        )
    result_context = replace(
        representation.context,
        metric_type="representation_deficiency",
        estimand_name="optimal_information_value_difference",
    )
    return RepresentationDeficiency(
        full.value - representation.value, full, representation, result_context
    )

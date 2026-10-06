"""Common-reference PIR accounting and finite capability preservation.

PIR normalizes the *optimal actual-information value*, not the deployed
controller value. Information losses and support gaps remain signed unless
policy-class nesting is established separately. A finite direction sweep
describes only the supplied directions, not inclusion of whole capability sets.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence, cast

from physioworld.core.contracts import (
    CausalStatus,
    EstimandFirewall,
    EvidenceClass,
    Metric,
    MetricContext,
    VersionedRecord,
    finite,
)
from physioworld.core.evidence import IdentificationRecord

from .region import CapabilityRegion, PIRResult, _vector, compute_pir, support_gap
from .values import OptimalValue, optimal_task_value, representation_deficiency


def information_loss(ideal: OptimalValue, actual: OptimalValue) -> float:
    """Return V*_ideal - V*_actual under common, distinct information references."""
    return representation_deficiency(ideal, actual).value


def policy_loss(actual: OptimalValue, deployed: Metric) -> float:
    """Return V*_actual - V_actual(pi_A) under the same information condition."""
    EstimandFirewall.require_metric(deployed.context, "task_value", "policy_value")
    EstimandFirewall.common_reference(actual.context, deployed.context)
    loss = actual.value - deployed.value
    finite(loss, "policy loss")
    if loss < -1e-12:
        raise ValueError("deployed policy value exceeds the declared actual optimum")
    return loss


def total_loss(ideal: OptimalValue, actual: OptimalValue, deployed: Metric) -> float:
    """Return L_info + L_policy with both reference comparisons checked."""
    loss = information_loss(ideal, actual) + policy_loss(actual, deployed)
    finite(loss, "total loss")
    return loss


def pir_identity(result: PIRResult) -> float | None:
    """Evaluate PIR = 1 - L_info / (V*_ideal - V_min).

    Per-value common references must have been checked by ``compute_pir``.
    The identity is undefined (``None``) at a zero reference span, just as PIR
    itself is. Controller loss is deliberately absent from this identity.
    """
    if not result.reference_checked:
        raise ValueError("PIR identity requires checked per-value common references")
    span = result.ideal_value - result.minimum_value
    return None if span == 0 else 1.0 - result.information_loss / span


def _value_context(region: CapabilityRegion) -> MetricContext:
    if region.context is None:
        raise ValueError("finite policy PIR requires both capability value contexts")
    EstimandFirewall.require_metric(region.context, "task_value", "policy_value")
    EstimandFirewall.require_complete_reference(region.context)
    if region.dimension != len(region.context.task_family) or (
        region.task_labels and region.task_labels != region.context.task_family
    ):
        raise ValueError("capability task coordinates disagree with the value context")
    if region.context.information_condition == "unspecified":
        raise ValueError("finite policy PIR requires explicit information conditions")
    return region.context


def finite_policy_pir(
    ideal: CapabilityRegion,
    actual: CapabilityRegion,
    direction: Sequence[float],
    *,
    deployed_policy_id: str,
    minimum: Metric,
) -> PIRResult:
    """Independently enumerate two finite policy images and account for their loss.

    All values use the same scalar utility ``direction dot V(pi)``. The
    supplied minimum is a reference for that scalar utility, not a vector of
    coordinatewise minima. Policy IDs may differ between the two images.
    Explicit convexification does not change a linear support optimum.
    """
    ideal_context, actual_context = _value_context(ideal), _value_context(actual)
    weights = _vector(direction, ideal.dimension, "PIR utility direction")
    # Reuse the existing independent-support comparison and its reference gate.
    support_gap(ideal, actual, weights)
    if deployed_policy_id not in actual.policy_values:
        raise ValueError("deployed policy must belong to the actual feasible policy image")
    EstimandFirewall.require_metric(minimum.context, "task_value", "policy_value")
    EstimandFirewall.common_reference(actual_context, minimum.context)

    def scalar_values(region: CapabilityRegion) -> dict[str, float]:
        return {
            policy: sum(q * value for q, value in zip(weights, vector))
            for policy, vector in region.policy_values.items()
        }

    ideal_optimum = optimal_task_value(scalar_values(ideal), context=ideal_context)
    actual_values = scalar_values(actual)
    actual_optimum = optimal_task_value(actual_values, context=actual_context)
    return compute_pir(
        ideal_optimum.value,
        actual_optimum.value,
        actual_values[deployed_policy_id],
        minimum.value,
        references={
            "ideal_policy": ideal_optimum.policy_id,
            "actual_optimal_policy": actual_optimum.policy_id,
            "deployed_policy": deployed_policy_id,
            "minimum_artifact": minimum.source_artifact,
            "utility_direction": repr(weights),
        },
        value_contexts={
            "ideal": ideal_context,
            "actual_optimal": actual_context,
            "actual_policy": actual_context,
            "minimum": minimum.context,
        },
        information_conditions=(
            ideal_context.information_condition,
            actual_context.information_condition,
        ),
    )


def finite_causal_pir(
    ideal: CapabilityRegion,
    actual: CapabilityRegion,
    direction: Sequence[float],
    *,
    deployed_policy_id: str,
    minimum: Metric,
    identifications: Mapping[str, IdentificationRecord | Any] | None = None,
) -> PIRResult:
    """Normalized PIR for finite interventional policy values.

    Model-interventional inputs retain MODEL/SIMULATION evidence. Identified
    inputs require the core identification record for each of the four value
    references, with every optimized policy covered by declared strategies.
    This accounting does not identify inputs or issue safety/uncertainty
    certificates. The distinct resource-indexed robust success-value estimand
    remains in ``causal.finite_robust_causal_value``.
    """
    result = finite_policy_pir(
        ideal, actual, direction, deployed_policy_id=deployed_policy_id, minimum=minimum
    )
    context = cast(MetricContext, result.metric_context)
    if context.causal_status == CausalStatus.MODEL_INTERVENTIONAL:
        if context.evidence_class not in {EvidenceClass.MODEL, EvidenceClass.SIMULATION}:
            raise ValueError("model-interventional PIR must retain MODEL or SIMULATION evidence")
        return replace(result, limitations=("finite supplied interventional policy images only",))
    if context.causal_status != CausalStatus.IDENTIFIED:
        raise ValueError("causal PIR requires model-interventional or identified value contexts")
    if identifications is None or set(identifications) != set(result.value_contexts):
        raise ValueError("identified causal PIR requires all four value identification records")
    required_policies = {
        "ideal": set(ideal.policy_values),
        "actual_optimal": set(actual.policy_values),
        "actual_policy": {deployed_policy_id},
        "minimum": set(),
    }
    references = dict(result.references)
    limitations = ["finite supplied policy images; identification is an external dependency"]
    for name, record in identifications.items():
        value_context = result.value_contexts[name]
        status = getattr(record, "status", None)
        causal_identified = status == CausalStatus.IDENTIFIED or status in {
            "MODEL_IDENTIFIED",
            "IDENTIFIED_UNDER_EXPLICIT_CAUSAL_ASSUMPTIONS",
        }
        if not causal_identified:
            raise ValueError(f"{name}: an identified causal record is required")
        record.require_identified()
        strategies = getattr(record, "treatment_strategies", None)
        if strategies is None:
            trial = getattr(record, "trial", None)
            strategies = getattr(trial, "treatment_strategies", ())
        record_id = getattr(record, "identification_id", None) or getattr(record, "model_id", None)
        uncertainty = getattr(record, "uncertainty", None)
        if (
            not record_id
            or record.estimand != value_context.estimand_name
            or record.estimator != value_context.estimator
            or uncertainty != value_context.uncertainty_treatment
        ):
            raise ValueError(f"{name}: identification record disagrees with the value context")
        if not required_policies[name] <= set(cast(Iterable[str], strategies)):
            raise ValueError(f"{name}: identification must cover every feasible policy strategy")
        references[f"{name}_identification"] = record_id
        limitations.extend(getattr(record, "limitations", ()))
    return replace(result, references=references, limitations=tuple(dict.fromkeys(limitations)))


@dataclass(frozen=True)
class SupportGapDiagnostic(VersionedRecord):
    """Two independently selected support policies for one finite direction."""

    direction: tuple[float, ...]
    full_support: float
    reduced_support: float
    full_policy: str
    reduced_policy: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "direction", _vector(self.direction, len(self.direction), "direction")
        )
        for value in (self.full_support, self.reduced_support, self.gap):
            finite(value, "support gap diagnostic")
        if not self.direction or not self.full_policy or not self.reduced_policy:
            raise ValueError(
                "support gap diagnostic requires a direction and both selected policies"
            )

    @property
    def gap(self) -> float:
        """Signed h_full(q) - h_reduced(q), without clipping or a nesting claim."""
        return self.full_support - self.reduced_support


def finite_support_gap(
    full: CapabilityRegion, reduced: CapabilityRegion, direction: Sequence[float]
) -> SupportGapDiagnostic:
    """Return one independently optimized finite support-gap diagnostic."""
    gap = support_gap(full, reduced, direction)
    return SupportGapDiagnostic(
        tuple(direction),
        cast(float, gap["full_support"]),
        cast(float, gap["reduced_support"]),
        cast(str, gap["full_policy"]),
        cast(str, gap["reduced_policy"]),
    )


@dataclass(frozen=True)
class CapabilityPreservationRegion(VersionedRecord):
    """Finite utility-direction region where independently optimized supports agree.

    Membership is ``abs(h_full(q) - h_reduced(q)) <= tolerance`` among the
    supplied directions only. It is two-sided preservation in utility units,
    not a confidence bound, whole-set inclusion, or joint attainability test.
    Direction scaling therefore scales the gap; directions are not normalized
    implicitly. Both deterministic and explicitly convexified images are valid.
    """

    full: CapabilityRegion
    reduced: CapabilityRegion
    directions: tuple[tuple[float, ...], ...]
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        finite(self.tolerance, "preservation tolerance")
        if self.tolerance < 0:
            raise ValueError("preservation tolerance must be nonnegative")
        directions = tuple(
            _vector(direction, self.full.dimension, "preservation direction")
            for direction in self.directions
        )
        if not directions or len(set(directions)) != len(directions):
            raise ValueError("preservation diagnostic requires nonempty unique finite directions")
        object.__setattr__(self, "directions", directions)
        # Check nonempty images, aligned task coordinates, and common references.
        self.diagnostics

    @property
    def diagnostics(self) -> tuple[SupportGapDiagnostic, ...]:
        return tuple(
            finite_support_gap(self.full, self.reduced, direction) for direction in self.directions
        )

    @property
    def preserved_directions(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            diagnostic.direction
            for diagnostic in self.diagnostics
            if abs(diagnostic.gap) <= self.tolerance
        )

    @property
    def maximum_absolute_gap(self) -> float:
        return max(abs(diagnostic.gap) for diagnostic in self.diagnostics)

    @property
    def preserves_sampled_directions(self) -> bool:
        return len(self.preserved_directions) == len(self.directions)

    def contains(self, direction: Sequence[float]) -> bool:
        candidate = _vector(direction, self.full.dimension, "preservation direction")
        return candidate in self.preserved_directions


def capability_preservation_region(
    full: CapabilityRegion,
    reduced: CapabilityRegion,
    directions: Iterable[Sequence[float]],
    *,
    tolerance: float = 0.0,
) -> CapabilityPreservationRegion:
    """Construct a finite, auditable support-gap preservation diagnostic."""
    return CapabilityPreservationRegion(
        full, reduced, tuple(tuple(direction) for direction in directions), tolerance
    )


# Verbose aliases make the scalar accounting and finite-region boundaries
# discoverable without changing the established ``compute_pir`` API.
compute_information_loss = information_loss
compute_policy_loss = policy_loss
compute_total_loss = total_loss
compute_pir_identity = pir_identity
finite_capability_preservation_region = capability_preservation_region
support_gap_diagnostic = finite_support_gap
causal_pir = finite_causal_pir

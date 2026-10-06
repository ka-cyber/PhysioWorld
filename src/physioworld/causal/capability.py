"""Executable gate for finite resource-indexed common-policy causal capability.

This computes a robust *policy value*, not the normalized observational PIR ratio.
External safety/uncertainty certificates are explicit dependencies, not issued here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .identification import AssumptionStatus, IdentificationError
from .longitudinal import PolicyValueEstimate


@dataclass(frozen=True)
class SafePolicyCertificate:
    policy_id: str
    resource_regime: str
    status: AssumptionStatus
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class KernelUncertaintySet:
    kernel_ids: tuple[str, ...]
    resource_regime: str
    coverage: float
    status: AssumptionStatus
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class RobustCausalValue:
    value: float
    selected_policy_id: str
    policy_worst_case_values: tuple[tuple[str, float], ...]
    kernel_ids: tuple[str, ...]
    resource_regime: str
    estimand: str
    uncertainty_set: KernelUncertaintySet
    safety_certificates: tuple[SafePolicyCertificate, ...]
    claim_status: str
    limitations: tuple[str, ...]


def finite_robust_causal_value(
    estimates: Mapping[tuple[str, str], PolicyValueEstimate],
    *,
    safety_certificates: Sequence[SafePolicyCertificate],
    uncertainty_set: KernelUncertaintySet,
    resource_regime: str,
) -> RobustCausalValue:
    """max_policy min_kernel value with identification, safety, and uncertainty gates.

    A caller seeking the success-probability version must supply identified endpoint
    success estimates with that estimand. No confidence bound is inferred from a
    finite parameter range; coverage is the external uncertainty-set declaration.
    """
    if (
        not resource_regime
        or uncertainty_set.resource_regime != resource_regime
        or uncertainty_set.status != AssumptionStatus.VERIFIED
        or not uncertainty_set.evidence_refs
        or not uncertainty_set.kernel_ids
        or len(set(uncertainty_set.kernel_ids)) != len(uncertainty_set.kernel_ids)
        or any(not k for k in uncertainty_set.kernel_ids)
        or not math.isfinite(uncertainty_set.coverage)
        or not 0 < uncertainty_set.coverage <= 1
    ):
        raise IdentificationError("valid externally verified kernel uncertainty set is required")
    certificates = tuple(safety_certificates)
    if (
        not certificates
        or len({c.policy_id for c in certificates}) != len(certificates)
        or any(
            not c.policy_id
            or c.status != AssumptionStatus.VERIFIED
            or not c.evidence_refs
            or c.resource_regime != resource_regime
            for c in certificates
        )
    ):
        raise IdentificationError("externally verified safety-certified policy class is required")
    required = {(c.policy_id, k) for c in certificates for k in uncertainty_set.kernel_ids}
    if set(estimates) != required:
        raise IdentificationError("complete common-policy by kernel evaluation grid required")
    first_estimate = next(iter(estimates.values()))
    first = first_estimate.identification
    worst = []
    assumed = False
    for certificate in certificates:
        values = []
        for kernel in uncertainty_set.kernel_ids:
            estimate = estimates[(certificate.policy_id, kernel)]
            record = estimate.identification
            record.require_identified()
            if (
                estimate.policy_id != certificate.policy_id
                or record.estimand != first.estimand
                or record.trial != first.trial
                or record.treatment != first.treatment
                or record.model_id != kernel
                or record.resource_regime != resource_regime
                or estimate.discount != first_estimate.discount
                or estimate.method != record.estimator
                or not math.isfinite(estimate.value)
            ):
                raise IdentificationError("incompatible policy/kernel estimands or target trials")
            assumed |= record.status == "IDENTIFIED_UNDER_EXPLICIT_ASSUMPTIONS"
            values.append(estimate.value)
        worst.append((certificate.policy_id, min(values)))
    selected = max(worst, key=lambda pair: pair[1])
    return RobustCausalValue(
        selected[1],
        selected[0],
        tuple(worst),
        uncertainty_set.kernel_ids,
        resource_regime,
        first.estimand,
        uncertainty_set,
        certificates,
        "ROBUST_VALUE_UNDER_EXPLICIT_CAUSAL_ASSUMPTIONS"
        if assumed
        else "FINITE_MODEL_ROBUST_VALUE",
        (
            "finite policy and kernel classes only",
            "finite-sample estimation error remains",
            "external certificate references are dependencies, not certifications issued by this API",
        ),
    )

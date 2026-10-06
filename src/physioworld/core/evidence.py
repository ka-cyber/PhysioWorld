"""Assumption-gated certificates and machine-readable scientific lineage."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from .contracts import CausalStatus, EvidenceClass, VersionedRecord, finite


class AssumptionStatus(str, Enum):
    VERIFIED = "VERIFIED"
    TESTED = "TESTED"
    PLAUSIBLE = "PLAUSIBLE"
    UNCHECKABLE = "UNCHECKABLE"
    VIOLATED = "VIOLATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Assumption(VersionedRecord):
    assumption_id: str
    statement: str
    status: AssumptionStatus = AssumptionStatus.UNCHECKABLE
    mandatory: bool = True
    evidence: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", AssumptionStatus(self.status))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not self.assumption_id or not self.statement:
            raise ValueError("assumption needs an ID and statement")
        if (
            self.status in {AssumptionStatus.VERIFIED, AssumptionStatus.TESTED}
            and not self.evidence
        ):
            raise ValueError(
                f"{self.assumption_id}: {self.status.value} requires evidence references"
            )


@dataclass(frozen=True)
class AssumptionSet(VersionedRecord):
    assumptions: tuple[Assumption, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        ids = [assumption.assumption_id for assumption in self.assumptions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate assumption IDs")

    def require(self, required: tuple[str, ...] = (), *, high_confidence: bool = True) -> None:
        lookup = {assumption.assumption_id: assumption for assumption in self.assumptions}
        problems = [
            f"{name}: missing mandatory assumption" for name in required if name not in lookup
        ]
        for assumption in self.assumptions:
            if not assumption.mandatory and assumption.assumption_id not in required:
                continue
            accepted = {AssumptionStatus.VERIFIED, AssumptionStatus.TESTED}
            if (high_confidence and assumption.status not in accepted) or (
                not high_confidence and assumption.status == AssumptionStatus.VIOLATED
            ):
                problems.append(
                    f"{assumption.assumption_id}: {assumption.status.value}"
                    + (f" ({assumption.rationale})" if assumption.rationale else "")
                )
        if problems:
            raise ValueError("certificate assumption guard: " + "; ".join(problems))


@dataclass(frozen=True)
class OperatingEnvelope(VersionedRecord):
    envelope_id: str
    description: str
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    categories: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.envelope_id or not self.description:
            raise ValueError("operating envelope requires ID and description")
        object.__setattr__(
            self, "bounds", {key: tuple(value) for key, value in self.bounds.items()}
        )
        object.__setattr__(
            self, "categories", {key: tuple(value) for key, value in self.categories.items()}
        )
        for name, interval in self.bounds.items():
            if len(interval) != 2:
                raise ValueError(f"{name}: envelope bounds need two endpoints")
            low, high = interval
            finite(low, name)
            finite(high, name)
            if low > high:
                raise ValueError(f"{name}: reversed envelope bounds")
        if any(not values for values in self.categories.values()):
            raise ValueError("categorical envelopes cannot be empty")

    def contains(self, conditions: Mapping[str, float | str]) -> bool:
        if any(name not in conditions for name in (*self.bounds, *self.categories)):
            return False
        for name, (low, high) in self.bounds.items():
            value = conditions[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not low <= value <= high:
                return False
        return all(conditions[name] in allowed for name, allowed in self.categories.items())


@dataclass(frozen=True)
class ContextOfUse(VersionedRecord):
    context_id: str
    purpose: str
    population: str
    setting: str
    endpoints: tuple[str, ...]
    excluded_uses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.context_id, self.purpose, self.population, self.setting, self.endpoints)):
            raise ValueError("context of use must declare purpose, population, setting, endpoints")


class ProofStatus(str, Enum):
    UNPROVED = "UNPROVED"
    PROOF_SKETCH = "PROOF_SKETCH"
    PROVED_WITH_REFERENCE = "PROVED_WITH_REFERENCE"
    MACHINE_CHECKED = "MACHINE_CHECKED"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"


@dataclass(frozen=True)
class Claim(VersionedRecord):
    claim_id: str
    text: str
    evidence_class: EvidenceClass
    source_artifact: str
    code_path: str
    tests: tuple[str, ...]
    uncertainty: str
    causal_status: CausalStatus
    scope: str
    operating_envelope: OperatingEnvelope
    assumptions: AssumptionSet = field(default_factory=AssumptionSet)
    proof_status: ProofStatus | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "causal_status", CausalStatus(self.causal_status))
        if self.proof_status is not None:
            object.__setattr__(self, "proof_status", ProofStatus(self.proof_status))
        if not all((self.claim_id, self.text, self.source_artifact, self.code_path, self.scope)):
            raise ValueError("claim must have traceable ID, source, code, and scope")
        if self.evidence_class == EvidenceClass.THEOREM and self.proof_status is None:
            raise ValueError("theorem claims require explicit proof status")


@dataclass(frozen=True)
class Certificate(VersionedRecord):
    certificate_id: str
    claim_id: str
    guarantee: str
    evidence_class: EvidenceClass
    assumptions: AssumptionSet
    operating_envelope: OperatingEnvelope
    context_of_use: ContextOfUse
    source_artifacts: tuple[str, ...]
    high_confidence: bool = True
    failure_conditions: tuple[str, ...] = ()
    proof_status: ProofStatus | None = None
    proof_artifact: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        if not all((self.certificate_id, self.claim_id, self.guarantee, self.source_artifacts)):
            raise ValueError("certificate requires a claim, guarantee, and source artifacts")
        self.assumptions.require(high_confidence=self.high_confidence)
        if not self.high_confidence and not self.failure_conditions:
            raise ValueError("conditional certificate must explain its failure conditions")
        if self.proof_status is not None:
            object.__setattr__(self, "proof_status", ProofStatus(self.proof_status))
        if self.evidence_class == EvidenceClass.THEOREM and self.high_confidence:
            if (
                self.proof_status
                not in {ProofStatus.PROVED_WITH_REFERENCE, ProofStatus.MACHINE_CHECKED}
                or not self.proof_artifact
            ):
                raise ValueError(
                    "a high-confidence theorem certificate requires an explicit proof artifact/status"
                )


@dataclass(frozen=True)
class Counterexample(VersionedRecord):
    counterexample_id: str
    rejected_statement: str
    construction: str
    scope: str
    source_artifact: str


@dataclass(frozen=True)
class ReproductionRecord(VersionedRecord):
    reproduction_id: str
    source_artifact: str
    implementation_version: str
    configuration_digest: str
    command: str
    outcome: str
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION


CAUSAL_REQUIREMENTS = (
    "temporal_ordering",
    "consistency",
    "exchangeability_or_randomization",
    "positivity",
    "action_support",
    "safety_constraints",
    "intervention_variation",
    "valid_outcome_measurement",
)


@dataclass(frozen=True)
class IdentificationRecord(VersionedRecord):
    identification_id: str
    estimand: str
    treatment: str
    outcome: str
    time_zero: str
    eligibility: str
    treatment_strategies: tuple[str, ...]
    follow_up: str
    censoring: str
    assumptions: AssumptionSet
    observed_support: str
    estimator: str
    uncertainty: str
    sensitivity: str
    limitations: tuple[str, ...]
    status: CausalStatus = CausalStatus.UNIDENTIFIED

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CausalStatus(self.status))
        if self.status == CausalStatus.IDENTIFIED:
            self.require_identified()

    def require_identified(self) -> None:
        self.assumptions.require(CAUSAL_REQUIREMENTS)
        if not self.observed_support or not self.treatment_strategies:
            raise ValueError(
                "identification needs observed action support and treatment strategies"
            )


class EvidenceNodeKind(str, Enum):
    CLAIM = "claim"
    ASSUMPTION = "assumption"
    MODEL = "model"
    IMPLEMENTATION = "implementation"
    TEST = "test"
    RESULT = "result"
    VALIDATION = "validation"
    OPERATING_ENVELOPE = "operating_envelope"
    RELEASE = "release"


@dataclass(frozen=True)
class EvidenceNode(VersionedRecord):
    node_id: str
    kind: EvidenceNodeKind
    artifact: str
    evidence_class: EvidenceClass

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EvidenceNodeKind(self.kind))
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        if not self.node_id or not self.artifact:
            raise ValueError("evidence node requires an ID and artifact")
        if self.kind == EvidenceNodeKind.TEST and self.evidence_class == EvidenceClass.THEOREM:
            raise ValueError("a numerical/software test is not a theorem proof")


@dataclass(frozen=True)
class EvidenceEdge(VersionedRecord):
    source: str
    target: str
    relation: str = "supports"

    def __post_init__(self) -> None:
        if self.relation not in {"supports", "depends_on"}:
            raise ValueError("evidence relations must declare supports or depends_on direction")

    def dependency_direction(self) -> tuple[str, str]:
        return (
            (self.target, self.source)
            if self.relation == "depends_on"
            else (self.source, self.target)
        )


def validate_dag(node_ids: set[str], edges: tuple[tuple[str, str], ...]) -> None:
    children: dict[str, list[str]] = {name: [] for name in node_ids}
    indegree = dict.fromkeys(node_ids, 0)
    for source, target in edges:
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"dangling graph edge: {source} -> {target}")
        children[source].append(target)
        indegree[target] += 1
    queue = [name for name, degree in indegree.items() if degree == 0]
    count = 0
    while queue:
        source = queue.pop()
        count += 1
        for target in children[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if count != len(node_ids):
        raise ValueError("graph contains a directed cycle; use time-unrolled nodes for feedback")


@dataclass(frozen=True)
class EvidenceGraph(VersionedRecord):
    nodes: tuple[EvidenceNode, ...] = ()
    edges: tuple[EvidenceEdge, ...] = ()

    def __post_init__(self) -> None:
        ids = {node.node_id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate evidence node IDs")
        validate_dag(ids, tuple((edge.source, edge.target) for edge in self.edges))
        validate_dag(ids, tuple(edge.dependency_direction() for edge in self.edges))

    def affected_by(self, artifact_id: str) -> tuple[str, ...]:
        """Return impacted dependents, respecting supports versus depends_on direction."""
        if artifact_id not in {node.node_id for node in self.nodes}:
            raise KeyError(artifact_id)
        affected = {artifact_id}
        changed = True
        while changed:
            before = len(affected)
            affected.update(
                target
                for source, target in (edge.dependency_direction() for edge in self.edges)
                if source in affected
            )
            changed = len(affected) != before
        return tuple(sorted(affected - {artifact_id}))


@dataclass(frozen=True)
class TheoremRecord(VersionedRecord):
    theorem_id: str
    statement: str
    assumptions: tuple[str, ...]
    proof_status: ProofStatus
    proof_reference: str
    implementation_relation: str
    validation_tests: tuple[str, ...]
    scope: str
    counterexamples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "proof_status", ProofStatus(self.proof_status))
        if not all((self.theorem_id, self.statement, self.assumptions, self.scope)):
            raise ValueError("theorem record requires statement, assumptions and scope")
        if self.proof_status in {ProofStatus.PROVED_WITH_REFERENCE, ProofStatus.MACHINE_CHECKED}:
            if not self.proof_reference:
                raise ValueError("proved status requires a proof artifact, not a numerical test")


def theorem_registry() -> tuple[TheoremRecord, ...]:
    """Explicit conditional propositions; numerical implementations are not proofs."""
    entries = (
        (
            "bellman_optimality",
            "V_t(s)=sup_a [r_t(s,a)+E V_{t+1}(S')].",
            ("Markov state", "nonanticipative admissible policies", "measurable selectors"),
            "finite horizon dynamic programming",
            "finite-state bounded rewards",
        ),
        (
            "markov_quotient",
            "A reward/action/kernel-preserving quotient preserves optimal value.",
            ("lumpability", "reward agreement", "admissible action agreement"),
            "resource abstraction contract",
            "exact Markov quotient",
        ),
        (
            "resource_capability_dominance",
            "Policy emulation preserving values implies capability inclusion.",
            ("declared emulation for every policy", "common utility/reference"),
            "capability.resource_dominates",
            "declared policy classes",
        ),
        (
            "strategic_measure",
            "Inclusion of attainable trajectory laws preserves bounded utility supports.",
            ("common trajectory space", "all bounded measurable utility tests"),
            "capability.StrategicMeasure",
            "trajectory-law inclusion",
        ),
        (
            "requirement_frontier",
            "Inclusion of capabilities implies inclusion of downward requirement regions.",
            ("all coordinates are benefits", "common coordinate semantics"),
            "CapabilityRegion.requirement_region",
            "finite capability images",
        ),
        (
            "schedule_dominance",
            "A causal feasible schedule emulation preserves schedule-dependent value.",
            (
                "nonanticipative schedule emulation",
                "equal initial reference",
                "safety preservation",
            ),
            "declared policy relation",
            "dynamic resource schedules",
        ),
        (
            "equal_harvest_counterexample",
            "Equal total harvest need not give equal deadline capability.",
            ("energy needed before a deadline", "harvest after deadline is unusable"),
            "counterexample construction",
            "two-stage deadline task",
        ),
        (
            "marginal_nonnegative",
            "Nested feasible policy sets imply nondecreasing optimal value.",
            ("feasible-set nesting", "same reward", "no compulsory resource penalty"),
            "capability inclusion",
            "no differentiability assertion",
        ),
        (
            "information_energy_separability",
            "Information/energy separability requires explicit factorization.",
            ("factorized feasibility", "separable declared objective"),
            "hypothesis boundary",
            "conditional factorization only",
        ),
        (
            "robust_minimax",
            "sup_pi inf_theta J <= inf_theta sup_pi J.",
            ("common policies", "same uncertainty set", "finite bounded values"),
            "RobustCapabilityRegion.support/optimistic_support",
            "finite policy/kernel matrix",
        ),
        (
            "physical_design_order",
            "Physical designs are ordered only via capability-preserving simulation.",
            ("declared dynamics simulation", "safety/action/value preservation"),
            "capability.resource_dominates",
            "not coordinate energy/ESR ordering",
        ),
        (
            "value_transfer",
            "Delta_t <= eps_r,t + 2 M_{t+1} eps_P,t + Delta_{t+1}.",
            ("TV=half L1", "absolute continuation bound M", "common feasible actions"),
            "capability.finite_kernel_error_bound",
            "finite horizon",
        ),
        (
            "finite_kernel_accumulation",
            "Composed Markov TV error is bounded by the sum of local uniform TV errors.",
            ("aligned kernels", "uniform conditional error", "TV contraction"),
            "InformationFlowGraph.certify_pipeline",
            "serial common-space kernels",
        ),
        (
            "dobrushin_stability",
            "e_{t+1} <= alpha e_t + eps; alpha<1 gives limsup e <= eps/(1-alpha).",
            ("TV=half L1", "uniform kernel perturbation", "Dobrushin alpha<1"),
            "capability.dobrushin_stability_bound",
            "distribution error, not clinical safety",
        ),
    )
    test_prefix = "tests/test_contracts_capability.py::"
    validation_tests = {
        "resource_capability_dominance": "CapabilityTests::test_resource_dominance_requires_policy_relation_not_resource_coordinates",
        "strategic_measure": "CapabilityTests::test_blackwell_garbling_and_strategic_measure",
        "requirement_frontier": "CapabilityTests::test_coordinate_maxima_not_joint_and_downward_closure",
        "equal_harvest_counterexample": "CapabilityTests::test_equal_total_harvest_does_not_preserve_deadline_capability",
        "robust_minimax": "CapabilityTests::test_common_policy_robustness_differs_from_oracle",
        "physical_design_order": "CapabilityTests::test_resource_dominance_requires_policy_relation_not_resource_coordinates",
        "value_transfer": "FlowAndBoundTests::test_finite_value_transfer_coefficient_matches_half_l1_tv",
        "finite_kernel_accumulation": "FlowAndBoundTests::test_real_composition_uses_contraction_not_blanket_sum",
        "dobrushin_stability": "FlowAndBoundTests::test_kernel_validation_and_dobrushin_analytic_bound",
    }
    return tuple(
        TheoremRecord(
            identifier,
            statement,
            assumptions,
            ProofStatus.COUNTEREXAMPLE
            if identifier == "equal_harvest_counterexample"
            else ProofStatus.UNPROVED,
            "Construction: one energy unit arriving before versus after the required action."
            if identifier == "equal_harvest_counterexample"
            else "Conditional proposition recorded for audit; no machine-checked proof artifact supplied.",
            implementation,
            (test_prefix + validation_tests[identifier],) if identifier in validation_tests else (),
            scope,
            ("componentwise resource ordering is insufficient",)
            if identifier in {"physical_design_order", "resource_capability_dominance"}
            else (),
        )
        for identifier, statement, assumptions, implementation, scope in entries
    )


def scientific_claim_registry() -> tuple[Claim, ...]:
    """Reported context from the specification, without fabricating reproductions."""
    descriptions = (
        (
            "temporal_history",
            "Recent history improves the declared forecasting tasks in the supplied corpus.",
        ),
        ("task_dependence", "Task performance depends on the declared task family."),
        ("crossing_tasks", "Compact representations can cross in task performance."),
        ("mean_vs_worst", "Mean and worst-task rankings can disagree."),
        (
            "semantic_decode",
            "Semantic permutation can destroy task preservation despite successful decode.",
        ),
        ("resource_conditioning", "Attainable capability depends on the declared resource regime."),
        ("predictive_not_causal", "Predictive transport is distinct from causal identification."),
        ("model_not_measurement", "Model-derived resource values are not hardware measurements."),
        (
            "independent_support",
            "Policy-optimized support gaps optimize both policy sets independently.",
        ),
    )
    envelope = OperatingEnvelope("specification-context", "Declared tasks, cohorts and models only")
    return tuple(
        Claim(
            f"PW-{identifier}",
            text,
            EvidenceClass.INTERPRETATION,
            "prompt 2.md, section 168",
            "physioworld.core.evidence.scientific_claim_registry",
            (),
            "Historical corpus not loaded or re-estimated by this registry.",
            CausalStatus.NOT_APPLICABLE,
            "Supplied scientific context; no new empirical result",
            envelope,
        )
        for identifier, text in descriptions
    )

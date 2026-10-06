"""Typed information-flow DAGs and finite-kernel composition certificates.

The implemented composition is a total-variation kernel perturbation bound,
not addition of unrelated reconstruction, AUC, packet or task-value losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .contracts import CausalStatus, EvidenceClass, VersionedRecord, finite
from .evidence import Assumption, AssumptionSet, OperatingEnvelope, validate_dag


class FlowNodeType(str, Enum):
    STATE = "state"
    OBSERVATION = "observation"
    HISTORY = "history"
    REPRESENTATION = "representation"
    PREDICTION = "prediction"
    TASK_LAW = "task_law"
    CAPABILITY = "capability"
    RESOURCE = "resource"
    POLICY = "policy"
    INTERVENTION = "intervention"
    ENDPOINT = "endpoint"


class FlowEdgeType(str, Enum):
    SENSING = "sensing"
    FILTERING = "filtering"
    PREPROCESSING = "preprocessing"
    COMPRESSION = "compression"
    QUANTIZATION = "quantization"
    SERIALIZATION = "serialization"
    PACKETIZATION = "packetization"
    TRANSMISSION = "transmission"
    DECODING = "decoding"
    RECONSTRUCTION = "reconstruction"
    INFERENCE = "inference"
    POLICY_TRANSFORMATION = "policy_transformation"
    RESOURCE_TRANSITION = "resource_transition"


@dataclass(frozen=True)
class FlowNode(VersionedRecord):
    node_id: str
    kind: FlowNodeType
    semantic_space: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", FlowNodeType(self.kind))
        if not self.node_id or not self.semantic_space:
            raise ValueError("flow node needs an ID and semantic space")


@dataclass(frozen=True)
class FiniteKernel(VersionedRecord):
    input_space: str
    output_space: str
    input_states: tuple[str, ...]
    output_states: tuple[str, ...]
    probabilities: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_states", tuple(self.input_states))
        object.__setattr__(self, "output_states", tuple(self.output_states))
        object.__setattr__(self, "probabilities", tuple(tuple(row) for row in self.probabilities))
        if (
            not self.input_space
            or not self.output_space
            or not self.input_states
            or not self.output_states
        ):
            raise ValueError("kernel requires named nonempty input and output spaces")
        if len(set(self.input_states)) != len(self.input_states) or len(
            set(self.output_states)
        ) != len(self.output_states):
            raise ValueError("kernel state labels must be unique")
        if len(self.probabilities) != len(self.input_states):
            raise ValueError("kernel row count does not match input states")
        for row in self.probabilities:
            if len(row) != len(self.output_states):
                raise ValueError("kernel column count does not match output states")
            for probability in row:
                finite(probability, "kernel probability")
            if (
                any(value < 0 or value > 1 for value in row)
                or abs(sum(float(value) for value in row) - 1) > 1e-12
            ):
                raise ValueError("kernel rows must be probability distributions")
        object.__setattr__(
            self,
            "probabilities",
            tuple(tuple(float(value) for value in row) for row in self.probabilities),
        )

    def uniform_tv_error(self, reference: "FiniteKernel") -> float:
        if (self.input_space, self.output_space, self.input_states, self.output_states) != (
            reference.input_space,
            reference.output_space,
            reference.input_states,
            reference.output_states,
        ):
            raise ValueError("kernel comparisons need identical ordered state spaces")
        return max(
            0.5 * sum(abs(p - q) for p, q in zip(row, ref))
            for row, ref in zip(self.probabilities, reference.probabilities)
        )

    def dobrushin_coefficient(self) -> float:
        return max(
            0.5 * sum(abs(p - q) for p, q in zip(left, right))
            for left in self.probabilities
            for right in self.probabilities
        )


@dataclass(frozen=True)
class LocalGuarantee(VersionedRecord):
    metric: str
    bound: float
    domain: str
    assumptions: AssumptionSet = field(default_factory=AssumptionSet)
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        finite(self.bound, "local guarantee bound")
        if self.bound < 0 or not self.metric or not self.domain:
            raise ValueError("local guarantee needs a nonnegative bound, metric and domain")
        if self.metric == "uniform_kernel_tv" and self.bound > 1:
            raise ValueError("half-L1 total variation cannot exceed one")


@dataclass(frozen=True)
class FlowEdge(VersionedRecord):
    edge_id: str
    source: str
    target: str
    kind: FlowEdgeType
    transformation: str
    stochastic_kernel: FiniteKernel | None = None
    reference_kernel: FiniteKernel | None = None
    latency_s: float = 0.0
    resource_cost: Mapping[str, float] = field(default_factory=dict)
    uncertainty: str = "none_declared"
    local_guarantee: LocalGuarantee | None = None
    evidence_class: EvidenceClass = EvidenceClass.MODEL
    assumptions: AssumptionSet = field(default_factory=AssumptionSet)
    causal_status: CausalStatus = CausalStatus.NOT_APPLICABLE

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", FlowEdgeType(self.kind))
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "causal_status", CausalStatus(self.causal_status))
        finite(self.latency_s, "edge latency")
        if self.latency_s < 0 or not self.edge_id or not self.transformation:
            raise ValueError("edge requires a transformation and nonnegative latency")
        for value in self.resource_cost.values():
            finite(value, "resource cost")
            if value < 0:
                raise ValueError("edge resource costs cannot be negative")
        if self.reference_kernel is not None and self.stochastic_kernel is None:
            raise ValueError("reference kernel without actual kernel")


@dataclass(frozen=True)
class ScientificPipelineCertificate(VersionedRecord):
    certificate_id: str
    edge_ids: tuple[str, ...]
    local_guarantees: tuple[LocalGuarantee, ...]
    assumptions: AssumptionSet
    composition_rule: str
    resulting_guarantee: LocalGuarantee
    failure_conditions: tuple[str, ...]
    validity_domain: OperatingEnvelope
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION
    contraction_coefficients: tuple[float, ...] = ()
    initial_tv_error: float = 0.0

    def __post_init__(self) -> None:
        self.assumptions.require()
        if not self.edge_ids or len(self.edge_ids) != len(self.local_guarantees):
            raise ValueError("pipeline certificate needs a guarantee for every edge")
        if len(self.contraction_coefficients) != len(self.edge_ids):
            raise ValueError(
                "pipeline certificate must retain its reference-kernel contraction coefficients"
            )
        finite(self.initial_tv_error, "initial TV error")
        if not 0 <= self.initial_tv_error <= 1:
            raise ValueError("certificate initial TV error must lie in [0,1]")
        bound = self.initial_tv_error
        for guarantee, contraction in zip(self.local_guarantees, self.contraction_coefficients):
            guarantee.assumptions.require()
            if guarantee.metric != "uniform_kernel_tv":
                raise ValueError("pipeline certificate cannot compose unrelated local loss metrics")
            finite(contraction, "kernel contraction")
            if not 0 <= contraction <= 1:
                raise ValueError("Dobrushin contraction must lie in [0,1]")
            bound = min(1.0, guarantee.bound + contraction * bound)
        if (
            self.resulting_guarantee.metric != "output_distribution_tv"
            or abs(self.resulting_guarantee.bound - bound) > 1e-12
        ):
            raise ValueError("pipeline certificate has an inconsistent composition bound")


@dataclass(frozen=True)
class InformationFlowGraph(VersionedRecord):
    nodes: tuple[FlowNode, ...]
    edges: tuple[FlowEdge, ...]

    def __post_init__(self) -> None:
        node_map = {node.node_id: node for node in self.nodes}
        if len(node_map) != len(self.nodes) or len({edge.edge_id for edge in self.edges}) != len(
            self.edges
        ):
            raise ValueError("flow node and edge IDs must be unique")
        validate_dag(set(node_map), tuple((edge.source, edge.target) for edge in self.edges))
        for edge in self.edges:
            source, target = node_map[edge.source], node_map[edge.target]
            if edge.kind == FlowEdgeType.SENSING and (source.kind, target.kind) != (
                FlowNodeType.STATE,
                FlowNodeType.OBSERVATION,
            ):
                raise ValueError("sensing must connect a state to an observation")
            if edge.kind == FlowEdgeType.RESOURCE_TRANSITION and (
                source.kind != FlowNodeType.RESOURCE or target.kind != FlowNodeType.RESOURCE
            ):
                raise ValueError("resource transition must connect resource nodes")
            for kernel in (edge.stochastic_kernel, edge.reference_kernel):
                if kernel and (
                    kernel.input_space != source.semantic_space
                    or kernel.output_space != target.semantic_space
                ):
                    raise ValueError(f"{edge.edge_id}: kernel/node semantic spaces differ")

    def certify_pipeline(
        self,
        edge_ids: Sequence[str],
        *,
        certificate_id: str,
        validity_domain: OperatingEnvelope,
        initial_tv_error: float = 0.0,
    ) -> ScientificPipelineCertificate:
        """Derive e_next <= eps + alpha(reference) e for a serial finite-kernel path.

        Triangle inequality splits the perturbation into the uniform local error
        and the reference-kernel contraction. TV uses half L1. Independence of
        error events is not assumed. Feedback must be explicitly time-unrolled.
        """
        finite(initial_tv_error, "initial TV error")
        if not 0 <= initial_tv_error <= 1:
            raise ValueError("initial TV error must lie in [0, 1]")
        lookup = {edge.edge_id: edge for edge in self.edges}
        if not edge_ids or len(set(edge_ids)) != len(edge_ids):
            raise ValueError("pipeline requires a nonempty path without repeated edges")
        path = [lookup[name] for name in edge_ids]
        guarantees = []
        contractions = []
        assumptions: dict[str, Assumption] = {}
        bound = initial_tv_error
        previous_kernel: FiniteKernel | None = None
        output_space = ""
        for index, edge in enumerate(path):
            if index and path[index - 1].target != edge.source:
                raise ValueError("pipeline edges do not form a connected serial path")
            actual, reference, guarantee = (
                edge.stochastic_kernel,
                edge.reference_kernel,
                edge.local_guarantee,
            )
            if actual is None or reference is None or guarantee is None:
                raise ValueError(
                    f"{edge.edge_id}: TV composition needs actual/reference kernels and a local guarantee"
                )
            if guarantee.metric != "uniform_kernel_tv" or guarantee.domain != actual.input_space:
                raise ValueError(
                    f"{edge.edge_id}: incompatible local loss descriptor for TV composition"
                )
            if previous_kernel and previous_kernel.output_states != actual.input_states:
                raise ValueError("pipeline state labels/order do not align")
            error = actual.uniform_tv_error(reference)
            if error > guarantee.bound + 1e-12:
                raise ValueError(
                    f"{edge.edge_id}: supplied guarantee understates uniform TV error {error}"
                )
            edge.assumptions.require()
            guarantee.assumptions.require()
            for assumption in (*edge.assumptions.assumptions, *guarantee.assumptions.assumptions):
                if (
                    assumption.assumption_id in assumptions
                    and assumptions[assumption.assumption_id] != assumption
                ):
                    raise ValueError("conflicting pipeline assumptions")
                assumptions[assumption.assumption_id] = assumption
            bound = min(1.0, guarantee.bound + reference.dobrushin_coefficient() * bound)
            guarantees.append(guarantee)
            contractions.append(reference.dobrushin_coefficient())
            previous_kernel = actual
            output_space = reference.output_space
        assumption_set = AssumptionSet(tuple(assumptions.values()))
        return ScientificPipelineCertificate(
            certificate_id,
            tuple(edge_ids),
            tuple(guarantees),
            assumption_set,
            "e_next <= uniform_kernel_tv + Dobrushin(reference_kernel)*e; TV=half_L1; cap at 1",
            LocalGuarantee(
                "output_distribution_tv",
                bound,
                output_space,
                assumption_set,
                description="Uniform finite-kernel perturbation bound",
            ),
            (
                "kernels or aligned state spaces change",
                "mandatory assumptions cease to hold",
                "initial input TV exceeds the declared bound",
                "unmodeled feedback or history dependence",
            ),
            validity_domain,
            contraction_coefficients=tuple(contractions),
            initial_tv_error=initial_tv_error,
        )

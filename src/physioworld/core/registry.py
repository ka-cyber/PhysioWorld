"""Typed model-family registry and scientific change-impact governance.

The registry is intentionally independent of a model runtime.  It records what
an implementation claims to be, its schemas and envelope, and the evidence
that governs it.  Registering a new version never mutates an older descriptor
and refuses incompatible API changes hidden behind the old version.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .contracts import EvidenceClass, VersionedRecord
from .evidence import (
    ContextOfUse,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeKind,
    OperatingEnvelope,
)


class ModelGovernanceError(ValueError):
    """Raised when model identity, versioning, or compatibility is invalid."""


class ModelFamily(str, Enum):
    PHYSIOLOGY = "physiology"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    COMMUNICATION = "communication"
    RESOURCE = "resource"
    POLICY = "policy"


class ModelType(str, Enum):
    MECHANISTIC = "mechanistic"
    STATE_SPACE = "state_space"
    REGRESSION = "regression"
    CLASSIFIER = "classifier"
    FORECASTER = "forecaster"
    NEURAL = "neural"
    CHANNEL = "channel"
    RESOURCE_DYNAMICS = "resource_dynamics"
    CONTROLLER = "controller"
    REDUCED_ORDER = "reduced_order"


class DeprecationStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True, order=True)
class SemanticVersion(VersionedRecord):
    """Strict major/minor/patch version used by governance compatibility rules."""

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0 for value in (self.major, self.minor, self.patch)
        ):
            raise ValueError("semantic version components must be nonnegative integers")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, value: str) -> "SemanticVersion":
        match = _SEMVER.fullmatch(value)
        if match is None:
            raise ValueError(f"invalid semantic version: {value!r}")
        return cls(*(int(part) for part in match.groups()))


@dataclass(frozen=True)
class DeprecationPolicy(VersionedRecord):
    """Explicit lifecycle status; retirement is never inferred from age."""

    status: DeprecationStatus = DeprecationStatus.ACTIVE
    reason: str = ""
    replacement_model_id: str | None = None
    sunset_version: SemanticVersion | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", DeprecationStatus(self.status))
        if self.status != DeprecationStatus.ACTIVE and not self.reason:
            raise ValueError("deprecated/retired models require a reason")
        if self.status == DeprecationStatus.DEPRECATED and not (
            self.replacement_model_id or self.sunset_version
        ):
            raise ValueError("deprecated models require a replacement or sunset version")


@dataclass(frozen=True)
class ModelDescriptor(VersionedRecord):
    """Complete typed descriptor for a governed scientific model."""

    model_id: str
    family: ModelFamily
    model_type: ModelType
    version: SemanticVersion
    equations_or_algorithm: str
    assumptions: tuple[str, ...]
    parameters: Mapping[str, Any]
    input_schema: Mapping[str, str]
    output_schema: Mapping[str, str]
    operating_envelope: OperatingEnvelope
    context_of_use: ContextOfUse
    evidence_class: EvidenceClass
    evidence_refs: tuple[str, ...] = ()
    affected_claims: tuple[str, ...] = ()
    affected_tests: tuple[str, ...] = ()
    affected_benchmarks: tuple[str, ...] = ()
    release_tag: str | None = None
    deprecation: DeprecationPolicy = field(default_factory=DeprecationPolicy)

    def __post_init__(self) -> None:
        if isinstance(self.version, str):
            object.__setattr__(self, "version", SemanticVersion.parse(self.version))
        object.__setattr__(self, "family", ModelFamily(self.family))
        object.__setattr__(self, "model_type", ModelType(self.model_type))
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(self, "affected_claims", tuple(self.affected_claims))
        object.__setattr__(self, "affected_tests", tuple(self.affected_tests))
        object.__setattr__(self, "affected_benchmarks", tuple(self.affected_benchmarks))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "input_schema", dict(self.input_schema))
        object.__setattr__(self, "output_schema", dict(self.output_schema))
        if not self.model_id or not self.equations_or_algorithm:
            raise ValueError("model descriptor needs an ID and algorithm/equations")
        if not self.assumptions:
            raise ValueError("model descriptor needs explicit assumptions")
        if not self.input_schema or not self.output_schema:
            raise ValueError("model descriptor needs input and output schemas")
        if self.release_tag is not None and not self.release_tag:
            raise ValueError("release_tag cannot be empty")
        if any(
            not isinstance(key, str) or not key for key in (*self.input_schema, *self.output_schema)
        ):
            raise ValueError("model schema field names must be nonempty strings")
        if any(
            not isinstance(value, str) or not value
            for value in (*self.input_schema.values(), *self.output_schema.values())
        ):
            raise ValueError("model schema types must be nonempty strings")

    @classmethod
    def with_version(cls, version: str, **kwargs: Any) -> "ModelDescriptor":
        return cls(version=SemanticVersion.parse(version), **kwargs)

    def content_hash(self) -> str:
        payload = self.to_dict()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @property
    def algorithm(self) -> str:
        """Concise API name for the declared equations/algorithm text."""

        return self.equations_or_algorithm

    @property
    def params(self) -> Mapping[str, Any]:
        return self.parameters

    @property
    def model_family(self) -> ModelFamily:
        return self.family

    @property
    def evidence(self) -> EvidenceClass:
        return self.evidence_class

    @property
    def envelope(self) -> OperatingEnvelope:
        return self.operating_envelope

    @property
    def context(self) -> ContextOfUse:
        return self.context_of_use

    def api_signature(self) -> tuple[Any, ...]:
        return (
            self.family.value,
            self.model_type.value,
            tuple(sorted(self.input_schema.items())),
            tuple(sorted(self.output_schema.items())),
        )

    def api_compatible_with(self, previous: "ModelDescriptor") -> bool:
        return self.api_signature() == previous.api_signature()


@dataclass(frozen=True)
class RegisteredModel(VersionedRecord):
    descriptor: ModelDescriptor
    content_hash: str

    def __post_init__(self) -> None:
        if self.content_hash != self.descriptor.content_hash():
            raise ModelGovernanceError("registered model hash does not match descriptor content")


@dataclass(frozen=True)
class ChangeImpact(VersionedRecord):
    """Machine-readable impact assessment emitted for a model version change."""

    model_id: str
    previous_version: SemanticVersion | None
    current_version: SemanticVersion
    previous_hash: str | None
    current_hash: str
    changed_fields: tuple[str, ...]
    affected_claims: tuple[str, ...]
    affected_tests: tuple[str, ...]
    affected_benchmarks: tuple[str, ...]
    api_compatible: bool
    results_invalidated: bool
    invalidation_reason: str
    evidence_graph: EvidenceGraph


def _changed_fields(previous: ModelDescriptor, current: ModelDescriptor) -> tuple[str, ...]:
    fields = (
        "family",
        "model_type",
        "equations_or_algorithm",
        "assumptions",
        "parameters",
        "input_schema",
        "output_schema",
        "operating_envelope",
        "context_of_use",
        "evidence_class",
    )
    return tuple(name for name in fields if getattr(previous, name) != getattr(current, name))


def _impact_graph(
    previous_hash: str | None,
    current: ModelDescriptor,
    claims: tuple[str, ...],
    tests: tuple[str, ...],
    benchmarks: tuple[str, ...],
) -> EvidenceGraph:
    current_hash = current.content_hash()
    nodes = [
        EvidenceNode(
            current_hash,
            EvidenceNodeKind.MODEL,
            f"model:{current.model_id}@{current.version}",
            current.evidence_class,
        )
    ]
    edges: list[EvidenceEdge] = []
    if previous_hash is not None and previous_hash != current_hash:
        nodes.append(
            EvidenceNode(
                previous_hash,
                EvidenceNodeKind.MODEL,
                f"model:{current.model_id}@previous",
                current.evidence_class,
            )
        )
        edges.append(EvidenceEdge(previous_hash, current_hash, "depends_on"))
    for kind, identifiers, evidence_class in (
        (EvidenceNodeKind.CLAIM, claims, EvidenceClass.INTERPRETATION),
        (EvidenceNodeKind.TEST, tests, EvidenceClass.NUMERICAL_VERIFICATION),
        (EvidenceNodeKind.RESULT, benchmarks, EvidenceClass.EMPIRICAL_ESTIMATE),
    ):
        for identifier in identifiers:
            node_id = f"{kind.value}:{identifier}"
            nodes.append(EvidenceNode(node_id, kind, identifier, evidence_class))
            edges.append(EvidenceEdge(current_hash, node_id, "supports"))
    return EvidenceGraph(tuple(nodes), tuple(edges))


@dataclass
class ModelRegistry(VersionedRecord):
    """Versioned registry with append-only model lineage per ``model_id``."""

    models: tuple[RegisteredModel, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(self.models))
        keys = [(entry.descriptor.model_id, entry.descriptor.version) for entry in self.models]
        if len(keys) != len(set(keys)):
            raise ModelGovernanceError("duplicate model ID/version in registry")
        for entry in self.models:
            if entry.content_hash != entry.descriptor.content_hash():
                raise ModelGovernanceError("registry contains an invalid model hash")

    def _lineage(self, model_id: str) -> tuple[RegisteredModel, ...]:
        return tuple(
            sorted(
                (entry for entry in self.models if entry.descriptor.model_id == model_id),
                key=lambda entry: entry.descriptor.version,
            )
        )

    def latest(self, model_id: str) -> RegisteredModel:
        lineage = self._lineage(model_id)
        if not lineage:
            raise KeyError(model_id)
        return lineage[-1]

    def get(self, model_id: str, version: str | SemanticVersion | None = None) -> RegisteredModel:
        if version is None:
            return self.latest(model_id)
        selected = SemanticVersion.parse(version) if isinstance(version, str) else version
        for entry in self.models:
            if entry.descriptor.model_id == model_id and entry.descriptor.version == selected:
                return entry
        raise KeyError(f"{model_id}@{selected}")

    def register(self, descriptor: ModelDescriptor) -> ChangeImpact:
        """Append a descriptor after enforcing semver/API compatibility.

        The registry is mutated only after all checks pass.  Re-registering the
        exact same descriptor is idempotent and emits a no-change impact.
        """

        current_hash = descriptor.content_hash()
        existing = self._lineage(descriptor.model_id)
        previous = existing[-1] if existing else None
        if previous is not None and descriptor.version < previous.descriptor.version:
            raise ModelGovernanceError("model versions must be monotonically increasing")
        if previous is not None and descriptor.version == previous.descriptor.version:
            if current_hash != previous.content_hash:
                raise ModelGovernanceError(
                    "a model version cannot be reused with different content"
                )
            return self._make_impact(previous, descriptor, ())
        changed = _changed_fields(previous.descriptor, descriptor) if previous else ()
        api_compatible = previous is None or descriptor.api_compatible_with(previous.descriptor)
        if (
            previous is not None
            and not api_compatible
            and descriptor.version.major == previous.descriptor.version.major
        ):
            raise ModelGovernanceError(
                "incompatible input/output/model-family change requires a semantic major-version bump"
            )
        entry = RegisteredModel(descriptor, current_hash)
        object.__setattr__(self, "models", self.models + (entry,))
        return self._make_impact(previous, descriptor, changed)

    def _make_impact(
        self,
        previous: RegisteredModel | None,
        descriptor: ModelDescriptor,
        changed: tuple[str, ...],
    ) -> ChangeImpact:
        claims = descriptor.affected_claims
        tests = descriptor.affected_tests
        benchmarks = descriptor.affected_benchmarks
        scientific_change = any(
            name in changed
            for name in (
                "equations_or_algorithm",
                "assumptions",
                "parameters",
                "input_schema",
                "output_schema",
                "operating_envelope",
                "context_of_use",
                "family",
                "model_type",
            )
        )
        invalidated = previous is not None and scientific_change
        reason = (
            "scientific kernel or validity contract changed; prior results require re-evaluation"
            if invalidated
            else "no prior scientific result invalidation declared"
        )
        return ChangeImpact(
            descriptor.model_id,
            previous.descriptor.version if previous else None,
            descriptor.version,
            previous.content_hash if previous else None,
            descriptor.content_hash(),
            changed,
            claims,
            tests,
            benchmarks,
            previous is None or descriptor.api_compatible_with(previous.descriptor),
            invalidated,
            reason,
            _impact_graph(
                previous.content_hash
                if previous and previous.content_hash != descriptor.content_hash()
                else None,
                descriptor,
                claims,
                tests,
                benchmarks,
            ),
        )

    def compatible(self, model_id: str, version: str | SemanticVersion) -> bool:
        selected = self.get(model_id, version)
        latest = self.latest(model_id)
        return selected.descriptor.api_compatible_with(latest.descriptor)

    def require_active(
        self, model_id: str, version: str | SemanticVersion | None = None
    ) -> RegisteredModel:
        """Resolve a descriptor while enforcing its explicit lifecycle status."""

        selected = self.get(model_id, version)
        if selected.descriptor.deprecation.status != DeprecationStatus.ACTIVE:
            raise ModelGovernanceError(
                f"{model_id}@{selected.descriptor.version} is "
                f"{selected.descriptor.deprecation.status.value}"
            )
        return selected


__all__ = [
    "ChangeImpact",
    "DeprecationPolicy",
    "DeprecationStatus",
    "ModelDescriptor",
    "ModelFamily",
    "ModelGovernanceError",
    "ModelRegistry",
    "ModelType",
    "RegisteredModel",
    "SemanticVersion",
]

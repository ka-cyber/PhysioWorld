"""Serializable experiment result contract with complete scientific context."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from physioworld.core.contracts import EvidenceClass, VersionedRecord
from physioworld.core.evidence import OperatingEnvelope


def _freeze(value: Any) -> Any:
    """Canonical immutable JSON tree, matching VersionedRecord's Any decoder."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("result mappings require string keys")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        from physioworld.core.contracts import finite

        finite(value, "result value")
        return value
    raise ValueError(f"result contains a non-JSON value: {type(value).__name__}")


@dataclass(frozen=True)
class ExperimentResult(VersionedRecord):
    result_id: str
    software_version: str
    commit: str | None
    experiment_id: str
    dataset_manifest: Mapping[str, Any]
    task: Mapping[str, Any]
    estimand: str
    model: Mapping[str, Any]
    representation: Mapping[str, Any]
    channel: Mapping[str, Any]
    resource: Mapping[str, Any]
    policy: Mapping[str, Any]
    seed: int
    uncertainty: Mapping[str, Any]
    evidence_class: EvidenceClass
    operating_envelope: OperatingEnvelope
    provenance: Mapping[str, Any]
    outputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        if not all(
            isinstance(value, str) and value
            for value in (self.result_id, self.software_version, self.experiment_id, self.estimand)
        ):
            raise ValueError("experiment results require IDs, software version and an estimand")
        if self.commit is not None and (not isinstance(self.commit, str) or not self.commit):
            raise ValueError("result commit must be a nonempty string or null")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("result seed must be an integer")
        for name in (
            "dataset_manifest",
            "task",
            "model",
            "representation",
            "channel",
            "resource",
            "policy",
            "uncertainty",
            "provenance",
            "outputs",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"result {name} must be a mapping")
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        if not isinstance(self.operating_envelope, OperatingEnvelope):
            raise ValueError("result operating envelope must be an OperatingEnvelope")

"""Canonical record and event model independent of any dataset format."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .stream import SignalStream
from .time import TimeMetadata


@dataclass(frozen=True)
class RecordEvent:
    onset: float
    label: str
    duration: float = 0.0
    source: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    clock: TimeMetadata = field(default_factory=TimeMetadata)
    kind: str = "event"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.onset)
            or not math.isfinite(self.duration)
            or self.duration < 0
            or not self.label
        ):
            raise ValueError(
                "record events require finite timing, non-negative duration, and a label"
            )


@dataclass(frozen=True)
class PhysiologicalRecord:
    subject_id: str
    record_id: str
    session_id: str | None
    streams: Mapping[str, SignalStream]
    events: tuple[RecordEvent, ...] = ()
    clinical_labels: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    site_id: str | None = None
    device_id: str | None = None
    subject: Mapping[str, Any] = field(default_factory=dict)
    session: Mapping[str, Any] = field(default_factory=dict)
    devices: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    images: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subject_id or not self.record_id:
            raise ValueError("subject_id and record_id are required")
        if not self.streams and not self.images:
            raise ValueError("a physiological record requires streams or imaging metadata")
        for modality, stream in self.streams.items():
            if stream.metadata.record_id not in {None, self.record_id}:
                raise ValueError(f"stream {modality} has a conflicting record_id")
            if stream.metadata.subject_id not in {None, self.subject_id}:
                raise ValueError(f"stream {modality} has a conflicting subject_id")
            if stream.metadata.session_id not in {None, self.session_id}:
                raise ValueError(f"stream {modality} has a conflicting session_id")
            if self.site_id is not None and stream.metadata.site_id not in {None, self.site_id}:
                raise ValueError(f"stream {modality} has a conflicting site_id")

    def to_dict(self) -> dict[str, Any]:
        from physioworld.io.canonical import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysiologicalRecord":
        from physioworld.io.canonical import from_dict

        result = from_dict(payload)
        if not isinstance(result, cls):
            raise ValueError("expected a canonical record")
        return result

    def semantic_equal(self, other: object) -> bool:
        return isinstance(other, PhysiologicalRecord) and self.to_dict() == other.to_dict()

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.streams)


@dataclass
class RecordCatalog:
    records: list[PhysiologicalRecord] = field(default_factory=list)

    def add(self, record: PhysiologicalRecord) -> None:
        if any(existing.record_id == record.record_id for existing in self.records):
            raise ValueError(f"duplicate record_id: {record.record_id}")
        self.records.append(record)

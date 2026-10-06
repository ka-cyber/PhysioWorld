"""Framed packets and separately typed transport and task-semantic results.

Times are simulation seconds. CRC32 detects accidental corruption, not malicious
modification. A packet's timestamp is acquisition time, never its sequence ID.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ._validation import finite, integer, probability

_PREFIX = struct.Struct("!4sBII")
_CRC = struct.Struct("!I")


@dataclass(frozen=True)
class PacketHeader:
    protocol_version: int
    source_id: str
    sequence: int
    timestamp: float
    modality: str | None = None
    representation: str | None = None
    payload_length: int = 0

    def __post_init__(self) -> None:
        if integer(self.protocol_version, "protocol_version", 1) != 1:
            raise ValueError("unsupported packet protocol version")
        integer(self.sequence, "sequence")
        finite(self.timestamp, "timestamp", 0)
        integer(self.payload_length, "payload_length")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string")
        for name in ("modality", "representation"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("packet metadata must be finite JSON data") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


@dataclass(frozen=True)
class Packet:
    sequence: int
    payload: bytes
    timestamp: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    header: PacketHeader | None = None

    def __post_init__(self) -> None:
        integer(self.sequence, "packet sequence")
        finite(self.timestamp, "packet timestamp", 0)
        if not isinstance(self.payload, bytes):
            raise ValueError("packet payload must be bytes")
        if not isinstance(self.metadata, Mapping) or any(
            not isinstance(key, str) for key in self.metadata
        ):
            raise ValueError("metadata must be a string-keyed mapping")
        # Snapshot input metadata; callers cannot mutate the top-level mapping.
        metadata = json.loads(_json_bytes(dict(self.metadata)))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        header = (
            self.header
            if self.header is not None
            else PacketHeader(
                1, "default", self.sequence, self.timestamp, payload_length=len(self.payload)
            )
        )
        if not isinstance(header, PacketHeader) or (
            header.sequence != self.sequence
            or header.payload_length != len(self.payload)
            or header.timestamp != self.timestamp
        ):
            raise ValueError("packet header does not match sequence, timestamp or payload length")
        object.__setattr__(self, "header", header)

    @property
    def payload_length(self) -> int:
        return len(self.payload)

    @property
    def wire_length(self) -> int:
        return len(self.serialize())

    @property
    def checksum(self) -> int:
        return _CRC.unpack(self.serialize()[-_CRC.size :])[0]

    def serialize(self) -> bytes:
        """PWPK/version/header length/payload length/JSON header/payload/CRC32."""
        header_value = self.header
        if header_value is None:
            raise RuntimeError("packet header invariant was not established")
        header = _json_bytes({"header": asdict(header_value), "metadata": dict(self.metadata)})
        try:
            frame = _PREFIX.pack(b"PWPK", 1, len(header), len(self.payload))
        except struct.error as exc:
            raise ValueError("packet exceeds framing length limits") from exc
        frame += header + self.payload
        return frame + _CRC.pack(zlib.crc32(frame))

    @classmethod
    def deserialize(cls, frame: bytes) -> "Packet":
        if not isinstance(frame, bytes) or len(frame) < _PREFIX.size + _CRC.size:
            raise ValueError("truncated packet frame")
        magic, version, header_size, payload_size = _PREFIX.unpack_from(frame)
        if magic != b"PWPK" or version != 1:
            raise ValueError("invalid packet framing or version")
        if len(frame) != _PREFIX.size + header_size + payload_size + _CRC.size:
            raise ValueError("packet frame length mismatch")
        if zlib.crc32(frame[: -_CRC.size]) != _CRC.unpack(frame[-_CRC.size :])[0]:
            raise ValueError("packet checksum mismatch")
        end = _PREFIX.size + header_size
        try:
            document = json.loads(frame[_PREFIX.size : end], object_pairs_hook=_unique_object)
            if set(document) != {"header", "metadata"}:
                raise ValueError("invalid frame fields")
            header = PacketHeader(**document["header"])
            return cls(
                header.sequence,
                frame[end : -_CRC.size],
                header.timestamp,
                document["metadata"],
                header,
            )
        except (TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed packet header") from exc


@dataclass(frozen=True)
class TransmissionEvent:
    sequence: int
    attempt: int
    departure: float
    arrival: float | None
    status: str
    packet: Packet | None = None
    channel_state: str | None = None
    duplicate: bool = False

    def __post_init__(self) -> None:
        integer(self.sequence, "event sequence")
        integer(self.attempt, "attempt")
        finite(self.departure, "departure", 0)
        if self.arrival is not None:
            finite(self.arrival, "arrival", self.departure)
        if self.status not in {
            "delivered",
            "lost",
            "erased",
            "corrupt",
            "deadline",
            "queue_full",
            "expired",
        }:
            raise ValueError("unknown transmission event status")
        if self.status == "delivered":
            if self.packet is None or self.packet.sequence != self.sequence or self.arrival is None:
                raise ValueError("a delivered event requires its packet and arrival time")
        elif self.packet is not None:
            raise ValueError("unavailable events cannot expose a recovered packet")


@dataclass(frozen=True)
class TransportStatistics:
    sent_packets: int
    delivered_packets: int
    transmission_attempts: int
    duplicate_arrivals: int
    loss_events: int
    erasure_events: int
    corrupt_events: int
    deadline_events: int
    queue_drops: int
    wire_bytes: int
    packet_delivery: float
    mean_latency: float | None
    max_latency: float | None
    latency_jitter: float | None


@dataclass(frozen=True)
class ChannelResult:
    sent: tuple[Packet, ...]
    received: tuple[Packet, ...]
    lost_sequences: tuple[int, ...]
    delays: Mapping[int, float] = field(default_factory=dict)
    events: tuple[TransmissionEvent, ...] = ()
    arrival_times: Mapping[int, float] = field(default_factory=dict)
    wire_bytes: int = 0

    def __post_init__(self) -> None:
        integer(self.wire_bytes, "wire_bytes")
        ids = {packet.sequence for packet in self.sent}
        if len(ids) != len(self.sent):
            raise ValueError("channel source IDs must be unique")
        received = {packet.sequence for packet in self.received}
        lost = set(self.lost_sequences)
        if received - ids or lost - ids or received & lost:
            raise ValueError("inconsistent channel result packet IDs")
        for name in ("delays", "arrival_times"):
            values = getattr(self, name)
            if set(values) - received:
                raise ValueError(f"{name} must refer to received packets")
            for value in values.values():
                finite(value, name, 0)
            object.__setattr__(self, name, MappingProxyType(dict(values)))

    @property
    def transport(self) -> TransportStatistics:
        unique = len({packet.sequence for packet in self.received})
        values = tuple(self.delays.values())
        # Scale first so even very large finite delays do not overflow the
        # mean/variance computation and introduce NaN diagnostics.
        if values:
            mean_value = math.fsum(v / len(values) for v in values)
            scale = max(abs(v - mean_value) for v in values)
            jitter: float | None = (
                scale
                * math.sqrt(
                    math.fsum(((v - mean_value) / scale) ** 2 / len(values) for v in values)
                )
                if scale
                else 0.0
            )
            mean: float | None = mean_value
            maximum: float | None = max(values)
        else:
            mean = None
            jitter = None
            maximum = None
        return TransportStatistics(
            len(self.sent),
            unique,
            sum(
                event.status != "queue_full" and event.status != "expired" and not event.duplicate
                for event in self.events
            ),
            len(self.received) - unique,
            sum(event.status == "lost" for event in self.events),
            sum(event.status == "erased" for event in self.events),
            sum(event.status == "corrupt" for event in self.events),
            sum(event.status in {"deadline", "expired"} for event in self.events),
            sum(event.status == "queue_full" for event in self.events),
            self.wire_bytes,
            unique / len(self.sent) if self.sent else 1.0,
            mean,
            maximum,
            jitter,
        )

    def evaluate_semantics(
        self,
        reference: tuple[float, ...],
        observed: tuple[float | None, ...],
        *,
        context: "ScientificContext",
        **kwargs: Any,
    ) -> "SemanticStatistics":
        from .codec import evaluate_semantics

        return evaluate_semantics(reference, observed, context=context, **kwargs)


class EvidenceClass(str, Enum):
    THEOREM = "THEOREM"
    DEFINITION = "DEFINITION"
    ASSUMPTION = "ASSUMPTION"
    MODEL = "MODEL"
    SIMULATION = "SIMULATION"
    NUMERICAL_VERIFICATION = "NUMERICAL_VERIFICATION"
    EMPIRICAL_ESTIMATE = "EMPIRICAL_ESTIMATE"
    EXTERNAL_VALIDATION = "EXTERNAL_VALIDATION"
    CAUSAL_IDENTIFICATION = "CAUSAL_IDENTIFICATION"
    CLINICAL_EVIDENCE = "CLINICAL_EVIDENCE"
    HYPOTHESIS = "HYPOTHESIS"
    INTERPRETATION = "INTERPRETATION"


@dataclass(frozen=True)
class ScientificContext:
    estimand: str
    data_domain: str
    evaluation_regime: str
    conditioning: tuple[str, ...]
    policy_class: str
    reference_state: str
    time_horizon: float
    task_family: tuple[str, ...]
    resource_state: str
    uncertainty_treatment: str
    evidence_class: EvidenceClass
    assumptions: tuple[str, ...] = ()
    causal_status: str = "predictive"

    def __post_init__(self) -> None:
        finite(self.time_horizon, "time_horizon", 0)
        for name in (
            "estimand",
            "data_domain",
            "evaluation_regime",
            "policy_class",
            "reference_state",
            "resource_state",
            "uncertainty_treatment",
            "causal_status",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be declared")
        for name in ("conditioning", "task_family", "assumptions"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(v, str) or not v for v in values
            ):
                raise ValueError(f"{name} must be a tuple of non-empty strings")
        if not self.task_family:
            raise ValueError("task_family must be declared")
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))


@dataclass(frozen=True)
class SemanticStatistics:
    context: ScientificContext
    available_fraction: float
    reconstruction_distortion: float | None
    task_law_discrepancy: float | None = None
    task_value: float | None = None
    reference_task_value: float | None = None
    capability_preservation: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.context, ScientificContext):
            raise ValueError("a scientific context is required")
        probability(self.available_fraction, "available_fraction")
        for name in (
            "reconstruction_distortion",
            "task_law_discrepancy",
            "task_value",
            "reference_task_value",
            "capability_preservation",
        ):
            value = getattr(self, name)
            if value is not None:
                minimum = (
                    0 if name in {"reconstruction_distortion", "task_law_discrepancy"} else None
                )
                finite(value, name, minimum)


@dataclass(frozen=True)
class CommunicationMetrics:
    """Compatibility aggregate; downstream values must be supplied explicitly."""

    packet_delivery: float
    symbol_recovery: float
    bit_recovery: float
    reconstruction_distortion: float | None
    prediction_metric: float | None = None
    task_law_metric: float | None = None
    clinical_utility: float | None = None

    def __post_init__(self) -> None:
        for name in ("packet_delivery", "symbol_recovery", "bit_recovery"):
            probability(getattr(self, name), name)
        for name in (
            "reconstruction_distortion",
            "prediction_metric",
            "task_law_metric",
            "clinical_utility",
        ):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)

    @classmethod
    def from_counts(
        cls,
        sent: int,
        received: int,
        symbols_sent: int,
        symbols_recovered: int,
        bits_sent: int,
        bits_recovered: int,
        **kwargs: Any,
    ) -> "CommunicationMetrics":
        for total, recovered in (
            (sent, received),
            (symbols_sent, symbols_recovered),
            (bits_sent, bits_recovered),
        ):
            integer(total, "sent count")
            integer(recovered, "recovered count")
            if recovered > total:
                raise ValueError("recovered count cannot exceed sent count")
        kwargs.setdefault("reconstruction_distortion", None)
        return cls(
            received / sent if sent else 1.0,
            symbols_recovered / symbols_sent if symbols_sent else 1.0,
            bits_recovered / bits_sent if bits_sent else 1.0,
            **kwargs,
        )


@dataclass
class PacketBuffer:
    packets: dict[int, Packet] = field(default_factory=dict)

    def add(self, packet: Packet) -> bool:
        if packet.sequence in self.packets:
            raise ValueError(f"duplicate packet sequence: {packet.sequence}")
        self.packets[packet.sequence] = packet
        return True

    def ordered(self) -> tuple[Packet, ...]:
        return tuple(self.packets[sequence] for sequence in sorted(self.packets))


@dataclass
class PacketReceiver(PacketBuffer):
    """Explicit wire-duplicate policy: identical retransmissions are idempotent.

    Source sequence IDs must be unique within one receiver stream. Conflicting
    repeated IDs are invalid; create another receiver for another stream.
    """

    duplicate_count: int = 0

    def add(self, packet: Packet) -> bool:
        existing = self.packets.get(packet.sequence)
        if existing is not None:
            if existing.serialize() != packet.serialize():
                raise ValueError(f"conflicting repeated packet sequence: {packet.sequence}")
            self.duplicate_count += 1
            return False
        super().add(packet)
        return True

    def available(self, expected_sequences: tuple[int, ...]) -> tuple[Packet | None, ...]:
        for sequence in expected_sequences:
            integer(sequence, "expected sequence")
        if len(set(expected_sequences)) != len(expected_sequences):
            raise ValueError("expected sequence IDs must be unique")
        return tuple(self.packets.get(sequence) for sequence in expected_sequences)

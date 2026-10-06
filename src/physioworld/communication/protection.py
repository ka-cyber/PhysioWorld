"""Packet-level RLNC, XOR parity and budgeted semantic-priority repetition.

Every coding operation acts on serialized source frames, including semantics,
timestamps, lengths and checksums. Padding bytes are removed using an explicit
length prefix; padding is not interpreted as recovered erased source data.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ._validation import finite, integer
from .models import Packet, PacketReceiver
from .rlnc import Codeword, RLNCDecoder, RLNCEncoder, _combine

_LENGTH = struct.Struct("!I")


def _sources(packets: Sequence[Packet]) -> tuple[Packet, ...]:
    result = tuple(packets)
    if not result or any(not isinstance(packet, Packet) for packet in result):
        raise ValueError("coding needs non-empty source packets")
    if len({packet.sequence for packet in result}) != len(result):
        raise ValueError("source packet sequence IDs must be unique")
    return result


def _generation(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("generation_id must be a non-empty string")
    return value


def _metadata_integer(metadata: Mapping[str, object], key: str, minimum: int = 0) -> int:
    value = metadata.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return integer(value, key, minimum)


def _source_symbol(packet: Packet) -> bytes:
    frame = packet.serialize()
    return _LENGTH.pack(len(frame)) + frame


def _source_packet(symbol: bytes) -> Packet:
    if len(symbol) < _LENGTH.size:
        raise ValueError("truncated source symbol")
    length = _LENGTH.unpack_from(symbol)[0]
    end = _LENGTH.size + length
    if end > len(symbol) or any(symbol[end:]):
        raise ValueError("invalid source length or padding")
    return Packet.deserialize(symbol[_LENGTH.size : end])


@dataclass
class RLNCPacketEncoder:
    packets: tuple[Packet, ...]
    seed: int = 0
    generation_id: str = "default"

    def __post_init__(self) -> None:
        self.packets = _sources(self.packets)
        _generation(self.generation_id)
        symbols = tuple(_source_symbol(packet) for packet in self.packets)
        self.source_width = max(map(len, symbols))
        self._encoder = RLNCEncoder(
            tuple(s.ljust(self.source_width, b"\0") for s in symbols), self.seed
        )

    @property
    def dimension(self) -> int:
        return len(self.packets)

    def _wrap(self, word: Codeword, codec: str, timestamp: float) -> Packet:
        return Packet(
            word.sequence,
            word.serialize(),
            timestamp,
            {
                "codec": codec,
                "generation_id": self.generation_id,
                "dimension": self.dimension,
                "source_width": self.source_width,
            },
        )

    def encode(
        self, count: int | None = None, *, timestamp: float | None = None
    ) -> tuple[Packet, ...]:
        # A block cannot be coded before its most recently acquired source.
        acquired = max(packet.timestamp for packet in self.packets)
        time = acquired if timestamp is None else finite(timestamp, "coding timestamp", 0)
        if time < acquired:
            raise ValueError("coding timestamp precedes source acquisition")
        return tuple(self._wrap(word, "rlnc", time) for word in self._encoder.encode(count))


@dataclass
class RLNCPacketDecoder:
    dimension: int
    source_width: int
    generation_id: str = "default"
    _codec: str = field(init=False, default="rlnc")
    _wire_receiver: PacketReceiver = field(init=False, default_factory=PacketReceiver)

    def __post_init__(self) -> None:
        _generation(self.generation_id)
        self._decoder = RLNCDecoder(self.dimension, self.source_width)

    def add(self, packet: Packet) -> bool:
        info = packet.metadata
        _metadata_integer(info, "dimension", 1)
        _metadata_integer(info, "source_width", 1)
        existing = self._wire_receiver.packets.get(packet.sequence)
        if existing is not None and existing.serialize() != packet.serialize():
            raise ValueError("conflicting repeated wire packet sequence")
        if (
            info.get("codec") != self._codec
            or info.get("generation_id") != self.generation_id
            or info.get("dimension") != self.dimension
            or info.get("source_width") != self.source_width
        ):
            raise ValueError("packet coding metadata does not match decoder")
        word = Codeword.deserialize(packet.payload)
        if self._codec == "xor-parity":
            coeffs = word.coefficients
            if not (
                all(value == 1 for value in coeffs)
                or (sum(value != 0 for value in coeffs) == 1 and 1 in coeffs)
            ):
                raise ValueError("not a systematic or XOR parity word")
        innovative = self._decoder.add(word)
        self._wire_receiver.add(packet)
        return innovative

    @property
    def rank(self) -> int:
        return self._decoder.rank

    @property
    def complete(self) -> bool:
        return self._decoder.complete

    def available(self) -> tuple[Packet | None, ...]:
        return tuple(
            None if symbol is None else _source_packet(symbol)
            for symbol in self._decoder.available()
        )

    def decode(self) -> tuple[Packet, ...]:
        source = tuple(_source_packet(symbol) for symbol in self._decoder.decode())
        if len({packet.sequence for packet in source}) != len(source):
            raise ValueError("decoded source IDs are not unique")
        return source


@dataclass
class ParityEncoder(RLNCPacketEncoder):
    """Systematic source frames plus one XOR parity frame per generation."""

    def encode(
        self,
        count: int | None = None,
        *,
        timestamp: float | None = None,
        packet_budget: int | None = None,
        byte_budget: int | None = None,
    ) -> tuple[Packet, ...]:
        """Emit one complete parity block or reject an insufficient hard budget."""
        if count is not None:
            if packet_budget is not None and packet_budget != count:
                raise ValueError("count and packet_budget must agree")
            packet_budget = count
        if packet_budget is not None:
            integer(packet_budget, "packet_budget")
            if packet_budget < self.dimension + 1:
                raise ValueError("packet budget cannot carry the complete parity block")
        if byte_budget is not None:
            integer(byte_budget, "byte_budget")
        acquired = max(packet.timestamp for packet in self.packets)
        time = acquired if timestamp is None else finite(timestamp, "coding timestamp", 0)
        if time < acquired:
            raise ValueError("coding timestamp precedes source acquisition")
        words = []
        # Fresh transport/codeword IDs on every batch; same generation sources.
        for i in range(self.dimension + 1):
            coefficients = bytes(
                [1 if i == self.dimension or j == i else 0 for j in range(self.dimension)]
            )
            word = Codeword(
                coefficients,
                _combine(coefficients, self._encoder.source_symbols),
                self._encoder._sequence + i,
            )
            words.append(self._wrap(word, "xor-parity", time))
        if byte_budget is not None and sum(packet.wire_length for packet in words) > byte_budget:
            raise ValueError("byte budget cannot carry the complete parity block")
        self._encoder._sequence += len(words)
        return tuple(words)


@dataclass
class ParityDecoder(RLNCPacketDecoder):
    _codec: str = field(init=False, default="xor-parity")


@dataclass
class SemanticPriorityEncoder:
    """Unequal error protection using real, independently transportable copies.

    First send one copy per source in descending declared priority, then allocate
    repeats by priority/(copies+1). Ties retain source order. A constrained budget
    may leave lower-priority sources unavailable; it never fabricates their data.
    Byte budgets include the complete serialized outer frames and repeated inner
    frames. Encoder inputs remain unique; emitted copies have unique wire IDs.
    """

    packets: tuple[Packet, ...]
    priorities: Mapping[int, float]
    generation_id: str = "default"
    max_copies: int | Mapping[int, int] = 8

    def __post_init__(self) -> None:
        self.packets = _sources(self.packets)
        _generation(self.generation_id)
        ids = {packet.sequence for packet in self.packets}
        if set(self.priorities) != ids:
            raise ValueError("priorities must be declared for exactly the source IDs")
        self.priorities = {
            key: finite(value, "semantic priority", 0) for key, value in self.priorities.items()
        }
        if isinstance(self.max_copies, Mapping):
            if set(self.max_copies) != ids:
                raise ValueError("copy limits must match source IDs")
            self._limits = {
                key: integer(value, "max_copies", 1) for key, value in self.max_copies.items()
            }
        else:
            limit = integer(self.max_copies, "max_copies", 1)
            self._limits = dict.fromkeys(ids, limit)
        self._sequence = 0
        self.last_allocation: dict[int, int] = dict.fromkeys(ids, 0)

    def encode(
        self, packet_budget: int, *, byte_budget: int | None = None, timestamp: float | None = None
    ) -> tuple[Packet, ...]:
        integer(packet_budget, "packet_budget")
        if byte_budget is not None:
            integer(byte_budget, "byte_budget")
        acquired = max(packet.timestamp for packet in self.packets)
        time = acquired if timestamp is None else finite(timestamp, "coding timestamp", 0)
        if time < acquired:
            raise ValueError("coding timestamp precedes source acquisition")
        copies = {packet.sequence: 0 for packet in self.packets}
        output: list[Packet] = []
        used_bytes = 0
        while len(output) < packet_budget:
            candidates = []
            for index, source in enumerate(self.packets):
                count = copies[source.sequence]
                if count >= self._limits[source.sequence]:
                    continue
                candidate = Packet(
                    self._sequence,
                    source.serialize(),
                    time,
                    {
                        "codec": "semantic-repetition",
                        "generation_id": self.generation_id,
                        "source_sequence": source.sequence,
                        "copy": count,
                    },
                )
                size = candidate.wire_length
                if byte_budget is not None and used_bytes + size > byte_budget:
                    continue
                priority = self.priorities[source.sequence] / (count + 1)
                candidates.append((count == 0, priority, -index, candidate, size))
            if not candidates:
                break
            _, _, _, candidate, size = max(candidates, key=lambda item: item[:3])
            output.append(candidate)
            used_bytes += size
            copies[candidate.metadata["source_sequence"]] += 1
            self._sequence += 1
        self.last_allocation = copies
        return tuple(output)


@dataclass
class SemanticPriorityDecoder:
    expected_sequences: tuple[int, ...]
    generation_id: str = "default"
    _receiver: PacketReceiver = field(init=False, default_factory=PacketReceiver)
    _wire_receiver: PacketReceiver = field(init=False, default_factory=PacketReceiver)

    def __post_init__(self) -> None:
        self.expected_sequences = tuple(self.expected_sequences)
        for sequence in self.expected_sequences:
            integer(sequence, "source sequence")
        if len(set(self.expected_sequences)) != len(self.expected_sequences):
            raise ValueError("expected source IDs must be unique")
        _generation(self.generation_id)

    def add(self, packet: Packet) -> bool:
        info = packet.metadata
        if (
            info.get("codec") != "semantic-repetition"
            or info.get("generation_id") != self.generation_id
        ):
            raise ValueError("packet protection metadata does not match decoder")
        source = Packet.deserialize(packet.payload)
        if (
            source.sequence != info.get("source_sequence")
            or source.sequence not in self.expected_sequences
        ):
            raise ValueError("unexpected protected source ID")
        _metadata_integer(info, "copy")
        self._wire_receiver.add(packet)
        return self._receiver.add(source)

    @property
    def complete(self) -> bool:
        return len(self._receiver.packets) == len(self.expected_sequences)

    def available(self) -> tuple[Packet | None, ...]:
        return self._receiver.available(self.expected_sequences)

    def decode(self) -> tuple[Packet, ...]:
        if not self.complete:
            raise ValueError("protected source packets are still unavailable")
        return tuple(self._receiver.packets[sequence] for sequence in self.expected_sequences)

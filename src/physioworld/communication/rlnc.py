"""Systematic RLNC over GF(2^8), with incremental reduced-row elimination.

The field uses primitive polynomial x^8+x^4+x^3+x^2+1 (0x11D).
Random redundancy has a rank-failure probability; decoding requires actual
full rank rather than assuming that a packet count implies recoverability.
"""

from __future__ import annotations

import random
import struct
import zlib
from dataclasses import dataclass, field
from typing import Sequence

from ._validation import integer


class GF256:
    polynomial = 0x11D

    @staticmethod
    def _element(value: int) -> int:
        integer(value, "GF256 element")
        if value > 255:
            raise ValueError("GF256 elements must be in [0, 255]")
        return value

    @classmethod
    def add(cls, a: int, b: int) -> int:
        return cls._element(a) ^ cls._element(b)

    @classmethod
    def mul(cls, a: int, b: int) -> int:
        cls._element(a)
        cls._element(b)
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= cls.polynomial
        return result

    @classmethod
    def pow(cls, a: int, exponent: int) -> int:
        cls._element(a)
        integer(exponent, "exponent")
        result = 1
        while exponent:
            if exponent & 1:
                result = cls.mul(result, a)
            a = cls.mul(a, a)
            exponent >>= 1
        return result

    @classmethod
    def inv(cls, a: int) -> int:
        cls._element(a)
        if a == 0:
            raise ZeroDivisionError("zero has no GF256 inverse")
        return cls.pow(a, 254)


_WORD = struct.Struct("!4sIIQ")
_CRC = struct.Struct("!I")


@dataclass(frozen=True)
class Codeword:
    coefficients: bytes
    payload: bytes
    sequence: int

    def __post_init__(self) -> None:
        integer(self.sequence, "codeword sequence")
        if self.sequence >= 2**64:
            raise ValueError("codeword sequence exceeds wire format")
        if not isinstance(self.coefficients, bytes) or not self.coefficients:
            raise ValueError("coefficients must be non-empty bytes")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("codeword payload must be non-empty bytes")

    def serialize(self) -> bytes:
        frame = _WORD.pack(b"PWNC", len(self.coefficients), len(self.payload), self.sequence)
        frame += self.coefficients + self.payload
        return frame + _CRC.pack(zlib.crc32(frame))

    @classmethod
    def deserialize(cls, frame: bytes) -> "Codeword":
        if not isinstance(frame, bytes) or len(frame) < _WORD.size + _CRC.size:
            raise ValueError("truncated codeword")
        magic, dimension, width, sequence = _WORD.unpack_from(frame)
        if magic != b"PWNC" or len(frame) != _WORD.size + dimension + width + _CRC.size:
            raise ValueError("invalid codeword framing")
        if zlib.crc32(frame[:-4]) != _CRC.unpack(frame[-4:])[0]:
            raise ValueError("codeword checksum mismatch")
        return cls(
            frame[_WORD.size : _WORD.size + dimension], frame[_WORD.size + dimension : -4], sequence
        )


def _combine(coefficients: Sequence[int], symbols: Sequence[bytes]) -> bytes:
    if len(coefficients) != len(symbols):
        raise ValueError("coefficient count must match source dimension")
    if not symbols:
        return b""
    width = len(symbols[0])
    if any(len(symbol) != width for symbol in symbols):
        raise ValueError("symbols must have equal width")
    result = bytearray(width)
    for coefficient, symbol in zip(coefficients, symbols):
        GF256._element(coefficient)
        if coefficient:
            for index, value in enumerate(symbol):
                result[index] ^= GF256.mul(coefficient, value)
    return bytes(result)


@dataclass
class RLNCEncoder:
    source_symbols: tuple[bytes, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        self.source_symbols = tuple(self.source_symbols)
        if not self.source_symbols:
            raise ValueError("RLNC requires source symbols")
        if any(not isinstance(symbol, bytes) for symbol in self.source_symbols):
            raise ValueError("source symbols must be bytes")
        width = len(self.source_symbols[0])
        if width == 0 or any(len(s) != width for s in self.source_symbols):
            raise ValueError("source symbols must be non-empty and equal width")
        integer(self.seed, "seed")
        self._rng = random.Random(self.seed)
        self._sequence = 0

    @property
    def dimension(self) -> int:
        return len(self.source_symbols)

    def encode(self, count: int | None = None) -> tuple[Codeword, ...]:
        """Emit ``count`` new words (default dimension), continuing the stream.

        The first dimension words over the encoder lifetime are systematic.
        Later calls emit fresh random repairs, not repeated systematic packets.
        A count of zero is genuinely empty; small repair batches are allowed.
        """
        count = self.dimension if count is None else integer(count, "count")
        words = []
        for _ in range(count):
            if self._sequence < self.dimension:
                coefficients = [0] * self.dimension
                coefficients[self._sequence] = 1
            else:
                coefficients = [self._rng.randrange(256) for _ in range(self.dimension)]
                while not any(coefficients):
                    coefficients = [self._rng.randrange(256) for _ in range(self.dimension)]
            words.append(
                Codeword(
                    bytes(coefficients), _combine(coefficients, self.source_symbols), self._sequence
                )
            )
            self._sequence += 1
        return tuple(words)


@dataclass
class RLNCDecoder:
    dimension: int
    source_width: int
    _basis: dict[int, tuple[list[int], bytearray]] = field(init=False, default_factory=dict)
    _seen: dict[int, Codeword] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        integer(self.dimension, "dimension", 1)
        integer(self.source_width, "source_width", 1)

    def add(self, codeword: Codeword) -> bool:
        if not isinstance(codeword, Codeword) or (
            len(codeword.coefficients) != self.dimension
            or len(codeword.payload) != self.source_width
        ):
            raise ValueError("codeword dimensions do not match decoder")
        previous = self._seen.get(codeword.sequence)
        if previous is not None:
            if previous != codeword:
                raise ValueError("conflicting repeated codeword sequence")
            return False
        row = list(codeword.coefficients)
        payload = bytearray(codeword.payload)
        # The basis is fully reduced, so a late earlier pivot cannot invalidate
        # a previously stored later pivot or revive an eliminated coefficient.
        for pivot in sorted(self._basis):
            existing, existing_payload = self._basis[pivot]
            factor = row[pivot]
            if factor:
                for i in range(self.dimension):
                    row[i] ^= GF256.mul(factor, existing[i])
                for i in range(self.source_width):
                    payload[i] ^= GF256.mul(factor, existing_payload[i])
        pivot_index = next((i for i, value in enumerate(row) if value), None)
        if pivot_index is None:
            if any(payload):
                raise ValueError("inconsistent dependent codeword")
            self._seen[codeword.sequence] = codeword
            return False
        scale = GF256.inv(row[pivot_index])
        row = [GF256.mul(scale, value) for value in row]
        payload = bytearray(GF256.mul(scale, value) for value in payload)
        for existing, existing_payload in self._basis.values():
            factor = existing[pivot_index]
            if factor:
                for i in range(self.dimension):
                    existing[i] ^= GF256.mul(factor, row[i])
                for i in range(self.source_width):
                    existing_payload[i] ^= GF256.mul(factor, payload[i])
        self._basis[pivot_index] = row, payload
        self._seen[codeword.sequence] = codeword
        return True

    @property
    def rank(self) -> int:
        return len(self._basis)

    @property
    def complete(self) -> bool:
        return self.rank == self.dimension

    def available(self) -> tuple[bytes | None, ...]:
        """Only individually determined source symbols; None is an erasure."""
        result: list[bytes | None] = [None] * self.dimension
        for pivot, (row, payload) in self._basis.items():
            if sum(value != 0 for value in row) == 1:
                result[pivot] = bytes(payload)
        return tuple(result)

    def decode(self) -> tuple[bytes, ...]:
        if not self.complete:
            raise ValueError("insufficient independent codewords for decoding")
        return tuple(bytes(self._basis[i][1]) for i in range(self.dimension))

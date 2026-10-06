"""Validated communication configuration, compatible with perturbation helpers.

The standalone mask/delay helpers replay their seed on every call, preserving
their existing API. ErasureChannel owns persistent random streams instead.
Base configurations imported from physioworld.perturbations.network are also
accepted and validated at the channel boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from physioworld.perturbations.network import LatencyJitter as _LatencyJitter
from physioworld.perturbations.network import PacketLoss as _PacketLoss

from ._validation import finite, integer, probability


@dataclass(frozen=True)
class PacketLoss(_PacketLoss):
    def __post_init__(self) -> None:
        probability(self.probability, "loss probability")
        integer(self.burst_length, "burst_length", 1)
        integer(self.seed, "seed")
        super().__post_init__()

    def mask(self, count: int, *, timestamps_s: Sequence[float] | None = None) -> tuple[bool, ...]:
        integer(count, "count")
        return super().mask(count, timestamps_s=timestamps_s)


@dataclass(frozen=True)
class LatencyJitter(_LatencyJitter):
    def __post_init__(self) -> None:
        finite(self.latency, "latency", 0)
        finite(self.jitter, "jitter", 0)
        integer(self.seed, "seed")
        super().__post_init__()

    def delays(
        self, count: int, *, timestamps_s: Sequence[float] | None = None
    ) -> tuple[float, ...]:
        integer(count, "count")
        return super().delays(count, timestamps_s=timestamps_s)

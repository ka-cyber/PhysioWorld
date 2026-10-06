"""Network perturbations kept independent from signal semantics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from .base import Perturbation


@dataclass(frozen=True)
class PacketLoss(Perturbation):
    probability: float
    burst_length: int = 1
    seed: int = 0
    name: str = "packet_loss"
    cause: str = field(default="controlled synthetic communication degradation", kw_only=True)
    expected_state_effect = "packet delivery changes only; physiology unchanged"

    def mask(self, count: int, *, timestamps_s: Sequence[float] | None = None) -> tuple[bool, ...]:
        if not 0 <= self.probability <= 1 or self.burst_length < 1:
            raise ValueError("invalid loss parameters")
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be nonnegative")
        if timestamps_s is None and (self.onset_s != 0 or self.end_s is not None):
            raise ValueError("timed packet loss requires explicit timestamps_s")
        if timestamps_s is not None and len(timestamps_s) != count:
            raise ValueError("one timestamp per packet required")
        active = self.timing_mask(timestamps_s) if timestamps_s is not None else (True,) * count
        rng = random.Random(self.seed)
        lost = [False] * count
        for i in range(count):
            if active[i] and rng.random() < self.probability:
                for j in range(i, min(count, i + self.burst_length)):
                    if active[j]:
                        lost[j] = True
        return tuple(lost)


@dataclass(frozen=True)
class LatencyJitter(Perturbation):
    latency: float
    jitter: float = 0.0
    seed: int = 0
    name: str = "latency_jitter"
    cause: str = field(default="controlled synthetic communication degradation", kw_only=True)
    expected_state_effect = "transport delay changes only; acquisition and physiology unchanged"

    def delays(
        self, count: int, *, timestamps_s: Sequence[float] | None = None
    ) -> tuple[float, ...]:
        if (
            not math.isfinite(self.latency)
            or not math.isfinite(self.jitter)
            or self.latency < 0
            or self.jitter < 0
        ):
            raise ValueError("latency and jitter must be non-negative")
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be nonnegative")
        if timestamps_s is None and (self.onset_s != 0 or self.end_s is not None):
            raise ValueError("timed latency requires explicit timestamps_s")
        if timestamps_s is not None and len(timestamps_s) != count:
            raise ValueError("one timestamp per packet required")
        active = self.timing_mask(timestamps_s) if timestamps_s is not None else (True,) * count
        rng = random.Random(self.seed)
        return tuple(
            max(0.0, self.latency + rng.uniform(-self.jitter, self.jitter)) if active[i] else 0.0
            for i in range(count)
        )


@dataclass(frozen=True)
class Connectivity:
    """Intermittent connectivity schedule over discrete time steps."""

    available: tuple[bool, ...]

    def is_available(self, step: int) -> bool:
        if step < 0:
            raise ValueError("step must be non-negative")
        return (
            self.available[step]
            if step < len(self.available)
            else self.available[-1]
            if self.available
            else False
        )

"""Controlled signal perturbations with deterministic random seeds."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from physioworld.physiology import SignalStream

from .base import Perturbation


def _rng(seed: int | None) -> random.Random:
    return random.Random(seed)


@dataclass(frozen=True)
class AdditiveNoise(Perturbation):
    sigma: float
    seed: int = 0
    name: str = "additive_noise"

    def apply(self, stream: SignalStream) -> SignalStream:
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        rng = _rng(self.seed)
        values = [
            [x + rng.gauss(0.0, self.sigma) if self.active(t) else x for x in row]
            for t, row in zip(stream.timestamps, stream.values)
        ]
        return stream.replace_values(
            values,
            operation=self.name,
            provenance=self.provenance(sigma=self.sigma),
        )


@dataclass(frozen=True)
class BaselineWander(Perturbation):
    amplitude: float
    frequency_hz: float
    name: str = "baseline_wander"
    seed: int = 0

    def apply(self, stream: SignalStream) -> SignalStream:
        values = [
            [
                x + self.amplitude * math.sin(2 * math.pi * self.frequency_hz * (t - self.onset_s))
                if self.active(t)
                else x
                for x in row
            ]
            for t, row in zip(stream.timestamps, stream.values)
        ]
        return stream.replace_values(
            values,
            operation=self.name,
            provenance=self.provenance(amplitude=self.amplitude, frequency_hz=self.frequency_hz),
        )


@dataclass(frozen=True)
class Clipping(Perturbation):
    low: float
    high: float
    name: str = "clipping"
    seed: int = 0

    def __post_init__(self) -> None:
        self.validate_timing()
        if self.low > self.high:
            raise ValueError("low must not exceed high")

    def apply(self, stream: SignalStream) -> SignalStream:
        values = [
            [min(self.high, max(self.low, x)) if self.active(t) else x for x in row]
            for t, row in zip(stream.timestamps, stream.values)
        ]
        return stream.replace_values(
            values, operation=self.name, provenance=self.provenance(low=self.low, high=self.high)
        )


@dataclass(frozen=True)
class Quantization(Perturbation):
    step: float
    name: str = "quantization"
    seed: int = 0

    def apply(self, stream: SignalStream) -> SignalStream:
        if self.step <= 0:
            raise ValueError("quantization step must be positive")
        values = [
            [round(x / self.step) * self.step if self.active(t) else x for x in row]
            for t, row in zip(stream.timestamps, stream.values)
        ]
        return stream.replace_values(
            values, operation=self.name, provenance=self.provenance(step=self.step)
        )


@dataclass(frozen=True)
class SensorDrift(Perturbation):
    slope_per_second: float
    name: str = "sensor_drift"
    seed: int = 0

    def apply(self, stream: SignalStream) -> SignalStream:
        values = [
            [
                x + self.slope_per_second * (t - max(self.onset_s, stream.timestamps[0]))
                if self.active(t)
                else x
                for x in row
            ]
            for t, row in zip(stream.timestamps, stream.values)
        ]
        return stream.replace_values(
            values,
            operation=self.name,
            provenance=self.provenance(slope_per_second=self.slope_per_second),
        )


@dataclass(frozen=True)
class Missingness(Perturbation):
    probability: float = 0.1
    block_length: int = 1
    seed: int = 0
    name: str = "missingness"

    def apply(self, stream: SignalStream) -> SignalStream:
        if not 0 <= self.probability <= 1 or self.block_length < 1:
            raise ValueError("invalid missingness parameters")
        rng = _rng(self.seed)
        missing = list(stream.metadata.missing or (False,) * stream.n_samples)
        for i in range(stream.n_samples):
            if self.active(stream.timestamps[i]) and rng.random() < self.probability:
                for j in range(i, min(stream.n_samples, i + self.block_length)):
                    if self.active(stream.timestamps[j]):
                        missing[j] = True
        values = [
            [float("nan") if missing[i] else x for x in row] for i, row in enumerate(stream.values)
        ]
        return stream.replace_values(
            values,
            operation=self.name,
            missing=missing,
            provenance=self.provenance(
                block_length=self.block_length, probability=self.probability
            ),
        )

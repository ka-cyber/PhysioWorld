"""Named artifact and missingness generators used in controlled sweeps."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from physioworld.physiology import SignalStream

from .base import Perturbation
from .signal import Clipping


@dataclass(frozen=True)
class MotionArtifact(Perturbation):
    amplitude: float = 1.0
    frequency_hz: float = 1.0
    seed: int = 0
    name: str = "motion_artifact"

    def apply(self, stream: SignalStream) -> SignalStream:
        rng = random.Random(self.seed)
        values = [
            [
                x
                + self.amplitude * math.sin(2 * math.pi * self.frequency_hz * (t - self.onset_s))
                + rng.gauss(0, self.amplitude * 0.05)
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
class ElectrodeArtifact(Perturbation):
    probability: float = 0.01
    amplitude: float = 2.0
    duration: int = 1
    seed: int = 0
    name: str = "electrode_artifact"

    def apply(self, stream: SignalStream) -> SignalStream:
        if not 0 <= self.probability <= 1 or self.duration < 1:
            raise ValueError("invalid electrode artifact parameters")
        rng = random.Random(self.seed)
        values = [list(row) for row in stream.values]
        for i in range(stream.n_samples):
            if self.active(stream.timestamps[i]) and rng.random() < self.probability:
                for j in range(i, min(stream.n_samples, i + self.duration)):
                    if self.active(stream.timestamps[j]):
                        values[j] = [x + self.amplitude for x in values[j]]
        return stream.replace_values(
            values,
            operation=self.name,
            provenance=self.provenance(
                probability=self.probability,
                amplitude=self.amplitude,
                duration_samples=self.duration,
            ),
        )


@dataclass(frozen=True)
class EMGContamination(Perturbation):
    sigma: float = 0.1
    seed: int = 0
    name: str = "emg_contamination"

    def apply(self, stream: SignalStream) -> SignalStream:
        rng = random.Random(self.seed)
        values = [
            [x + rng.gauss(0.0, self.sigma) if self.active(t) else x for x in row]
            for t, row in zip(stream.timestamps, stream.values)
        ]
        return stream.replace_values(
            values, operation=self.name, provenance=self.provenance(sigma=self.sigma)
        )


class SensorDropout:
    """Named alias retaining sensor-level semantics for block missingness."""

    def __init__(
        self, probability: float = 0.1, block_length: int = 1, seed: int = 0, **timing_metadata
    ):
        from .signal import Missingness

        self._inner = Missingness(
            probability, block_length, seed, "sensor_dropout", **timing_metadata
        )

    def apply(self, stream: SignalStream) -> SignalStream:
        return self._inner.apply(stream)


class CorrelatedMissingness(SensorDropout):
    """Missingness with a contiguous block, preserving ground-truth metadata."""


@dataclass(frozen=True)
class DistributionShift(Perturbation):
    scale: float = 1.0
    offset: float = 0.0
    seed: int = 0
    name: str = "distribution_shift"

    def apply(self, stream: SignalStream) -> SignalStream:
        values = [
            [self.scale * x + self.offset if self.active(t) else x for x in row]
            for t, row in zip(stream.timestamps, stream.values)
        ]
        return stream.replace_values(
            values,
            operation=self.name,
            provenance=self.provenance(scale=self.scale, offset=self.offset),
        )


@dataclass(frozen=True)
class GainError(DistributionShift):
    gain_error: float = 0.0
    name: str = "gain_error"

    def apply(self, stream: SignalStream) -> SignalStream:
        return DistributionShift(
            self.scale + self.gain_error,
            self.offset,
            self.seed,
            self.name,
            onset_s=self.onset_s,
            end_s=self.end_s,
            cause=self.cause,
            evidence_basis=self.evidence_basis,
            safety_status=self.safety_status,
        ).apply(stream)


@dataclass(frozen=True)
class OffsetError(DistributionShift):
    offset_error: float = 0.0
    name: str = "offset_error"

    def apply(self, stream: SignalStream) -> SignalStream:
        return DistributionShift(
            self.scale,
            self.offset + self.offset_error,
            self.seed,
            self.name,
            onset_s=self.onset_s,
            end_s=self.end_s,
            cause=self.cause,
            evidence_basis=self.evidence_basis,
            safety_status=self.safety_status,
        ).apply(stream)


@dataclass(frozen=True)
class CalibrationFailure(Perturbation):
    gain: float = 1.0
    offset: float = 0.0
    seed: int = 0
    name: str = "calibration_failure"

    def apply(self, stream: SignalStream) -> SignalStream:
        if self.gain == 0:
            raise ValueError("calibration failure gain cannot be zero")
        values = [
            [self.gain * value + self.offset if self.active(t) else value for value in row]
            for t, row in zip(stream.timestamps, stream.values)
        ]
        return stream.replace_values(
            values,
            operation=self.name,
            provenance=self.provenance(gain=self.gain, offset=self.offset),
        )


@dataclass(frozen=True)
class ADCSaturation(Clipping):
    name: str = "adc_saturation"

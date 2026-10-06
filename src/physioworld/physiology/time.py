"""Explicit time domains and transformations for physiological records."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class ClockDomain(str, Enum):
    SOURCE_RECORD = "source_record"
    SIMULATION = "simulation"
    WALL_CLOCK = "wall_clock"
    DEVICE = "device"
    ACQUISITION = "acquisition"
    TRANSPORT = "transport"
    COMPUTATION = "computation"
    DECISION = "decision"
    INTERVENTION = "intervention"
    OUTCOME = "outcome"
    EVENT = "event"


@dataclass(frozen=True)
class TimeTransform:
    """Affine clock map `t_target = offset + scale * t_source`."""

    source: ClockDomain
    target: ClockDomain
    offset_s: float = 0.0
    scale: float = 1.0
    method: str = "declared_affine"
    uncertainty_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", ClockDomain(self.source))
        object.__setattr__(self, "target", ClockDomain(self.target))
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("clock scale must be positive")
        if not math.isfinite(self.offset_s):
            raise ValueError("clock offset must be finite seconds")
        if not math.isfinite(self.uncertainty_s) or self.uncertainty_s < 0:
            raise ValueError("clock uncertainty must be finite and non-negative")

    def apply(self, timestamp: float) -> float:
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite seconds")
        result = self.offset_s + self.scale * timestamp
        if not math.isfinite(result):
            raise ValueError("clock transformation overflowed finite seconds")
        return result

    def apply_many(self, timestamps: Sequence[float]) -> tuple[float, ...]:
        return tuple(self.apply(float(timestamp)) for timestamp in timestamps)


@dataclass(frozen=True)
class TimeMetadata:
    """Timestamps are seconds in the declared domain, never sample indices.

    ``origin`` retains an original epoch/ISO-8601 reference; wall-clock samples
    use Unix seconds. Offset/drift specify a *declared* map, not an inferred one.
    """

    clock_domain: ClockDomain = ClockDomain.SOURCE_RECORD
    origin: str | None = None
    wall_clock_start_s: float | None = None
    clock_offset_s: float = 0.0
    sampling_drift_ppm: float = 0.0
    time_unit: str = "s"
    uncertainty_s: float = 0.0
    clock_id: str | None = None
    synchronization_method: str | None = None
    time_basis: str = "relative"  # relative to origin, or Unix epoch seconds

    def __post_init__(self) -> None:
        object.__setattr__(self, "clock_domain", ClockDomain(self.clock_domain))
        if self.time_unit != "s":
            raise ValueError("canonical time_unit must be 's'; convert explicitly at ingestion")
        if self.time_basis not in {"relative", "unix"}:
            raise ValueError("time_basis must be relative or unix")
        if self.clock_domain == ClockDomain.WALL_CLOCK and self.time_basis != "unix":
            raise ValueError("wall-clock time requires an explicit unix time_basis")
        numbers = (self.clock_offset_s, self.sampling_drift_ppm, self.uncertainty_s)
        if any(not math.isfinite(v) for v in numbers) or self.uncertainty_s < 0:
            raise ValueError("clock parameters must be finite, uncertainty non-negative")
        if 1 + self.sampling_drift_ppm * 1e-6 <= 0:
            raise ValueError("clock drift implies a non-positive scale")
        if self.wall_clock_start_s is not None and not math.isfinite(self.wall_clock_start_s):
            raise ValueError("wall clock start must be finite")

    def transform_to(self, target: ClockDomain) -> TimeTransform:
        scale = 1.0 + self.sampling_drift_ppm * 1e-6
        return TimeTransform(
            self.clock_domain,
            target,
            self.clock_offset_s,
            scale,
            "metadata_affine",
            self.uncertainty_s,
        )


@dataclass(frozen=True)
class CalibrationMetadata:
    gain: float = 1.0
    offset: float = 0.0
    adc_bits: int | None = None
    saturation_low: float | None = None
    saturation_high: float | None = None
    method: str = "unspecified"
    input_unit: str | None = None
    output_unit: str | None = None
    version: str = "1.0"
    input_data: tuple[str, ...] = ()
    checksum: str | None = None
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    operating_conditions: Mapping[str, Any] = field(default_factory=dict)
    status: str = "ASSUMED"

    def __post_init__(self) -> None:
        if not math.isfinite(self.gain) or self.gain == 0:
            raise ValueError("calibration gain cannot be zero")
        if not math.isfinite(self.offset):
            raise ValueError("calibration offset must be finite")
        if any(
            v is not None and not math.isfinite(v)
            for v in (self.saturation_low, self.saturation_high)
        ):
            raise ValueError("calibration bounds must be finite")
        if self.adc_bits is not None and self.adc_bits < 1:
            raise ValueError("adc_bits must be positive")
        if (
            self.saturation_low is not None
            and self.saturation_high is not None
            and self.saturation_low > self.saturation_high
        ):
            raise ValueError("calibration saturation bounds are reversed")
        if self.status not in {"ASSUMED", "MEASURED", "FITTED", "BOUNDED"}:
            raise ValueError("invalid calibration status")

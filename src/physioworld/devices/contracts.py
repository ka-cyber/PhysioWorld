"""Vendor-neutral acquisition/HIL contracts and explicitly bounded stimulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .calibration import Calibration


class DeviceState(str, Enum):
    SAFE = "safe"
    CONFIGURED = "configured"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class SensorDescriptor:
    id: str
    unit: str
    minimum: float
    maximum: float
    description: str = ""

    def __post_init__(self) -> None:
        if (
            not self.id
            or not math.isfinite(self.minimum)
            or not math.isfinite(self.maximum)
            or self.minimum >= self.maximum
        ):
            raise ValueError("sensor needs finite increasing range")


@dataclass(frozen=True)
class ActuatorDescriptor:
    id: str
    unit: str
    minimum: float
    maximum: float
    safe_value: float = 0.0
    digital: bool = False

    def __post_init__(self) -> None:
        if (
            not all(map(math.isfinite, (self.minimum, self.maximum, self.safe_value)))
            or not self.minimum <= self.safe_value <= self.maximum
        ):
            raise ValueError("actuator safe value must lie in finite declared bounds")


@dataclass(frozen=True)
class DeviceCapabilities:
    identity: str
    firmware: str
    sensors: tuple[SensorDescriptor, ...]
    actuators: tuple[ActuatorDescriptor, ...] = ()
    min_sample_rate: float = 1.0
    max_sample_rate: float = 10000.0
    trigger: bool = False
    stimulation: bool = False
    transport: str = "simulator"

    def __post_init__(self) -> None:
        ids = [s.id for s in self.sensors]
        outputs = [a.id for a in self.actuators]
        if (
            not self.identity
            or not self.firmware
            or len(set(ids)) != len(ids)
            or len(set(outputs)) != len(outputs)
            or not math.isfinite(self.max_sample_rate)
            or not 0 < self.min_sample_rate <= self.max_sample_rate
        ):
            raise ValueError("invalid discovery identity/channels/rates")


@dataclass(frozen=True)
class DeviceConfiguration:
    sample_rate: float
    channels: tuple[str, ...]


@dataclass(frozen=True)
class SampleBlock:
    timestamps: tuple[float, ...]
    rows: tuple[tuple[float, ...], ...]
    sequence_numbers: tuple[int, ...]
    sensors: tuple[SensorDescriptor, ...]
    dropped_samples: int = 0

    def __post_init__(self) -> None:
        if (
            len(self.timestamps) != len(self.rows)
            or len(self.rows) != len(self.sequence_numbers)
            or any(len(row) != len(self.sensors) for row in self.rows)
            or self.dropped_samples < 0
            or not all(map(math.isfinite, self.timestamps))
            or any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:]))
            or any(b <= a for a, b in zip(self.sequence_numbers, self.sequence_numbers[1:]))
            or not all(math.isfinite(v) for row in self.rows for v in row)
        ):
            raise ValueError("malformed acquisition block")


@dataclass(frozen=True)
class TriggerReceipt:
    requested_time: float
    observed_time: float
    clock_id: str

    @property
    def latency_s(self) -> float:
        return self.observed_time - self.requested_time


@dataclass(frozen=True)
class Stimulus:
    actuator: str
    amplitude: float
    unit: str
    duration_s: float
    frequency_hz: float
    pulse_width_s: float


@dataclass(frozen=True)
class StimulationBounds:
    unit: str
    max_abs_amplitude: float
    max_duration_s: float
    max_frequency_hz: float
    max_duty_cycle: float
    max_abs_charge_c: float | None = None

    def validate(self, stimulus: Stimulus) -> None:
        from physioworld.instrumentation.units import unit

        values = (
            self.max_abs_amplitude,
            self.max_duration_s,
            self.max_frequency_hz,
            self.max_duty_cycle,
            stimulus.amplitude,
            stimulus.duration_s,
            stimulus.frequency_hz,
            stimulus.pulse_width_s,
        )
        if not all(map(math.isfinite, values)) or min(values[:4]) <= 0 or self.max_duty_cycle > 1:
            raise ValueError("stimulation bounds and parameters must be finite and valid")
        if unit(stimulus.unit) != unit(self.unit):
            raise ValueError("stimulation units must match bounds explicitly")
        if (
            abs(stimulus.amplitude) > self.max_abs_amplitude
            or not 0 < stimulus.duration_s <= self.max_duration_s
            or not 0 < stimulus.frequency_hz <= self.max_frequency_hz
            or not 0 < stimulus.pulse_width_s <= stimulus.duration_s
            or stimulus.frequency_hz * stimulus.pulse_width_s > self.max_duty_cycle
        ):
            raise ValueError(
                "stimulation exceeds declared amplitude/duration/frequency/duty bounds"
            )
        if self.max_abs_charge_c is not None:
            if not math.isfinite(self.max_abs_charge_c) or self.max_abs_charge_c <= 0:
                raise ValueError("charge bound must be positive and finite")
            current_a = stimulus.amplitude * unit(stimulus.unit).factor_to(unit("A"))
            # Conservative pulse-count bound including a pulse starting at t=0.
            pulse_count = math.ceil(stimulus.duration_s * stimulus.frequency_hz)
            if abs(current_a) * stimulus.pulse_width_s * pulse_count > self.max_abs_charge_c:
                raise ValueError("stimulation exceeds declared absolute charge bound")


@runtime_checkable
class Device(Protocol):
    state: DeviceState

    def discover(self) -> DeviceCapabilities: ...
    def configure(self, configuration: DeviceConfiguration) -> None: ...
    def calibrate(self, channel: str, calibration: Calibration) -> None: ...
    def start(self, start_time: float = 0.0) -> None: ...
    def read(self, sample_count: int) -> SampleBlock: ...
    def stop(self) -> None: ...
    def emergency_stop(self) -> None: ...
    def reset(self) -> None: ...
    def trigger(self, timestamp: float) -> TriggerReceipt: ...
    def analog_write(self, actuator: str, value: float) -> None: ...
    def digital_write(self, actuator: str, value: bool) -> None: ...
    def stimulate(self, stimulus: Stimulus, bounds: StimulationBounds) -> None: ...

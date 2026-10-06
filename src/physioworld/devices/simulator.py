"""Deterministic, sampled virtual instrument with latched safe failure."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Callable

from .calibration import Calibration
from .contracts import (
    ActuatorDescriptor,
    DeviceCapabilities,
    DeviceConfiguration,
    DeviceState,
    SampleBlock,
    SensorDescriptor,
    StimulationBounds,
    Stimulus,
    TriggerReceipt,
)


class SimulatedDevice:
    def __init__(
        self,
        identity: str = "simulated-lab",
        *,
        sensors: tuple[SensorDescriptor, ...] | None = None,
        actuators: tuple[ActuatorDescriptor, ...] | None = None,
        waveform: Callable[[float, str], float] | None = None,
        seed: int = 0,
        noise_std: float = 0.0,
        jitter_s: float = 0.0,
        drop_every: int = 0,
        fail_after: int | None = None,
        clock_drift_ppm: float = 0.0,
        trigger_latency_s: float = 0.001,
    ) -> None:
        if (
            noise_std < 0
            or jitter_s < 0
            or drop_every < 0
            or trigger_latency_s < 0
            or not all(
                map(math.isfinite, (noise_std, jitter_s, clock_drift_ppm, trigger_latency_s))
            )
            or 1 + clock_drift_ppm * 1e-6 <= 0
        ):
            raise ValueError("invalid simulator fault/timing parameters")
        self.capabilities = DeviceCapabilities(
            identity,
            "simulator-1",
            sensors or (SensorDescriptor("signal", "mV", -10, 10),),
            actuators
            if actuators is not None
            else (
                ActuatorDescriptor("analog", "V", -5, 5),
                ActuatorDescriptor("digital", "1", 0, 1, digital=True),
            ),
            trigger=True,
            stimulation=True,
        )
        self.waveform = waveform or (lambda t, channel: math.sin(2 * math.pi * t))
        self.seed, self.noise_std, self.jitter_s = seed, noise_std, jitter_s
        self.drop_every, self.fail_after = drop_every, fail_after
        self.clock_drift_ppm, self.trigger_latency_s = clock_drift_ppm, trigger_latency_s
        self.calibrations: dict[str, Calibration] = {}
        self.configuration: DeviceConfiguration | None = None
        self.state = DeviceState.SAFE
        self.outputs = {a.id: a.safe_value for a in self.capabilities.actuators}
        self.triggers: list[TriggerReceipt] = []
        self.stimuli: list[Stimulus] = []
        self.stimulus_schedule: list[tuple[float, Stimulus]] = []
        self._sequence, self._start = 0, 0.0
        self._rng = random.Random(seed)

    def discover(self) -> DeviceCapabilities:
        return self.capabilities

    def configure(self, configuration: DeviceConfiguration) -> None:
        if self.state not in {DeviceState.SAFE, DeviceState.STOPPED, DeviceState.CONFIGURED}:
            raise RuntimeError("device must be safe/stopped before configuration")
        ids = {s.id for s in self.capabilities.sensors}
        if (
            not math.isfinite(configuration.sample_rate)
            or not self.capabilities.min_sample_rate
            <= configuration.sample_rate
            <= self.capabilities.max_sample_rate
            or not configuration.channels
            or len(set(configuration.channels)) != len(configuration.channels)
            or not set(configuration.channels) <= ids
            or self.jitter_s * 2 >= (1 + self.clock_drift_ppm * 1e-6) / configuration.sample_rate
        ):
            raise ValueError("unsupported channels/rate or jitter may reverse timestamps")
        self.configuration = configuration
        self.state = DeviceState.CONFIGURED

    def calibrate(self, channel: str, calibration: Calibration) -> None:
        if self.state == DeviceState.RUNNING:
            raise RuntimeError("calibration cannot change while acquiring")
        sensor = next((s for s in self.capabilities.sensors if s.id == channel), None)
        if sensor is None or sensor.unit != calibration.input_unit:
            raise ValueError("calibration input channel/unit mismatch")
        self.calibrations[channel] = calibration

    def start(self, start_time: float = 0.0) -> None:
        if self.state != DeviceState.CONFIGURED or not math.isfinite(start_time):
            raise RuntimeError("safe configuration required before startup")
        self._safe_outputs()
        self._sequence, self._start = 0, start_time
        self._rng = random.Random(self.seed)
        self.triggers.clear()
        self.stimuli.clear()
        self.stimulus_schedule.clear()
        self.state = DeviceState.RUNNING

    def read(self, sample_count: int) -> SampleBlock:
        if self.state != DeviceState.RUNNING or self.configuration is None:
            raise RuntimeError("acquisition is not running")
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        source = {s.id: s for s in self.capabilities.sensors}
        sensors = tuple(source[c] for c in self.configuration.channels)
        output_sensors = tuple(
            replace(
                s,
                unit=self.calibrations[s.id].output_unit,
                minimum=self.calibrations[s.id].output_bounds[0],
                maximum=self.calibrations[s.id].output_bounds[1],
            )
            if s.id in self.calibrations
            else s
            for s in sensors
        )
        times, rows, sequences, dropped = [], [], [], 0
        try:
            for _ in range(sample_count):
                sequence = self._sequence
                if self.fail_after is not None and sequence >= self.fail_after:
                    raise IOError("declared simulator acquisition failure")
                t = self._start + sequence / self.configuration.sample_rate
                clock_t = self._start + (t - self._start) * (1 + self.clock_drift_ppm * 1e-6)
                clock_t += self._rng.uniform(-self.jitter_s, self.jitter_s)
                values = []
                for sensor in sensors:
                    raw = self.waveform(t, sensor.id) + self._rng.gauss(0, self.noise_std)
                    if not math.isfinite(raw):
                        raise ValueError("waveform returned a nonfinite sample")
                    raw = min(sensor.maximum, max(sensor.minimum, raw))
                    values.append(
                        self.calibrations[sensor.id].apply(raw)
                        if sensor.id in self.calibrations
                        else raw
                    )
                self._sequence += 1
                if self.drop_every and (sequence + 1) % self.drop_every == 0:
                    dropped += 1
                    continue
                times.append(clock_t)
                rows.append(tuple(values))
                sequences.append(sequence)
            return SampleBlock(tuple(times), tuple(rows), tuple(sequences), output_sensors, dropped)
        except Exception:
            self.emergency_stop()
            raise

    def _safe_outputs(self) -> None:
        self.outputs = {a.id: a.safe_value for a in self.capabilities.actuators}
        self.stimulus_schedule.clear()

    def stop(self) -> None:
        self._safe_outputs()
        if self.state != DeviceState.FAILED:
            self.state = DeviceState.STOPPED

    def emergency_stop(self) -> None:
        self._safe_outputs()
        self.state = DeviceState.FAILED

    def reset(self) -> None:
        self._safe_outputs()
        self.configuration = None
        self.state = DeviceState.SAFE

    def trigger(self, timestamp: float) -> TriggerReceipt:
        if self.state != DeviceState.RUNNING or not math.isfinite(timestamp):
            raise RuntimeError("trigger requires running device and finite timestamp")
        receipt = TriggerReceipt(
            timestamp, timestamp + self.trigger_latency_s, self.capabilities.identity
        )
        self.triggers.append(receipt)
        return receipt

    def _actuator(self, name: str) -> ActuatorDescriptor:
        if self.state != DeviceState.RUNNING:
            raise RuntimeError("output requires safe startup")
        actuator = next((a for a in self.capabilities.actuators if a.id == name), None)
        if actuator is None:
            raise KeyError(name)
        return actuator

    def analog_write(self, actuator: str, value: float) -> None:
        try:
            descriptor = self._actuator(actuator)
            if (
                descriptor.digital
                or not math.isfinite(value)
                or not descriptor.minimum <= value <= descriptor.maximum
            ):
                raise ValueError("analog output exceeds declared bounds or targets digital IO")
            self.outputs[actuator] = value
        except Exception:
            self.emergency_stop()
            raise

    def digital_write(self, actuator: str, value: bool) -> None:
        try:
            descriptor = self._actuator(actuator)
            if not descriptor.digital or not isinstance(value, bool):
                raise ValueError("digital output requires bool and digital actuator")
            self.outputs[actuator] = float(value)
        except Exception:
            self.emergency_stop()
            raise

    def stimulate(self, stimulus: Stimulus, bounds: StimulationBounds) -> None:
        try:
            bounds.validate(stimulus)
            descriptor = self._actuator(stimulus.actuator)
            if descriptor.unit != stimulus.unit or descriptor.digital:
                raise ValueError("stimulus actuator/unit mismatch")
            configuration = self.configuration
            if configuration is None:
                raise RuntimeError("stimulation requires an active device configuration")
            start = self._start + self._sequence / configuration.sample_rate
            if any(
                s.actuator == stimulus.actuator and start < onset + s.duration_s
                for onset, s in self.stimulus_schedule
            ):
                raise ValueError("overlapping stimuli on an actuator are not supported")
            self.analog_write(stimulus.actuator, stimulus.amplitude)
            self.stimuli.append(stimulus)
            self.stimulus_schedule.append((start, stimulus))
            # Virtual waveform is scheduled, never a persistent energized output.
            self.outputs[stimulus.actuator] = descriptor.safe_value
        except Exception:
            self.emergency_stop()
            raise

    def output_at(self, actuator: str, timestamp: float) -> float:
        """Evaluate a generated virtual rectangular pulse train at a device time.

        Researchers can wire this to the input waveform callable for closed-loop
        simulated IO. Overlapping stimuli on one actuator are rejected.
        """
        descriptor = next(a for a in self.capabilities.actuators if a.id == actuator)
        if not math.isfinite(timestamp):
            raise ValueError("output timestamp must be finite")
        if self.state != DeviceState.RUNNING:
            return descriptor.safe_value
        active = [
            (start, s)
            for start, s in self.stimulus_schedule
            if s.actuator == actuator and start <= timestamp < start + s.duration_s
        ]
        if not active:
            return self.outputs[actuator]
        start, stimulus = active[-1]
        cycles = (timestamp - start) * stimulus.frequency_hz
        nearest = round(cycles)
        if math.isclose(cycles, nearest, abs_tol=1e-12, rel_tol=0):
            cycles = nearest
        phase = (cycles % 1) / stimulus.frequency_hz
        return stimulus.amplitude if phase < stimulus.pulse_width_s else descriptor.safe_value

    def analog_read(self, channel: str) -> float:
        if self.configuration is None or channel not in self.configuration.channels:
            raise ValueError("channel is not configured")
        block = self.read(1)
        if not block.rows:
            raise IOError("single sample was dropped")
        return block.rows[0][self.configuration.channels.index(channel)]

    def digital_read(self, actuator: str) -> bool:
        descriptor = self._actuator(actuator)
        if not descriptor.digital:
            raise ValueError("digital read requires digital IO")
        return bool(self.outputs[actuator])

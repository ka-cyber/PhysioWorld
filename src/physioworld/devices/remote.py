"""Executable JSON-line device adapter for serial/network instrument firmware.

Wire contract: {operation, arguments} -> {ok: true, result: ...} or
{ok: false, error: ...}. Both ends must implement this declared contract.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

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
from .transports import LineTransport


class JsonLineDevice:
    def __init__(self, transport: LineTransport) -> None:
        self.transport, self.state = transport, DeviceState.SAFE
        self.configuration: DeviceConfiguration | None = None
        self.calibrations: dict[str, Calibration] = {}
        self.capabilities: DeviceCapabilities | None = None
        self._rpc("safe_startup")

    def _rpc(self, operation: str, **arguments: Any) -> Any:
        try:
            response = json.loads(
                self.transport.exchange(
                    json.dumps(
                        {"operation": operation, "arguments": arguments}, allow_nan=False
                    ).encode()
                )
            )
            if response.get("ok") is not True:
                raise IOError(response.get("error", "device rejected command"))
            return response.get("result")
        except Exception:
            self.state = DeviceState.FAILED
            if operation != "emergency_stop":
                try:
                    self.transport.exchange(b'{"operation":"emergency_stop","arguments":{}}')
                except Exception:
                    pass
            raise

    def discover(self) -> DeviceCapabilities:
        if self.capabilities is None:
            info = self._rpc("discover")
            info["sensors"] = tuple(SensorDescriptor(**s) for s in info["sensors"])
            info["actuators"] = tuple(ActuatorDescriptor(**a) for a in info.get("actuators", ()))
            self.capabilities = DeviceCapabilities(**info)
        return self.capabilities

    def configure(self, configuration: DeviceConfiguration) -> None:
        caps = self.discover()
        if self.state not in {DeviceState.SAFE, DeviceState.STOPPED, DeviceState.CONFIGURED}:
            raise RuntimeError("device is not safe for configuration")
        if (
            not configuration.channels
            or len(set(configuration.channels)) != len(configuration.channels)
            or not set(configuration.channels) <= {s.id for s in caps.sensors}
            or not caps.min_sample_rate <= configuration.sample_rate <= caps.max_sample_rate
        ):
            raise ValueError("unsupported device configuration")
        self._rpc("configure", **asdict(configuration))
        self.configuration, self.state = configuration, DeviceState.CONFIGURED

    def calibrate(self, channel: str, calibration: Calibration) -> None:
        if self.state == DeviceState.RUNNING:
            raise RuntimeError("cannot recalibrate during acquisition")
        self._rpc("calibrate", channel=channel, calibration=asdict(calibration))
        self.calibrations[channel] = calibration

    def start(self, start_time: float = 0.0) -> None:
        if self.state != DeviceState.CONFIGURED or not math.isfinite(start_time):
            raise RuntimeError("configuration and finite start time required")
        self._rpc("start", start_time=start_time)
        self.state = DeviceState.RUNNING

    def read(self, sample_count: int) -> SampleBlock:
        if self.state != DeviceState.RUNNING or sample_count < 1:
            raise RuntimeError("running acquisition and positive sample count required")
        try:
            block = self._rpc("read", sample_count=sample_count)
            return SampleBlock(
                tuple(block["timestamps"]),
                tuple(tuple(r) for r in block["rows"]),
                tuple(block["sequence_numbers"]),
                tuple(SensorDescriptor(**s) for s in block["sensors"]),
                block.get("dropped_samples", 0),
            )
        except Exception:
            self.emergency_stop()
            raise

    def stop(self) -> None:
        self._rpc("stop")
        self.state = DeviceState.STOPPED

    def emergency_stop(self) -> None:
        self.state = DeviceState.FAILED
        self._rpc("emergency_stop")

    def reset(self) -> None:
        self._rpc("reset")
        self.state, self.configuration = DeviceState.SAFE, None

    def trigger(self, timestamp: float) -> TriggerReceipt:
        if self.state != DeviceState.RUNNING or not self.discover().trigger:
            raise RuntimeError("trigger unavailable or device is not running")
        return TriggerReceipt(**self._rpc("trigger", timestamp=timestamp))

    def _output(self, actuator: str, value: float, digital: bool) -> ActuatorDescriptor:
        descriptor = next(a for a in self.discover().actuators if a.id == actuator)
        if (
            self.state != DeviceState.RUNNING
            or descriptor.digital != digital
            or not math.isfinite(value)
            or not descriptor.minimum <= value <= descriptor.maximum
        ):
            raise ValueError("output violates device state/type/bounds")
        return descriptor

    def analog_write(self, actuator: str, value: float) -> None:
        try:
            self._output(actuator, value, False)
            self._rpc("analog_write", actuator=actuator, value=value)
        except Exception:
            self.emergency_stop()
            raise

    def digital_write(self, actuator: str, value: bool) -> None:
        try:
            if not isinstance(value, bool):
                raise ValueError("digital value must be bool")
            self._output(actuator, float(value), True)
            self._rpc("digital_write", actuator=actuator, value=value)
        except Exception:
            self.emergency_stop()
            raise

    def stimulate(self, stimulus: Stimulus, bounds: StimulationBounds) -> None:
        try:
            bounds.validate(stimulus)
            descriptor = self._output(stimulus.actuator, stimulus.amplitude, False)
            if not self.discover().stimulation or descriptor.unit != stimulus.unit:
                raise ValueError("stimulation capability/unit mismatch")
            self._rpc("stimulate", stimulus=asdict(stimulus), bounds=asdict(bounds))
        except Exception:
            self.emergency_stop()
            raise

    def close(self) -> None:
        try:
            self.emergency_stop() if self.state == DeviceState.FAILED else self.stop()
        finally:
            self.transport.close()

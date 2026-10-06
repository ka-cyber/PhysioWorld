"""Synchronized block acquisition using declared affine device clock maps."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Mapping

from physioworld.core import Provenance
from physioworld.instrumentation.recording import Recording
from physioworld.physiology.stream import ChannelMetadata, SignalStream, StreamMetadata
from physioworld.physiology.time import ClockDomain, TimeMetadata

from .contracts import Device, DeviceConfiguration


@dataclass(frozen=True)
class ClockSynchronization:
    offset_s: float
    scale: float = 1.0
    uncertainty_s: float = 0.0
    method: str = "declared_affine"

    def __post_init__(self) -> None:
        if (
            not all(map(math.isfinite, (self.offset_s, self.scale, self.uncertainty_s)))
            or self.scale <= 0
            or self.uncertainty_s < 0
        ):
            raise ValueError("invalid clock synchronization")

    def to_session(self, device_time: float) -> float:
        return self.offset_s + self.scale * device_time


@dataclass(frozen=True)
class AcquisitionConfiguration:
    duration_s: float
    block_samples: int = 256
    session_start_s: float = 0.0
    realtime: bool = False
    session_clock: ClockDomain = ClockDomain.ACQUISITION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_clock", ClockDomain(self.session_clock))
        if (
            not math.isfinite(self.duration_s)
            or self.duration_s <= 0
            or self.block_samples < 1
            or not math.isfinite(self.session_start_s)
        ):
            raise ValueError("positive duration/block size and finite start required")


@dataclass(frozen=True)
class AcquisitionResult:
    recording: Recording
    sequence_numbers: dict[str, tuple[int, ...]]
    dropped_samples: dict[str, int]
    clock_maps: dict[str, ClockSynchronization]


class AcquisitionRunner:
    def __init__(
        self,
        devices: Mapping[str, Device],
        configurations: Mapping[str, DeviceConfiguration],
        clock_maps: Mapping[str, ClockSynchronization],
    ) -> None:
        if not devices or set(devices) != set(configurations) or set(devices) != set(clock_maps):
            raise ValueError("every device requires configuration and an explicit clock map")
        if len({id(d) for d in devices.values()}) != len(devices):
            raise ValueError("a device instance cannot be acquired twice")
        self.devices, self.configurations, self.clock_maps = (
            dict(devices),
            dict(configurations),
            dict(clock_maps),
        )

    def run(
        self,
        configuration: AcquisitionConfiguration,
        recording: Recording | None = None,
        on_block: Callable[[str, SignalStream], None] | None = None,
    ) -> AcquisitionResult:
        recording = recording or Recording("acquisition", "unassigned", "session")
        sequences: dict[str, list[int]] = {k: [] for k in self.devices}
        loss = {k: 0 for k in self.devices}
        remaining = {
            k: math.floor(
                configuration.duration_s * c.sample_rate / self.clock_maps[k].scale + 1e-9
            )
            for k, c in self.configurations.items()
        }
        if min(remaining.values()) < 1:
            raise ValueError("duration must include at least one sample for every device")
        requested = {k: 0 for k in self.devices}
        wall_start = time.monotonic()
        try:
            for name, device in self.devices.items():
                device.configure(self.configurations[name])
            for name, device in self.devices.items():
                clock = self.clock_maps[name]
                device.start((configuration.session_start_s - clock.offset_s) / clock.scale)
            while any(remaining.values()):
                for name, device in self.devices.items():
                    if not remaining[name]:
                        continue
                    count = min(configuration.block_samples, remaining[name])
                    block = device.read(count)
                    if len(block.rows) + block.dropped_samples != count:
                        raise IOError("device must account explicitly for every requested sample")
                    if tuple(s.id for s in block.sensors) != self.configurations[name].channels:
                        raise IOError("device returned different channels/order than configured")
                    if not set(block.sequence_numbers) <= set(
                        range(requested[name], requested[name] + count)
                    ):
                        raise IOError(
                            "device sequence numbers do not account for the requested sample range"
                        )
                    remaining[name] -= count
                    requested[name] += count
                    clock = self.clock_maps[name]
                    if (
                        sequences[name]
                        and block.sequence_numbers
                        and block.sequence_numbers[0] <= sequences[name][-1]
                    ):
                        raise IOError("device sequence restarted during acquisition")
                    sequences[name].extend(block.sequence_numbers)
                    loss[name] += block.dropped_samples
                    if not block.rows:
                        continue
                    capabilities = device.discover()
                    metadata = StreamMetadata(
                        modality="acquired",
                        device_id=capabilities.identity,
                        data_kind="simulation"
                        if capabilities.transport == "simulator"
                        else "sensor_telemetry",
                        clock=TimeMetadata(
                            configuration.session_clock,
                            origin=recording.session_id,
                            clock_id=recording.session_id,
                            uncertainty_s=clock.uncertainty_s,
                            synchronization_method=clock.method,
                        ),
                        channels=tuple(
                            ChannelMetadata(s.id, s.unit, description=s.description)
                            for s in block.sensors
                        ),
                        provenance=Provenance(
                            "physioworld.devices",
                            "acquisition",
                            notes={
                                "device": asdict(capabilities),
                                "clock_map": asdict(clock),
                                "sequence_numbers": block.sequence_numbers,
                                "dropped_samples": block.dropped_samples,
                                "calibrations": {
                                    k: asdict(v)
                                    for k, v in getattr(device, "calibrations", {}).items()
                                },
                            },
                        ),
                    )
                    stream = SignalStream(
                        tuple(clock.to_session(t) for t in block.timestamps),
                        block.rows,
                        self.configurations[name].sample_rate / clock.scale,
                        metadata,
                    )
                    recording.append(name, stream)
                    if on_block is not None:
                        on_block(name, stream)
                if configuration.realtime:
                    progress = min(
                        requested[k] * self.clock_maps[k].scale / self.configurations[k].sample_rate
                        for k in self.devices
                    )
                    delay = progress - (time.monotonic() - wall_start)
                    if delay > 0:
                        time.sleep(delay)
            for device in self.devices.values():
                device.stop()
        except BaseException:
            for device in self.devices.values():
                try:
                    device.emergency_stop()
                except Exception:
                    pass  # Preserve the originating failure; stop was attempted for every device.
            raise
        return AcquisitionResult(
            recording, {k: tuple(v) for k, v in sequences.items()}, loss, self.clock_maps
        )

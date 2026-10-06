"""Static device qualification definitions and executable software contract checks.

Results are software/simulator observations. They are not physical validation,
electrical safety certification, or proof of clock synchronization to a standard.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from .calibration import Calibration
from .contracts import Device, DeviceConfiguration, DeviceState


@dataclass(frozen=True)
class ReferenceDefinition:
    id: str
    context: str
    input_artifact: str
    reference_output: str
    tolerance: str
    comparison: str
    pass_rule: str
    software_version: str = "physioworld device contract 1"
    scientific_status: str = "software contract verification; physical validation not performed"


QUALIFICATION_DEFINITIONS = tuple(
    ReferenceDefinition(*row)
    for row in (
        (
            "DEV-ID",
            "discovery",
            "declared identity/firmware/channels",
            "matching metadata",
            "exact",
            "caller declaration comparison",
            "all fields match",
        ),
        (
            "DEV-TIME",
            "timestamps and jitter",
            "sequence-indexed uniform clock",
            "t_i = i/fs",
            "plan.max_jitter_s",
            "analytic time grid independent of device",
            "monotonic and residual within tolerance",
        ),
        (
            "DEV-TRIGGER",
            "trigger latency",
            "trigger at declared time",
            "nonnegative bounded response",
            "plan.max_trigger_latency_s",
            "observed minus requested",
            "0 <= latency <= tolerance",
        ),
        (
            "DEV-LOSS",
            "loss/dropout",
            "N requested samples",
            "N sequence-accounted samples",
            "plan.max_loss_fraction",
            "count and sequence gaps",
            "reported loss agrees and fraction within tolerance",
        ),
        (
            "DEV-SAT",
            "saturation",
            "declared sensor range",
            "endpoint fraction",
            "plan.max_saturation_fraction",
            "compare acquired samples to descriptor endpoints",
            "fraction within tolerance",
        ),
        (
            "DEV-SAFE",
            "safe failure",
            "emergency stop",
            "latched failure; read rejected",
            "exact",
            "state check and attempted acquisition",
            "FAILED state and acquisition rejected",
        ),
        (
            "DEV-RESTART",
            "restart",
            "reset/configure/start/read",
            "new acquisition starts at sequence 0",
            "exact",
            "sequence comparison",
            "valid block and reset sequence",
        ),
        (
            "DEV-CAL",
            "calibration",
            "independent reference points",
            "declared converted values",
            "caller tolerance",
            "polynomial evaluation against reference",
            "all errors <= tolerance",
        ),
    )
)


@dataclass(frozen=True)
class QualificationPlan:
    configuration: DeviceConfiguration
    sample_count: int = 128
    expected_identity: str | None = None
    expected_firmware: str | None = None
    expected_units: tuple[str, ...] | None = None
    max_jitter_s: float = 1e-6
    max_trigger_latency_s: float = 0.01
    max_loss_fraction: float = 0.0
    max_saturation_fraction: float = 0.0
    deterministic_replay: bool = False

    def __post_init__(self) -> None:
        if (
            self.sample_count < 2
            or not all(
                map(
                    math.isfinite,
                    (
                        self.max_jitter_s,
                        self.max_trigger_latency_s,
                        self.max_loss_fraction,
                        self.max_saturation_fraction,
                    ),
                )
            )
            or min(self.max_jitter_s, self.max_trigger_latency_s) < 0
            or not 0 <= self.max_loss_fraction <= 1
            or not 0 <= self.max_saturation_fraction <= 1
        ):
            raise ValueError("invalid qualification tolerances")


@dataclass(frozen=True)
class QualificationCheck:
    id: str
    passed: bool | None
    measured: float | str | None
    detail: str


@dataclass(frozen=True)
class QualificationReport:
    identity: str
    checks: tuple[QualificationCheck, ...]
    evidence_scope: str = "software interface observations; no physical validation"

    @property
    def passed(self) -> bool:
        return all(c.passed is not False for c in self.checks)


def qualify_device(factory: Callable[[], Device], plan: QualificationPlan) -> QualificationReport:
    device = factory()
    checks = []
    caps = device.discover()
    identity_ok = (plan.expected_identity is None or caps.identity == plan.expected_identity) and (
        plan.expected_firmware is None or caps.firmware == plan.expected_firmware
    )
    checks.append(QualificationCheck("DEV-ID", identity_ok, caps.identity, caps.firmware))
    try:
        device.configure(plan.configuration)
        device.start(0.0)
        start = time.monotonic()
        block = device.read(plan.sample_count)
        checks.append(
            QualificationCheck(
                "DEV-READ-LATENCY",
                None,
                time.monotonic() - start,
                "Host elapsed seconds, descriptive, no physical latency reference.",
            )
        )
        timestamps_ok = bool(block.timestamps) and all(
            b > a for a, b in zip(block.timestamps, block.timestamps[1:])
        )
        jitter = max(
            (
                abs(t - i / plan.configuration.sample_rate)
                for t, i in zip(block.timestamps, block.sequence_numbers)
            ),
            default=math.inf,
        )
        checks.append(
            QualificationCheck(
                "DEV-TIME",
                timestamps_ok and jitter <= plan.max_jitter_s,
                jitter,
                "Residual to declared clock grid, seconds.",
            )
        )
        fraction = (plan.sample_count - len(block.rows)) / plan.sample_count
        expected_sequences = set(range(plan.sample_count))
        accounting = (
            set(block.sequence_numbers) <= expected_sequences
            and len(block.rows) + block.dropped_samples == plan.sample_count
        )
        checks.append(
            QualificationCheck(
                "DEV-LOSS",
                accounting and fraction <= plan.max_loss_fraction,
                fraction,
                "Requested samples and sequence numbers, no filling.",
            )
        )
        units_ok = (
            plan.expected_units is None
            or tuple(s.unit for s in block.sensors) == plan.expected_units
        )
        checks.append(
            QualificationCheck(
                "DEV-UNITS",
                units_ok,
                ",".join(s.unit for s in block.sensors),
                "Declared channel units.",
            )
        )
        saturation: float = sum(
            v <= s.minimum or v >= s.maximum
            for row in block.rows
            for v, s in zip(row, block.sensors)
        )
        saturation /= max(1, len(block.rows) * len(block.sensors))
        checks.append(
            QualificationCheck(
                "DEV-SAT",
                saturation <= plan.max_saturation_fraction,
                saturation,
                "Endpoint saturation fraction; depends on supplied test waveform.",
            )
        )
        if caps.trigger:
            receipt = device.trigger(0.0)
            checks.append(
                QualificationCheck(
                    "DEV-TRIGGER",
                    0 <= receipt.latency_s <= plan.max_trigger_latency_s,
                    receipt.latency_s,
                    "Device-reported clock-domain latency.",
                )
            )
        else:
            checks.append(
                QualificationCheck(
                    "DEV-TRIGGER", None, None, "Trigger not supported by discovered device."
                )
            )
        device.emergency_stop()
        rejected = False
        try:
            device.read(1)
        except Exception:
            rejected = True
        checks.append(
            QualificationCheck(
                "DEV-SAFE",
                device.state == DeviceState.FAILED and rejected,
                device.state.value,
                "Emergency-stop latch and rejected acquisition.",
            )
        )
        device.reset()
        device.configure(plan.configuration)
        device.start(0.0)
        restart = device.read(plan.sample_count)
        checks.append(
            QualificationCheck(
                "DEV-RESTART",
                bool(restart.sequence_numbers) and restart.sequence_numbers[0] == 0,
                None,
                "Sequence restarts after explicit reset.",
            )
        )
        if plan.deterministic_replay:
            checks.append(
                QualificationCheck(
                    "DEV-REPRO",
                    block == restart,
                    None,
                    "Exact simulator replay requested by caller.",
                )
            )
        for name in ("DEV-TEMPERATURE", "DEV-POWER", "DEV-CAL", "DEV-EXTERNAL-SYNC"):
            checks.append(
                QualificationCheck(
                    name, None, None, "No independent measurement supplied; not qualified."
                )
            )
        return QualificationReport(caps.identity, tuple(checks))
    finally:
        device.stop()


def qualify_calibration(
    calibration: Calibration, reference: tuple[tuple[float, float], ...], tolerance: float
) -> QualificationCheck:
    if not reference or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("independent reference points and finite tolerance required")
    error = max(abs(calibration.apply(x) - y) for x, y in reference)
    return QualificationCheck(
        "DEV-CAL", error <= tolerance, error, "Absolute reference output error."
    )

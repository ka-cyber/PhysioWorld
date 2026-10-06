"""Data-quality, metadata, and leakage-oriented validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .record import PhysiologicalRecord
from .stream import SignalStream


class ValidationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ValidationIssue:
    status: ValidationStatus
    code: str
    location: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    status: ValidationStatus
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.status == ValidationStatus.ERROR)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "issues": [
                {
                    "status": i.status.value,
                    "code": i.code,
                    "location": i.location,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }


def _status(issues: list[ValidationIssue]) -> ValidationStatus:
    if any(issue.status == ValidationStatus.ERROR for issue in issues):
        return ValidationStatus.ERROR
    if any(issue.status == ValidationStatus.WARNING for issue in issues):
        return ValidationStatus.WARNING
    return ValidationStatus.PASS


def validate_stream(
    stream: SignalStream, *, physiologic_ranges: Mapping[str, tuple[float, float]] | None = None
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    if not stream.metadata.channels:
        issues.append(
            ValidationIssue(
                ValidationStatus.ERROR,
                "missing_channel_metadata",
                "metadata.channels",
                "channel metadata is required",
            )
        )
    names = [channel.name for channel in stream.metadata.channels]
    if stream.metadata.access_class not in {"OPEN", "RESTRICTED", "CREDENTIALED"}:
        issues.append(
            ValidationIssue(
                ValidationStatus.ERROR,
                "invalid_access_class",
                "metadata.access_class",
                "access class must be OPEN, RESTRICTED, or CREDENTIALED",
            )
        )
    if len(names) != len(set(names)):
        issues.append(
            ValidationIssue(
                ValidationStatus.ERROR,
                "duplicate_channel",
                "metadata.channels",
                "channel names must be unique",
            )
        )
    if any(not channel.unit for channel in stream.metadata.channels):
        issues.append(
            ValidationIssue(
                ValidationStatus.ERROR,
                "missing_unit",
                "metadata.channels",
                "every channel requires a unit",
            )
        )
    if stream.sample_rate is None and stream.metadata.sampling_semantics in {"regular", "declared"}:
        issues.append(
            ValidationIssue(
                ValidationStatus.WARNING,
                "missing_sampling_rate",
                "sample_rate",
                "sampling rate is unknown",
            )
        )
    elif stream.sample_rate is not None and (
        not math.isfinite(stream.sample_rate) or stream.sample_rate <= 0
    ):
        issues.append(
            ValidationIssue(
                ValidationStatus.ERROR,
                "invalid_sampling_rate",
                "sample_rate",
                "sampling rate must be positive",
            )
        )
    if any(not math.isfinite(t) for t in stream.timestamps):
        issues.append(
            ValidationIssue(
                ValidationStatus.ERROR,
                "invalid_time",
                "timestamps",
                "timestamps must be finite seconds",
            )
        )
    if any(b <= a for a, b in zip(stream.timestamps, stream.timestamps[1:])):
        issues.append(
            ValidationIssue(
                ValidationStatus.ERROR,
                "non_strict_timestamps",
                "timestamps",
                "timestamps must be strictly increasing for analysis",
            )
        )
    if stream.metadata.quality and any(
        not 0 <= quality <= 1 for quality in stream.metadata.quality
    ):
        issues.append(
            ValidationIssue(
                ValidationStatus.ERROR,
                "invalid_quality",
                "metadata.quality",
                "quality values must be in [0,1]",
            )
        )
    for index, row in enumerate(stream.values):
        if any(
            not math.isfinite(value)
            and not (
                (stream.metadata.missing[index] if stream.metadata.missing else False)
                or (
                    stream.metadata.missing_channels[index][c]
                    if stream.metadata.missing_channels
                    else False
                )
            )
            for c, value in enumerate(row)
        ):
            issues.append(
                ValidationIssue(
                    ValidationStatus.ERROR,
                    "invalid_value",
                    f"values[{index}]",
                    "non-finite values require explicit missingness",
                )
            )
            break
    if any(b.onset < a.onset for a, b in zip(stream.annotations, stream.annotations[1:])):
        issues.append(
            ValidationIssue(
                ValidationStatus.ERROR,
                "annotation_order",
                "annotations",
                "annotations must be ordered by onset",
            )
        )
    if physiologic_ranges and stream.metadata.modality in physiologic_ranges:
        low, high = physiologic_ranges[stream.metadata.modality]
        if any(
            value < low or value > high
            for row in stream.values
            for value in row
            if math.isfinite(value)
        ):
            issues.append(
                ValidationIssue(
                    ValidationStatus.WARNING,
                    "out_of_range",
                    "values",
                    f"values exceed declared {stream.metadata.modality} range",
                )
            )
    if stream.metadata.calibration is None:
        issues.append(
            ValidationIssue(
                ValidationStatus.WARNING,
                "missing_calibration",
                "metadata.calibration",
                "calibration metadata is not declared",
            )
        )
    return ValidationReport(_status(issues), tuple(issues))


def validate_records(records: Iterable[PhysiologicalRecord]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    seen_records: set[str] = set()
    for record in records:
        if record.record_id in seen_records:
            issues.append(
                ValidationIssue(
                    ValidationStatus.ERROR,
                    "duplicate_record",
                    record.record_id,
                    "record_id appears more than once",
                )
            )
        seen_records.add(record.record_id)
        for modality, stream in record.streams.items():
            stream_report = validate_stream(stream)
            issues.extend(
                ValidationIssue(
                    issue.status,
                    issue.code,
                    f"{record.record_id}/{modality}/{issue.location}",
                    issue.message,
                )
                for issue in stream_report.issues
            )
        if any(b.onset < a.onset for a, b in zip(record.events, record.events[1:])):
            issues.append(
                ValidationIssue(
                    ValidationStatus.ERROR,
                    "event_order",
                    record.record_id,
                    "record events must be ordered by onset",
                )
            )
    return ValidationReport(_status(issues), tuple(issues))

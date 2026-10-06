"""Validated signal containers with explicit units and provenance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

from physioworld.core import Provenance

from .time import CalibrationMetadata, TimeMetadata, TimeTransform


def _as_rows(values: Sequence[Any]) -> tuple[tuple[float, ...], ...]:
    if len(values) == 0:
        return ()
    first = values[0]
    if hasattr(first, "__iter__") and not isinstance(first, (str, bytes)):
        rows = tuple(tuple(float(v) for v in row) for row in values)
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("multichannel values must be rectangular")
        return rows
    return tuple((float(v),) for v in values)


@dataclass(frozen=True)
class ChannelMetadata:
    name: str
    unit: str
    anatomical_site: str | None = None
    description: str | None = None
    unit_system: str = "UCUM"
    calibration: CalibrationMetadata | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("channels require explicit names and units")


@dataclass(frozen=True)
class Annotation:
    onset: float
    duration: float = 0.0
    label: str = ""
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.onset) or not math.isfinite(self.duration) or self.duration < 0:
            raise ValueError("annotation timing must be finite and duration non-negative")


@dataclass(frozen=True)
class StreamMetadata:
    record_id: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    modality: str = "unknown"
    physical_quantity: str | None = None
    source_dataset: str | None = None
    dataset_version: str | None = None
    access_class: str = "OPEN"
    clock: TimeMetadata = field(default_factory=TimeMetadata)
    calibration: CalibrationMetadata | None = None
    channels: tuple[ChannelMetadata, ...] = ()
    quality: tuple[float, ...] = ()
    missing: tuple[bool, ...] = ()
    task_labels: Mapping[str, Any] = field(default_factory=dict)
    clinical_labels: Mapping[str, Any] = field(default_factory=dict)
    provenance: Provenance | None = None
    device_id: str | None = None
    site_id: str | None = None
    sampling_semantics: str = "declared"
    missing_channels: tuple[tuple[bool, ...], ...] = ()
    uncertainty: tuple[tuple[float, ...], ...] = ()  # standard deviations in channel units
    lineage: tuple[Provenance, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    data_kind: str = "sensor_telemetry"


@dataclass(frozen=True)
class SignalStream:
    """A timestamped scalar or multichannel stream.

    Samples are stored as `(time, channel)` rows. Non-finite sample values require
    declared row or channel missingness; no imputation is performed. Annotation
    onsets/durations use the stream's declared clock/basis and seconds.
    """

    timestamps: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    sample_rate: float | None
    metadata: StreamMetadata
    annotations: tuple[Annotation, ...] = ()

    def __post_init__(self) -> None:
        if any(isinstance(t, bool) or not math.isfinite(t) for t in self.timestamps):
            raise ValueError("timestamps must be finite seconds")
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have equal length")
        if any(b < a for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("timestamps must be monotone non-decreasing")
        if self.sample_rate is not None and (
            isinstance(self.sample_rate, bool)
            or not math.isfinite(self.sample_rate)
            or self.sample_rate <= 0
        ):
            raise ValueError("sample_rate must be finite and positive")
        width = len(self.values[0]) if self.values else len(self.metadata.channels)
        if any(len(row) != width for row in self.values):
            raise ValueError("values must be rectangular")
        if width != len(self.metadata.channels):
            raise ValueError("channel metadata must match values width")
        if len({c.name for c in self.metadata.channels}) != width:
            raise ValueError("channel names must be unique")
        if self.metadata.quality and len(self.metadata.quality) != len(self.values):
            raise ValueError("quality must have one value per sample")
        if self.metadata.missing and len(self.metadata.missing) != len(self.values):
            raise ValueError("missingness must have one value per sample")
        if any(not math.isfinite(q) or not 0 <= q <= 1 for q in self.metadata.quality):
            raise ValueError("quality must be finite in [0,1]")
        for name in ("missing_channels", "uncertainty"):
            data = getattr(self.metadata, name)
            if data and (len(data) != len(self.values) or any(len(row) != width for row in data)):
                raise ValueError(f"{name} must match sample/channel shape")
        if any(not isinstance(flag, bool) for flag in self.metadata.missing) or any(
            not isinstance(flag, bool) for row in self.metadata.missing_channels for flag in row
        ):
            raise ValueError("missingness flags must be boolean")
        for index, row in enumerate(self.values):
            if any(
                not math.isfinite(value)
                and not (
                    (self.metadata.missing[index] if self.metadata.missing else False)
                    or (
                        self.metadata.missing_channels[index][channel]
                        if self.metadata.missing_channels
                        else False
                    )
                )
                for channel, value in enumerate(row)
            ):
                raise ValueError("non-finite values require declared missingness")
        if any(not math.isfinite(u) or u < 0 for row in self.metadata.uncertainty for u in row):
            raise ValueError("uncertainty must be finite non-negative standard deviations")
        if self.metadata.sampling_semantics not in {"declared", "regular", "irregular", "event"}:
            raise ValueError("invalid sampling semantics")
        if self.metadata.sampling_semantics == "regular" and len(self.timestamps) > 1:
            if self.sample_rate is None or any(
                not math.isclose(
                    b - a,
                    1 / self.sample_rate,
                    rel_tol=1e-6,
                    abs_tol=max(1e-12, math.ulp(a) + math.ulp(b)),
                )
                for a, b in zip(self.timestamps, self.timestamps[1:])
            ):
                raise ValueError("regular sampling timestamps disagree with sample_rate")

    @classmethod
    def from_samples(
        cls,
        values: Sequence[Any],
        sample_rate: float | None,
        unit: str,
        modality: str,
        *,
        timestamps: Sequence[float] | None = None,
        record_id: str | None = None,
        subject_id: str | None = None,
        session_id: str | None = None,
        channel_names: Sequence[str] | None = None,
        provenance: Provenance | None = None,
        physical_quantity: str | None = None,
        source_dataset: str | None = None,
        dataset_version: str | None = None,
        access_class: str = "OPEN",
        clock: TimeMetadata | None = None,
        calibration: CalibrationMetadata | None = None,
        device_id: str | None = None,
        site_id: str | None = None,
        missing: Sequence[bool] = (),
        missing_channels: Sequence[Sequence[bool]] = (),
        quality: Sequence[float] = (),
        uncertainty: Sequence[Sequence[float]] = (),
        attributes: Mapping[str, Any] | None = None,
        data_kind: str = "sensor_telemetry",
        sampling_semantics: str | None = None,
    ) -> "SignalStream":
        rows = _as_rows(values)
        if sample_rate is not None and (
            isinstance(sample_rate, bool) or not math.isfinite(sample_rate) or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be finite and positive")
        if timestamps is None and sample_rate is None:
            raise ValueError("explicit timestamps or a finite positive sample_rate are required")
        if timestamps is not None and any(isinstance(t, bool) for t in timestamps):
            raise ValueError("timestamps must be numeric seconds, not booleans")
        if timestamps is not None:
            times = tuple(float(t) for t in timestamps)
        else:
            assert sample_rate is not None
            times = tuple(i / sample_rate for i in range(len(rows)))
        if len(times) != len(rows):
            raise ValueError("timestamps and values must have equal length")
        width = len(rows[0]) if rows else len(channel_names) if channel_names is not None else 1
        names = (
            tuple(channel_names)
            if channel_names is not None
            else tuple(f"{modality.lower()}_{i}" for i in range(width))
        )
        if len(names) != width:
            raise ValueError("channel_names has wrong width")
        metadata = StreamMetadata(
            record_id=record_id,
            subject_id=subject_id,
            session_id=session_id,
            modality=modality,
            physical_quantity=physical_quantity,
            source_dataset=source_dataset,
            dataset_version=dataset_version,
            access_class=access_class,
            clock=clock or TimeMetadata(),
            calibration=calibration,
            channels=tuple(ChannelMetadata(name=n, unit=unit) for n in names),
            provenance=provenance,
            device_id=device_id,
            site_id=site_id,
            sampling_semantics=sampling_semantics
            or ("regular" if timestamps is None else "irregular"),
            missing=tuple(missing.tolist() if hasattr(missing, "tolist") else missing),
            missing_channels=tuple(
                tuple(r)
                for r in (
                    missing_channels.tolist()
                    if hasattr(missing_channels, "tolist")
                    else missing_channels
                )
            ),
            quality=tuple(float(q) for q in quality),
            uncertainty=tuple(tuple(float(u) for u in r) for r in uncertainty),
            attributes=attributes or {},
            data_kind=data_kind,
        )
        return cls(times, rows, float(sample_rate) if sample_rate is not None else None, metadata)

    @property
    def n_samples(self) -> int:
        return len(self.values)

    @property
    def n_channels(self) -> int:
        return len(self.values[0]) if self.values else len(self.metadata.channels)

    @property
    def duration(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def channel(self, index_or_name: int | str) -> tuple[float, ...]:
        if isinstance(index_or_name, str):
            names = [c.name for c in self.metadata.channels]
            if index_or_name not in names:
                raise KeyError(index_or_name)
            index = names.index(index_or_name)
        else:
            index = index_or_name
        return tuple(row[index] for row in self.values)

    def replace_values(
        self,
        values: Sequence[Any],
        *,
        operation: str,
        provenance: Provenance | None = None,
        missing: Sequence[bool] | None = None,
    ) -> "SignalStream":
        rows = _as_rows(values)
        if len(rows) != self.n_samples:
            raise ValueError("replacement values have wrong sample count")
        metadata = self.metadata
        if missing is not None:
            metadata = replace(metadata, missing=tuple(missing))
        if provenance is not None:
            metadata = replace(metadata, provenance=provenance)
        elif not operation:
            raise ValueError("replacement requires a named operation")
        else:
            metadata = replace(metadata, provenance=Provenance("physioworld", operation))
        metadata = replace(metadata, lineage=self._lineage())
        return replace(self, values=rows, metadata=metadata)

    def _lineage(self) -> tuple[Provenance, ...]:
        return self.metadata.lineage + (
            (self.metadata.provenance,) if self.metadata.provenance else ()
        )

    def is_missing(self, sample: int, channel: int) -> bool:
        return (
            (bool(self.metadata.missing[sample]) if self.metadata.missing else False)
            or (
                bool(self.metadata.missing_channels[sample][channel])
                if self.metadata.missing_channels
                else False
            )
            or not math.isfinite(self.values[sample][channel])
        )

    def to_dict(self) -> dict[str, Any]:
        from physioworld.io.canonical import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SignalStream":
        from physioworld.io.canonical import from_dict

        result = from_dict(payload)
        if not isinstance(result, cls):
            raise ValueError("expected a canonical signal")
        return result

    def semantic_equal(self, other: object) -> bool:
        return isinstance(other, SignalStream) and self.to_dict() == other.to_dict()

    def convert_units(self, units: str | Sequence[str]) -> "SignalStream":
        from physioworld.io.units import convert_stream_units

        return convert_stream_units(self, units)

    def with_annotations(self, annotations: Iterable[Annotation]) -> "SignalStream":
        return replace(
            self,
            annotations=tuple(annotations),
            metadata=replace(
                self.metadata,
                lineage=self._lineage(),
                provenance=Provenance("physioworld", "annotate"),
            ),
        )

    def transform_time(
        self,
        transform: TimeTransform,
        *,
        operation: str = "time_transform",
        target_time_basis: str | None = None,
        target_origin: str | None = None,
    ) -> "SignalStream":
        """Apply an explicit clock transformation and record its semantics."""
        if transform.source != self.metadata.clock.clock_domain:
            raise ValueError("time transform source does not match stream clock domain")
        metadata = replace(
            self.metadata,
            clock=replace(
                self.metadata.clock,
                clock_domain=transform.target,
                clock_offset_s=0.0,
                sampling_drift_ppm=0.0,
                time_basis=target_time_basis or self.metadata.clock.time_basis,
                origin=target_origin if target_origin is not None else self.metadata.clock.origin,
                uncertainty_s=math.hypot(
                    transform.scale * self.metadata.clock.uncertainty_s, transform.uncertainty_s
                ),
            ),
            provenance=Provenance(
                "physioworld",
                operation,
                notes={
                    "time_transform": transform.__dict__,
                    "target_time_basis": target_time_basis or self.metadata.clock.time_basis,
                    "target_origin": target_origin
                    if target_origin is not None
                    else self.metadata.clock.origin,
                },
            ),
            lineage=self._lineage(),
        )
        return replace(
            self,
            timestamps=transform.apply_many(self.timestamps),
            metadata=metadata,
            sample_rate=self.sample_rate / transform.scale if self.sample_rate else None,
            annotations=tuple(
                replace(a, onset=transform.apply(a.onset), duration=a.duration * transform.scale)
                for a in self.annotations
            ),
        )

    def resample(
        self, new_sample_rate: float, *, method: str = "linear", phase_delay_s: float = 0.0
    ) -> "SignalStream":
        """Explicitly resample and retain original/new rate and delay metadata."""
        if (
            isinstance(new_sample_rate, bool)
            or not math.isfinite(new_sample_rate)
            or new_sample_rate <= 0
        ):
            raise ValueError("new_sample_rate must be finite and positive")
        if not math.isfinite(phase_delay_s):
            raise ValueError("phase delay must be finite seconds")
        if method not in {"linear", "nearest"}:
            raise ValueError("method must be linear or nearest")
        if any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("resampling requires strictly increasing timestamps")
        start, end = (self.timestamps[0], self.timestamps[-1]) if self.timestamps else (0.0, 0.0)
        count = (
            int(math.floor((end - start) * new_sample_rate + 1e-9)) + 1 if self.timestamps else 0
        )
        new_times = tuple(start + i / new_sample_rate + phase_delay_s for i in range(count))
        values, missing_channels, quality, uncertainty = [], [], [], []
        right = 0  # monotone two-pointer scan: O(input samples + output samples * channels)
        for sample_index in range(count):
            source_time = start + sample_index / new_sample_rate
            while right < self.n_samples - 1 and self.timestamps[right] < source_time:
                right += 1
            left = max(0, right - 1)
            if math.isclose(source_time, self.timestamps[right], abs_tol=1e-12, rel_tol=0):
                left = right
            if method == "nearest":
                index = (
                    left
                    if source_time - self.timestamps[left] <= self.timestamps[right] - source_time
                    else right
                )
                left = right = index
            fraction = (
                (
                    (source_time - self.timestamps[left])
                    / (self.timestamps[right] - self.timestamps[left])
                )
                if right != left
                else 0.0
            )
            flags = tuple(
                self.is_missing(left, c) or self.is_missing(right, c)
                for c in range(self.n_channels)
            )
            missing_channels.append(flags)
            values.append(
                tuple(
                    float("nan")
                    if flags[c]
                    else self.values[left][c]
                    + fraction * (self.values[right][c] - self.values[left][c])
                    for c in range(self.n_channels)
                )
            )
            if self.metadata.quality:
                quality.append(min(self.metadata.quality[left], self.metadata.quality[right]))
            if self.metadata.uncertainty:
                # Conservative bound; no independence between adjacent samples is assumed.
                uncertainty.append(
                    tuple(
                        (1 - fraction) * a + fraction * b
                        for a, b in zip(
                            self.metadata.uncertainty[left], self.metadata.uncertainty[right]
                        )
                    )
                )
        metadata = replace(
            self.metadata,
            provenance=Provenance(
                "physioworld",
                "resample",
                notes={
                    "original_rate": self.sample_rate,
                    "new_rate": new_sample_rate,
                    "method": method,
                    "phase_delay_s": phase_delay_s,
                    "missing_policy": "propagate_support",
                    "quality_policy": "minimum_support",
                    "uncertainty_policy": "correlated_upper_bound",
                    "anti_alias_filter": None,
                },
            ),
            lineage=self._lineage(),
            sampling_semantics="regular",
            missing=tuple(all(r) for r in missing_channels),
            missing_channels=tuple(missing_channels),
            quality=tuple(quality),
            uncertainty=tuple(uncertainty),
        )
        # Row missing flags mean the entire sample is missing; partial gaps use channel flags.
        return replace(
            self,
            timestamps=new_times,
            values=tuple(values),
            sample_rate=new_sample_rate,
            metadata=metadata,
            annotations=tuple(replace(a, onset=a.onset + phase_delay_s) for a in self.annotations),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "record_id": self.metadata.record_id,
            "subject_id": self.metadata.subject_id,
            "session_id": self.metadata.session_id,
            "modality": self.metadata.modality,
            "clock_domain": self.metadata.clock.clock_domain.value,
            "source_dataset": self.metadata.source_dataset,
            "dataset_version": self.metadata.dataset_version,
            "access_class": self.metadata.access_class,
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "sample_rate": self.sample_rate,
            "unit": self.metadata.channels[0].unit if self.metadata.channels else None,
            "duration": self.duration,
            "missing_samples": sum(
                any(self.is_missing(i, c) for c in range(self.n_channels))
                for i in range(self.n_samples)
            ),
            "annotation_count": len(self.annotations),
        }

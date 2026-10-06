"""Researcher-native recording organization and renderer-independent data views."""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from physioworld.core import Provenance
from physioworld.physiology.stream import Annotation, ChannelMetadata, SignalStream, StreamMetadata
from physioworld.physiology.time import CalibrationMetadata, ClockDomain, TimeMetadata

from .analysis import EventSeries, Measurements, same_clock, statistics_extract


@dataclass(frozen=True)
class VideoTrack:
    """PTS metadata only: recording_time = offset_s + rate_scale * video_pts.

    The media URI is descriptive and is not opened or decoded during replay.
    """

    uri: str
    pts_seconds: tuple[float, ...]
    offset_s: float
    rate_scale: float = 1.0
    synchronization_method: str = "declared_affine"
    uncertainty_s: float = 0.0

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(v)
                for v in (*self.pts_seconds, self.offset_s, self.rate_scale, self.uncertainty_s)
            )
            or self.rate_scale <= 0
            or self.uncertainty_s < 0
            or any(b <= a for a, b in zip(self.pts_seconds, self.pts_seconds[1:]))
        ):
            raise ValueError("invalid video timestamp map")

    @property
    def recording_times(self) -> tuple[float, ...]:
        return tuple(self.offset_s + self.rate_scale * t for t in self.pts_seconds)


def _stream_dict(stream: SignalStream) -> dict[str, Any]:
    data = asdict(stream)
    data["values"] = [[v if math.isfinite(v) else str(v) for v in row] for row in stream.values]
    return data


def _json_encode(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"$tuple": [_json_encode(v) for v in value]}
    if isinstance(value, list):
        return [_json_encode(v) for v in value]
    if isinstance(value, dict):
        encoded = {k: _json_encode(v) for k, v in value.items()}
        return {"$mapping": encoded} if set(value) & {"$tuple", "$mapping"} else encoded
    return value


def _json_decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_decode(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"$tuple"}:
            return tuple(_json_decode(v) for v in value["$tuple"])
        if set(value) == {"$mapping"}:
            return {k: _json_decode(v) for k, v in value["$mapping"].items()}
        return {k: _json_decode(v) for k, v in value.items()}
    return value


def _stream_load(data: dict[str, Any]) -> SignalStream:
    meta = dict(data["metadata"])
    clock = dict(meta["clock"])
    clock["clock_domain"] = ClockDomain(clock["clock_domain"])
    meta["clock"] = TimeMetadata(**clock)

    def calibration_load(value: dict[str, Any]) -> CalibrationMetadata:
        value = dict(value)
        value["input_data"] = tuple(value.get("input_data", ()))
        return CalibrationMetadata(**value)

    def provenance_load(value: dict[str, Any]) -> Provenance:
        value = dict(value)
        value["parent_ids"] = tuple(value["parent_ids"])
        return Provenance(**value)

    meta["channels"] = tuple(
        ChannelMetadata(
            **{
                **c,
                "calibration": calibration_load(c["calibration"]) if c.get("calibration") else None,
            }
        )
        for c in meta["channels"]
    )
    meta["quality"], meta["missing"] = tuple(meta["quality"]), tuple(meta["missing"])
    if meta["calibration"]:
        meta["calibration"] = calibration_load(meta["calibration"])
    if meta["provenance"]:
        meta["provenance"] = provenance_load(meta["provenance"])
    meta["missing_channels"] = tuple(tuple(row) for row in meta.get("missing_channels", ()))
    meta["uncertainty"] = tuple(tuple(row) for row in meta.get("uncertainty", ()))
    meta["lineage"] = tuple(provenance_load(p) for p in meta.get("lineage", ()))
    return SignalStream(
        tuple(data["timestamps"]),
        tuple(tuple(float(v) for v in row) for row in data["values"]),
        data["sample_rate"],
        StreamMetadata(**meta),
        tuple(Annotation(**a) for a in data["annotations"]),
    )


@dataclass
class Recording:
    id: str
    subject_id: str
    session_id: str
    group: str | None = None
    channels: dict[str, SignalStream] = field(default_factory=dict)
    annotations: list[Annotation] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    video: list[VideoTrack] = field(default_factory=list)
    journal: list[dict[str, Any]] = field(default_factory=list)
    autosave_path: Path | None = None

    def append(self, channel: str, block: SignalStream) -> None:
        old = self.channels.get(channel)
        block = replace(
            block,
            metadata=replace(
                block.metadata,
                record_id=self.id,
                subject_id=self.subject_id,
                session_id=self.session_id,
            ),
        )
        if old is not None:
            if (
                old.metadata.channels != block.metadata.channels
                or old.sample_rate != block.sample_rate
                or old.metadata.clock != block.metadata.clock
                or old.metadata.calibration != block.metadata.calibration
                or old.metadata.device_id != block.metadata.device_id
                or bool(old.metadata.missing_channels) != bool(block.metadata.missing_channels)
                or bool(old.metadata.uncertainty) != bool(block.metadata.uncertainty)
                or bool(old.metadata.quality) != bool(block.metadata.quality)
                or bool(old.metadata.missing) != bool(block.metadata.missing)
            ):
                raise ValueError(
                    "recording block channel/rate/clock/calibration/mask contract changed"
                )
            if old.timestamps and block.timestamps and block.timestamps[0] <= old.timestamps[-1]:
                raise ValueError("recording blocks overlap or go backward")
            block = replace(
                block,
                timestamps=old.timestamps + block.timestamps,
                values=old.values + block.values,
                annotations=old.annotations + block.annotations,
                metadata=replace(
                    block.metadata,
                    quality=old.metadata.quality + block.metadata.quality,
                    missing=old.metadata.missing + block.metadata.missing,
                    missing_channels=old.metadata.missing_channels
                    + block.metadata.missing_channels,
                    uncertainty=old.metadata.uncertainty + block.metadata.uncertainty,
                    lineage=old.metadata.lineage
                    + block.metadata.lineage
                    + ((block.metadata.provenance,) if block.metadata.provenance else ()),
                    provenance=Provenance(
                        "physioworld.recording",
                        "append",
                        parent_ids=(self.id,),
                        notes={
                            "channel": channel,
                            "previous_samples": old.n_samples,
                            "appended_samples": block.n_samples,
                            "lineage_storage": "flat_block_provenance",
                        },
                    ),
                ),
            )
        elif block.metadata.provenance is not None:
            block = replace(
                block,
                metadata=replace(
                    block.metadata, lineage=block.metadata.lineage + (block.metadata.provenance,)
                ),
            )
        self.channels[channel] = block
        self.journal.append(
            {"operation": "append", "channel": channel, "total_samples": block.n_samples}
        )
        self._autosave()

    def mark(self, onset: float, label: str, duration: float = 0, **attributes: Any) -> None:
        self.annotations.append(Annotation(onset, duration, label, attributes))
        self.journal.append({"operation": "mark", "onset": onset, "label": label})
        self._autosave()

    def _autosave(self) -> None:
        if self.autosave_path is not None:
            self.save(self.autosave_path)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "format": "physioworld.recording/1",
            "id": self.id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "group": self.group,
            "tags": sorted(self.tags),
            "channels": {k: _stream_dict(v) for k, v in self.channels.items()},
            "annotations": [asdict(a) for a in self.annotations],
            "video": [asdict(v) for v in self.video],
            "journal": self.journal,
        }
        payload = json.dumps(_json_encode(data), allow_nan=False, sort_keys=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        data = _json_decode(json.loads(Path(path).read_text(encoding="utf-8")))
        if data.get("format") != "physioworld.recording/1":
            raise ValueError("unsupported recording format")
        recording = cls(data["id"], data["subject_id"], data["session_id"], data["group"])
        recording.channels = {k: _stream_load(v) for k, v in data["channels"].items()}
        recording.annotations = [Annotation(**a) for a in data["annotations"]]
        recording.tags = set(data["tags"])
        recording.video = [
            VideoTrack(**{**v, "pts_seconds": tuple(v["pts_seconds"])}) for v in data["video"]
        ]
        recording.journal = data["journal"]
        return recording

    def replay(self, channel: str, block_samples: int) -> Iterable[SignalStream]:
        if block_samples < 1:
            raise ValueError("block_samples must be positive")
        s = self.channels[channel]
        for start in range(0, s.n_samples, block_samples):
            stop = start + block_samples
            yield replace(
                s,
                timestamps=s.timestamps[start:stop],
                values=s.values[start:stop],
                metadata=replace(
                    s.metadata,
                    quality=s.metadata.quality[start:stop],
                    missing=s.metadata.missing[start:stop],
                    missing_channels=s.metadata.missing_channels[start:stop],
                    uncertainty=s.metadata.uncertainty[start:stop],
                ),
                annotations=tuple(
                    a
                    for a in s.annotations
                    if s.timestamps[start] <= a.onset <= s.timestamps[min(stop, s.n_samples) - 1]
                ),
            )

    def export_csv(self, path: str | Path) -> None:
        """Long-form text/Excel-compatible interchange, including units and clocks."""
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["recording", "channel", "component", "time_s", "value", "unit", "clock", "missing"]
            )
            for name, stream in self.channels.items():
                for i, (t, row) in enumerate(zip(stream.timestamps, stream.values)):
                    for j, (c, value) in enumerate(zip(stream.metadata.channels, row)):
                        writer.writerow(
                            [
                                self.id,
                                name,
                                c.name,
                                t,
                                value,
                                c.unit,
                                stream.metadata.clock.clock_domain.value,
                                int(
                                    (
                                        stream.metadata.missing[i]
                                        if stream.metadata.missing
                                        else False
                                    )
                                    or (
                                        stream.metadata.missing_channels[i][j]
                                        if stream.metadata.missing_channels
                                        else False
                                    )
                                ),
                            ]
                        )

    @classmethod
    def import_csv(
        cls,
        path: str | Path,
        *,
        subject_id: str,
        session_id: str,
        recording_id: str | None = None,
        sample_rates: dict[str, float] | None = None,
    ) -> Recording:
        """Import the native long-form CSV. Rates are declared, never inferred.

        All components of each logical channel must have identical time grids;
        incomplete component rows are rejected rather than padded or imputed.
        """
        path = Path(path)
        grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
        record_ids = set()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record_ids.add(row["recording"])
                grouped.setdefault(row["channel"], {}).setdefault(row["component"], []).append(row)
        if not grouped or len(record_ids) != 1:
            raise ValueError("CSV must contain nonempty data from one recording")
        recording = cls(recording_id or next(iter(record_ids)), subject_id, session_id)
        provenance = Provenance(
            "physioworld.recording",
            "import_csv",
            notes={
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sample_rates": sample_rates,
                "sampling_rate_inferred": False,
                "source_recording": next(iter(record_ids)),
            },
        )
        for name, components in grouped.items():
            columns: list[tuple[float, ...]] = []
            masks: list[tuple[bool, ...]] = []
            channels: list[ChannelMetadata] = []
            clocks: set[str] = set()
            reference_times: tuple[float, ...] | None = None
            for component, rows in components.items():
                times = tuple(float(row["time_s"]) for row in rows)
                declared_units = {row["unit"] for row in rows}
                clocks.update(row["clock"] for row in rows)
                if (
                    len(declared_units) != 1
                    or any(b <= a for a, b in zip(times, times[1:]))
                    or (reference_times is not None and times != reference_times)
                ):
                    raise ValueError(
                        "CSV component units/time grid changed; explicit alignment required"
                    )
                reference_times = times
                columns.append(tuple(float(row["value"]) for row in rows))
                if any(row["missing"] not in {"0", "1"} for row in rows):
                    raise ValueError("CSV missing flags must be 0 or 1")
                masks.append(tuple(row["missing"] == "1" for row in rows))
                channels.append(ChannelMetadata(component, next(iter(declared_units))))
            if len(clocks) != 1:
                raise ValueError("CSV logical channel mixes clocks")
            if reference_times is None:
                raise ValueError("CSV logical channel has no timestamp rows")
            metadata = StreamMetadata(
                modality="imported",
                channels=tuple(channels),
                clock=TimeMetadata(ClockDomain(next(iter(clocks)))),
                provenance=provenance,
                missing_channels=tuple(zip(*masks)),
                sampling_semantics="irregular",
            )
            stream = SignalStream(
                reference_times, tuple(zip(*columns)), (sample_rates or {}).get(name), metadata
            )
            recording.append(name, stream)
        return recording

    def export_mat(self, path: str | Path) -> None:
        try:
            from scipy.io import savemat
        except ImportError as exc:
            raise ImportError("MATLAB .mat export requires optional package scipy") from exc
        # Cell-like nested records preserve arbitrary researcher channel names.
        savemat(
            str(path),
            {
                "recording_id": self.id,
                "channels": [
                    {
                        "name": name,
                        "timestamps": s.timestamps,
                        "values": s.values,
                        "units": [c.unit for c in s.metadata.channels],
                    }
                    for name, s in self.channels.items()
                ],
            },
        )


@dataclass
class Session:
    id: str
    subject_id: str
    group: str | None = None
    recordings: dict[str, Recording] = field(default_factory=dict)

    def add(self, recording: Recording) -> None:
        if recording.session_id != self.id or recording.subject_id != self.subject_id:
            raise ValueError("recording subject/session mismatch")
        if recording.id in self.recordings:
            raise ValueError("duplicate recording id")
        self.recordings[recording.id] = recording

    def batch(self, channel: str, calculation: Callable[[SignalStream], Any]) -> dict[str, Any]:
        return {name: calculation(r.channels[channel]) for name, r in self.recordings.items()}


@dataclass
class LinkedCursor:
    time_s: float = 0.0


@dataclass
class ChartView:
    recording: Recording
    cursor: LinkedCursor = field(default_factory=LinkedCursor)
    start: float = -math.inf
    stop: float = math.inf

    def zoom(self, start: float, stop: float) -> None:
        if start >= stop:
            raise ValueError("zoom must have positive extent")
        self.start, self.stop = start, stop

    def data(self, channel: str) -> tuple[tuple[float, tuple[float, ...]], ...]:
        s = self.recording.channels[channel]
        left = bisect.bisect_left(s.timestamps, self.start)
        right = bisect.bisect_right(s.timestamps, self.stop)
        return tuple(zip(s.timestamps[left:right], s.values[left:right]))

    def events(self) -> tuple[Annotation, ...]:
        annotations = list(self.recording.annotations) + [
            a for s in self.recording.channels.values() for a in s.annotations
        ]
        return tuple(
            sorted(
                (
                    a
                    for a in annotations
                    if a.onset <= self.stop and a.onset + a.duration >= self.start
                ),
                key=lambda a: a.onset,
            )
        )

    def meter(self, channel: str) -> tuple[float, ...]:
        s = self.recording.channels[channel]
        if not s.timestamps:
            raise ValueError("empty meter channel")
        i = min(bisect.bisect_left(s.timestamps, self.cursor.time_s), s.n_samples - 1)
        if i and abs(s.timestamps[i - 1] - self.cursor.time_s) <= abs(
            s.timestamps[i] - self.cursor.time_s
        ):
            i -= 1
        return s.values[i]

    def xy(
        self, x: str, y: str, x_component: int = 0, y_component: int = 0
    ) -> tuple[tuple[float, float], ...]:
        sx, sy = self.recording.channels[x], self.recording.channels[y]
        if sx.timestamps != sy.timestamps or not same_clock(sx, sy):
            raise ValueError("XY requires explicitly aligned timestamps/clocks")
        return tuple(
            (a[x_component], b[y_component]) for (_, a), (_, b) in zip(self.data(x), self.data(y))
        )

    def histogram(
        self, channel: str, edges: tuple[float, ...], component: int = 0
    ) -> tuple[int, ...]:
        if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
            raise ValueError("histogram edges must strictly increase")
        counts = [0] * (len(edges) - 1)
        for _, row in self.data(channel):
            v = row[component]
            if math.isfinite(v) and edges[0] <= v <= edges[-1]:
                counts[min(bisect.bisect_right(edges, v) - 1, len(counts) - 1)] += 1
        return tuple(counts)

    def extract(self, channel: str, component: int = 0) -> Measurements:
        s = self.recording.channels[channel]
        indices = [i for i, t in enumerate(s.timestamps) if self.start <= t <= self.stop]
        sub = replace(
            s,
            timestamps=tuple(s.timestamps[i] for i in indices),
            values=tuple(s.values[i] for i in indices),
            metadata=replace(
                s.metadata,
                missing=tuple(s.metadata.missing[i] for i in indices) if s.metadata.missing else (),
                quality=tuple(s.metadata.quality[i] for i in indices) if s.metadata.quality else (),
                missing_channels=tuple(s.metadata.missing_channels[i] for i in indices)
                if s.metadata.missing_channels
                else (),
                uncertainty=tuple(s.metadata.uncertainty[i] for i in indices)
                if s.metadata.uncertainty
                else (),
            ),
        )
        return statistics_extract(sub, component)


@dataclass
class ScopeView(ChartView):
    def overlays(
        self, channel: str, events: EventSeries, before_s: float, after_s: float
    ) -> tuple[tuple[tuple[float, tuple[float, ...]], ...], ...]:
        if before_s < 0 or after_s < 0:
            raise ValueError("scope extents must be nonnegative")
        s = self.recording.channels[channel]
        return tuple(
            tuple(
                (t - event, row)
                for t, row in zip(s.timestamps, s.values)
                if event - before_s <= t <= event + after_s
            )
            for event in events.times
        )

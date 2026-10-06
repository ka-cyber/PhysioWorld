"""Real CSV, canonical JSON, and streaming NDJSON interchange using the stdlib."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from physioworld.core import Provenance
from physioworld.physiology.record import PhysiologicalRecord
from physioworld.physiology.stream import ChannelMetadata, SignalStream
from physioworld.physiology.time import TimeMetadata

from .canonical import decode_json, dumps, loads, pack, unpack

ScientificObject = SignalStream | PhysiologicalRecord


class JSONAdapter:
    def write(self, obj: ScientificObject, path: str | Path) -> None:
        Path(path).write_text(dumps(obj) + "\n", encoding="utf-8")

    def read(self, path: str | Path) -> ScientificObject:
        return loads(Path(path).read_text(encoding="utf-8"))


class NDJSONAdapter:
    def write(self, objects: Iterable[ScientificObject], path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            for obj in objects:
                handle.write(dumps(obj) + "\n")

    def read(self, path: str | Path) -> Iterator[ScientificObject]:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f"blank NDJSON record at line {line_number}")
                try:
                    yield loads(line)
                except ValueError as exc:
                    raise ValueError(f"NDJSON line {line_number}: {exc}") from exc


def signal_header(stream: SignalStream) -> str:
    # Keep full per-sample metadata but store sample numbers in actual table columns.
    md = replace(stream.metadata, sampling_semantics="declared")
    skeleton = replace(
        stream,
        timestamps=(),
        values=(),
        metadata=replace(md, quality=(), missing=(), missing_channels=(), uncertainty=()),
    )
    return json.dumps(
        {
            "signal": json.loads(dumps(skeleton)),
            "sample_metadata": pack(
                {
                    "quality": stream.metadata.quality,
                    "missing": stream.metadata.missing,
                    "missing_channels": stream.metadata.missing_channels,
                    "uncertainty": stream.metadata.uncertainty,
                    "sampling_semantics": stream.metadata.sampling_semantics,
                }
            ),
        },
        allow_nan=False,
        sort_keys=True,
    )


def restore_signal(
    header: str, timestamps: Sequence[float], values: Sequence[Sequence[float]]
) -> SignalStream:
    raw = decode_json(header)
    skeleton = loads(json.dumps(raw["signal"]))
    if not isinstance(skeleton, SignalStream):
        raise ValueError("table envelope must contain a signal")
    md = replace(skeleton.metadata, **unpack(raw["sample_metadata"]))
    return replace(
        skeleton, timestamps=tuple(timestamps), values=tuple(tuple(r) for r in values), metadata=md
    )


class CSVAdapter:
    """Canonical CSV has a metadata preamble and numeric time/channel columns.

    Ordinary CSV requires an explicit column->unit map, timestamp column, clock,
    and identity. Empty numeric cells become *declared* missing NaNs. Row order is
    retained and invalid/nonmonotone clocks are rejected by SignalStream.
    """

    def write(self, stream: SignalStream, path: str | Path) -> None:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            handle.write("#physioworld " + signal_header(stream) + "\n")
            writer = csv.writer(handle)
            writer.writerow(["timestamp_s"] + [c.name for c in stream.metadata.channels])
            writer.writerows(
                (repr(t), *(repr(x) for x in row))
                for t, row in zip(stream.timestamps, stream.values)
            )

    def read(
        self,
        path: str | Path,
        *,
        columns: Mapping[str, str] | None = None,
        timestamp_column: str | None = None,
        clock: TimeMetadata | None = None,
        modality: str = "unknown",
        subject_id: str | None = None,
        record_id: str | None = None,
        session_id: str | None = None,
        sample_rate: float | None = None,
    ) -> SignalStream:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            first = handle.readline()
            if first.startswith("#physioworld "):
                header = first[len("#physioworld ") :].strip()
                canonical_reader = csv.reader(handle)
                names = next(canonical_reader)
                skeleton = loads(json.dumps(decode_json(header)["signal"]))
                if not isinstance(skeleton, SignalStream):
                    raise ValueError("table envelope must contain a signal")
                if names != ["timestamp_s"] + [c.name for c in skeleton.metadata.channels]:
                    raise ValueError("CSV columns disagree with canonical channel metadata")
                times, rows = [], []
                for line in canonical_reader:
                    if len(line) != len(names):
                        raise ValueError("CSV sample width mismatch")
                    times.append(float(line[0]))
                    rows.append(tuple(float(x) for x in line[1:]))
                return restore_signal(header, times, rows)
            handle.seek(0)
            if not columns or not timestamp_column or clock is None:
                raise ValueError(
                    "ordinary CSV requires columns with units, timestamp_column, and clock"
                )
            dict_reader = csv.DictReader(handle)
            if not dict_reader.fieldnames or not (set(columns) | {timestamp_column}) <= set(
                dict_reader.fieldnames
            ):
                raise ValueError("missing declared CSV columns")
            if len(dict_reader.fieldnames) != len(set(dict_reader.fieldnames)):
                raise ValueError("duplicate CSV column names")
            times, rows, gaps, unmapped = [], [], [], []
            other_columns = tuple(
                c for c in dict_reader.fieldnames if c not in columns and c != timestamp_column
            )
            for row in dict_reader:
                if None in row or any(row[c] is None for c in (*columns, timestamp_column)):
                    raise ValueError("malformed CSV row")
                clean_row = {
                    key: value
                    for key, value in row.items()
                    if key is not None and value is not None
                }
                times.append(float(clean_row[timestamp_column]))
                values = tuple(
                    float(clean_row[c]) if clean_row[c].strip() else float("nan") for c in columns
                )
                rows.append(values)
                gaps.append(tuple(not math.isfinite(v) for v in values))
                unmapped.append({c: clean_row[c] for c in other_columns})
        stream = SignalStream.from_samples(
            rows,
            sample_rate,
            next(iter(columns.values())),
            modality,
            timestamps=times,
            channel_names=tuple(columns),
            subject_id=subject_id,
            record_id=record_id,
            session_id=session_id,
            clock=clock,
            missing_channels=gaps,
            attributes={"unmapped_columns": other_columns, "unmapped_rows": tuple(unmapped)},
            provenance=Provenance(
                "local_csv",
                "read",
                notes={
                    "columns": dict(columns),
                    "timestamp_column": timestamp_column,
                    "missing_policy": "empty_or_nonfinite",
                },
            ),
        )
        return replace(
            stream,
            metadata=replace(
                stream.metadata,
                channels=tuple(ChannelMetadata(c, unit) for c, unit in columns.items()),
            ),
        )

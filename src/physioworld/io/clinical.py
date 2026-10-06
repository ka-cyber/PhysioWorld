"""Explicit clinical/sensor standard mappings into canonical scientific semantics.

Supported profiles: FHIR R4 numeric Observation/components/SampledData, Open
mHealth numeric body values, OMOP CDM measurement rows with supplied concept
and UCUM mappings, and BIDS physiological TSV sidecars. Source documents are
retained in provenance; unsupported profiles fail instead of inventing values.
"""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from physioworld.core import Provenance
from physioworld.physiology.record import PhysiologicalRecord, RecordEvent
from physioworld.physiology.stream import ChannelMetadata, SignalStream
from physioworld.physiology.time import CalibrationMetadata, ClockDomain, TimeMetadata

from .canonical import decode_json


def iso_seconds(value: str) -> float:
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if instant.tzinfo is None:
        raise ValueError("absolute timestamps require an explicit timezone")
    return instant.timestamp()


def _ucum(quantity: Mapping[str, Any]) -> str:
    if quantity.get("system") != "http://unitsofmeasure.org" or not quantity.get("code"):
        raise ValueError("FHIR numeric quantities require explicit UCUM system and code")
    return quantity["code"]


def _reference(data: Mapping[str, Any] | None, field: str) -> str:
    if not data or not data.get("reference"):
        raise ValueError(f"explicit {field} reference is required")
    return str(data["reference"])


class FHIRAdapter:
    def from_resource(self, resource: Mapping[str, Any]) -> PhysiologicalRecord:
        if resource.get("resourceType") != "Observation":
            raise ValueError("FHIR adapter supports numeric Observation resources")
        subject = _reference(resource.get("subject"), "subject")
        rid = resource.get("id")
        if not rid:
            raise ValueError("FHIR Observation.id is required")
        session = (
            _reference(resource["encounter"], "encounter") if resource.get("encounter") else None
        )
        device = _reference(resource["device"], "device") if resource.get("device") else None
        timestamp = resource.get("effectiveDateTime")
        if timestamp is None:
            timestamp = resource.get("effectivePeriod", {}).get("start")
        if timestamp is None:
            raise ValueError("FHIR Observation requires effectiveDateTime or effectivePeriod.start")
        start = iso_seconds(timestamp)
        clock = TimeMetadata(
            ClockDomain.ACQUISITION,
            origin=timestamp,
            wall_clock_start_s=start,
            synchronization_method="ISO8601",
            time_basis="unix",
        )
        components = resource.get("component") or (resource,)
        streams = {}
        period = resource.get("effectivePeriod")
        duration = 0.0
        if period and period.get("end"):
            duration = iso_seconds(period["end"]) - start
            if duration < 0:
                raise ValueError("FHIR effectivePeriod end precedes start")
        for component in components:
            coding = component.get("code", {}).get("coding", ())
            name = (
                str(coding[0]["code"])
                if coding and coding[0].get("code")
                else component.get("code", {}).get("text")
            )
            if not name:
                raise ValueError("FHIR observation/component requires a coded or textual variable")
            if name in streams:
                raise ValueError("duplicate FHIR component code")
            attrs = {
                "code": component["code"],
                "status": resource.get("status"),
                "issued": resource.get("issued"),
                "interpretation": component.get("interpretation", ()),
            }
            if "valueSampledData" in component:
                sampled = component["valueSampledData"]
                unit = _ucum(sampled["origin"])
                dimensions = sampled["dimensions"]
                period_ms = sampled["period"]
                if (
                    not isinstance(dimensions, int)
                    or dimensions < 1
                    or not math.isfinite(period_ms)
                    or period_ms <= 0
                ):
                    raise ValueError("invalid FHIR SampledData dimensions/period")
                tokens = sampled["data"].split()
                if not tokens or len(tokens) % dimensions:
                    raise ValueError("FHIR SampledData count not divisible by dimensions")
                factor, origin = float(sampled.get("factor", 1)), float(sampled["origin"]["value"])
                if not math.isfinite(factor) or not math.isfinite(origin) or factor == 0:
                    raise ValueError("invalid FHIR SampledData calibration")
                rows: list[tuple[float, ...]] = []
                missing: list[tuple[bool, ...]] = []
                for offset in range(0, len(tokens), dimensions):
                    chunk = tokens[offset : offset + dimensions]
                    flags = tuple(t in {"E", "L", "U"} for t in chunk)
                    missing.append(flags)
                    rows.append(
                        tuple(
                            float("nan") if flag else float(t) * factor + origin
                            for t, flag in zip(chunk, flags)
                        )
                    )
                if any(
                    not math.isfinite(v) and not flag
                    for row, flags in zip(rows, missing)
                    for v, flag in zip(row, flags)
                ):
                    raise ValueError(
                        "FHIR SampledData numeric tokens must be finite; use E/L/U for missing values"
                    )
                times = [start + i * period_ms / 1000 for i in range(len(rows))]
                rate = 1000 / period_ms
                attrs["sampled_data_tokens"] = tuple(tokens)
                calibration = CalibrationMetadata(
                    factor,
                    origin,
                    method="FHIR.SampledData",
                    output_unit=unit,
                    status="ASSUMED",
                    operating_conditions={"profile": "FHIR R4"},
                )
            elif "valueQuantity" in component and "value" in component["valueQuantity"]:
                quantity = component["valueQuantity"]
                if "comparator" in quantity:
                    raise ValueError(
                        "censored FHIR Quantity cannot be treated as a point measurement"
                    )
                unit = _ucum(quantity)
                rows = [(float(quantity["value"]),)]
                times = [start]
                rate = None
                missing = []
                calibration = None
                if not math.isfinite(rows[0][0]):
                    raise ValueError("FHIR valueQuantity.value must be finite")
            elif "dataAbsentReason" in component:
                # Missing values still require declared measurement units.
                unit = _ucum(component.get("valueQuantity", component.get("unit", {})))
                rows = [(float("nan"),)]
                times = [start]
                rate = None
                missing = [(True,)]
                calibration = None
                attrs["dataAbsentReason"] = component["dataAbsentReason"]
            else:
                raise ValueError("unsupported nonnumeric FHIR Observation value")
            stream = SignalStream.from_samples(
                rows,
                rate,
                unit,
                "clinical_measurement",
                timestamps=times,
                channel_names=[name]
                if len(rows[0]) == 1
                else [f"{name}:{i}" for i in range(len(rows[0]))],
                subject_id=subject,
                record_id=str(rid),
                session_id=session,
                device_id=device,
                clock=clock,
                missing_channels=missing,
                calibration=calibration,
                attributes=attrs,
                data_kind="clinical",
                provenance=Provenance(
                    "FHIR R4", "map_observation", notes={"source_resource": dict(resource)}
                ),
            )
            attrs["measurement_interval"] = period
            streams[name] = replace(
                stream,
                metadata=replace(
                    stream.metadata,
                    sampling_semantics="regular" if rate is not None else "event",
                    attributes=attrs,
                ),
            )
        return PhysiologicalRecord(
            subject,
            str(rid),
            session,
            streams,
            device_id=device,
            events=(
                RecordEvent(
                    start,
                    "measurement_interval",
                    duration,
                    "FHIR R4",
                    {"effectivePeriod": period},
                    clock,
                    "measurement_interval",
                ),
            )
            if period
            else (),
            clinical_labels={"status": resource.get("status")},
            provenance={"standard": "FHIR R4", "resource": dict(resource)},
            metadata={
                "data_kind": "clinical",
                "effective_time": timestamp,
                "effective_period": resource.get("effectivePeriod"),
            },
        )

    def read(self, path: str | Path) -> PhysiologicalRecord:
        return self.from_resource(decode_json(Path(path).read_text(encoding="utf-8")))


class OpenMHealthAdapter:
    def read(
        self,
        path: str | Path,
        *,
        subject_id: str,
        unit_map: Mapping[str, str],
        session_id: str | None = None,
    ) -> PhysiologicalRecord:
        return self.from_document(
            decode_json(Path(path).read_text(encoding="utf-8")),
            subject_id=subject_id,
            unit_map=unit_map,
            session_id=session_id,
        )

    def from_document(
        self,
        document: Mapping[str, Any],
        *,
        subject_id: str,
        unit_map: Mapping[str, str],
        session_id: str | None = None,
    ) -> PhysiologicalRecord:
        header, body = document["header"], document["body"]
        rid = header.get("id")
        if not rid or not subject_id:
            raise ValueError("Open mHealth requires header.id and explicit subject identity")
        frame = body.get("effective_time_frame", {})
        time = frame.get("date_time") or frame.get("time_interval", {}).get("start_date_time")
        if not time:
            raise ValueError("Open mHealth requires an effective time with timezone")
        start = iso_seconds(time)
        clock = TimeMetadata(
            ClockDomain.ACQUISITION, origin=time, wall_clock_start_s=start, time_basis="unix"
        )
        streams = {}
        for name, ucum in unit_map.items():
            quantity = body.get(name)
            if (
                not isinstance(quantity, Mapping)
                or "value" not in quantity
                or "unit" not in quantity
            ):
                raise ValueError(f"missing Open mHealth numeric field {name}")
            if not ucum:
                raise ValueError("explicit Open mHealth->UCUM unit mapping required")
            # Only declared equivalent unit spellings are mapped, never rescaled implicitly.
            aliases = {
                "beats/min": "/min",
                "breaths/min": "/min",
                "mmHg": "mm[Hg]",
                "percent": "%",
                "C": "Cel",
                "kg": "kg",
                "mg/dL": "mg/dL",
            }
            source_ucum = aliases.get(quantity["unit"], quantity["unit"])
            if source_ucum != ucum:
                raise ValueError(
                    f"Open mHealth unit {quantity['unit']!r} does not match declared UCUM {ucum!r}; convert explicitly"
                )
            if not math.isfinite(float(quantity["value"])):
                raise ValueError("Open mHealth numeric quantities must be finite")
            streams[name] = SignalStream.from_samples(
                [quantity["value"]],
                None,
                ucum,
                "health_measurement",
                timestamps=[start],
                channel_names=[name],
                subject_id=subject_id,
                record_id=rid,
                session_id=session_id,
                clock=clock,
                data_kind="clinical",
                attributes={"source_unit": quantity["unit"], "unit_mapping": ucum},
                provenance=Provenance(
                    "Open mHealth", "map_numeric_body", notes={"document": dict(document)}
                ),
            )
            streams[name] = replace(
                streams[name],
                metadata=replace(
                    streams[name].metadata,
                    sampling_semantics="event",
                    attributes={
                        "source_unit": quantity["unit"],
                        "unit_mapping": ucum,
                        "effective_time_frame": frame,
                    },
                ),
            )
        if not streams:
            raise ValueError("Open mHealth requires at least one explicit numeric field mapping")
        return PhysiologicalRecord(
            subject_id,
            rid,
            session_id,
            streams,
            provenance={"standard": "Open mHealth", "document": dict(document)},
            metadata={
                "schema_id": header.get("schema_id"),
                "effective_time_frame": frame,
                "acquisition_provenance": header.get("acquisition_provenance"),
            },
        )


class OMOPAdapter:
    def read(
        self,
        path: str | Path,
        *,
        format: str,
        concept_names: Mapping[int, str],
        unit_concepts: Mapping[int, str],
        timezone_name: str | None = None,
    ) -> tuple[PhysiologicalRecord, ...]:
        if format == "json":
            rows = decode_json(Path(path).read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError("OMOP JSON input must be an array of measurement rows")
        elif format == "csv":
            with Path(path).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError("OMOP CSV requires unique column names")
                rows = list(reader)
                if any(None in row or any(value is None for value in row.values()) for row in rows):
                    raise ValueError("malformed OMOP CSV row")
        else:
            raise ValueError("OMOP format must be explicitly json or csv")
        return self.from_measurements(
            rows,
            concept_names=concept_names,
            unit_concepts=unit_concepts,
            timezone_name=timezone_name,
        )

    def from_measurements(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        concept_names: Mapping[int, str],
        unit_concepts: Mapping[int, str],
        timezone_name: str | None = None,
    ) -> tuple[PhysiologicalRecord, ...]:
        """Map numeric CDM measurement rows. Naive CDM times require a supplied IANA zone."""
        from zoneinfo import ZoneInfo

        records, seen = [], set()
        for row in rows:
            if row.get("measurement_id") in (None, "") or row.get("person_id") in (None, ""):
                raise ValueError("OMOP requires measurement_id and person_id")
            rid, subject = str(row["measurement_id"]), str(row["person_id"])
            if rid in seen:
                raise ValueError(f"duplicate OMOP measurement_id: {rid}")
            seen.add(rid)
            time = row.get("measurement_datetime")
            if not time:
                raise ValueError(
                    "OMOP requires measurement_datetime; date alone is not acquisition time"
                )
            instant = datetime.fromisoformat(str(time).replace("Z", "+00:00"))
            if instant.tzinfo is None:
                if timezone_name is None:
                    raise ValueError(
                        "naive OMOP measurement_datetime requires explicit timezone_name"
                    )
                instant = instant.replace(tzinfo=ZoneInfo(timezone_name))
                if instant.utcoffset() != instant.replace(fold=1).utcoffset():
                    raise ValueError("ambiguous OMOP local time; provide an offset-aware timestamp")
                if instant.astimezone(timezone.utc).astimezone(instant.tzinfo).replace(
                    tzinfo=None
                ) != instant.replace(tzinfo=None):
                    raise ValueError("nonexistent OMOP local time")
            name = concept_names.get(int(row["measurement_concept_id"]))
            unit = unit_concepts.get(int(row["unit_concept_id"]))
            if not name or not unit:
                raise ValueError(
                    "OMOP measurement and unit concepts require explicit name/UCUM mappings"
                )
            if row.get("operator_concept_id") not in (None, "", 0, "0"):
                raise ValueError("OMOP operator/censored measurements are not point values")
            value = row.get("value_as_number")
            value = None if value == "" else value
            if value is None and row.get("value_as_concept_id") not in (None, "", 0, "0"):
                raise ValueError("categorical OMOP measurement is not a numeric signal")
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("OMOP numeric measurements must be finite or explicitly absent")
            session = (
                str(row["visit_occurrence_id"])
                if row.get("visit_occurrence_id") not in (None, "")
                else None
            )
            stream = SignalStream.from_samples(
                [float(value) if value is not None else float("nan")],
                None,
                unit,
                "clinical_measurement",
                timestamps=[instant.timestamp()],
                subject_id=subject,
                record_id=rid,
                session_id=session,
                channel_names=[name],
                missing=[value is None],
                clock=TimeMetadata(ClockDomain.ACQUISITION, origin=str(time), time_basis="unix"),
                data_kind="clinical",
                provenance=Provenance(
                    "OMOP CDM", "map_measurement", notes={"source_row": dict(row)}
                ),
                attributes={
                    "measurement_concept_id": row["measurement_concept_id"],
                    "unit_concept_id": row["unit_concept_id"],
                    "range_low": row.get("range_low"),
                    "range_high": row.get("range_high"),
                },
            )
            stream = replace(stream, metadata=replace(stream.metadata, sampling_semantics="event"))
            records.append(
                PhysiologicalRecord(
                    subject,
                    rid,
                    session,
                    {name: stream},
                    provenance={"standard": "OMOP CDM", "measurement": dict(row)},
                    metadata={"data_kind": "clinical", "timezone": timezone_name},
                )
            )
        return tuple(records)


class BIDSAdapter:
    def read(
        self,
        tsv: str | Path,
        sidecar: str | Path,
        *,
        subject_id: str,
        record_id: str,
        session_id: str | None,
        clock: TimeMetadata,
        events_tsv: str | Path | None = None,
    ) -> PhysiologicalRecord:
        if clock.time_basis != "relative":
            raise ValueError(
                "BIDS StartTime uses a relative acquisition clock; transform to absolute time explicitly"
            )
        metadata = decode_json(Path(sidecar).read_text(encoding="utf-8"))
        rate, start, columns = (
            metadata["SamplingFrequency"],
            metadata["StartTime"],
            metadata["Columns"],
        )
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(start):
            raise ValueError("invalid BIDS SamplingFrequency/StartTime")
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("BIDS Columns must be nonempty and unique")
        units: list[str] = []
        for column in columns:
            unit = metadata.get(column, {}).get("Units")
            if not unit:
                raise ValueError(f"BIDS channel {column} requires explicit Units")
            units.append(unit)
        if str(tsv).endswith(".gz"):
            import gzip

            handle = gzip.open(tsv, "rt", encoding="utf-8")
        else:
            handle = Path(tsv).open(encoding="utf-8")
        rows: list[tuple[float, ...]] = []
        with handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) != len(columns):
                    raise ValueError("BIDS sample width differs from Columns")
                rows.append(tuple(float("nan") if x == "n/a" else float(x) for x in row))
        stream = SignalStream.from_samples(
            rows,
            rate,
            units[0],
            "BIDS_physio",
            timestamps=[start + i / rate for i in range(len(rows))],
            channel_names=columns,
            subject_id=subject_id,
            record_id=record_id,
            session_id=session_id,
            clock=clock,
            missing_channels=[tuple(not math.isfinite(v) for v in row) for row in rows],
            provenance=Provenance("BIDS", "read_physio", notes={"sidecar": metadata}),
        )
        stream = replace(
            stream,
            metadata=replace(
                stream.metadata,
                sampling_semantics="regular",
                channels=tuple(
                    ChannelMetadata(name, unit, description=metadata[name].get("Description"))
                    for name, unit in zip(columns, units)
                ),
                attributes={"bids_sidecar": metadata},
            ),
        )
        events = []
        if events_tsv:
            with Path(events_tsv).open(encoding="utf-8") as handle:
                for event_row in csv.DictReader(handle, delimiter="\t"):
                    label = event_row.get("trial_type") or event_row.get("value")
                    if not label:
                        raise ValueError("BIDS event requires trial_type or value")
                    events.append(
                        RecordEvent(
                            float(event_row["onset"]),
                            label,
                            float(event_row["duration"]),
                            "BIDS",
                            dict(event_row),
                            clock,
                        )
                    )
        return PhysiologicalRecord(
            subject_id,
            record_id,
            session_id,
            {"physio": stream},
            tuple(events),
            provenance={"standard": "BIDS", "sidecar": metadata},
        )

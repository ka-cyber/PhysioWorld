"""Versioned, dependency-free scientific interchange (no pickle or NaN JSON).

Tuples and non-finite sample values are tagged to preserve their semantics in
arbitrary nested metadata. Unknown versions/fields are rejected, not guessed.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Mapping

from physioworld.core import Provenance
from physioworld.physiology.record import PhysiologicalRecord, RecordEvent
from physioworld.physiology.stream import Annotation, ChannelMetadata, SignalStream, StreamMetadata
from physioworld.physiology.time import CalibrationMetadata, TimeMetadata

SCHEMA_VERSION = "1.0.0"
FORMAT = "physioworld.canonical"


def pack(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: pack(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return {"$tuple": [pack(v) for v in value]}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("canonical metadata keys must be strings")
        # Escape reserved keys even when they occur in user-authored attributes.
        result = {k: pack(v) for k, v in value.items()}
        return {"$mapping": result} if any(k.startswith("$") for k in value) else result
    if isinstance(value, float) and not math.isfinite(value):
        return {"$float": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"unsupported canonical metadata type: {type(value).__name__}")


def unpack(value: Any) -> Any:
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"$tuple"}:
            if not isinstance(value["$tuple"], list):
                raise ValueError("malformed tuple tag")
            return tuple(unpack(v) for v in value["$tuple"])
        if set(value) == {"$float"}:
            options = {"nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf")}
            if value["$float"] not in options:
                raise ValueError("malformed float tag")
            return options[value["$float"]]
        if set(value) == {"$mapping"}:
            if not isinstance(value["$mapping"], dict):
                raise ValueError("malformed mapping tag")
            return {k: unpack(v) for k, v in value["$mapping"].items()}
        return {k: unpack(v) for k, v in value.items()}
    return value


def _provenance(data: Any) -> Provenance | None:
    return Provenance(**data) if data is not None else None


def _clock(data: Any) -> TimeMetadata:
    if not isinstance(data, Mapping) or not {"clock_domain", "time_unit", "time_basis"} <= set(
        data
    ):
        raise ValueError(
            "canonical clocks require explicit clock_domain, time_unit, and time_basis"
        )
    return TimeMetadata(**data)


def _calibration(data: Any) -> CalibrationMetadata | None:
    if data is None:
        return None
    if not isinstance(data, Mapping) or not {"gain", "offset", "method"} <= set(data):
        raise ValueError("canonical calibration must declare gain, offset, and method (or be null)")
    return CalibrationMetadata(**data)


def _stream(data: Mapping[str, Any]) -> SignalStream:
    md = dict(data["metadata"])
    md["clock"] = _clock(md["clock"])
    md["calibration"] = _calibration(md.get("calibration"))
    channels = []
    for channel in md["channels"]:
        channel = dict(channel)
        channel["calibration"] = _calibration(channel.get("calibration"))
        channels.append(ChannelMetadata(**channel))
    md["channels"] = tuple(channels)
    md["provenance"] = _provenance(md.get("provenance"))
    md["lineage"] = tuple(Provenance(**p) for p in md.get("lineage", ()))
    kwargs = dict(data)
    kwargs["metadata"] = StreamMetadata(**md)
    kwargs["annotations"] = tuple(Annotation(**a) for a in data["annotations"])
    kwargs["timestamps"] = tuple(data["timestamps"])
    kwargs["values"] = tuple(tuple(row) for row in data["values"])
    return SignalStream(**kwargs)


def to_dict(obj: SignalStream | PhysiologicalRecord) -> dict[str, Any]:
    if not isinstance(obj, (SignalStream, PhysiologicalRecord)):
        raise TypeError("canonical interchange supports signals and physiological records")
    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "object_type": "signal" if isinstance(obj, SignalStream) else "record",
        "data": pack(obj),
    }


def from_dict(payload: Mapping[str, Any]) -> SignalStream | PhysiologicalRecord:
    if not isinstance(payload, Mapping) or set(payload) != {
        "format",
        "schema_version",
        "object_type",
        "data",
    }:
        raise ValueError("malformed canonical envelope")
    if payload["format"] != FORMAT or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical format/version: {payload.get('schema_version')!r}")
    try:
        data = unpack(payload["data"])
        if payload["object_type"] == "signal":
            return _stream(data)
        if payload["object_type"] == "record":
            kwargs = dict(data)
            kwargs["streams"] = {k: _stream(v) for k, v in data["streams"].items()}
            events = []
            for event in data["events"]:
                event = dict(event)
                event["clock"] = _clock(event["clock"])
                events.append(RecordEvent(**event))
            kwargs["events"] = tuple(events)
            return PhysiologicalRecord(**kwargs)
        raise ValueError("unknown canonical object_type")
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"malformed canonical {payload['object_type']}: {exc}") from exc


def dumps(obj: SignalStream | PhysiologicalRecord) -> str:
    return json.dumps(to_dict(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_json(text: str) -> Any:
    """Strict JSON parser shared by native-format metadata readers."""

    def reject(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}; use canonical float tags")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject, object_pairs_hook=pairs)


def loads(text: str) -> SignalStream | PhysiologicalRecord:
    return from_dict(decode_json(text))

"""Typed experiment declarations and dependency-free scientific validation.

Compilation validates configuration, never executes a pipeline or acquires data.
The installed schemas are the structural contract; cross-field scientific checks
below complement them without requiring the optional ``jsonschema`` package.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, Mapping

from physioworld.evidence_origin import source_evidence_class
from physioworld.io import load_config


class ExperimentValidationError(ValueError):
    """A declaration cannot be compiled without implicit scientific choices."""


ConfigurationKind = Literal["synthetic_samples", "vertical", "dataset_readiness"]
DataMode = Literal["synthetic", "replay", "dataset"]
SCHEMA_VERSION = "1.0.0"


def schema_names() -> tuple[str, ...]:
    """Return reviewed schema resources available in an installed distribution."""
    return tuple(
        sorted(
            entry.name
            for entry in files("physioworld").joinpath("schemas").iterdir()
            if entry.name.endswith(".schema.json") and entry.is_file()
        )
    )


def load_schema(name: str) -> dict[str, Any]:
    """Load a bundled schema by basename, independent of the source checkout."""
    if name not in schema_names():
        raise ExperimentValidationError(f"no reviewed schema named {name!r}")
    resource = files("physioworld").joinpath("schemas").joinpath(name)
    return json.loads(resource.read_text(encoding="utf-8"))


def configuration_schema_name(configuration: Mapping[str, Any]) -> str:
    """Identify the scientific branch, rather than trusting a schema declaration."""
    raw = configuration.get("experiment", configuration)
    if not isinstance(raw, Mapping):
        raise ExperimentValidationError("experiment root must be an object")
    corpus_markers = {
        "corpus_id",
        "execution_profile",
        "fixture_root",
        "seed_policy",
        "external_inputs",
        "cases",
    }
    if corpus_markers.intersection(raw):
        raise ExperimentValidationError(
            "corpus fields cannot be mixed with experiment declarations; "
            "use validate --family native_level3_synthetic for a corpus configuration"
        )
    if "vertical" in raw:
        return "vertical.schema.json"
    if "samples" in raw:
        return "world.schema.json"
    return "experiment.schema.json"


def _json_value(value: Any, path: str = "experiment") -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ExperimentValidationError(f"{path}: object keys must be strings")
        return {key: _json_value(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ExperimentValidationError(f"{path}: unsupported JSON value {type(value).__name__}")


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "null": value is None,
    }[expected]


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) in (int, float) and type(right) in (int, float):
        return bool(left == right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _validate_structure(
    value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], path: str = "experiment"
) -> None:
    """Evaluate the structural keywords used by the reviewed execution schemas.

    This intentionally is not a general JSON Schema validator. The optional
    schema checker validates the full Draft 2020-12 documents independently.
    """

    def fail(message: str) -> None:
        raise ExperimentValidationError(f"{path}: {message}")

    def valid(candidate: Mapping[str, Any]) -> bool:
        try:
            _validate_structure(value, candidate, root, path)
        except ExperimentValidationError:
            return False
        return True

    if "$ref" in schema:
        name, _, fragment = schema["$ref"].partition("#")
        document = load_schema(name) if name else root
        referenced = document
        for part in fragment.strip("/").split("/") if fragment else ():
            referenced = referenced[part.replace("~1", "/").replace("~0", "~")]
        _validate_structure(value, referenced, document, path)
    if "type" in schema:
        expected = schema["type"]
        types = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, item) for item in types):
            fail(f"must have type {expected}")
    if "const" in schema and not _json_equal(value, schema["const"]):
        fail(f"must equal {schema['const']!r}")
    if "enum" in schema and not any(_json_equal(value, choice) for choice in schema["enum"]):
        fail(f"must be one of {schema['enum']}")
    for keyword, count in (("oneOf", 1), ("anyOf", None)):
        if keyword in schema:
            matches = sum(valid(candidate) for candidate in schema[keyword])
            if (count is not None and matches != count) or (count is None and not matches):
                fail(f"does not match {keyword} declaration branches")
    for candidate in schema.get("allOf", ()):
        _validate_structure(value, candidate, root, path)
    if "not" in schema and valid(schema["not"]):
        fail("contains fields reserved for another declaration branch")
    if "if" in schema:
        branch = "then" if valid(schema["if"]) else "else"
        if branch in schema:
            _validate_structure(value, schema[branch], root, path)
    if isinstance(value, dict):
        missing = [key for key in schema.get("required", ()) if key not in value]
        if missing:
            fail(f"missing required fields: {missing}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _validate_structure(item, properties[key], root, f"{path}.{key}")
            elif additional is False:
                fail(f"unknown field {key!r}")
            elif isinstance(additional, Mapping):
                _validate_structure(item, additional, root, f"{path}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            fail(f"requires at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            fail(f"allows at most {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            if any(
                _json_equal(item, previous)
                for index, item in enumerate(value)
                for previous in value[:index]
            ):
                fail("items must be distinct")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_structure(item, schema["items"], root, f"{path}[{index}]")
    if isinstance(value, str) and len(value.strip()) < schema.get("minLength", 0):
        fail("must be a non-empty string")
    if type(value) in (int, float):
        for keyword, compare in (
            ("minimum", lambda a, b: a >= b),
            ("maximum", lambda a, b: a <= b),
            ("exclusiveMinimum", lambda a, b: a > b),
        ):
            if keyword in schema and not compare(value, schema[keyword]):
                fail(f"violates {keyword} {schema[keyword]}")


def _validate_time(
    samples: list[Any], timestamps: list[Any] | None, sample_rate: float | None, path: str
) -> None:
    if timestamps is None:
        if sample_rate is None:
            raise ExperimentValidationError(
                f"{path}: sampling rate or timestamps in seconds required"
            )
        return
    if len(timestamps) != len(samples):
        raise ExperimentValidationError(f"{path}: timestamps must match the samples length")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ExperimentValidationError(f"{path}: timestamps must be strictly increasing seconds")
    if sample_rate is not None and any(
        not math.isclose(b - a, 1 / sample_rate, rel_tol=1e-6, abs_tol=1e-9)
        for a, b in zip(timestamps, timestamps[1:])
    ):
        raise ExperimentValidationError(
            f"{path}: timestamps disagree with the declared sampling rate"
        )


def _validate_vertical(raw: Mapping[str, Any]) -> DataMode:
    dataset, task = raw["dataset"], raw["task"]
    unit = dataset["unit"]
    if task["target"] != dataset["channel"]:
        raise ExperimentValidationError("task.target must equal the declared dataset.channel")
    if task["threshold_unit"] != unit or raw["policy"]["threshold_unit"] != unit:
        raise ExperimentValidationError("task and policy threshold units must match dataset.unit")
    if task["metric"] not in raw["metrics"]:
        raise ExperimentValidationError("task.metric must be included in metrics")
    policy = raw["policy"]
    if not any(weight > 0 for weight in policy["support_direction"]):
        raise ExperimentValidationError("policy.support_direction requires a positive weight")
    resource = raw["resource"]
    if not resource["v_min"] <= resource["initial_voltage_v"] <= resource["v_max"]:
        raise ExperimentValidationError("resource initial voltage must lie within [v_min, v_max]")
    if resource["model"] == "capacitor_p0" and (
        resource["esr_ohm"] != 0 or resource["leakage_siemens"] != 0
    ):
        raise ExperimentValidationError("capacitor_p0 requires zero ESR and zero leakage")
    if dataset["origin"] == "REPLAY":
        _validate_time(
            dataset["samples"], dataset["timestamps_s"], dataset.get("sample_rate_hz"), "dataset"
        )
        if raw["split"]["strategy"] != "replay_fixed":
            raise ExperimentValidationError("REPLAY requires split.strategy replay_fixed")
        if any(item["type"] == "physiological_action" for item in raw["perturbations"]):
            raise ExperimentValidationError("physiological actions require a SYNTHETIC source")
        try:
            source_evidence_class(dataset)
        except ValueError as error:
            raise ExperimentValidationError(str(error)) from error
        duration = dataset["timestamps_s"][-1] - dataset["timestamps_s"][0]
        mode: DataMode = "replay"
    else:
        from physioworld.physiology.models import VARIABLE_UNITS

        duration = dataset["duration_s"]
        rate = dataset["sample_rate_hz"]
        interval = dataset["sample_interval_s"]
        if not math.isclose(interval, 1 / rate, rel_tol=1e-9, abs_tol=1e-12):
            raise ExperimentValidationError(
                "dataset.sample_interval_s must equal the reciprocal of sample_rate_hz"
            )
        count = duration * rate
        if (
            not math.isfinite(count)
            or round(count) < 1
            or not math.isclose(count, round(count), abs_tol=1e-9)
        ):
            raise ExperimentValidationError(
                "synthetic duration_s * sample_rate_hz must be a positive integer"
            )
        if dataset["modality"] == "eeg" and rate < 64:
            raise ExperimentValidationError("synthetic EEG requires sample_rate_hz >= 64")
        expected_unit = (
            "uV"
            if dataset["modality"] == "eeg" and dataset["channel"] == "EEG"
            else VARIABLE_UNITS.get(dataset["channel"])
            if dataset["modality"] == "telemetry"
            else None
        )
        if expected_unit is None or unit != expected_unit:
            raise ExperimentValidationError(
                "synthetic channel and unit must match the declared model"
            )
        if raw["split"]["strategy"] != "synthetic_fixed":
            raise ExperimentValidationError("SYNTHETIC requires split.strategy synthetic_fixed")
        for item in raw["perturbations"]:
            if item["type"] == "physiological_action" and item["time_s"] > duration:
                raise ExperimentValidationError(
                    "physiological action scheduled beyond dataset.duration_s"
                )
        mode = "synthetic"
    if task["horizon_s"] > duration:
        raise ExperimentValidationError("task.horizon_s exceeds the declared data duration")
    return mode


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    configuration: Mapping[str, Any]
    configuration_hash: str
    data_mode: DataMode
    configuration_kind: ConfigurationKind = "dataset_readiness"
    schema_name: str = "experiment.schema.json"

    @property
    def readiness_status(self) -> str:
        return (
            "CONFIGURATION_ONLY"
            if self.configuration_kind == "dataset_readiness"
            else "READY_FOR_EXECUTION"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "configuration": dict(self.configuration),
            "configuration_hash": self.configuration_hash,
            "data_mode": self.data_mode,
            "configuration_kind": self.configuration_kind,
            "schema_name": self.schema_name,
            "readiness_status": self.readiness_status,
            "execution_status": "NOT_EXECUTED",
        }


class ExperimentCompiler:
    required_fields = (
        "dataset",
        "task",
        "representation",
        "perturbations",
        "channel",
        "resource",
        "compute",
        "policy",
        "seed",
        "split",
        "metrics",
        "outputs",
    )

    def compile(
        self, configuration: Mapping[str, Any], *, experiment_id: str | None = None
    ) -> ExperimentPlan:
        if not isinstance(configuration, Mapping):
            raise ExperimentValidationError("configuration root must be an object")
        if "experiment" in configuration:
            unknown = set(configuration) - {"experiment", "$schema"}
            if unknown:
                raise ExperimentValidationError(
                    f"experiment wrapper contains unknown fields: {unknown}"
                )
            declaration = configuration.get("$schema")
            if declaration is not None and (
                not isinstance(declaration, str)
                or declaration.rsplit("/", 1)[-1] != "experiment.schema.json"
            ):
                raise ExperimentValidationError(
                    "experiment wrapper requires experiment.schema.json"
                )
        raw = configuration.get("experiment", configuration)
        if not isinstance(raw, Mapping):
            raise ExperimentValidationError("experiment root must be an object")
        try:
            canonical = json.loads(json.dumps(_json_value(raw), sort_keys=True, allow_nan=False))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ExperimentValidationError(
                f"configuration must contain finite JSON values: {exc}"
            ) from exc
        schema_name = configuration_schema_name(canonical)
        declaration = canonical.get("$schema")
        if declaration is not None:
            if not isinstance(declaration, str) or declaration.rsplit("/", 1)[-1] not in {
                schema_name,
                "experiment.schema.json",
            }:
                raise ExperimentValidationError(
                    "$schema does not identify the reviewed declaration branch"
                )
        schema = load_schema(schema_name)
        if schema_name == "world.schema.json":
            # The historical smoke simulator always computes these two summaries.
            canonical.setdefault("metrics", ["mean", "rms"])
            kind: ConfigurationKind = "synthetic_samples"
            mode: DataMode = "synthetic"
        elif schema_name == "vertical.schema.json":
            kind = "vertical"
            mode = "synthetic"  # Replaced after origin-specific validation below.
        else:
            kind = "dataset_readiness"
            mode = "dataset"
        _validate_structure(canonical, schema, schema)
        if kind == "synthetic_samples":
            _validate_time(
                canonical["samples"],
                canonical.get("timestamps"),
                canonical.get("sample_rate"),
                "experiment",
            )
        elif kind == "vertical":
            mode = _validate_vertical(canonical)
        else:
            task = canonical["task"]
            if "horizon" in task and "horizon_s" in task and task["horizon"] != task["horizon_s"]:
                raise ExperimentValidationError("task horizon and horizon_s disagree")
            if not canonical["split"]:
                raise ExperimentValidationError("split must be explicit")
        identifier = experiment_id or canonical.get("name") or "experiment"
        if not isinstance(identifier, str) or not identifier.strip():
            raise ExperimentValidationError("experiment_id must be a non-empty string")
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        digest = hashlib.sha256(encoded).hexdigest()
        return ExperimentPlan(identifier, canonical, digest, mode, kind, schema_name)

    def compile_file(self, path: str | Path) -> ExperimentPlan:
        path = Path(path)
        return self.compile(load_config(path), experiment_id=path.stem)

"""Frozen clinical-validation protocol and readiness reporting."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

_PLACEHOLDERS = frozenset(
    {
        "",
        "none",
        "null",
        "unspecified",
        "unknown",
        "n/a",
        "na",
        "tbd",
        "todo",
        "placeholder",
        "default",
        "not specified",
        "not provided",
        "to be determined",
        "to be defined",
        "{}",
        "[]",
        "nan",
        "inf",
        "infinity",
        "research evaluation",
        "declared dataset population",
        "dataset protocol",
        "declared dataset labels",
        "declared task target",
        "all eligible subjects",
        "missingness reported; no implicit imputation",
        "declared model",
        "declared before evaluation",
        "declared resource",
        "declared channel",
        "provider version recorded by adapter",
    }
)


def _require_explicit_text(value: Any, name: str) -> None:
    """Reject absent text and known placeholders without attesting to its contents."""
    if not isinstance(value, str):
        raise ValueError(f"{name} requires explicit non-placeholder text")
    normalized = " ".join(value.strip().strip("\"'").lower().split())
    if (
        normalized in _PLACEHOLDERS
        or normalized.startswith("default ")
        or normalized.startswith("placeholder ")
        or re.fullmatch(r"<[^>]+>", normalized)
    ):
        raise ValueError(f"{name} requires explicit non-placeholder text")


def _declaration_value(value: Any, name: str) -> Any:
    """Validate structured declarations before canonical JSON serialization."""
    if isinstance(value, str):
        _require_explicit_text(value, name)
        return value
    if isinstance(value, Mapping):
        if not value or any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError(f"{name} requires a non-empty declaration with string keys")
        return {key: _declaration_value(item, f"{name}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} requires a non-empty declaration")
        return [_declaration_value(item, f"{name}[{i}]") for i, item in enumerate(value)]
    if type(value) is bool or type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"{name} requires explicit finite values, not null or placeholders")


def _freeze_declaration(value: Any, name: str) -> str:
    if not isinstance(value, (str, Mapping)):
        raise ValueError(f"{name} requires explicit text or a non-empty object")
    validated = _declaration_value(value, name)
    if isinstance(validated, str):
        return validated
    return json.dumps(validated, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class ClinicalValidationProtocol:
    intended_use: str
    population: str
    data_provenance: str
    inclusion_exclusion: str
    reference_standard: str
    endpoint: str
    analysis_population: str
    patient_split: str
    missing_data: str
    model: str
    threshold: str
    resource_regime: str
    channel_regime: str
    dataset_version: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            _require_explicit_text(value, name)
            if name in ("patient_split", "model", "resource_regime", "channel_regime"):
                if value.lstrip().startswith(("{", "[")):
                    try:
                        declaration = json.loads(value)
                    except ValueError as exc:
                        raise ValueError(f"{name} contains an invalid JSON declaration") from exc
                    _declaration_value(declaration, name)
        if not re.search(r"(?<![a-z])(test|temporal)(?![a-z])", self.patient_split.lower()):
            raise ValueError(
                "patient_split must explicitly define a held-out test or temporal evaluation"
            )

    @property
    def frozen_hash(self) -> str:
        self.validate()
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "frozen_hash": self.frozen_hash}


def protocol_from_config(config: Mapping[str, Any]) -> ClinicalValidationProtocol:
    """Freeze only caller-declared clinical fields; experiment defaults are insufficient."""
    if not isinstance(config, Mapping):
        raise ValueError("clinical protocol configuration must be an object")
    sources = {
        "intended_use": "intended_use",
        "population": "population",
        "data_provenance": "data_provenance",
        "inclusion_exclusion": "inclusion_exclusion",
        "reference_standard": "reference_standard",
        "endpoint": "endpoint",
        "analysis_population": "analysis_population",
        "patient_split": "split",
        "missing_data": "missing_data",
        "model": "model",
        "threshold": "threshold",
        "resource_regime": "resource",
        "channel_regime": "channel",
        "dataset_version": "dataset_version",
    }
    missing = [source for source in sources.values() if source not in config]
    if missing:
        raise ValueError(f"clinical validation protocol requires explicit fields: {missing}")
    fields: dict[str, str] = {}
    for name, source in sources.items():
        value = config[source]
        if name in ("patient_split", "model", "resource_regime", "channel_regime"):
            fields[name] = _freeze_declaration(value, source)
        elif name == "threshold" and type(value) in (int, float):
            fields[name] = json.dumps(_declaration_value(value, source), allow_nan=False)
        else:
            _require_explicit_text(value, source)
            fields[name] = value
    protocol = ClinicalValidationProtocol(**fields)
    protocol.validate()
    return protocol


def validation_readiness(data_mode: str) -> dict[str, str]:
    """Classify configuration readiness without inferring evaluation or validation."""
    modes = {
        "synthetic": ("READY_FOR_SIMULATION", "AVAILABLE_SYNTHETIC", "SIMULATION"),
        "replay": ("READY_FOR_REPLAY", "AVAILABLE_REPLAY", "REPLAY"),
        "dataset": (
            "CONFIGURATION_ONLY",
            "REQUIRES_USER_DATASET_ACCESS",
            "EXTERNAL_PREDICTIVE_EVALUATION",
        ),
    }
    if data_mode not in modes:
        raise ValueError(f"unsupported validation data mode: {data_mode!r}")
    status, data_status, scope = modes[data_mode]
    return {
        "status": status,
        "data_status": data_status,
        "evaluation_scope": scope,
        "external_predictive_evaluation_status": "NOT_EVALUATED",
        "clinical_validation_status": "NOT_ESTABLISHED",
    }

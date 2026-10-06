"""JSON-first configuration loader with optional YAML support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            # JSON is a valid YAML 1.2 subset and keeps minimal installations
            # able to consume machine-generated YAML-extension configs.
            result = json.loads(text)
        except json.JSONDecodeError:
            result = None
        if result is not None:
            if not isinstance(result, dict):
                raise ValueError("configuration root must be an object")
            return result
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML configuration requires the 'yaml' optional dependency"
            ) from exc
        result = yaml.safe_load(text)
    else:
        result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("configuration root must be an object")
    return result


def save_json(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )

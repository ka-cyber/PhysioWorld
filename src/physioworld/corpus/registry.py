"""Machine-readable source and experiment lineage for the Level 3 corpus.

The registry is deliberately metadata-only: an extracted reference is not a
dataset, and an unavailable source cannot become active evidence merely by
appearing in a manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping


@dataclass(frozen=True)
class CorpusSource:
    source_id: str
    title: str
    artifact: str
    role: str
    authority: str
    evidence_class: str
    access_status: str
    supersedes: tuple[str, ...]
    mappings: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.source_id,
            "title": self.title,
            "artifact": self.artifact,
            "role": self.role,
            "authority": self.authority,
            "evidence_class": self.evidence_class,
            "access_status": self.access_status,
            "supersedes": list(self.supersedes),
            "maps_to": {key: list(value) for key, value in self.mappings.items()},
        }


def load_registry() -> dict[str, Any]:
    """Load and validate the bundled registry as finite JSON."""
    payload = files("physioworld.corpus").joinpath("registry.json").read_text(encoding="utf-8")
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("registry_version") != "1.0.0":
        raise ValueError("bundled corpus registry has an unsupported version")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("bundled corpus registry must contain sources")
    identifiers = [item.get("id") for item in sources if isinstance(item, dict)]
    if len(identifiers) != len(set(identifiers)) or any(not item for item in identifiers):
        raise ValueError("bundled corpus registry source IDs must be unique")
    json.dumps(value, allow_nan=False)
    return value


def source_registry() -> tuple[CorpusSource, ...]:
    """Return typed source records with stable IDs and tuple-valued mappings."""
    records = []
    for item in load_registry()["sources"]:
        mappings = item.get("maps_to", {})
        records.append(
            CorpusSource(
                item["id"],
                item["title"],
                item["artifact"],
                item["role"],
                item["authority"],
                item["evidence_class"],
                item["access_status"],
                tuple(item.get("supersedes", ())),
                {key: tuple(value) for key, value in mappings.items()},
            )
        )
    return tuple(records)


def source_by_id(source_id: str) -> CorpusSource:
    for source in source_registry():
        if source.source_id == source_id:
            return source
    raise KeyError(source_id)


__all__ = ["CorpusSource", "load_registry", "source_registry", "source_by_id"]

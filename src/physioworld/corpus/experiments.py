"""Typed access to the executable Level 3 experiment registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    title: str
    source_ids: tuple[str, ...]
    estimand: str
    unit_of_analysis: str
    seed: int
    configuration: str
    output_schema: str
    evidence_class: str
    expected_diagnostic: str
    falsification_conditions: tuple[str, ...]

    @property
    def is_external(self) -> bool:
        return self.configuration.startswith("external.required") or self.configuration.startswith(
            "historical.required"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.experiment_id,
            "title": self.title,
            "source_ids": list(self.source_ids),
            "estimand": self.estimand,
            "unit_of_analysis": self.unit_of_analysis,
            "seed": self.seed,
            "configuration": self.configuration,
            "output_schema": self.output_schema,
            "evidence_class": self.evidence_class,
            "expected_diagnostic": self.expected_diagnostic,
            "falsification_conditions": list(self.falsification_conditions),
            "availability": "EXTERNAL_INPUT_REQUIRED" if self.is_external else "EXECUTABLE",
        }


def load_experiment_registry() -> tuple[ExperimentSpec, ...]:
    payload = files("physioworld.corpus").joinpath("experiments.json").read_text(encoding="utf-8")
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get("registry_version") != "1.0.0":
        raise ValueError("unsupported Level 3 experiment registry version")
    raw = document.get("experiments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Level 3 experiment registry is empty")
    result = tuple(
        ExperimentSpec(
            experiment_id=item["id"],
            title=item["title"],
            source_ids=tuple(item["source_ids"]),
            estimand=item["estimand"],
            unit_of_analysis=item["unit_of_analysis"],
            seed=item["seed"],
            configuration=item["configuration"],
            output_schema=item["output_schema"],
            evidence_class=item["evidence_class"],
            expected_diagnostic=item["expected_diagnostic"],
            falsification_conditions=tuple(item["falsification_conditions"]),
        )
        for item in raw
    )
    identifiers = [item.experiment_id for item in result]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Level 3 experiment IDs must be unique")
    if any(
        item.seed < 0 or not item.source_ids or not item.falsification_conditions for item in result
    ):
        raise ValueError(
            "every experiment needs a nonnegative seed, source and falsification condition"
        )
    return result


def experiment_by_id(experiment_id: str) -> ExperimentSpec:
    for experiment in load_experiment_registry():
        if experiment.experiment_id == experiment_id:
            return experiment
    raise KeyError(experiment_id)


__all__ = ["ExperimentSpec", "load_experiment_registry", "experiment_by_id"]

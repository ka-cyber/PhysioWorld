"""Status-only flagship reproduction registry.

Reference numbers from external analyses are never presented as produced
results. A reproduction becomes REPRODUCED only after its data and estimator
artifacts are supplied to the runner.
"""

from __future__ import annotations

from typing import Any

REFERENCE_IDS = (
    "flagship_temporal",
    "flagship_representation",
    "flagship_communication",
    "flagship_resource",
)


def flagship_status(available_inputs: dict[str, bool] | None = None) -> tuple[dict[str, Any], ...]:
    available_inputs = available_inputs or {}
    return tuple(
        {
            "analysis": analysis,
            "status": "REPRODUCED"
            if available_inputs.get(analysis, False)
            else "NOT REPRODUCIBLE FROM AVAILABLE INPUTS",
            "reference_inputs_required": True,
            "claim": "No numerical result is asserted by the status registry.",
        }
        for analysis in REFERENCE_IDS
    )

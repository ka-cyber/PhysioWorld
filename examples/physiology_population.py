"""Run an assumed synthetic adult population and a common-random-number fork.

Run from an installed checkout: ``python examples/physiology_population.py``.
No participant data or clinical validity are claimed; output is model simulation.
"""

import json

from physioworld.physiology import (
    DataRequest,
    PhysiologicalAction,
    PhysiologicalScenario,
    run_population,
    sample_population,
)


def main() -> None:
    scenario = PhysiologicalScenario(
        name="synthetic_orthostatic_population",
        duration_s=45,
        seed=23,
        requests=DataRequest(("heart_rate_bpm", "mean_pressure_mmhg", "cardiac_output_l_min")),
        actions=(PhysiologicalAction(10, "orthostasis", {"orthostasis": 1}),),
    )
    summaries = []
    for result in run_population(scenario, sample_population(4, seed=23)):
        summaries.append(
            {
                "subject_id": result.manifest["patient"]["subject_id"],
                "endpoint": result.observations[-1].to_dict(),
                "conservation_residuals": result.manifest["conservation_residuals"],
            }
        )
    engine = scenario.create_engine()
    engine.advance(15)
    treated = engine.fork("saline", (PhysiologicalAction(15, "bolus", {"volume_ml": 250}),))
    engine.advance_to(45)
    treated.advance_to(45)
    print(
        json.dumps(
            {
                "evidence_class": "SIMULATION",
                "population": summaries,
                "model_counterfactual": {
                    "untreated": engine.observe().to_dict(),
                    "saline": treated.observe().to_dict(),
                },
                "fork_lineage": treated.manifest()["lineage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

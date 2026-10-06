"""Run: python examples/resource_design.py (standard-library-only design study).

All designs/corners below are explicitly synthetic; unavailable historical
fixtures are reported without reverse-fitting their quoted results.
"""

import json
from dataclasses import asdict

from physioworld.resources import (
    CapacitorResource,
    DesignEvaluation,
    canonical_fixture_status,
    corner_analysis,
    design_search,
    evaluate_harvest_schedule,
    optimize_robust_policy,
    phase_diagram,
    resource_frontier,
    shadow_value,
    theorem_registry,
)


def main():
    def evaluate(resources):
        candidates = [("sleep", 0, 0, 0), ("local_summary", 0.5, 1, 0.7), ("raw_edge", 2, 8, 1)]
        feasible = [
            p for p in candidates if p[1] <= resources["energy"] and p[2] <= resources["bandwidth"]
        ]
        policy = max(feasible, key=lambda p: p[3])
        return DesignEvaluation((policy[3],), policy[0])

    diagram = phase_diagram({"energy": (0, 0.5, 1, 2), "bandwidth": (0, 1, 4, 8)}, evaluate, (0.7,))
    corners = corner_analysis(
        {"capacitance": (0.18, 0.22), "esr": (7.2, 8.8), "leakage": (0.0005, 0.001)},
        lambda theta: CapacitorResource(
            theta["capacitance"], theta["esr"], theta["leakage"], v_min=1.85
        ).constant_current_limit(2.65, 10),
    )
    robust = optimize_robust_policy(
        {"fast": {"low": 1, "high": 0.1}, "economy": {"low": 0.7, "high": 0.7}}
    )
    physical_candidates = tuple((voltage, esr) for voltage in (2.5, 2.65, 2.8) for esr in (6, 8))

    def physical(candidate):
        voltage, esr = candidate
        resource = CapacitorResource(0.2, esr, 0.001, v_min=1.85, safety_margin_v=0.025)
        current = resource.constant_current_limit(voltage, 10)
        return DesignEvaluation((current,), "constant_current_10s", current >= 0.01)

    best, _ = design_search(
        physical_candidates, physical, lambda candidate, result: -0.5 * 0.2 * candidate[0] ** 2
    )
    result = {
        "evidence": "MODEL_DERIVED",
        "parameter_origin": "explicit synthetic design family",
        "phase_diagram": asdict(diagram),
        "frontier": [asdict(p) for p in resource_frontier(diagram)],
        "corners": asdict(corners),
        "worst_current_a": corners.worst_value,
        "robust_policy": asdict(robust),
        "physical_design": {
            "initial_voltage_v": best[0][0],
            "esr_ohm": best[0][1],
            "ready_energy_j": -best[2],
            "deliverable_current_a": best[1].capability[0],
            "scope": "minimum energy over this explicit synthetic finite candidate set",
        },
        "shadow": asdict(
            shadow_value(
                lambda r: evaluate(r).capability[0], {"energy": 0.5, "bandwidth": 1}, "energy", 0.1
            )
        ),
        "harvest_timing": {
            "early": asdict(evaluate_harvest_schedule((1, 0), (1, 0), (1, 0))),
            "late": asdict(evaluate_harvest_schedule((0, 1), (1, 0), (1, 0))),
        },
        "canonical_fixtures": canonical_fixture_status(),
        "proposition_ids": list(theorem_registry()),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

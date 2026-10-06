"""Finite model comparisons/reachability ensembles with explicitly limited scope."""

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .engine import PhysiologicalAction
from .information import DataRequest
from .scenario import PhysiologicalScenario


def sampled_reachability(
    scenario: PhysiologicalScenario,
    action_branches: Mapping[str, Sequence[PhysiologicalAction]],
    *,
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, Any]:
    """Enumerate trajectories and safe-set membership at scenario sample times.

    This finite sampled set is neither an outer bound on all reachable states
    nor a viability kernel; between-sample threshold crossings are not excluded.
    """
    outcomes, extrema = {}, {n: [float("inf"), -float("inf")] for n in bounds}
    for name, actions in action_branches.items():
        result = replace(
            scenario,
            actions=tuple((*scenario.actions, *actions)),
            requests=DataRequest(tuple(bounds)),
        ).run()
        safe = True
        for observation in result.observations:
            for n, (low, high) in bounds.items():
                value = observation.scalar(n)
                safe = safe and low <= value <= high
                extrema[n][0], extrema[n][1] = min(extrema[n][0], value), max(extrema[n][1], value)
        outcomes[name] = {"sampled_safe": safe, "endpoint": dict(result.observations[-1].values)}
    if not outcomes:
        raise ValueError("at least one action branch required")
    return {
        "evidence_class": "SIMULATION",
        "causal_status": "MODEL_COUNTERFACTUAL",
        "method": "FINITE_SAMPLED_REACHABILITY",
        "bounds": dict(bounds),
        "sampled_extrema": extrema,
        "branches": outcomes,
        "is_viability_kernel": False,
        "continuous_time_safety_proven": False,
    }


def compare_fidelities(
    scenario: PhysiologicalScenario,
    *,
    variables: Sequence[str],
    atol: Mapping[str, float],
) -> dict[str, Any]:
    """Declared trajectory max-error comparison of open_loop to coupled model."""
    if not variables or any(n not in atol or atol[n] <= 0 for n in variables):
        raise ValueError("declare a positive native-unit error tolerance per variable")
    common = replace(scenario, requests=DataRequest(tuple(variables)), observation_noise={})
    reference = replace(common, fidelity="coupled").run()
    reduced = replace(common, fidelity="open_loop").run()
    errors = {
        n: max(
            abs(a.scalar(n) - b.scalar(n))
            for a, b in zip(reference.observations, reduced.observations)
        )
        for n in variables
    }
    return {
        "evidence_class": "NUMERICAL_VERIFICATION",
        "reference": "coupled",
        "reduced": "open_loop",
        "error_criterion": "max_absolute_requested_trajectory_error",
        "max_errors": errors,
        "tolerances": dict(atol),
        "criterion_met": all(errors[n] <= atol[n] for n in variables),
        "scope": "declared scenario only; no full-order human equivalence",
    }

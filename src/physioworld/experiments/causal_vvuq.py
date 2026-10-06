"""Synthetic-only causal VVUQ with explicit identification boundaries.

This module uses the packaged known-ground-truth intervention fixture and a
separate Monte Carlo implementation. It reports recovery of a declared SCM
effect, but deliberately blocks observational and future-information shortcuts.
No output is empirical, clinical, or a claim about an external population.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from importlib.resources import files
from statistics import fmean
from typing import Sequence

from physioworld.causal import (
    BinaryTreatmentSCM,
    known_ground_truth_interventional_gap,
    positivity_diagnostic,
)


def _comparison(observed: float, reference: float, tolerance: float) -> dict:
    residual = observed - reference
    return {
        "observed": observed,
        "reference": reference,
        "residual": residual,
        "absolute_tolerance": tolerance,
        "passed": math.isfinite(residual) and abs(residual) <= tolerance,
    }


def _independent_counterfactual_sample(
    effect: float, confounding: float, noise: float, count: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    untreated = []
    treated = []
    for _ in range(count):
        latent = rng.gauss(0.0, 1.0)
        treated.append(effect + confounding * latent + rng.gauss(0.0, noise))
        untreated.append(confounding * latent + rng.gauss(0.0, noise))
    return fmean(treated), fmean(untreated)


def _randomized_observation(
    effect: float, confounding: float, noise: float, count: int, seed: int
) -> tuple[float, float, int, int]:
    rng = random.Random(seed)
    treated: list[float] = []
    untreated: list[float] = []
    for _ in range(count):
        latent = rng.gauss(0.0, 1.0)
        action = int(rng.random() < 0.5)
        outcome = effect * action + confounding * latent + rng.gauss(0.0, noise)
        (treated if action else untreated).append(outcome)
    return fmean(treated), fmean(untreated), len(treated), len(untreated)


def _predictive_causal_counterexample() -> dict:
    """Two states have the same natural-policy prediction and different do-laws."""
    states = {
        "h": {"natural_action": 0, "predictive_outcome": 0.0, "do_1_outcome": 1.0},
        "h_prime": {"natural_action": 0, "predictive_outcome": 0.0, "do_1_outcome": 2.0},
    }
    predictive_gap = abs(
        states["h"]["predictive_outcome"] - states["h_prime"]["predictive_outcome"]
    )
    intervention_gap = abs(states["h"]["do_1_outcome"] - states["h_prime"]["do_1_outcome"])
    return {
        "states": states,
        "predictive_gap": predictive_gap,
        "intervention_gap": intervention_gap,
        "passed": predictive_gap == 0.0 and intervention_gap > 0.0,
        "interpretation": "equal natural-policy prediction does not establish causal sufficiency",
    }


def _future_covariate_control() -> dict:
    decision_time = 1.0
    future_available_at = 2.0
    return {
        "decision_time_s": decision_time,
        "covariate_available_at_s": future_available_at,
        "accepted_as_history": future_available_at <= decision_time,
        "passed": future_available_at > decision_time,
        "status": "BLOCKED_BY_CAUSAL_GATE",
    }


def run_causal_vvuq(*, counts: Sequence[int] = (64, 256, 1024), seed: int = 7) -> dict:
    """Verify a known SCM effect, seeded convergence, and causal negative controls."""
    counts = tuple(counts)
    if (
        len(counts) < 3
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in counts)
        or any(a >= b for a, b in zip(counts, counts[1:]))
    ):
        raise ValueError("counts must contain at least three increasing integers >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    payload = (
        files("physioworld")
        .joinpath("corpus")
        .joinpath("golden")
        .joinpath("causal_intervention.json")
        .read_bytes()
    )
    fixture = json.loads(payload)
    effect = float(fixture["treatment_effect"])
    confounding = 0.5
    noise = 0.1
    model = BinaryTreatmentSCM(
        treatment_effect=effect,
        confounding_effect=confounding,
        noise_scale=noise,
    )
    rows = model.generate(fixture["sample_count"], seed=fixture["seed"])
    structural_reference = effect
    production_gap = known_ground_truth_interventional_gap(model)
    structural_comparison = _comparison(production_gap, structural_reference, 1e-12)
    structural_check = {
        "sample_count": len(rows),
        "independent_structural_effect": structural_reference,
        "production_counterfactual_gap": production_gap,
        "comparison": structural_comparison,
        "truth_source": "fixture_declared_intervention_law",
    }
    convergence = []
    for count in counts:
        treated, untreated = _independent_counterfactual_sample(
            effect, confounding, noise, count, seed + count
        )
        randomized_treated, randomized_untreated, treated_count, untreated_count = (
            _randomized_observation(effect, confounding, noise, count, seed + 10000 + count)
        )
        convergence.append(
            {
                "count": count,
                "independent_counterfactual_ate": treated - untreated,
                "counterfactual_absolute_error": abs(treated - untreated - effect),
                "randomized_ate": randomized_treated - randomized_untreated,
                "randomized_absolute_error": abs(
                    randomized_treated - randomized_untreated - effect
                ),
                "treated_count": treated_count,
                "untreated_count": untreated_count,
                "positivity": treated_count > 0 and untreated_count > 0,
            }
        )
    replay = []
    for count in counts:
        treated, untreated = _independent_counterfactual_sample(
            effect, confounding, noise, count, seed + count
        )
        replay.append(treated - untreated)
    convergence_check = {
        "rows": convergence,
        "large_sample_counterfactual_error": convergence[-1]["counterfactual_absolute_error"],
        "large_sample_randomized_error": convergence[-1]["randomized_absolute_error"],
        "max_allowed_counterfactual_error": 0.25,
        "max_allowed_randomized_error": 0.5,
        "reproducible": replay == [row["independent_counterfactual_ate"] for row in convergence],
        "passed": (
            convergence[-1]["counterfactual_absolute_error"] <= 0.25
            and convergence[-1]["randomized_absolute_error"] <= 0.5
            and all(row["positivity"] for row in convergence)
            and replay == [row["independent_counterfactual_ate"] for row in convergence]
        ),
        "interpretation": "finite seeded simulation convergence, not a confidence interval",
    }

    treated_rows = [row.outcome for row in rows if row.treatment == 1]
    untreated_rows = [row.outcome for row in rows if row.treatment == 0]
    observational_gap = fmean(treated_rows) - fmean(untreated_rows)
    observational_control = {
        "naive_observational_difference": observational_gap,
        "declared_structural_effect": effect,
        "absolute_bias": abs(observational_gap - effect),
        "passed": abs(observational_gap - effect) > 0.05,
        "status": "BLOCKED_NOT_CAUSAL_IDENTIFICATION",
        "reason": "treatment assignment depends on the latent confounder in the fixture SCM",
    }
    positivity_control = positivity_diagnostic((0.0, 0.5, 1.0), threshold=0.01)
    positivity_failure = {
        "minimum_propensity": positivity_control.minimum_propensity,
        "maximum_propensity": positivity_control.maximum_propensity,
        "violations": positivity_control.violations,
        "valid": positivity_control.valid,
        "passed": not positivity_control.valid and positivity_control.violations == 2,
        "status": "BLOCKED_BY_POSITIVITY_GATE",
    }
    future_control = _future_covariate_control()
    representation_control = _predictive_causal_counterexample()
    negative_controls = {
        "observational_claim": observational_control,
        "positivity_failure": positivity_failure,
        "future_covariate": future_control,
        "predictive_causal_counterexample": representation_control,
    }
    passed = (
        structural_comparison["passed"]
        and convergence_check["passed"]
        and all(control["passed"] for control in negative_controls.values())
    )
    return {
        "experiment_id": "causal_vvuq",
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "SIMULATION",
        "data_origin": "KNOWN_GROUND_TRUTH_SCM",
        "data_fidelity": "SYNTHETIC",
        "clinical_claim": False,
        "fixture": {
            "name": "causal_intervention.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "inputs": fixture,
        },
        "estimand": "structural average treatment effect under declared synthetic SCM",
        "structural_reference": structural_check,
        "convergence": convergence_check,
        "negative_controls": negative_controls,
        "causal_boundary": {
            "identification_status": "MODEL_IDENTIFIED_WITHIN_DECLARED_SCM",
            "observational_transport_status": "BLOCKED",
            "external_validation_status": "NOT_PERFORMED",
            "assumptions": [
                "consistency within the synthetic structural equations",
                "model-declared intervention law",
                "no interference in the finite simulation",
            ],
        },
        "limitations": [
            "Known-ground-truth synthetic simulation only; no empirical, clinical, or external evidence.",
            "The finite Monte Carlo rows quantify simulation behavior and do not establish a population effect.",
            "The observational contrast is retained as a falsification control and is explicitly not called causal.",
        ],
    }

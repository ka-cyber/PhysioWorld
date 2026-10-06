"""Focused known-ground-truth synthetic causal VVUQ checks."""

from physioworld.experiments.causal_vvuq import run_causal_vvuq


def test_causal_vvuq_recovers_declared_effect_with_independent_simulation():
    report = run_causal_vvuq(counts=(64, 256, 1024), seed=7)

    assert report["status"] == "PASS"
    assert report["evidence_class"] == "SIMULATION"
    assert report["clinical_claim"] is False
    assert report["structural_reference"]["comparison"]["passed"] is True
    assert report["convergence"]["passed"] is True
    assert report["convergence"]["reproducible"] is True


def test_causal_vvuq_blocks_observational_and_representation_shortcuts():
    report = run_causal_vvuq(counts=(64, 256, 1024), seed=7)
    controls = report["negative_controls"]

    assert controls["observational_claim"]["status"] == "BLOCKED_NOT_CAUSAL_IDENTIFICATION"
    assert controls["observational_claim"]["passed"] is True
    assert controls["positivity_failure"]["status"] == "BLOCKED_BY_POSITIVITY_GATE"
    assert controls["positivity_failure"]["passed"] is True
    assert controls["future_covariate"]["status"] == "BLOCKED_BY_CAUSAL_GATE"
    assert controls["future_covariate"]["passed"] is True
    assert controls["predictive_causal_counterexample"]["passed"] is True

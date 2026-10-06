"""Focused finite-field and seeded transport VVUQ checks."""

from physioworld.experiments.communication_vvuq import run_communication_vvuq


def test_communication_vvuq_cross_checks_gf_and_converges_seeded_frequencies():
    report = run_communication_vvuq(seed=17, trials=16)

    assert report["status"] == "PASS"
    assert report["evidence_class"] == "NUMERICAL_VERIFICATION_AND_SIMULATION"
    assert report["clinical_claim"] is False
    assert all(
        value is True
        for value in report["independent_reference"]["checks"].values()
        if isinstance(value, bool)
    )
    assert report["stochastic_convergence"]["checks"]["reproducible"] is True
    assert report["stochastic_convergence"]["checks"]["large_sample_error"] is True


def test_communication_vvuq_semantic_permutation_preserves_transport_but_fails_task_law():
    report = run_communication_vvuq(seed=17, trials=16)
    control = report["semantic_negative_control"]["check"]

    assert control["paired_transport_difference"] == 0.0
    assert control["valid_task_law_discrepancy"] == 0.0
    assert control["permuted_task_law_discrepancy"] > control["valid_task_law_discrepancy"]
    assert control["passed"] is True

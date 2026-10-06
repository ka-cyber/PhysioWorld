"""Focused synthetic numerical verification for the resource VVUQ experiment."""

from physioworld.experiments.resource_vvuq import run_resource_vvuq


def test_resource_vvuq_checks_independent_convergence_and_energy_controls():
    report = run_resource_vvuq(steps=(16, 32, 64))

    assert report["status"] == "PASS"
    assert report["evidence_class"] == "NUMERICAL_VERIFICATION"
    assert report["clinical_claim"] is False
    assert report["convergence"]["passed"] is True
    assert report["convergence"]["observed_orders"][-1] >= 3.5
    assert report["p2"]["checks"]["independent_energy_balance"]["passed"] is True
    assert report["uncertainty"]["corner_count"] == 128
    assert report["negative_controls"]["wrong_discharge_sign"]["passed"] is True
    assert report["negative_controls"]["omitted_leakage_loss"]["passed"] is True
    assert report["negative_controls"]["equal_energy_different_feasibility"]["passed"] is True

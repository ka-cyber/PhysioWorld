"""Verification of the experimental resource-coupled physiology reduction."""

import math

import pytest

from physioworld.physiology import (
    COUPLED_MODEL_VERSION,
    CoupledEnvelopeError,
    CoupledInputs,
    CoupledIntervention,
    CoupledPhysiology,
    CoupledResources,
    resource_limited_scenario,
)


def _event(result, kind, resource):
    return next(
        event
        for event in result.manifest["events"]
        if event["kind"] == kind and event.get("resource") == resource
    )


def test_resource_limited_scenario_is_end_to_end_and_scientifically_labelled():
    result = resource_limited_scenario().run(max_step_s=0.5)
    initial = result.observations[0]
    supported = next(observation for observation in result.observations if observation.time_s == 70)
    final = result.observations[-1]
    model = result.manifest["model"]

    assert result.observations[-1].time_s == 240
    assert len(result.observations) == 25
    assert model["version"] == COUPLED_MODEL_VERSION
    assert model["maturity"] == "EXPERIMENTAL_REDUCED_ORDER"
    assert model["operating_envelope_label"] == "ASSUMED_SYNTHETIC_ADULT_SHORT_HORIZON"
    assert model["evidence_class"] == "SIMULATION"
    assert model["validation_status"] == "NOT_VALIDATED"
    assert model["calibration_status"] == "NOT_CALIBRATED"
    assert model["calibration_data"] == []
    assert model["empirical_evaluation_evidence"] == []
    assert model["state_units"]["arterial_o2_ml"] == "mL STPD"
    assert model["state_units"]["arterial_co2_ml"] == "mL STPD"
    assert set(model["state_vector_order"]) <= set(model["state_units"])
    assert "dO2_supply=-O2_draw/60; dO2_drawn=O2_draw/60" in model["equations"]
    assert set(model["event_definitions"]) == {
        "input_change",
        "resource_depleted",
        "operating_envelope_exit",
    }
    assert model["provenance"]

    # The resource state feeds back into actual delivery, rather than being an
    # observational side channel: oxygen and support are active after t=40 s.
    assert supported["oxygen_draw_ml_min"] == 4000
    assert supported["infusion_delivered_ml_min"] == 100
    assert supported["device_power_w"] == 15
    assert supported["arterial_po2_mmhg"] > initial["arterial_po2_mmhg"]
    assert final["oxygen_supply_ml"] == 0
    assert final["fluid_supply_l"] == 0
    assert final["battery_energy_j"] == 0
    assert final["device_power_w"] == 0
    assert final["ventilatory_assist_active"] == 0

    oxygen_event = _event(result, "resource_depleted", "oxygen_supply_ml")
    fluid_event = _event(result, "resource_depleted", "fluid_supply_l")
    battery_event = _event(result, "resource_depleted", "battery_energy_j")
    assert oxygen_event["time_s"] == pytest.approx(100, abs=1e-9)
    assert fluid_event["time_s"] == pytest.approx(130, abs=1e-9)
    assert battery_event["time_s"] == pytest.approx(150, abs=1e-9)
    assert result.manifest["conservation_residuals"]["oxygen_supply_ml"] == pytest.approx(0)
    assert result.manifest["conservation_residuals"]["fluid_supply_l"] == pytest.approx(0)
    assert result.manifest["conservation_residuals"]["battery_energy_j"] == pytest.approx(0)
    assert max(abs(value) for value in result.manifest["conservation_residuals"].values()) < 1e-7


def test_rhs_exposes_all_states_and_resource_rates_without_mutation():
    world = CoupledPhysiology(
        resources=CoupledResources(oxygen_supply_ml=600, fluid_supply_l=0.02, battery_energy_j=100),
        inputs=CoupledInputs(
            oxygen_flow_l_min_stpd=1,
            assisted_ventilation_l_min_btps=8,
            infusion_request_ml_min=20,
            peep_cmh2o=4,
        ),
    )
    before = world.state.vector()
    derivative = world.derivative(world.state, world.inputs)

    assert len(derivative) == len(before)
    assert all(math.isfinite(value) for value in derivative)
    assert derivative[-6:-3] == pytest.approx((-1000 / 60, -20 / 60000, -15))
    assert derivative[-3:] == pytest.approx((1000 / 60, 20 / 60000, 15))
    assert world.state.vector() == before

    world.advance_to(1)
    assert world.observe()["oxygen_supply_ml"] == pytest.approx(600 - 1 / 60 * 1000)
    assert world.observe()["fluid_supply_l"] == pytest.approx(0.02 - 20 / 60000)
    assert world.observe()["battery_energy_j"] == pytest.approx(85)


def test_exact_interventions_and_empty_resource_stop_delivery():
    world = CoupledPhysiology(resources=CoupledResources(oxygen_supply_ml=100))
    world.schedule(CoupledIntervention(2.5, {"oxygen_flow_l_min_stpd": 1}, "oxygen start"))
    assert world.advance_to(2.499)["oxygen_draw_ml_min"] == 0
    at_start = world.advance_to(2.5)
    assert at_start["oxygen_draw_ml_min"] == 1000
    after_empty = world.advance_to(8.5)
    assert after_empty["oxygen_supply_ml"] == 0
    assert after_empty["oxygen_draw_ml_min"] == 0
    event = next(event for event in world.events if event["kind"] == "resource_depleted")
    assert event["time_s"] == pytest.approx(8.5, abs=1e-9)


def test_bounds_and_recorded_operating_exit_are_explicit():
    with pytest.raises(ValueError, match="oxygen_supply_ml"):
        CoupledResources(oxygen_supply_ml=-1)
    with pytest.raises(ValueError, match="exercise_mets"):
        CoupledInputs(exercise_mets=7)
    with pytest.raises(CoupledEnvelopeError, match="blood_volume_ratio"):
        world = CoupledPhysiology(envelope_policy="fail")
        world.schedule(CoupledIntervention(0, {"hemorrhage_ml_min": 300}, "declared bleed"))
        world.advance_to(1800)

    # A recording run retains the first exit in the provenance stream rather
    # than silently changing the configured bounds.
    world = CoupledPhysiology(envelope_policy="record")
    world.schedule(CoupledIntervention(0, {"hemorrhage_ml_min": 300}, "declared bleed"))
    world.advance_to(400)
    assert world.manifest()["envelope_violations"]
    assert all("bounds" in violation for violation in world.manifest()["envelope_violations"])

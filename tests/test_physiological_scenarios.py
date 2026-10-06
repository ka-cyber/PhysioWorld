"""Numerical/software verification of the declared synthetic physiological model."""

import json
import math
from dataclasses import replace

import pytest

from physioworld.perturbations import (
    AdditiveNoise,
    BaselineWander,
    Clipping,
    DistributionShift,
    ElectrodeArtifact,
    LatencyJitter,
    Missingness,
    MotionArtifact,
    PacketLoss,
    exercise_challenge,
    hemorrhage_challenge,
)
from physioworld.physiology import (
    VARIABLE_UNITS,
    DataRequest,
    EnvironmentConditions,
    EquipmentState,
    OperatingEnvelopeError,
    OperatingEnvelopeWarning,
    PatientParameters,
    PhysiologicalAction,
    PhysiologicalScenario,
    PhysiologicalWorld,
    PhysiologyEngine,
    RequestAccessError,
    ScenarioRunner,
    SignalStream,
    SyntheticPhysiology,
    calibrate_patient,
    compare_fidelities,
    golden_scenarios,
    local_identifiability,
    parameter_sweep,
    run_population,
    run_scenario,
    sample_population,
    sampled_reachability,
)

ALL = DataRequest(tuple(VARIABLE_UNITS), ("blood_gas", "renal"), events=True)


def action(time, kind, **parameters):
    return PhysiologicalAction(time, kind, parameters)


def test_resting_equilibrium_and_analytical_gas_balance():
    world = PhysiologyEngine(requests=ALL)
    before = world.observe()
    after = world.advance(120)
    for name in before.values:
        if name != "urine_l":
            assert after[name] == pytest.approx(before[name], abs=2e-8)
    # Independent alveolar equation: PAO2 = FiO2*Pdry - 863*VO2/VA.
    expected = 0.2095 * 713 - 0.863 * 245 / before["ventilation_l_min"]
    assert after["alveolar_po2_mmhg"] == pytest.approx(expected, abs=1e-9)
    assert after["urine_l"] == pytest.approx(0.8 * 2 / 1000, abs=1e-12)


def test_conservation_of_fluid_sodium_and_respiratory_gases_with_interventions():
    world = PhysiologyEngine(requests=ALL)
    for event in (
        action(1.125, "hemorrhage", hemorrhage_ml_min=200),
        action(10.375, "exercise", exercise_mets=4),
        action(25.2, "infusion", infusion_ml_min=100),
        action(30.77, "bolus", volume_ml=250),
        action(40.1, "oxygen", oxygen_fraction=0.4),
        action(60.675, "hemorrhage", hemorrhage_ml_min=0),
    ):
        world.schedule(event)
    world.advance(90)
    residuals = world.conservation_residuals()
    assert abs(residuals["fluid_l"]) < 1e-10
    assert abs(residuals["sodium_mmol"]) < 1e-8
    assert abs(residuals["oxygen_ml"]) < 1e-7
    assert abs(residuals["co2_ml"]) < 1e-7
    assert world.observe()["blood_loss_l"] == pytest.approx(
        200 / 60000 * (60.675 - 1.125), abs=1e-12
    )


def test_positivity_and_debt_under_demand_delivery_mismatch():
    world = PhysiologyEngine(requests=ALL)
    world.act("exercise", exercise_mets=8)
    world.act("arrhythmia", arrhythmia_rate_bpm=170, rhythm_efficiency=0.25)
    world.act("hemorrhage", hemorrhage_ml_min=250)
    result = world.advance(80)
    assert all(math.isfinite(v) and v >= 0 for v in world.checkpoint()["state"].values())
    assert 0 <= result["oxygen_saturation"] <= 1
    assert result["oxygen_demand_ml_min"] > result["oxygen_consumption_ml_min"]
    assert result["oxygen_debt_ml"] > 0
    assert result["lactate_mmol_l"] > 1
    assert result["renal_filtration_ml_min"] < 120


def test_step_halving_converges_for_coupled_transient():
    def run(step):
        world = PhysiologyEngine(requests=ALL, max_step_s=step, rtol=0.1, atol=1e-6)
        world.schedule(action(0.37, "orthostasis", orthostasis=1))
        world.schedule(action(2.43, "exercise", exercise_mets=3))
        world.schedule(action(4.77, "oxygen", oxygen_fraction=0.4))
        return world.advance(12)

    coarse, medium, fine, reference = [run(s) for s in (1, 0.5, 0.25, 0.0625)]
    names = ("mean_pressure_mmhg", "cardiac_output_l_min", "arterial_pco2_mmhg")
    errors = [max(abs(o[n] - reference[n]) for n in names) for o in (coarse, medium, fine)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[0] < 0.001
    assert errors[1] < errors[0] / 3


def test_error_controller_rejects_inaccurate_large_steps():
    world = PhysiologyEngine(max_step_s=1, rtol=1e-11, atol=1e-12)
    world.act("oxygen", oxygen_fraction=0.5)
    world.advance(2)
    assert world.solver_statistics["rejected_steps"] > 0


def test_exact_hybrid_event_timing_and_conservative_impulse():
    world = PhysiologyEngine(requests=ALL, max_step_s=1)
    world.schedule(action(0.375, "hemorrhage", hemorrhage_ml_min=200))
    world.schedule(action(4.125, "hemorrhage", hemorrhage_ml_min=0))
    world.advance(0.374)
    assert world.observe()["blood_loss_l"] == 0
    world.advance_to(0.375)
    assert world.events[0]["time_s"] == 0.375
    assert world.observe()["blood_loss_l"] == 0
    world.advance_to(10)
    assert world.observe()["blood_loss_l"] == pytest.approx(200 / 60000 * 3.75, abs=1e-13)
    checkpoint = world.checkpoint()
    world.act("bolus", volume_ml=500)
    after = world.checkpoint()
    assert after["state"]["plasma_volume_l"] - checkpoint["state"][
        "plasma_volume_l"
    ] == pytest.approx(0.5)
    for gas in ("arterial_o2_ml", "venous_o2_ml", "arterial_co2_ml", "venous_co2_ml"):
        assert after["state"][gas] == checkpoint["state"][gas]
    assert (
        world.observe()["hemoglobin_g_dl"]
        < PhysiologyEngine.from_checkpoint(checkpoint).observe()["hemoglobin_g_dl"]
    )


def test_challenges_couple_pressure_gases_renal_and_regulation():
    world = PhysiologyEngine(requests=ALL)
    resting = world.observe()
    hemorrhage_challenge(0, rate_ml_min=300, duration_s=90).schedule(world)
    result = world.advance(90)
    assert result["blood_volume_l"] < resting["blood_volume_l"]
    assert result["cardiac_output_l_min"] < resting["cardiac_output_l_min"]
    assert result["mean_pressure_mmhg"] < resting["mean_pressure_mmhg"]
    assert result["sympathetic"] > resting["sympathetic"]
    assert result["adh"] > resting["adh"]
    assert result["renal_filtration_ml_min"] < resting["renal_filtration_ml_min"]
    assert result["oxygen_delivery_ml_min"] < resting["oxygen_delivery_ml_min"]
    assert world.events[-1]["parameters"]["hemorrhage_ml_min"] == 0


def test_environment_and_equipment_change_gas_exchange_and_preload():
    baseline = PhysiologyEngine(requests=ALL)
    co2 = baseline.fork("CO2", [action(0, "environment", co2_fraction=0.01)])
    ventilator = baseline.fork(
        "ventilation", [action(0, "ventilation", ventilation_l_min=8, peep_cmh2o=8)]
    )
    oxygen = baseline.fork("oxygen", [action(0, "oxygen", oxygen_fraction=0.5)])
    a, b, c = co2.advance(45), ventilator.advance(45), oxygen.advance(45)
    assert a["arterial_pco2_mmhg"] > baseline.observe()["arterial_pco2_mmhg"]
    assert a["ventilation_l_min"] > baseline.observe()["ventilation_l_min"]
    assert b["arterial_pco2_mmhg"] < baseline.observe()["arterial_pco2_mmhg"]
    assert b["cardiac_output_l_min"] < baseline.observe()["cardiac_output_l_min"]
    assert c["arterial_po2_mmhg"] > baseline.observe()["arterial_po2_mmhg"]


def test_fork_preserves_identity_future_events_resources_and_random_pairing():
    world = PhysiologyEngine(
        seed=42,
        process_noise_std=0.04,
        requests=ALL,
        resources={"energy_j": 10},
        resource_units={"energy_j": "J"},
        observation_noise={"mean_pressure_mmhg": 0.5},
    )
    world.schedule(action(10.375, "exercise", exercise_mets=2))
    world.advance(3.5)
    left = world.fork("untreated")
    right = world.fork("oxygen", [action(4.1, "oxygen", oxygen_fraction=0.5)])
    for branch in (left, right):
        assert branch.patient.subject_id == world.patient.subject_id
        assert branch.resources == world.resources
        assert branch.time_s == world.time_s
        branch.advance_to(20.5)
    assert left.checkpoint()["rng_state"] == right.checkpoint()["rng_state"]
    assert left.rng_identity == right.rng_identity

    def residual(branch):
        return (
            branch.observe()["mean_pressure_mmhg"]
            - branch.checkpoint()["state"]["mean_pressure_mmhg"]
        )

    assert residual(left) == pytest.approx(residual(right), abs=1e-13)
    assert left.observe()["oxygen_saturation"] != right.observe()["oxygen_saturation"]
    assert left.manifest()["causal_status"] == "MODEL_COUNTERFACTUAL"
    assert left.events[-1]["time_s"] == 10.375
    assert len(world.events) == 0


def test_checkpoint_file_restart_equivalence_and_diff(tmp_path):
    world = PhysiologyEngine(
        seed=7,
        process_noise_std=0.03,
        requests=ALL,
        resources={"energy_j": 10},
        resource_units={"energy_j": "J"},
    )
    world.schedule(action(4.125, "hemorrhage", hemorrhage_ml_min=120))
    world.schedule(action(8.25, "resource", energy_j=2))
    world.advance(5.375)
    path = tmp_path / "checkpoint.json"
    world.save(path)
    restarted = PhysiologyEngine.load(path)
    assert restarted.checkpoint() == world.checkpoint()
    world.advance_to(15)
    restarted.advance_to(15)
    assert restarted.checkpoint() == world.checkpoint()
    assert all(not value for value in world.diff(restarted).values())
    branch = world.fork("resource_branch")
    branch.act("resource", energy_j=0)
    branch.act("policy", policy_id="fallback")
    difference = world.diff(branch)
    assert difference["resources"] and difference["policy"] and difference["interventions"]
    bad = world.checkpoint()
    bad["model_version"] = "unsupported"
    with pytest.raises(ValueError, match="version"):
        PhysiologyEngine.from_checkpoint(bad)


def test_no_hidden_observation_access_or_rng_side_effects():
    world = PhysiologyEngine(requests=DataRequest(("heart_rate_bpm",), ("renal",)))
    rng_before = world.checkpoint()["rng_state"]
    observation = world.observe()
    assert set(observation.values) == {"heart_rate_bpm"}
    assert "sympathetic" not in observation.to_dict()["values"]
    with pytest.raises(RequestAccessError):
        _ = observation["sympathetic"]
    with pytest.raises(RequestAccessError):
        world.observe(DataRequest(("sympathetic",)))
    with pytest.raises(RequestAccessError):
        world.assess("blood_gas")
    with pytest.raises(TypeError):
        observation.values["sympathetic"] = 0.3
    assert world.assess("renal").evidence_class == "SIMULATION"
    assert world.checkpoint()["rng_state"] == rng_before
    with pytest.raises(RequestAccessError):
        PhysiologyEngine(requests=DataRequest(("secret_organ",)))
    with pytest.raises(RequestAccessError):
        PhysiologyEngine(requests=DataRequest(("resource.energy_j",)), resources={"energy_j": 1})


def test_sensor_failure_is_observation_only():
    world = PhysiologyEngine(requests=DataRequest(("oxygen_saturation",)))
    state_before = world.checkpoint()["state"]
    world.act("sensor", oxygen_saturation=False)
    assert world.observe()["oxygen_saturation"] is None
    assert world.checkpoint()["state"] == state_before


def test_scenario_json_chunking_and_restart_are_exact(tmp_path):
    config = {
        "duration_s": 12.3,
        "sample_interval_s": 0.7,
        "seed": 11,
        "process_noise_std": 0.02,
        "requests": ["mean_pressure_mmhg", "heart_rate_bpm"],
        "actions": [{"time_s": 2.125, "kind": "orthostasis", "parameters": {"orthostasis": 1}}],
    }
    scenario = PhysiologicalScenario.from_json(json.dumps(config))
    whole = scenario.run()
    runner = scenario.runner()
    first = runner.run_chunk(5)
    path = tmp_path / "runner.json"
    runner.save(path)
    restored = ScenarioRunner.load(path)
    chunked = (*first, *(o for chunk in restored.chunks(3) for o in chunk))
    assert [o.to_dict() for o in chunked] == [o.to_dict() for o in whole.observations]
    assert restored.engine.checkpoint() == whole.final_checkpoint
    assert chunked[-1].time_s == 12.3
    assert restored.run_chunk(5) == ()
    assert run_scenario(config).manifest["scenario_sha256"] == whole.manifest["scenario_sha256"]


def test_population_and_sweep_are_deterministic_and_declared_synthetic():
    patients = sample_population(3, seed=17)
    assert patients == sample_population(3, seed=17)
    assert len({p.subject_id for p in patients}) == 3
    assert all(p.parameter_origin == "ASSUMED_SYNTHETIC" for p in patients)
    scenario = PhysiologicalScenario(duration_s=2)
    results = list(run_population(scenario, patients))
    assert [r.manifest["patient"]["subject_id"] for r in results] == [
        p.subject_id for p in patients
    ]
    sweeps = list(parameter_sweep(scenario, {"patient.hemoglobin_g_dl": [10, 15]}))
    assert len(sweeps) == 2
    assert all(r.manifest["causal_status"] == "MODEL_COUNTERFACTUAL" for _, r in sweeps)
    assert sweeps[0][1].manifest["rng_identity"] == sweeps[1][1].manifest["rng_identity"]


@pytest.mark.parametrize(
    "perturbation",
    [
        AdditiveNoise(1, onset_s=1, end_s=3),
        BaselineWander(1, 0.25, onset_s=1, end_s=3),
        Clipping(-0.5, 0.5, onset_s=1, end_s=3),
        DistributionShift(2, 1, onset_s=1, end_s=3),
        ElectrodeArtifact(1, 2, 10, onset_s=1, end_s=3),
        MotionArtifact(1, 1, onset_s=1, end_s=3),
    ],
)
def test_signal_perturbations_have_actual_timing_masks_and_evidence(perturbation):
    source = SignalStream.from_samples([1, 1, 1, 1, 1], 1, "mV", "ECG")
    modified = perturbation.apply(source)
    assert modified.values[0] == source.values[0]
    assert modified.values[3:] == source.values[3:]
    assert modified.values[1:3] != source.values[1:3]
    assert perturbation.timing_mask(source.timestamps) == (False, True, True, False, False)
    metadata = modified.metadata.provenance.notes
    assert all(
        k in metadata for k in ("cause", "timing", "mechanism", "evidence_basis", "safety_status")
    )


def test_missingness_blocks_never_escape_declared_time_interval():
    source = SignalStream.from_samples([1] * 6, 1, "mV", "ECG")
    modified = Missingness(1, block_length=10, onset_s=2, end_s=4).apply(source)
    assert modified.metadata.missing == (False, False, True, True, False, False)
    assert modified.values[4] == source.values[4]


def test_network_timing_uses_seconds_and_never_packet_indices():
    times = (10, 11, 12, 13, 14)
    loss = PacketLoss(1, burst_length=10, onset_s=11, end_s=13)
    assert loss.mask(5, timestamps_s=times) == (False, True, True, False, False)
    latency = LatencyJitter(2, onset_s=11, end_s=13)
    assert latency.delays(5, timestamps_s=times) == (0, 2, 2, 0, 0)
    with pytest.raises(ValueError, match="timestamps"):
        loss.mask(5)


def test_pure_oxygen_and_telemetry_missingness_are_supported():
    world = PhysiologyEngine(requests=ALL)
    world.act("oxygen", oxygen_fraction=1.0)
    before = world.observe()["arterial_po2_mmhg"]
    after = world.advance(30)
    assert after["arterial_po2_mmhg"] > before
    assert abs(world.conservation_residuals()["oxygen_ml"]) < 1e-7
    scenario = PhysiologicalScenario(
        duration_s=3, actions=(action(1, "sensor", oxygen_saturation=False),)
    )
    stream = SyntheticPhysiology().telemetry(duration_s=3, scenario=scenario)
    assert stream.metadata.missing == (False, True, True)
    assert math.isnan(stream.channel("oxygen_saturation")[1])


def test_coupling_boundary_and_checkpoint_clock_validation():
    world = PhysiologyEngine(requests=DataRequest(("heart_rate_bpm",)))
    frame = world.coupling_step(2, (action(0.5, "exercise", exercise_mets=2),))
    assert frame["time_s"] == 2 and set(frame["values"]) == {"heart_rate_bpm"}
    bad = world.checkpoint()
    bad["noise_tick"] = 0
    with pytest.raises(ValueError, match="clock"):
        PhysiologyEngine.from_checkpoint(bad)


def test_generators_have_native_units_coupling_and_honest_labels():
    generator = SyntheticPhysiology(seed=1)
    source = PhysiologicalScenario(duration_s=4, actions=(action(1, "exercise", exercise_mets=3),))
    telemetry = generator.telemetry(duration_s=4, scenario=source)
    assert telemetry.n_channels == 9
    assert telemetry.metadata.channels[0].unit == "1/min"
    assert telemetry.metadata.channels[1].unit == "mmHg"
    assert telemetry.channel("heart_rate_bpm")[-1] > telemetry.channel("heart_rate_bpm")[0]
    ecg = generator.ecg(duration_s=1, sample_rate=100)
    assert ecg.n_samples == 100 and ecg.metadata.channels[0].unit == "mV"
    eeg = generator.eeg(duration_s=1, sample_rate=64)
    assert eeg.n_samples == 64 and eeg.metadata.channels[0].unit == "uV"
    assert eeg.values == generator.eeg(duration_s=1, sample_rate=64).values
    assert "surrogate" in eeg.metadata.provenance.notes["model"]
    for stream in (telemetry, ecg, eeg):
        assert stream.metadata.provenance.notes["evidence_class"] == "SIMULATION"
        assert not stream.metadata.provenance.notes["clinical_validation"]


def test_original_stream_world_and_chunked_replay_api():
    stream = SignalStream.from_samples([1, 2, 3], 1, "mV", "ECG")
    world = PhysiologicalWorld.synthetic({"ECG": stream}, {"synthetic": True})
    branch = world.fork()
    snapshots = [s for chunk in world.replay_chunks(2) for s in chunk]
    assert [s.time for s in snapshots] == [0, 1, 2]
    assert snapshots[-1].physiological == {"ECG": (3.0,)}
    assert [s.time for s in branch.replay()] == [0, 1, 2]
    assert isinstance(PhysiologicalWorld.physiological(), PhysiologyEngine)


def test_calibration_checksum_grid_uncertainty_and_units():
    scenario = PhysiologicalScenario(duration_s=2, requests=DataRequest(("heart_rate_bpm",)))
    rows = [{"time_s": o.time_s, "values": dict(o.values)} for o in scenario.run().observations]
    kwargs = dict(
        parameter_grid={"heart_rate_bpm": [68, 72, 76]},
        units={"heart_rate_bpm": "1/min"},
        scales={"heart_rate_bpm": 1},
        dataset_id="synthetic-calibration-fixture",
        data_origin="SYNTHETIC",
        operating_conditions={"exercise_mets": 1},
    )
    fit = calibrate_patient(scenario, rows, **kwargs)
    assert fit.parameters == {"heart_rate_bpm": 72}
    assert fit.objective == 0
    assert not fit.uncertainty["is_confidence_interval"]
    assert fit.input_sha256 == calibrate_patient(scenario, rows, **kwargs).input_sha256
    with pytest.raises(ValueError, match="unit mismatch"):
        calibrate_patient(scenario, rows, **{**kwargs, "units": {"heart_rate_bpm": "Hz"}})


def test_local_excitation_detects_unidentifiable_resting_controller_time_constant():
    scenario = PhysiologicalScenario(duration_s=3)
    report = local_identifiability(
        scenario,
        ["autonomic_tau_s"],
        variables=["heart_rate_bpm"],
        times_s=[1, 2, 3],
        scales={"heart_rate_bpm": 1},
    )
    assert report["local_numerical_rank"] == 0
    excited = replace(scenario, actions=(action(0, "orthostasis", orthostasis=1),))
    report = local_identifiability(
        excited,
        ["autonomic_tau_s"],
        variables=["heart_rate_bpm"],
        times_s=[1, 2, 3],
        scales={"heart_rate_bpm": 1},
    )
    assert report["local_numerical_rank"] == 1
    assert report["structural_identifiability"] == "NOT_ANALYZED"


def test_reduction_and_reachability_are_scoped_numerical_comparisons():
    scenario = PhysiologicalScenario(duration_s=4)
    comparison = compare_fidelities(
        scenario, variables=["mean_pressure_mmhg"], atol={"mean_pressure_mmhg": 0.01}
    )
    assert comparison["criterion_met"]
    result = sampled_reachability(
        scenario,
        {"rest": (), "rhythm": (action(0, "arrhythmia", rhythm_efficiency=0.2),)},
        bounds={"mean_pressure_mmhg": (65, 120)},
    )
    assert result["branches"]["rest"]["sampled_safe"]
    assert not result["branches"]["rhythm"]["sampled_safe"]
    assert not result["is_viability_kernel"]


def test_envelopes_and_invalid_configuration_fail_explicitly():
    with pytest.raises(ValueError):
        PatientParameters(mass_kg=float("nan"))
    with pytest.raises(ValueError):
        EquipmentState(ventilation_l_min=-1)
    with pytest.raises(ValueError):
        EnvironmentConditions(oxygen_fraction=2)
    world = PhysiologyEngine(envelope_policy="fail")
    with pytest.raises(ValueError):
        world.schedule(action(0, "hemorrhage", hemorrhage_ml_min=10000))
    with pytest.raises(ValueError):
        world.advance(-1)
    with pytest.raises(OperatingEnvelopeError):
        world.advance(21601)
    world.act("arrhythmia", rhythm_efficiency=0.1)
    with pytest.raises(OperatingEnvelopeError):
        world.advance(10)
    with pytest.raises(ValueError):
        world.schedule(action(0, "exercise", exercise_mets=2))
    recording = PhysiologyEngine()
    recording.act("arrhythmia", rhythm_efficiency=0.1)
    with pytest.warns(OperatingEnvelopeWarning):
        recording.advance(5)
    assert recording.manifest()["envelope_violations"]


def test_golden_world_directional_behaviors():
    golden = golden_scenarios(20)
    baseline = golden["ideal"].run().observations[-1]
    oxygen = golden["oxygen"].run().observations[-1]
    exercise = golden["exercise"].run().observations[-1]
    assert oxygen["oxygen_saturation"] > baseline["oxygen_saturation"]
    assert exercise["heart_rate_bpm"] > baseline["heart_rate_bpm"]
    assert golden["missing_modality"].run().observations[-1]["oxygen_saturation"] is None
    assert golden["resource_depletion"].run().observations[-1]["resource.energy_j"] == 0
    assert exercise_challenge(1, duration_s=2).metadata()["evidence_class"] == "SIMULATION"

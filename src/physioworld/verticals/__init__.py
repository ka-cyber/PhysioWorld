"""Config-driven finite-population physiology, telemetry and EEG workflows.

The task-law estimator is an explicitly in-sample finite empirical law, not a
forecast calibration or clinical-validation claim. Policies are evaluated on
the entire followed-up population, including unavailable observations. The
runtime never conditions utility or discrepancy on successful packet delivery.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import struct
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from physioworld.capability import CapabilityRegion, support_gap
from physioworld.communication import (
    ErasureChannel,
    LatencyJitter,
    PacketLoss,
    packetize,
    reconstruct,
)
from physioworld.compute import ComputeEngine, ComputeNode, ModelProfile
from physioworld.core import (
    CausalStatus,
    EvidenceClass,
    MetricContext,
    OperatingEnvelope,
    Task,
    TaskKind,
)
from physioworld.evidence_origin import (
    estimator_evidence_metadata,
    source_evidence_class,
)
from physioworld.experiments import ExperimentCompiler, ExperimentResult
from physioworld.physiology import (
    PhysiologicalAction,
    PhysiologicalScenario,
    SyntheticPhysiology,
)
from physioworld.representations import FiniteHistoryRepresentation, MultiscaleHistoryRepresentation
from physioworld.resources import CapacitorResource, ResourceState
from physioworld.tasklaws import conditional_task_law
from physioworld.version import __version__


def vertical_names() -> tuple[str, ...]:
    return ("cardiovascular", "neurophysiology", "icu")


def default_configuration(name: str, *, seed: int = 0) -> dict[str, Any]:
    """Complete, schema-visible demonstration declarations, also used by CLI qualification."""
    if name not in vertical_names():
        raise ValueError(f"unknown vertical {name!r}; choose one of {vertical_names()}")
    neuro = name == "neurophysiology"
    channel = "EEG" if neuro else "oxygen_saturation" if name == "icu" else "mean_pressure_mmhg"
    unit = "uV" if neuro else "1" if name == "icu" else "mmHg"
    threshold = 5.0 if neuro else 0.979 if name == "icu" else 89.0
    rate = 64 if neuro else 1
    return {
        "$schema": "vertical.schema.json",
        "schema_version": "1.0.0",
        "name": f"{name}_vertical",
        "vertical": name,
        "seed": seed,
        "dataset": {
            "id": f"synthetic_{name}",
            "origin": "SYNTHETIC",
            "model": "SyntheticPhysiology",
            "modality": "eeg" if neuro else "telemetry",
            "channel": channel,
            "unit": unit,
            "time_unit": "s",
            "duration_s": 12.0,
            "sample_rate_hz": rate,
            "sample_interval_s": 1 / rate,
        },
        "task": {
            "id": f"{name}_threshold",
            "kind": "threshold_exceedance",
            "target": channel,
            "horizon_s": 0.0,
            "metric": "task_law_discrepancy",
            "threshold": threshold,
            "threshold_unit": unit,
            "loss": "zero_one",
            "reference": "full_information",
            "missingness": "unavailable",
            "uncertainty": "observed_availability",
        },
        "representation": (
            {
                "name": "multiscale_history",
                "windows": [1, 4, 8],
                "window_unit": "samples",
                "statistics": ["mean", "std", "trend", "last"],
            }
            if neuro
            else {"name": "finite_history", "window": 5, "window_unit": "samples", "pad_value": 0.0}
        ),
        "perturbations": [
            {
                "type": "physiological_action",
                "time_s": 4.0,
                "kind": "hemorrhage" if name == "icu" else "orthostasis",
                "parameters": {"hemorrhage_ml_min": 120.0}
                if name == "icu"
                else {"orthostasis": 1.0},
            }
        ],
        "channel": {
            "model": "erasure",
            "loss_probability": 0.05 if neuro else 0.2 if name == "icu" else 0.0,
            "latency_s": 0.002 if neuro else 0.02,
            "jitter_s": 0.001 if neuro else 0.005,
            "packet_payload_bytes": 256,
            "encoding": "float32_be",
            "deadline_s": 1 / rate,
            "semantic_permutation": False,
        },
        "resource": {
            "model": "capacitor_p1",
            "capacitance_f": 0.2,
            "esr_ohm": 8.0,
            "leakage_siemens": 0.0,
            "initial_voltage_v": 2.65,
            "v_min": 1.85,
            "v_max": 3.3,
            "max_current_a": 0.02,
            "load_current_a": 0.001,
            "step_duration_s": 0.0001,
        },
        "compute": {
            "location": "local",
            "operations_per_sample": 100.0,
            "operations_per_second": 100000.0,
            "memory_bytes": 4096,
            "required_memory_bytes": 64,
            "energy_per_operation_j": 1e-9,
            "network_latency_s": 0.0,
        },
        "policy": {
            "class": "declared_threshold_policy",
            "thresholds": [threshold - 0.001, threshold, threshold + 0.001],
            "threshold_unit": unit,
            "information_set": "received_history",
            "action_set": ["negative", "positive", "abstain"],
            "fallback_action": "abstain",
            "rewards": {
                "true_positive": 1.0,
                "false_positive": -1.0,
                "true_negative": 1.0,
                "false_negative": -1.0,
                "unavailable": 0.0,
            },
            "support_direction": [1.0, 0.0],
        },
        "split": {"strategy": "synthetic_fixed"},
        "metrics": [
            "packet_delivery",
            "available_fraction",
            "task_law_discrepancy",
            "task_value",
            "capability_support",
            "energy_j",
            "terminal_voltage_v",
            "compute_latency_s",
        ],
        "outputs": {"format": "json"},
    }


@dataclass(frozen=True)
class VerticalResult:
    """The public result is the versioned scientific contract actually persisted."""

    result: ExperimentResult

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned result plus stable vertical convenience fields.

        The :class:`ExperimentResult` contract keeps scientific outputs under
        ``outputs`` so that the persisted schema is uniform across workflows.
        Vertical callers historically consumed ``transport`` and ``semantic``
        directly, however.  These are read-only projections of the exact
        contract values; they are not independently computed metrics and cannot
        change the estimand.
        """
        payload = self.result.to_dict()
        outputs = payload["outputs"]
        payload.update(
            {
                "vertical_id": outputs["vertical_id"],
                "question": (
                    "How do the declared physiological information, transport, "
                    "resource, and policy conditions affect the task?"
                ),
                "limitations": list(payload["provenance"].get("limitations", ())),
                "transport": outputs["transport"],
                "semantic": outputs["semantics"],
            }
        )
        return payload


def _source(
    config: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...], dict[str, Any]]:
    data = config["dataset"]
    if data["origin"] == "REPLAY":
        evidence_origin, evidence_class = source_evidence_class(data)
        source = {
            "source": data["source"],
            "origin": "REPLAY",
            "evidence_origin": evidence_origin,
            "evidence_class": evidence_class.value,
        }
        if "artifact_metadata" in data:
            source["artifact_metadata"] = data["artifact_metadata"]
        return (
            tuple(data["timestamps_s"]),
            tuple(data["samples"]),
            {
                **source,
                "units": {data["channel"]: data["unit"]},
                "causal_status": "PREDICTIVE_REPLAY",
                "interventions": [],
            },
        )
    actions = tuple(
        PhysiologicalAction(p["time_s"], p["kind"], p["parameters"])
        for p in config["perturbations"]
        if p["type"] == "physiological_action"
    )
    scenario = PhysiologicalScenario(
        name=config["name"] if "name" in config else config["vertical"],
        duration_s=data["duration_s"],
        sample_interval_s=data["sample_interval_s"],
        seed=config["seed"],
        actions=actions,
    )
    generator = SyntheticPhysiology(seed=config["seed"])
    if data["modality"] == "eeg":
        stream = generator.eeg(
            duration_s=data["duration_s"], sample_rate=data["sample_rate_hz"], scenario=scenario
        )
        values = stream.channel(0)
    else:
        stream = generator.telemetry(
            duration_s=data["duration_s"],
            sample_rate=data["sample_rate_hz"],
            scenario=scenario,
            variables=(data["channel"],),
        )
        values = stream.channel(data["channel"])
    return (
        stream.timestamps,
        values,
        {
            "source": "SyntheticPhysiology",
            "origin": "SYNTHETIC",
            "evidence_origin": "SYNTHETIC_MODEL",
            "evidence_class": EvidenceClass.SIMULATION.value,
            "scenario": scenario.to_dict(),
            "generator": stream.metadata.provenance.notes if stream.metadata.provenance else {},
            "units": {data["channel"]: data["unit"]},
            "causal_status": "MODEL_SIMULATION",
            "interventions": [asdict(action) for action in actions],
        },
    )


def _features(config: Mapping[str, Any], observations: tuple[float, ...]):
    declaration = config["representation"]
    unit = config["dataset"]["unit"]
    method: Any
    if declaration["name"] == "finite_history":
        method = FiniteHistoryRepresentation(declaration["window"], declaration["pad_value"])
        descriptors = tuple(
            {"name": f"lag_{declaration['window'] - i - 1}", "unit": unit}
            for i in range(declaration["window"])
        )
        decision_index = len(descriptors) - 1
    else:
        stats = tuple(declaration["statistics"])
        if "last" not in stats:
            raise ValueError(
                "threshold policy requires a declared multiscale 'last' statistic in native units"
            )
        method = MultiscaleHistoryRepresentation(tuple(declaration["windows"]), statistics=stats)
        descriptors = tuple(
            {
                "name": f"window_{window}_{stat}",
                "unit": f"{unit}/sample" if stat == "trend" else unit,
            }
            for window in declaration["windows"]
            for stat in stats
        )
        decision_index = stats.index("last")
    represented = tuple(method.transform(observations[: i + 1]) for i in range(len(observations)))
    return represented, descriptors, decision_index


def _reward(label: int, action: str, rewards: Mapping[str, float]) -> float:
    if action == "abstain":
        return rewards["unavailable"]
    return rewards[
        ("true_" if bool(label) == (action == "positive") else "false_")
        + ("positive" if action == "positive" else "negative")
    ]


def run_vertical(name: str | Mapping[str, Any], *, seed: int | None = None) -> VerticalResult:
    """Execute all declarations; scalar task laws use finite followed-up atoms.

    Float32 conversion is explicit. Each feature vector is fragmented and only
    reconstructs when every fragment arrives before the acquisition-relative
    deadline. Resource work is billed before transmission, including compute
    energy through a conservative constant-current envelope at ``v_min``.
    Counterfactual full-information and received-information policy sets are
    independently optimized; policy utilities include the explicit abstention.
    """
    declaration = (
        default_configuration(name, seed=0 if seed is None else seed)
        if isinstance(name, str)
        else dict(name)
    )
    if seed is not None and not isinstance(name, str) and seed != declaration["seed"]:
        raise ValueError("seed override conflicts with the declared experiment")
    plan = ExperimentCompiler().compile(declaration)
    config = plan.configuration
    if plan.configuration_kind != "vertical":
        raise ValueError("run_vertical requires a compiled vertical declaration")
    times, latent_values, source = _source(config)
    observed_values = list(latent_values)
    for index, p in enumerate(config["perturbations"]):
        if p["type"] == "additive_noise":
            rng = random.Random(p.get("seed", config["seed"] + index))
            observed_values = [value + rng.gauss(0, p["sigma"]) for value in observed_values]
    represented, descriptors, decision_index = _features(config, tuple(observed_values))
    c, r, compute = config["channel"], config["resource"], config["compute"]
    resource = CapacitorResource(
        r["capacitance_f"],
        r["esr_ohm"],
        r["leakage_siemens"],
        v_min=r["v_min"],
        v_max=r["v_max"],
        max_current_a=r["max_current_a"],
        model_name=r["model"],
    )
    state = ResourceState(r["initial_voltage_v"], time=times[0])
    initial_energy = resource.energy(state).value_joules
    node = ComputeNode(
        "declared",
        compute["location"],
        compute["operations_per_second"],
        compute["memory_bytes"],
        compute["energy_per_operation_j"] * compute["operations_per_second"],
        network_latency_s=compute["network_latency_s"],
    )
    model = ModelProfile(
        "representation", compute["required_memory_bytes"], compute["operations_per_sample"]
    )
    executor = ComputeEngine({node.node_id: node}, {model.model_id: model}, seed=config["seed"])
    channel = ErasureChannel(
        PacketLoss(c["loss_probability"], seed=config["seed"]),
        LatencyJitter(c["latency_s"], c["jitter_s"], seed=config["seed"]),
        deadline=c["deadline_s"],
        seed=config["seed"],
    )
    row_records: list[dict[str, Any]] = []
    sent_packets = delivered_packets = wire_bytes = 0
    delays: list[float] = []
    clocks: list[float] = []
    transitions: list[dict[str, Any]] = []
    sequence = 10
    depleted = False
    for index, (time, feature) in enumerate(zip(times, represented)):
        payload = struct.pack(f"!{len(feature.values)}f", *feature.values)
        quantized = struct.unpack(f"!{len(feature.values)}f", payload)
        if any(not math.isfinite(v) for v in quantized):
            raise ValueError("float32 encoding overflow; explicitly rescale the representation")
        count = math.ceil(len(payload) / c["packet_payload_bytes"])
        packets = packetize(
            payload,
            c["packet_payload_bytes"],
            timestamp=time,
            sequences=tuple(sequence + j * 10 for j in range(count)),
            source_id=config["dataset"]["id"],
            modality=config["dataset"]["modality"],
            representation=config["representation"]["name"],
        )
        sequence += (count + 1) * 10
        if c.get("semantic_permutation"):
            permuted = tuple(reversed(feature.values))
            payload = struct.pack(f"!{len(permuted)}f", *permuted)
            packets = tuple(
                type(p)(
                    p.sequence,
                    payload[offset : offset + c["packet_payload_bytes"]],
                    p.timestamp,
                    p.metadata,
                    p.header,
                )
                for p, offset in zip(packets, range(0, len(payload), c["packet_payload_bytes"]))
            )
        deadline = time + c["deadline_s"]
        start = max(time, state.time, channel.time)
        availability_reason = None
        idle = resource.step(state, 0.0, start - state.time)
        if idle.feasible:
            state = idle.next
        else:
            depleted = True
        quote = executor.execute("declared", "representation", arrival_time_s=start)
        received_features = None
        final_time = start
        if depleted:
            availability_reason = "resource_idle_envelope"
        elif start > deadline or quote.completion_time_s > deadline:
            availability_reason = "compute:deadline exceeded"
        elif not quote.feasible:
            availability_reason = f"compute:{quote.reason}"
        else:
            duration = quote.execution_time_s + count * r["step_duration_s"]
            if r["v_min"] <= 0 and quote.energy_j > 0:
                raise ValueError(
                    "positive compute energy requires a positive declared v_min for the current envelope"
                )
            current = r["load_current_a"] + (
                quote.energy_j / (r["v_min"] * duration) if quote.energy_j else 0
            )
            transition = resource.step(state, current, duration)
            transitions.append(
                {
                    "row": index,
                    "start_s": state.time,
                    "duration_s": duration,
                    "current_a": current,
                    "feasible": transition.feasible,
                    "reason": transition.reason,
                    "energy_residual_j": transition.energy_balance.residual_j,
                    "terminal_voltage_v": transition.terminal_voltage,
                }
            )
            if not transition.feasible:
                availability_reason = f"resource:{transition.reason}"
            else:
                execution = executor.submit(
                    "declared", "representation", arrival_time_s=start, deadline_s=deadline
                )
                state = transition.next
                transmitted = channel.transmit(
                    packets, ready_times={p.sequence: state.time for p in packets}
                )
                sent_packets += len(packets)
                delivered_packets += transmitted.transport.delivered_packets
                wire_bytes += transmitted.wire_bytes
                delays.extend(transmitted.delays.values())
                clocks.append(execution.execution_time_s)
                final_time = max(state.time, channel.time)
                if len(transmitted.received) == len(packets):
                    decoded = reconstruct(
                        transmitted.received, expected_sequences=tuple(p.sequence for p in packets)
                    )
                    received_features = struct.unpack(f"!{len(feature.values)}f", decoded)
                else:
                    availability_reason = "channel_loss_or_deadline"
        future_index = bisect.bisect_left(times, time + config["task"]["horizon_s"])
        label = (
            int(latent_values[future_index] >= config["task"]["threshold"])
            if future_index < len(times)
            else None
        )
        endpoint_time = times[future_index] if label is not None else None
        # For a genuinely prospective horizon, a result arriving after the
        # endpoint is not an available prediction. Horizon-zero threshold
        # tasks are explicitly contemporaneous classifications and retain the
        # declared channel deadline as their operational timing regime.
        if (
            label is not None
            and endpoint_time is not None
            and config["task"]["horizon_s"] > 0
            and final_time > endpoint_time + 1e-12
        ):
            received_features = None
            availability_reason = "decision_after_endpoint"
        row_records.append(
            {
                "acquisition_time_s": time,
                "decision_time_s": final_time,
                "deadline_s": deadline,
                "representation": feature.values,
                "quantized_representation": quantized,
                "received_representation": received_features,
                "representation_metadata": feature.metadata,
                "endpoint": label,
                "endpoint_time_s": endpoint_time,
                "endpoint_status": "OBSERVED" if label is not None else "INCOMPLETE_FOLLOW_UP",
                "available": received_features is not None,
                "unavailability_reason": availability_reason,
                "fallback_action": config["policy"]["fallback_action"]
                if received_features is None
                else None,
                "resource_voltage_v": state.voltage,
                "full_decision_statistic": observed_values[index],
                "received_decision_statistic": received_features[decision_index]
                if received_features is not None
                else None,
            }
        )
    evaluated = tuple(row for row in row_records if row["endpoint"] is not None)
    if not evaluated:
        raise ValueError("no fully followed-up endpoint at the declared horizon")
    task = Task(
        config["task"]["id"],
        TaskKind.CLASSIFICATION,
        config["task"]["horizon_s"],
        config["task"]["target"],
        "zero_one",
        threshold=config["task"]["threshold"],
        evaluation_population="declared finite followed-up atoms",
        resource_regime=plan.configuration_hash,
        reference_condition="full_information",
        uncertainty_method=config["task"]["uncertainty"],
    )
    context = MetricContext(
        "sup_task_mean_absolute_conditional_expectation_difference",
        config["dataset"]["id"],
        "resource_channel_conditioned_finite_population",
        ("complete_follow_up",),
        "declared_threshold_policy",
        "same_followed_up_atoms",
        task.horizon,
        (task.task_id,),
        json.dumps(r, sort_keys=True),
        config["task"]["uncertainty"],
        metric_type="task_law_discrepancy",
        utility="zero_one",
        population="declared finite atoms",
        causal_status=CausalStatus.PREDICTIVE,
        evidence_class=EvidenceClass(source["evidence_class"]),
        communication_regime=json.dumps(c, sort_keys=True),
        information_condition="received_representation",
        estimator="in_sample_empirical_conditional_distribution",
        time_unit="s",
    )
    groups = tuple(
        ("unavailable",) if not row["available"] else tuple(row["received_representation"])
        for row in evaluated
    )
    law = conditional_task_law(
        tuple(row["endpoint"] for row in evaluated),
        groups,
        task,
        history_groups=tuple(range(len(evaluated))),
        context=context,
        evidence_class=context.evidence_class,
        evidence_metadata=estimator_evidence_metadata(
            source["evidence_origin"], context.evidence_class, source
        ),
    )
    representation_groups = tuple(tuple(row["representation"]) for row in evaluated)
    intrinsic = conditional_task_law(
        tuple(row["endpoint"] for row in evaluated),
        representation_groups,
        task,
        history_groups=tuple(range(len(evaluated))),
        evidence_class=context.evidence_class,
        evidence_metadata=estimator_evidence_metadata(
            source["evidence_origin"], context.evidence_class, source
        ),
    )
    thresholds = tuple(config["policy"]["thresholds"])
    rewards = config["policy"]["rewards"]
    policy_ids = tuple(f"threshold:{threshold.hex()}" for threshold in map(float, thresholds)) + (
        "always_abstain",
    )
    available = sum(row["available"] for row in evaluated) / len(evaluated)
    full_values, actual_values, actions = {}, {}, {}
    for policy_id, threshold in zip(policy_ids, (*thresholds, None)):
        full_actions = tuple(
            "abstain"
            if threshold is None
            else "positive"
            if row["full_decision_statistic"] >= threshold
            else "negative"
            for row in evaluated
        )
        actual_actions = tuple(
            "abstain"
            if threshold is None or not row["available"]
            else "positive"
            if row["received_decision_statistic"] >= threshold
            else "negative"
            for row in evaluated
        )
        full_values[policy_id] = (
            sum(
                _reward(row["endpoint"], action, rewards)
                for row, action in zip(evaluated, full_actions)
            )
            / len(evaluated),
            1.0,
        )
        actual_values[policy_id] = (
            sum(
                _reward(row["endpoint"], action, rewards)
                for row, action in zip(evaluated, actual_actions)
            )
            / len(evaluated),
            available,
        )
        actions[policy_id] = actual_actions
    full_region = CapabilityRegion(
        full_values, "full_information", task_labels=("policy_utility", "availability")
    )
    actual_region = CapabilityRegion(
        actual_values, "received_representation", task_labels=full_region.task_labels
    )
    direction = config["policy"]["support_direction"]
    support = support_gap(full_region, actual_region, direction)
    selected = support["reduced_policy"]
    if not isinstance(selected, str):
        raise ValueError("capability support did not select a reduced policy")
    distortion = (
        sum(
            (row["full_decision_statistic"] - row["received_decision_statistic"]) ** 2
            for row in evaluated
        )
        / len(evaluated)
        if available == 1
        else None
    )
    outputs = {
        "vertical_id": config["vertical"],
        "rows": row_records,
        "evaluation": {
            "total_atoms": len(row_records),
            "evaluated_atoms": len(evaluated),
            "incomplete_follow_up_atoms": len(row_records) - len(evaluated),
            "independent_subjects": 1,
            "split": config["split"],
            "scope": "finite in-sample descriptive population; no generalization interval",
        },
        "transport": {
            "offered_vectors": len(times),
            "sent_packets": sent_packets,
            "delivered_packets": delivered_packets,
            "wire_bytes": wire_bytes,
            "packet_delivery": delivered_packets / sent_packets if sent_packets else None,
            "mean_latency_s": sum(delays) / len(delays) if delays else None,
        },
        "semantics": {
            "available_fraction": available,
            "reconstruction_distortion": distortion,
            "task_law_discrepancy": law.discrepancy(),
            "intrinsic_task_law_discrepancy": intrinsic.discrepancy(),
            "task_law": law.to_dict(),
            "metric_context": context.to_dict(),
        },
        "compute": {
            "executed_jobs": len(clocks),
            "mean_latency_s": sum(clocks) / len(clocks) if clocks else None,
        },
        "resource": {
            "initial_energy_j": initial_energy,
            "final_energy_j": resource.energy(state).value_joules,
            "consumed_energy_j": initial_energy - resource.energy(state).value_joules,
            "terminal_voltage_v": state.voltage,
            "transitions": transitions,
            "cost_rule": "constant current: declared load + compute_energy/(v_min*job_duration); idle leakage explicit",
            "energy_status": "MODEL_DERIVED",
        },
        "capability": {
            "full": full_region.to_dict(),
            "actual": actual_region.to_dict(),
            "support": support,
            "selected_policy": selected,
            "selected_actions": actions[selected],
            "endpoint_task_value": actual_values[selected][0],
            "support_direction": direction,
        },
        "causal_gate": {
            "status": "UNIDENTIFIED",
            "reason": "prediction/replay does not supply an intervention identification record",
        },
    }
    result = ExperimentResult(
        plan.configuration_hash,
        __version__,
        None,
        plan.experiment_id,
        dict(config["dataset"]),
        task.to_dict(),
        "finite_population_empirical_task_law_and_independently_optimized_policy_utility",
        {"source": source, "fidelity": "declared_reduced_order_or_replay"},
        {
            **dict(config["representation"]),
            "features": descriptors,
            "decision_feature_index": decision_index,
            "encoding": "IEEE754_float32_big_endian",
            "nonanticipative": True,
        },
        dict(c),
        dict(r),
        dict(config["policy"]),
        config["seed"],
        {
            "method": config["task"]["uncertainty"],
            "interpretation": "seeded simulation/availability; not a subject confidence interval",
        },
        context.evidence_class,
        OperatingEnvelope(
            f"{config['vertical']}-finite-world",
            "declared configuration only",
            {"acquisition_time_s": (times[0], times[-1])},
        ),
        {
            "configuration_hash": plan.configuration_hash,
            "source": source,
            "resolved_configuration": config,
            "schema_name": plan.schema_name,
            "execution_mode": plan.data_mode,
            "limitations": [
                "In-sample empirical laws are not calibrated out-of-sample forecasts.",
                "Model-derived resource work is not a hardware measurement.",
                "The full-information reference is a privileged information counterfactual, not an observed causal intervention.",
            ],
        },
        outputs,
    )
    return VerticalResult(result)

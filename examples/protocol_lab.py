"""Execute a native declarative protocol against the reduced-order physiology engine.

This is simulation-only: the handlers below inject a ``PhysiologyEngine`` and a
declared sensor calibration.  No hardware measurement or clinical evidence is
claimed.  Run from an installed checkout with::

    python examples/protocol_lab.py

The printed manifest shows chronological execution, a calibrated sensor
declaration, and a derived shock-index measurement.
"""

from __future__ import annotations

from physioworld.devices import Calibration
from physioworld.experiments.protocol import (
    Baseline,
    Computation,
    Condition,
    Event,
    Export,
    HandlerRegistry,
    If,
    Intervention,
    Measurement,
    Protocol,
    ProtocolRuntime,
    Quantity,
    Reference,
    Setup,
    Stop,
    Wait,
)
from physioworld.experiments.protocol import Calibration as CalibrationNode
from physioworld.physiology import DataRequest, PhysiologyEngine


def main() -> None:
    engine = PhysiologyEngine(
        seed=19,
        requests=DataRequest(
            variables=("heart_rate_bpm", "mean_pressure_mmhg", "oxygen_delivery_ml_min")
        ),
    )
    calibrated: dict[str, Calibration] = {}

    def observe(context) -> None:
        observation = engine.observe()
        context.variables.update(observation.values)
        context.provenance.append(
            {
                "kind": "physiology_observation",
                "time_s": engine.time_s,
                "evidence_class": observation.evidence_class,
                "units": dict(observation.units),
            }
        )

    def setup(node, context):
        context.variables["simulation"] = "reduced_order_physiology"

    def calibration(node, context):
        artifact = Calibration.linear(
            tuple(node.parameters["points"]),
            node.input_unit,
            node.output_unit,
            node.parameters["reference_id"],
        )
        calibrated[node.channel] = artifact
        context.provenance.append(
            {
                "kind": "calibration",
                "channel": node.channel,
                "method": artifact.method,
                "reference_id": artifact.reference_id,
                "input_unit": artifact.input_unit,
                "output_unit": artifact.output_unit,
                "evidence_class": "DECLARED_CALIBRATION",
            }
        )

    def baseline(node, context):
        engine.advance(node.duration_s)
        observe(context)

    def intervention(node, context):
        parameters = dict(node.parameters)
        engine.act(node.kind, **parameters)
        observe(context)

    def wait(node, context):
        engine.advance(node.duration_s)
        observe(context)

    def measurement(node, context):
        return context.resolve(node.source)

    def computation(node, context):
        heart_rate = context.resolve(node.inputs[0])
        pressure = context.resolve(node.inputs[1])
        return heart_rate.value / pressure.value

    def export(node, context):
        # The injected sink is intentionally a manifest marker; no unrequested
        # file or hardware export is implied by a declarative export node.
        context.provenance.append(
            {
                "kind": "export_requested",
                "target": node.target,
                "format": node.format,
                "evidence_class": "SIMULATION_TRACE",
            }
        )

    def no_op(node, context):
        return None

    handlers = HandlerRegistry(
        {
            "setup": setup,
            "calibration": calibration,
            "baseline": baseline,
            "intervention": intervention,
            "wait": wait,
            "measurement": measurement,
            "computation": computation,
            "event": no_op,
            "export": export,
            "stop": no_op,
        }
    )

    protocol = Protocol(
        protocol_id="reduced-order-cardiorespiratory-lab",
        horizon_s=10,
        seed=19,
        metadata={
            "purpose": "demonstrate declarative simulation protocol execution",
            "evidence_class": "SIMULATION",
            "hardware_measurement": False,
        },
        nodes=(
            Setup("setup", {"subject": "synthetic-adult"}),
            CalibrationNode(
                "pressure-calibration",
                "pressure-sensor",
                "count",
                "mmHg",
                parameters={"points": ((0, 0), (100, 100)), "reference_id": "declared-demo-points"},
            ),
            Event("baseline-onset", "baseline"),
            # The baseline node advances the explicit protocol and physiology clocks together.
            Baseline("baseline", 1.0),
            Intervention(
                "exercise",
                "exercise",
                {"exercise_mets": 2.0},
                {"exercise_mets": "1"},
            ),
            Wait("recovery-window", 1.0),
            Measurement("heart-rate", "heart_rate_measurement", "heart_rate_bpm", "bpm"),
            Measurement("mean-pressure", "mean_pressure_measurement", "mean_pressure_mmhg", "mmHg"),
            Computation(
                "derived-shock-index",
                "shock_index",
                "bpm/mmHg",
                inputs=(
                    "measurement.heart_rate_measurement",
                    "measurement.mean_pressure_measurement",
                ),
            ),
            If(
                "safety-branch",
                Condition(Reference("measurement.shock_index"), ">", Quantity(1.0, "bpm/mmHg")),
                then_nodes=(Event("elevated-index", "elevated-index"),),
                else_nodes=(Event("index-within-demo-range", "index-within-demo-range"),),
            ),
            Export("manifest-export", "protocol-result", "json"),
            Stop("complete", "declared protocol complete"),
        ),
        emergency=(
            Intervention(
                "emergency-rest",
                "control",
                {"exercise_mets": 1.0},
                {"exercise_mets": "1"},
            ),
        ),
    )

    result = ProtocolRuntime(handlers).run(protocol)
    print(
        {
            "status": result.status,
            "time_s": result.time_s,
            "shock_index": result.measurements["shock_index"].to_dict(),
            "calibration": calibrated["pressure-sensor"].method,
            "chronology_entries": len(result.chronology),
            "evidence_class": "SIMULATION",
        }
    )


if __name__ == "__main__":
    main()

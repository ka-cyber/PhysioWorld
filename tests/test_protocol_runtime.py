import json
import unittest
from dataclasses import replace

from physioworld.experiments.protocol import (
    Computation,
    Condition,
    Event,
    ExecutionLimits,
    HandlerContract,
    HandlerRegistry,
    If,
    Intervention,
    Loop,
    Measurement,
    ParameterRef,
    ParameterSweep,
    Protocol,
    ProtocolBuilder,
    ProtocolError,
    ProtocolExecutionError,
    ProtocolRuntime,
    ProtocolValidationError,
    Quantity,
    Reference,
    Schedule,
    ScheduleCase,
    Setup,
    SimulationClock,
    Wait,
)


class ProtocolRuntimeTests(unittest.TestCase):
    def make_handlers(self, calls):
        def handler(node, context):
            calls.append((node.op, node.node_id, context.clock.now_s))
            if isinstance(node, Setup):
                context.variables.update(node.parameters)
            if isinstance(node, Measurement):
                return context.resolve(node.source)
            if isinstance(node, Computation):
                left = context.resolve(node.inputs[0])
                right = context.resolve(node.inputs[1])
                left = left.value if isinstance(left, Quantity) else left
                right = right.value if isinstance(right, Quantity) else right
                return left / right
            return None

        return HandlerRegistry(
            {
                name: handler
                for name in (
                    "setup",
                    "event",
                    "wait",
                    "measurement",
                    "computation",
                    "acquire",
                    "calibration",
                    "baseline",
                    "stimulus",
                    "intervention",
                    "resource",
                    "stop",
                    "export",
                )
            }
        )

    def test_chronology_units_branch_loop_and_derived_computation(self):
        calls = []
        handlers = self.make_handlers(calls)
        protocol = (
            ProtocolBuilder("chronology", horizon_s=10, seed=9)
            .setup(
                "values",
                {"heart_rate": 80.0, "pressure": 40.0},
                units={"heart_rate": "bpm", "pressure": "mmHg"},
            )
            .add(
                Measurement(
                    "hr", "heart_rate_measurement", "heart_rate", "bpm", handler="measurement"
                )
            )
            .add(
                Measurement(
                    "pressure", "pressure_measurement", "pressure", "mmHg", handler="measurement"
                )
            )
            .add(
                Computation(
                    "index",
                    "shock_index",
                    "bpm/mmHg",
                    inputs=(
                        "measurement.heart_rate_measurement",
                        "measurement.pressure_measurement",
                    ),
                )
            )
            .add(
                If(
                    "branch",
                    Condition(Reference("measurement.shock_index"), ">", Quantity(1, "bpm/mmHg")),
                    then_nodes=(Event("high", "high-index"),),
                    else_nodes=(Event("low", "low-index"),),
                )
            )
            .add(Loop("repeat", 2, (Wait("pause", 0.5), Event("beat", "beat"))))
            .build()
        )

        result = ProtocolRuntime(handlers).run(protocol)
        self.assertEqual(result.status, "completed")
        self.assertAlmostEqual(result.time_s, 1.0)
        self.assertAlmostEqual(result.measurements["shock_index"].value, 2.0)
        self.assertEqual(
            [event["label"] for event in result.events], ["high-index", "beat", "beat"]
        )
        dispatch_times = [
            item["time_s"] for item in result.chronology if item.get("kind") == "dispatch"
        ]
        self.assertEqual(dispatch_times, sorted(dispatch_times))
        self.assertEqual(result.measurements["shock_index"].unit, "bpm/mmHg")

    def test_seeded_randomized_schedule_is_reproducible_and_counterbalanced(self):
        def run(seed, mode="randomized"):
            protocol = Protocol(
                "schedule",
                10,
                (
                    Schedule(
                        "order",
                        (
                            ScheduleCase("a", (Event("a-event", "a"),)),
                            ScheduleCase("b", (Event("b-event", "b"),)),
                            ScheduleCase("c", (Event("c-event", "c"),)),
                        ),
                        mode=mode,
                        repetitions=2,
                    ),
                ),
                seed=seed,
            )
            return ProtocolRuntime(self.make_handlers([])).run(protocol)

        left, right = run(12), run(12)
        self.assertEqual(left.to_dict(), right.to_dict())
        self.assertNotEqual(
            [item["order"] for item in left.provenance if item["kind"] == "schedule"],
            [["a", "b", "c"], ["a", "b", "c"]],
        )
        balanced = run(1, "counterbalanced")
        orders = [item["order"] for item in balanced.provenance if item["kind"] == "schedule"]
        self.assertEqual(orders, [["b", "c", "a"], ["c", "a", "b"]])

    def test_parameter_sweep_is_bounded_and_materializes_only_data(self):
        protocol = Protocol(
            "sweep",
            10,
            (
                ParameterSweep(
                    "sweep",
                    {"level": (1, 2)},
                    (Event("marker", "level", attributes={"level": ParameterRef("level")}),),
                ),
            ),
            seed=3,
        )
        result = ProtocolRuntime(self.make_handlers([])).run(protocol)
        self.assertEqual([event["attributes"]["level"] for event in result.events], [1, 2])
        self.assertNotIn("eval", json.dumps(result.to_dict()))
        limited = replace(protocol, limits=ExecutionLimits(max_sweep_cases=1))
        with self.assertRaises(ProtocolExecutionError) as raised:
            ProtocolRuntime(self.make_handlers([])).run(limited)
        self.assertIn("max_sweep_cases", str(raised.exception))

    def test_abort_runs_explicit_emergency_fallback_and_records_it(self):
        calls = []

        def abort(node, context):
            calls.append(node.node_id)
            context.abort("operator abort")

        def safe(node, context):
            calls.append(node.node_id)

        protocol = Protocol(
            "abort",
            5,
            (Event("abort-now", "abort", handler="abort"),),
            emergency=(Event("safe-output", "safe", handler="safe"),),
        )
        result = ProtocolRuntime(
            HandlerRegistry({"abort": abort, "safe": safe, "event": safe})
        ).run(protocol)
        self.assertEqual(result.status, "aborted")
        self.assertEqual(calls, ["abort-now", "safe-output"])
        self.assertEqual(
            [p["status"] for p in result.provenance if p["kind"] == "emergency"],
            ["started", "completed"],
        )

    def test_no_future_clock_and_action_are_rejected_with_failure_trace(self):
        clock = SimulationClock(2)
        clock.advance_to(1)
        with self.assertRaises(ProtocolError):
            clock.advance_to(0.5)
        protocol = Protocol(
            "past", 3, (Event("future", "future", at_s=2), Event("past", "past", at_s=1))
        )
        with self.assertRaises(ProtocolExecutionError) as raised:
            ProtocolRuntime(self.make_handlers([])).run(protocol)
        self.assertIn("past", str(raised.exception))
        self.assertEqual(raised.exception.result.status, "failed")

    def test_interventions_are_explicitly_timed_and_recorded_as_actions(self):
        protocol = Protocol(
            "action-trace",
            2,
            (
                Intervention(
                    "dose", "control", {"exercise_mets": 2.0}, {"exercise_mets": "1"}, at_s=0.5
                ),
            ),
        )
        result = ProtocolRuntime(self.make_handlers([])).run(protocol)
        self.assertEqual(result.actions[0]["time_s"], 0.5)
        self.assertEqual(result.actions[0]["kind"], "control")
        self.assertEqual(result.actions[0]["evidence_class"], "DECLARED_ACTION")

    def test_strict_schema_unknown_ops_fields_versions_and_handler_contracts(self):
        raw = ProtocolBuilder("json", horizon_s=2).add(Event("mark", "mark")).build().to_dict()
        restored = Protocol.from_dict(raw)
        self.assertEqual(restored.to_dict(), raw)
        with self.assertRaises(ProtocolValidationError):
            Protocol.from_dict({**raw, "schema_version": "2.0.0"})
        bad_node = dict(raw["nodes"][0])
        bad_node["op"] = "arbitrary_python"
        with self.assertRaises(ProtocolValidationError):
            Protocol.from_dict({**raw, "nodes": [bad_node]})
        bad_field = dict(raw["nodes"][0])
        bad_field["unknown"] = True
        with self.assertRaises(ProtocolValidationError):
            Protocol.from_dict({**raw, "nodes": [bad_field]})
        handlers = HandlerRegistry(
            {"event": lambda node, context: None},
            contracts={
                "event": HandlerContract("event", required_parameters=("missing",)),
            },
        )
        with self.assertRaises(ProtocolExecutionError):
            ProtocolRuntime(handlers).run(Protocol("contract", 1, (Event("mark", "mark"),)))

    def test_handler_source_is_not_a_protocol_language(self):
        with self.assertRaises(ProtocolValidationError):
            Protocol("no-code", 1, (Event("x", "x", handler="__import__('os').system('bad')"),))


if __name__ == "__main__":
    unittest.main()

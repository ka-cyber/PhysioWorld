import json
import unittest
from dataclasses import replace

from physioworld.communication import ChannelResult, Packet, TransmissionEvent
from physioworld.core import OperatingEnvelope
from physioworld.core.contracts import EvidenceClass
from physioworld.devices import QualificationCheck, QualificationReport
from physioworld.experiments import ExecutionResult, ExperimentResult
from physioworld.instrumentation import (
    ExperimentObservation,
    TraceStatus,
    build_inspection,
    render_inspection,
)
from physioworld.provenance import create_manifest
from physioworld.version import __version__


def result(result_id: str = "result") -> ExperimentResult:
    return ExperimentResult(
        result_id,
        __version__,
        None,
        "cardio",
        {"id": "synthetic"},
        {"id": "hypotension", "horizon_s": 1.0},
        "task_value",
        {"id": "model", "source": "reference"},
        {"name": "finite_history", "window": 2},
        {"deadline_s": 0.2},
        {"model": "capacitor", "v_min": 2.0},
        {"class": "threshold", "fallback_action": "abstain"},
        4,
        {"method": "seeded_simulation"},
        EvidenceClass.SIMULATION,
        OperatingEnvelope("env", "synthetic local run", {"time_s": (0.0, 1.0)}),
        {"configuration_hash": "declared", "source": "synthetic", "limitations": ["in sample"]},
        {
            "rows": [
                {
                    "acquisition_time_s": 0.0,
                    "decision_time_s": 0.1,
                    "deadline_s": 0.2,
                    "endpoint_time_s": 1.0,
                    "representation": [1.0, 2.0],
                    "quantized_representation": [1.0, 2.0],
                    "received_representation": [1.0, 2.0],
                    "representation_metadata": {"unit": "mV"},
                    "endpoint": 1,
                    "endpoint_status": "OBSERVED",
                    "available": True,
                    "unavailability_reason": None,
                    "fallback_action": None,
                },
                {
                    "acquisition_time_s": 1.0,
                    "decision_time_s": 1.4,
                    "deadline_s": 1.2,
                    "endpoint_time_s": None,
                    "representation": [3.0, 4.0],
                    "quantized_representation": [3.0, 4.0],
                    "received_representation": None,
                    "representation_metadata": {"unit": "mV"},
                    "endpoint": None,
                    "endpoint_status": "INCOMPLETE_FOLLOW_UP",
                    "available": False,
                    "unavailability_reason": "channel_loss_or_deadline",
                    "fallback_action": "abstain",
                },
            ],
            "transport": {"sent_packets": 2, "delivered_packets": 1, "wire_bytes": 10},
            "resource": {
                "initial_energy_j": 1.0,
                "final_energy_j": 0.8,
                "transitions": [
                    {
                        "row": 0,
                        "start_s": 0.0,
                        "duration_s": 0.1,
                        "feasible": True,
                        "reason": None,
                    }
                ],
            },
            "capability": {
                "selected_policy": "threshold:1",
                "selected_actions": ["positive"],
                "endpoint_task_value": 1.0,
            },
        },
    )


class ObservabilityTests(unittest.TestCase):
    def test_projects_existing_result_traces_and_preserves_not_recorded_state(self):
        packet_a = Packet(10, b"a", timestamp=0.0, metadata={"row": 0})
        packet_b = Packet(20, b"b", timestamp=1.0, metadata={"row": 1})
        channel = ChannelResult(
            (packet_a, packet_b),
            (packet_a,),
            (20,),
            delays={10: 0.1},
            arrival_times={10: 0.1},
            wire_bytes=10,
            events=(
                TransmissionEvent(10, 0, 0.0, 0.1, "delivered", packet_a),
                TransmissionEvent(20, 0, 1.0, None, "lost"),
            ),
        )
        protocol = ExecutionResult(
            "protocol",
            "completed",
            1.0,
            4,
            {},
            {},
            (),
            (),
            ({"kind": "decision", "time_s": 0.5, "action": "positive"},),
            ({"kind": "protocol", "source": "test"},),
        )
        qualification = QualificationReport(
            "simulator",
            (QualificationCheck("DEV-ID", True, "simulator", "reference"),),
        )
        inspection = build_inspection(
            [
                ExperimentObservation(
                    result(), channel=channel, protocol=protocol, qualification=qualification
                )
            ]
        )
        self.assertEqual(inspection.selected_run_id, "result")
        selected = inspection.selected
        assert selected is not None
        self.assertEqual(selected.packets.trace_status, TraceStatus.RECORDED)
        self.assertEqual([item.status for item in selected.packets.attempts], ["delivered", "lost"])
        self.assertEqual(selected.representations.samples[0].received, (1.0, 2.0))
        self.assertEqual(selected.policy.decisions[0].action, "positive")
        self.assertIsNone(selected.policy.decisions[1].action)
        self.assertEqual(selected.resources.transitions[0].row_index, 0)
        self.assertEqual(selected.qualification.source, "device")
        self.assertEqual(selected.provenance.integrity_status, "NOT_VERIFIED")
        self.assertGreaterEqual(len(selected.timeline), 5)
        bare = build_inspection([ExperimentObservation(result())]).selected
        assert bare is not None
        self.assertEqual(bare.packets.trace_status, TraceStatus.NOT_RECORDED)

    def test_catalog_selection_manifest_provenance_and_strict_json(self):
        manifest = create_manifest("run-b", {"seed": 4}, seed=4, outputs={})
        manifest_result = replace(result("b"), provenance={})
        inspection = build_inspection(
            [
                ExperimentObservation(manifest_result, manifest=manifest),
                ExperimentObservation(result("a")),
            ],
            selected_run_id="run-b",
        )
        self.assertEqual([item.run_id for item in inspection.experiments], ["a", "run-b"])
        self.assertEqual(inspection.selected.summary.run_id, "run-b")  # type: ignore[union-attr]
        self.assertEqual(inspection.selected.provenance.manifest["run_id"], "run-b")  # type: ignore[union-attr]
        payload = json.loads(render_inspection(inspection, format="json"))
        self.assertEqual(payload["schema_version"], "1.0.0")
        self.assertEqual(payload["record_type"], "LabInspection")
        self.assertIn("RECORDED", render_inspection(inspection))
        self.assertNotIn("NaN", render_inspection(inspection, format="json"))

    def test_rejects_ambiguous_selection_and_misaligned_policy_trace(self):
        with self.assertRaisesRegex(ValueError, "unknown selected run"):
            build_inspection([ExperimentObservation(result())], selected_run_id="other")
        bad = result("bad")
        bad_outputs = dict(bad.outputs)
        capability = dict(bad_outputs["capability"])
        capability["selected_actions"] = ["positive", "negative"]
        bad_outputs["capability"] = capability
        bad = ExperimentResult(
            bad.result_id,
            bad.software_version,
            bad.commit,
            bad.experiment_id,
            bad.dataset_manifest,
            bad.task,
            bad.estimand,
            bad.model,
            bad.representation,
            bad.channel,
            bad.resource,
            bad.policy,
            bad.seed,
            bad.uncertainty,
            bad.evidence_class,
            bad.operating_envelope,
            bad.provenance,
            bad_outputs,
        )
        with self.assertRaisesRegex(ValueError, "align"):
            build_inspection([ExperimentObservation(bad)])

    def test_renderer_escapes_terminal_control_bytes(self):
        unsafe = replace(result(), provenance={"source": "\x1b[31m\nresearch"})
        text = render_inspection(build_inspection([ExperimentObservation(unsafe)]))
        self.assertNotIn("\x1b", text)
        self.assertIn("\\u001b[31m\\nresearch", text)


if __name__ == "__main__":
    unittest.main()

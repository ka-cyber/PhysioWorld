"""Execution-layer qualification tests for the continuation slice."""

import tempfile
import unittest
from pathlib import Path

from physioworld.cli import (
    qualify_install,
    qualify_plugin_command,
    reproduce_manifest,
    run_declared_experiment,
)
from physioworld.experiments import ExperimentResult
from physioworld.verticals import run_vertical


class ExecutionQualificationTests(unittest.TestCase):
    def test_verticals_are_deterministic_and_have_separate_transport_semantics(self):
        first = run_vertical("cardiovascular", seed=4).to_dict()
        second = run_vertical("cardiovascular", seed=4).to_dict()
        self.assertEqual(first, second)
        self.assertIn("packet_delivery", first["transport"])
        self.assertIn("task_law_discrepancy", first["semantic"])
        self.assertEqual(first["evidence_class"], "SIMULATION")

    def test_declared_examples_execute_and_reproduce(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cardio"
            result = run_declared_experiment(
                str(root / "examples" / "cardio_world.json"), str(output)
            )
            self.assertEqual(result["vertical_id"], "cardiovascular")
            replay = Path(directory) / "replay"
            reproduced = reproduce_manifest(str(output / "result_manifest.json"), str(replay))
            self.assertEqual(reproduced["status"], "REPRODUCED")
            self.assertEqual(reproduced["result"], result)
            self.assertEqual(
                reproduced["comparison"]["comparison"], "EXACT_SCIENTIFIC_OUTPUT_HASHES"
            )
            self.assertIn("EXACT_SCIENTIFIC_OUTPUT_HASHES", (replay / "report.md").read_text())

    def test_plugin_qualification_does_not_equate_installation_with_validation(self):
        result = qualify_plugin_command("builtin:verticals")
        self.assertFalse(result["passed"])
        self.assertEqual(result["assurance_tier"], "INTERFACE_VERIFIED")
        self.assertEqual(result["evidence_status"], "SOFTWARE_INTERFACE_VERIFIED")

    def test_installation_qualification_is_explicit(self):
        result = qualify_install()
        self.assertEqual(result["status"], "PASS", result)

    def test_result_object_roundtrip(self):
        from physioworld.core import OperatingEnvelope
        from physioworld.core.contracts import EvidenceClass

        result = ExperimentResult(
            "result",
            "0.1.0",
            None,
            "vertical",
            {"id": "synthetic"},
            {"id": "task"},
            "task_value",
            {"id": "model"},
            {"id": "representation"},
            {"id": "channel"},
            {"id": "resource"},
            {"id": "policy"},
            0,
            {"method": "none"},
            EvidenceClass.SIMULATION,
            OperatingEnvelope("env", "synthetic", {"time_s": (0, 12)}),
            {"source": "test"},
            {"value": 1},
        )
        self.assertEqual(ExperimentResult.from_json(result.to_json()), result)


if __name__ == "__main__":
    unittest.main()

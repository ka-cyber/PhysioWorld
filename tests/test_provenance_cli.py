import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from physioworld.cli import clinical_validate, doctor, run_declared_experiment, simulate
from physioworld.experiments import ExperimentValidationError
from physioworld.provenance import ManifestVerificationError, load_manifest


class ProvenanceTests(unittest.TestCase):
    def test_simulation_writes_manifest_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            output = Path(directory) / "run"
            config.write_text(
                json.dumps(
                    {
                        "seed": 2,
                        "modality": "ECG",
                        "sample_rate": 2,
                        "unit": "mV",
                        "samples": [0, 1],
                    }
                )
            )
            simulate(str(config), str(output))
            manifest = load_manifest(output)
            self.assertEqual(manifest.seed, 2)
            self.assertIn("summary", manifest.output_hashes)
            self.assertTrue((output / "configuration.json").exists())
            self.assertTrue((output / "report.md").exists())

    def test_provenance_command_uses_manifest_to_dict(self):
        from physioworld.cli import main

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            output = Path(directory) / "run"
            config.write_text(
                json.dumps(
                    {
                        "seed": 2,
                        "modality": "ECG",
                        "sample_rate": 2,
                        "unit": "mV",
                        "samples": [0, 1],
                    }
                )
            )
            simulate(str(config), str(output))
            stream = io.StringIO()
            with redirect_stdout(stream):
                self.assertEqual(main(["provenance", str(output)]), 0)
            payload = json.loads(stream.getvalue())
            self.assertEqual(payload["manifest_version"], 1)
            self.assertIsInstance(payload["configuration"], dict)

    def test_doctor_has_no_import_errors(self):
        result = doctor()
        self.assertNotEqual(result["status"], "ERROR", result)

    def test_dataset_run_does_not_fake_a_clinical_result(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            output = Path(directory) / "run"
            config.write_text(
                json.dumps(
                    {
                        "dataset": "mimic_waveform",
                        "task": {
                            "id": "hypotension",
                            "horizon_s": 300,
                            "target": "future_map",
                            "metric": "auprc",
                        },
                        "representation": "raw",
                        "perturbations": [],
                        "channel": {},
                        "resource": {},
                        "compute": {},
                        "policy": {},
                        "seed": 1,
                        "split": "patient_temporal_test",
                        "metrics": ["auprc"],
                        "outputs": {},
                    }
                )
            )
            result = run_declared_experiment(str(config), str(output))
            self.assertEqual(result["status"], "CONFIGURATION_ONLY")
            manifest = load_manifest(output)
            self.assertEqual(manifest.status, "PARTIAL")

    def test_vertical_configuration_mutation_changes_cli_scientific_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            source = json.loads((root / "examples" / "cardio_world.json").read_text())
            first = Path(directory) / "first"
            second_config = Path(directory) / "changed.json"
            second = Path(directory) / "second"
            run_declared_experiment(str(root / "examples" / "cardio_world.json"), str(first))
            source["channel"]["loss_probability"] = 1.0
            second_config.write_text(json.dumps(source))
            changed = run_declared_experiment(str(second_config), str(second))
            original = json.loads((first / "result.json").read_text())
            self.assertNotEqual(original["outputs"]["transport"], changed["outputs"]["transport"])
            self.assertNotEqual(
                load_manifest(first).configuration_hash, load_manifest(second).configuration_hash
            )

    def test_malformed_configuration_rejects_before_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.json"
            output = Path(directory) / "run"
            config.write_text(
                json.dumps({"seed": 1, "modality": "ECG", "unit": "mV", "samples": []})
            )
            with self.assertRaises(ExperimentValidationError):
                simulate(str(config), str(output))
            self.assertFalse(output.exists())

    def test_tampered_manifest_rejects_replay_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            output = Path(directory) / "run"
            config.write_text(
                json.dumps(
                    {
                        "seed": 2,
                        "modality": "ECG",
                        "sample_rate": 2,
                        "unit": "mV",
                        "samples": [0, 1],
                    }
                )
            )
            simulate(str(config), str(output))
            manifest = output / "result_manifest.json"
            payload = json.loads(manifest.read_text())
            payload["configuration"]["seed"] = 99
            manifest.write_text(json.dumps(payload))
            with self.assertRaises(ManifestVerificationError):
                from physioworld.cli import reproduce_manifest

                reproduce_manifest(str(manifest), str(Path(directory) / "replay"))

    def test_schema_command_uses_bundled_resources(self):
        from physioworld.cli import main

        self.assertEqual(main(["schema", "vertical.schema.json"]), 0)

    def test_clinical_validation_freezes_protocol_without_claiming_performance(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            output = Path(directory) / "clinical"
            config.write_text(
                json.dumps(
                    {
                        "dataset": "mitdb",
                        "intended_use": "arrhythmia rhythm annotation research evaluation",
                        "population": "adult MIT-BIH records meeting declared eligibility",
                        "data_provenance": "MIT-BIH Arrhythmia Database release 1.0",
                        "inclusion_exclusion": "include records with reference rhythm annotations; exclude incomplete records",
                        "reference_standard": "expert rhythm annotations from the database release",
                        "endpoint": "rhythm annotation at the record level",
                        "analysis_population": "all eligible records with complete endpoint annotations",
                        "missing_data": "exclude records missing the prespecified reference annotation",
                        "model": "raw-signal rhythm classifier frozen before test evaluation",
                        "threshold": "macro_f1 threshold and scoring rule prespecified before evaluation",
                        "dataset_version": "MIT-BIH release 1.0",
                        "task": {
                            "id": "arrhythmia",
                            "horizon_s": 0,
                            "target": "rhythm",
                            "metric": "f1",
                        },
                        "representation": "raw",
                        "perturbations": [],
                        "channel": {"sampling": "360 Hz", "transport": "local"},
                        "resource": {
                            "device": "reference workstation",
                            "power_mode": "unconstrained",
                        },
                        "compute": {},
                        "policy": {},
                        "seed": 1,
                        "split": "subject_test",
                        "metrics": ["f1"],
                        "outputs": {},
                    }
                )
            )
            result = clinical_validate(str(config), str(output))
            self.assertEqual(result["status"], "CONFIGURATION_ONLY")
            self.assertEqual(result["evaluation_scope"], "EXTERNAL_PREDICTIVE_EVALUATION")
            self.assertEqual(result["external_predictive_evaluation_status"], "NOT_EVALUATED")
            self.assertEqual(result["clinical_validation_status"], "NOT_ESTABLISHED")
            self.assertIsNone(result["performance"])


if __name__ == "__main__":
    unittest.main()

"""Path ownership and explicit CLI configuration routing regressions."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from physioworld.cli import (
    clinical_validate,
    main,
    replay,
    run_declared_experiment,
    run_vertical_experiment,
    simulate,
    validate_config,
)
from physioworld.experiments import ExperimentCompiler, ExperimentManager, ExperimentValidationError
from physioworld.experiments.compiler import configuration_schema_name
from physioworld.provenance import verify_manifest

ROOT = Path(__file__).resolve().parents[1]
CORPUS_CONFIG = ROOT / "configs" / "corpus" / "benchmark_corpus_level3.json"
CORPUS_FAMILY = "native_level3_synthetic"


def sample_configuration():
    return {
        "seed": 2,
        "modality": "ECG",
        "unit": "mV",
        "sample_rate": 2,
        "samples": [0, 1],
    }


def legacy_configuration():
    return {
        "dataset": "mitdb",
        "task": {"id": "rhythm", "target": "annotation", "horizon_s": 0, "metric": "f1"},
        "representation": "raw",
        "perturbations": [],
        "channel": {},
        "resource": {},
        "compute": {},
        "policy": {},
        "seed": 1,
        "split": "subject_test",
        "metrics": ["f1"],
        "outputs": {},
    }


class PathConfigCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(sample_configuration()), encoding="utf-8")
        self.output = self.root / "run"

    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_manager_rejects_unsafe_names_without_computation_or_writes(self):
        function = Mock(return_value={"value": 1})
        output_root = self.root / "results"
        for name in (
            "",
            " ",
            ".",
            "..",
            "../escape",
            "nested/run",
            "./run",
            "run/",
            str(self.root / "escape"),
            "nested\\run",
            "C:/escape",
            "C:escape",
            "\\\\server\\run",
            "nul\x00name",
            None,
            False,
            2,
        ):
            configuration = {**sample_configuration(), "name": name}
            self.config.write_text(json.dumps(configuration), encoding="utf-8")
            with self.subTest(name=name), self.assertRaises(ExperimentValidationError):
                ExperimentManager(output_root).run(self.config, function)
            function.assert_not_called()
            self.assertFalse(output_root.exists())
            self.assertFalse((self.root / "escape").exists())

    def test_manager_checks_filename_fallback_as_well_as_declared_names(self):
        path = self.root / "C:escape.json"
        path.write_text(json.dumps(sample_configuration()), encoding="utf-8")
        function = Mock()
        with self.assertRaisesRegex(ExperimentValidationError, "safe path component"):
            ExperimentManager(self.root / "results").run(path, function)
        function.assert_not_called()
        self.assertFalse((self.root / "results").exists())

    def test_manager_rejects_occupied_output_before_computation(self):
        for occupied in ("directory", "file"):
            output_root = self.root / occupied
            output_root.mkdir()
            target = output_root / self.config.stem
            if occupied == "directory":
                target.mkdir()
                sentinel = target / "metrics.json"
            else:
                sentinel = target
            sentinel.write_bytes(b"caller-owned")
            function = Mock()
            with self.subTest(occupied=occupied), self.assertRaises(FileExistsError):
                ExperimentManager(output_root).run(self.config, function)
            function.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"caller-owned")
            if target.is_dir():
                self.assertEqual(list(target.iterdir()), [sentinel])

    def test_manager_rejects_symlink_paths_before_computation(self):
        outside = self.root / "outside"
        outside.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "absent", target_is_directory=True)
        for output_root in (alias, dangling):
            function = Mock()
            with (
                self.subTest(output_root=output_root),
                self.assertRaisesRegex(ValueError, "symlink"),
            ):
                ExperimentManager(output_root).run(self.config, function)
            function.assert_not_called()
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.root / "absent").exists())

    def test_manager_writes_verifiable_artifacts_to_a_fresh_directory(self):
        for declared_name in (None, "named-run"):
            configuration = sample_configuration()
            if declared_name is not None:
                configuration["name"] = declared_name
            self.config.write_text(json.dumps(configuration), encoding="utf-8")
            manager = ExperimentManager(self.root / "results")
            function = Mock(return_value={"value": 3})
            run = manager.run(self.config, function)
            with self.subTest(name=declared_name):
                self.assertEqual(run.run_id, declared_name or "config")
                self.assertEqual(run.result, {"value": 3})
                verified = verify_manifest(run.output_directory)
                self.assertEqual(verified.manifest.run_id, run.run_id)
                self.assertEqual(verified.manifest.scientific_outputs, ("metrics",))
                with self.assertRaises(FileExistsError):
                    manager.run(self.config, function)
                self.assertEqual(function.call_count, 1)

    def test_manager_rechecks_freshness_after_computation(self):
        manager = ExperimentManager(self.root / "results")
        target = manager.output_root / self.config.stem

        def occupy_output(configuration):
            target.mkdir(parents=True)
            (target / "metrics.json").write_bytes(b"new-owner")
            return {"value": 3}

        with self.assertRaises(FileExistsError):
            manager.run(self.config, occupy_output)
        self.assertEqual((target / "metrics.json").read_bytes(), b"new-owner")
        self.assertEqual([path.name for path in target.iterdir()], ["metrics.json"])

    def test_execution_routes_reject_occupied_outputs_before_scientific_work(self):
        self.output.mkdir()
        sentinel = self.output / "sentinel.txt"
        sentinel.write_bytes(b"caller-owned")
        with (
            patch("physioworld.cli.SignalStream.from_samples") as samples,
            patch("physioworld.cli.run_vertical") as vertical,
            patch("physioworld.cli.validation_readiness") as readiness,
        ):
            for runner in (simulate, replay, run_declared_experiment, clinical_validate):
                with self.subTest(runner=runner.__name__), self.assertRaises(FileExistsError):
                    runner(str(self.config), str(self.output))
            configuration = json.loads((ROOT / "examples" / "cardio_world.json").read_text())
            with self.assertRaises(FileExistsError):
                run_vertical_experiment(configuration, str(self.output))
            samples.assert_not_called()
            vertical.assert_not_called()
            readiness.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"caller-owned")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_cli_rejects_file_symlink_ancestor_and_traversal_before_computation(self):
        file = self.root / "file"
        file.write_bytes(b"caller-owned")
        outside = self.root / "outside"
        outside.mkdir()
        alias = self.root / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "absent", target_is_directory=True)
        for output in (
            file,
            file / "child",
            alias,
            alias / "child",
            dangling,
            self.root / ".." / "escape",
        ):
            with (
                self.subTest(output=output),
                patch("physioworld.cli.SignalStream.from_samples") as work,
            ):
                code, stdout, stderr = self.invoke(
                    ["simulate", str(self.config), "--output", str(output)]
                )
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn("execution error", stderr)
                work.assert_not_called()
        self.assertEqual(file.read_bytes(), b"caller-owned")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.root / "absent").exists())

    def test_legacy_benchmarks_reject_ignored_config_and_output_options(self):
        families = (
            None,
            "representation",
            "task-law",
            "communication",
            "resource",
            "policy",
            "clinical",
            "causal",
            "robustness",
        )
        options = (
            ["--config", str(self.root / "missing.json")],
            ["--output", str(self.output)],
            ["--config", str(self.config), "--output", str(self.output)],
        )
        with (
            patch("physioworld.cli.run_benchmark_family") as family_runner,
            patch("physioworld.cli.run_smoke_benchmark") as smoke,
        ):
            for family in families:
                for flags in options:
                    arguments = ["benchmark"]
                    if family is not None:
                        arguments += ["--family", family]
                    with self.subTest(family=family, flags=flags):
                        code, stdout, stderr = self.invoke(arguments + flags)
                        self.assertEqual(code, 2)
                        self.assertEqual(stdout, "")
                        self.assertIn("require --family native_level3_synthetic", stderr)
            family_runner.assert_not_called()
            smoke.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_legacy_benchmark_calls_without_unsupported_flags_still_work(self):
        for arguments in (["benchmark"], ["benchmark", "--family", "representation"]):
            with self.subTest(arguments=arguments):
                code, stdout, stderr = self.invoke(arguments)
                self.assertEqual(code, 0, stderr)
                self.assertTrue(json.loads(stdout))

    def test_corpus_validation_is_family_aware_and_does_not_execute(self):
        with patch("physioworld.benchmarks.corpus._resource_fixture") as fixtures:
            code, stdout, stderr = self.invoke(
                ["validate", str(CORPUS_CONFIG), "--family", CORPUS_FAMILY]
            )
            self.assertEqual(code, 0, stderr)
            result = json.loads(stdout)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["schema_name"], "corpus.schema.json")
            self.assertEqual(result["configuration"], json.loads(CORPUS_CONFIG.read_text()))
            self.assertEqual(result["execution_status"], "NOT_EXECUTED")
            fixtures.assert_not_called()
        code, stdout, stderr = self.invoke(["validate", str(CORPUS_CONFIG)])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("use validate --family native_level3_synthetic", stderr)

    def test_corpus_validation_rejects_invalid_and_legacy_declarations(self):
        configuration = json.loads(CORPUS_CONFIG.read_text())
        for change in (
            {"clinical_claim": True},
            {"cases": []},
            {"seed": 3},
            {"cases": ["unknown"]},
        ):
            self.config.write_text(json.dumps({**configuration, **change}), encoding="utf-8")
            with self.subTest(change=change):
                code, stdout, stderr = self.invoke(
                    ["validate", str(self.config), "--family", CORPUS_FAMILY]
                )
                self.assertEqual(code, 2)
                self.assertEqual(stdout, "")
                self.assertIn("execution error", stderr)
        self.config.write_text(json.dumps(legacy_configuration()), encoding="utf-8")
        with self.assertRaises(ExperimentValidationError):
            validate_config(str(self.config), family=CORPUS_FAMILY)
        with self.assertRaisesRegex(ExperimentValidationError, "unsupported validation family"):
            validate_config(str(self.config), family="representation")

    def test_native_benchmark_consumes_config_and_output_and_rejects_reuse(self):
        configuration = json.loads(CORPUS_CONFIG.read_text())
        configuration["cases"] = ["resource_p1"]
        self.config.write_text(json.dumps(configuration), encoding="utf-8")
        arguments = [
            "benchmark",
            "--family",
            CORPUS_FAMILY,
            "--config",
            str(self.config),
            "--output",
            str(self.output),
        ]
        code, stdout, stderr = self.invoke(arguments)
        self.assertEqual(code, 0, stderr)
        result = json.loads(stdout)
        self.assertEqual(result["case_count"], 1)
        self.assertEqual(result["cases"][0]["case_id"], "resource_p1")
        manifest_path = self.output / "corpus_manifest.json"
        before = manifest_path.read_bytes()
        self.assertEqual(json.loads(before)["configuration"], configuration)
        with patch("physioworld.benchmarks.corpus._resource_fixture") as work:
            code, stdout, stderr = self.invoke(arguments)
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("must be empty", stderr)
            work.assert_not_called()
        self.assertEqual(manifest_path.read_bytes(), before)

    def test_legacy_compiler_cannot_ignore_corpus_markers(self):
        for field in (
            "corpus_id",
            "execution_profile",
            "fixture_root",
            "seed_policy",
            "external_inputs",
            "cases",
        ):
            for wrapped in (False, True):
                configuration = {**legacy_configuration(), field: None}
                if wrapped:
                    configuration = {"experiment": configuration}
                with self.subTest(field=field, wrapped=wrapped):
                    with self.assertRaisesRegex(ExperimentValidationError, "corpus fields"):
                        ExperimentCompiler().compile(configuration)
                    with self.assertRaisesRegex(ExperimentValidationError, "corpus fields"):
                        configuration_schema_name(configuration)

    def test_examples_have_resolvable_document_relative_schema_references(self):
        for path in sorted((ROOT / "examples").glob("*.json")):
            with self.subTest(example=path.name):
                configuration = json.loads(path.read_text())
                reference = configuration["$schema"]
                self.assertTrue(reference.startswith("../schemas/"))
                schema_path = (path.parent / reference).resolve()
                self.assertTrue(schema_path.is_file())
                plan = ExperimentCompiler().compile_file(path)
                self.assertEqual(plan.schema_name, schema_path.name)


if __name__ == "__main__":
    unittest.main()

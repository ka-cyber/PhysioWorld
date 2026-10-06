"""Stdlib installation checks, independent references and failure controls."""

import copy
import importlib.metadata
import json
import math
import pkgutil
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import physioworld
from physioworld import qualification
from physioworld.experiments import compiler
from physioworld.resources import CapacitorResource
from physioworld.verticals import run_vertical


def find_check(report, check, location):
    return next(
        item for item in report["checks"] if item["check"] == check and item["location"] == location
    )


def find_comparison(check, quantity):
    return next(item for item in check["comparisons"] if item["quantity"] == quantity)


class InstallQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = qualification.qualify_install()

    def test_real_kernels_and_all_verticals_have_executed_evidence(self):
        self.assertEqual(self.report["status"], "PASS", self.report)
        self.assertEqual(self.report["evidence_scope"], "installed_package_numerical_verification")
        self.assertEqual(self.report["source_tests"]["status"], "NOT_RUN")
        json.dumps(self.report, allow_nan=False)
        for name in ("capacitor", "pca", "rlnc"):
            check = find_check(self.report, "numerical_reference", name)
            self.assertEqual(check["status"], "PASS", check)
            self.assertTrue(check["comparisons"])
            for comparison in check["comparisons"]:
                self.assertTrue(comparison["passed"], comparison)
                self.assertLessEqual(comparison["residual"], comparison["tolerance"])
        for name in ("cardiovascular", "neurophysiology", "icu"):
            check = find_check(self.report, "vertical_replay", name)
            self.assertEqual(check["status"], "PASS", check)
            self.assertGreater(check["observed"]["evaluated_atoms"], 0)
            self.assertGreater(check["observed"]["sent_packets"], 0)
            self.assertEqual(check["observed"]["causal_status"], "UNIDENTIFIED")
            self.assertEqual(check["observed"]["energy_status"], "MODEL_DERIVED")
        lost = find_check(self.report, "analytic_replay_reference", "complete_packet_loss")
        discrepancy = find_comparison(lost, "task_law_discrepancy")
        self.assertAlmostEqual(discrepancy["observed"], 4 / 9, places=12)
        self.assertEqual(lost["observed"]["endpoint_labels"], [1, 1, 0])
        self.assertEqual(find_comparison(lost, "endpoint_task_value")["observed"], 0)

    def test_doctor_imports_every_discoverable_module_and_reads_runtime_schemas(self):
        report = qualification.doctor()
        expected = {"physioworld"} | {
            item.name for item in pkgutil.walk_packages(physioworld.__path__, "physioworld.")
        }
        observed = {item["location"] for item in report["checks"] if item["check"] == "import"}
        self.assertEqual(observed, expected)
        self.assertEqual(
            find_check(report, "import", "physioworld.instrumentation.units")["status"], "PASS"
        )
        self.assertEqual(
            find_check(report, "import", "physioworld.devices.transports")["status"], "PASS"
        )
        for name in compiler.schema_names():
            check = find_check(report, "schema_json", name)
            self.assertEqual(check["status"], "PASS", check)
            self.assertGreater(check["observed"]["json_bytes"], 0)
            self.assertEqual(len(check["sha256"]), 64)
        self.assertFalse(
            any(
                item["check"] in {"required_file", "tests", "credential_scan"}
                for item in report["checks"]
            )
        )

    def test_absent_optional_packages_have_exact_nonfatal_capability_groups(self):
        with (
            patch(
                "importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError
            ),
            patch("importlib.util.find_spec") as discovery,
        ):
            report = qualification.doctor()
            qualified = qualification.qualify_install()
        discovery.assert_not_called()
        self.assertEqual(report["status"], "WARNING", report)
        self.assertEqual(qualified["status"], "PASS", qualified)
        groups = {
            item["location"]: item
            for item in report["checks"]
            if item["check"] == "optional_dependency"
        }
        self.assertEqual(
            set(groups),
            {
                "scientific",
                "plot",
                "wfdb",
                "yaml",
                "schema",
                "arrow",
                "hdf5",
                "nwb",
                "dicom",
                "serial",
                "fmi",
            },
        )
        self.assertEqual(groups["scientific"]["missing"], ["numpy", "scipy"])
        self.assertEqual(groups["hdf5"]["missing"], ["h5py", "numpy"])
        self.assertEqual(groups["serial"]["missing"], ["pyserial"])
        self.assertEqual(groups["yaml"]["missing"], ["PyYAML"])
        self.assertTrue(
            all(item["status"] == "WARNING" and not item["required"] for item in groups.values())
        )

    def test_interpreter_compatibility_has_both_bounds(self):
        for version, expected in (
            ((3, 9), "ERROR"),
            ((3, 10), "PASS"),
            ((3, 14), "PASS"),
            ((3, 15), "ERROR"),
            ((4, 0), "ERROR"),
        ):
            with (
                self.subTest(version=version),
                patch.object(sys, "version_info", (*version, 0)),
                patch("platform.python_version", return_value=".".join(map(str, version))),
            ):
                check = find_check(qualification.doctor(), "interpreter", "python")
                self.assertEqual(check["status"], expected)

    def test_package_import_failure_is_reported_and_siblings_continue(self):
        real_import = qualification.importlib.import_module

        def broken_import(name):
            if name == "physioworld.instrumentation.units":
                raise RuntimeError("injected installed module failure")
            return real_import(name)

        with patch.object(qualification.importlib, "import_module", side_effect=broken_import):
            report = qualification.doctor()
        self.assertEqual(report["status"], "ERROR")
        failure = find_check(report, "import", "physioworld.instrumentation.units")
        self.assertEqual(failure["status"], "ERROR")
        self.assertIn("injected installed module failure", failure["message"])
        self.assertEqual(
            find_check(report, "import", "physioworld.communication.rlnc")["status"], "PASS"
        )

    def test_wrong_resource_numerics_retain_observation_and_residual(self):
        real_step = CapacitorResource.step

        def wrong_step(resource, *args, **kwargs):
            result = real_step(resource, *args, **kwargs)
            return replace(result, terminal_voltage=result.terminal_voltage + 0.1)

        with patch.object(CapacitorResource, "step", wrong_step):
            report = qualification.qualify_install()
        check = find_check(report, "numerical_reference", "capacitor")
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(check["status"], "ERROR")
        comparison = find_comparison(check, "terminal_voltage_v")
        self.assertAlmostEqual(comparison["reference"], 1.85)
        self.assertAlmostEqual(comparison["observed"], 1.95)
        self.assertAlmostEqual(comparison["residual"], 0.1)
        self.assertFalse(comparison["passed"])
        with patch("physioworld.resources.canonical_current_limit", return_value=0.0):
            report = qualification.qualify_install()
        self.assertFalse(
            find_comparison(
                find_check(report, "numerical_reference", "capacitor"), "canonical_current_a"
            )["passed"]
        )

    def test_pca_diagonal_proxy_is_rejected_by_eigen_direction(self):
        from physioworld.representations import PCARepresentation

        real_fit = PCARepresentation.fit

        def diagonal_proxy(model, samples):
            real_fit(model, samples)
            model.components_ = ((1.0, 0.0), (0.0, 1.0))
            return model

        with patch.object(PCARepresentation, "fit", diagonal_proxy):
            report = qualification.qualify_install()
        check = find_check(report, "numerical_reference", "pca")
        self.assertEqual(check["status"], "ERROR")
        self.assertFalse(find_comparison(check, "sign_invariant_eigenvector_error")["passed"])

    def test_rlnc_roundtrip_cannot_hide_a_wrong_field_product(self):
        from physioworld.communication.rlnc import GF256

        with patch.object(GF256, "mul", side_effect=lambda a, b: (a * b) % 256):
            report = qualification.qualify_install()
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(find_check(report, "numerical_reference", "rlnc")["status"], "ERROR")

    def test_constant_garbage_vertical_results_do_not_qualify_by_equality(self):
        with patch(
            "physioworld.verticals.run_vertical",
            return_value=SimpleNamespace(to_dict=lambda: {"constant": "garbage"}),
        ):
            report = qualification.qualify_install()
        self.assertEqual(report["status"], "ERROR")
        failures = [
            item
            for item in report["checks"]
            if item["check"] in ("vertical_replay", "analytic_replay_reference")
        ]
        self.assertEqual(len(failures), 5)
        self.assertTrue(all(item["status"] == "ERROR" for item in failures))

    def test_plausible_but_wrong_replay_discrepancy_is_rejected(self):
        def wrong_law(name, **kwargs):
            result = copy.deepcopy(run_vertical(name, **kwargs).to_dict())
            result["outputs"]["semantics"]["task_law_discrepancy"] = 0.0
            return SimpleNamespace(to_dict=lambda: result)

        with patch("physioworld.verticals.run_vertical", side_effect=wrong_law):
            report = qualification.qualify_install()
        self.assertEqual(report["status"], "ERROR")
        check = find_check(report, "analytic_replay_reference", "complete_packet_loss")
        self.assertEqual(check["status"], "ERROR")
        comparison = find_comparison(check, "task_law_discrepancy")
        self.assertEqual(comparison["observed"], 0)
        self.assertAlmostEqual(comparison["reference"], 4 / 9)
        self.assertAlmostEqual(comparison["residual"], 4 / 9)

    def test_nondeterministic_replay_is_rejected(self):
        counter = 0

        def changing_result(name, **kwargs):
            nonlocal counter
            counter += 1
            result = run_vertical(name, **kwargs).to_dict()
            result["provenance"]["injected_replay_counter"] = counter
            return SimpleNamespace(to_dict=lambda: result)

        with patch("physioworld.verticals.run_vertical", side_effect=changing_result):
            report = qualification.qualify_install()
        self.assertEqual(report["status"], "ERROR")
        for name in ("cardiovascular", "neurophysiology", "icu"):
            check = find_check(report, "vertical_replay", name)
            self.assertEqual(check["status"], "ERROR")
            self.assertEqual(find_comparison(check, "paired_result_mismatch")["observed"], 1)

    def test_nonfinite_kernel_observation_fails_and_stays_json_ready(self):
        with patch("physioworld.resources.canonical_current_limit", return_value=math.nan):
            report = qualification.qualify_install()
        self.assertEqual(report["status"], "ERROR")
        check = find_check(report, "numerical_reference", "capacitor")
        comparison = find_comparison(check, "canonical_current_a")
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["observed"], "nan")
        self.assertIsNone(comparison["residual"])
        json.dumps(report, allow_nan=False)

    def test_invalid_parsed_schema_and_missing_resources_fail(self):
        real_load = compiler.load_schema

        def invalid_schema(name):
            document = copy.deepcopy(real_load(name))
            if name == "world.schema.json":
                document["type"] = "not_a_json_schema_type"
            return document

        with patch.object(compiler, "load_schema", side_effect=invalid_schema):
            report = qualification.doctor()
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(find_check(report, "schema_json", "world.schema.json")["status"], "ERROR")
        with patch.object(compiler, "schema_names", return_value=()):
            report = qualification.doctor()
        self.assertEqual(report["status"], "ERROR")
        self.assertIn(
            "vertical.schema.json",
            find_check(report, "bundled_schemas", "physioworld.schemas")["missing"],
        )

    def isolated_reports(self, *, corrupt=False):
        with tempfile.TemporaryDirectory() as directory:
            installed = Path(directory) / "physioworld"
            shutil.copytree(
                Path(qualification.__file__).parent,
                installed,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            if corrupt:
                (installed / "schemas" / "world.schema.json").write_text(
                    "{invalid JSON", encoding="utf-8"
                )
            script = (
                "import json,sys; sys.path.insert(0,sys.argv[1]); "
                "from physioworld.qualification import doctor,qualify_install; "
                "print(json.dumps({'doctor':doctor(),'qualification':qualify_install(),"
                "'optional_loaded':[name for name in "
                "('numpy','scipy','matplotlib','wfdb','yaml','jsonschema','pyarrow','h5py',"
                "'pynwb','pydicom','serial','fmpy') if name in sys.modules]},allow_nan=False))"
            )
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", script, directory],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return json.loads(completed.stdout)

    def test_package_only_installation_works_without_checkout_or_optional_imports(self):
        reports = self.isolated_reports()
        self.assertEqual(reports["doctor"]["status"], "WARNING", reports)
        self.assertEqual(reports["qualification"]["status"], "PASS", reports)
        self.assertEqual(reports["optional_loaded"], [])
        self.assertEqual(reports["qualification"]["source_tests"]["status"], "NOT_RUN")

    def test_actually_corrupt_installed_schema_is_not_a_presence_pass(self):
        reports = self.isolated_reports(corrupt=True)
        for name in ("doctor", "qualification"):
            self.assertEqual(reports[name]["status"], "ERROR")
            check = find_check(reports[name], "schema_json", "world.schema.json")
            self.assertEqual(check["status"], "ERROR")
            self.assertIn("JSONDecodeError", check["message"])


if __name__ == "__main__":
    unittest.main()

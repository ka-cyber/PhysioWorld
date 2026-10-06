"""Public exports, fresh stdlib-only imports, and optional dependency boundaries."""

import builtins
import importlib
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from physioworld.io.optional import MissingOptionalDependency, require
from physioworld.monitoring.reduction import PODDependencyError, fit_pod

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def stdlib_process(script, *arguments):
    """Ignore site packages and PYTHONPATH so installed extras cannot mask failures."""
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            f"import sys; sys.path.insert(0, {str(SOURCE_ROOT)!r})\n" + textwrap.dedent(script),
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


class PublicExportTests(unittest.TestCase):
    def assert_exports(self, package_name, module_name, names):
        package = importlib.import_module(package_name)
        implementation = importlib.import_module(module_name)
        namespace = {}
        exec(f"from {package_name} import *", namespace)
        self.assertEqual(len(package.__all__), len(set(package.__all__)))
        for name in names:
            with self.subTest(package=package_name, name=name):
                self.assertIn(name, package.__all__)
                self.assertIs(getattr(package, name), getattr(implementation, name))
                self.assertIs(namespace[name], getattr(implementation, name))

    def test_task_law_state_exports(self):
        self.assert_exports(
            "physioworld.tasklaws",
            "physioworld.tasklaws.state",
            (
                "TaskLawState",
                "TaskLawSufficiencyDiagnostic",
                "SufficiencyKind",
                "SufficiencyViolation",
                "predictive_intervention_counterexample",
            ),
        )

    def test_nirs_exports_include_semantic_aliases(self):
        self.assert_exports(
            "physioworld.instrumentation",
            "physioworld.instrumentation.nirs",
            (
                "NIRSQuantity",
                "IntensitySignal",
                "OpticalDensitySignal",
                "HemoglobinSignal",
                "Intensity",
                "OpticalDensity",
                "Hemoglobin",
                "RawProxyResult",
                "optical_density",
                "simulate_motion",
                "motion_correct",
                "filter_optical_density",
                "raw_proxy_experiment",
            ),
        )

    def test_protocol_exports_include_error_hierarchy_and_handler_types(self):
        protocol = importlib.import_module("physioworld.experiments.protocol")
        self.assert_exports(
            "physioworld.experiments",
            protocol.__name__,
            (*protocol.__all__, "Handler"),
        )

    def test_vvuq_exports_are_lazy_and_preserve_function_identity(self):
        result = stdlib_process(
            """
            import importlib
            import physioworld.experiments as experiments

            for kind in ('causal', 'communication', 'resource'):
                module_name = f'physioworld.experiments.{kind}_vvuq'
                name = f'run_{kind}_vvuq'
                assert module_name not in sys.modules, module_name
                assert name in experiments.__all__
                assert name in dir(experiments)
                namespace = {}
                exec(f'from physioworld.experiments import {name}', namespace)
                runner = getattr(importlib.import_module(module_name), name)
                assert namespace[name] is runner
                assert getattr(experiments, name) is runner
                assert experiments.__dict__[name] is runner

            assert not hasattr(experiments, 'missing_runner')
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")


class StdlibImportTests(unittest.TestCase):
    def test_every_discoverable_module_imports_without_extras_or_cli_execution(self):
        result = stdlib_process(
            """
            import importlib
            import pkgutil
            import physioworld

            assert all('site-packages' not in path for path in sys.path)
            modules = list(pkgutil.walk_packages(physioworld.__path__, 'physioworld.'))
            assert 'physioworld.__main__' in {module.name for module in modules}
            for module in modules:
                importlib.import_module(module.name)
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_dependency_modules_can_be_imported_first_without_cycles(self):
        modules = (
            "physioworld.tasklaws.state",
            "physioworld.instrumentation.nirs",
            "physioworld.instrumentation.observability",
            "physioworld.physiology.stream",
            "physioworld.devices",
            "physioworld.coupling",
            "physioworld.experiments.protocol",
            "physioworld.experiments.causal_vvuq",
            "physioworld.experiments.communication_vvuq",
            "physioworld.experiments.resource_vvuq",
            "physioworld.causal",
            "physioworld.communication",
            "physioworld.resources",
            "physioworld.verification.numerics",
            "physioworld.monitoring.reduction",
            "physioworld.io.optional",
        )
        for first in modules:
            with self.subTest(first=first):
                result = stdlib_process(
                    """
                    import importlib
                    importlib.import_module(sys.argv[1])
                    for name in ('tasklaws', 'instrumentation', 'experiments'):
                        package = importlib.import_module(f'physioworld.{name}')
                        for exported in package.__all__:
                            getattr(package, exported)
                    """,
                    first,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")

    def test_module_entry_point_forwards_cli_output_and_exit_codes(self):
        script = """
            import runpy
            sys.argv = ['physioworld', *sys.argv[1:]]
            runpy.run_module('physioworld', run_name='__main__', alter_sys=True)
        """
        version = importlib.import_module("physioworld").__version__
        for arguments, code, output in (
            (("--version",), 0, version + "\n"),
            (("version",), 0, version + "\n"),
            (("--help",), 0, "usage: physioworld"),
            (("unknown-command",), 2, ""),
        ):
            with self.subTest(arguments=arguments):
                result = stdlib_process(script, *arguments)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertTrue(result.stdout.startswith(output), result.stdout)
                if code:
                    self.assertIn("invalid choice", result.stderr)

    def test_module_entry_point_propagates_main_return_value(self):
        result = stdlib_process(
            """
            import runpy
            from unittest.mock import patch

            with patch('physioworld.cli.main', return_value=7) as main:
                try:
                    runpy.run_module('physioworld', run_name='__main__', alter_sys=True)
                except SystemExit as exc:
                    assert exc.code == 7, exc.code
                else:
                    raise AssertionError('module entry point did not exit')
                main.assert_called_once_with()
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class OptionalDependencyTests(unittest.TestCase):
    @staticmethod
    def validate_with_jsonschema_error(error):
        from physioworld.benchmarks.corpus import _validate_document

        original_import = builtins.__import__

        def broken_jsonschema(name, *args, **kwargs):
            if name == "jsonschema":
                raise error
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=broken_jsonschema):
            return _validate_document({"value": 1}, {"type": "object"})

    def test_absent_jsonschema_keeps_dependency_free_validation(self):
        missing = ModuleNotFoundError("No module named 'jsonschema'", name="jsonschema")
        self.assertEqual(self.validate_with_jsonschema_error(missing), {"value": 1})

    def test_broken_jsonschema_install_propagates_original_error(self):
        for error in (
            ImportError("validator symbol missing", name="jsonschema"),
            ModuleNotFoundError("missing dependency", name="jsonschema_dependency"),
            ModuleNotFoundError("missing validator module", name="jsonschema.validators"),
        ):
            with self.subTest(error=error), self.assertRaises(type(error)) as caught:
                self.validate_with_jsonschema_error(error)
            self.assertIs(caught.exception, error)

    @staticmethod
    def fit_with_numpy_error(error):
        original_import = builtins.__import__

        def broken_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise error
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=broken_numpy):
            return fit_pod(((1.0, 2.0), (2.0, 3.0)), 1, reference_id="import-boundary")

    def test_absent_numpy_has_actionable_pod_error_and_original_cause(self):
        missing = ModuleNotFoundError("No module named 'numpy'", name="numpy")
        with self.assertRaisesRegex(PODDependencyError, "optional dependency 'numpy'") as caught:
            self.fit_with_numpy_error(missing)
        self.assertIs(caught.exception.__cause__, missing)

    def test_broken_numpy_install_propagates_original_error(self):
        for error in (
            ImportError("NumPy extension could not be loaded", name="numpy"),
            ModuleNotFoundError("missing NumPy extension", name="numpy._core._multiarray_umath"),
            ModuleNotFoundError("missing transitive dependency", name="numpy_dependency"),
            ModuleNotFoundError("missing module without a name"),
        ):
            with self.subTest(error=error), self.assertRaises(type(error)) as caught:
                self.fit_with_numpy_error(error)
            self.assertIs(caught.exception, error)

    def test_optional_loader_translates_only_absent_requested_modules(self):
        for package, missing_name in (("numpy", "numpy"), ("pyarrow.parquet", "pyarrow")):
            missing = ModuleNotFoundError("absent optional package", name=missing_name)
            with (
                self.subTest(package=package),
                patch("physioworld.io.optional.import_module", side_effect=missing),
                self.assertRaises(MissingOptionalDependency) as caught,
            ):
                require(package, "test format", "scientific")
            self.assertIs(caught.exception.__cause__, missing)
            self.assertEqual(caught.exception.package, package.split(".")[0])
            self.assertIn("physioworld[scientific]", str(caught.exception))

    def test_optional_loader_preserves_broken_install_errors(self):
        for error in (
            ImportError("broken optional package", name="numpy"),
            ModuleNotFoundError("missing extension", name="numpy._core"),
            ModuleNotFoundError("missing dependency", name="numpy_dependency"),
            ModuleNotFoundError("unknown missing module"),
        ):
            with (
                self.subTest(error=error),
                patch("physioworld.io.optional.import_module", side_effect=error),
                self.assertRaises(type(error)) as caught,
            ):
                require("numpy", "test format", "scientific")
            self.assertIs(caught.exception, error)


if __name__ == "__main__":
    unittest.main()

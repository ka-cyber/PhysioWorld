"""Adversarial, stdlib-only checks for evidence-based plugin qualification."""

import inspect
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from physioworld.plugins import (
    PLUGIN_CONTRACTS,
    AssuranceTier,
    BuiltInVerticalPlugin,
    CheckCategory,
    EvidenceArtifact,
    EvidenceTestCase,
    PluginKind,
    PluginMetadata,
    PluginProvenance,
    PluginQualificationPlan,
    PluginRegistry,
    QualificationSuite,
    load_plugin,
    qualification_scope,
    qualify_plugin,
)


class MeanPlugin:
    """An analytic fixture, not a physiological model or external validation."""

    schemas = {
        "transform": {
            "input": {
                "$id": "test:mean:input:1",
                "type": "object",
                "properties": {"history": {"type": "array", "items": {"type": "number"}}},
                "required": ["history"],
                "additionalProperties": False,
            },
            "output": {"$id": "test:mean:output:1", "type": "number"},
        }
    }

    def __init__(self):
        artifact = EvidenceArtifact.from_file(__file__, "analytic fixture and reviewer test source")
        self.metadata = PluginMetadata(
            "test.mean",
            PluginKind.REPRESENTATION,
            "1.0.0",
            "physioworld>=0.1,<0.2",
            (),
            ("Analytic synthetic arithmetic-mean fixture; no empirical evidence",),
            "SELF_REPORTED_PASS",
            "MIT",
            "Reviewed local synthetic fixture",
            {"history_length": (1, 3), "history_values": (-10, 10)},
            provenance=PluginProvenance(
                "test:analytic-mean", "fixture-1", "test fixture", (artifact,)
            ),
        )
        self.schemas = json.loads(json.dumps(type(self).schemas))

    def transform(self, history):
        if not isinstance(history, (list, tuple)) or not 1 <= len(history) <= 3:
            raise ValueError("history length outside declared envelope")
        if any(
            type(x) not in (int, float) or not math.isfinite(x) or not -10 <= x <= 10
            for x in history
        ):
            raise ValueError("history values outside declared envelope")
        return math.fsum(history) / len(history)


class MeanEvidence(EvidenceTestCase):
    """Reviewer-owned oracles with observed, serializable comparisons."""

    plugin = None

    def check_api(self):
        result = self.plugin.transform(history=[1, 2, 3])
        self.assertEvidenceEqual(type(result).__name__, "float", label="keyword API result type")

    def check_schema(self):
        self.assertEvidenceEqual(
            self.plugin.schemas["transform"]["input"]["required"],
            ["history"],
            label="input has required history",
        )
        self.assertEvidenceEqual(
            isinstance(self.plugin.transform([1, 2, 3]), (int, float)),
            True,
            label="actual output satisfies declared number type",
        )

    def check_unit(self):
        self.assertEvidenceEqual(self.plugin.transform([3]), 3.0, label="singleton mean identity")

    def check_reference(self):
        self.assertEvidenceEqual(
            self.plugin.transform([-3, 0, 3]), 0.0, label="symmetric analytic reference"
        )

    def check_numerical(self):
        self.assertEvidenceClose(
            self.plugin.transform([1, 2, 3]),
            2.0,
            abs_tol=1e-12,
            label="independent analytic mean with absolute tolerance",
        )
        self.assertEvidenceClose(
            self.plugin.transform([-10, 10, 0]),
            0.0,
            abs_tol=1e-12,
            label="finite envelope endpoint cancellation",
        )

    def check_reproducibility(self):
        actual = [self.plugin.transform([1, 2, 3]) for _ in range(3)]
        self.assertEvidenceEqual(actual, [2.0, 2.0, 2.0], label="three deterministic replays")

    def check_security(self):
        # Concrete malformed-input checks; this is a narrow software security
        # fixture, not a universal security audit or supply-chain certification.
        rejected = []
        for payload in ([], [float("nan")], [float("inf")], [True], ["__import__('os')"], [11]):
            try:
                self.plugin.transform(payload)
            except ValueError:
                rejected.append(True)
            else:
                rejected.append(False)
        self.assertEvidenceEqual(rejected, [True] * 6, label="malformed input scan rejection")

    def check_noop(self):
        pass

    def check_suppressed_failure(self):
        try:
            self.assertEvidenceEqual(1, 2, label="deliberately suppressed failed assertion")
        except AssertionError:
            pass

    def check_bad_numerical_evidence(self):
        self.assertEvidenceEqual(
            self.plugin.metadata.numerical_test_status, "SELF_REPORTED_PASS", label="metadata only"
        )

    def check_failed_numerical(self):
        self.assertEvidenceClose(
            self.plugin.transform([1, 2, 3]), 9, abs_tol=1e-12, label="wrong reference"
        )

    def check_nonfinite_numerical(self):
        self.assertEvidenceClose(float("nan"), 0, abs_tol=1e-12, label="nonfinite reference")

    def check_skipped(self):
        self.skipTest("no current scan available")

    def check_mutated_schema(self):
        self.plugin.schemas["transform"]["output"]["$id"] = "changed-during-run"
        self.assertEvidenceEqual(True, True, label="mutation exercised")


def _case(plugin, method):
    case = MeanEvidence(method)
    case.plugin = plugin
    return unittest.TestSuite([case])


def api_suite(plugin):
    return _case(plugin, "check_api")


def schema_suite(plugin):
    return _case(plugin, "check_schema")


def unit_suite(plugin):
    return _case(plugin, "check_unit")


def reference_suite(plugin):
    return _case(plugin, "check_reference")


def numerical_suite(plugin):
    return _case(plugin, "check_numerical")


def replay_suite(plugin):
    return _case(plugin, "check_reproducibility")


def security_suite(plugin):
    return _case(plugin, "check_security")


def noop_suite(plugin):
    return _case(plugin, "check_noop")


def suppressed_suite(plugin):
    return _case(plugin, "check_suppressed_failure")


def bad_numerical_suite(plugin):
    return _case(plugin, "check_bad_numerical_evidence")


def failed_numerical_suite(plugin):
    return _case(plugin, "check_failed_numerical")


def nonfinite_suite(plugin):
    return _case(plugin, "check_nonfinite_numerical")


def skipped_suite(plugin):
    return _case(plugin, "check_skipped")


def mutated_suite(plugin):
    return _case(plugin, "check_mutated_schema")


def empty_suite(plugin):
    return unittest.TestSuite()


def plain_unittest_suite(plugin):
    return unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])


FACTORIES = {
    CheckCategory.API: api_suite,
    CheckCategory.SCHEMA: schema_suite,
    CheckCategory.UNIT: unit_suite,
    CheckCategory.REFERENCE: reference_suite,
    CheckCategory.NUMERICAL: numerical_suite,
    CheckCategory.REPRODUCIBILITY: replay_suite,
    CheckCategory.SECURITY: security_suite,
}


def make_plan(plugin, *, missing=(), replacement=None):
    artifact = EvidenceArtifact.from_file(__file__, "reviewer-owned qualification suites")
    suites = tuple(
        QualificationSuite(
            f"mean-{category.value}",
            category,
            replacement[1] if replacement and category == replacement[0] else factory,
            artifact,
            f"synthetic analytic fixture: {category.value}",
        )
        for category, factory in FACTORIES.items()
        if category not in missing
    )
    return PluginQualificationPlan(qualification_scope(plugin), suites)


class PluginQualificationTests(unittest.TestCase):
    def assertUnqualified(self, report, *, interface=True):
        self.assertFalse(report.passed, report.to_dict())
        expected = (
            AssuranceTier.INTERFACE_VERIFIED if interface else AssuranceTier.INSTALLED_UNQUALIFIED
        )
        self.assertEqual(report.assurance_tier, expected.value)

    def test_no_plan_and_metadata_lambda_cannot_qualify(self):
        for tests in ((), (lambda p: p.metadata,), (lambda p: None,), (lambda p: True,)):
            with self.subTest(tests=tests):
                report = qualify_plugin(MeanPlugin(), tests=tests)
                self.assertUnqualified(report)
                self.assertEqual(report.evidence_status, "SOFTWARE_INTERFACE_VERIFIED")
                self.assertEqual(
                    next(c.status for c in report.checks if c.check_id == "suite:numerical"),
                    "NOT_EVALUATED",
                )

    def test_builtin_compatibility_is_interface_only_and_no_vertical_import(self):
        with patch("physioworld.plugins.importlib.import_module") as importer:
            plugin = load_plugin("builtin:verticals")
            report = qualify_plugin(plugin, tests=(lambda p: p.metadata,))
        importer.assert_not_called()
        self.assertIsInstance(plugin, BuiltInVerticalPlugin)
        self.assertUnqualified(report)
        self.assertIn("seed", inspect.signature(plugin.execute).parameters)

    def test_builtin_forwards_seed_and_rejects_undeclared_envelope(self):
        plugin = BuiltInVerticalPlugin()
        runner = Mock(return_value=SimpleNamespace(to_dict=lambda: {"seed": 19}))
        with patch.dict(
            "sys.modules", {"physioworld.verticals": SimpleNamespace(run_vertical=runner)}
        ):
            self.assertEqual(plugin.execute(seed=19), {"seed": 19})
        runner.assert_called_once_with("cardiovascular", seed=19)
        for configuration in ({"name": "unknown"}, {"seed": -1}, {"seed": True}, {"seed": 2**31}):
            with self.subTest(configuration=configuration), self.assertRaises(ValueError):
                plugin.execute(**configuration)

    def test_complete_live_plan_qualifies_with_serializable_current_evidence(self):
        plugin = MeanPlugin()
        report = qualify_plugin(plugin, plan=make_plan(plugin))
        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(report.assurance_tier, "REFERENCE_QUALIFIED")
        self.assertEqual(report.evidence_status, "SOFTWARE_REFERENCE_VERIFIED")
        payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["scope"]["plugin_version"], "1.0.0")
        numerical = next(c for c in payload["checks"] if c["check_id"] == "test:mean-numerical")
        self.assertEqual(numerical["evidence"]["tests_run"], 1)
        self.assertEqual(
            numerical["evidence"]["comparisons"][0]["observations"][0]["abs_tol"], 1e-12
        )

    def test_missing_each_required_suite_blocks_reference_tier(self):
        for category in CheckCategory:
            with self.subTest(category=category):
                plugin = MeanPlugin()
                report = qualify_plugin(plugin, plan=make_plan(plugin, missing=(category,)))
                self.assertUnqualified(report)
                self.assertEqual(
                    next(
                        c.status for c in report.checks if c.check_id == f"suite:{category.value}"
                    ),
                    "NOT_EVALUATED",
                )

    def test_absent_numerical_and_security_not_replaced_by_self_report(self):
        plugin = MeanPlugin()
        report = qualify_plugin(
            plugin,
            plan=make_plan(plugin, missing=(CheckCategory.NUMERICAL, CheckCategory.SECURITY)),
        )
        self.assertUnqualified(report)
        self.assertEqual(plugin.metadata.numerical_test_status, "SELF_REPORTED_PASS")

    def test_noop_empty_plain_skipped_and_suppressed_failure_not_evidence(self):
        for factory in (
            noop_suite,
            empty_suite,
            plain_unittest_suite,
            skipped_suite,
            suppressed_suite,
        ):
            with self.subTest(factory=factory.__name__):
                plugin = MeanPlugin()
                report = qualify_plugin(
                    plugin, plan=make_plan(plugin, replacement=(CheckCategory.SECURITY, factory))
                )
                self.assertUnqualified(report)
                self.assertEqual(
                    next(c.status for c in report.checks if c.check_id == "suite:security"), "FAIL"
                )

    def test_numerical_requires_real_finite_comparisons(self):
        for factory in (bad_numerical_suite, failed_numerical_suite, nonfinite_suite):
            with self.subTest(factory=factory.__name__):
                plugin = MeanPlugin()
                report = qualify_plugin(
                    plugin, plan=make_plan(plugin, replacement=(CheckCategory.NUMERICAL, factory))
                )
                self.assertUnqualified(report)
                self.assertEqual(
                    next(c.status for c in report.checks if c.check_id == "suite:numerical"), "FAIL"
                )

    def test_version_api_and_host_mismatch(self):
        for changes in (
            {"api_version": "2.0"},
            {"api_compatibility": "physioworld>=9.0"},
            {"api_compatibility": "unrelated>=0.1"},
        ):
            with self.subTest(changes=changes):
                plugin = MeanPlugin()
                plugin.metadata = replace(plugin.metadata, **changes)
                report = qualify_plugin(plugin)
                self.assertUnqualified(report, interface=False)
                self.assertEqual(
                    next(c.status for c in report.checks if c.check_id == "api-version"), "FAIL"
                )

    def test_stale_plugin_version_and_metadata_scope_blocks_suites(self):
        for changes in (
            {"version": "1.1.0"},
            {"license": "BSD-3-Clause"},
            {"operating_envelope": {"history_length": (1, 2)}},
        ):
            with self.subTest(changes=changes):
                plugin = MeanPlugin()
                plan = make_plan(plugin)
                plugin.metadata = replace(plugin.metadata, **changes)
                report = qualify_plugin(plugin, plan=plan)
                self.assertUnqualified(report)
                self.assertEqual(
                    next(c.status for c in report.checks if c.check_id == "plan-scope"), "FAIL"
                )
                self.assertFalse(any(c.check_id.startswith("test:") for c in report.checks))

    def test_stale_schema_scope_is_rejected(self):
        plugin = MeanPlugin()
        plan = make_plan(plugin)
        plugin.schemas["transform"]["output"]["$id"] = "test:changed-schema"
        self.assertUnqualified(qualify_plugin(plugin, plan=plan))

    def test_stale_runtime_entrypoint_is_rejected(self):
        plugin = MeanPlugin()
        plan = make_plan(plugin)
        plugin.transform = lambda history: math.fsum(history)
        self.assertUnqualified(qualify_plugin(plugin, plan=plan))

    def test_stale_source_and_test_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "fixture.txt"
            file.write_text("reviewed fixture", encoding="utf-8")
            artifact = EvidenceArtifact.from_file(file, "explicit extra source fixture")
            plugin = MeanPlugin()
            plugin.metadata = replace(
                plugin.metadata,
                provenance=replace(
                    plugin.metadata.provenance,
                    artifacts=plugin.metadata.provenance.artifacts + (artifact,),
                ),
            )
            plan = make_plan(plugin)
            file.write_text("changed fixture", encoding="utf-8")
            self.assertUnqualified(qualify_plugin(plugin, plan=plan))
        plugin = MeanPlugin()
        plan = make_plan(plugin)
        corrupted = replace(
            plan.suites[-1], artifact=replace(plan.suites[-1].artifact, sha256="0" * 64)
        )
        report = qualify_plugin(plugin, plan=replace(plan, suites=plan.suites[:-1] + (corrupted,)))
        self.assertUnqualified(report)
        self.assertIn(
            "checksum mismatch",
            next(c.detail for c in report.checks if c.check_id == "test:mean-security"),
        )

    def test_schema_mutation_during_run_invalidates_interface_and_reference(self):
        plugin = MeanPlugin()
        report = qualify_plugin(
            plugin, plan=make_plan(plugin, replacement=(CheckCategory.SECURITY, mutated_suite))
        )
        self.assertUnqualified(report, interface=False)
        self.assertEqual(
            next(c.status for c in report.checks if c.check_id == "scope-stability"), "FAIL"
        )

    def test_smoke_failure_is_visible_and_cannot_sneak_mutation_after_scope_check(self):
        plugin = MeanPlugin()
        report = qualify_plugin(plugin, plan=make_plan(plugin), tests=(lambda p: False,))
        self.assertUnqualified(report)
        self.assertEqual(next(c.status for c in report.checks if c.check_id == "smoke:1"), "FAIL")
        plugin = MeanPlugin()
        plan = make_plan(plugin)

        def mutate(value):
            value.metadata = replace(value.metadata, version="1.0.1")

        self.assertUnqualified(qualify_plugin(plugin, plan=plan, tests=(mutate,)), interface=False)
        plugin = MeanPlugin()
        self.assertUnqualified(qualify_plugin(plugin, tests=(mutate,)), interface=False)

    def test_boolean_numbers_are_not_equal_evidence(self):
        case = EvidenceTestCase()
        with self.assertRaises(AssertionError):
            case.assertEvidenceEqual(
                [1], [True], label="boolean and number are different JSON types"
            )

    def test_contract_failure_even_with_complete_metadata(self):
        plugin = MeanPlugin()
        plugin.transform = None
        self.assertUnqualified(qualify_plugin(plugin), interface=False)
        plugin = MeanPlugin()
        plugin.transform = lambda wrong: 0
        self.assertUnqualified(qualify_plugin(plugin), interface=False)
        plugin = MeanPlugin()
        plugin.transform = lambda **kwargs: 0
        self.assertUnqualified(qualify_plugin(plugin), interface=False)

    def test_every_kind_has_explicit_entrypoints_and_missing_contract_is_rejected(self):
        self.assertEqual(set(PLUGIN_CONTRACTS), set(PluginKind))
        for kind, contract in PLUGIN_CONTRACTS.items():
            with self.subTest(kind=kind):
                self.assertTrue(contract.methods)
                plugin = MeanPlugin()
                plugin.metadata = replace(plugin.metadata, kind=kind)
                if kind != PluginKind.REPRESENTATION:
                    self.assertUnqualified(qualify_plugin(plugin), interface=False)

    def test_dependencies_use_distribution_metadata_without_importing(self):
        for actual in ("1.5.0", "2.0.0", "1.5.0rc1"):
            with self.subTest(actual=actual):
                plugin = MeanPlugin()
                plugin.metadata = replace(plugin.metadata, dependencies=("test-dependency>=1,<2",))
                with (
                    patch("physioworld.plugins.version", return_value=actual),
                    patch("physioworld.plugins.importlib.import_module") as importer,
                ):
                    report = qualify_plugin(plugin)
                importer.assert_not_called()
                self.assertUnqualified(report, interface=actual == "1.5.0")

    def test_missing_dependency_and_unsupported_requirement_fail_closed(self):
        from importlib.metadata import PackageNotFoundError

        plugin = MeanPlugin()
        plugin.metadata = replace(plugin.metadata, dependencies=("missing>=1",))
        with patch("physioworld.plugins.version", side_effect=PackageNotFoundError("missing")):
            self.assertUnqualified(qualify_plugin(plugin), interface=False)
        for requirement in ("unbounded", "package~=1.0", "package>=1; python_version>'3'"):
            plugin.metadata = replace(plugin.metadata, dependencies=(requirement,))
            self.assertUnqualified(qualify_plugin(plugin), interface=False)

    def test_dependency_version_change_invalidates_plan(self):
        plugin = MeanPlugin()
        plugin.metadata = replace(plugin.metadata, dependencies=("example>=1,<2",))
        with patch("physioworld.plugins.version", return_value="1.1.0"):
            plan = make_plan(plugin)
        with patch("physioworld.plugins.version", return_value="1.2.0"):
            self.assertUnqualified(qualify_plugin(plugin, plan=plan))

    def test_missing_provenance_license_evidence_and_envelope_prevent_reference_tier(self):
        for changes in (
            {"provenance": None},
            {"license": ""},
            {"scientific_evidence": ()},
            {"security_provenance": ""},
            {"operating_envelope": {}},
        ):
            with self.subTest(changes=changes):
                plugin = MeanPlugin()
                plugin.metadata = replace(plugin.metadata, **changes)
                self.assertUnqualified(qualify_plugin(plugin, plan=make_plan(plugin)))

    def test_invalid_schema_declarations_block_interface(self):
        for schema in (
            {},
            {"$id": "test:bad", "type": "not-a-type"},
            {"$id": "test:bad", "type": "object", "required": ["missing"]},
        ):
            with self.subTest(schema=schema):
                plugin = MeanPlugin()
                plugin.schemas["transform"]["input"] = schema
                self.assertUnqualified(qualify_plugin(plugin), interface=False)

    def test_metadata_roundtrip_immutability_and_version_rejection(self):
        metadata = MeanPlugin().metadata
        restored = PluginMetadata.from_dict(json.loads(json.dumps(metadata.to_dict())))
        self.assertEqual(restored, metadata)
        with self.assertRaises(TypeError):
            metadata.operating_envelope["history_length"] = (1, 20)
        for changes in (
            {"schema_version": "2.0"},
            {"version": "1.0.0rc1"},
            {"operating_envelope": {"values": (0, float("nan"))}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(metadata, **changes)
        payload = metadata.to_dict()
        del payload["schema_version"]
        with self.assertRaises(ValueError):
            PluginMetadata.from_dict(payload)

    def test_registry_does_not_qualify_or_discover_on_registration(self):
        plugin = MeanPlugin()
        registry = PluginRegistry()
        with (
            patch("physioworld.plugins.qualify_plugin") as qualifier,
            patch("physioworld.plugins.importlib.import_module") as importer,
        ):
            registry.register(plugin)
            self.assertIs(registry.get(plugin.metadata.plugin_id), plugin)
        qualifier.assert_not_called()
        importer.assert_not_called()
        self.assertUnqualified(registry.qualify(plugin.metadata.plugin_id))
        self.assertTrue(registry.qualify(plugin.metadata.plugin_id, plan=make_plan(plugin)).passed)
        with self.assertRaises(ValueError):
            registry.register(plugin)
        with self.assertRaises(KeyError):
            registry.get("unknown")

    def test_loading_only_imports_explicitly_requested_reference(self):
        class Module:
            plugin = MeanPlugin

        with patch("physioworld.plugins.importlib.import_module", return_value=Module) as importer:
            self.assertIsInstance(load_plugin("requested_module:plugin"), MeanPlugin)
        importer.assert_called_once_with("requested_module")
        for reference in (
            "module",
            "module:",
            ":attribute",
            "module:attr:other",
            "../module:plugin",
        ):
            with (
                self.subTest(reference=reference),
                patch("physioworld.plugins.importlib.import_module") as importer,
            ):
                with self.assertRaises(ValueError):
                    load_plugin(reference)
                importer.assert_not_called()

    def test_duplicate_suite_identity_and_unpinned_factories_are_rejected(self):
        plugin = MeanPlugin()
        plan = make_plan(plugin)
        with self.assertRaises(ValueError):
            replace(plan, suites=plan.suites + (plan.suites[0],))
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "unrelated.txt"
            artifact_path.write_text("not executable test source", encoding="utf-8")
            artifact = EvidenceArtifact.from_file(artifact_path, "unrelated artifact")
            bad = replace(plan.suites[-1], artifact=artifact)
            self.assertUnqualified(
                qualify_plugin(plugin, plan=replace(plan, suites=plan.suites[:-1] + (bad,)))
            )

    def test_additional_failed_suite_cannot_hide_behind_passing_category(self):
        plugin = MeanPlugin()
        plan = make_plan(plugin)
        extra = replace(plan.suites[-1], suite_id="extra-security", factory=noop_suite)
        self.assertUnqualified(
            qualify_plugin(plugin, plan=replace(plan, suites=plan.suites + (extra,)))
        )


if __name__ == "__main__":
    unittest.main()

"""Behavioral checks for the native Level 3 benchmark corpus."""

import hashlib
import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from unittest.mock import patch

from physioworld.benchmarks import corpus, run_corpus_campaign
from physioworld.experiments import ExperimentValidationError
from physioworld.io.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION = ROOT / "configs" / "corpus" / "benchmark_corpus_level3.json"
FIXTURE_HASHES = {
    "normal_physiology.json": "9d0635708e43d070b6e3f542e589f36e680659088eab937b7b61b1cd54e5b998",  # pragma: allowlist secret
    "artifact.json": "ceeb61c309b784ad1c72ad179dea46df6b004a68d9df5506d53350415a50bd23",  # pragma: allowlist secret
    "missing_modality.json": "1cd737a8439d4258dde2b5458e6340656f9e28662007432dffe8176bf5d32938",  # pragma: allowlist secret
    "packet_loss.json": "3be3d288189fe74782afceed20146d2007b080f372b9fd4b1b834bf05388e8d2",  # pragma: allowlist secret
    "resource_depletion.json": "154c75f8f2028184f5ac78925ae79398426be3967e8e78eda264f2c1b24a76b4",  # pragma: allowlist secret
    "task_transition.json": "1a7876dde096a0a7b8ce65da4b024ce85c7bec7f264d01914a6866654399f84f",  # pragma: allowlist secret
    "causal_intervention.json": "428b86a0e08eee18635ce68adbc4482cd8365b8819cda974b0e0760fac124a71",  # pragma: allowlist secret
}
FIXTURE_SEEDS = {
    "normal_physiology.json": 1,
    "artifact.json": 2,
    "missing_modality.json": 3,
    "packet_loss.json": 4,
    "resource_depletion.json": 5,
    "task_transition.json": 6,
    "causal_intervention.json": 7,
}
NATIVE_CASES = (
    "representation_temporal",
    "tasklaw_transition",
    "communication_semantic",
    "resource_p1",
    "policy_capability",
    "causal_synthetic",
    "robustness_common_policy",
    "cardiovascular",
    "neurophysiology",
    "icu",
)
EXPECTED_EXTERNAL_INPUTS = [
    {
        "id": "UPIT_EMPIRICAL_RAW",
        "status": "EXTERNAL_ACQUISITION_REQUIRED",
        "role": "raw empirical cohort and source tables",
    },
    {
        "id": "EXTERNAL_31_RECORDING_COHORT",
        "status": "EXTERNAL_ACQUISITION_REQUIRED",
        "role": "independent predictive transport cohort",
    },
    {
        "id": "WEARABLE_CPAP_MAYO_LBNP",
        "status": "EXTERNAL_ACQUISITION_REQUIRED",
        "role": "external stress and validation datasets",
    },
    {
        "id": "HARDWARE_MEASUREMENTS",
        "status": "NOT_SUPPLIED",
        "role": "physical device and hardware validation",
    },
]


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


class BenchmarkCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "corpus"

    def load_campaign(self, case_groups=None):
        configuration_path = CONFIGURATION
        if case_groups is not None:
            configuration = load_config(CONFIGURATION)
            configuration["cases"] = case_groups
            configuration_path = Path(self.temporary.name) / "selected.json"
            configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
        run_corpus_campaign(self.output, configuration=configuration_path)
        self.manifest = json.loads(
            (self.output / "corpus_manifest.json").read_text(encoding="utf-8")
        )
        self.summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.cases = {case["case_id"]: case for case in self.summary["cases"]}

    def test_config_is_loaded_and_validated_by_the_corpus_schema(self):
        configuration = load_config(CONFIGURATION)
        schema = json.loads((ROOT / "schemas" / "corpus.schema.json").read_text(encoding="utf-8"))

        self.assertEqual(configuration["corpus_id"], "native_level3_synthetic")
        self.assertEqual(configuration["execution_profile"], "native_level3_targeted")
        self.assertEqual(configuration["target_fidelity"], "HYBRID")
        self.assertEqual(configuration["fixture_root"], "package:physioworld/corpus/golden")
        self.assertEqual(configuration["external_inputs"], "NOT_REPRODUCIBLE_FROM_AVAILABLE_INPUTS")
        self.assertFalse(configuration["clinical_claim"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], "1.0.0")
        self.assertEqual(schema["properties"]["clinical_claim"]["const"], False)

        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            self.skipTest("jsonschema is not installed")
        Draft202012Validator(schema).validate(configuration)

    def test_golden_fixtures_load_with_pinned_hashes(self):
        fixture_root = files("physioworld").joinpath("corpus", "golden")

        for filename, expected_hash in FIXTURE_HASHES.items():
            with self.subTest(fixture=filename):
                resource = fixture_root.joinpath(filename)
                payload = resource.read_bytes()
                fixture = json.loads(payload)
                fixture_id = filename.removesuffix(".json")
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected_hash)
                self.assertEqual(fixture["fixture"], fixture_id)
                self.assertEqual(fixture["seed"], FIXTURE_SEEDS[filename])

    def test_campaign_executes_every_native_level3_case(self):
        self.load_campaign()
        self.assertEqual(self.manifest["fixture_hashes"], FIXTURE_HASHES)
        self.assertEqual(self.manifest["cases"], list(NATIVE_CASES))
        self.assertEqual(self.summary["cases"], [self.cases[case_id] for case_id in NATIVE_CASES])
        self.assertEqual(self.summary["case_count"], len(NATIVE_CASES))
        self.assertEqual(set(self.cases), set(NATIVE_CASES))
        self.assertTrue(
            all(
                case["status"] in {"COMPLETE_SYNTHETIC_ONLY", "COMPLETE_NUMERICAL_REPRODUCTION"}
                for case in self.cases.values()
            )
        )

    def test_p1_policy_and_robustness_use_exact_independent_references(self):
        self.load_campaign()
        p1 = self.cases["resource_p1"]
        self.assertEqual(p1["status"], "COMPLETE_NUMERICAL_REPRODUCTION")
        self.assertEqual(p1["independent_reference"]["method"], "closed_form_constant_current_p1")
        self.assertAlmostEqual(p1["independent_reference"]["current_limit_a"], 0.013793103448275862)
        self.assertEqual(p1["independent_reference"]["terminal_voltage_v"], 2.07)
        self.assertEqual(p1["observed"]["terminal_voltage_v"], 2.07)
        self.assertTrue(p1["comparisons"]["terminal_voltage"]["passed"])
        self.assertTrue(p1["comparisons"]["current_limit"]["passed"])
        self.assertEqual(
            self.manifest["reference_artifacts"]["ra_upit_resource"]["values"],
            {
                "p0_current_ma": 13.7931,
                "p1_current_ma": 13.39603,
                "p2_nominal_ma": 6.86462,
                "p2_robust_ma": 6.113567,
                "tailored_energy_j": 0.529841,
                "uniform_energy_j": 0.646043,
                "reduction_fraction": 0.1799,
            },
        )

        policy = self.cases["policy_capability"]
        self.assertEqual(policy["observed"], {"oracle_value": 2.0, "oracle_action": "go"})
        self.assertEqual(
            policy["independent_reference"],
            {"method": "exhaustive_action_sequence_enumeration", "value": 2.0},
        )
        self.assertEqual(policy["comparison"]["residual"], 0.0)
        self.assertTrue(policy["comparison"]["passed"])

        robustness = self.cases["robustness_common_policy"]
        self.assertEqual(
            robustness["observed"], {"robust_value_sup_inf": 0.0, "optimistic_value_inf_sup": 1.0}
        )
        self.assertEqual(
            robustness["independent_reference"],
            {"method": "direct_policy_kernel_enumeration", "sup_inf": 0.0, "inf_sup": 1.0},
        )
        self.assertTrue(robustness["comparisons"]["sup_inf"]["passed"])
        self.assertTrue(robustness["comparisons"]["inf_sup"]["passed"])

    def test_causal_case_has_simulation_only_evidence_labels(self):
        self.load_campaign()
        causal = self.cases["causal_synthetic"]
        self.assertEqual(causal["status"], "COMPLETE_SYNTHETIC_ONLY")
        self.assertEqual(causal["data_origin"], "KNOWN_GROUND_TRUTH_SCM")
        self.assertEqual(causal["data_fidelity"], "SYNTHETIC")
        self.assertEqual(causal["evidence_class"], "SIMULATION")
        self.assertEqual(causal["observed"], {"sample_count": 100, "model_ate": 2.0})
        self.assertEqual(
            causal["independent_reference"],
            {"method": "declared_structural_effect", "ate": 2.0},
        )
        self.assertEqual(causal["controls"]["observational_claim"], "BLOCKED")
        self.assertEqual(causal["controls"]["future_covariate"], "BLOCKED_BY_CAUSAL_GATE")
        self.assertTrue(
            any("not empirical causal evidence" in item for item in causal["limitations"])
        )

    def test_verticals_have_deterministic_replay_hashes(self):
        self.load_campaign()
        for vertical in ("cardiovascular", "neurophysiology", "icu"):
            with self.subTest(vertical=vertical):
                case = self.cases[vertical]
                self.assertEqual(case["status"], "COMPLETE_SYNTHETIC_ONLY")
                self.assertEqual(case["evidence_class"], "SIMULATION")
                self.assertTrue(case["replay_equal"])
                self.assertEqual(case["scientific_output_hash"], case["replay_hash"])
                self.assertEqual(case["scientific_output_hash"], canonical_hash(case["outputs"]))

    def test_prompt_references_are_manually_ingested_and_not_active_evidence(self):
        self.load_campaign()
        self.assertEqual(self.manifest["source_status"], "MANUALLY_INGESTED_SPECIFICATION")
        self.assertTrue(self.manifest["reference_artifacts"])
        for reference in self.manifest["reference_artifacts"].values():
            with self.subTest(estimand=reference["estimand"]):
                self.assertEqual(reference["source_id"], "LEVEL3_PROMPT")
                self.assertEqual(reference["source_status"], "MANUALLY_INGESTED")
                self.assertFalse(reference.get("active_evidence", False))
                self.assertIn("reproduction_status", reference)

        for case in self.cases.values():
            reference = case.get("reference_comparison")
            if reference is not None:
                self.assertEqual(reference["source_status"], "MANUALLY_INGESTED")
                self.assertFalse(reference["active_evidence"])

    def test_external_inputs_are_explicitly_unavailable(self):
        self.load_campaign()
        self.assertEqual(self.manifest["external_inputs"], EXPECTED_EXTERNAL_INPUTS)
        self.assertEqual(
            json.loads((self.output / "unavailable_inputs.json").read_text(encoding="utf-8")),
            EXPECTED_EXTERNAL_INPUTS,
        )
        self.assertEqual(self.manifest["actual_fidelity"], "SYNTHETIC_AND_MATHEMATICAL")
        self.assertNotIn("EMPIRICAL", self.manifest["evidence_class"])
        self.assertNotIn("CLINICAL", self.manifest["evidence_class"])
        self.assertNotIn("HARDWARE", self.manifest["evidence_class"])

        statuses = {item["status"] for item in self.manifest["external_inputs"]}
        self.assertEqual(statuses, {"EXTERNAL_ACQUISITION_REQUIRED", "NOT_SUPPLIED"})
        self.assertNotIn("UPIT_EMPIRICAL_RAW", self.manifest["cases"])

    def test_campaign_rejects_a_nonfresh_output_directory(self):
        occupied = Path(self.temporary.name) / "occupied"
        occupied.mkdir()
        sentinel = occupied / "sentinel.txt"
        sentinel.write_text("caller-owned", encoding="utf-8")
        with (
            patch.object(corpus, "_resource_fixture") as computation,
            self.assertRaises(FileExistsError),
        ):
            run_corpus_campaign(occupied, configuration=CONFIGURATION)
        computation.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "caller-owned")

    def test_campaign_rejects_unsafe_or_file_paths_before_computation(self):
        root = Path(self.temporary.name)
        file = root / "file"
        file.write_bytes(b"caller-owned")
        outside = root / "outside"
        outside.mkdir()
        alias = root / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        dangling = root / "dangling"
        dangling.symlink_to(root / "absent", target_is_directory=True)
        for destination in (
            file,
            file / "child",
            alias,
            alias / "child",
            dangling,
            root / ".." / "escape",
        ):
            with (
                self.subTest(destination=destination),
                patch.object(corpus, "_resource_fixture") as work,
            ):
                with self.assertRaises((ValueError, FileExistsError, NotADirectoryError)):
                    run_corpus_campaign(destination, configuration=CONFIGURATION)
                work.assert_not_called()
        self.assertEqual(file.read_bytes(), b"caller-owned")
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((root / "absent").exists())

    def test_invalid_configuration_and_mixed_markers_reject_before_outputs(self):
        original = load_config(CONFIGURATION)
        path = Path(self.temporary.name) / "invalid.json"
        for change in (
            {"clinical_claim": True},
            {"cases": []},
            {"cases": ["resource_p1", "resource_p1"]},
            {"cases": ["unknown"]},
            {"seed": 2},
            {"dataset": "mitdb"},
            {"experiment": {"seed": 2}},
        ):
            path.write_text(json.dumps({**original, **change}), encoding="utf-8")
            with self.subTest(change=change), patch.object(corpus, "_resource_fixture") as work:
                with self.assertRaises(ExperimentValidationError):
                    run_corpus_campaign(self.output, configuration=path)
                work.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_campaign_rechecks_freshness_after_computation(self):
        original = corpus._resource_fixture

        def occupy_output():
            self.output.mkdir()
            (self.output / "summary.json").write_bytes(b"new-owner")
            return original()

        with patch.object(corpus, "_resource_fixture", side_effect=occupy_output):
            with self.assertRaises(FileExistsError):
                self.load_campaign(["resource_p1"])
        self.assertEqual((self.output / "summary.json").read_bytes(), b"new-owner")
        self.assertEqual([path.name for path in self.output.iterdir()], ["summary.json"])

    def test_case_schema_references_resolve_relative_to_each_case_document(self):
        self.load_campaign(["resource_p1", "policy_capability"])
        schema_path = self.output / "corpus_result.schema.json"
        schema = json.loads(schema_path.read_text())
        self.assertEqual(
            schema,
            json.loads((ROOT / "src" / "physioworld" / "schemas" / schema_path.name).read_text()),
        )
        self.assertEqual(
            schema["oneOf"][0]["properties"]["$schema"]["const"],
            "../../corpus_result.schema.json",
        )
        for case_id, summary_case in self.cases.items():
            path = self.output / "cases" / case_id / "result.json"
            case = json.loads(path.read_text())
            with self.subTest(case_id=case_id):
                self.assertEqual(case, summary_case)
                self.assertEqual((path.parent / case["$schema"]).resolve(), schema_path.resolve())
                corpus._validate_document(case, schema)
                wrong_reference = {**case, "$schema": "corpus_result.schema.json"}
                with self.assertRaises(ExperimentValidationError):
                    corpus._validate_document(wrong_reference, schema)
        for name in ("summary.json", "corpus_manifest.json"):
            path = self.output / name
            document = json.loads(path.read_text())
            self.assertEqual((path.parent / document["$schema"]).resolve(), schema_path.resolve())


if __name__ == "__main__":
    unittest.main()

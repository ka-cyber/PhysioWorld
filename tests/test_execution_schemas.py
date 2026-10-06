"""Execution declarations cannot bypass scientific validation or imply execution."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from physioworld.experiments import ExperimentCompiler, ExperimentValidationError
from physioworld.experiments.compiler import (
    configuration_schema_name,
    load_schema,
    schema_names,
)

ROOT = Path(__file__).resolve().parents[1]
HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None


def minimal_configuration():
    return {
        "seed": 7,
        "modality": "ECG",
        "unit": "mV",
        "sample_rate": 10,
        "samples": [0.0, 0.2, 0.8],
        "metrics": ["mean", "rms"],
    }


def legacy_configuration():
    return {
        "dataset": "mitdb",
        "task": {
            "id": "arrhythmia",
            "target": "rhythm_annotation",
            "horizon_s": 0,
            "metric": "macro_f1",
        },
        "representation": "raw",
        "perturbations": [],
        "channel": {},
        "resource": {},
        "compute": {},
        "policy": {},
        "seed": 4,
        "split": "subject_test",
        "metrics": ["macro_f1"],
        "outputs": {},
    }


def vertical_configuration():
    """A complete configuration for runtime integration; every cost is declared."""
    return {
        "$schema": "experiment.schema.json",
        "schema_version": "1.0.0",
        "name": "cardiovascular_vertical",
        "vertical": "cardiovascular",
        "dataset": {
            "id": "synthetic_cardio",
            "origin": "SYNTHETIC",
            "model": "SyntheticPhysiology",
            "modality": "telemetry",
            "channel": "mean_pressure_mmhg",
            "unit": "mmHg",
            "time_unit": "s",
            "duration_s": 12,
            "sample_rate_hz": 1,
            "sample_interval_s": 1,
        },
        "task": {
            "id": "pressure_threshold",
            "kind": "threshold_exceedance",
            "target": "mean_pressure_mmhg",
            "horizon_s": 0,
            "metric": "task_law_discrepancy",
            "threshold": 80,
            "threshold_unit": "mmHg",
            "loss": "zero_one",
            "reference": "full_information",
            "missingness": "unavailable",
            "uncertainty": "seeded_simulation",
        },
        "representation": {
            "name": "finite_history",
            "window": 5,
            "window_unit": "samples",
            "pad_value": 0,
        },
        "perturbations": [
            {
                "type": "physiological_action",
                "time_s": 4,
                "kind": "orthostasis",
                "parameters": {"orthostasis": 1},
            },
        ],
        "channel": {
            "model": "erasure",
            "loss_probability": 0,
            "latency_s": 0.02,
            "jitter_s": 0.005,
            "packet_payload_bytes": 4,
            "encoding": "float32_be",
            "deadline_s": 1,
        },
        "resource": {
            "model": "capacitor_p1",
            "capacitance_f": 0.2,
            "esr_ohm": 8,
            "leakage_siemens": 0,
            "initial_voltage_v": 2.65,
            "v_min": 1.85,
            "v_max": 3.3,
            "max_current_a": 0.02,
            "load_current_a": 0.001,
            "step_duration_s": 1,
        },
        "compute": {
            "location": "local",
            "operations_per_sample": 100,
            "operations_per_second": 100000,
            "memory_bytes": 4096,
            "required_memory_bytes": 64,
            "energy_per_operation_j": 1e-9,
            "network_latency_s": 0,
        },
        "policy": {
            "class": "declared_threshold_policy",
            "thresholds": [75, 80, 85],
            "threshold_unit": "mmHg",
            "information_set": "received_history",
            "action_set": ["negative", "positive", "abstain"],
            "fallback_action": "abstain",
            "rewards": {
                "true_positive": 1,
                "false_positive": -1,
                "true_negative": 1,
                "false_negative": -1,
                "unavailable": 0,
            },
            "support_direction": [1, 1],
        },
        "seed": 0,
        "split": {"strategy": "synthetic_fixed"},
        "metrics": ["packet_delivery", "task_law_discrepancy", "capability_support"],
        "outputs": {"format": "json"},
    }


def replay_configuration():
    result = vertical_configuration()
    result["dataset"] = {
        "id": "declared_replay",
        "origin": "REPLAY",
        "modality": "telemetry",
        "channel": "mean_pressure_mmhg",
        "unit": "mmHg",
        "time_unit": "s",
        "source": "synthetic test fixture; not a human recording",
        "samples": [90, 85, 75],
        "timestamps_s": [0, 0.5, 2],
    }
    result["perturbations"] = []
    result["split"] = {"strategy": "replay_fixed"}
    return result


class ExecutionCompilerTests(unittest.TestCase):
    def setUp(self):
        self.compiler = ExperimentCompiler()

    def test_shipped_minimal_world_compiles_as_synthetic(self):
        plan = self.compiler.compile_file(ROOT / "examples" / "minimal_world.json")
        self.assertEqual(plan.data_mode, "synthetic")
        self.assertEqual(plan.configuration_kind, "synthetic_samples")
        self.assertNotIn("dataset", plan.configuration)
        self.assertEqual(plan.to_dict()["execution_status"], "NOT_EXECUTED")

    def test_historical_summary_defaults_are_explicit_in_plan(self):
        config = minimal_configuration()
        del config["metrics"]
        plan = self.compiler.compile(config)
        self.assertEqual(plan.configuration["metrics"], ["mean", "rms"])
        self.assertNotIn("metrics", config)

    def test_legacy_empty_pipeline_components_remain_configuration_only(self):
        config = legacy_configuration()
        for dataset in ("mitdb", {"id": "synthetic_but_not_available", "origin": "SYNTHETIC"}):
            with self.subTest(dataset=dataset):
                config["dataset"] = dataset
                plan = self.compiler.compile(config)
                self.assertEqual(plan.data_mode, "dataset")
                self.assertEqual(plan.configuration_kind, "dataset_readiness")
                self.assertEqual(plan.readiness_status, "CONFIGURATION_ONLY")
                self.assertEqual(plan.to_dict()["execution_status"], "NOT_EXECUTED")

    def test_public_legacy_configs_remain_valid(self):
        for path in sorted((ROOT / "configs").glob("*.json")):
            with self.subTest(config=path.name):
                plan = self.compiler.compile_file(path)
                self.assertEqual(plan.readiness_status, "CONFIGURATION_ONLY")

    def test_complete_vertical_is_typed_and_has_never_executed(self):
        config = vertical_configuration()
        with patch.dict("sys.modules", {"jsonschema": None}):
            plan = self.compiler.compile(config)
        self.assertEqual(plan.configuration_kind, "vertical")
        self.assertEqual(plan.schema_name, "vertical.schema.json")
        self.assertEqual(plan.data_mode, "synthetic")
        self.assertEqual(plan.readiness_status, "READY_FOR_EXECUTION")
        self.assertEqual(plan.to_dict()["execution_status"], "NOT_EXECUTED")

    def test_inline_irregular_replay_has_distinct_mode(self):
        plan = self.compiler.compile(replay_configuration())
        self.assertEqual(plan.configuration_kind, "vertical")
        self.assertEqual(plan.data_mode, "replay")

    def test_vertical_names_never_fall_back_to_legacy_readiness(self):
        for name in ("cardiovascular", "neurophysiology", "icu", "unknown"):
            config = legacy_configuration()
            config["vertical"] = name
            with self.subTest(vertical=name), self.assertRaises(ExperimentValidationError):
                self.compiler.compile(config)

    def test_empty_vertical_pipeline_components_are_rejected(self):
        for component in (
            "dataset",
            "task",
            "representation",
            "channel",
            "resource",
            "compute",
            "policy",
            "split",
            "outputs",
        ):
            config = vertical_configuration()
            config[component] = {}
            with self.subTest(component=component), self.assertRaises(ExperimentValidationError):
                self.compiler.compile(config)

    def test_executable_vertical_version_is_mandatory(self):
        for version in (None, "0.1.0", 1, True):
            config = vertical_configuration()
            if version is None:
                del config["schema_version"]
            else:
                config["schema_version"] = version
            with self.subTest(version=version), self.assertRaises(ExperimentValidationError):
                self.compiler.compile(config)

    def test_explicit_units_and_time_are_required(self):
        for component, field in (
            ("dataset", "unit"),
            ("dataset", "time_unit"),
            ("task", "threshold_unit"),
            ("task", "horizon_s"),
            ("representation", "window_unit"),
            ("channel", "deadline_s"),
        ):
            config = vertical_configuration()
            del config[component][field]
            with self.subTest(field=field), self.assertRaises(ExperimentValidationError):
                self.compiler.compile(config)
        config = vertical_configuration()
        config["task"]["threshold_unit"] = "Pa"
        with self.assertRaisesRegex(ExperimentValidationError, "units"):
            self.compiler.compile(config)

    def test_boolean_or_noninteger_seeds_are_rejected_in_every_branch(self):
        for factory in (minimal_configuration, legacy_configuration, vertical_configuration):
            for seed in (True, False, None, 1.0, "1"):
                config = factory()
                config["seed"] = seed
                with self.subTest(branch=factory.__name__, seed=seed):
                    with self.assertRaises(ExperimentValidationError):
                        self.compiler.compile(config)

    def test_nonfinite_values_are_rejected_even_in_legacy_extensions(self):
        for factory in (minimal_configuration, legacy_configuration, vertical_configuration):
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                config = factory()
                config["unreviewed_metadata"] = {"nested": [nonfinite]}
                with self.subTest(branch=factory.__name__, nonfinite=nonfinite):
                    with self.assertRaisesRegex(ExperimentValidationError, "finite JSON"):
                        self.compiler.compile(config)

    def test_arbitrary_objects_are_not_stringified_for_hashing(self):
        config = legacy_configuration()
        config["metadata"] = {"path": Path("not-a-json-value")}
        with self.assertRaisesRegex(ExperimentValidationError, "unsupported JSON value"):
            self.compiler.compile(config)
        config["metadata"] = {1: "ambiguous key"}
        with self.assertRaisesRegex(ExperimentValidationError, "keys must be strings"):
            self.compiler.compile(config)

    def test_horizons_must_be_nonnegative_finite_numbers(self):
        for factory in (legacy_configuration, vertical_configuration):
            for horizon in (-1, True, None, "12", float("nan"), float("inf")):
                config = factory()
                config["task"]["horizon_s"] = horizon
                with self.subTest(branch=factory.__name__, horizon=horizon):
                    with self.assertRaises(ExperimentValidationError):
                        self.compiler.compile(config)
        config = vertical_configuration()
        config["task"]["horizon_s"] = 13
        with self.assertRaisesRegex(ExperimentValidationError, "duration"):
            self.compiler.compile(config)

    def test_metric_declarations_are_real_nonempty_names(self):
        metric_cases: tuple[object, ...] = ([], "mean", [True], [None], [{}], [""])
        for factory in (minimal_configuration, legacy_configuration, vertical_configuration):
            for metrics in metric_cases:
                config = factory()
                config["metrics"] = metrics
                with self.subTest(branch=factory.__name__, metrics=metrics):
                    with self.assertRaises(ExperimentValidationError):
                        self.compiler.compile(config)
        for factory in (minimal_configuration, vertical_configuration):
            config = factory()
            config["metrics"] = ["invented_metric"]
            with self.assertRaises(ExperimentValidationError):
                self.compiler.compile(config)
        config = vertical_configuration()
        config["metrics"] = ["packet_delivery"]
        with self.assertRaisesRegex(ExperimentValidationError, "task.metric"):
            self.compiler.compile(config)

    def test_replay_timestamps_cannot_be_implicitly_repaired(self):
        for timestamps in ([0, 1], [0, 0, 1], [0, 2, 1], [0, True, 2]):
            config = replay_configuration()
            config["dataset"]["timestamps_s"] = timestamps
            with self.subTest(timestamps=timestamps), self.assertRaises(ExperimentValidationError):
                self.compiler.compile(config)
        config = replay_configuration()
        config["dataset"]["sample_rate_hz"] = 1
        with self.assertRaisesRegex(ExperimentValidationError, "sampling rate"):
            self.compiler.compile(config)

    def test_minimal_sampling_is_required_and_consistent(self):
        sampling_changes: tuple[dict[str, object], ...] = (
            {"sample_rate": None},
            {"sample_rate": 0},
            {"sample_rate": True},
            {"timestamps": [0, 0.1]},
            {"timestamps": [0, 0, 0.2]},
            {"timestamps": [0, 0.2, 0.4]},
            {"samples": []},
        )
        for change in sampling_changes:
            config = minimal_configuration()
            config.update(change)
            with self.subTest(change=change), self.assertRaises(ExperimentValidationError):
                self.compiler.compile(config)
        config = minimal_configuration()
        del config["sample_rate"]
        config["timestamps"] = [0, 0.25, 1]
        self.assertEqual(self.compiler.compile(config).data_mode, "synthetic")

    def test_replay_source_cannot_be_mislabelled_synthetic(self):
        config = replay_configuration()
        config["dataset"]["origin"] = "SYNTHETIC"
        with self.assertRaises(ExperimentValidationError):
            self.compiler.compile(config)
        config = replay_configuration()
        config["perturbations"] = vertical_configuration()["perturbations"]
        with self.assertRaisesRegex(ExperimentValidationError, "actions require"):
            self.compiler.compile(config)

    def test_physical_model_names_and_units_cannot_hide_bad_parameters(self):
        for updates in (
            {"initial_voltage_v": 1},
            {"capacitance_f": 0},
            {"model": "capacitor_p0"},
            {"esr_ohm": -1},
            {"step_duration_s": True},
        ):
            config = vertical_configuration()
            config["resource"].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ExperimentValidationError):
                self.compiler.compile(config)

    def test_synthetic_model_units_are_checked_without_implicit_conversion(self):
        config = vertical_configuration()
        config["dataset"]["unit"] = "Pa"
        config["task"]["threshold_unit"] = "Pa"
        config["policy"]["threshold_unit"] = "Pa"
        with self.assertRaisesRegex(ExperimentValidationError, "declared model"):
            self.compiler.compile(config)

    def test_eeg_pipeline_declares_its_actual_model_units_and_sample_rate(self):
        config = vertical_configuration()
        config["vertical"] = "neurophysiology"
        config["dataset"].update(
            {
                "modality": "eeg",
                "channel": "EEG",
                "unit": "uV",
                "sample_rate_hz": 128,
                "sample_interval_s": 1 / 128,
            }
        )
        config["task"].update({"target": "EEG", "threshold_unit": "uV"})
        config["policy"]["threshold_unit"] = "uV"
        self.assertEqual(self.compiler.compile(config).configuration_kind, "vertical")
        config["dataset"]["sample_rate_hz"] = 32
        config["dataset"]["sample_interval_s"] = 1 / 32
        with self.assertRaisesRegex(ExperimentValidationError, ">= 64"):
            self.compiler.compile(config)

    def test_unreviewed_pipeline_fields_and_false_estimands_are_rejected(self):
        config = vertical_configuration()
        config["compute"]["free_energy"] = True
        with self.assertRaisesRegex(ExperimentValidationError, "unknown field"):
            self.compiler.compile(config)
        config = vertical_configuration()
        config["task"]["metric"] = "packet_delivery"
        with self.assertRaises(ExperimentValidationError):
            self.compiler.compile(config)

    def test_multiscale_history_declares_sample_windows_and_statistics(self):
        config = vertical_configuration()
        config["representation"] = {
            "name": "multiscale_history",
            "windows": [1, 4, 8],
            "window_unit": "samples",
            "statistics": ["mean", "std", "trend"],
        }
        self.compiler.compile(config)
        config["representation"]["windows"] = [1, 1]
        with self.assertRaises(ExperimentValidationError):
            self.compiler.compile(config)

    def test_hash_is_strict_canonical_json_and_changes_with_costs(self):
        config = vertical_configuration()
        first = self.compiler.compile(config)
        reordered = dict(reversed(list(config.items())))
        self.assertEqual(
            first.configuration_hash, self.compiler.compile(reordered).configuration_hash
        )
        encoded = json.dumps(
            first.configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        self.assertEqual(first.configuration_hash, hashlib.sha256(encoded).hexdigest())
        config["compute"]["operations_per_sample"] += 1
        self.assertNotEqual(
            first.configuration_hash, self.compiler.compile(config).configuration_hash
        )
        self.assertEqual(first.configuration["compute"]["operations_per_sample"], 100)

    def test_wrapped_and_file_configs_select_the_same_branch(self):
        config = vertical_configuration()
        wrapped = self.compiler.compile({"experiment": config})
        self.assertEqual(
            wrapped.configuration_hash, self.compiler.compile(config).configuration_hash
        )
        self.assertEqual(configuration_schema_name({"experiment": config}), "vertical.schema.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(self.compiler.compile_file(path).experiment_id, "pipeline")

    def test_wrapper_cannot_discard_conflicting_scientific_fields(self):
        with self.assertRaisesRegex(ExperimentValidationError, "wrapper"):
            self.compiler.compile({"experiment": vertical_configuration(), "seed": 99})

    def test_numerically_equal_policy_thresholds_are_duplicates(self):
        config = vertical_configuration()
        config["policy"]["thresholds"] = [80, 80.0]
        with self.assertRaisesRegex(ExperimentValidationError, "distinct"):
            self.compiler.compile(config)

    def test_schema_declaration_cannot_redirect_or_downgrade_validation(self):
        for declaration in ("world.schema.json", "missing.schema.json", True):
            config = vertical_configuration()
            config["$schema"] = declaration
            with (
                self.subTest(declaration=declaration),
                self.assertRaises(ExperimentValidationError),
            ):
                self.compiler.compile(config)


class SchemaResourceTests(unittest.TestCase):
    def test_bundled_parity_and_installed_resource_api(self):
        source = ROOT / "schemas"
        self.assertEqual(set(schema_names()), {path.name for path in source.glob("*.json")})
        for name in schema_names():
            with self.subTest(schema=name):
                self.assertEqual(load_schema(name), json.loads((source / name).read_text()))
                self.assertEqual(
                    (source / name).read_bytes(),
                    (ROOT / "src" / "physioworld" / "schemas" / name).read_bytes(),
                )
        with self.assertRaises(ExperimentValidationError):
            load_schema("../experiment.schema.json")

    @unittest.skipUnless(HAS_JSONSCHEMA, "optional jsonschema is not installed")
    def test_draft_schemas_and_named_branches_validate_offline(self):
        from jsonschema.validators import validator_for
        from referencing import Registry, Resource

        documents = {name: load_schema(name) for name in schema_names()}
        registry = Registry().with_resources(
            (document["$id"], Resource.from_contents(document)) for document in documents.values()
        )
        for schema in documents.values():
            validator_for(schema).check_schema(schema)
        entry = documents["experiment.schema.json"]
        validator = validator_for(entry)(entry, registry=registry)
        for config in (
            minimal_configuration(),
            legacy_configuration(),
            vertical_configuration(),
            replay_configuration(),
            {"experiment": vertical_configuration()},
        ):
            with self.subTest(kind=configuration_schema_name(config)):
                validator.validate(config)
                original = copy.deepcopy(config)
                raw = config.get("experiment", config)
                raw["seed"] = True
                self.assertFalse(validator.is_valid(config))
                self.assertNotEqual(config, original)
        legacy_branch = {
            "$schema": entry["$schema"],
            "$id": entry["$id"],
            "$defs": entry["$defs"],
            "$ref": "#/$defs/dataset_readiness",
        }
        legacy_validator = validator_for(entry)(legacy_branch, registry=registry)
        self.assertTrue(legacy_validator.is_valid(legacy_configuration()))
        self.assertFalse(legacy_validator.is_valid(vertical_configuration()))
        self.assertFalse(legacy_validator.is_valid(minimal_configuration()))


if __name__ == "__main__":
    unittest.main()

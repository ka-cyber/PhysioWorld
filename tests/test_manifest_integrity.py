"""Behavioral integrity tests using only the standard library."""

import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

from physioworld.provenance import (
    MANIFEST_VERSION,
    ManifestVerificationError,
    compare_scientific_outputs,
    configuration_snapshot,
    create_manifest,
    current_schema_digest,
    current_source_digest,
    hash_configuration,
    hash_file,
    load_manifest,
    prepare_run_directory,
    verify_manifest,
    write_json_output,
    write_output,
    write_report,
)


class ManifestIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.configuration = {"seed": 7, "model": {"gains": [1.0, {"value": 2}]}}

    def run_directory(self, name="run", *, configuration=None, status="COMPLETE", outputs=None):
        directory = prepare_run_directory(self.root / name)
        outputs = outputs or {"result": "result.json"}
        for key, relative_path in outputs.items():
            write_json_output(directory, relative_path, {"output": key, "value": 0.25})
        manifest = create_manifest(
            name,
            self.configuration if configuration is None else configuration,
            outputs=outputs,
            status=status,
        )
        manifest.write(directory)
        return directory

    def edit_manifest(self, directory, change):
        path = directory / "result_manifest.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        change(raw)
        path.write_text(json.dumps(raw, allow_nan=False), encoding="utf-8")

    def report(self, directory, value=1):
        return write_report(
            directory,
            question="Question?",
            hypothesis="Deterministic",
            estimand="Value",
            data="synthetic",
            method="seeded",
            results={"value": value},
        )

    def test_manifest_round_trip_and_verification_receipt(self):
        directory = self.run_directory()
        receipt = verify_manifest(directory / "result_manifest.json")
        self.assertTrue(receipt.passed)
        self.assertEqual(receipt.manifest.manifest_version, MANIFEST_VERSION)
        self.assertEqual(
            receipt.manifest.configuration_hash, hash_configuration(self.configuration)
        )
        self.assertEqual(
            receipt.manifest.output_hashes["result"], hash_file(directory / "result.json")
        )
        self.assertEqual(receipt.manifest_hash, hash_file(directory / "result_manifest.json"))
        self.assertEqual(configuration_snapshot(receipt.manifest), self.configuration)
        self.assertEqual(receipt.manifest, load_manifest(directory))
        self.assertNotIn("manifest", receipt.to_dict())
        self.assertEqual(
            receipt.to_dict(include_manifest=True)["manifest"], receipt.manifest.to_dict()
        )
        json.dumps(receipt.to_dict(include_manifest=True), allow_nan=False)

    def test_hash_is_canonical_strict_json_and_cannot_be_overridden(self):
        expected = hashlib.sha256(
            json.dumps(
                self.configuration,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.assertEqual(hash_configuration(self.configuration), expected)
        reordered = dict(reversed(list(self.configuration.items())))
        self.assertEqual(hash_configuration(reordered), expected)
        manifest = create_manifest("run", reordered, input_hashes={"configuration": expected})
        self.assertEqual(manifest.input_hashes["configuration"], expected)
        with self.assertRaisesRegex(ValueError, "configuration hash conflicts"):
            create_manifest("run", self.configuration, input_hashes={"configuration": "0" * 64})
        with self.assertRaisesRegex(ValueError, "configuration hash conflicts"):
            replace(manifest, input_hashes={"configuration": "1" * 64})

    def test_seed_is_inferred_and_conflicts_are_rejected(self):
        self.assertEqual(create_manifest("run", {"seed": 7}).seed, 7)
        self.assertEqual(create_manifest("run", {}, seed=7).seed, 7)
        with self.assertRaisesRegex(ValueError, "seed conflicts"):
            create_manifest("run", {"seed": 7}, seed=8)
        for invalid in (True, "7", 7.0):
            with self.subTest(seed=invalid), self.assertRaisesRegex(ValueError, "seed must"):
                create_manifest("run", {"seed": invalid})
        directory = self.run_directory()
        self.edit_manifest(directory, lambda raw: raw.update(seed=8))
        with self.assertRaisesRegex(ManifestVerificationError, "seed conflicts"):
            verify_manifest(directory)

    def test_all_nested_snapshot_state_is_detached_and_read_only(self):
        datasets = ({"id": "synthetic", "metadata": {"subjects": ["a"]}},)
        outputs = {"result": "result.json"}
        manifest = create_manifest("run", self.configuration, datasets=datasets, outputs=outputs)
        original_hash = manifest.configuration_hash
        self.configuration["model"]["gains"][1]["value"] = 999
        self.configuration["model"]["gains"].append(3)
        datasets[0]["metadata"]["subjects"].append("b")
        outputs["result"] = "different.json"
        self.assertEqual(manifest.configuration_hash, original_hash)
        self.assertEqual(manifest.configuration["model"]["gains"][1]["value"], 2)
        self.assertEqual(manifest.datasets[0]["metadata"]["subjects"], ("a",))
        self.assertEqual(manifest.outputs["result"], "result.json")
        self.assertIsInstance(manifest.configuration, MappingProxyType)
        self.assertIsInstance(manifest.configuration["model"], MappingProxyType)
        with self.assertRaises(TypeError):
            manifest.configuration["model"]["gains"][1]["value"] = 3
        with self.assertRaises(FrozenInstanceError):
            manifest.seed = 8
        snapshot = configuration_snapshot(manifest)
        snapshot["model"]["gains"][1]["value"] = 111
        serialized = manifest.to_dict()
        serialized["configuration"]["model"]["gains"].append(0)
        self.assertEqual(manifest.configuration["model"]["gains"], (1.0, {"value": 2}))
        directory = prepare_run_directory(self.root / "immutable")
        write_output(directory, "result.json", "result")
        manifest.write(directory)
        self.assertEqual(verify_manifest(directory).manifest.configuration_hash, original_hash)

    def test_nonfinite_unsupported_keys_and_cycles_are_rejected(self):
        cyclic = []
        cyclic.append(cyclic)
        for invalid in (
            float("nan"),
            float("inf"),
            -float("inf"),
            Path("private"),
            {1: "x"},
            cyclic,
        ):
            with (
                self.subTest(invalid_type=type(invalid).__name__),
                self.assertRaises((TypeError, ValueError)),
            ):
                create_manifest("run", {"nested": [{"value": invalid}]})
        directory = self.run_directory()
        with self.assertRaises(ValueError):
            write_json_output(directory, "nonfinite.json", {"x": float("nan")})
        self.assertFalse((directory / "nonfinite.json").exists())
        before = (directory / "result_manifest.json").read_bytes()
        with self.assertRaises(ValueError):
            self.report(directory, float("inf"))
        self.assertFalse((directory / "report.md").exists())
        self.assertEqual((directory / "result_manifest.json").read_bytes(), before)

    def test_literal_sensitive_keys_rejected_without_disclosing_values(self):
        for key in (
            "password",
            "TOKEN",
            "api_key",
            "api-key",
            "client_secret",
            "access_token",
            "private_key",
            "authorization",
            "credentials",
            "clientSecret",
            "accessToken",
            "privateKey",
            "secret_key",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError) as caught:
                create_manifest("run", {"connection": [{key: "DO-NOT-DISCLOSE"}]})
            self.assertNotIn("DO-NOT-DISCLOSE", str(caught.exception))
        for empty in (None, "", [], {}):
            create_manifest("run", {"password": empty})
        manifest = create_manifest(
            "references",
            {
                "data_reference": "user-provided/record.csv",
                "credential_env": "DATA_ACCESS",
                "api" + "_key_env": "SERVICE_ACCESS",
                "dataset_id": "credentialed-source",
            },
        )
        self.assertEqual(manifest.configuration["credential_env"], "DATA_ACCESS")
        directory = self.run_directory()
        with self.assertRaisesRegex(ValueError, "sensitive"):
            self.report(directory, {"token": "DO-NOT-DISCLOSE"})

    def test_configuration_tampering_fails_even_if_output_is_untouched(self):
        directory = self.run_directory()
        self.edit_manifest(directory, lambda raw: raw["configuration"]["model"].update(gains=[9]))
        with self.assertRaisesRegex(ManifestVerificationError, "configuration hash conflicts"):
            verify_manifest(directory)

    def test_configuration_file_must_match_manifest_and_be_present(self):
        for name, value in (
            ("changed", {"seed": 7}),
            ("numeric_type", {"seed": 7, "model": {"gains": [1, {"value": 2}]}}),
        ):
            with self.subTest(name=name):
                directory = self.run_directory(name)
                (directory / "configuration.json").write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaisesRegex(
                    ManifestVerificationError, "configuration.json differs"
                ):
                    verify_manifest(directory)
        directory = self.run_directory("missing_configuration")
        (directory / "configuration.json").unlink()
        with self.assertRaises(ManifestVerificationError):
            verify_manifest(directory)

    def test_output_tampering_and_missing_artifacts_fail(self):
        directory = self.run_directory()
        (directory / "result.json").write_text('{"value": 100}', encoding="utf-8")
        with self.assertRaisesRegex(ManifestVerificationError, "output hash mismatch: result"):
            verify_manifest(directory)
        (directory / "result.json").unlink()
        with self.assertRaises(ManifestVerificationError):
            verify_manifest(directory)

    def test_missing_declared_output_cannot_be_written_as_complete(self):
        directory = prepare_run_directory(self.root / "missing")
        manifest = create_manifest("missing", {}, outputs={"result": "missing.json"})
        with self.assertRaises(FileNotFoundError):
            manifest.write(directory)
        self.assertFalse((directory / "result_manifest.json").exists())
        self.assertFalse((directory / "configuration.json").exists())

    def test_output_hash_keys_are_exact_and_supplied_hashes_cannot_lie(self):
        for name, change in (
            ("missing_hash", lambda raw: raw["output_hashes"].pop("result")),
            ("extra_hash", lambda raw: raw["output_hashes"].update(unknown="0" * 64)),
        ):
            with self.subTest(name=name):
                directory = self.run_directory(name)
                self.edit_manifest(directory, change)
                with self.assertRaises(ManifestVerificationError):
                    verify_manifest(directory)
        directory = prepare_run_directory(self.root / "false_hash")
        write_output(directory, "result.json", "actual")
        manifest = create_manifest("false_hash", {}, outputs={"result": "result.json"})
        with self.assertRaisesRegex(ValueError, "output hash conflicts"):
            replace(manifest, output_hashes={"result": "0" * 64}).write(directory)

    def test_complete_status_required_and_partial_is_explicit(self):
        directory = self.run_directory(status="PARTIAL")
        with self.assertRaisesRegex(ManifestVerificationError, "COMPLETE"):
            verify_manifest(directory)
        receipt = verify_manifest(directory, allow_partial=True)
        self.assertEqual(receipt.to_dict()["manifest_status"], "PARTIAL")
        self.assertEqual(receipt.compatibility_overrides, ("allow_partial",))
        self.report(directory)
        self.assertEqual(verify_manifest(directory, allow_partial=True).manifest.status, "PARTIAL")
        for status in ("FAILED", "UNKNOWN"):
            with self.subTest(status=status):
                directory = self.run_directory(status.lower(), status=status)
                with self.assertRaises(ManifestVerificationError):
                    verify_manifest(directory, allow_partial=True)

    def test_exact_software_version_and_explicit_override(self):
        directory = self.run_directory()
        self.edit_manifest(directory, lambda raw: raw.update(software_version="0.1.1"))
        with self.assertRaisesRegex(ManifestVerificationError, "software version"):
            verify_manifest(directory)
        receipt = verify_manifest(directory, allow_version_mismatch=True)
        self.assertEqual(receipt.compatibility_overrides, ("allow_version_mismatch",))
        self.edit_manifest(directory, lambda raw: raw.update(source_digest="0" * 64))
        with self.assertRaisesRegex(ManifestVerificationError, "source digest"):
            verify_manifest(directory, allow_version_mismatch=True)

    def test_source_and_schema_bindings_are_required_and_separate(self):
        for field, option in (
            ("source_digest", "allow_source_mismatch"),
            ("schema_digest", "allow_schema_mismatch"),
        ):
            with self.subTest(field=field):
                directory = self.run_directory(field)
                self.edit_manifest(directory, lambda raw: raw.update({field: "0" * 64}))
                with self.assertRaisesRegex(ManifestVerificationError, field.replace("_", " ")):
                    verify_manifest(directory)
                receipt = verify_manifest(directory, **{option: True})
                self.assertEqual(receipt.compatibility_overrides, (option,))
                self.edit_manifest(directory, lambda raw: raw.pop(field))
                with self.assertRaisesRegex(ManifestVerificationError, "missing"):
                    verify_manifest(directory, **{option: True})

    def identity_tree(self, name):
        package = self.root / name / "physioworld"
        (package / "provenance").mkdir(parents=True)
        (package / "schemas").mkdir()
        (package / "provenance" / "manifest.py").write_text("# source\n", encoding="utf-8")
        (package / "runner.py").write_text("value = 1\n", encoding="utf-8")
        (package / "schemas" / "experiment.schema.json").write_text(
            '{"type":"object"}', encoding="utf-8"
        )
        return package

    def test_current_code_and_schema_changes_reject_same_version_replay(self):
        package = self.identity_tree("installed")
        with patch(
            "physioworld.provenance.manifest.__file__", str(package / "provenance" / "manifest.py")
        ):
            directory = self.run_directory()
            original = verify_manifest(directory).manifest
            (package / "runner.py").write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestVerificationError, "source digest"):
                verify_manifest(directory)
            with self.assertRaisesRegex(ValueError, "stale"):
                original.write(prepare_run_directory(self.root / "stale"))
            (package / "runner.py").write_text("value = 1\n", encoding="utf-8")
            (package / "schemas" / "experiment.schema.json").write_text(
                '{"type":"array"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ManifestVerificationError, "schema digest"):
                verify_manifest(directory)
            self.assertEqual(load_manifest(directory).software_version, original.software_version)

    def test_identity_digests_are_relocatable_and_exclude_bytecode(self):
        digests = []
        for name in ("first", "second"):
            package = self.identity_tree(name)
            (package / "__pycache__").mkdir()
            (package / "__pycache__" / "runner.pyc").write_bytes(name.encode())
            with patch(
                "physioworld.provenance.manifest.__file__",
                str(package / "provenance" / "manifest.py"),
            ):
                digests.append((current_source_digest(), current_schema_digest()))
        self.assertEqual(digests[0], digests[1])

    def test_legacy_manifest_is_loadable_but_cannot_claim_verified_replay(self):
        directory = self.run_directory()

        def remove_bindings(raw):
            for key in ("manifest_version", "source_digest", "schema_digest", "scientific_outputs"):
                raw.pop(key)

        self.edit_manifest(directory, remove_bindings)
        self.assertEqual(load_manifest(directory).manifest_version, 0)
        with self.assertRaisesRegex(ManifestVerificationError, "unversioned"):
            verify_manifest(directory, allow_source_mismatch=True, allow_schema_mismatch=True)
        self.edit_manifest(directory, lambda raw: raw.update(manifest_version=MANIFEST_VERSION + 1))
        with self.assertRaisesRegex(ManifestVerificationError, "unsupported"):
            verify_manifest(directory)

    def test_legacy_loader_preserves_optional_field_defaults(self):
        manifest = create_manifest("legacy", {}).to_dict()
        legacy = {
            key: manifest[key]
            for key in (
                "run_id",
                "created_at",
                "software_version",
                "python_version",
                "platform",
            )
        }
        path = self.root / "legacy.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        loaded = load_manifest(path)
        self.assertIsNone(loaded.seed)
        self.assertEqual(configuration_snapshot(loaded), {})
        self.assertEqual(loaded.status, "UNKNOWN")
        self.assertEqual(loaded.manifest_version, 0)
        with self.assertRaisesRegex(ManifestVerificationError, "unversioned"):
            verify_manifest(path)

    def test_escaping_and_ambiguous_artifact_names_are_rejected(self):
        directory = prepare_run_directory(self.root / "paths")
        for name in (
            "/tmp/outside",
            "../outside",
            "nested/../../outside",
            "nested\\outside",
            "C:/outside",
            "C:outside",
            "//server/file",
            "./file",
            "a//b",
            "a/",
            "",
            "a\x00b",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_manifest("paths", {}, outputs={"result": name})
            with self.subTest(writer=name), self.assertRaises(ValueError):
                write_output(directory, name, "payload")
        with self.assertRaisesRegex(ValueError, "manifest-owned"):
            create_manifest("paths", {}, outputs={"result": "configuration.json"})
        with self.assertRaisesRegex(ValueError, "unique"):
            create_manifest("paths", {}, outputs={"a": "same.json", "b": "same.json"})
        directory = self.run_directory()
        self.edit_manifest(directory, lambda raw: raw["outputs"].update(result="../outside.json"))
        with self.assertRaisesRegex(ManifestVerificationError, "unsafe"):
            verify_manifest(directory)

    def test_symlink_leaf_directory_and_ancestor_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "data.json").write_text("private", encoding="utf-8")
        alias = self.root / "alias"
        alias.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare_run_directory(alias / "new")
        directory = prepare_run_directory(self.root / "paths")
        (directory / "nested").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            write_output(directory, "nested/data.json", "overwrite")
        (directory / "leaf.json").symlink_to(outside / "data.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            write_output(directory, "leaf.json", "overwrite")
        manifest = create_manifest("paths", {}, outputs={"result": "leaf.json"})
        with self.assertRaisesRegex(ValueError, "symlink"):
            manifest.write(directory)
        self.assertEqual((outside / "data.json").read_text(encoding="utf-8"), "private")
        directory = self.run_directory()
        (directory / "result.json").unlink()
        (directory / "result.json").symlink_to(outside / "data.json")
        with self.assertRaisesRegex(ManifestVerificationError, "symlink"):
            verify_manifest(directory)
        linked_run = self.root / "linked_run"
        linked_run.symlink_to(directory, target_is_directory=True)
        with self.assertRaisesRegex(ManifestVerificationError, "symlink"):
            verify_manifest(linked_run)

    def test_manifest_and_configuration_symlinks_are_rejected(self):
        for filename in ("result_manifest.json", "configuration.json"):
            with self.subTest(filename=filename):
                directory = self.run_directory(filename.replace(".", "_"))
                saved = self.root / ("saved_" + filename)
                (directory / filename).rename(saved)
                (directory / filename).symlink_to(saved)
                with self.assertRaisesRegex(ManifestVerificationError, "symlink"):
                    verify_manifest(directory)

    def test_prepare_and_writers_refuse_run_reuse_and_overwrites(self):
        directory = prepare_run_directory(self.root / "empty")
        self.assertEqual(prepare_run_directory(directory), directory)
        artifact = write_output(directory, "nested/result.txt", "original")
        with self.assertRaises(FileExistsError):
            prepare_run_directory(directory)
        with self.assertRaises(FileExistsError):
            write_output(directory, "nested/result.txt", "replacement")
        self.assertEqual(artifact.read_text(encoding="utf-8"), "original")
        file_target = self.root / "file_target"
        file_target.write_text("file", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            prepare_run_directory(file_target)
        directory = self.run_directory()
        before = (directory / "result_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            load_manifest(directory).write(directory)
        self.assertEqual((directory / "result_manifest.json").read_bytes(), before)
        with self.assertRaises(ValueError):
            prepare_run_directory(self.root / ".." / "escape")

    def test_existing_configuration_is_never_overwritten(self):
        directory = prepare_run_directory(self.root / "configuration_exists")
        write_output(directory, "configuration.json", "owned by caller")
        with self.assertRaises(FileExistsError):
            create_manifest("run", {}).write(directory)
        self.assertEqual(
            (directory / "configuration.json").read_text(encoding="utf-8"), "owned by caller"
        )

    def test_report_updates_owned_manifest_and_accurate_hashes(self):
        directory = self.run_directory()
        before = verify_manifest(directory)
        path = self.report(directory, 1)
        after = verify_manifest(directory)
        self.assertEqual(after.manifest.output_hashes["report"], hash_file(path))
        self.assertEqual(after.manifest.scientific_outputs, ("result",))
        self.assertEqual(
            after.manifest.output_hashes["result"], before.manifest.output_hashes["result"]
        )
        self.assertNotEqual(after.manifest_hash, before.manifest_hash)
        self.report(directory, 2)
        updated = verify_manifest(directory)
        self.assertNotEqual(
            updated.manifest.output_hashes["report"], after.manifest.output_hashes["report"]
        )
        self.assertEqual(updated.manifest.output_hashes["report"], hash_file(path))
        path.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ManifestVerificationError, "output hash mismatch: report"):
            self.report(directory, 3)

    def test_unowned_report_is_not_overwritten_or_adopted(self):
        directory = self.run_directory()
        write_output(directory, "report.md", "caller report")
        before = (directory / "result_manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.report(directory)
        self.assertEqual((directory / "report.md").read_text(encoding="utf-8"), "caller report")
        self.assertEqual((directory / "result_manifest.json").read_bytes(), before)

    def test_comparison_excludes_report_and_compiled_plan_but_verifies_them(self):
        outputs = {"result": "result.json", "compiled_plan": "compiled_plan.json"}
        first, second = (
            self.run_directory("first", outputs=outputs),
            self.run_directory("second", outputs=outputs),
        )
        self.report(first, 1)
        self.report(second, 2)
        comparison = compare_scientific_outputs(first, second)
        self.assertTrue(comparison.passed)
        self.assertEqual(comparison.scientific_outputs, ("result",))
        json.dumps(comparison.to_dict(), allow_nan=False)
        self.assertEqual(
            verify_manifest(first).compare_scientific_outputs(verify_manifest(second)), comparison
        )
        self.assertEqual(
            load_manifest(first).compare_scientific_outputs(load_manifest(second)), comparison
        )
        (second / "compiled_plan.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(
            ManifestVerificationError, "output hash mismatch: compiled_plan"
        ):
            compare_scientific_outputs(first, second)
        self.edit_manifest(
            second,
            lambda raw: raw["output_hashes"].update(
                compiled_plan=hash_file(second / "compiled_plan.json")
            ),
        )
        self.assertEqual(compare_scientific_outputs(first, second), comparison)
        (second / "result.json").write_text("changed scientific output", encoding="utf-8")
        self.edit_manifest(
            second,
            lambda raw: raw["output_hashes"].update(result=hash_file(second / "result.json")),
        )
        with self.assertRaisesRegex(ManifestVerificationError, "scientific output hash mismatch"):
            compare_scientific_outputs(first, second)

    def test_scientific_output_selection_is_explicit_and_validated(self):
        for selection in (("report",), ("compiled_plan",), ("unknown",), ("result", "result")):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                create_manifest(
                    "run",
                    {},
                    outputs={
                        "result": "result.json",
                        "report": "report.md",
                        "compiled_plan": "compiled_plan.json",
                    },
                    scientific_outputs=selection,
                )
        first, second = self.run_directory("first"), self.run_directory("second")
        self.edit_manifest(second, lambda raw: raw.update(scientific_outputs=[]))
        with self.assertRaisesRegex(ManifestVerificationError, "selections differ"):
            compare_scientific_outputs(first, second)

    def test_strict_manifest_and_configuration_json_reject_nonfinite_and_duplicates(self):
        for name, payload in (
            ("nan", '{"seed":NaN}'),
            ("infinity", '{"seed":1e999}'),
            ("duplicate", '{"seed":7,"seed":7}'),
        ):
            with self.subTest(name=name):
                directory = self.run_directory(name)
                (directory / "configuration.json").write_text(payload, encoding="utf-8")
                with self.assertRaises(ManifestVerificationError):
                    verify_manifest(directory)
        directory = self.run_directory("duplicate_manifest")
        path = directory / "result_manifest.json"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('"seed": 7,', '"seed": 7, "seed": 7,', 1), encoding="utf-8")
        with self.assertRaisesRegex(ManifestVerificationError, "duplicate"):
            verify_manifest(directory)


if __name__ == "__main__":
    unittest.main()

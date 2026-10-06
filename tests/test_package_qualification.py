"""Packaging resource failures, archive parity, and recursive family discovery."""

import copy
import io
import json
import shutil
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from physioworld import qualification
from physioworld.experiments import compiler
from tools import check_schemas, smoke_install

ROOT = Path(__file__).resolve().parents[1]


def find_check(report, check, location):
    return next(
        item for item in report["checks"] if item["check"] == check and item["location"] == location
    )


class PackageResourceTests(unittest.TestCase):
    def test_doctor_reads_both_corpus_schemas_and_all_seven_fixtures(self):
        report = qualification.doctor()
        for name in ("corpus.schema.json", "corpus_result.schema.json"):
            self.assertEqual(find_check(report, "schema_json", name)["status"], "PASS")
        inventory = find_check(report, "bundled_golden_fixtures", "physioworld.corpus.golden")
        self.assertEqual(inventory["status"], "PASS", inventory)
        self.assertEqual(set(inventory["observed"]), set(smoke_install.REQUIRED_FIXTURES))
        for name in smoke_install.REQUIRED_FIXTURES:
            check = find_check(report, "golden_fixture_json", name)
            self.assertEqual(check["status"], "PASS", check)
            self.assertEqual(len(check["sha256"]), 64)
            self.assertGreater(check["observed"]["json_bytes"], 0)

    def test_either_missing_corpus_schema_fails_install_qualification(self):
        names = compiler.schema_names()
        for missing in ("corpus.schema.json", "corpus_result.schema.json"):
            with (
                self.subTest(schema=missing),
                patch.object(
                    compiler,
                    "schema_names",
                    return_value=tuple(name for name in names if name != missing),
                ),
            ):
                report = qualification.qualify_install()
            self.assertEqual(report["status"], "ERROR")
            self.assertIn(
                missing, find_check(report, "bundled_schemas", "physioworld.schemas")["missing"]
            )

    def test_each_missing_fixture_is_required_and_siblings_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "physioworld"
            golden = package / "corpus" / "golden"
            shutil.copytree(ROOT / "src" / "physioworld" / "corpus" / "golden", golden)
            for name in smoke_install.REQUIRED_FIXTURES:
                resource = golden / name
                payload = resource.read_bytes()
                resource.unlink()
                with (
                    self.subTest(fixture=name),
                    patch.object(qualification, "files", return_value=package),
                ):
                    checks = qualification._golden_fixture_checks()
                inventory = checks[0]
                self.assertEqual(inventory["status"], "ERROR", inventory)
                self.assertEqual(inventory["missing"], [name])
                self.assertEqual(sum(item["status"] == "PASS" for item in checks), 6)
                resource.write_bytes(payload)

            (golden / "artifact.json").unlink()
            with patch.object(qualification, "files", return_value=package):
                report = qualification.qualify_install()
            self.assertEqual(report["status"], "ERROR")
            self.assertEqual(
                find_check(report, "golden_fixture_json", "artifact.json")["status"], "ERROR"
            )

    def test_present_but_corrupt_or_misidentified_fixture_does_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "physioworld"
            golden = package / "corpus" / "golden"
            shutil.copytree(ROOT / "src" / "physioworld" / "corpus" / "golden", golden)
            resource = golden / "artifact.json"
            document = json.loads(resource.read_bytes())
            corruptions = (
                "{invalid JSON",
                json.dumps({**document, "fixture": "wrong"}),
                json.dumps({**document, "seed": True}),
                json.dumps({**document, "samples": [float("nan")]}),
            )
            for payload in corruptions:
                resource.write_text(payload, encoding="utf-8")
                with (
                    self.subTest(payload=payload),
                    patch.object(qualification, "files", return_value=package),
                ):
                    checks = qualification._golden_fixture_checks()
                failure = next(item for item in checks if item.get("location") == "artifact.json")
                self.assertEqual(failure["status"], "ERROR", failure)
                self.assertEqual(sum(item["status"] == "PASS" for item in checks), 7)


class ArchiveParityTests(unittest.TestCase):
    def test_wheel_detects_missing_changed_and_unreviewed_modules_or_resources(self):
        contents = smoke_install._package_files(ROOT)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "physioworld.whl"
            with zipfile.ZipFile(artifact, "w") as archive:
                for name, payload in contents.items():
                    archive.writestr(name, payload)
            self.assertEqual(smoke_install.check_archive(artifact, ROOT)["status"], "PASS")

            missing_schema = "physioworld/schemas/corpus_result.schema.json"
            missing_fixture = "physioworld/corpus/golden/artifact.json"
            changed = "physioworld/qualification.py"
            unexpected = "physioworld/unreviewed.py"
            with zipfile.ZipFile(artifact, "w") as archive:
                for name, payload in contents.items():
                    if name not in {missing_schema, missing_fixture}:
                        archive.writestr(name, b"changed" if name == changed else payload)
                archive.writestr(unexpected, "unreviewed = True\n")
            result = smoke_install.check_archive(artifact, ROOT)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["missing"], sorted([missing_schema, missing_fixture]))
            self.assertEqual(result["differing"], [changed])
            self.assertEqual(result["unexpected"], [unexpected])

    def test_sdist_requires_ledgers_governance_and_readmes_and_excludes_artifacts(self):
        contents = {
            f"src/{name}": payload for name, payload in smoke_install._package_files(ROOT).items()
        }
        contents.update(smoke_install._public_sdist_files(ROOT))
        required = (
            "src/physioworld/corpus/golden/README.md",
            "src/physioworld/representations/README.md",
            "docs/SCIENTIFIC_SOURCE_LEDGER.yaml",
            ".github/workflows/ci.yml",
            "CLINICAL_VALIDATION.md",
            "risk/risk_register.json",
        )
        for name in required:
            self.assertIn(name, contents)

        def write_archive(artifact, entries):
            with tarfile.open(artifact, "w:gz") as archive:
                for name, payload in entries.items():
                    member = tarfile.TarInfo(f"physioworld-0.1.0/{name}")
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))

        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "physioworld.tar.gz"
            write_archive(artifact, contents)
            self.assertEqual(smoke_install.check_archive(artifact, ROOT)["status"], "PASS")
            incomplete = {
                name: payload for name, payload in contents.items() if name not in required
            }
            incomplete["release/artifacts/nested/evidence.json"] = b"{}"
            write_archive(artifact, incomplete)
            result = smoke_install.check_archive(artifact, ROOT)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["missing"], sorted(required))
            self.assertEqual(result["forbidden"], ["release/artifacts/nested/evidence.json"])


class SchemaDiscoveryTests(unittest.TestCase):
    def test_recursive_corpus_configs_use_corpus_contract_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "schemas", root / "schemas")
            shutil.copytree(
                ROOT / "src" / "physioworld" / "schemas", root / "src" / "physioworld" / "schemas"
            )
            configs = root / "configs" / "corpus" / "nested"
            configs.mkdir(parents=True)
            path = configs / "campaign.json"
            configuration = json.loads(
                (ROOT / "configs" / "corpus" / "benchmark_corpus_level3.json").read_text()
            )
            path.write_text(json.dumps(configuration), encoding="utf-8")
            with (
                patch.object(check_schemas, "ROOT", root),
                patch(
                    "physioworld.benchmarks.corpus.run_corpus_campaign",
                    side_effect=AssertionError("validation executed a campaign"),
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(check_schemas.main(), 0, output.getvalue())
            self.assertIn("'corpus': 1", output.getvalue())
            invalid = copy.deepcopy(configuration)
            invalid["cases"] = ["unknown_case"]
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with (
                patch.object(check_schemas, "ROOT", root),
                redirect_stdout(io.StringIO()) as output,
            ):
                self.assertEqual(check_schemas.main(), 1)
            self.assertIn("configs/corpus/nested/campaign.json", output.getvalue())


if __name__ == "__main__":
    unittest.main()

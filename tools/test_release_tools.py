"""Reference tests for release integrity, immutable output and offline docs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import release_manifest as release
import smoke_install

import docs


class ReleaseToolTests(unittest.TestCase):
    def source_args(self, root: Path) -> argparse.Namespace:
        (root / "pyproject.toml").write_text('version = "0.1.0"\n', encoding="utf-8")
        (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "model.py").write_text("value = 1\n", encoding="utf-8")
        return argparse.Namespace(
            root=root,
            output=root / "bundle",
            evidence=None,
            release_level="DEVELOPMENT",
            reproducibility_level=0,
            container_digest=None,
            artifact=[],
        )

    def git_results(
        self,
        root: Path,
        *,
        commit: str | None = "a" * 40,
        status: str = "",
        status_available: bool = True,
    ) -> list[subprocess.CompletedProcess[str]]:
        results = [
            subprocess.CompletedProcess([], 0, str(root) + "\n", ""),
            subprocess.CompletedProcess([], 0 if commit else 128, (commit or "") + "\n", ""),
            subprocess.CompletedProcess([], 0 if status_available else 128, status, ""),
        ]
        if commit:
            results.append(subprocess.CompletedProcess([], 0, "v0.1.0\n", ""))
        return results

    def test_generated_hashes_bind_actual_inventory_and_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.source_args(root)
            release.generate(args)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertIsNone(manifest["git"]["commit"])
            self.assertIsNone(manifest["git"]["dirty"])
            self.assertFalse(manifest["git"]["status_available"])
            self.assertEqual(manifest["git"]["provenance_status"], "SOURCE_TREE")
            self.assertIn("without commit provenance", manifest["git"]["provenance_reason"])
            self.assertEqual(manifest["release_level"], "DEVELOPMENT")
            self.assertEqual(manifest["acceptance"]["status"], "INCOMPLETE")
            self.assertEqual(
                manifest["source"]["sha256"],
                hashlib.sha256(release.canonical(manifest["source"]["files"])).hexdigest(),
            )
            self.assertEqual(
                manifest["sbom"]["sha256"],
                hashlib.sha256((args.output / "sbom.json").read_bytes()).hexdigest(),
            )
            with self.assertRaises(ValueError):
                release.generate(args)
            args.output = root / "production"
            args.release_level = "PRODUCTION_READY_RESEARCH_PLATFORM"
            with self.assertRaises(ValueError):
                release.generate(args)

    def test_manifest_hashes_public_configs_requirements_governance_and_ledgers(self) -> None:
        inputs = {
            "source": (
                ".github/dependabot.yml",
                ".github/workflows/ci.yml",
                ".github/ISSUE_TEMPLATE/bug_report.yml",
                "CODEOWNERS",
                "requirements.txt",
                "requirements/requirements-core.txt",
                "requirements/requirements-scientific.txt",
                "requirements/requirements-dev.txt",
                "requirements/requirements-citation.txt",
                "requirements/citation.in",
                "requirements/README.md",
            ),
            "documentation": (
                "CLINICAL_VALIDATION.md",
                "risk/README.md",
                "risk/risk_register.json",
                "change_control/README.md",
                "clinical_protocols/README.md",
                "validation/README.md",
                "verification/README.md",
                "docs/ASSUMPTION_LEDGER.yaml",
                "docs/CLAIM_LEDGER.yaml",
                "docs/THEOREM_LEDGER.yaml",
                "docs/SCIENTIFIC_SOURCE_LEDGER.yaml",
                "docs/REFERENCE_ARTIFACTS.yaml",
                "docs/quality/data-governance.md",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.source_args(root)
            for paths in inputs.values():
                for name in paths:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"Public release input: {name}\n", encoding="utf-8")
            with patch.object(release, "installed_components", return_value=([], [])):
                release.generate(args)
            manifest = json.loads((args.output / "manifest.json").read_text())
            for section, paths in inputs.items():
                entries = {entry["path"]: entry for entry in manifest[section]["files"]}
                for name in paths:
                    with self.subTest(path=name):
                        content = (root / name).read_bytes()
                        self.assertEqual(
                            entries[name]["sha256"], hashlib.sha256(content).hexdigest()
                        )
                        self.assertEqual(entries[name]["bytes"], len(content))
                    self.assertEqual(
                        manifest[section]["sha256"],
                        hashlib.sha256(release.canonical(manifest[section]["files"])).hexdigest(),
                    )

    def test_experiments_and_artifacts_are_inventoried_and_bound_to_bundle_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.source_args(root)
            experiments = root / "experiments"
            experiments.mkdir()
            (experiments / "suite.json").write_text('{"profile":"fast"}\n', encoding="utf-8")
            first = root / "physioworld-0.1.0-py3-none-any.whl"
            second = root / "physioworld-0.1.0.tar.gz"
            first.write_bytes(b"wheel bytes")
            second.write_bytes(b"sdist bytes")
            args.artifact = [first, second]
            with patch.object(release, "installed_components", return_value=([], [])):
                release.generate(args)

            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["experiments"]["files"][0]["path"], "experiments/suite.json")
            self.assertEqual(
                [item["name"] for item in manifest["artifacts"]],
                [first.name, second.name],
            )
            self.assertEqual((args.output / first.name).read_bytes(), first.read_bytes())
            self.assertEqual((args.output / second.name).read_bytes(), second.read_bytes())
            checksums = {}
            for line in (args.output / "checksums.sha256").read_text().splitlines():
                digest, name = line.split("  ", 1)
                checksums[name] = digest
            self.assertEqual(
                set(checksums), {"manifest.json", "sbom.json", first.name, second.name}
            )
            for name, digest in checksums.items():
                self.assertEqual(
                    digest, hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                )

    def test_artifact_basename_collisions_reserved_names_and_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.source_args(root)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            first = left / "duplicate.whl"
            second = right / first.name
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            args.artifact = [first, second]
            with self.assertRaisesRegex(ValueError, "unique basenames"):
                release.generate(args)
            self.assertFalse(args.output.exists())

            reserved = root / "manifest.json"
            reserved.write_bytes(b"artifact")
            args.output = root / "reserved-bundle"
            args.artifact = [reserved]
            with self.assertRaisesRegex(ValueError, "reserved filename"):
                release.generate(args)
            self.assertFalse(args.output.exists())

            try:
                linked = root / "linked.whl"
                linked.symlink_to(first)
            except OSError:
                self.skipTest("symlinks unavailable")
            args.output = root / "symlink-bundle"
            args.artifact = [linked]
            with self.assertRaisesRegex(ValueError, "symlink"):
                release.generate(args)
            self.assertFalse(args.output.exists())

    def test_sdist_inputs_include_corpus_experiments_and_curated_governance_json(self) -> None:
        root = Path(__file__).resolve().parents[1]
        contents = smoke_install._public_sdist_files(root)
        for name in (
            "experiments/level3_suite.json",
            "docs/corpus_inventory.json",
            "change_control/policy.json",
            "clinical_protocols/research_readiness.json",
            "test_cases/golden_worlds.json",
            "validation/validation_scope.json",
        ):
            with self.subTest(path=name):
                self.assertIn(name, contents)

    def test_output_cannot_enter_new_inventory_trees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.source_args(root)
            for name in (
                ".github",
                "requirements",
                "docs",
                "risk",
                "change_control",
                "clinical_protocols",
                "validation",
                "verification",
                "experiments",
            ):
                with self.subTest(tree=name):
                    (root / name).mkdir()
                    args.output = root / name / "bundle"
                    with self.assertRaisesRegex(
                        ValueError, "output cannot be inside input inventory"
                    ):
                        release.generate(args)
                    self.assertFalse(args.output.exists())

    def test_git_state_does_not_attribute_parent_repository_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "extracted-source"
            root.mkdir()
            top_level = subprocess.CompletedProcess([], 0, str(parent) + "\n", "")
            with patch.object(release.subprocess, "run", return_value=top_level) as run:
                state = release.git_state(root)
            self.assertIsNone(state["commit"])
            self.assertIsNone(state["dirty"])
            self.assertEqual(state["tags_at_commit"], [])
            self.assertFalse(state["status_available"])
            self.assertEqual(state["provenance_status"], "SOURCE_TREE")
            self.assertIn("top-level does not match --root", state["provenance_reason"])
            run.assert_called_once_with(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )

    def test_git_state_reports_committed_dirty_uncommitted_and_unknown_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                ("clean", "a" * 40, "", True, False),
                ("dirty", "a" * 40, " M src/model.py\n", True, True),
                ("untracked", "a" * 40, "?? requirements.txt\n", True, True),
                ("uncommitted", None, "?? src/model.py\n", True, True),
                ("unknown", "a" * 40, "", False, None),
            )
            for name, commit, status, available, dirty in cases:
                with self.subTest(state=name):
                    results = self.git_results(
                        root, commit=commit, status=status, status_available=available
                    )
                    with patch.object(release.subprocess, "run", side_effect=results):
                        state = release.git_state(root)
                    self.assertEqual(state["commit"], commit)
                    self.assertIs(state["dirty"], dirty)
                    self.assertIs(state["status_available"], available)
                    self.assertEqual(state["tags_at_commit"], ["v0.1.0"] if commit else [])
                    self.assertEqual(state["provenance_status"], "GIT" if commit else "UNCOMMITTED")
                    if commit is None:
                        self.assertIn("No verified HEAD commit", state["provenance_reason"])

    def test_git_state_compares_resolved_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = self.git_results(root)
            results[0].stdout = str(root / "child" / "..") + "\n"
            with patch.object(release.subprocess, "run", side_effect=results):
                state = release.git_state(root)
            self.assertEqual(state["commit"], "a" * 40)
            self.assertIs(state["dirty"], False)

    def test_git_inspection_errors_have_explicit_source_tree_fallback(self) -> None:
        for error, reason in (
            (FileNotFoundError("git unavailable"), "Git inspection unavailable"),
            (subprocess.TimeoutExpired("git", 30), "Git inspection timed out"),
        ):
            with self.subTest(reason=reason):
                with patch.object(release.subprocess, "run", side_effect=error):
                    state = release.git_state(Path(tempfile.gettempdir()))
                self.assertIsNone(state["commit"])
                self.assertIsNone(state["dirty"])
                self.assertEqual(state["tags_at_commit"], [])
                self.assertFalse(state["status_available"])
                self.assertEqual(state["provenance_status"], "SOURCE_TREE")
                self.assertIn(reason, state["provenance_reason"])

    def test_external_source_tree_and_uncommitted_git_generate_development_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            for name in ("external-source", "nested-source", "uncommitted"):
                with self.subTest(tree=name):
                    root = parent / name
                    root.mkdir()
                    args = self.source_args(root)
                    args.output = parent / f"{name}-bundle"
                    if name == "uncommitted":
                        results = self.git_results(root, commit=None, status="?? src/model.py\n")
                    else:
                        results = [
                            subprocess.CompletedProcess(
                                [], 128 if name == "external-source" else 0, str(parent), ""
                            )
                        ]
                    with (
                        patch.object(release.subprocess, "run", side_effect=results),
                        patch.object(release, "installed_components", return_value=([], [])),
                    ):
                        release.generate(args)
                    manifest = json.loads((args.output / "manifest.json").read_text())
                    self.assertIsNone(manifest["git"]["commit"])
                    self.assertEqual(manifest["release_level"], "DEVELOPMENT")
                    self.assertEqual(manifest["acceptance"]["status"], "INCOMPLETE")
                    self.assertEqual(manifest["scientific_claim_status"]["status"], "NOT_EVALUATED")
                    self.assertEqual(
                        manifest["git"]["provenance_status"],
                        "UNCOMMITTED" if name == "uncommitted" else "SOURCE_TREE",
                    )
                    self.assertTrue(manifest["source"]["files"])

    def test_production_requires_all_gates_and_exact_root_clean_committed_git(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.source_args(root)
            args.release_level = "PRODUCTION_READY_RESEARCH_PLATFORM"
            artifact = root / "gate-result.txt"
            artifact.write_text("Observed test result\n", encoding="utf-8")
            args.evidence = root / "evidence.json"
            checks = [
                {"id": gate, "status": "PASS", "artifact": artifact.name}
                for gate in sorted(release.GATES)
            ]
            args.evidence.write_text(json.dumps({"checks": checks}), encoding="utf-8")
            cases = {
                "no-git": [subprocess.CompletedProcess([], 128, "", "")],
                "parent-git": [subprocess.CompletedProcess([], 0, str(root.parent), "")],
                "uncommitted": self.git_results(root, commit=None, status="?? src/model.py\n"),
                "dirty": self.git_results(root, status=" M src/model.py\n"),
                "untracked": self.git_results(root, status="?? requirements.txt\n"),
                "unknown-status": self.git_results(root, status_available=False),
                "missing-gate": self.git_results(root),
                "clean-committed": self.git_results(root),
            }
            for name, results in cases.items():
                with self.subTest(state=name):
                    args.output = root / f"production-{name}"
                    args.evidence.write_text(
                        json.dumps({"checks": checks[:-1] if name == "missing-gate" else checks}),
                        encoding="utf-8",
                    )
                    with (
                        patch.object(release.subprocess, "run", side_effect=results),
                        patch.object(release, "installed_components", return_value=([], [])),
                    ):
                        if name == "clean-committed":
                            release.generate(args)
                            manifest = json.loads((args.output / "manifest.json").read_text())
                            self.assertEqual(manifest["git"]["commit"], "a" * 40)
                            self.assertIs(manifest["git"]["dirty"], False)
                            self.assertEqual(manifest["acceptance"]["status"], "EVIDENCED")
                        else:
                            with self.assertRaisesRegex(ValueError, "clean committed tree"):
                                release.generate(args)
                            self.assertFalse(args.output.exists())

    def test_inventory_changes_when_input_changes_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            file = root / "src" / "model.py"
            file.write_text("first", encoding="utf-8")
            first = release.inventory(root, ("src",))["sha256"]
            file.write_text("second", encoding="utf-8")
            self.assertNotEqual(first, release.inventory(root, ("src",))["sha256"])
            try:
                (root / "src" / "escape").symlink_to(root / "uv.lock")
            except OSError:
                self.skipTest("symlinks unavailable")
            (root / "uv.lock").write_text("lock", encoding="utf-8")
            with self.assertRaises(ValueError):
                release.inventory(root, ("src",))

    def test_docs_detect_bad_fragments_and_build_escaped_linked_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            index = root / "docs" / "index.md"
            index.write_text("# Guide\n[Page](page.md#correct)\n", encoding="utf-8")
            page = root / "docs" / "page.md"
            page.write_text(
                "# Correct\n```python\n<script>alert(1)</script>\n```\n", encoding="utf-8"
            )
            paths = docs.documents(root)
            self.assertEqual(docs.check(root, paths, False), [])
            output = root / "site"
            docs.build(root, paths, output)
            self.assertIn("page.html#correct", (output / "docs" / "index.html").read_text())
            rendered = (output / "docs" / "page.html").read_text()
            self.assertNotIn("<script>", rendered)
            self.assertIn("&lt;script&gt;", rendered)
            index.write_text("[Bad](page.md#absent)\n", encoding="utf-8")
            self.assertTrue(docs.check(root, paths, False))
            with self.assertRaises(ValueError):
                docs.build(root, paths, output)

    def test_web_links_are_escaped_once_and_code_heading_fragments_match(self) -> None:
        link = docs.inline("[Example](https://example.org/?a=1&b=2)")
        self.assertIn("?a=1&amp;b=2", link)
        self.assertNotIn("&amp;amp;", link)
        self.assertEqual(docs.anchors("# Use `Model`\n"), {"use-model"})
        redirect = docs.WebRedirect()
        with self.assertRaises(ValueError):
            redirect.redirect_request(None, None, 302, "Found", {}, "file:///private")


if __name__ == "__main__":
    unittest.main()

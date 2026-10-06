"""Install a wheel and sdist in separate clean venvs outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path

COMMANDS = (
    "doctor",
    "datasets",
    "validate",
    "clinical-validate",
    "simulate",
    "replay",
    "benchmark",
    "run",
    "evaluate",
    "reproduce",
    "provenance",
    "report",
    "qualify-install",
    "qualify-plugin",
    "inspect",
    "schema",
    "scientific-suite",
    "version",
)
REQUIRED_SCHEMAS = (
    "experiment.schema.json",
    "world.schema.json",
    "vertical.schema.json",
    "result_manifest.schema.json",
    "protocol.schema.json",
    "corpus.schema.json",
    "corpus_result.schema.json",
)
REQUIRED_FIXTURES = (
    "normal_physiology.json",
    "artifact.json",
    "missing_modality.json",
    "packet_loss.json",
    "resource_depletion.json",
    "task_transition.json",
    "causal_intervention.json",
)
REQUIRED_CORPUS_RESOURCES = ("experiments.json", "registry.json")
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

CHECKSUM_EVIDENCE_SCRIPT = """
import hashlib
import sys
from pathlib import Path

output = Path(sys.argv[1]).resolve()
lines = [line for line in (output / 'checksums.sha256').read_text().splitlines() if line]
assert lines
names = []
for line in lines:
    digest, separator, name = line.partition('  ')
    assert separator and len(digest) == 64 and name and name not in names
    target = (output / name).resolve()
    assert not (output / name).is_symlink()
    assert target.is_file() and target.is_relative_to(output)
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    names.append(name)
assert set(names) == {'suite_result.json', 'experiment_registry.json', 'source_registry.json'}
"""


def _reviewed_package_path(path: Path) -> bool:
    return "__pycache__" not in path.parts and (
        path.suffix in {".py", ".json", ".yaml", ".yml", ".md"} or path.name == "py.typed"
    )


def _assert_safe_source_path(path: Path, root: Path) -> None:
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        raise ValueError(f"unsafe source symlink or escape: {relative}")


def _package_files(root: Path) -> dict[str, bytes]:
    source = root / "src"
    _assert_safe_source_path(source, root)
    files = {}
    for path in sorted((source / "physioworld").rglob("*")):
        _assert_safe_source_path(path, root)
        if path.is_file() and _reviewed_package_path(path):
            files[path.relative_to(source).as_posix()] = path.read_bytes()
    return files


def _public_sdist_files(root: Path) -> dict[str, bytes]:
    paths = [
        root / name
        for name in (
            "LICENSE",
            "CITATION.cff",
            "README.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "CODEOWNERS",
            "CLINICAL_VALIDATION.md",
            "release/README.md",
        )
    ]
    for directory in (
        ".github",
        "docs",
        "change_control",
        "clinical_protocols",
        "risk",
        "test_cases",
        "validation",
        "verification",
        "experiments",
    ):
        _assert_safe_source_path(root / directory, root)
        for path in (root / directory).rglob("*"):
            _assert_safe_source_path(path, root)
            if path.is_file() and path.suffix in {".md", ".yaml", ".yml", ".json"}:
                paths.append(path)
    for path in paths:
        _assert_safe_source_path(path, root)
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}


def check_archive(artifact: Path, root: Path) -> dict[str, object]:
    """Compare reviewed modules/resources with archive bytes before installation."""
    try:
        wheel = artifact.suffix == ".whl"
        prefix = "physioworld/" if wheel else "src/physioworld/"
        expected = {
            (name if wheel else f"src/{name}"): payload
            for name, payload in _package_files(root).items()
        }
        public = {} if wheel else _public_sdist_files(root)
        expected.update(public)
        observed = {}
        names = set()
        unsafe_members = []
        unsafe_symlinks = []

        def safe_archive_name(name: str) -> bool:
            parts = Path(name.replace("\\", "/")).parts
            return (
                bool(name)
                and not Path(name.replace("\\", "/")).is_absolute()
                and all(part not in {"", ".", ".."} for part in parts)
            )

        def reviewed(name: str) -> bool:
            names.add(name)
            return name in public or (
                name.startswith(prefix) and _reviewed_package_path(Path(name))
            )

        if wheel:
            with zipfile.ZipFile(artifact) as wheel_archive:
                for entry in wheel_archive.infolist():
                    if not safe_archive_name(entry.filename):
                        unsafe_members.append(entry.filename)
                        continue
                    mode = (entry.external_attr >> 16) & 0o170000
                    if mode == stat.S_IFLNK:
                        unsafe_symlinks.append(entry.filename)
                        continue
                    if not entry.is_dir() and reviewed(entry.filename):
                        if entry.filename in observed:
                            raise ValueError(f"duplicate archive member: {entry.filename}")
                        observed[entry.filename] = wheel_archive.read(entry)
        else:
            with tarfile.open(artifact, "r:gz") as source_archive:
                for member in source_archive.getmembers():
                    if not safe_archive_name(member.name):
                        unsafe_members.append(member.name)
                        continue
                    if member.issym() or member.islnk():
                        unsafe_symlinks.append(member.name)
                        continue
                    if not member.name.partition("/")[2]:
                        continue
                    name = member.name.partition("/")[2]
                    if member.isfile() and reviewed(name):
                        if name in observed:
                            raise ValueError(f"duplicate archive member: {name}")
                        handle = source_archive.extractfile(member)
                        if handle is None:
                            raise ValueError(f"unreadable archive member: {name}")
                        with handle:
                            observed[name] = handle.read()
        missing = sorted(expected.keys() - observed.keys())
        unexpected = sorted(observed.keys() - expected.keys())
        differing = sorted(
            name for name in expected.keys() & observed.keys() if expected[name] != observed[name]
        )
        forbidden = sorted(
            name
            for name in names
            if name == "release/artifacts" or name.startswith("release/artifacts/")
        )
        return {
            "id": "archive-parity",
            "status": "FAIL"
            if missing or unexpected or differing or forbidden or unsafe_members or unsafe_symlinks
            else "PASS",
            "reviewed_files": len(expected),
            "missing": missing,
            "unexpected": unexpected,
            "differing": differing,
            "forbidden": forbidden,
            "unsafe_members": sorted(unsafe_members),
            "unsafe_symlinks": sorted(unsafe_symlinks),
        }
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        return {"id": "archive-parity", "status": "FAIL", "reason": str(exc)}


CORPUS_EVIDENCE_SCRIPT = """
import hashlib
import json
import sys
from importlib.resources import files
from pathlib import Path

output = Path(sys.argv[1])
expected_cases = json.loads(sys.argv[2])
expected_fixtures = json.loads(sys.argv[3])
summary = json.loads((output / 'summary.json').read_text())
manifest = json.loads((output / 'corpus_manifest.json').read_text())
assert summary['status'] == manifest['status'] == 'COMPLETE_SYNTHETIC_ONLY'
assert summary['corpus_id'] == manifest['corpus_id'] == 'native_level3_synthetic'
assert not summary['failed_cases']
assert summary['case_count'] == len(expected_cases)
assert [case['case_id'] for case in summary['cases']] == manifest['cases'] == expected_cases
assert all(case['status'] in ('COMPLETE_SYNTHETIC_ONLY', 'COMPLETE_NUMERICAL_REPRODUCTION')
           for case in summary['cases'])
fixtures = files('physioworld').joinpath('corpus', 'golden')
assert manifest['fixture_hashes'] == {
    name: hashlib.sha256(fixtures.joinpath(name).read_bytes()).hexdigest()
    for name in expected_fixtures
}
assert set(manifest['case_hashes']) == {
    'cases/' + name + '/result.json' for name in expected_cases
}
for case in summary['cases']:
    name = 'cases/' + case['case_id'] + '/result.json'
    payload = (output / name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest['case_hashes'][name]
    assert json.loads(payload) == case
canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
assert hashlib.sha256(canonical).hexdigest() == summary['manifest_hash']
assert json.loads((output / 'corpus_result.schema.json').read_text()) == json.loads(
    files('physioworld').joinpath('schemas', 'corpus_result.schema.json').read_text())
"""


def smoke(artifact: Path, parent: Path, examples: Path | None) -> dict[str, object]:
    workspace = Path(tempfile.mkdtemp(prefix="physioworld-install-", dir=parent))
    env = workspace / "venv"
    venv.EnvBuilder(with_pip=True).create(env)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment["PYTHONNOUSERSITE"] = "1"
    results = [check_archive(artifact, Path(__file__).resolve().parents[1])]
    print(f"{artifact.name}: archive-parity: {results[0]['status']}")
    if results[0]["status"] != "PASS":
        print(json.dumps(results[0], indent=2), file=sys.stderr)

    def execute(label: str, *arguments: str) -> bool:
        result = subprocess.run(
            [str(python), *arguments],
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        results.append(
            {
                "id": label,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "exit_code": result.returncode,
            }
        )
        print(f"{artifact.name}: {label}: {results[-1]['status']}")
        if result.returncode:
            print(result.stdout[-3000:] + result.stderr[-3000:], file=sys.stderr)
        return result.returncode == 0

    if execute("install", "-m", "pip", "install", "--no-deps", str(artifact.resolve())):
        execute("dependency-consistency", "-m", "pip", "check")
        execute("module-entrypoint", "-I", "-m", "physioworld", "--help")
        execute(
            "imports",
            "-I",
            "-c",
            "import importlib, importlib.resources as r, pkgutil, physioworld; "
            "[importlib.import_module(m.name) for m in pkgutil.walk_packages(physioworld.__path__, 'physioworld.')]; "
            "assert physioworld.__file__.startswith(__import__('sys').prefix)",
        )
        execute(
            "bundled-resources",
            "-I",
            "-c",
            "import importlib.resources as r, json, sys; "
            "package = r.files('physioworld'); "
            "assert package.joinpath('py.typed').is_file(); "
            "schemas = package.joinpath('schemas'); "
            "[json.dumps(json.loads(schemas.joinpath(name).read_text()), allow_nan=False) "
            "for name in json.loads(sys.argv[1])]; "
            "fixtures = package.joinpath('corpus', 'golden'); "
            "corpus = package.joinpath('corpus'); "
            "[json.dumps(json.loads(corpus.joinpath(name).read_text()), allow_nan=False) "
            "for name in json.loads(sys.argv[3])]; "
            "documents = [json.loads(fixtures.joinpath(name).read_text()) "
            "for name in json.loads(sys.argv[2])]; "
            "[json.dumps(document, allow_nan=False) for document in documents]; "
            "assert all(document['fixture'] == name.removesuffix('.json') "
            "for document, name in zip(documents, json.loads(sys.argv[2]))); "
            "assert package.joinpath('representations', 'README.md').is_file(); "
            "assert fixtures.joinpath('README.md').is_file()",
            json.dumps(REQUIRED_SCHEMAS),
            json.dumps(REQUIRED_FIXTURES),
            json.dumps(REQUIRED_CORPUS_RESOURCES),
        )
        for command in COMMANDS:
            execute(f"help:{command}", "-I", "-m", "physioworld.cli", command, "--help")
        for command in ("doctor", "qualify-install", "benchmark"):
            execute(command, "-I", "-m", "physioworld.cli", command)
        suite_output = workspace / "scientific-suite"
        if execute(
            "scientific-suite-fast",
            "-I",
            "-m",
            "physioworld.cli",
            "scientific-suite",
            "--profile",
            "fast",
            "--output",
            str(suite_output),
        ):
            execute(
                "scientific-suite-checksums",
                "-I",
                "-c",
                CHECKSUM_EVIDENCE_SCRIPT,
                str(suite_output),
            )
        corpus_output = workspace / "native-corpus"
        if execute(
            "native-corpus",
            "-I",
            "-m",
            "physioworld.cli",
            "benchmark",
            "--family",
            "native_level3_synthetic",
            "--output",
            str(corpus_output),
        ):
            execute(
                "native-corpus-evidence",
                "-I",
                "-c",
                CORPUS_EVIDENCE_SCRIPT,
                str(corpus_output),
                json.dumps(NATIVE_CASES),
                json.dumps(REQUIRED_FIXTURES),
            )
        if examples:
            for name in ("minimal_world", "cardio_world", "neuro_world", "icu_world"):
                config = examples / f"{name}.json"
                if not config.is_file():
                    results.append({"id": name, "status": "FAIL", "reason": "example missing"})
                    continue
                local = workspace / config.name
                local.write_bytes(config.read_bytes())
                execute(
                    f"run:{name}",
                    "-I",
                    "-m",
                    "physioworld.cli",
                    "run",
                    str(local),
                    "--output",
                    str(workspace / name),
                )
                manifest = workspace / name / "result_manifest.json"
                if manifest.is_file():
                    execute(
                        f"reproduce:{name}",
                        "-I",
                        "-m",
                        "physioworld.cli",
                        "reproduce",
                        str(manifest),
                        "--output",
                        str(workspace / f"{name}-reproduced"),
                    )
                else:
                    results.append({"id": f"manifest:{name}", "status": "FAIL"})
    return {
        "artifact": artifact.name,
        "checks": results,
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--parent", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.parent.is_dir():
        parser.error("the external workspace parent must already exist")
    root = Path(__file__).resolve().parents[1]
    if args.parent.resolve().is_relative_to(root):
        parser.error("clean installations must be outside the source checkout")
    results = [
        smoke(artifact, args.parent.resolve(), args.examples.resolve() if args.examples else None)
        for artifact in args.artifacts
    ]
    if args.report:
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")
    return any(item["status"] != "PASS" for item in results)


if __name__ == "__main__":
    raise SystemExit(main())

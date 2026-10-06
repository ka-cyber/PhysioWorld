"""Generate CycloneDX 1.5 inventory and immutable, observed release evidence.

Uses only the standard library; run with the interpreter whose installed
environment is being inventoried. Output paths must be new directories.
Source trees without their own Git repository retain source hashes without
claiming commit provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (
    "DEVELOPMENT",
    "RESEARCH",
    "REFERENCE",
    "VALIDATED_RESEARCH",
    "CLINICAL_VALIDATION_PACKAGE",
    "PRODUCTION_READY_RESEARCH_PLATFORM",
)
IGNORED = {"__pycache__", ".git", ".venv", "build", "dist", ".pytest_cache"}
# Shared by hashing and output exclusion so new inputs cannot contain the bundle.
INVENTORY_PATHS = {
    "source": (
        "src",
        "tests",
        "tools",
        "scripts",
        "examples",
        "configs",
        ".github",
        "requirements",
        "requirements.txt",
        "pyproject.toml",
        "MANIFEST.in",
        "LICENSE",
        "CITATION.cff",
        "CODEOWNERS",
        ".pre-commit-config.yaml",
        ".gitignore",
    ),
    "documentation": (
        "docs",
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CLINICAL_VALIDATION.md",
        "CHANGELOG.md",
        "release/README.md",
        "risk",
        "change_control",
        "clinical_protocols",
        "validation",
        "verification",
    ),
    "schemas": ("schemas", "src/physioworld/schemas"),
    "benchmarks": ("benchmarks", "src/physioworld/benchmarks"),
    "dataset_manifests": ("datasets",),
    "experiments": ("experiments",),
}
RESERVED_ARTIFACT_FILENAMES = frozenset({"manifest.json", "sbom.json", "checksums.sha256"})
GATES = {
    "installation",
    "cli",
    "imports",
    "typing",
    "lint",
    "format",
    "platform_matrix",
    "dependency_lock",
    "security",
    "licenses",
    "scientific_regression",
    "numerical",
    "benchmarks",
    "schemas",
    "data_governance",
    "reproducibility",
    "independent_checks",
    "documentation",
    "public_hygiene",
    "verticals",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def inventory(root: Path, directories: tuple[str, ...]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for directory in directories:
        base = root / directory
        if base.is_symlink():
            raise ValueError(f"inventory refuses symlink/escape: {directory}")
        paths = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in paths:
            relative = path.relative_to(root)
            if path.is_symlink():
                raise ValueError(f"inventory refuses symlink/escape: {relative}")
            if not path.is_file() or any(
                part in IGNORED or part.endswith(".egg-info") for part in relative.parts
            ):
                continue
            if path.suffix in {".pyc", ".pyo", ".log"}:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"inventory refuses symlink/escape: {relative}")
            files.append(
                {"path": relative.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            )
    files.sort(key=lambda entry: entry["path"])
    return {
        "sha256": hashlib.sha256(canonical(files)).hexdigest(),
        "files": files,
        "hash_convention": "sha256 of sorted canonical JSON file inventory",
    }


def git_state(root: Path) -> dict[str, Any]:
    root = root.resolve()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False, timeout=30
        )

    def source_tree(reason: str) -> dict[str, Any]:
        return {
            "commit": None,
            "dirty": None,
            "tags_at_commit": [],
            "status_available": False,
            "provenance_status": "SOURCE_TREE",
            "provenance_reason": f"{reason}; using source-tree hashes without commit provenance",
        }

    try:
        top_level = git("rev-parse", "--show-toplevel")
        if top_level.returncode != 0 or not top_level.stdout.strip():
            return source_tree("Git working tree unavailable at --root")
        if Path(top_level.stdout.strip()).resolve() != root:
            return source_tree("Git top-level does not match --root")
        commit = git("rev-parse", "--verify", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
        tags = git("tag", "--points-at", "HEAD") if commit.returncode == 0 else None
        commit_id = (commit.stdout.strip() or None) if commit.returncode == 0 else None
        return {
            "commit": commit_id,
            "dirty": bool(status.stdout) if status.returncode == 0 else None,
            "tags_at_commit": tags.stdout.splitlines() if tags and tags.returncode == 0 else [],
            "status_available": status.returncode == 0,
            "provenance_status": "GIT" if commit_id else "UNCOMMITTED",
            "provenance_reason": (
                None
                if commit_id
                else "No verified HEAD commit exists at --root; source files remain uncommitted"
            ),
        }
    except OSError:
        return source_tree("Git inspection unavailable")
    except subprocess.TimeoutExpired:
        return source_tree("Git inspection timed out")


def installed_components() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    components = []
    versions = []
    for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name = dist.metadata["Name"]
        if not name:
            continue
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        purl = f"pkg:pypi/{normalized}@{dist.version}"
        component: dict[str, Any] = {
            "type": "library",
            "name": name,
            "version": dist.version,
            "bom-ref": purl,
            "purl": purl,
            "properties": [
                {"name": "physioworld:inventory-scope", "value": "installed-environment"}
            ],
        }
        expression = (
            dist.metadata["License-Expression"] if "License-Expression" in dist.metadata else None
        )
        classifiers = [
            line.split(" :: ")[-1]
            for line in dist.metadata.get_all("Classifier", [])
            if line.startswith("License ::")
        ]
        legacy = (dist.metadata["License"] if "License" in dist.metadata else "").strip()
        license_name = "; ".join(classifiers) or (legacy if legacy not in {"UNKNOWN", ""} else "")
        if expression:
            component["licenses"] = [{"expression": expression}]
        elif license_name:
            component["licenses"] = [{"license": {"name": license_name}}]
        else:
            component["properties"].append(
                {"name": "physioworld:license-evidence", "value": "missing"}
            )
        record = dist.read_text("RECORD")
        if record:
            component["properties"].append(
                {
                    "name": "physioworld:installed-record-sha256",
                    "value": hashlib.sha256(record.encode()).hexdigest(),
                }
            )
        for requirement in dist.requires or []:
            component["properties"].append(
                {"name": "physioworld:declared-requirement", "value": requirement}
            )
        components.append(component)
        versions.append({"name": name, "version": dist.version})
    return components, versions


def load_evidence(path: Path | None, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {"status": "NOT_EVALUATED", "claims": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("checks"), list):
        raise ValueError("evidence must be an object containing a checks array")
    checks = []
    seen = set()
    for check in value["checks"]:
        if check["id"] in seen or check["status"] not in {"PASS", "FAIL", "NOT_EVALUATED"}:
            raise ValueError("evidence check IDs must be unique and statuses explicit")
        seen.add(check["id"])
        item = dict(check)
        if item["status"] in {"PASS", "FAIL"}:
            artifact = Path(item["artifact"])
            artifact = artifact if artifact.is_absolute() else root / artifact
            if not artifact.is_file():
                raise ValueError(f"evidence artifact missing for {item['id']}")
            item["artifact"] = artifact.name
            item["artifact_sha256"] = sha256(artifact)
        checks.append(item)
    scientific = value.get("scientific_claim_status", {"status": "NOT_EVALUATED", "claims": []})
    return checks, scientific


def artifact_metadata(paths: list[Path]) -> list[dict[str, Any]]:
    """Validate artifact inputs and record their bytes before creating output."""
    seen: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        name = path.name
        if name in RESERVED_ARTIFACT_FILENAMES:
            raise ValueError(f"release artifact uses reserved filename: {name}")
        if not name or name in seen:
            raise ValueError(f"release artifacts must have unique basenames: {name or path}")
        if path.is_symlink():
            raise ValueError(f"release artifact symlink is not allowed: {path}")
        if not path.is_file():
            raise ValueError(f"release artifact missing: {name}")
        seen.add(name)
        artifacts.append({"name": name, "sha256": sha256(path), "bytes": path.stat().st_size})
    return artifacts


def copy_artifacts(paths: list[Path], artifacts: list[dict[str, Any]], output: Path) -> None:
    """Copy regular artifact bytes into a new bundle without following links."""
    for path, record in zip(paths, artifacts):
        destination = output / record["name"]
        with path.open("rb") as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if sha256(destination) != record["sha256"] or destination.stat().st_size != record["bytes"]:
            raise ValueError(f"release artifact changed while copying: {record['name']}")


def generate(args: argparse.Namespace) -> Path:
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output directory already exists; immutable evidence is never overwritten")
    if not output.parent.is_dir():
        raise ValueError("output parent must exist")
    # Refuse outputs inside any hashed input tree.
    for paths in INVENTORY_PATHS.values():
        for name in paths:
            if output.is_relative_to(root / name):
                raise ValueError(f"output cannot be inside input inventory {name}")
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (root / "pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise ValueError("project version is missing")
    version = match.group(1)
    state = git_state(root)
    checks, scientific = load_evidence(args.evidence, root)
    passed = {item["id"] for item in checks if item["status"] == "PASS"}
    if args.release_level == "PRODUCTION_READY_RESEARCH_PLATFORM":
        if GATES - passed or state["commit"] is None or state["dirty"] is not False:
            raise ValueError(
                "production level requires all evidenced gates and a clean committed tree"
            )
    if args.reproducibility_level > 0 and "reproducibility" not in passed:
        raise ValueError("a reproducibility level above 0 requires a PASS evidence artifact")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    components, dependencies = installed_components()
    inventories = {name: inventory(root, paths) for name, paths in INVENTORY_PATHS.items()}
    lock = root / "uv.lock"
    if not lock.is_file():
        raise ValueError("uv.lock is required")
    sbom_id = f"urn:uuid:{uuid.uuid4()}"
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": sbom_id,
        "version": 1,
        "metadata": {
            "timestamp": now,
            "component": {
                "type": "application",
                "name": "physioworld",
                "version": version,
                "bom-ref": f"pkg:pypi/physioworld@{version}",
                "purl": f"pkg:pypi/physioworld@{version}",
                "licenses": [{"license": {"id": "MIT"}}],
            },
        },
        "components": [
            component for component in components if component["name"].lower() != "physioworld"
        ],
    }
    artifacts = artifact_metadata(args.artifact)
    manifest = {
        "manifest_version": "1.0",
        "software": {"name": "physioworld", "version": version},
        "generated_at": now,
        "release_level": args.release_level,
        "git": state,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "os": platform.system(),
            "architecture": platform.machine(),
        },
        "dependency_versions": dependencies,
        "lockfile": {"path": "uv.lock", "sha256": sha256(lock)},
        **inventories,
        "dataset_evidence": "metadata inventory only; access, acquisition and study validation are not inferred",
        "container_digest": args.container_digest,
        "sbom": {
            "identifier": sbom_id,
            "path": "sbom.json",
            "sha256": hashlib.sha256(canonical(sbom)).hexdigest(),
        },
        "artifacts": artifacts,
        "reproducibility": {
            "level": args.reproducibility_level,
            "evidence_required_for_higher_levels": True,
        },
        "scientific_claim_status": scientific,
        "acceptance": {
            "checks": checks,
            "unassessed_gates": sorted(GATES - passed),
            "status": "EVIDENCED" if GATES <= passed else "INCOMPLETE",
        },
    }
    output.mkdir()
    copy_artifacts(args.artifact, artifacts, output)
    for name, value in (("sbom.json", sbom), ("manifest.json", manifest)):
        with (output / name).open("xb") as handle:
            handle.write(canonical(value))
    checksum_names = ["manifest.json", "sbom.json", *(item["name"] for item in artifacts)]
    checksum_lines = [f"{sha256(output / name)}  {name}" for name in checksum_names]
    (output / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="source tree to inventory; Git provenance requires a matching repository top-level",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-level", choices=LEVELS, default="DEVELOPMENT")
    parser.add_argument("--reproducibility-level", type=int, choices=range(6), default=0)
    parser.add_argument(
        "--evidence", type=Path, help="JSON checks with actual evidence artifact paths"
    )
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--container-digest", help="observed OCI digest, if a container was built")
    args = parser.parse_args(argv)
    if args.container_digest and not re.fullmatch(r"sha256:[0-9a-f]{64}", args.container_digest):
        parser.error("container digest must be an observed sha256 OCI digest")
    try:
        output = generate(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"release evidence: {exc}\n")
    print(f"Generated manifest.json, sbom.json and checksums.sha256 in {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

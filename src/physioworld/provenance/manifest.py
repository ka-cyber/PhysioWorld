"""Immutable run snapshots, exclusive artifact writes, and replay integrity checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping

from physioworld.version import __version__

MANIFEST_VERSION = 1
_MANIFEST_NAME = "result_manifest.json"
_CONFIGURATION_NAME = "configuration.json"
_NONSCIENTIFIC_OUTPUTS = {"report", "compiled_plan"}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SENSITIVE_KEYS = {
    "password",
    "passwd",
    "pwd",
    "token",
    "api_key",
    "apikey",
    "secret",
    "private_key",
    "access_key",
    "secret_key",
    "credentials",
    "authorization",
}


class ManifestVerificationError(ValueError):
    """An artifact cannot be trusted for replay under the requested policy."""


def _json_value(value: Any, active: set[int] | None = None) -> Any:
    """Copy a JSON tree without stringifying unsupported objects or losing key types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    if not isinstance(value, (Mapping, list, tuple)):
        raise TypeError(f"unsupported JSON value type: {type(value).__name__}")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic JSON value")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("JSON object keys must be strings")
            return {key: _json_value(item, active) for key, item in value.items()}
        return [_json_value(item, active) for item in value]
    finally:
        active.remove(identity)


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
            normalized = re.sub(r"[^a-z0-9]+", "_", separated.lower()).strip("_")
            sensitive = normalized in _SENSITIVE_KEYS or any(
                normalized.endswith("_" + name) for name in _SENSITIVE_KEYS
            )
            if sensitive and item not in (None, "", [], {}):
                # Never include the credential value in an exception or receipt.
                raise ValueError(f"literal sensitive configuration key is forbidden: {key}")
            _reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secrets(item)


def configuration_snapshot(configuration: Mapping[str, Any] | ResultManifest) -> dict[str, Any]:
    """Return a detached, readable dict/list tree from a mapping or manifest."""
    if isinstance(configuration, ResultManifest):
        configuration = configuration.configuration
    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a JSON object")
    snapshot = _json_value(configuration)
    _reject_secrets(snapshot)
    return snapshot


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_text(value: Any) -> str:
    return json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"


def hash_configuration(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(configuration_snapshot(configuration))).hexdigest()


def hash_file(path: str | os.PathLike[str], algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError("artifact names must be nonempty relative POSIX paths")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or PureWindowsPath(name).drive
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError(f"unsafe relative artifact name: {name}")
    return name


def _checked_path(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    if ".." in target.parts:
        raise ValueError("paths must not traverse parent directories")
    target = target.absolute()
    for ancestor in (*reversed(target.parents), target):
        if ancestor.is_symlink():
            raise ValueError(f"symlink paths are forbidden: {ancestor}")
    return target


def _artifact_path(directory: Path, name: str) -> Path:
    return _checked_path(directory / _safe_relative_name(name))


def _regular_file(path: Path) -> None:
    _checked_path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"artifact is not a regular file: {path.name}")


def _read_bytes(path: Path) -> bytes:
    _regular_file(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read()


def _artifact_hash(path: Path) -> str:
    _regular_file(path)
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_run_directory(path: str | os.PathLike[str]) -> Path:
    """Create a new run directory, or accept an empty one; reject symlink ancestry."""
    target = _checked_path(path)
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"run output is not a directory: {target}")
        if any(target.iterdir()):
            raise FileExistsError(f"run output directory must be empty: {target}")
    else:
        target.mkdir(parents=True, exist_ok=False)
    return _checked_path(target)


def write_output(
    directory: str | os.PathLike[str],
    relative_path: str,
    content: str | bytes,
) -> Path:
    """Exclusively create an artifact under an existing run directory."""
    target = _checked_path(directory)
    if not target.is_dir():
        raise FileNotFoundError(f"run output directory does not exist: {target}")
    path = _artifact_path(target, relative_path)
    if not isinstance(content, (str, bytes)):
        raise TypeError("output content must be text or bytes")
    payload = content.encode("utf-8") if isinstance(content, str) else content
    path.parent.mkdir(parents=True, exist_ok=True)
    _checked_path(path)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    return path


def write_json_output(
    directory: str | os.PathLike[str],
    relative_path: str,
    value: Any,
) -> Path:
    """Write strict JSON exclusively; sensitive literal keys are rejected."""
    snapshot = _json_value(value)
    _reject_secrets(snapshot)
    return write_output(directory, relative_path, _json_text(snapshot))


def _replace_owned(path: Path, content: str, expected_hash: str) -> None:
    """Atomically update a previously verified, owned report or manifest."""
    _checked_path(path)
    if _artifact_hash(path) != expected_hash:
        raise ManifestVerificationError(f"artifact changed during report update: {path.name}")
    descriptor, temporary = tempfile.mkstemp(prefix=".provenance-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        _checked_path(path)
        if _artifact_hash(path) != expected_hash:
            raise ManifestVerificationError(f"artifact changed during report update: {path.name}")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _tree_digest(root: Path, suffixes: tuple[str, ...]) -> str:
    _checked_path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"runtime identity tree is missing: {root.name}")
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlinks are forbidden in runtime identity trees: {path.name}")
        if path.suffix in suffixes:
            hashes[path.relative_to(root).as_posix()] = _artifact_hash(path)
    if not hashes:
        raise ValueError(f"runtime identity tree is empty: {root.name}")
    return hashlib.sha256(_canonical(hashes)).hexdigest()


def current_source_digest() -> str:
    """Bind installed package Python source using relative names, not checkout paths."""
    return _tree_digest(Path(__file__).absolute().parents[1], (".py",))


def current_schema_digest() -> str:
    """Bind the bundled schemas used by the installed compiler."""
    return _tree_digest(Path(__file__).absolute().parents[1] / "schemas", (".json", ".yaml"))


def _validate_hashes(hashes: Mapping[str, Any], label: str) -> None:
    for key, value in hashes.items():
        if not key or not isinstance(value, str) or not _DIGEST.fullmatch(value):
            raise ValueError(f"{label} must contain named SHA-256 hex digests")


@dataclass(frozen=True)
class ResultManifest:
    run_id: str
    created_at: str
    software_version: str
    python_version: str
    platform: str
    seed: int | None
    configuration: Mapping[str, Any]
    datasets: tuple[Mapping[str, Any], ...] = ()
    outputs: Mapping[str, str] = field(default_factory=dict)
    input_hashes: Mapping[str, str] = field(default_factory=dict)
    output_hashes: Mapping[str, str] = field(default_factory=dict)
    git_commit: str | None = None
    status: str = "COMPLETE"
    manifest_version: int = MANIFEST_VERSION
    source_digest: str | None = None
    schema_digest: str | None = None
    scientific_outputs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        configuration = configuration_snapshot(self.configuration)
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("manifest seed must be an integer or null")
        if "seed" in configuration:
            if configuration["seed"] is not None and type(configuration["seed"]) is not int:
                raise ValueError("configuration seed must be an integer or null")
            if configuration["seed"] != self.seed:
                raise ValueError("manifest seed conflicts with configuration seed")
        for name in ("run_id", "created_at", "software_version", "python_version", "platform"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if self.git_commit is not None and not isinstance(self.git_commit, str):
            raise TypeError("git_commit must be a string or null")
        if type(self.manifest_version) is not int or self.manifest_version < 0:
            raise ValueError("manifest_version must be a nonnegative integer")
        if self.status not in ("COMPLETE", "PARTIAL", "FAILED", "UNKNOWN"):
            raise ValueError("invalid manifest status")
        object.__setattr__(self, "configuration", _freeze(configuration))
        if not isinstance(self.datasets, (list, tuple)):
            raise TypeError("datasets must be a sequence of JSON objects")
        datasets = tuple(configuration_snapshot(item) for item in self.datasets)
        object.__setattr__(self, "datasets", _freeze(datasets))
        for name in ("outputs", "input_hashes", "output_hashes"):
            snapshot = configuration_snapshot(getattr(self, name))
            object.__setattr__(self, name, _freeze(snapshot))
        for key, name in self.outputs.items():
            if not key:
                raise ValueError("output keys must be nonempty")
            _safe_relative_name(name)
            if name in (_MANIFEST_NAME, _CONFIGURATION_NAME):
                raise ValueError("outputs cannot alias manifest-owned files")
        if len(set(self.outputs.values())) != len(self.outputs):
            raise ValueError("output paths must be unique")
        _validate_hashes(self.input_hashes, "input_hashes")
        _validate_hashes(self.output_hashes, "output_hashes")
        if set(self.output_hashes) - set(self.outputs):
            raise ValueError("output_hashes contains undeclared outputs")
        computed_hash = hash_configuration(configuration)
        if self.input_hashes.get("configuration", computed_hash) != computed_hash:
            raise ValueError("configuration hash conflicts with computed content")
        for name in ("source_digest", "schema_digest"):
            digest = getattr(self, name)
            if digest is not None and (
                not isinstance(digest, str) or not _DIGEST.fullmatch(digest)
            ):
                raise ValueError(f"{name} must be a SHA-256 hex digest")
        selection = self.scientific_outputs
        if selection is None:
            selection = tuple(key for key in self.outputs if key not in _NONSCIENTIFIC_OUTPUTS)
        if not isinstance(selection, (tuple, list)) or any(
            not isinstance(key, str) for key in selection
        ):
            raise TypeError("scientific_outputs must be a sequence of output keys")
        if len(set(selection)) != len(selection) or set(selection) - set(self.outputs):
            raise ValueError("scientific_outputs must contain unique declared output keys")
        if set(selection) & _NONSCIENTIFIC_OUTPUTS:
            raise ValueError("report and compiled_plan are not scientific outputs")
        object.__setattr__(self, "scientific_outputs", tuple(selection))

    @property
    def configuration_hash(self) -> str:
        return hash_configuration(self.configuration)

    def to_dict(self) -> dict[str, Any]:
        """Serialize explicitly: dataclasses.asdict cannot copy MappingProxyType."""
        return {item.name: _json_value(getattr(self, item.name)) for item in fields(self)}

    def write(self, directory: str | os.PathLike[str]) -> Path:
        target = _checked_path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = _artifact_path(target, _MANIFEST_NAME)
        configuration_path = _artifact_path(target, _CONFIGURATION_NAME)
        if path.exists() or configuration_path.exists():
            raise FileExistsError("manifest/configuration overwrite is forbidden")
        if self.manifest_version != MANIFEST_VERSION:
            raise ValueError("cannot write an unsupported manifest version")
        if self.software_version != __version__:
            raise ValueError("cannot write a manifest for a different software version")
        if (
            self.source_digest != current_source_digest()
            or self.schema_digest != current_schema_digest()
        ):
            raise ValueError("manifest source/schema binding is missing or stale")
        if self.input_hashes.get("configuration") != self.configuration_hash:
            raise ValueError("manifest must record its computed configuration hash")
        output_hashes = {
            key: _artifact_hash(_artifact_path(target, name)) for key, name in self.outputs.items()
        }
        for key, expected in self.output_hashes.items():
            if output_hashes[key] != expected:
                raise ValueError(f"output hash conflicts with artifact: {key}")
        manifest = replace(self, output_hashes=output_hashes)
        write_output(target, _CONFIGURATION_NAME, _json_text(configuration_snapshot(self)))
        return write_output(target, _MANIFEST_NAME, _json_text(manifest.to_dict()))

    def compare_scientific_outputs(
        self, other: ResultManifest | Verification | str | os.PathLike[str]
    ) -> ScientificOutputComparison:
        return compare_scientific_outputs(self, other)


def create_manifest(
    run_id: str,
    configuration: Mapping[str, Any],
    *,
    seed: int | None = None,
    datasets: tuple[Mapping[str, Any], ...] = (),
    outputs: Mapping[str, str] | None = None,
    input_hashes: Mapping[str, str] | None = None,
    status: str = "COMPLETE",
    scientific_outputs: tuple[str, ...] | None = None,
) -> ResultManifest:
    snapshot = configuration_snapshot(configuration)
    if seed is None:
        seed = snapshot.get("seed")
    computed_hash = hash_configuration(snapshot)
    hashes = dict(input_hashes or {})
    if hashes.get("configuration", computed_hash) != computed_hash:
        raise ValueError("configuration hash conflicts with computed content")
    hashes["configuration"] = computed_hash
    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).absolute().parent,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            or None
        )
    except OSError:
        git_commit = None
    return ResultManifest(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        software_version=__version__,
        python_version=sys.version,
        platform=platform.platform(),
        seed=seed,
        configuration=snapshot,
        datasets=datasets,
        outputs=outputs or {},
        input_hashes=hashes,
        git_commit=git_commit,
        status=status,
        source_digest=current_source_digest(),
        schema_digest=current_schema_digest(),
        scientific_outputs=scientific_outputs,
    )


def _strict_json(payload: bytes) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    return _json_value(
        json.loads(payload, parse_constant=reject_constant, object_pairs_hook=unique_object)
    )


def _manifest_path(path: str | os.PathLike[str]) -> Path:
    target = _checked_path(path)
    return _artifact_path(target, _MANIFEST_NAME) if target.is_dir() else target


def _load_payload(payload: bytes) -> ResultManifest:
    raw = _strict_json(payload)
    if not isinstance(raw, dict):
        raise ValueError("manifest root must be a JSON object")
    # Legacy documents remain inspectable, but verification requires current bindings.
    raw.setdefault("manifest_version", 0)
    raw.setdefault("status", "UNKNOWN")
    raw.setdefault("seed", None)
    raw.setdefault("configuration", {})
    return ResultManifest(**raw)


def load_manifest(directory: str | os.PathLike[str]) -> ResultManifest:
    """Load an immutable snapshot. Use verify_manifest before trusting it for replay."""
    return _load_payload(_read_bytes(_manifest_path(directory)))


@dataclass(frozen=True)
class Verification:
    """Successful integrity receipt, with the verified snapshot available to the runner."""

    manifest: ResultManifest
    manifest_hash: str
    compatibility_overrides: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return True

    def to_dict(self, *, include_manifest: bool = False) -> dict[str, Any]:
        result = {
            "status": "PASS",
            "verification": "MANIFEST_INTEGRITY",
            "manifest_version": self.manifest.manifest_version,
            "manifest_hash": self.manifest_hash,
            "manifest_status": self.manifest.status,
            "configuration_hash": self.manifest.configuration_hash,
            "software_version": self.manifest.software_version,
            "source_digest": self.manifest.source_digest,
            "schema_digest": self.manifest.schema_digest,
            "output_hashes": dict(self.manifest.output_hashes),
            "scientific_outputs": list(self.manifest.scientific_outputs or ()),
            "compatibility_overrides": list(self.compatibility_overrides),
        }
        if include_manifest:
            result["manifest"] = self.manifest.to_dict()
        return result

    def compare_scientific_outputs(
        self, other: ResultManifest | Verification | str | os.PathLike[str]
    ) -> ScientificOutputComparison:
        return compare_scientific_outputs(self, other)


def verify_manifest(
    path: str | os.PathLike[str],
    *,
    allow_partial: bool = False,
    allow_version_mismatch: bool = False,
    allow_source_mismatch: bool = False,
    allow_schema_mismatch: bool = False,
) -> Verification:
    """Verify every declared artifact and exact runtime identity; raise on failure."""
    try:
        manifest_path = _manifest_path(path)
        payload = _read_bytes(manifest_path)
        manifest = _load_payload(payload)
        if manifest.manifest_version != MANIFEST_VERSION:
            raise ValueError("unsupported or unversioned manifest; current bindings are required")
        overrides = []
        if manifest.status != "COMPLETE":
            if manifest.status == "PARTIAL" and allow_partial:
                overrides.append("allow_partial")
            else:
                raise ValueError(
                    "manifest status must be COMPLETE (PARTIAL requires allow_partial)"
                )
        checks = (
            (
                "software version",
                manifest.software_version,
                __version__,
                allow_version_mismatch,
                "allow_version_mismatch",
            ),
            (
                "source digest",
                manifest.source_digest,
                current_source_digest(),
                allow_source_mismatch,
                "allow_source_mismatch",
            ),
            (
                "schema digest",
                manifest.schema_digest,
                current_schema_digest(),
                allow_schema_mismatch,
                "allow_schema_mismatch",
            ),
        )
        for label, recorded, current, allowed, option in checks:
            if recorded is None:
                raise ValueError(f"manifest is missing its {label}")
            if recorded != current:
                if not allowed:
                    raise ValueError(f"manifest {label} is incompatible with the current runtime")
                overrides.append(option)
        if manifest.input_hashes.get("configuration") != manifest.configuration_hash:
            raise ValueError("configuration hash is missing or does not match content")
        configuration = configuration_snapshot(
            _strict_json(
                _read_bytes(
                    _artifact_path(manifest_path.parent, _CONFIGURATION_NAME),
                )
            )
        )
        if _canonical(configuration) != _canonical(manifest.configuration):
            raise ValueError("configuration.json differs from the manifest configuration")
        if set(manifest.output_hashes) != set(manifest.outputs):
            raise ValueError("output_hashes must match all declared output keys exactly")
        for key, name in manifest.outputs.items():
            actual = _artifact_hash(_artifact_path(manifest_path.parent, name))
            if actual != manifest.output_hashes[key]:
                raise ValueError(f"output hash mismatch: {key}")
        return Verification(manifest, hashlib.sha256(payload).hexdigest(), tuple(overrides))
    except (OSError, TypeError, ValueError, RecursionError) as exc:
        raise ManifestVerificationError(str(exc)) from exc


@dataclass(frozen=True)
class ScientificOutputComparison:
    scientific_outputs: tuple[str, ...]
    output_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_hashes", _freeze(dict(self.output_hashes)))

    @property
    def passed(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "comparison": "EXACT_SCIENTIFIC_OUTPUT_HASHES",
            "scientific_outputs": list(self.scientific_outputs),
            "output_hashes": dict(self.output_hashes),
        }


def compare_scientific_outputs(
    original: ResultManifest | Verification | str | os.PathLike[str],
    reproduced: ResultManifest | Verification | str | os.PathLike[str],
) -> ScientificOutputComparison:
    """Compare only selected output bytes; paths are verified before comparison."""

    def snapshot(value: ResultManifest | Verification | str | os.PathLike[str]) -> ResultManifest:
        if isinstance(value, Verification):
            return value.manifest
        if isinstance(value, ResultManifest):
            return value
        return verify_manifest(value).manifest

    first, second = snapshot(original), snapshot(reproduced)
    selection = tuple(sorted(first.scientific_outputs or ()))
    if set(selection) != set(second.scientific_outputs or ()):
        raise ManifestVerificationError("scientific output selections differ")
    hashes = {}
    for key in selection:
        expected, actual = first.output_hashes.get(key), second.output_hashes.get(key)
        if expected is None or actual is None or expected != actual:
            raise ManifestVerificationError(f"scientific output hash mismatch: {key}")
        hashes[key] = expected
    return ScientificOutputComparison(selection, hashes)


def write_report(
    directory: str | os.PathLike[str],
    *,
    question: str,
    hypothesis: str,
    estimand: str,
    data: str,
    method: str,
    results: Mapping[str, Any],
    uncertainty: Mapping[str, Any] | None = None,
    limitations: list[str] | None = None,
    failure_cases: list[str] | None = None,
) -> Path:
    target = _checked_path(directory)
    receipt = verify_manifest(target, allow_partial=True)
    manifest = receipt.manifest
    result_snapshot = configuration_snapshot(results)
    uncertainty_snapshot = configuration_snapshot(uncertainty or {})
    lines = [
        f"# Experiment report: {manifest.run_id}",
        "",
        "## Question",
        question,
        "",
        "## Hypothesis",
        hypothesis,
        "",
        "## Estimand",
        estimand,
        "",
        "## Data",
        data,
        "",
        "## Configuration and method",
        method,
        "",
        "## Results",
        "```json",
        _json_text(result_snapshot).rstrip(),
        "```",
        "",
        "## Uncertainty",
        "```json",
        _json_text(uncertainty_snapshot).rstrip(),
        "```",
        "",
        "## Limitations",
        *[f"- {item}" for item in limitations or ["No limitations supplied."]],
        "",
        "## Failure cases",
        *[f"- {item}" for item in failure_cases or ["No failure cases recorded."]],
        "",
        "## Reproducibility",
        "Manifest: `result_manifest.json`",
        f"Software version: `{manifest.software_version}`",
        f"Run status: `{manifest.status}`",
        "",
    ]
    path = _artifact_path(target, "report.md")
    content = "\n".join(lines)
    if "report" in manifest.outputs:
        if manifest.outputs["report"] != "report.md":
            raise ValueError("the report output must be report.md")
        _replace_owned(path, content, manifest.output_hashes["report"])
    else:
        if "report.md" in manifest.outputs.values():
            raise ValueError("report.md is already owned by another output")
        write_output(target, "report.md", content)
    outputs = {**manifest.outputs, "report": "report.md"}
    output_hashes = {**manifest.output_hashes, "report": _artifact_hash(path)}
    updated = replace(manifest, outputs=outputs, output_hashes=output_hashes)
    _replace_owned(target / _MANIFEST_NAME, _json_text(updated.to_dict()), receipt.manifest_hash)
    return path

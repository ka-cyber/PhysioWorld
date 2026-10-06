"""Explicit, versioned plugin interfaces and evidence-scoped software qualification.

Importing or registering a plugin never runs qualification or discovers extensions.
Reference qualification requires reviewer-supplied, live unittest suites; declarations
and ordinary smoke callables cannot supply that evidence. No tier is clinical,
hardware, causal, or regulatory validation.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import marshal
import math
import re
import sys
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping
from uuid import uuid4

from .version import __version__

PLUGIN_API_VERSION = "1.0"
METADATA_SCHEMA_VERSION = "1.0"
QUALIFICATION_SCHEMA_VERSION = "1.0"


class PluginKind(str, Enum):
    PHYSIOLOGY_MODEL = "physiology_model"
    SENSOR = "sensor"
    DEVICE = "device"
    COMMUNICATION_CHANNEL = "communication_channel"
    RESOURCE = "resource"
    REPRESENTATION = "representation"
    TASK = "task"
    ESTIMATOR = "estimator"
    POLICY = "policy"
    CAUSAL_ESTIMATOR = "causal_estimator"
    DATASET_ADAPTER = "dataset_adapter"
    BENCHMARK = "benchmark"
    VISUALIZATION = "visualization"
    REPORT_GENERATOR = "report_generator"


class AssuranceTier(str, Enum):
    INSTALLED_UNQUALIFIED = "INSTALLED_UNQUALIFIED"
    INTERFACE_VERIFIED = "INTERFACE_VERIFIED"
    REFERENCE_QUALIFIED = "REFERENCE_QUALIFIED"


class CheckCategory(str, Enum):
    API = "api"
    SCHEMA = "schema"
    UNIT = "unit"
    REFERENCE = "reference"
    NUMERICAL = "numerical"
    REPRODUCIBILITY = "reproducibility"
    SECURITY = "security"


@dataclass(frozen=True)
class PluginContract:
    """Required keyword-callable entry points, checked without invoking them."""

    kind: PluginKind
    methods: Mapping[str, tuple[str, ...]]
    api_version: str = PLUGIN_API_VERSION


_ENTRYPOINTS: dict[PluginKind, dict[str, tuple[str, ...]]] = {
    PluginKind.PHYSIOLOGY_MODEL: {
        "initialize": ("parameters", "seed"),
        "advance": ("state", "actions", "dt_s"),
        "observe": ("state",),
    },
    PluginKind.SENSOR: {"observe": ("state", "time_s")},
    PluginKind.DEVICE: {
        "configure": ("configuration",),
        "start": (),
        "read": ("max_samples",),
        "stop": (),
    },
    PluginKind.COMMUNICATION_CHANNEL: {"transmit": ("packets", "seed")},
    PluginKind.RESOURCE: {"transition": ("state", "action", "dt_s")},
    PluginKind.REPRESENTATION: {"transform": ("history",)},
    PluginKind.TASK: {"evaluate": ("prediction", "target")},
    PluginKind.ESTIMATOR: {"fit": ("data",), "predict": ("data",)},
    PluginKind.POLICY: {"act": ("observation", "resource_state", "time_s")},
    PluginKind.CAUSAL_ESTIMATOR: {
        "identify": ("problem",),
        "estimate": ("data", "identification"),
    },
    PluginKind.DATASET_ADAPTER: {"manifest": (), "load": ("configuration",)},
    PluginKind.BENCHMARK: {"execute": ("name", "seed")},
    PluginKind.VISUALIZATION: {"render": ("data", "configuration")},
    PluginKind.REPORT_GENERATOR: {"generate": ("result",)},
}
PLUGIN_CONTRACTS: Mapping[PluginKind, PluginContract] = MappingProxyType(
    {
        kind: PluginContract(kind, MappingProxyType(dict(methods)))
        for kind, methods in _ENTRYPOINTS.items()
    }
)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _json(value: Any) -> Any:
    """Copy JSON data, rejecting nonfinite values and non-JSON objects."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _equal_json(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _equal_json(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _equal_json(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("evidence requires a lowercase SHA-256 digest")
    return value


def _envelope(envelope: Mapping[str, Any]) -> Mapping[str, tuple[Any, ...]]:
    normalized = {}
    for name, bound in envelope.items():
        _text(name, "operating-envelope dimension")
        if not isinstance(bound, (tuple, list)) or not bound:
            raise ValueError("operating-envelope bounds must be nonempty lists or tuples")
        values = tuple(bound)
        if len(values) == 2 and all(type(v) in (int, float) for v in values):
            low, high = map(float, values)
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError("numeric operating-envelope bounds must be finite and ordered")
            normalized[name] = (low, high)
        elif all(isinstance(v, str) and v.strip() for v in values):
            normalized[name] = values
        else:
            raise ValueError("operating-envelope entries must be numeric bounds or categories")
    return MappingProxyType(normalized)


def _release(value: str) -> tuple[int, ...]:
    # Deliberately conservative stdlib-only release/specifier grammar. Never
    # guess the ordering of unsupported prereleases, markers, extras or URLs.
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d+)*", value):
        raise ValueError(f"unsupported stable release version: {value!r}")
    parts = tuple(map(int, value.split(".")))
    return parts + (0,) * max(0, 4 - len(parts))


def _requirement(requirement: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)\s*(.*)", requirement)
    if not match or not match[2]:
        raise ValueError("requirements must declare a distribution and version bounds")
    bounds = []
    for bound in match[2].split(","):
        parsed = re.fullmatch(r"\s*(==|!=|>=|<=|>|<)\s*(\d+(?:\.\d+)*)\s*", bound)
        if not parsed:
            raise ValueError(f"unsupported requirement: {requirement!r}")
        bounds.append((parsed[1], parsed[2]))
    return match[1], tuple(bounds)


def _satisfies(actual: str, bounds: tuple[tuple[str, str], ...]) -> bool:
    release = _release(actual)
    for operator, expected in bounds:
        target = _release(expected)
        if not {
            "==": release == target,
            "!=": release != target,
            ">=": release >= target,
            "<=": release <= target,
            ">": release > target,
            "<": release < target,
        }[operator]:
            return False
    return True


def _schema_declaration(schema: Any, *, root: bool = True) -> None:
    """Check the supported declaration grammar, not arbitrary schema behavior."""
    if not isinstance(schema, dict):
        raise ValueError("schema declaration must be a JSON object")
    if root:
        _text(schema.get("$id"), "schema $id")
    if schema.get("type") not in {
        "object",
        "array",
        "string",
        "integer",
        "number",
        "boolean",
        "null",
    }:
        raise ValueError("schema requires a single supported JSON type")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("schema properties must be a JSON object")
    for name, subschema in properties.items():
        _text(name, "schema property")
        _schema_declaration(subschema, root=False)
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(n, str) for n in required):
        raise ValueError("schema required must be an array of property names")
    if len(set(required)) != len(required) or not set(required).issubset(properties):
        raise ValueError("required schema properties must be unique and declared")
    if "items" in schema:
        _schema_declaration(schema["items"], root=False)
    additional = schema.get("additionalProperties", True)
    if isinstance(additional, dict):
        _schema_declaration(additional, root=False)
    elif not isinstance(additional, bool):
        raise ValueError("additionalProperties must be boolean or a supported schema")


@dataclass(frozen=True)
class EvidenceArtifact:
    """An explicitly requested local artifact with a pinned, observed checksum."""

    path: str
    sha256: str
    description: str

    def __post_init__(self) -> None:
        _text(self.path, "artifact path")
        _sha256(self.sha256)
        _text(self.description, "artifact description")

    @classmethod
    def from_file(cls, path: str | Path, description: str) -> EvidenceArtifact:
        resolved = Path(path).resolve()
        return cls(str(resolved), hashlib.sha256(resolved.read_bytes()).hexdigest(), description)

    def verify(self) -> None:
        if hashlib.sha256(Path(self.path).read_bytes()).hexdigest() != self.sha256:
            raise ValueError(f"artifact checksum mismatch: {self.path}")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256, "description": self.description}


@dataclass(frozen=True)
class PluginProvenance:
    source_uri: str
    revision: str
    publisher: str
    artifacts: tuple[EvidenceArtifact, ...]

    def __post_init__(self) -> None:
        for name in ("source_uri", "revision", "publisher"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if not self.artifacts or not all(isinstance(a, EvidenceArtifact) for a in self.artifacts):
            raise ValueError("provenance requires pinned source artifacts")
        if len({str(Path(a.path).resolve()) for a in self.artifacts}) != len(self.artifacts):
            raise ValueError("duplicate provenance artifact")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_uri": self.source_uri,
            "revision": self.revision,
            "publisher": self.publisher,
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


@dataclass(frozen=True)
class PluginMetadata:
    """Declarations, never execution evidence. Legacy positional fields are retained."""

    plugin_id: str
    kind: PluginKind
    version: str
    api_compatibility: str
    dependencies: tuple[str, ...]
    scientific_evidence: tuple[str, ...]
    numerical_test_status: str
    license: str
    security_provenance: str
    operating_envelope: Mapping[str, Any] = field(default_factory=dict)
    api_version: str = PLUGIN_API_VERSION
    provenance: PluginProvenance | None = None
    schema_version: str = METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PluginKind(self.kind))
        if self.schema_version != METADATA_SCHEMA_VERSION:
            raise ValueError("unsupported plugin metadata schema version")
        for name in ("plugin_id", "version", "api_compatibility", "api_version"):
            _text(getattr(self, name), name)
        if not re.fullmatch(r"\d+\.\d+\.\d+", self.version):
            raise ValueError("plugin version must be a stable major.minor.patch release")
        for name in ("dependencies", "scientific_evidence"):
            values = getattr(self, name)
            if isinstance(values, str):
                raise ValueError(f"{name} must be a sequence, not a string")
            values = tuple(values)
            for value in values:
                _text(value, name)
            object.__setattr__(self, name, values)
        for name in ("numerical_test_status", "license", "security_provenance"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if self.provenance is not None and not isinstance(self.provenance, PluginProvenance):
            raise ValueError("provenance must be PluginProvenance")
        object.__setattr__(self, "operating_envelope", _envelope(self.operating_envelope))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plugin_id": self.plugin_id,
            "kind": self.kind.value,
            "version": self.version,
            "api_version": self.api_version,
            "api_compatibility": self.api_compatibility,
            "dependencies": list(self.dependencies),
            "scientific_evidence": list(self.scientific_evidence),
            "numerical_test_status": self.numerical_test_status,
            "license": self.license,
            "security_provenance": self.security_provenance,
            "operating_envelope": {k: list(v) for k, v in self.operating_envelope.items()},
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PluginMetadata:
        values = dict(data)
        if values.get("schema_version") != METADATA_SCHEMA_VERSION:
            raise ValueError("missing or unsupported metadata schema version")
        provenance = values.get("provenance")
        if provenance is not None:
            values["provenance"] = PluginProvenance(
                provenance["source_uri"],
                provenance["revision"],
                provenance["publisher"],
                tuple(EvidenceArtifact(**artifact) for artifact in provenance["artifacts"]),
            )
        return cls(**values)


@dataclass(frozen=True)
class QualificationScope:
    plugin_id: str
    plugin_version: str
    api_version: str
    metadata_sha256: str
    implementation_sha256: str
    interface_sha256: str
    dependency_versions: tuple[tuple[str, str], ...]
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "plugin_version": self.plugin_version,
            "api_version": self.api_version,
            "metadata_sha256": self.metadata_sha256,
            "implementation_sha256": self.implementation_sha256,
            "interface_sha256": self.interface_sha256,
            "dependency_versions": dict(self.dependency_versions),
            "python_version": self.python_version,
        }


def _metadata(plugin: Any) -> PluginMetadata:
    metadata = getattr(plugin, "metadata", None)
    if not isinstance(metadata, PluginMetadata):
        raise ValueError("plugin exposes no PluginMetadata")
    return metadata


def _source_artifacts(plugin: Any) -> tuple[EvidenceArtifact, ...]:
    source = inspect.getsourcefile(type(plugin))
    if source is None:
        raise ValueError("plugin defining source file is unavailable")
    defining = EvidenceArtifact.from_file(source, "plugin defining Python source")
    provenance = _metadata(plugin).provenance
    if provenance is None:
        return (defining,)
    covered = {str(Path(a.path).resolve()) for a in provenance.artifacts}
    method_sources = {
        inspect.getsourcefile(getattr(plugin, name))
        for name in PLUGIN_CONTRACTS[_metadata(plugin).kind].methods
    }
    if any(
        path is None or str(Path(path).resolve()) not in covered
        for path in method_sources | {source}
    ):
        raise ValueError("provenance does not cover plugin defining source")
    for artifact in provenance.artifacts:
        artifact.verify()
    return provenance.artifacts


def qualification_scope(plugin: Any) -> QualificationScope:
    """Observe the current scope without importing any dependency or running tests."""
    metadata = _metadata(plugin)
    dependencies = []
    for requirement in metadata.dependencies:
        name, bounds = _requirement(requirement)
        actual = version(name)
        if not _satisfies(actual, bounds):
            raise ValueError(f"dependency version mismatch: {name} {actual}")
        dependencies.append((name, actual))
    artifacts = sorted(_source_artifacts(plugin), key=lambda artifact: artifact.path)
    entrypoints = {}
    for name in PLUGIN_CONTRACTS[metadata.kind].methods:
        method = getattr(plugin, name)
        code = getattr(method, "__code__", None)
        if code is None:
            raise ValueError("reference scope requires inspectable Python entry points")
        entrypoints[name] = {
            "code_sha256": hashlib.sha256(marshal.dumps(code)).hexdigest(),
            "defaults": _json(getattr(method, "__defaults__", None)),
            "keyword_defaults": _json(getattr(method, "__kwdefaults__", None)),
        }
    return QualificationScope(
        metadata.plugin_id,
        metadata.version,
        metadata.api_version,
        _digest(metadata.to_dict()),
        _digest({"artifacts": [a.to_dict() for a in artifacts], "entrypoints": entrypoints}),
        _digest(getattr(plugin, "schemas", None)),
        tuple(sorted(dependencies)),
        sys.version,
    )


class EvidenceTestCase(unittest.TestCase):
    """unittest case that records actual comparisons for the qualification report.

    Every test must exercise an evidence assertion. A no-op test, an empty suite,
    a skipped test, and a metadata-returning callable are not passing evidence.
    """

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.observations: list[dict[str, Any]] = []

    def assertEvidenceEqual(self, actual: Any, expected: Any, *, label: str) -> None:
        _text(label, "comparison label")
        actual, expected = _json(actual), _json(expected)
        self.observations.append(
            {"label": label, "comparison": "equal", "actual": actual, "expected": expected}
        )
        self.assertTrue(_equal_json(actual, expected), label)

    def assertEvidenceClose(
        self,
        actual: float,
        expected: float,
        *,
        abs_tol: float,
        label: str,
        rel_tol: float = 0.0,
    ) -> None:
        _text(label, "comparison label")
        values = (actual, expected, abs_tol, rel_tol)
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            self.fail("numerical comparisons require finite real numbers")
        if abs_tol < 0 or rel_tol < 0:
            self.fail("numerical tolerances must be nonnegative")
        self.observations.append(
            {
                "label": label,
                "comparison": "numerical",
                "actual": actual,
                "expected": expected,
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
            }
        )
        self.assertTrue(math.isclose(actual, expected, abs_tol=abs_tol, rel_tol=rel_tol), label)


@dataclass(frozen=True)
class QualificationSuite:
    suite_id: str
    category: CheckCategory
    factory: Callable[[Any], unittest.TestSuite]
    artifact: EvidenceArtifact
    reference: str

    def __post_init__(self) -> None:
        _text(self.suite_id, "suite ID")
        object.__setattr__(self, "category", CheckCategory(self.category))
        if not callable(self.factory) or not isinstance(self.artifact, EvidenceArtifact):
            raise ValueError("suite requires a callable factory and pinned test artifact")
        _text(self.reference, "test oracle/reference")


@dataclass(frozen=True)
class PluginQualificationPlan:
    scope: QualificationScope
    suites: tuple[QualificationSuite, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, QualificationScope):
            raise ValueError("plan requires an observed QualificationScope")
        object.__setattr__(self, "suites", tuple(self.suites))
        if not all(isinstance(s, QualificationSuite) for s in self.suites):
            raise ValueError("plan requires typed qualification suites")
        if len({s.suite_id for s in self.suites}) != len(self.suites):
            raise ValueError("duplicate qualification suite ID")


@dataclass(frozen=True)
class QualificationCheck:
    check_id: str
    status: str
    detail: str
    category: str = "declaration"
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.check_id, "check ID")
        _text(self.detail, "check detail")
        if self.status not in {"PASS", "FAIL", "NOT_EVALUATED"}:
            raise ValueError("invalid qualification check status")
        object.__setattr__(self, "evidence", MappingProxyType(_json(dict(self.evidence))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "detail": self.detail,
            "category": self.category,
            "evidence": _json(dict(self.evidence)),
        }


@dataclass(frozen=True)
class PluginQualification:
    plugin_id: str
    metadata: PluginMetadata
    checks: tuple[QualificationCheck, ...]
    assurance_tier: str
    evidence_status: str
    scope: QualificationScope | None = None
    run_id: str = field(default_factory=lambda: str(uuid4()))
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = QUALIFICATION_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        """True only for complete reference qualification, not interface verification."""
        required = {f"suite:{category.value}" for category in CheckCategory} | {
            "api-version",
            "dependencies",
            "kind-contract",
            "schema-declarations",
            "source-scope",
            "scientific-evidence",
            "license",
            "security-provenance",
            "operating-envelope",
            "provenance",
            "plan-scope",
            "scope-stability",
        }
        return (
            self.assurance_tier == AssuranceTier.REFERENCE_QUALIFIED.value
            and self.scope is not None
            and required.issubset({c.check_id for c in self.checks})
            and all(c.status == "PASS" for c in self.checks if c.category != "smoke")
            and not any(c.status == "FAIL" for c in self.checks)
        )

    @property
    def interface_verified(self) -> bool:
        return self.assurance_tier in {
            AssuranceTier.INTERFACE_VERIFIED.value,
            AssuranceTier.REFERENCE_QUALIFIED.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plugin_id": self.plugin_id,
            "metadata": self.metadata.to_dict(),
            "checks": [c.to_dict() for c in self.checks],
            "assurance_tier": self.assurance_tier,
            "evidence_status": self.evidence_status,
            "passed": self.passed,
            "interface_verified": self.interface_verified,
            "scope": self.scope.to_dict() if self.scope else None,
            "run_id": self.run_id,
            "evaluated_at": self.evaluated_at,
        }


@dataclass
class PluginRegistry:
    """An explicit in-process registry. Registering confers no assurance."""

    plugins: dict[str, Any] = field(default_factory=dict)

    def register(self, plugin: Any) -> None:
        metadata = _metadata(plugin)
        if metadata.plugin_id in self.plugins:
            raise ValueError(f"duplicate plugin ID: {metadata.plugin_id}")
        self.plugins[metadata.plugin_id] = plugin

    def get(self, plugin_id: str) -> Any:
        try:
            return self.plugins[plugin_id]
        except KeyError as exc:
            raise KeyError(f"unknown plugin: {plugin_id}") from exc

    def qualify(
        self,
        plugin_id: str,
        *,
        tests: tuple[Callable[[Any], Any], ...] = (),
        plan: PluginQualificationPlan | None = None,
    ) -> PluginQualification:
        return qualify_plugin(self.get(plugin_id), tests=tests, plan=plan)


def load_plugin(reference: str) -> Any:
    """Explicitly import module:attribute (and instantiate classes).

    This executes Python in the calling process. It is not a sandbox or a scan.
    No entry-point discovery, directory scanning or dependency importing occurs.
    """
    if reference == "builtin:verticals":
        return BuiltInVerticalPlugin()
    if not isinstance(reference, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", reference, re.ASCII
    ):
        raise ValueError("plugin reference must use module:attribute syntax")
    module_name, attribute = reference.split(":")
    module = importlib.import_module(module_name)
    try:
        plugin = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"plugin attribute not found: {reference}") from exc
    return plugin() if inspect.isclass(plugin) else plugin


def _interface_checks(plugin: Any, metadata: PluginMetadata) -> list[QualificationCheck]:
    checks = []

    def check(name: str, operation: Callable[[], Any]) -> None:
        try:
            evidence = operation() or {}
        except Exception as exc:
            checks.append(
                QualificationCheck(name, "FAIL", f"{type(exc).__name__}: {exc}", "interface")
            )
        else:
            checks.append(
                QualificationCheck(
                    name, "PASS", "current interface check passed", "interface", evidence
                )
            )

    def api() -> dict[str, str]:
        if metadata.api_version != PLUGIN_API_VERSION:
            raise ValueError(
                f"plugin API mismatch: {metadata.api_version}; expected {PLUGIN_API_VERSION}"
            )
        name, bounds = _requirement(metadata.api_compatibility)
        if name != "physioworld" or not _satisfies(__version__, bounds):
            raise ValueError(
                f"PhysioWorld version {__version__} does not satisfy {metadata.api_compatibility}"
            )
        return {"plugin_api": PLUGIN_API_VERSION, "physioworld_version": __version__}

    def dependencies() -> dict[str, str]:
        observed = {}
        for requirement in metadata.dependencies:
            name, bounds = _requirement(requirement)
            actual = version(name)
            if not _satisfies(actual, bounds):
                raise ValueError(f"dependency mismatch: {name} {actual}; expected {requirement}")
            observed[name] = actual
        return observed

    def contract() -> dict[str, Any]:
        for name, arguments in PLUGIN_CONTRACTS[metadata.kind].methods.items():
            method = getattr(plugin, name, None)
            if not callable(method):
                raise ValueError(f"{metadata.kind.value} requires callable {name}")
            signature = inspect.signature(method)
            if any(
                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in signature.parameters.values()
            ):
                raise ValueError(f"{name} must declare explicit contract arguments")
            signature.bind(**dict.fromkeys(arguments))
            if inspect.iscoroutinefunction(method):
                raise ValueError(f"{name} must be synchronous for plugin API {PLUGIN_API_VERSION}")
        return {
            "kind": metadata.kind.value,
            "methods": list(PLUGIN_CONTRACTS[metadata.kind].methods),
        }

    def schemas() -> dict[str, Any]:
        # Declaration syntax is part of the interface. Behavioral schema checks
        # must still run in the SCHEMA suite before reference qualification.
        declared = getattr(plugin, "schemas", None)
        methods = PLUGIN_CONTRACTS[metadata.kind].methods
        if not isinstance(declared, Mapping) or set(declared) != set(methods):
            raise ValueError("schemas must cover exactly the kind's entry points")
        for name, arguments in methods.items():
            pair = declared[name]
            if not isinstance(pair, Mapping) or set(pair) != {"input", "output"}:
                raise ValueError(f"{name} requires input and output JSON Schema declarations")
            for direction in ("input", "output"):
                schema = _json(pair[direction])
                _schema_declaration(schema)
            input_schema = pair["input"]
            if input_schema["type"] != "object" or not set(arguments).issubset(
                input_schema.get("properties", {})
            ):
                raise ValueError(f"{name} input schema must describe all contract arguments")
        return {"schemas_sha256": _digest(declared)}

    check("api-version", api)
    check("dependencies", dependencies)
    check("kind-contract", contract)
    check("schema-declarations", schemas)
    return checks


def _flatten(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    cases = []
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            cases.extend(_flatten(test))
        elif isinstance(test, unittest.TestCase):
            cases.append(test)
        else:
            raise ValueError("suite contains a non-unittest test")
    return cases


class _EvidenceResult(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.observed: list[dict[str, Any]] = []

    def startTest(self, test: unittest.TestCase) -> None:
        super().startTest(test)
        if isinstance(test, EvidenceTestCase):
            test.observations.clear()

    def stopTest(self, test: unittest.TestCase) -> None:
        if isinstance(test, EvidenceTestCase):
            self.observed.append({"test_id": test.id(), "observations": _json(test.observations)})
        super().stopTest(test)


def _run_suite(plugin: Any, spec: QualificationSuite) -> QualificationCheck:
    evidence: dict[str, Any] = {"artifact": spec.artifact.to_dict(), "reference": spec.reference}
    try:
        spec.artifact.verify()
        artifact_path = Path(spec.artifact.path).resolve()
        factory_source = inspect.getsourcefile(spec.factory)
        if factory_source is None or Path(factory_source).resolve() != artifact_path:
            raise ValueError("suite factory is not covered by its pinned test artifact")
        suite = spec.factory(plugin)
        if not isinstance(suite, unittest.TestSuite):
            raise ValueError("suite factory must return unittest.TestSuite")
        cases = _flatten(suite)
        if not cases or not all(isinstance(test, EvidenceTestCase) for test in cases):
            raise ValueError("suite requires nonempty EvidenceTestCase tests")
        if len({test.id() for test in cases}) != len(cases):
            raise ValueError("duplicate test IDs within qualification suite")
        for test in cases:
            method = getattr(test, test._testMethodName)
            source = inspect.getsourcefile(method)
            if source is None or Path(source).resolve() != artifact_path:
                raise ValueError("test methods are not covered by the pinned test artifact")
        result = _EvidenceResult()
        suite.run(result)
        evidence.update(
            {
                "tests_run": result.testsRun,
                "test_ids": [test.id() for test in cases],
                "comparisons": result.observed,
                "skipped": len(result.skipped),
                "failures": len(result.failures),
                "errors": len(result.errors),
                "unexpected_successes": len(result.unexpectedSuccesses),
            }
        )
        evidence["failure_details"] = [
            {"test_id": test.id(), "message": message.strip().splitlines()[-1]}
            for test, message in result.failures + result.errors
        ]
        spec.artifact.verify()
        if not result.wasSuccessful() or result.expectedFailures or result.skipped:
            raise ValueError("tests failed, errored, skipped, or used expected-failure status")
        if result.testsRun != len(cases) or len(result.observed) != len(cases):
            raise ValueError("not all declared tests executed")
        if any(not item["observations"] for item in result.observed):
            raise ValueError("test completed without an evidence comparison")
        for item in result.observed:
            for observation in item["observations"]:
                if observation["comparison"] == "equal":
                    matches = _equal_json(observation["actual"], observation["expected"])
                elif observation["comparison"] == "numerical":
                    matches = math.isclose(
                        observation["actual"],
                        observation["expected"],
                        abs_tol=observation["abs_tol"],
                        rel_tol=observation["rel_tol"],
                    )
                else:
                    matches = False
                if not matches:
                    raise ValueError("recorded evidence comparison failed")
        if spec.category == CheckCategory.NUMERICAL and any(
            not any(o["comparison"] == "numerical" for o in item["observations"])
            for item in result.observed
        ):
            raise ValueError(
                "numerical tests must record finite numerical comparisons and tolerances"
            )
    except Exception as exc:
        return QualificationCheck(
            f"test:{spec.suite_id}",
            "FAIL",
            f"{type(exc).__name__}: {exc}",
            spec.category.value,
            evidence,
        )
    return QualificationCheck(
        f"test:{spec.suite_id}",
        "PASS",
        "live evidence comparisons passed",
        spec.category.value,
        evidence,
    )


def qualify_plugin(
    plugin: Any,
    *,
    tests: tuple[Callable[[Any], Any], ...] = (),
    plan: PluginQualificationPlan | None = None,
) -> PluginQualification:
    """Check the current interface and optionally execute a pinned reference plan.

    Legacy ``tests=`` callables are smoke checks only, even if they return None,
    True, or metadata. Only a complete live plan can produce REFERENCE_QUALIFIED.
    """
    metadata = _metadata(plugin)
    if plan is not None and not isinstance(plan, PluginQualificationPlan):
        raise ValueError("reference qualification requires PluginQualificationPlan")
    checks = _interface_checks(plugin, metadata)
    interface_ok = all(c.status == "PASS" for c in checks)
    scope = None
    try:
        scope = qualification_scope(plugin)
    except (ValueError, OSError, TypeError, AttributeError, PackageNotFoundError) as exc:
        checks.append(QualificationCheck("source-scope", "NOT_EVALUATED", str(exc)))
    else:
        checks.append(
            QualificationCheck(
                "source-scope",
                "PASS",
                "current source and dependency scope observed",
                evidence=scope.to_dict(),
            )
        )

    for name, present in (
        ("scientific-evidence", bool(metadata.scientific_evidence)),
        ("license", bool(metadata.license.strip())),
        ("security-provenance", bool(metadata.security_provenance.strip())),
        ("operating-envelope", bool(metadata.operating_envelope)),
        ("provenance", metadata.provenance is not None),
    ):
        checks.append(
            QualificationCheck(
                name,
                "PASS" if present else "NOT_EVALUATED",
                "declaration present (not validation)"
                if present
                else "required declaration absent",
            )
        )

    plan_ok = plan is not None and scope is not None and plan.scope == scope and interface_ok
    checks.append(
        QualificationCheck(
            "plan-scope",
            "PASS" if plan_ok else "FAIL" if plan is not None else "NOT_EVALUATED",
            "plan matches current identity, API, metadata, source and dependencies"
            if plan_ok
            else "plan does not match current scope/interface"
            if plan is not None
            else "no reference plan requested",
        )
    )
    suite_results: dict[CheckCategory, list[QualificationCheck]] = {c: [] for c in CheckCategory}
    if plan_ok and plan is not None:
        for spec in plan.suites:
            result = _run_suite(plugin, spec)
            checks.append(result)
            suite_results[spec.category].append(result)
    for category in CheckCategory:
        results = suite_results[category]
        status = (
            "PASS"
            if results and all(r.status == "PASS" for r in results)
            else "FAIL"
            if results
            else "NOT_EVALUATED"
        )
        checks.append(
            QualificationCheck(
                f"suite:{category.value}",
                status,
                "live categorized suites passed"
                if status == "PASS"
                else "required live evidence absent or failed",
                category.value,
            )
        )

    for index, test in enumerate(tests):
        try:
            returned = test(plugin)
            if returned is False:
                raise AssertionError("smoke callable returned False")
        except Exception as exc:
            checks.append(
                QualificationCheck(
                    f"smoke:{index + 1}", "FAIL", f"{type(exc).__name__}: {exc}", "smoke"
                )
            )
        else:
            checks.append(
                QualificationCheck(
                    f"smoke:{index + 1}",
                    "NOT_EVALUATED",
                    "smoke callable completed; no qualification evidence",
                    "smoke",
                )
            )

    # Scope/schema changes made while any checks execute invalidate qualification.
    try:
        if (scope is not None and qualification_scope(plugin) != scope) or _interface_checks(
            plugin, _metadata(plugin)
        ) != checks[:4]:
            raise ValueError("plugin scope or interface changed during qualification")
    except Exception as exc:
        interface_ok = False
        checks.append(QualificationCheck("scope-stability", "FAIL", f"{type(exc).__name__}: {exc}"))
    else:
        checks.append(
            QualificationCheck(
                "scope-stability",
                "PASS" if scope is not None else "NOT_EVALUATED",
                "source, metadata, dependencies and schemas remained unchanged"
                if scope is not None
                else "interface unchanged; source scope unavailable",
            )
        )

    qualified = (
        plan_ok
        and all(c.status == "PASS" for c in checks if c.category != "smoke")
        and not any(c.status == "FAIL" for c in checks)
    )
    tier = (
        AssuranceTier.REFERENCE_QUALIFIED
        if qualified
        else AssuranceTier.INTERFACE_VERIFIED
        if interface_ok
        else AssuranceTier.INSTALLED_UNQUALIFIED
    )
    evidence_status = {
        AssuranceTier.INSTALLED_UNQUALIFIED: "INSTALLED_ONLY",
        AssuranceTier.INTERFACE_VERIFIED: "SOFTWARE_INTERFACE_VERIFIED",
        AssuranceTier.REFERENCE_QUALIFIED: "SOFTWARE_REFERENCE_VERIFIED",
    }[tier]
    return PluginQualification(
        metadata.plugin_id, metadata, tuple(checks), tier.value, evidence_status, scope
    )


class BuiltInVerticalPlugin:
    """Explicit synthetic vertical adapter; it ships no claimed reference certification."""

    metadata = PluginMetadata(
        "physioworld.verticals",
        PluginKind.BENCHMARK,
        "1.0.0",
        "physioworld>=0.1,<0.2",
        (),
        ("docs/scientific-scope.md: synthetic/model-derived execution only",),
        "NOT_EVALUATED",
        "MIT",
        "built-in source; no security scan implied",
        {"vertical": ("cardiovascular", "neurophysiology", "icu"), "seed": (0.0, 2147483647.0)},
    )
    schemas = {
        "execute": {
            "input": {
                "$id": "physioworld:verticals:execute:input:1.0",
                "type": "object",
                "properties": {"name": {"type": "string"}, "seed": {"type": "integer"}},
            },
            "output": {"$id": "physioworld:verticals:execute:output:1.0", "type": "object"},
        }
    }

    def execute(self, name: str = "cardiovascular", *, seed: int = 0) -> Mapping[str, Any]:
        if name not in self.metadata.operating_envelope["vertical"]:
            raise ValueError("vertical is outside the declared plugin envelope")
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            raise ValueError("seed is outside the declared plugin envelope")
        from .verticals import run_vertical

        return run_vertical(name, seed=seed).to_dict()

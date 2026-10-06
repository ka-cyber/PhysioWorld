# Plugin contracts and software qualification

PhysioWorld separates an available plugin, an inspected interface, and a
reference-tested implementation. Importability, registration, metadata, a
self-reported numerical status, and a successful smoke callback do not establish
reference qualification or scientific validation.

## Explicit loading

```python
from physioworld.plugins import PluginRegistry, load_plugin

plugin = load_plugin("builtin:verticals")
registry = PluginRegistry()
registry.register(plugin)
report = registry.qualify(plugin.metadata.plugin_id)
print(report.assurance_tier)  # INTERFACE_VERIFIED
print(report.passed)  # False: no reference plan was executed
print(report.interface_verified)  # True
```

`load_plugin("package.module:attribute")` imports only the explicitly requested
module. Classes are instantiated without arguments; other objects are returned
as supplied. Loading executes Python in the caller's process. Registration and
qualification never search directories, discover entry points, or import
declared dependency modules. Dependency checks inspect installed distribution
metadata. Qualification invokes only explicitly supplied suites or smoke checks.

## Assurance tiers

| Tier | Meaning |
| --- | --- |
| `INSTALLED_UNQUALIFIED` | An object is available, but its current kind-specific interface, API compatibility, dependency bounds, or schema declarations failed. Registration itself confers no tier. |
| `INTERFACE_VERIFIED` | Current static interface checks passed. Behavioral reference qualification is absent, incomplete, stale, or failed. |
| `REFERENCE_QUALIFIED` | All required live, evidence-recording test categories and metadata/provenance gates passed for the exact reported scope. |

`passed` is true **only** for `REFERENCE_QUALIFIED`. `interface_verified` reports
the narrower interface result. The corresponding `evidence_status` values are
`INSTALLED_ONLY`, `SOFTWARE_INTERFACE_VERIFIED`, and
`SOFTWARE_REFERENCE_VERIFIED`. These are software evidence classifications,
not hardware measurements, clinical validation, causal identification, proof of
a theorem, regulatory approval, or security certification.

Every category—API, schema, unit, reference, numerical, reproducibility, and
security—needs at least one passing live suite. Missing categories are
`NOT_EVALUATED`; failures remain explicit. Additional suites within a category
must all pass. An all-PASS subset cannot qualify a plugin.

## Plugin API 1.0

Each object exposes `metadata: PluginMetadata`, `schemas`, and the entry points
for its declared `PluginKind`. Entry points are synchronous and support the
listed argument names as keywords. Required extra arguments, positional-only
contract arguments, missing methods, and catch-all `*args`/`**kwargs` wrappers
fail the contract. Existing native APIs can be wrapped in an explicit adapter.

| Kind | Required entry points |
| --- | --- |
| `physiology_model` | `initialize(parameters, seed)`, `advance(state, actions, dt_s)`, `observe(state)` |
| `sensor` | `observe(state, time_s)` |
| `device` | `configure(configuration)`, `start()`, `read(max_samples)`, `stop()` |
| `communication_channel` | `transmit(packets, seed)` |
| `resource` | `transition(state, action, dt_s)` |
| `representation` | `transform(history)` |
| `task` | `evaluate(prediction, target)` |
| `estimator` | `fit(data)`, `predict(data)` |
| `policy` | `act(observation, resource_state, time_s)` |
| `causal_estimator` | `identify(problem)`, `estimate(data, identification)` |
| `dataset_adapter` | `manifest()`, `load(configuration)` |
| `benchmark` | `execute(name, seed)` |
| `visualization` | `render(data, configuration)` |
| `report_generator` | `generate(result)` |

`PLUGIN_CONTRACTS` is the immutable, inspectable contract table. Time arguments
with `_s` use seconds. Domain units, clock semantics, estimands, valid inputs,
outputs, and failure behavior belong in the schemas and reviewed reference
fixtures; method names alone do not establish scientific correctness.

### Schema declarations

`schemas` maps **exactly** the required method names to `{"input": ..., "output":
...}` JSON Schema objects. Each root declares a nonempty `$id` and a single JSON
`type`. Input roots are objects with `properties` for every contract argument.
The static declaration checker supports single types, nested `properties`,
`required`, `items`, and boolean or typed-schema `additionalProperties`.
Declarations must serialize as finite JSON data.

This syntax check does not evaluate arbitrary JSON Schema keywords or validate
actual method outputs. The required live schema suite must verify the chosen
schema dialect, valid and invalid inputs, actual outputs, and domain-specific
units/time semantics. A full JSON Schema implementation can be used by that
explicitly supplied suite.

### Metadata and version constraints

Metadata schema `1.0` preserves the original positional fields and adds
`api_version` (plugin API `1.0`), typed `provenance`, and `schema_version`.
`to_dict()` / `from_dict()` support finite JSON round-trips; missing or unknown
metadata schema versions fail. Operating-envelope bounds are copied and
immutable, finite ordered numerical intervals or nonempty categorical lists.

* Plugin releases use stable `major.minor.patch` strings.
* `api_compatibility` constrains the running PhysioWorld distribution, for example
  `physioworld>=0.1,<0.2`; it is distinct from `api_version`.
* Dependencies are distribution requirements, for example `example>=1.2,<2`.
  The stdlib-only parser accepts numeric stable releases and comma-separated
  `==`, `!=`, `>=`, `<=`, `>`, `<` comparisons. It fails closed on unbounded
  requirements, prereleases, wildcard versions, extras, URLs, environment markers,
  and other specifier syntax. Such dependencies need an explicit supported
  distribution constraint or an adapter; their semantics are never guessed.
* Reference qualification also requires scientific evidence references, license
  metadata, security provenance, a declared operating envelope, and
  `PluginProvenance(source_uri, revision, publisher, artifacts)`.

`EvidenceArtifact` pins an explicitly supplied local path, SHA-256 digest, and
description. Provenance artifacts must include the plugin defining file and the
files defining its entry points. Add transitive implementation files, models,
configuration, and reference fixtures relevant to the declared envelope to the
manifest. Checksums verify the reported local artifacts; they do not authenticate
a publisher or establish the truth of a citation or legal license approval.

## Executing a reference plan

Reviewers supply `QualificationSuite` objects rather than plugin-authored PASS
strings or historical result records. Every suite has an ID, category, a factory
receiving the current plugin, a pinned test-source artifact, and a reference/oracle
description. The factory returns a nonempty `unittest.TestSuite` containing
`EvidenceTestCase` instances defined in that artifact. The defining source of
both the factory and test methods must be covered by the pinned test artifact.

The following example targets a separately loaded arithmetic-mean representation
adapter. `plugin` denotes that adapter, with its own metadata and schemas.

```python
import unittest
from physioworld.plugins import (
    CheckCategory,
    EvidenceArtifact,
    EvidenceTestCase,
    PluginQualificationPlan,
    QualificationSuite,
    qualification_scope,
    qualify_plugin,
)


class NumericalChecks(EvidenceTestCase):
    def verify_known_mean(self):
        self.assertEvidenceClose(
            self.plugin.transform(history=[1.0, 2.0, 3.0]),
            2.0,
            abs_tol=1e-12,
            label="analytic arithmetic-mean fixture",
        )


def numerical_tests(plugin):
    case = NumericalChecks("verify_known_mean")
    case.plugin = plugin
    return unittest.TestSuite([case])


# This code lives in a reviewed Python test file; __file__ pins that file.
numerical = QualificationSuite(
    "analytic-mean",
    CheckCategory.NUMERICAL,
    numerical_tests,
    EvidenceArtifact.from_file(__file__, "reviewed numerical tests"),
    "Arithmetic mean of 1, 2, 3 equals 2; synthetic analytic oracle",
)
# Supply reviewed suites for every other category as well.
plan = PluginQualificationPlan(qualification_scope(plugin), (numerical,))
report = qualify_plugin(plugin, plan=plan)
assert not report.passed  # A numerical suite alone is incomplete.
```

Use `assertEvidenceEqual(actual, expected, label=...)` for JSON-compatible
comparisons and `assertEvidenceClose(actual, expected, abs_tol=..., rel_tol=...,
label=...)` for finite numerical comparisons. Every test must execute an evidence
comparison. Every numerical test must execute a finite numerical comparison with
declared tolerances. Skips, expected failures, unexpected successes, failed
comparisons (even if an assertion exception is caught), empty suites, no-op tests,
and plain unrecorded unittest tests do not qualify.

The security suite must perform a real, scoped current check with observable
results—such as reviewed malformed-input tests and scan findings compared with
the declared policy. A security-provenance string is not a scan. Reviewers must
judge whether each oracle, numerical tolerance, scan scope, reproducibility
protocol, and tested envelope is adequate. The runner records execution and
comparisons; it cannot establish independence or scientific validity of a
reviewer's tests, and does not isolate malicious Python code.

### Freshness and serialization

`qualification_scope(plugin)` observes identity, release, plugin API, metadata
and schema fingerprints, pinned implementation artifacts, entry-point bytecode
and defaults, installed dependency versions, and the Python runtime. The plan
must exactly match this scope before its suites run. Source and test artifacts
are verified against their pinned digests. Scope and interface are checked again
after suites and legacy smoke callbacks; changes invalidate qualification.
Opaque/native entry points need an inspectable Python adapter and explicit
artifact/runtime evidence for reference qualification.

Qualification schema `1.0` serializes with `to_dict()` and includes the scope,
run ID, UTC evaluation time, individual statuses, test IDs/counts, observed
comparisons, tolerances, artifact checksums, and failure summaries. Serialized
reports are historical snapshots, not authenticated certificates or reusable
live evidence. Re-run a plan against the current installation. Undeclared
transitive files, instance state, external systems, and untested parameter
regimes are outside the reported scope.

## Built-in adapter and API migration

`BuiltInVerticalPlugin`, `load_plugin`, `PluginRegistry`, and `qualify_plugin`
remain available. `builtin:verticals` works through `load_plugin` as well as the
CLI's existing explicit built-in branch. `execute(name="cardiovascular", seed=0)`
forwards the seed; the declared adapter envelope is the three supported vertical
names and integer seeds `0..2147483647`. Results remain synthetic/model-derived.
The adapter supplies an interface, not a built-in reference/security certificate.

`registry.qualify(plugin_id)` inspects the interface; `registry.qualify(plugin_id,
plan=reviewed_plan)` requests the corresponding live reference checks. Legacy
`tests=` remains compatible but reports successful smoke execution as
`NOT_EVALUATED` evidence, including metadata-returning callbacks. Consumers use
`interface_verified` for static interface success and `passed` for complete
reference qualification. The former `QUALIFIED_SOFTWARE_PLUGIN` result is
replaced by the specific tier actually established. New types are importable
from `physioworld.plugins`.

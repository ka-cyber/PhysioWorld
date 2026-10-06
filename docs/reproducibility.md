# Reproducing an experiment

Every run needs a declarative configuration, seed, software/runtime identity,
dataset/version, input and output hashes, explicit preprocessing/split, task,
estimand, model, policy, resource/channel regime and uncertainty treatment.

```bash
physioworld run examples/minimal_world.json --output .physioworld/original
physioworld provenance .physioworld/original
physioworld reproduce .physioworld/original/result_manifest.json --output .physioworld/reproduced
physioworld report .physioworld/reproduced
```

Use a fresh output directory for each execution. `provenance` inspects the
manifest; `report` displays the run report. A verified reproduction first checks
the original manifest, executes its configuration, verifies the new artifacts,
and compares the scientific output hashes exactly. Generation timestamps,
run names, reports, and compiled-plan metadata are not scientific comparisons.
Missing data must produce an explicit readiness result or failure rather than a
replacement synthetic experiment.

## Immutable, versioned snapshots

New manifests use `manifest_version: 1`. They record the canonical configuration
SHA-256 hash, seed, software/runtime identity, installed Python-source digest,
bundled-schema digest, declared output names, and each output's SHA-256 hash.
The source and schema digests use sorted relative file names and file contents;
they are independent of the checkout's absolute location. Python bytecode and
run-directory artifacts are not part of source identity. An unchanged version
string does not permit replay with changed code or schemas.

Configurations must contain finite JSON values with string object keys. Hashing
uses sorted keys, compact separators, UTF-8, and `allow_nan=False`, with no
fallback stringification. A caller-supplied `input_hashes["configuration"]` must
equal the computed hash. A configuration's declared seed must agree with the
manifest seed; `create_manifest` infers the seed when omitted.

`ResultManifest` is a frozen dataclass. Its `.configuration` is a recursively
read-only `MappingProxyType`: nested objects are also read-only mappings and
JSON arrays are tuples. Datasets and hash/output mappings are copied and frozen
as well. Changing the caller's original nested objects cannot change a manifest.

Use `manifest.to_dict()` for a detached JSON-serializable manifest and
`configuration_snapshot(manifest)` or
`configuration_snapshot(manifest.configuration)` for a detached **dict/list**
configuration. `dict(manifest.configuration)` only copies the outer mapping;
`dataclasses.asdict(manifest)` cannot copy `MappingProxyType`.

## Integrity verification and receipts

```python
from physioworld.provenance import (
    compare_scientific_outputs,
    configuration_snapshot,
    verify_manifest,
)

original = verify_manifest(".physioworld/original/result_manifest.json")
configuration = configuration_snapshot(original.manifest)
# Execute configuration into a fresh directory using the relevant runner.
reproduced = verify_manifest(".physioworld/reproduced")
comparison = compare_scientific_outputs(original, reproduced)
receipt = {
    "original": original.to_dict(),
    "reproduced": reproduced.to_dict(),
    "comparison": comparison.to_dict(),
}
```

`verify_manifest(path)` accepts a directory or manifest file and returns a
`Verification` receipt. `.manifest` is the verified immutable snapshot;
`.to_dict()` is a JSON-ready integrity receipt, and
`.to_dict(include_manifest=True)` also includes the configuration snapshot.
Failures raise `ManifestVerificationError`, a `ValueError` subclass.

Verification requires:

- The current manifest format and a `COMPLETE` status.
- Exactly the current software version, source digest, and schema digest.
- The correct canonical configuration hash and equality of `configuration.json`
  with the manifest configuration, including JSON numeric representations.
- Exactly one output hash per declared output key, every declared file present
  and regular, and exact byte hashes for **all** outputs, including reports and
  compiled plans.
- Safe relative POSIX artifact names: no absolute paths, Windows drives,
  parent traversal, backslashes, ambiguous empty/dot components, or symlinks.
  Symlinks in the run path's ancestors are also rejected.
- Strict JSON parsing: duplicate keys, nonfinite constants, and numeric
  overflow to infinity are rejected.

Readiness-only runs should be created with `status="PARTIAL"`. Their integrity
can be inspected with `verify_manifest(path, allow_partial=True)`; the receipt
preserves `PARTIAL` and records the policy override. This does not establish that
the experiment ran or that clinical performance was measured. `FAILED` and
`UNKNOWN` never qualify for successful verification.

Compatibility exceptions are explicit and independent:
`allow_version_mismatch=True`, `allow_source_mismatch=True`, and
`allow_schema_mismatch=True`. Each actual exception appears in
`compatibility_overrides` in the receipt. These options do not permit missing
bindings or unsupported manifest formats. Exact compatibility is the default.

`load_manifest(path)` remains available for inspection, including older
unversioned manifests, but does not verify files or runtime compatibility.
Unversioned manifests cannot produce a current integrity receipt.

`manifest.scientific_outputs` is a tuple of output **keys**, defaulting to every
declared key except `report` and `compiled_plan`. An explicit tuple can select a
scientific subset. `compare_scientific_outputs` requires matching selections and
exact selected hashes and returns a `ScientificOutputComparison` with
`.to_dict()`. Passing paths verifies their files first; passing manifests or
verification receipts compares their already-recorded hashes. A runner should
pass fresh verification receipts after execution. Manifest and verification
objects also expose `.compare_scientific_outputs(other)`.

## Output ownership

`prepare_run_directory(path)` creates a new path or accepts an empty directory.
It refuses nonempty directories, existing run outputs, and symlink ancestors.
Call it before the runner writes any outputs. `write_output(directory,
relative_path, text_or_bytes)` and `write_json_output(directory, relative_path,
value)` exclusively create safe artifacts, including nested relative paths;
they never overwrite existing files. The JSON writer uses strict JSON.

The legacy `create_manifest(...)` and `ResultManifest.write(directory)` APIs are
preserved. `.write` accepts the directory a runner has populated, requires every
declared output to exist, computes accurate hashes, and exclusively creates
`configuration.json` and `result_manifest.json`. It refuses an existing manifest
or configuration snapshot, including matching ones.

`write_report(...)` verifies the run first, writes strict JSON result/uncertainty
sections, and updates its owned manifest with the completed report hash. It may
replace a previously declared, verified `report.md`, but refuses unowned reports
and corrupted outputs. Report/manifest replacement is atomic per file; a failed
or interrupted update must be verified again before reuse. Updating a report
preserves the run status and scientific output selection.

## External data, credentials, and evidence scope

Credentialed dataset access remains user-side. Prefer dataset/version identifiers
and explicit `data_reference` values, with references such as `credential_env`
or `api_key_env` naming credentials supplied outside the snapshot. Do not embed
restricted raw records in configuration, reports, or output artifacts. Public
synthetic sample declarations can be recorded directly.

Nonempty literal sensitive configuration keys such as `password`, `token`,
`api_key`, `client_secret`, and `private_key` are rejected recursively and
case-insensitively. The same key check applies to report result/uncertainty
objects and the JSON output writer; exception messages omit credential values.
This key-based check cannot identify private data or secrets hidden in arbitrary
free-text fields or opaque file contents. A manifest does not assert that such
content has been de-identified.

Integrity verification establishes consistency of the recorded configuration,
runtime bindings, and declared output bytes. It does not authenticate a signed
publisher, inspect an external dataset reference, or prove scientific/clinical
validity. External input digests are recorded metadata; dataset adapters must
resolve those references and check the actual inputs before execution.

Use the frozen dependency environment from [installation](installation.md);
source identity does not capture installed dependency binaries. Exact scientific
byte equality is this replay API's criterion. A separate scientific evaluation
using floating-point tolerances must declare those tolerances explicitly.
Historical figures in [scientific context](scientific-context.md) require their
actual source data and estimator artifacts before they can be reproduced.

See [qualification levels](quality/reproducibility.md) and the
[release evidence format](../release/README.md).

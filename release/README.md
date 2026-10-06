# Release engineering

Version 0.1.0 remains the package version. Evidence levels are independent of
semantic versioning: `DEVELOPMENT`, `RESEARCH`, `REFERENCE`, `VALIDATED_RESEARCH`,
`CLINICAL_VALIDATION_PACKAGE`, `PRODUCTION_READY_RESEARCH_PLATFORM`.
See [release qualification](../docs/quality/release-qualification.md).

## Build and qualify

From the reviewed source tree, with uv 0.12.23 and the frozen lock:

```bash
uv sync --frozen --all-extras
uv run --no-sync python -m build --outdir /path/to/distributions
uv run --no-sync twine check /path/to/distributions/*
uv run --no-sync python tools/smoke_install.py /path/to/distributions/* --parent /path/to/external-workspace --examples examples --report /path/to/install-report.json
```

Build isolation uses exact backend pins from `pyproject.toml`. The standard
build frontend builds the wheel from the sdist, checking that packaging inputs
survive the source archive. Runtime schemas are package data. Examples, docs,
source schemas, tests and tools are sdist inputs; mutable outputs are excluded.

## Generate observed artifacts

Run the generator with the Python interpreter for the environment being
inventoried. The output must be a **new directory with an existing parent**;
the generator refuses to overwrite a prior bundle.

```bash
python tools/release_manifest.py \
  --output /path/to/new-release-bundle \
  --artifact /path/to/distributions/physioworld-0.1.0-py3-none-any.whl \
  --artifact /path/to/distributions/physioworld-0.1.0.tar.gz
```

Outputs:

- `manifest.json`: observed version, real commit or null, dirty state or unknown,
  tags actually pointing at the commit, runtime, installed dependency versions,
  lock hash, source/schema/benchmark/dataset inventories, optional observed
  container digest, artifact hashes, SBOM ID and evidence status.
- `sbom.json`: **CycloneDX 1.5 JSON** inventory of installed distributions,
  package URLs, versions, observed license metadata, declared requirements and
  installed RECORD hashes. It is an environment inventory, not an invented
  complete dependency graph. Run `tools/check_sbom.py` in the dev environment
  for schema-aware validation.
- `checksums.sha256`: SHA-256 of the two generated documents. Inventory aggregate
  hashes use sorted canonical JSON; each file also has its own SHA-256 and size.

No container digest, commit, release tag, dataset acquisition, scientific claim
or passing gate is inferred from a package version. Null/`NOT_EVALUATED` fields
are intentional when the evidence is unavailable. Keep execution bundles outside
the checkout or under ignored `release/artifacts/`; only reviewed frozen bundles
belong in a public release.

## Acceptance evidence input

`--evidence /path/to/evidence.json` accepts an object with `checks` and optional
`scientific_claim_status`. Each check has a unique `id`, explicit `status`
(`PASS`, `FAIL`, `NOT_EVALUATED`) and, for evaluated checks, an existing `artifact`
path. The generator hashes those artifacts. A minimal evidence object is:

```json
{
  "checks": [],
  "scientific_claim_status": {"status": "NOT_EVALUATED", "claims": []}
}
```

Set `--reproducibility-level 0..5` only for the evidenced level; levels above
zero require a PASS `reproducibility` artifact. Requesting production research
level requires every named gate in the generator plus a real clean committed
tree. The tool binds supplied evidence; it does not independently certify it.

An SBOM includes the actual generation environment, so generate a runtime SBOM
from a runtime environment, and a separate dev/build SBOM if needed. Review
dependency licenses and notices before redistribution. Regenerate scientific
reference evidence when a kernel, estimator, task or schema changes.

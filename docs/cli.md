# Command-line reference

Use `physioworld --help` and `physioworld COMMAND --help` for the installed
version's authoritative argument syntax. Execution commands validate declarations
before creating output artifacts. Invalid declarations and integrity errors
return nonzero; inspect each result's status as well, because readiness-only
outputs and some benchmark failure reports can return a successful exit code.

| Command | Purpose |
|---|---|
| `doctor` | Installed-package runtime, import, schema and optional-capability diagnostics |
| `qualify-install` | Installed-package, numerical smoke and replay qualification |
| `version` | Software version |
| `schema` | Discover/inspect bundled schema resources from the installed package |
| `inspect` | Inspect a scientific configuration or artifact |
| `datasets` | List registry metadata and validate user-side dataset access |
| `validate` | Validate a declarative scientific configuration |
| `simulate` | Execute explicitly synthetic samples/world inputs |
| `replay` | Replay declared observations with preserved time/provenance |
| `run` | Execute a compiled scientific workflow |
| `evaluate` | Evaluate a declared comparison/estimand |
| `benchmark` | Run a supported benchmark family or analytical smoke |
| `scientific-suite` | Run the registry-driven Level 3 scientific regression/evidence suite |
| `reproduce` | Verify a run manifest, re-execute it, and compare scientific output hashes |
| `provenance` | Inspect a run's manifest |
| `report` | Display the human-readable run report |
| `qualify-plugin` | Check plugin metadata and applicable contract requirements |
| `clinical-validate` | Freeze an explicit clinical protocol and report readiness without endpoint evaluation |

The clinical protocol command, `clinical-validate`, freezes a declared protocol
and reports readiness. It does not manufacture a clinical endpoint result.
Use the complete [clinical readiness configuration](clinical_validation_guide.md).

For the native Level 3 corpus, run:

```bash
physioworld validate configs/corpus/benchmark_corpus_level3.json --family native_level3_synthetic
physioworld benchmark --family native_level3_synthetic --config configs/corpus/benchmark_corpus_level3.json --output .physioworld/corpus-level3
```

The [corpus guide](corpus_guide.md) describes package-bound fixtures, the separate
corpus manifest/result format, and synthetic/numerical evidence boundaries.
`benchmark --config` and `--output` are supported for this family; legacy
benchmark families reject those options rather than consuming a configuration.

## Master scientific suite

Run the local scientific regression and evidence bundle with one command:

```bash
physioworld scientific-suite --profile fast --output .physioworld/scientific-suite
```

Profiles are `fast`, `full`, `independent-reference`, and `release`. The output
contains the source and experiment registries, native corpus results, kernel VVUQ
results, experiment receipts, claim mapping, uncertainty/operating-envelope
declarations, and SHA-256 checksums. External cohorts and historical controller
artifacts are recorded as `BLOCKED_EXTERNAL_INPUT` until their authorized source
artifacts are supplied. The `release` profile therefore exits nonzero while
those inputs remain unavailable; this is an explicit qualification boundary,
not a synthetic substitute for external evidence.

## Output and data semantics

Outputs default to `.physioworld/` and each run requires a new or empty output
directory. `simulate`, `replay`, and `run` compile and validate the complete
configuration before creating that directory. Dataset access and synthetic
execution are separate modes; configuration-only readiness writes a `PARTIAL`
manifest and is not successful dataset evaluation. Missing credentials, absent
data, malformed declarations, and unsupported methods are reported precisely.

`reproduce MANIFEST --output DIRECTORY` first verifies the original manifest and
all of its declared artifacts. It then executes the canonical configuration in
a fresh directory, verifies the reproduced manifest, compares the selected
scientific output SHA-256 hashes exactly, and writes a reproduction report.
Tampered manifests or output artifacts are rejected before replay.

Installation diagnostics and qualification use `physioworld.qualification` and
are independent of a source checkout. Add `--checkout-tests` to `doctor` or
`qualify-install` only when an explicit checkout test run is desired.

The `evaluate` command requires repeated explicit `--assumption` declarations;
it never invents binary-outcome, target, population, or measurability
assumptions.

Optional dependencies are capability-specific. Installation or plugin discovery
does not establish scientific qualification. See [installation](installation.md),
[scientific contracts](contracts.md) and [security](../SECURITY.md).

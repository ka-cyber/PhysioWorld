# Native Level 3 corpus guide

The native campaign exercises deterministic synthetic and mathematical fixtures
across representation, task-law, communication, resource, policy, causal, and
robustness boundaries, plus cardiovascular, neurophysiology, and ICU workflows.
Its evidence class is `SIMULATION_AND_NUMERICAL_VERIFICATION`. The Level 3
execution profile names a research scope, not clinical or production certification.

## Run from a checkout

The configuration is nested under `configs/corpus/`:

```bash
physioworld validate configs/corpus/benchmark_corpus_level3.json --family native_level3_synthetic
physioworld benchmark --family native_level3_synthetic --config configs/corpus/benchmark_corpus_level3.json --output .physioworld/corpus-level3
```

The [corpus configuration](../configs/corpus/benchmark_corpus_level3.json) is
validated against the [corpus schema](../schemas/corpus.schema.json), rather than
the ordinary experiment compiler. The family-specific `validate` command checks
that contract without executing it; use the benchmark command for execution.
The output directory must be new or empty.

Execution loads the installed fixtures from
`package:physioworld/corpus/golden`, independently of the current directory.
The checkout's `datasets/golden` directory is the source-side fixture collection;
it is not an external dataset. Omitting `--config` uses the same campaign defaults
from the installed package, so a wheel installation can run without a checkout.

## Python API

```python
from physioworld.benchmarks import run_corpus_campaign

# Installed-package defaults and package-bound fixtures; use a fresh output path.
summary = run_corpus_campaign(".physioworld/corpus-level3-api")
print(summary["status"], summary["case_count"])
```

To select the source-distribution configuration explicitly, pass
`configuration="configs/corpus/benchmark_corpus_level3.json"`. The
`run_benchmark_family("native_level3_synthetic", configuration=..., output=...)`
entry point dispatches to the same runner. Configuration and output contracts
are validated before artifacts are written.

## Cases and scope

The default configuration declares eight groups and produces ten case results:

| Group | Calculation / boundary |
|---|---|
| `representation_temporal` | Synthetic history-length RMSE sweep and multiscale features |
| `tasklaw_transition` | Four-row conditional task-law discrepancy with direct group-enumeration reference |
| `communication_semantic` | Seeded RLNC recovery under declared loss and a semantic-permutation control |
| `resource_p1` | Constant-current, zero-leakage capacitor/ESR transition and independent analytic current/voltage references |
| `policy_capability` | Two-step finite-MDP oracle versus exhaustive action-sequence enumeration |
| `causal_synthetic` | Known structural-model treatment effect; no human causal identification |
| `robustness_common_policy` | Finite policy/kernel table distinguishing common-policy robust and clairvoyant optimization orders |
| `verticals` | Separate cardiovascular, neurophysiology, and ICU synthetic results with repeated-run output-hash comparisons |

Some controls are recorded as registered controls rather than executed tests.
Read each case's `controls`, `comparisons`, and `limitations` fields before
using its result. Historical reference comparisons are explicitly inactive;
an analogue's success does not reproduce an external empirical headline value.

## Output and repeatability

The campaign writes:

- `summary.json`: campaign status, actual fidelity, case count, failed cases,
  configuration hash, manifest hash, and the case results.
- `corpus_manifest.json`: canonical configuration, seeds, fixture hashes,
  case-artifact hashes, reference metadata, and external-input boundaries.
- `cases/<case_id>/result.json`: observed calculations, estimands, independent
  references or controls, comparisons, and limitations.
- `unavailable_inputs.json`: external cohorts/source tables requiring acquisition
  and hardware measurements that are not supplied.
- `corpus_result.schema.json`: the versioned output contract, also available as
  the [source schema](../schemas/corpus_result.schema.json).

Successful default execution reports `COMPLETE_SYNTHETIC_ONLY` with
`actual_fidelity: SYNTHETIC_AND_MATHEMATICAL`. Individual numerical cases may
report `COMPLETE_NUMERICAL_REPRODUCTION`. A failed independent comparison or
vertical repeatability check produces `FAILED_REFERENCE`; inspect the returned
status and `failed_cases` even when the command exits successfully.

Repeat the benchmark with the same configuration into a second fresh directory
and compare its configuration, fixture, and case hashes. Corpus artifacts use
their own schema and `corpus_manifest.json`; the ordinary `provenance`, `report`,
and `reproduce` run-manifest commands apply to `result_manifest.json` workflows.

## Sources and interpretation

The [source ledger](SCIENTIFIC_SOURCE_LEDGER.yaml) maps stable source IDs to
external identifiers and distributed documentation extracts. IDs such as
`LEVEL3_PROMPT` remain compatibility labels in manifests; the original external
specification is not distributed. Legacy `MANUALLY_INGESTED` source statuses in
campaign output mean extracted specification/reference content, not loaded raw
data. All historical `reference_artifacts` have `active_evidence: false`.

Consult the [reference values](REFERENCE_ARTIFACTS.yaml),
[claim scopes and test references](CLAIM_LEDGER.yaml),
[assumptions](ASSUMPTION_LEDGER.yaml), and
[numerical-reference statements](THEOREM_LEDGER.yaml). `VERIFIED-NUMERICAL`
describes test-backed fixture scope, not a frozen production certification.
External raw-data reproduction requires authorized acquisition of the original
records, estimator/policy artifacts, and matching protocol. See
[known limitations](KNOWN_LIMITATIONS.md) and the
[Level 3 API map](api_reference.md#level-3-scientific-interfaces).

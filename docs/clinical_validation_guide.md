# Clinical validation guide

`physioworld clinical-validate` compiles an experiment and freezes an explicit
`ClinicalValidationProtocol`. It records intended use, population, data
provenance, inclusion/exclusion, reference standard, endpoint, analysis
population, subject/temporal split, missing-data policy, model, threshold,
resource, channel, and dataset version. The command reports protocol readiness;
it does not run predictive evaluation or calculate clinical performance.

## Complete synthetic readiness protocol

Use the JSON [simulation/readiness configuration](../configs/clinical/simulation_readiness.json)
from a source checkout or source distribution:

```bash
physioworld clinical-validate configs/clinical/simulation_readiness.json --output .physioworld/clinical-readiness
physioworld provenance .physioworld/clinical-readiness
physioworld report .physioworld/clinical-readiness
```

This configuration declares four analytic synthetic sequences, their units and
sampling clock, a one-second synthetic endpoint, an independent analytic label
rule, a fixed persistence model and threshold, a held-out temporal test split,
missingness/abstention rules, and explicit resource/channel/compute regimes.
It contains a generation recipe, not patient records, external reference data,
or generated samples. JSON requires no YAML dependency.

The current compiler routes declarations with these clinical fields through
the dataset-readiness branch. Declaring `origin: SYNTHETIC` inside `dataset`
does not execute that recipe or change the branch's `data_mode: dataset`.
The expected result is:

| Field | Value / interpretation |
|---|---|
| `status` | `CONFIGURATION_ONLY` |
| `data_status` | `REQUIRES_USER_DATASET_ACCESS`; the declared samples are not available to this command |
| `evaluation_scope` | `EXTERNAL_PREDICTIVE_EVALUATION`; the branch's readiness category, not evidence of external data or evaluation |
| `external_predictive_evaluation_status` | `NOT_EVALUATED` |
| `clinical_validation_status` | `NOT_ESTABLISHED` |
| `performance` | `null` |
| Run manifest status | `PARTIAL` |

A successful command freezes the configuration and protocol hashes and writes
`clinical_protocol.json`, `validation_result.json`, `result_manifest.json`, and
`report.md` along with the normal provenance artifacts. Use a new or empty output
directory. Structural compilation and non-placeholder checks do not independently
verify the scientific contents of the declarations.

## Evaluation and evidence boundaries

`protocol_from_config()` requires all fourteen clinical declarations explicitly;
experiment defaults do not supply them. Empty regimes, placeholder text, or a
split without a held-out test or temporal evaluation are rejected. The
[API reference](api_reference.md#level-3-scientific-interfaces) distinguishes this
readiness protocol from the locked cohort-evaluation API.

Independently acquired records and a valid reference standard are
required before evaluating clinical endpoints. Prespecified subject/temporal
splits, calibration, subgroup definitions, false-alarm burden and uncertainty
are reviewed against the intended context of use. See
[model credibility](quality/model-credibility.md).
Synthetic endpoint labels remain simulation evidence. The
[known limitations](KNOWN_LIMITATIONS.md) describe the unavailable-input boundary.

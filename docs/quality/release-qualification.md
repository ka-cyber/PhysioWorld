# Release qualification

Qualification is attached to an exact version, commit, source inventory and
environment. This document defines acceptance; current execution results live
in CI artifacts or an explicitly generated release bundle. It does not assert
that every acceptance gate has passed.

## Acceptance evidence

| Area | Required evidence |
|---|---|
| Installation | Wheel and sdist build, metadata validation, clean external installations, imports and bundled schemas |
| Interface | All required command help, meaningful failures, doctor, qualify-install and supported public API examples |
| Code quality | Full declared lint/format/type scope; no blanket failure suppression |
| Runtime support | Python 3.10–3.14 × Linux/macOS/Windows × core/scientific/dev CI outcomes |
| Scientific kernels | Estimand tests, negative controls, independently calculated numerical references and operating envelopes |
| Data | Schema/example validation, versioned contracts, units, acquisition/access metadata and leakage-safe split evidence |
| Reproduction | Hashed inputs/outputs, frozen seed/configuration, deterministic replay and independent reference comparisons |
| Supply chain | Frozen lock, advisory scan, secret scan, source security scan, license/notice inventory and CycloneDX SBOM |
| Public product | Built docs, checked links, accurate citation, scientific scope and curated reference artifacts |
| Integration | Cardiovascular, neurophysiology and ICU scenarios execute and reproduce with evidence classes |

The [CI workflow](../../.github/workflows/ci.yml) executes these checks where
automatable. License compatibility and context-of-use validation require
reviewed evidence; metadata presence is only one input to that review.

## Release semantics

Release levels are `DEVELOPMENT`, `RESEARCH`, `REFERENCE`, `VALIDATED_RESEARCH`,
`CLINICAL_VALIDATION_PACKAGE`, and `PRODUCTION_READY_RESEARCH_PLATFORM`.
They are evidence classifications, not synonyms for major version numbers.

The manifest generator defaults to `DEVELOPMENT` and records unassessed gates.
Production-level generation requires a real commit, a clean tree and supplied
PASS artifacts for every declared gate. It records evidence, not independent
authentication of the evidence author's assertions. A higher reproducibility
level requires a replay evidence artifact. Publication approval remains a
release-specific review.

## Scientific status

Software acceptance does not establish clinical benefit, regulatory approval,
causal identification or measured hardware performance. See
[model credibility](model-credibility.md), [scientific scope](../scientific-scope.md)
and [release commands](../../release/README.md).

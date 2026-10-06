# Model credibility and context of use

Credibility is specific to a scientific decision. The same model can be
supported for an analytical exercise and unsupported for patient treatment.
Use risk-informed V&V structure as engineering guidance; no standards certification
is implied by this documentation.

The implemented interfaces in `physioworld.verification` include
`ContextOfUse`, `CredibilityCase`, `ReferenceCase`, `ReferenceRegistry`,
`ValidationEvidence` and `EnvelopeBound`. A `ReferenceRegistry.run()` executes
the production and independent calculation now and emits `VerificationRun`
receipts with observed/reference values, residuals, input hashes and status.
`CredibilityCase.assess()` evaluates registered requirements and evidence for
the declared context rather than reusing a generic validated flag. The
[VVUQ tests](../../tests/test_causal_uncertainty_vvuq.py) exercise this boundary.

## Credibility case

For each model/version record:

1. **Context of use:** user, scientific question, decision and consequence of error.
2. **Model:** equations/algorithm, state, inputs, parameters, units and topology.
3. **Operating envelope:** physiological/resource/time ranges and action support.
4. **Verification:** analytic and independent numerical references with tolerances.
5. **Validation:** relevant measurements/cohorts, protocol, endpoints and discrepancy.
6. **Uncertainty:** parameter origin, statistical construction, propagation and sensitivity.
7. **Evidence:** typed claims, assumptions/status, data lineage and reproducibility.
8. **Interpretation:** supported uses, extrapolation rules and remaining scientific boundaries.

## Context examples

| Context | Credibility evidence | Boundary |
|---|---|---|
| Capacitor constant-current exercise | Analytic safety equation and differential integration | Model-derived, zero-leakage reference; not calibrated hardware |
| Synthetic cross-layer scenario | Configuration, seed, unit/time semantics, reference metrics and replay | A simulation tests the declared model; it does not establish population validity |
| Retrospective prediction | Acquired data, subject-safe splits, calibration and clustered uncertainty | Predictive validation does not identify intervention effects |
| Clinical protocol readiness | Frozen intended use, population, endpoint, thresholds and data access | Readiness is not clinical performance |

## Clinical and causal qualification

Clinical evaluation freezes preprocessing, task, representation, thresholds,
resource/channel regime and independent analysis population before evaluation.
Require reference standards, subgroup definitions, calibration, false-alarm burden,
uncertainty and error/harm analysis appropriate to the intended decision.

A causal identification record additionally specifies treatment/outcome, time
zero, eligibility, follow-up/censoring, temporal availability, consistency,
exchangeability, positivity, action support, estimator and sensitivity.
Mandatory violated or unchecked assumptions cannot support a high-assurance claim.

Review [scientific scope](../scientific-scope.md),
[clinical validation](../clinical_validation_guide.md), and the
[Verification Manual](verification-manual.md).

# Verification Manual

## Purpose and evidence model

This manual maps scientific requirements to executable references. Each
qualification execution must identify the software version/commit, test ID,
context, hashed inputs, reference outputs, tolerance, independent comparison,
pass/fail rule and evidence class. Specifications below are not execution results.

`NUMERICAL_VERIFICATION` describes computational agreement. `SIMULATION` describes
model-generated experiments. Empirical, external predictive, causal and clinical
evidence require their own data/protocol records. A unit test does not prove a theorem.

## Reference cases

| ID | API and context | Reference and pass rule | Test artifact |
|---|---|---|---|
| PW-V-001 | `canonical_current_limit`, zero-leakage constant current | `0.8/58 A`, terminal 1.85 V; 12-decimal analytic assertions | [scientific engines](../../tests/test_scientific_engines.py) |
| PW-V-002 | `CapacitorResource.step` | Independent Euler comparison; 8-decimal fixture assertion | [scientific engines](../../tests/test_scientific_engines.py) |
| PW-V-003 | `FosterResource.step` | Independent 10,000-step branch integration; 4-decimal branch and 8-decimal main voltage fixture assertions | [P2 and safety](../../tests/test_safety_causal_and_p2.py) |
| PW-V-004 | `support_gap` | Optimize H and Z policy sets separately; expected policies and gap 0.1 | [scientific engines](../../tests/test_scientific_engines.py) |
| PW-V-005 | `binary_forecasting_certificate` | Reject absent assumptions; valid binary comparison has nonnegative gap | [scientific engines](../../tests/test_scientific_engines.py) |
| PW-V-006 | `compute_pir` | Information + policy loss = total loss, explicit references | [scientific engines](../../tests/test_scientific_engines.py) |
| PW-V-007 | `RLNCEncoder`, `RLNCDecoder` | Exact source bytes after reordering/full rank; erasure path rejects reconstruction | [communication/policies](../../tests/test_communication_and_policies.py) |
| PW-V-008 | `DynamicProgrammingOracle` | Two-step finite-MDP optimum action `go`, value 2.0 | [communication/policies](../../tests/test_communication_and_policies.py) |
| PW-V-009 | Time transforms and compilation | Explicit resampling lineage, duplicate-record detection, missing-field rejection and configuration hash | [data/time/compiler](../../tests/test_data_time_and_compiler.py) |
| PW-V-010 | Synthetic intervention and safety | Known SCM ATE 2.0, unsafe voltage classification, emergency stop | [P2 and safety](../../tests/test_safety_causal_and_p2.py) |
| PW-V-011 | Provenance and clinical protocol | Run manifest/report, dataset/clinical readiness carries no fabricated performance | [provenance/CLI](../../tests/test_provenance_cli.py) |
| PW-V-012 | Coupled resource/control | Independent DOP853 state/energy, matrix exponential, threshold events, robust policy quantifiers and approximation bounds | [resource/control qualification](../../tests/test_resource_control_qualification.py) |
| PW-V-013 | Context-of-use VVUQ | Fresh independent reference receipts, uncertainty, discrepancy and credibility evidence | [causal/uncertainty/VVUQ](../../tests/test_causal_uncertainty_vvuq.py) |
| PW-V-014 | Laboratory/device/FMI boundaries | Calculation graph, synchronized simulated acquisition, calibration and executable coupling contracts | [instrumentation/device/FMI](../../tests/test_instrumentation_devices_fmi.py) |
| PW-V-015 | Physiological scenarios | Coupled model, scenario events, serialization, population and assimilation contracts | [physiological scenarios](../../tests/test_physiological_scenarios.py) |

Consult the test file at the evaluated commit for all parameters and assertions.
These cases are a reference map, not an exhaustive list of newly added tests.

## Execution

```bash
uv sync --frozen --all-extras
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m pytest -q
uv run --no-sync python tools/check_schemas.py
uv run --no-sync physioworld benchmark
```

For package qualification, build wheel/sdist and run
[clean-install smoke](../../tools/smoke_install.py) outside the repository.
CI checks the declared OS/Python/dependency profiles. Record actual outcomes
and attach compact evidence artifacts to a release bundle.

## Numerical and scientific review

Extend a reference envelope through convergence, conservation, positivity,
stability, threshold events and parameter corners. Use independent algorithms,
not copies of the production shortcut. Semantic permutation, label/time shuffle,
wrong-unit and future-information controls need their own expected failure rules.
Preserve subject clustering in statistical uncertainty.

Curated small fixtures can be versioned; raw run logs stay in CI/user workspaces.
See [numerical qualification](numerical-qualification.md),
[scientific verification](scientific-verification.md) and
[release qualification](release-qualification.md).

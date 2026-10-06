# Event-time monitoring and model governance

`physioworld.monitoring` provides deterministic, dependency-free diagnostics
for nonstationarity and explicit adaptation controls. The implementation is
intended for research workflows: a detected distribution change is evidence of
an observed statistical change, not evidence of a physiological cause or a
causal effect.

## Event-time drift

Create an `EventObservation` with an event timestamp, clock basis, feature
values, and—when relevant—prediction and label availability times. Configure a
`DriftMonitorConfig` with a mandatory training cutoff and half-open event-time
window geometry:

```python
from physioworld.monitoring import DriftMonitorConfig, EventObservation, WindowedDriftMonitor

config = DriftMonitorConfig(
    training_cutoff_s=100.0,
    clock_basis="study-clock",
    window_size_s=60.0,
    step_s=30.0,
    alpha=0.05,
)
monitor = WindowedDriftMonitor(reference_events, config)
reports = monitor.scan_covariate(current_events, "heart_rate")
```

`scan_covariate` uses a two-sample KS distance with its asymptotic two-sided
p-value. Ties are evaluated at unique observed values. Reference events must
be at or before the cutoff, and current events must be after it; invalid
chronology and incompatible clock bases fail loudly.

The `DriftDomain` declaration labels the monitored covariate as a physiology,
device, site, protocol, population, or calibration diagnostic. The label is a
governance/provenance statement; it does not turn a distributional comparison
into a causal attribution.

`scan_concept` is separate from covariate drift. It compares binned observed
conditional label means, requiring explicit label-availability times and an
`as_of_s` time. It does not identify a causal effect. Performance drift is a
third object: `TaskValuePoint` values are compared only after
`EstimandFirewall` exact comparison of every `MetricContext`. Consequently,
different utilities, populations, horizons, estimators, or metric types cannot
be averaged as if they were one task value.
`CapabilityValuePoint` and `track_capability_value` provide the corresponding
chronological capability series under a `metric_type="capability"` context.

## Adaptation and leakage policy

`AdaptationPolicy` declares whether a model is `frozen`, `periodic`, or
`drift_triggered`, as well as its cutoff, clock, interval, and embargo.
`compile_adaptation_split` converts event metadata to the existing
`SplitItem` contract and delegates chronology, overlap, future-input, label
leakage, and preprocessing semantics to `SplitCompiler`. `eligible_fit_events`
checks that all selected inputs and supervised labels were available at the
declared fit time. Adaptation cutoffs cannot be at or before the initial
training cutoff.

## Model registry and change impact

`physioworld.core.ModelDescriptor` is a typed descriptor for a physiology,
statistical, machine-learning, communication, resource, or policy model. It
records a strict semantic version, family/type, equations or algorithm,
assumptions, parameters, input/output schemas, `OperatingEnvelope`,
`ContextOfUse`, evidence class, and affected claim/test/benchmark IDs.

`ModelRegistry.register` computes a stable SHA-256 content hash, prevents
reusing a version with different content, and requires a semantic major bump
for incompatible model-family or input/output API changes. It returns a
`ChangeImpact` containing changed fields, affected evidence references,
whether prior results are invalidated, and an `EvidenceGraph` suitable for
impact traversal. Previous descriptors remain addressable by version.
`DeprecationPolicy` and `release_tag` make lifecycle and release labeling
explicit; `require_active` refuses deprecated or retired descriptors.

All registry, descriptor, diagnostic, policy, and reduction objects are
`VersionedRecord`s and support strict JSON round trips. Unknown fields and
schema versions are rejected by the shared serialization contract.

## Model reduction

`fit_pod` is an optional NumPy-backed proper orthogonal decomposition. It
centers a snapshot matrix, computes an SVD, retains orthonormal right-singular
modes, and records reference snapshot reconstruction RMSE plus an optional
tolerance. The model exposes projection and reconstruction; speed alone is
never treated as equivalence. NumPy is optional and is not required for drift
monitoring, governance, or the example.

The example is executable without optional dependencies:

```text
python examples/drift_monitoring.py
```

Its output is synthetic software-verification evidence, not empirical,
clinical, or causal validation.

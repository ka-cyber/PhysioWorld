# Getting started

## First run

After [installation](installation.md), use the explicitly synthetic example:

```bash
physioworld doctor
physioworld qualify-install
physioworld run examples/minimal_world.json --output .physioworld/minimal
physioworld provenance .physioworld/minimal
physioworld report .physioworld/minimal
physioworld reproduce .physioworld/minimal/result_manifest.json --output .physioworld/minimal-reproduced
```

Expected artifacts include `summary.json`, `result_manifest.json` and a
human-readable report. The seed, configuration and declared transformations
provide the context for the summary. Machine/runtime fields may differ across
reruns; deterministic scientific outputs are compared separately.

## Three verticals

```bash
physioworld run examples/cardio_world.json --output .physioworld/cardio
physioworld run examples/neuro_world.json --output .physioworld/neuro
physioworld run examples/icu_world.json --output .physioworld/icu
```

Inspect each task definition, horizon, unit, missingness rule, representation,
channel, resource model and seed. These shipped scenarios are synthetic;
physiological plausibility and repeatability do not establish clinical validity.

## First scientific calculation

The core capacitor reference is usable from any installed environment:

```python
from physioworld.resources import CapacitorResource, ResourceState, canonical_current_limit

current = canonical_current_limit(2.65, 1.85, 0.2, 8.0, 10.0)
model = CapacitorResource(0.2, 8.0, v_min=1.85)
transition = model.step(ResourceState(2.65), current, 10.0)
print(current, transition.terminal_voltage)  # about 0.0137931 A and 1.85 V
```

This evaluates a constant-current, zero-leakage model. The reference is
**numerical/model-derived evidence**, not a measured device current. The
[numerical qualification](quality/numerical-qualification.md) explains the
assumptions and independent comparison.

## Interpret an experiment

1. Identify the estimand and data mode before reading a metric.
2. Check the task, population, horizon, units, utility and reference.
3. Inspect assumptions, uncertainty and operating envelope.
4. Confirm input/output hashes and replay the manifest.
5. Compare only compatible evaluation regimes.

See [scientific scope](scientific-scope.md), [CLI](cli.md) and
[reproducibility](reproducibility.md) for the next steps.

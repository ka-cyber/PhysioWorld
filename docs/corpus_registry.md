# Corpus and experiment registry

PhysioWorld ships two package-bound registries:

- `physioworld.corpus/registry.json` maps source artifacts to equations, claims,
  estimands, experiments, datasets, fixtures, implementation modules, and
  validation gates.
- `physioworld.corpus/experiments.json` gives every Level 3 experiment a stable
  ID, seed, configuration identity, unit of analysis, evidence class, expected
  diagnostic, and falsification conditions.

The registries distinguish executable synthetic or mathematical fixtures,
software/numerical verification, extracted references from inaccessible source
artifacts, and external experiments blocked until authorized raw data and
protocol artifacts are supplied.

Inspect them programmatically:

```python
from physioworld.corpus import load_experiment_registry, source_registry

print([item.source_id for item in source_registry()])
print([item.experiment_id for item in load_experiment_registry()])
```

Run the complete local orchestration with:

```bash
physioworld scientific-suite --profile fast --output .physioworld/suite
```

The resulting `suite_result.json` is a curated qualification receipt. It is not
a publication result table and it does not promote inaccessible or historical
reference values to active evidence.

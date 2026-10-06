# PhysioWorld

**An executable computational laboratory for physiology and physiological-interface research.**

PhysioWorld connects physiological observations and temporal history to
representations, task laws, communication, computation, dynamic resources,
policies, endpoints, and reproducible evidence. It is useful for asking which
information a declared task needs, how transport and resource constraints change
attainable capability, and what evidence supports the resulting conclusion.

Version **0.1.0** has a dependency-light Python core. Scientific, dataset,
configuration, schema-validation, and plotting integrations are optional.
Software verification, simulation, predictive validation, and causal/clinical
evidence have distinct meanings throughout the platform.

## Install and run your first experiment

Python **3.10–3.14** is the declared support range. From a checkout:

```bash
python -m venv .venv
# POSIX; on Windows use .venv\Scripts\activate
source .venv/bin/activate
python -m pip install .
physioworld doctor
physioworld qualify-install
physioworld run examples/minimal_world.json --output .physioworld/first-run
physioworld provenance .physioworld/first-run
physioworld report .physioworld/first-run
physioworld reproduce .physioworld/first-run/result_manifest.json --output .physioworld/first-run-reproduced
```

The example uses explicit synthetic samples and a declared seed. Inspect the
summary, provenance, and report before interpreting its metrics. User-generated
outputs belong in an ignored workspace or a selected external directory.

For an exact development/reproduction environment, use **uv 0.12.23**:

```bash
uv sync --frozen --all-extras
uv run --no-sync physioworld doctor
```

For an installed wheel or sdist, installation qualification works independently
of a checkout: JSON schemas are package resources. Supply your own experiment
file or copy a documented example from the source distribution. See
[installation](docs/installation.md) for locked pip environments and extras.

## Three end-to-end research workflows

```bash
physioworld run examples/cardio_world.json --output .physioworld/cardio
physioworld run examples/neuro_world.json --output .physioworld/neuro
physioworld run examples/icu_world.json --output .physioworld/icu
```

| Workflow | Scientific question | Evidence interpretation |
|---|---|---|
| Cardiovascular | How do history, representation, channel and resource choices affect a declared physiological task? | Synthetic/model-derived task and capability results |
| Neurophysiology | Which temporal/multimodal information survives the declared communication and uncertainty regime? | Synthetic neurophysiology experiment, not a clinical EEG study |
| ICU telemetry | How do multivariate telemetry, delay and missingness affect a deterioration task? | Synthetic endpoint analysis with an explicit causal boundary |

All three use the same configuration → execution → evidence workflow. A dataset
registry entry is access metadata, not a downloaded dataset; dataset-backed
workflows require the declared records and permissions. Synthetic and acquired
data are never interchangeable evidence.

## Architecture

```text
world / dataset → observations → history → representation → task law
                                      ↓
                       channel → compute → resource → policy → endpoint
                                      ↓
                      evaluation → evidence → provenance → reproduction
```

The scientific environment is `E = (S, R, H, A, Y, K, Π, C, I)`.

- **Data/time:** signals, units, clocks, annotations, missingness and lineage.
- **Scientific engines:** task laws, representations, capability geometry,
  PIR/PIV, channels, physical resources, computation and policy evaluation.
- **Execution:** declarative experiments, simulation/replay and reproducible
  manifests with explicit input and output semantics.
- **Credibility:** assumptions, numerical references, uncertainty, context of
  use and guarded causal/clinical interpretation.

Packet decoding is not task-semantic preservation. AUC is not task-law
discrepancy. Coordinatewise task optima need not be jointly attainable. PIR is
relative to task, utility, horizon, policy class, resource regime and reference;
it is not a universal waveform score.

## Documentation

- **Use:** [getting started](docs/getting_started.md),
  [CLI](docs/cli.md), [data and time](docs/data_time_guide.md),
  [datasets](docs/dataset_guide.md), [reproducibility](docs/reproducibility.md),
  [native Level 3 corpus](docs/corpus_guide.md),
  [clinical protocol readiness](docs/clinical_validation_guide.md).
- **Extend:** [architecture](docs/architecture.md),
  [API reference](docs/api_reference.md), [developer guide](docs/developer_guide.md),
  [contributing](CONTRIBUTING.md).
- **Interpret:** [scientific specification](docs/scientific_specification.md),
  [equations and implementation](docs/mathematical_reference.md),
  [scientific scope](docs/scientific-scope.md),
  [known limitations](docs/KNOWN_LIMITATIONS.md).
- **Qualify:** [Verification Manual](docs/quality/verification-manual.md),
  [numerical qualification](docs/quality/numerical-qualification.md),
  [model credibility](docs/quality/model-credibility.md),
  [release qualification](docs/quality/release-qualification.md).

The [documentation index](docs/index.md) is the complete reading guide. Build
an offline HTML copy with `python tools/docs.py build --output /path/to/new-site`;
only the standard library is needed.

## Research and release scope

PhysioWorld provides research infrastructure for modeling, information
preservation and resource-aware experiments. Simulated physiology and
model-derived energy are not patient or hardware measurements. Causal
identification requires a supported intervention design and checked assumptions;
clinical validity requires study evidence for the intended context of use.

Release quality is established by actual qualification artifacts. A package
version, passing example or CI badge alone does not establish production
acceptance, clinical benefit or regulatory approval. See the
[release process](release/README.md) and [security policy](SECURITY.md).

## Citation and license

Author: **Navaneetha Krishnan Kamalakannan**. Cite the software version actually
used with the person metadata in [CITATION.cff](CITATION.cff); no DOI or ORCID is
assigned here. PhysioWorld is [MIT licensed](LICENSE). Dependency licenses and
notices remain applicable; see [third-party attribution](docs/third-party.md).

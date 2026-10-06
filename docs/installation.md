# Installation

## Runtime package

Python 3.10–3.14 is supported by the declared package policy. Use a virtual
environment. From a source checkout or extracted sdist:

```bash
python -m pip install .
physioworld doctor
physioworld qualify-install
```

For an already-built wheel, use `python -m pip install /path/to/physioworld-0.1.0-py3-none-any.whl`.
There is no mandatory third-party core runtime dependency. Package schemas are
bundled under `physioworld/schemas` and accessed through package resources;
installation qualification must not depend on root docs, tests or configurations.
Examples and release tools are included in the sdist, not assumed to be installed
beside the wheel. Pass explicit user configuration paths to runtime commands.

## Optional capabilities

```bash
python -m pip install '.[scientific,wfdb,yaml,plot,schema]'
```

| Extra | Dependency boundary |
|---|---|
| `scientific` | NumPy arrays and SciPy independent solver/matrix-exponential work |
| `wfdb` | WFDB waveform/annotation input, with WFDB's declared transitive dependencies |
| `yaml` | PyYAML safe configuration loading |
| `plot` | Matplotlib capability visualization |
| `schema` | jsonschema validation against declared JSON Schema drafts |
| `arrow` | PyArrow Arrow IPC and Parquet tables with canonical metadata |
| `hdf5` | h5py/NumPy scientific containers |
| `nwb` | PyNWB neurophysiology interchange |
| `dicom` | pydicom metadata/pixel interchange |
| `serial` | pyserial transport boundary; device qualification remains separate |
| `fmi` | FMPy native FMI 3 co-simulation and independent model-description validation; a compatible FMU binary is required |
| `dev` | Tests, property tests, coverage, lint, types, build and supply-chain checks |

An absent integration should be reported by capability name. Selecting a method
with an unmet dependency must not silently select a different scientific model.
JSON configurations and core analytical workflows do not require YAML or plotting.

Select a format extra explicitly, for example `python -m pip install '.[arrow,hdf5]'`.
Each adapter preserves the declared unit/time envelope; arbitrary files without
required scientific metadata must not be interpreted by guessed column names.

## Reproducible environment

Use uv 0.12.23 with the committed universal lock, which has Python-version and
platform markers plus distribution hashes:

```bash
uv sync --frozen --no-default-groups                  # core
uv sync --frozen --no-default-groups --extra scientific --extra schema
uv sync --frozen --no-default-groups --all-extras       # development
uv run --no-sync physioworld doctor
```

For pip, the [requirements exports](../requirements/README.md) contain the same
resolved versions and hashes:

```bash
python -m pip install --require-hashes -r requirements/requirements-scientific.txt
python -m pip install --no-deps .
```

Plain `pip install '.[scientific]'` resolves compatible dependencies at install
time; it is convenient installation, not an exact locked environment. Record
the installed environment when reporting results.

# Installation qualification

`physioworld.qualification` supplies two installed-package APIs:

```python
import json
from physioworld.qualification import doctor, qualify_install

diagnostics = doctor()
qualification = qualify_install()
print(json.dumps(qualification, indent=2, allow_nan=False))
```

Both return JSON-ready dictionaries with `status`, `version`, `checks`,
`evidence_scope`, and `source_tests`. They operate on installed package modules
and resources and work from an arbitrary current directory. Each check names
its `check` and `location`, and carries `PASS`, `WARNING`, or `ERROR`.

## Diagnostics

`doctor()` executes these checks:

- Interpreter compatibility: Python **3.10–3.14**, matching `>=3.10,<3.15`.
- Discovery and import of every module in the installed `physioworld` package,
  including leaf modules. Import failures are reported individually, with
  discovery continuing through sibling packages.
- Bundled schema inventory through
  `physioworld.experiments.compiler.schema_names()`, and actual schema loading
  through `load_schema()`. Every resource is parsed and checked for finite JSON,
  the declared Draft 2020-12 dialect, bounded structural consistency, and
  resolvable bundled references. The report includes a canonical parsed-document
  SHA-256 digest. This bounded check does not claim full Draft meta-schema
  validation; the `schema` extra supplies a general-purpose validator.
- Installed metadata and module discovery for exactly the capability groups
  below. Optional dependency code is not imported by these probes. Availability
  describes installation/discovery, rather than successful adapter execution or
  dependency version-range qualification.

| Capability extra | Distribution(s) | Import name(s) |
|---|---|---|
| `scientific` | numpy, scipy | numpy, scipy |
| `plot` | matplotlib | matplotlib |
| `wfdb` | wfdb | wfdb |
| `yaml` | PyYAML | yaml |
| `schema` | jsonschema | jsonschema |
| `arrow` | pyarrow | pyarrow |
| `hdf5` | h5py, numpy | h5py, numpy |
| `nwb` | pynwb | pynwb |
| `dicom` | pydicom | pydicom |
| `serial` | pyserial | serial |
| `fmi` | fmpy | fmpy |

A required diagnostic failure produces `ERROR`. Missing optional capabilities
produce `WARNING` with `required: false`, package details, and an exact extra
installation hint. If no warnings or errors occur, diagnostics return `PASS`.

## Executable qualification

`qualify_install()` includes the diagnostic checks and executes the following
standard-library-compatible numerical/reference checks. Its overall result is
`PASS` only when every required check passes. Optional capability warnings remain
visible but do not fail core installation qualification.

### Capacitor reference

For `C = 0.2 F`, `ESR = 8 ohm`, `V0 = 2.65 V`, `Vmin = 1.85 V`, zero leakage,
and `T = 10 s`, the independently calculated current is

```text
I = (V0 - Vmin) / (T/C + ESR) = 0.8/58 A
Vbulk(T) = V0 - I*T/C
Vterminal(T) = Vbulk(T) - I*ESR = 1.85 V
```

Qualification calls the package current-limit functions and executes the actual
resource transition. It compares the current, bulk voltage, terminal voltage,
and energy-balance residual with these references. Current tolerance is
`1e-14 A`; voltage and energy-residual tolerances are `2e-12` in their stated SI
units.

### Correlated PCA reference

The training rows `(12,-2)`, `(11,-1)`, `(8,-4)`, `(9,-5)` have mean `(10,-3)`
and sample covariance

```text
[[10/3, 8/3],
 [ 8/3,10/3]].
```

The eigenvalues are `6` and `2/3`; the leading eigenvector is
`(1,1)/sqrt(2)`, up to sign. Qualification executes fit, transform, and inverse
transform and checks the means, both eigenvalues, sign-invariant leading
direction, projected first row, and full-rank reconstruction. This fixture
detects a coordinate/diagonal proxy. Mean tolerance is `1e-12`; the other
absolute residual tolerances are `2e-12`.

### Three-dimensional RLNC reference

Three distinct five-byte symbols, including high-bit values, are encoded twice
with seed `19`. Each encoded payload is checked against an independent
polynomial-multiplication/long-division implementation of GF(256), polynomial
`0x11D`. All systematic words are discarded before decoding, so reconstruction
requires elimination of repair-only codewords. Qualification checks rank three,
wire serialization/deserialization, exact source-payload equality, emitted word
counts, and deterministic encoder equality. Byte and count tolerances are zero.

### Vertical execution and deterministic replay

Each of `cardiovascular`, `neurophysiology`, and `icu` executes twice through
`run_vertical(name, seed=19)`. The entire serialized pair must agree. The result
must identify the finite-population empirical task-law/independently optimized
policy-utility method, a nonempty evaluated population and real sent packets,
finite availability/discrepancy in `[0,1]`, `MODEL_DERIVED` energy, and an
`UNIDENTIFIED` causal gate. Equality alone cannot qualify arbitrary output.

Two additional executions use a built-in inline `REPLAY` declaration, each
repeated for deterministic equality. Samples are `[90,85,75] mmHg`, times are
`[0,1,2] s`, endpoint horizon is zero, and threshold is `80 mmHg`, giving labels
`[1,1,0]`. Unique full-history atoms are evaluated under ample resource/compute
capacity, zero latency/jitter, and either no packet loss or complete packet loss.

- **Full availability:** availability `1`, task-law discrepancy `0`, and
  selected policy utility `1`.
- **Complete packet loss:** availability `0`, every selected action abstains,
  and utility `0`. The actual unavailable conditional law is `2/3` on all three
  atoms. The independent discrepancy is
  `(abs(1-2/3) + abs(1-2/3) + abs(0-2/3))/3 = 4/9`.

Both cases check the complete endpoint population and zero intrinsic
representation discrepancy. Discrepancy and utility tolerances are `2e-12`;
labels, counts, actions, availability, and replay equality are exact.

Executed numerical checks retain `comparisons`, with each quantity's
`observed`, `reference`, absolute `residual`, `tolerance`, and `passed` flag.
Failed finite comparisons keep their actual observations. Nonfinite observations
are represented as strings with a null residual so failure reports remain valid
JSON. Exceptions carry an `ERROR` and the exception type/message.

## Evidence and integration scope

The public call signatures are `doctor()` and `qualify_install()` with no
arguments. CLI integration can delegate directly and use `status == "ERROR"`
as a failure condition. A diagnostic `WARNING` means optional capabilities are
missing; qualification can still return `PASS` for the core installation.

Every report explicitly records `source_tests.status = "NOT_RUN"`. Source-tree
tests, checkout inventory, release matrices, and provenance/artifact inspection
require separate explicit execution. The APIs neither launch a test subprocess
nor infer successful source-specific checks from their absence.

The evidence scope is installed-package diagnostics or installed-package
numerical verification. The reference fixtures establish limited software
execution and numerical consistency; they do not establish clinical benefit,
causal identification, measured hardware performance, or full release acceptance.

## Focused verification

From a source checkout, execute the explicit stdlib test suite:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_install_qualification.py -v
```

Failure controls cover deterministic garbage output, plausible but incorrect
task-law output, replay changes, incorrect capacitor output, coordinate-proxy
PCA, a wrong finite-field product, invalid bundled schemas, missing resources,
absent optional packages, unsupported interpreters, and leaf-module import
failures. A subprocess runs a package-only copy with `python -I -S` from an
external temporary directory to check operation without a checkout or optional
site packages, and a second copy verifies detection of actual malformed resource
JSON. These subprocesses belong to the explicit tests, not to the runtime APIs.

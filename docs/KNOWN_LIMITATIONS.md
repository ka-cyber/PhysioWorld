# Known limitations

The [source ledger](SCIENTIFIC_SOURCE_LEDGER.yaml) identifies the external
scientific specifications and PIR/UPIT/RA-UPIT research sources. The distributed
[reference ledger](REFERENCE_ARTIFACTS.yaml) contains extracted protocols,
estimands, reference values, and interpretation boundaries. Original research
documents, external cohorts, source tables, research archives, and hardware
measurements are not distributed with PhysioWorld.

The [native Level 3 corpus campaign](corpus_guide.md) executes synthetic and
mathematical fixtures with explicit independent comparisons. Historical values
remain inactive references; the campaign does not reproduce their empirical
headline values. It does not claim raw-data
reproduction, external transport validation, clinical validation, human
measurement, or hardware validation. Raw external data remain an explicit
acquisition boundary.

`VERIFIED-NUMERICAL` entries in the [claim ledger](CLAIM_LEDGER.yaml) describe
test-backed behavior for named fixtures, assumptions, and tolerances. They are
not frozen production certifications or release qualification results. A new
run needs its own hashes, environment, comparisons, and context of use.

The [clinical protocol workflow](clinical_validation_guide.md) currently accepts
complete clinical declarations through the configuration-readiness branch.
Even a declared synthetic dataset on that branch remains configuration-only;
freezing its protocol does not execute a simulation or establish clinical validity.

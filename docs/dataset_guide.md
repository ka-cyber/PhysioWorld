# Dataset guide

`physioworld datasets list` prints the registry. The registry contains
license/access metadata and acquisition instructions, not waveform files.
`OPEN` sources may be acquired according to their current provider license;
`RESTRICTED` and `CREDENTIALED` resources require user-side access and are
never downloaded by the default commands.

Adapters must preserve subject/record identity, sample rate, units,
annotations, provenance, and provider version. Splits are subject-level by
default for human data and must be temporal when the scientific question is
prospective forecasting.

The [provider registry](dataset-registry.md) lists source entry points. At
acquisition, freeze the version, checksum where legally permitted, license,
citation, schema, units, population, modality, intended use, preprocessing,
split rules and access class. Keep credentials and private raw records outside
source control. A synthetic adapter fixture demonstrates data-contract behavior,
not empirical validation of the provider cohort.

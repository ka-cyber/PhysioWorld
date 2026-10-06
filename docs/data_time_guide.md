# Data and time guide

The canonical model retains source-record, simulation, wall-clock, and device
clock domains. `TimeTransform` applies an explicit affine offset/scale;
`SignalStream.resample()` requires a method and records original rate, new
rate, and phase delay in provenance. No adapter or engine silently resamples.

`PhysiologicalRecord` groups streams by subject, record, and session. Quality
validation checks units, rate, timestamps, channels, finite/missing values,
annotation order, calibration, access class, and duplicate records.

`SignalStream` metadata retains subject, record, session, modality, channel,
physical unit/quantity, calibration, acquisition provenance, clock domain,
access class, quality and missingness. `validate_stream()` and
`validate_records()` return structured quality reports rather than silently
repairing the data. Signals grouped into a record retain events and labels.

Unit conversion, resampling, interpolation, imputation, label thresholds and
split changes must be explicitly configured and represented in lineage.
Device, acquisition, transport, computation, decision and outcome timestamps
must not be silently reinterpreted as one clock. For clinical cohorts, preserve
the independent subject/session/recording unit and avoid future-information
leakage. See [contracts](contracts.md) and the [dataset guide](dataset_guide.md).

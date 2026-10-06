"""Synthetic/local interchange fixtures and independent numerical references."""

import importlib.util
import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from physioworld.core import Provenance
from physioworld.datasets import (
    LocalWFDBAdapter,
    PreprocessingArtifact,
    SplitCompiler,
    SplitItem,
    SplitLeakageError,
    SplitSpec,
    default_registry,
    local_checksum,
)
from physioworld.io import (
    ArrowAdapter,
    BIDSAdapter,
    CSVAdapter,
    DICOMAdapter,
    FHIRAdapter,
    HDF5Adapter,
    JSONAdapter,
    MissingOptionalDependency,
    NDJSONAdapter,
    NWBAdapter,
    OMOPAdapter,
    OpenMHealthAdapter,
    ParquetAdapter,
    dumps,
    loads,
)
from physioworld.io.clinical import iso_seconds
from physioworld.io.optional import require
from physioworld.physiology.estimation import (
    EstimatorMetadata,
    ExtendedKalmanFilter,
    GaussianState,
    KalmanFilter,
    ParticleFilter,
    ParticleFilterResult,
    ParticleState,
    fit_affine_calibration,
    fuse_gaussians,
    observability_matrix,
    particle_smooth,
    rts_smooth,
)
from physioworld.physiology.quality import validate_stream
from physioworld.physiology.record import PhysiologicalRecord, RecordEvent
from physioworld.physiology.stream import Annotation, SignalStream
from physioworld.physiology.time import (
    CalibrationMetadata,
    ClockDomain,
    TimeMetadata,
    TimeTransform,
)


def synthetic_stream():
    cal = CalibrationMetadata(
        0.5,
        1.0,
        12,
        -2.0,
        2.0,
        "synthetic_fixture",
        input_unit="count",
        output_unit="mV",
        input_data=("fixture",),
        checksum="synthetic-checksum",
        uncertainty={"output_std": 0.2},
        operating_conditions={"temperature": {"value": 20, "unit": "Cel"}},
    )
    stream = SignalStream.from_samples(
        [[1.0, 2.0], [float("nan"), 3.0], [-0.0, 4.0]],
        None,
        "mV",
        "ECG",
        timestamps=[-1.0, 0.0, 0.7],
        subject_id="subject/α",
        record_id="record:odd/1",
        session_id="session:β",
        device_id="device:1",
        site_id="site:1",
        channel_names=["I", "II"],
        calibration=cal,
        clock=TimeMetadata(
            ClockDomain.DEVICE, origin="synthetic_epoch", uncertainty_s=0.02, clock_id="clock:1"
        ),
        missing_channels=[(False, False), (True, False), (False, False)],
        quality=[1, 0.4, 0.9],
        uncertainty=[(0.2, 0.3)] * 3,
        attributes={
            "nested": {"tuple": (1, {"$float": "literal"}), "list": [None, False]},
            "source_timestamps": ("original:1", "original:2", "original:3"),
        },
        provenance=Provenance("synthetic", "fixture", seed=17, created_at="2020-01-01T00:00:00Z"),
    )
    return replace(
        stream, annotations=(Annotation(-0.5, 0.1, "synthetic event", {"code": (1, 2)}),)
    )


def synthetic_record():
    stream = synthetic_stream()
    return PhysiologicalRecord(
        "subject/α",
        "record:odd/1",
        "session:β",
        {"ECG": stream},
        (
            RecordEvent(
                0.1,
                "synthetic intervention",
                source="fixture",
                clock=stream.metadata.clock,
                kind="intervention",
            ),
        ),
        {"outcome": {"value": None, "available": False}},
        {"source": "synthetic", "history": ("acquisition", "annotation")},
        site_id="site:1",
        device_id="device:1",
        subject={"pseudonymous": True},
        session={"protocol": "synthetic"},
        devices={"device:1": {"model": "fixture", "calibration": "assumed"}},
        metadata={"epochs": ((0.0, 0.7),), "evidence_class": "SIMULATION"},
    )


class CanonicalTests(unittest.TestCase):
    def test_semantic_roundtrip_all_metadata_and_nonfinite_missing_values(self):
        for obj in (synthetic_stream(), synthetic_record()):
            with self.subTest(type=type(obj).__name__):
                restored = loads(dumps(obj))
                self.assertTrue(obj.semantic_equal(restored))
                self.assertEqual(dumps(restored), dumps(obj))
                self.assertEqual(type(obj).from_dict(obj.to_dict()).to_dict(), obj.to_dict())
        self.assertTrue(math.isnan(loads(dumps(synthetic_stream())).values[1][0]))
        self.assertEqual(validate_stream(synthetic_stream()).errors, ())

    def test_long_history_and_empty_optional_fields(self):
        stream = SignalStream.from_samples(list(range(10001)), 100, "mV", "ECG")
        self.assertTrue(stream.semantic_equal(loads(dumps(stream))))
        empty = SignalStream.from_samples([], 10, "mV", "ECG")
        self.assertTrue(empty.semantic_equal(loads(dumps(empty))))

    def test_malformed_unknown_version_and_duplicate_json_keys(self):
        payload = synthetic_stream().to_dict()
        for update in ({"schema_version": "99.0.0"}, {"object_type": "pickle"}, {"extra": 1}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                SignalStream.from_dict({**payload, **update})
        for text in ('{"format":"x","format":"y"}', "NaN", '{"$float":"bogus"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                loads(text)
        payload["data"]["metadata"]["unknown_field"] = True
        with self.assertRaises(ValueError):
            SignalStream.from_dict(payload)
        for field, bad in (("calibration", {}), ("clock", {})):
            payload = synthetic_stream().to_dict()
            payload["data"]["metadata"][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                SignalStream.from_dict(payload)

    def test_no_index_time_invalid_rates_or_nonfinite_clocks(self):
        with self.assertRaisesRegex(ValueError, "explicit timestamps"):
            SignalStream.from_samples([1, 2], None, "mV", "ECG")
        for rate in (True, 0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                SignalStream.from_samples([1, 2], rate, "mV", "ECG")
        for times in ([0, float("nan")], [0, float("inf")], [1, 0]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                SignalStream.from_samples([1, 2], None, "mV", "ECG", timestamps=times)
        for factory in (
            lambda: TimeMetadata(time_unit="ms"),
            lambda: TimeMetadata(sampling_drift_ppm=float("nan")),
            lambda: TimeMetadata("wall_clock"),
            lambda: TimeTransform("device", "simulation", scale=float("nan")),
            lambda: Annotation(float("nan")),
            lambda: CalibrationMetadata(gain=float("nan")),
        ):
            with self.assertRaises(ValueError):
                factory()
        self.assertEqual(TimeMetadata("wall_clock", time_basis="unix").time_basis, "unix")
        with self.assertRaisesRegex(ValueError, "boolean"):
            SignalStream.from_samples([1], 1, "mV", "ECG", missing=["yes"])
        with self.assertRaisesRegex(ValueError, "quality"):
            SignalStream.from_samples([1], 1, "mV", "ECG", quality=[float("nan")])
        with self.assertRaisesRegex(ValueError, "declared missingness"):
            SignalStream.from_samples([float("nan")], 1, "mV", "ECG")

    def test_identity_conflicts_and_unknown_unit_retention(self):
        stream = synthetic_stream()
        for field in ("subject_id", "record_id", "session_id", "site_id"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                synthetic = replace(
                    stream, metadata=replace(stream.metadata, **{field: "conflict"})
                )
                replace(synthetic_record(), streams={"ECG": synthetic})
        unknown = SignalStream.from_samples([1], 1, "source-custom-unit", "sensor")
        self.assertTrue(unknown.semantic_equal(loads(dumps(unknown))))
        with self.assertRaisesRegex(ValueError, "unsupported unit"):
            unknown.convert_units("V")

    def test_explicit_unit_conversion_calibration_and_uncertainty(self):
        source = synthetic_stream()
        result = source.convert_units("V")
        self.assertAlmostEqual(result.values[0][0], 0.001)
        self.assertAlmostEqual(result.metadata.calibration.gain, 0.0005)
        self.assertAlmostEqual(result.metadata.calibration.offset, 0.001)
        self.assertAlmostEqual(result.metadata.calibration.saturation_high, 0.002)
        self.assertAlmostEqual(result.metadata.calibration.uncertainty["output_std"], 0.0002)
        self.assertAlmostEqual(result.metadata.uncertainty[0][0], 0.0002)
        self.assertEqual(result.metadata.missing_channels, source.metadata.missing_channels)
        self.assertEqual(result.metadata.lineage[-1], source.metadata.provenance)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            source.convert_units("s")
        cel = SignalStream.from_samples([0, 100], 1, "Cel", "temperature").convert_units("K")
        self.assertEqual(cel.channel(0), (273.15, 373.15))

    def test_time_transform_rate_annotations_and_lineage(self):
        source = SignalStream.from_samples(
            [1, 2], 2, "mV", "ECG", provenance=Provenance("synthetic", "acquire")
        )
        source = replace(source, annotations=(Annotation(0.25, 0.1, "marker"),))
        result = source.transform_time(
            TimeTransform("source_record", "simulation", 2, 2, uncertainty_s=0.1)
        )
        self.assertEqual(result.timestamps, (2, 3))
        self.assertEqual(result.sample_rate, 1)
        self.assertEqual(result.annotations[0].onset, 2.5)
        self.assertEqual(result.annotations[0].duration, 0.2)
        self.assertEqual(result.metadata.clock.uncertainty_s, 0.1)
        self.assertEqual(result.metadata.lineage[-1].operation, "acquire")

    def test_resampling_support_missing_quality_calibration_and_no_extrapolation(self):
        source = SignalStream.from_samples(
            [[0, 10], [2, float("nan")], [4, 14]],
            1,
            "mV",
            "ECG",
            missing_channels=[(False, False), (False, True), (False, False)],
            quality=[1, 0.2, 0.8],
            uncertainty=[(0.1, 0.2)] * 3,
            calibration=CalibrationMetadata(output_unit="mV"),
        )
        result = source.resample(2, phase_delay_s=0.1)
        self.assertEqual(result.n_samples, 5)
        self.assertEqual(result.channel(0), (0, 1, 2, 3, 4))
        self.assertEqual(
            result.metadata.missing_channels,
            ((False, False), (False, True), (False, True), (False, True), (False, False)),
        )
        self.assertEqual(result.metadata.quality, (1, 0.2, 0.2, 0.2, 0.8))
        self.assertEqual(len(result.metadata.uncertainty), 5)
        self.assertEqual(result.metadata.calibration, source.metadata.calibration)
        self.assertEqual(result.metadata.provenance.notes["missing_policy"], "propagate_support")
        nearest = source.resample(2, method="nearest")
        self.assertEqual(nearest.channel(0), (0, 0, 2, 2, 4))
        irregular = SignalStream.from_samples([0, 0.6], None, "mV", "ECG", timestamps=[0, 0.6])
        self.assertEqual(irregular.resample(1).timestamps, (0,))
        duplicate = SignalStream.from_samples([1, 2], None, "mV", "ECG", timestamps=[0, 0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            duplicate.resample(2)


class CoreAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_json_ndjson_csv_roundtrip_and_reproducible_bytes(self):
        for adapter, obj, suffix in (
            (JSONAdapter(), synthetic_record(), "json"),
            (CSVAdapter(), synthetic_stream(), "csv"),
        ):
            path = self.root / f"roundtrip.{suffix}"
            adapter.write(obj, path)
            self.assertTrue(obj.semantic_equal(adapter.read(path)))
            before = path.read_bytes()
            adapter.write(obj, path)
            self.assertEqual(path.read_bytes(), before)
        path = self.root / "history.ndjson"
        NDJSONAdapter().write([synthetic_stream(), synthetic_record()], path)
        values = list(NDJSONAdapter().read(path))
        self.assertTrue(synthetic_record().semantic_equal(values[1]))
        path.write_text(dumps(synthetic_stream()) + "\n\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            list(NDJSONAdapter().read(path))

    def test_plain_csv_requires_declared_units_and_clock_and_preserves_gaps(self):
        path = self.root / "plain.csv"
        path.write_text("time,ECG\n0,1\n0.3,\n0.9,2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            CSVAdapter().read(path)
        stream = CSVAdapter().read(
            path,
            columns={"ECG": "mV"},
            timestamp_column="time",
            clock=TimeMetadata("acquisition"),
            subject_id="s",
            record_id="r",
        )
        self.assertEqual(stream.timestamps, (0, 0.3, 0.9))
        self.assertEqual(stream.metadata.missing_channels, ((False,), (True,), (False,)))
        self.assertEqual(stream.metadata.channels[0].unit, "mV")
        self.assertEqual(validate_stream(stream).errors, ())
        path.write_text("time,ECG\nnan,1\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            CSVAdapter().read(
                path, columns={"ECG": "mV"}, timestamp_column="time", clock=TimeMetadata()
            )

    def test_precise_absent_dependency_and_broken_package_errors(self):
        for package, fmt, extra in (
            ("wfdb", "WFDB", "wfdb"),
            ("numpy", "estimation", "scientific"),
            ("pyarrow", "Arrow", None),
            ("h5py", "HDF5", None),
            ("pynwb", "NWB", None),
            ("pydicom", "DICOM", None),
        ):
            with self.subTest(package=package):
                error = ModuleNotFoundError("absent", name=package)
                with patch("physioworld.io.optional.import_module", side_effect=error):
                    with self.assertRaises(MissingOptionalDependency) as caught:
                        require(package, fmt, extra)
                    self.assertEqual(caught.exception.package, package)
                    self.assertIn("pip install", str(caught.exception))
        with patch(
            "physioworld.io.optional.import_module",
            side_effect=ModuleNotFoundError("broken", name="internal_dependency"),
        ):
            with self.assertRaises(ModuleNotFoundError) as caught:
                require("pynwb", "NWB")
            self.assertNotIsInstance(caught.exception, MissingOptionalDependency)

    def test_fhir_numeric_components_ucum_and_original_timestamp(self):
        resource = {
            "resourceType": "Observation",
            "id": "obs:1",
            "status": "final",
            "subject": {"reference": "Patient/synthetic"},
            "encounter": {"reference": "Encounter/synthetic"},
            "device": {"reference": "Device/synthetic"},
            "effectiveDateTime": "2020-01-02T03:04:05+05:30",
            "component": [
                {
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6"}]},
                    "valueQuantity": {
                        "value": 120,
                        "system": "http://unitsofmeasure.org",
                        "code": "mm[Hg]",
                    },
                }
            ],
        }
        record = FHIRAdapter().from_resource(resource)
        stream = record.streams["8480-6"]
        self.assertEqual(stream.values, ((120.0,),))
        self.assertEqual(stream.timestamps, (iso_seconds(resource["effectiveDateTime"]),))
        self.assertEqual(stream.metadata.channels[0].unit, "mm[Hg]")
        self.assertEqual(stream.metadata.clock.origin, resource["effectiveDateTime"])
        self.assertEqual(stream.metadata.clock.time_basis, "unix")
        self.assertEqual(record.session_id, "Encounter/synthetic")
        self.assertEqual(record.device_id, "Device/synthetic")
        self.assertEqual(stream.metadata.data_kind, "clinical")
        self.assertTrue(record.semantic_equal(loads(dumps(record))))
        bad = json.loads(json.dumps(resource))
        del bad["component"][0]["valueQuantity"]["system"]
        with self.assertRaisesRegex(ValueError, "UCUM"):
            FHIRAdapter().from_resource(bad)
        bad["effectiveDateTime"] = "2020-01-02T00:00:00"
        with self.assertRaisesRegex(ValueError, "timezone"):
            FHIRAdapter().from_resource(bad)

    def test_fhir_sampleddata_calibration_missing_tokens_and_count(self):
        resource = {
            "resourceType": "Observation",
            "id": "synthetic",
            "subject": {"reference": "Patient/s"},
            "effectiveDateTime": "2020-01-01T00:00:00Z",
            "code": {"text": "ECG"},
            "valueSampledData": {
                "origin": {"value": 1, "system": "http://unitsofmeasure.org", "code": "mV"},
                "period": 500,
                "factor": 2,
                "dimensions": 2,
                "data": "1 2 E 3 4 U",
            },
        }
        stream = FHIRAdapter().from_resource(resource).streams["ECG"]
        self.assertEqual(stream.n_samples, 3)
        self.assertEqual(stream.values[0], (3, 5))
        self.assertEqual(stream.metadata.missing_channels[1], (True, False))
        self.assertEqual(stream.metadata.calibration.gain, 2)
        self.assertEqual(stream.sample_rate, 2)
        self.assertEqual(stream.timestamps[2] - stream.timestamps[0], 1)
        resource["valueSampledData"]["data"] += " 1"
        with self.assertRaises(ValueError):
            FHIRAdapter().from_resource(resource)

    def test_fhir_missing_quantity_and_period_semantics(self):
        resource = {
            "resourceType": "Observation",
            "id": "missing",
            "subject": {"reference": "Patient/synthetic"},
            "code": {"text": "BP"},
            "effectivePeriod": {"start": "2020-01-01T00:00:00Z", "end": "2020-01-01T00:00:10Z"},
            "valueQuantity": {"system": "http://unitsofmeasure.org", "code": "mm[Hg]"},
            "dataAbsentReason": {"text": "synthetic dropout"},
        }
        record = FHIRAdapter().from_resource(resource)
        self.assertEqual(record.streams["BP"].metadata.missing_channels, ((True,),))
        self.assertEqual(record.streams["BP"].metadata.sampling_semantics, "event")
        self.assertEqual(record.events[0].duration, 10)
        self.assertEqual(record.events[0].clock.time_basis, "unix")
        self.assertTrue(record.semantic_equal(loads(dumps(record))))

    def test_open_mhealth_omop_actual_numeric_mapping_and_identity(self):
        document = {
            "header": {
                "id": "synthetic",
                "schema_id": {"namespace": "omh", "name": "heart-rate", "version": "1.0"},
            },
            "body": {
                "heart_rate": {"value": 70, "unit": "beats/min"},
                "effective_time_frame": {"date_time": "2020-01-01T10:00:00+01:00"},
            },
        }
        record = OpenMHealthAdapter().from_document(
            document, subject_id="synthetic-subject", unit_map={"heart_rate": "/min"}
        )
        stream = record.streams["heart_rate"]
        self.assertEqual(stream.values, ((70.0,),))
        self.assertEqual(stream.metadata.channels[0].unit, "/min")
        self.assertEqual(
            stream.metadata.clock.origin, document["body"]["effective_time_frame"]["date_time"]
        )
        with self.assertRaisesRegex(ValueError, "convert explicitly"):
            OpenMHealthAdapter().from_document(
                document, subject_id="s", unit_map={"heart_rate": "Hz"}
            )
        rows = [
            {
                "measurement_id": 1,
                "person_id": 2,
                "visit_occurrence_id": 3,
                "measurement_concept_id": 10,
                "unit_concept_id": 20,
                "value_as_number": 120,
                "measurement_datetime": "2020-01-01T10:00:00+01:00",
            }
        ]
        record = OMOPAdapter().from_measurements(
            rows, concept_names={10: "systolic_BP"}, unit_concepts={20: "mm[Hg]"}
        )[0]
        self.assertEqual((record.subject_id, record.session_id), ("2", "3"))
        self.assertEqual(record.streams["systolic_BP"].values, ((120.0,),))
        self.assertEqual(record.streams["systolic_BP"].timestamps, stream.timestamps)
        rows[0]["measurement_datetime"] = "2020-01-01T09:00:00"
        rows[0]["value_as_number"] = None
        with self.assertRaisesRegex(ValueError, "timezone_name"):
            OMOPAdapter().from_measurements(
                rows, concept_names={10: "BP"}, unit_concepts={20: "mm[Hg]"}
            )
        missing = OMOPAdapter().from_measurements(
            rows, concept_names={10: "BP"}, unit_concepts={20: "mm[Hg]"}, timezone_name="UTC"
        )[0]
        self.assertEqual(missing.streams["BP"].metadata.missing, (True,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            OMOPAdapter().from_measurements(
                rows * 2,
                concept_names={10: "BP"},
                unit_concepts={20: "mm[Hg]"},
                timezone_name="UTC",
            )
        path = self.root / "measurements.csv"
        path.write_text(
            "measurement_id,person_id,measurement_concept_id,unit_concept_id,measurement_datetime,value_as_number,operator_concept_id\n"
            "1,2,10,20,2020-01-01T09:00:00Z,,0\n",
            encoding="utf-8",
        )
        mapped = OMOPAdapter().read(
            path, format="csv", concept_names={10: "BP"}, unit_concepts={20: "mm[Hg]"}
        )[0]
        self.assertEqual(mapped.streams["BP"].metadata.missing, (True,))

    def test_bids_physiology_negative_start_explicit_units_events(self):
        tsv, sidecar, events = (
            self.root / "synthetic.tsv",
            self.root / "synthetic.json",
            self.root / "events.tsv",
        )
        tsv.write_text("1\t2\n3\tn/a\n5\t6\n", encoding="utf-8")
        sidecar.write_text(
            json.dumps(
                {
                    "SamplingFrequency": 2,
                    "StartTime": -0.5,
                    "Columns": ["ecg", "resp"],
                    "ecg": {"Units": "mV"},
                    "resp": {"Units": "1"},
                }
            ),
            encoding="utf-8",
        )
        events.write_text("onset\tduration\ttrial_type\n-0.2\t0.1\tbaseline\n", encoding="utf-8")
        record = BIDSAdapter().read(
            tsv,
            sidecar,
            subject_id="sub-synthetic",
            record_id="r",
            session_id="ses-1",
            clock=TimeMetadata("acquisition"),
            events_tsv=events,
        )
        stream = record.streams["physio"]
        self.assertEqual(stream.timestamps, (-0.5, 0, 0.5))
        self.assertEqual([c.unit for c in stream.metadata.channels], ["mV", "1"])
        self.assertEqual(stream.metadata.missing_channels[1], (False, True))
        self.assertEqual(record.events[0].onset, -0.2)
        self.assertTrue(record.semantic_equal(loads(dumps(record))))


class OptionalAdapterTests(unittest.TestCase):
    setUp = CoreAdapterTests.setUp

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow") or importlib.util.find_spec("h5py"),
        "optional table/container packages",
    )
    def test_native_container_roundtrips_signal_record_and_empty(self):
        for package, adapter, suffix in (
            ("pyarrow", ArrowAdapter(), "arrow"),
            ("pyarrow", ParquetAdapter(), "parquet"),
            ("h5py", HDF5Adapter(), "h5"),
        ):
            if importlib.util.find_spec(package) is None:
                continue
            for obj in (
                synthetic_stream(),
                synthetic_record(),
                SignalStream.from_samples([], 1, "mV", "ECG"),
            ):
                with self.subTest(format=suffix, object=type(obj).__name__):
                    path = self.root / f"synthetic.{suffix}"
                    adapter.write(obj, path)
                    self.assertTrue(obj.semantic_equal(adapter.read(path)))

    @unittest.skipUnless(importlib.util.find_spec("wfdb"), "optional wfdb package")
    def test_local_wfdb_actual_samples_missing_annotations_and_credentials(self):
        import numpy as np
        import wfdb

        values = np.array([[0, 1], [1, 2], [np.nan, 3], [-1, 4]], dtype=float)
        wfdb.wrsamp(
            "synthetic",
            2,
            ["mV", "mV"],
            ["I", "II"],
            p_signal=values,
            fmt=["16", "16"],
            adc_gain=[100, 100],
            baseline=[0, 0],
            write_dir=str(self.root),
        )
        wfdb.wrann(
            "synthetic",
            "atr",
            np.array([0, 2]),
            ["N", "V"],
            aux_note=["fixture", "missing"],
            write_dir=str(self.root),
        )
        spec = default_registry().get("mitdb")
        adapter = LocalWFDBAdapter(spec, self.root / "synthetic")
        self.assertEqual(adapter.validate()["status"], "PASS")
        stream = adapter.read(
            subject_id="verified-synthetic",
            record_id="r",
            session_id="s",
            annotation_extension="atr",
            modality="ECG",
        )
        self.assertEqual(stream.n_samples, 4)
        self.assertEqual(stream.channel("II"), (1, 2, 3, 4))
        self.assertTrue(math.isnan(stream.values[2][0]))
        self.assertEqual(stream.annotations[1].onset, 1)
        self.assertEqual(stream.annotations[0].attributes["aux_note"], "fixture")
        self.assertEqual(stream.metadata.subject_id, "verified-synthetic")
        self.assertEqual(stream.metadata.channels[0].calibration.gain, 0.01)
        reread = adapter.read(
            subject_id="verified-synthetic",
            record_id="r",
            session_id="s",
            annotation_extension="atr",
            modality="ECG",
        )
        self.assertEqual(stream.timestamps, reread.timestamps)
        self.assertEqual(
            SplitItem.from_record(
                PhysiologicalRecord("verified-synthetic", "r", "s", {"ECG": stream})
            ).content_hash,
            SplitItem.from_record(
                PhysiologicalRecord("verified-synthetic", "r", "s", {"ECG": reread})
            ).content_hash,
        )
        self.assertTrue(stream.semantic_equal(loads(dumps(stream))))
        checksum = local_checksum(self.root / "synthetic.dat")
        self.assertEqual(checksum, local_checksum(self.root / "synthetic.dat"))
        restricted = replace(
            spec, access="CREDENTIALED", credential_requirements=("provider authorization",)
        )
        guarded = LocalWFDBAdapter(restricted, self.root / "synthetic")
        with self.assertRaises(PermissionError):
            guarded.read(subject_id="s", record_id="r")
        self.assertEqual(guarded.read(subject_id="s", record_id="r", authorized=True).n_samples, 4)

    @unittest.skipUnless(importlib.util.find_spec("pynwb"), "optional pynwb package")
    def test_nwb_roundtrip_and_external_timeseries(self):
        import numpy as np
        from pynwb import NWBHDF5IO, NWBFile, TimeSeries
        from pynwb.file import Subject

        path = self.root / "synthetic.nwb"
        NWBAdapter().write(
            synthetic_record(), path, session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        self.assertTrue(synthetic_record().semantic_equal(NWBAdapter().read(path)))
        external = NWBFile(
            "synthetic", "record-external", datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        external.subject = Subject(subject_id="synthetic-subject")
        external.add_acquisition(
            TimeSeries(
                name="ECG",
                data=np.array([1, 2, 3]),
                unit="mV",
                starting_time=0.2,
                rate=2.0,
                conversion=0.5,
                offset=1.0,
            )
        )
        with NWBHDF5IO(str(path), "w") as handle:
            handle.write(external)
        record = NWBAdapter().read(path)
        self.assertEqual(record.streams["ECG"].values, ((1.5,), (2,), (2.5,)))
        self.assertEqual(record.streams["ECG"].timestamps, (0.2, 0.7, 1.2))
        self.assertEqual(record.subject_id, "synthetic-subject")
        absolute = SignalStream.from_samples(
            [1, 2],
            None,
            "mV",
            "ECG",
            timestamps=[1577836800, 1577836801],
            clock=TimeMetadata("acquisition", time_basis="unix"),
        )
        NWBAdapter().write(
            absolute, path, session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc)
        )
        self.assertTrue(absolute.semantic_equal(NWBAdapter().read(path)))
        with NWBHDF5IO(str(path), "r") as handle:
            self.assertEqual(tuple(handle.read().acquisition["stream_0"].timestamps[:]), (0, 1))

    @unittest.skipUnless(importlib.util.find_spec("pydicom"), "optional pydicom package")
    def test_dicom_imaging_metadata_identity_pixels_and_roundtrip(self):
        import numpy as np
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        data = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        data.SOPClassUID, data.SOPInstanceUID = (
            meta.MediaStorageSOPClassUID,
            meta.MediaStorageSOPInstanceUID,
        )
        data.StudyInstanceUID, data.SeriesInstanceUID = generate_uid(), generate_uid()
        data.PatientName, data.PatientID = "Synthetic^Fixture", "not-canonical-subject"
        data.Modality, data.AcquisitionDateTime = "OT", "20200101123456.123+0530"
        data.Rows, data.Columns, data.SamplesPerPixel = 2, 2, 1
        data.PhotometricInterpretation = "MONOCHROME2"
        data.BitsAllocated, data.BitsStored, data.HighBit, data.PixelRepresentation = 16, 16, 15, 0
        data.PixelSpacing, data.RescaleSlope, data.RescaleIntercept = [0.5, 0.5], 2, -1
        data.PixelData = np.array([[1, 2], [3, 4]], dtype=np.uint16).tobytes()
        path = self.root / "synthetic.dcm"
        data.save_as(path, enforce_file_format=True)
        record = DICOMAdapter().read(path, subject_id="canonical-synthetic", include_pixels=True)
        image = next(iter(record.images.values()))
        self.assertEqual(image["AcquisitionDateTime"], data.AcquisitionDateTime)
        self.assertAlmostEqual(
            image["acquisition_timestamp_s"], iso_seconds("2020-01-01T12:34:56.123+05:30")
        )
        self.assertEqual(image["pixels"], [[1, 2], [3, 4]])
        self.assertNotIn("PatientID", image)
        self.assertEqual(record.subject_id, "canonical-synthetic")
        self.assertTrue(record.semantic_equal(loads(dumps(record))))


class SplitTests(unittest.TestCase):
    def test_governed_public_registry(self):
        registry = default_registry()
        self.assertEqual(len(registry.list()), 9)
        for spec in registry.specs:
            for field in (
                "source_url",
                "version",
                "license",
                "citation",
                "expected_schema",
                "units",
                "population",
                "intended_use",
                "preprocessing",
                "split_rules",
                "leakage_risk",
                "checksum_policy",
            ):
                self.assertTrue(spec.to_dict()[field], f"{spec.dataset_id}:{field}")
        with self.assertRaises(ValueError):
            replace(registry.specs[0], license="")
        manifest = json.loads(
            (Path(__file__).parents[1] / "datasets" / "registry.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {s.dataset_id for s in registry.specs}, {s["id"] for s in manifest["datasets"]}
        )
        for entry in manifest["datasets"]:
            for field in (
                "license",
                "citation",
                "version",
                "expected_schema",
                "units",
                "population",
                "modality",
                "intended_use",
                "forbidden_use",
                "preprocessing",
                "split_rules",
                "leakage_risk",
                "credential_requirements",
                "checksum_policy",
            ):
                self.assertIn(field, entry)

    def test_deterministic_subject_split_and_protected_site_device(self):
        items = [
            SplitItem(
                f"r{i}",
                f"s{i // 2}",
                f"session{i % 2}",
                site_id=f"site{i // 4}",
                device_id=f"device{i // 4}",
            )
            for i in range(8)
        ]
        spec = SplitSpec("subject", protected_axes=("site", "device"), seed=3)
        first = SplitCompiler().compile(items, spec)
        second = SplitCompiler().compile(list(reversed(items)), spec)
        self.assertEqual(first.assignments, second.assignments)
        self.assertEqual(first.configuration_hash, second.configuration_hash)
        self.assertEqual(sum(first.independent_counts.values()), 4)
        # Interleaved lexical record IDs must not change group ordering on reversal.
        rows = [
            SplitItem(rid, subject)
            for rid, subject in (("z", "s1"), ("a", "s1"), ("m", "s2"), ("b", "s2"), ("c", "s3"))
        ]
        a = SplitCompiler().compile(rows, SplitSpec("subject", seed=12))
        b = SplitCompiler().compile(list(reversed(rows)), SplitSpec("subject", seed=12))
        self.assertEqual(a.configuration_hash, b.configuration_hash)

    def test_all_identity_axes_overlap_and_missing_identity(self):
        for axis in ("subject", "session", "site", "device"):
            first = SplitItem("a", "s", "session", "site", "device")
            second = replace(first, record_id="b")
            with self.subTest(axis=axis), self.assertRaises(SplitLeakageError) as caught:
                SplitCompiler().compile(
                    [first, second], SplitSpec(axis), assignments={"a": "train", "b": "test"}
                )
            self.assertIn(axis + "_overlap", {i.code for i in caught.exception.issues})
        with self.assertRaises(ValueError):
            SplitCompiler().compile([SplitItem("a", "s"), SplitItem("b", "t")], SplitSpec("device"))
        with self.assertRaises(SplitLeakageError):
            SplitCompiler().compile([SplitItem("a", "s")] * 2, SplitSpec("record"))

    def test_duplicate_content_future_input_label_and_preprocessing(self):
        a = SplitItem.from_record(synthetic_record())
        b = replace(a, record_id="copy", subject_id="another")
        with self.assertRaises(SplitLeakageError) as caught:
            SplitCompiler().compile(
                [a, b], SplitSpec("subject"), assignments={a.record_id: "train", "copy": "test"}
            )
        self.assertIn("duplicate_content", {i.code for i in caught.exception.issues})
        with self.assertRaises(SplitLeakageError) as caught:
            SplitCompiler().compile(
                [SplitItem("a", "s"), SplitItem("b", "t")],
                SplitSpec("subject", require_content_hash=True),
            )
        self.assertIn("missing_content_hash", {i.code for i in caught.exception.issues})
        items = [
            SplitItem(
                "a",
                "s",
                start_s=0,
                end_s=2,
                prediction_time_s=1,
                clock_basis="UTC",
                labels_used_as_features=True,
            ),
            SplitItem("b", "t", start_s=3, end_s=4, prediction_time_s=4, clock_basis="UTC"),
        ]
        artifact = PreprocessingArtifact(
            "normalizer", ("a", "b"), available_at_s=5, clock_basis="UTC", causal=False
        )
        with self.assertRaises(SplitLeakageError) as caught:
            SplitCompiler().compile(
                items,
                SplitSpec("subject"),
                assignments={"a": "train", "b": "test"},
                preprocessing=[artifact],
            )
        self.assertTrue(
            {"future_input", "label_leakage", "preprocessing_contamination", "future_preprocessing"}
            <= {i.code for i in caught.exception.issues}
        )

    def test_temporal_prospective_windows_future_labels_and_incomparable_clocks(self):
        items = [
            SplitItem("a", "s", start_s=0, end_s=5, clock_basis="UTC", label_available_s=6),
            SplitItem("b", "t", start_s=12, end_s=15, clock_basis="UTC"),
        ]
        spec = SplitSpec("record", strategy="prospective", cutoff_s=10, embargo_s=1)
        split = SplitCompiler().compile(items, spec)
        self.assertEqual(split.assignments, {"a": "train", "b": "test"})
        for bad, code in (
            (replace(items[0], label_available_s=11), "future_label"),
            (replace(items[0], clock_basis="device_clock"), "incomparable_clocks"),
        ):
            with self.assertRaises(SplitLeakageError) as caught:
                SplitCompiler().compile([bad, items[1]], spec)
            self.assertIn(code, {i.code for i in caught.exception.issues})
        crossing = replace(items[0], end_s=12)
        with self.assertRaises(SplitLeakageError):
            SplitCompiler().compile([crossing, items[1]], spec)

    def test_source_window_overlap_and_domain_holdout(self):
        items = [
            SplitItem(
                "a",
                "s",
                site_id="old",
                source_record_id="original",
                start_s=0,
                end_s=5,
                clock_basis="source",
            ),
            SplitItem(
                "b",
                "s",
                site_id="new",
                source_record_id="original",
                start_s=4,
                end_s=7,
                clock_basis="source",
            ),
        ]
        with self.assertRaises(SplitLeakageError) as caught:
            SplitCompiler().compile(
                items, SplitSpec("record"), assignments={"a": "train", "b": "test"}
            )
        self.assertIn("time_overlap", {i.code for i in caught.exception.issues})
        touching = replace(items[1], start_s=5)
        with self.assertRaises(SplitLeakageError) as caught:
            SplitCompiler().compile(
                [items[0], touching], SplitSpec("record"), assignments={"a": "train", "b": "test"}
            )
        self.assertIn("time_overlap", {i.code for i in caught.exception.issues})
        items[1] = replace(items[1], subject_id="t", start_s=6)
        split = SplitCompiler().compile(
            items, SplitSpec("site", strategy="domain", holdout_values=("new",))
        )
        self.assertEqual(split.assignments, {"a": "train", "b": "test"})
        subject_split = SplitCompiler().compile(
            items,
            SplitSpec("subject", strategy="domain", domain_axis="site", holdout_values=("new",)),
        )
        self.assertEqual(subject_split.assignments, split.assignments)
        self.assertEqual(subject_split.spec.independent_unit, "subject")


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional scientific NumPy dependency")
class EstimationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.metadata = EstimatorMetadata(
            "synthetic process",
            "synthetic measurement",
            {"mean": [0], "covariance": [[1]]},
            ("synthetic Gaussian fixture",),
            "posterior covariance",
        )

    def test_scalar_kalman_and_rts_against_joint_gaussian_conditioning(self):
        np = self.np
        observations = np.array([1.0, -0.5, 2.0])
        model = KalmanFilter([[1]], [[1]], [[0.2]], [[0.5]], GaussianState([0], [[1]]))
        result = model.filter([[z] for z in observations])
        # Independently condition the full random-walk trajectory's joint Gaussian.
        prior_cov = np.array([[1 + 0.2 * (min(i, j) + 1) for j in range(3)] for i in range(3)])
        for k, state in enumerate(result.filtered):
            cross = prior_cov[k, : k + 1]
            observed_cov = prior_cov[: k + 1, : k + 1] + np.eye(k + 1) * 0.5
            expected_mean = cross @ np.linalg.solve(observed_cov, observations[: k + 1])
            expected_var = prior_cov[k, k] - cross @ np.linalg.solve(observed_cov, cross)
            self.assertAlmostEqual(state.mean[0], expected_mean, places=12)
            self.assertAlmostEqual(state.covariance[0, 0], expected_var, places=12)
        expected_mean = prior_cov @ np.linalg.solve(prior_cov + np.eye(3) * 0.5, observations)
        expected_cov = prior_cov - prior_cov @ np.linalg.solve(
            prior_cov + np.eye(3) * 0.5, prior_cov
        )
        smoothed = rts_smooth(result)
        np.testing.assert_allclose([s.mean[0] for s in smoothed], expected_mean, atol=1e-12)
        np.testing.assert_allclose(
            [s.covariance[0, 0] for s in smoothed], np.diag(expected_cov), atol=1e-12
        )

    def test_missing_partial_measurements_and_correlated_multimodal_fusion(self):
        np = self.np
        model = KalmanFilter(
            np.eye(2),
            np.eye(2),
            np.eye(2) * 0.1,
            np.array([[1, 0.2], [0.2, 1]]),
            GaussianState([0, 0], np.eye(2)),
        )
        prior = model.predict()
        self.assertIs(model.update(None), prior)
        partial = model.update([1, float("nan")])
        self.assertAlmostEqual(partial.mean[0], 1.1 / 2.1)
        self.assertEqual(partial.mean[1], 0)
        model = KalmanFilter([[1]], [[1]], [[0]], [[1]], GaussianState([0], [[1]]))
        state = model.update_fused([[1], [2]], [[[1]], [[1]]], [[1, 0.5], [0.5, 2]])
        h, r = np.ones((2, 1)), np.array([[1, 0.5], [0.5, 2]])
        expected_var = 1 / (1 + (h.T @ np.linalg.solve(r, h))[0, 0])
        expected_mean = expected_var * (h.T @ np.linalg.solve(r, [1, 2]))[0]
        self.assertAlmostEqual(state.mean[0], expected_mean)
        self.assertAlmostEqual(state.covariance[0, 0], expected_var)
        fused = fuse_gaussians(
            [GaussianState([1], [[1]]), GaussianState([3], [[1]])], assume_independent=True
        )
        self.assertEqual(fused.mean[0], 2)
        self.assertEqual(fused.covariance[0, 0], 0.5)

    def test_ekf_real_nonlinear_jacobians_and_linear_limit(self):
        nonlinear = ExtendedKalmanFilter(
            lambda x, dt: x + dt * x**2,
            lambda x: x**2,
            lambda x, dt: [[1 + 2 * dt * x[0]]],
            lambda x: [[2 * x[0]]],
            [[0.1]],
            [[0.5]],
            GaussianState([1], [[0.2]]),
            metadata=self.metadata,
        )
        prior = nonlinear.predict(dt=0.1)
        self.assertAlmostEqual(prior.mean[0], 1.1)
        self.assertAlmostEqual(prior.covariance[0, 0], 1.2**2 * 0.2 + 0.1)
        updated = nonlinear.update([1.5])
        gain = prior.covariance[0, 0] * 2.2 / (2.2**2 * prior.covariance[0, 0] + 0.5)
        self.assertAlmostEqual(updated.mean[0], 1.1 + gain * (1.5 - 1.21))
        ekf = ExtendedKalmanFilter(
            lambda x, dt: x,
            lambda x: x,
            lambda x, dt: [[1]],
            lambda x: [[1]],
            [[0.2]],
            [[0.5]],
            GaussianState([0], [[1]]),
            metadata=self.metadata,
        )
        linear = KalmanFilter([[1]], [[1]], [[0.2]], [[0.5]], GaussianState([0], [[1]]))
        a = ekf.filter([[1], None, [2]], timestamps=[1, 2, 3], initial_time_s=0)
        b = linear.filter([[1], None, [2]], timestamps=[1, 2, 3])
        self.np.testing.assert_allclose([s.mean for s in a.filtered], [s.mean for s in b.filtered])
        with self.assertRaises(ValueError):
            ekf.predict(dt=float("nan"))
        model = ExtendedKalmanFilter(
            lambda x, dt: x,
            lambda x: x**2,
            lambda x, dt: [[1]],
            lambda x: [[2 * x[0]]],
            [[0]],
            [[0.5]],
            GaussianState([1], [[0.2]]),
            metadata=self.metadata,
        )
        fused = model.update_nonlinear_fused(
            [[1.2], [0.8]],
            [lambda x: x**2, lambda x: x],
            [lambda x: [[2 * x[0]]], lambda x: [[1]]],
            [[0.5, 0], [0, 0.3]],
        )
        h, r, p = self.np.array([[2.0], [1.0]]), self.np.diag([0.5, 0.3]), self.np.array([[0.2]])
        gain = self.np.linalg.solve(h @ p @ h.T + r, h @ p).T
        self.assertAlmostEqual(fused.mean[0], (1 + gain @ self.np.array([0.2, -0.2]))[0])
        self.assertAlmostEqual(fused.covariance[0, 0], (p - gain @ h @ p)[0, 0])
        self.assertEqual(
            model.update_fused([[float("nan")]], [[[1]]], [[1]]).to_dict(), fused.to_dict()
        )

    def test_invalid_covariances_models_priors_and_observability(self):
        for covariance in ([[-1]], [[float("nan")]], [[1, 2], [0, 1]]):
            with self.subTest(covariance=covariance), self.assertRaises(ValueError):
                GaussianState([0] if len(covariance) == 1 else [0, 0], covariance)
        with self.assertRaises(ValueError):
            KalmanFilter([[1]], [[1]], [[-1]], [[1]], GaussianState([0], [[1]]))
        with self.assertRaises(ValueError):
            KalmanFilter([[1]], [[1]], [[1]], [[0]], GaussianState([0], [[1]]))
        self.assertTrue(observability_matrix([[1, 1], [0, 1]], [[1, 0]])["observable"])
        self.assertFalse(observability_matrix([[1, 0], [0, 1]], [[1, 0]])["observable"])
        cal = fit_affine_calibration(
            [0, 1, 2, 3],
            [1, 3, 5, 7],
            input_unit="V",
            output_unit="mV",
            input_data=["synthetic"],
            checksum="fixture-checksum",
            operating_conditions={},
        )
        self.assertAlmostEqual(cal.gain, 2)
        self.assertAlmostEqual(cal.offset, 1)
        self.assertEqual(cal.status, "FITTED")
        with self.assertRaisesRegex(ValueError, "excitation"):
            fit_affine_calibration(
                [1, 1, 1],
                [1, 2, 3],
                input_unit="V",
                output_unit="mV",
                input_data=["fixture"],
                checksum="x",
                operating_conditions={},
            )

    def test_particle_weights_missing_seed_and_exact_discrete_reference(self):
        np = self.np
        prior = ParticleState([[0], [1], [2]], [0.2, 0.3, 0.5])

        def likelihood(z, particles, mask):
            return -0.5 * (particles[:, 0] - z[0]) ** 2

        model = ParticleFilter(
            lambda particles, dt, rng: particles, likelihood, prior, seed=4, metadata=self.metadata
        )
        weights = prior.weights * np.exp(-0.5 * (prior.particles[:, 0] - 1) ** 2)
        weights /= weights.sum()
        posterior = model.update([1])
        np.testing.assert_allclose(posterior.weights, weights, atol=1e-14)
        self.assertIs(model.update([float("nan")]), posterior)
        self.assertIs(model.update(None), posterior)
        a = model.resample()
        other = ParticleFilter(
            lambda particles, dt, rng: particles, likelihood, prior, seed=4, metadata=self.metadata
        )
        other.update([1])
        np.testing.assert_array_equal(a.particles, other.resample().particles)
        with self.assertRaises(ValueError):
            ParticleState([[0], [1]], [0.1, 0.1])
        unsupported = ParticleFilter(
            lambda p, dt, rng: p,
            lambda z, p, mask: np.full(len(p), -np.inf),
            prior,
            seed=0,
            metadata=self.metadata,
        )
        with self.assertRaisesRegex(ValueError, "zero support"):
            unsupported.update([1])

    def test_particle_gaussian_reference_and_backward_smoothing_enumeration(self):
        np = self.np
        rng = np.random.default_rng(7)
        prior = ParticleState(rng.normal(0, 1, (50000, 1)), np.full(50000, 1 / 50000))
        model = ParticleFilter(
            lambda p, dt, rng: p + rng.normal(0, math.sqrt(0.2), p.shape),
            lambda z, p, mask: (
                -0.5 * (p[:, 0] - z[0]) ** 2 / 0.5 - 0.5 * math.log(2 * math.pi * 0.5)
            ),
            prior,
            seed=11,
            metadata=self.metadata,
        )
        result = model.filter([[1]], timestamps=[1], initial_time_s=0)
        summary = result.states[0].gaussian_summary()
        self.assertAlmostEqual(summary.mean[0], 1.2 / 1.7, delta=0.015)
        self.assertAlmostEqual(summary.covariance[0, 0], 1.2 * 0.5 / 1.7, delta=0.015)
        # Exact two-state HMM forward/backward enumeration.
        transition = np.array([[0.8, 0.2], [0.3, 0.7]])
        emission_a, emission_b = np.array([0.9, 0.1]), np.array([0.2, 0.8])
        first = np.array([0.5, 0.5]) * emission_a
        first /= first.sum()
        second = (first @ transition) * emission_b
        second /= second.sum()
        trace = ParticleFilterResult(
            (ParticleState([[0], [1]], first), ParticleState([[0], [1]], second)), self.metadata, 0
        )
        smoothed = particle_smooth(trace, lambda a, b, dt: np.log(transition), timestamps=[0, 1])
        joint = first[:, None] * transition * emission_b[None, :]
        expected = joint.sum(axis=1) / joint.sum()
        np.testing.assert_allclose(smoothed[0].weights, expected, atol=1e-14)


if __name__ == "__main__":
    unittest.main()

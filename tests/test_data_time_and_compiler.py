import unittest

from physioworld.experiments import ExperimentCompiler, ExperimentValidationError
from physioworld.physiology import (
    ClockDomain,
    PhysiologicalRecord,
    SignalStream,
    TimeTransform,
    validate_records,
)


class DataAndCompilerTests(unittest.TestCase):
    def test_time_transform_and_resample_are_recorded(self):
        stream = SignalStream.from_samples([0.0, 1.0, 2.0], 1.0, "mV", "ECG")
        shifted = stream.transform_time(
            TimeTransform(ClockDomain.SOURCE_RECORD, ClockDomain.SIMULATION, offset_s=2.0)
        )
        self.assertEqual(shifted.timestamps, (2.0, 3.0, 4.0))
        sampled = shifted.resample(2.0, phase_delay_s=0.1)
        self.assertEqual(sampled.sample_rate, 2.0)
        self.assertEqual(sampled.metadata.provenance.notes["method"], "linear")

    def test_record_quality_detects_duplicate_records(self):
        stream = SignalStream.from_samples(
            [1.0, 2.0], 1.0, "mV", "ECG", record_id="r", subject_id="s"
        )
        first = PhysiologicalRecord("s", "r", None, {"ECG": stream})
        report = validate_records((first, first))
        self.assertEqual(report.status.value, "ERROR")

    def test_compiler_rejects_implicit_scientific_fields(self):
        with self.assertRaises(ExperimentValidationError):
            ExperimentCompiler().compile({"dataset": "mitdb", "task": "arrhythmia", "seed": 1})

    def test_compiler_produces_content_hash(self):
        plan = ExperimentCompiler().compile(
            {
                "name": "x",
                "dataset": "mitdb",
                "task": {"id": "a", "horizon_s": 1, "target": "y", "metric": "f1"},
                "representation": "raw",
                "perturbations": [],
                "channel": {},
                "resource": {},
                "compute": {},
                "policy": {},
                "seed": 4,
                "split": "subject_test",
                "metrics": ["f1"],
                "outputs": {},
            }
        )
        self.assertEqual(len(plan.configuration_hash), 64)
        self.assertEqual(plan.data_mode, "dataset")


if __name__ == "__main__":
    unittest.main()

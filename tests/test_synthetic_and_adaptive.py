import unittest

from physioworld.physiology import SyntheticPhysiology
from physioworld.policies import JointPIVUncertaintyPolicy, SensingAction
from physioworld.representations import VariationalInformationBottleneck


class SyntheticAndAdaptiveTests(unittest.TestCase):
    def test_synthetic_stream_has_provenance(self):
        stream = SyntheticPhysiology(seed=5).waveform("ECG", duration_s=1.0, sample_rate=100.0)
        self.assertEqual(stream.n_samples, 100)
        self.assertEqual(stream.metadata.provenance.operation, "waveform")

    def test_joint_policy_uses_declared_piv_and_uncertainty(self):
        ecg = SensingAction("sample", "ECG")
        ppg = SensingAction("sample", "PPG")
        policy = JointPIVUncertaintyPolicy({"ECG": 1.0, "PPG": 2.0}, {"ECG": 0.0, "PPG": 3.0})
        self.assertEqual(policy.action({}, (ecg, ppg)), ecg)

    def test_variational_information_bottleneck_fits_declared_objective(self):
        model = VariationalInformationBottleneck(2, latent_dimension=1, epochs=10, seed=3)
        model.fit(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)), (0, 1, 1, 1))
        self.assertEqual(len(model.transform((1.0, 1.0)).values), 1)
        self.assertEqual(model.spec.name, "variational_information_bottleneck")


if __name__ == "__main__":
    unittest.main()

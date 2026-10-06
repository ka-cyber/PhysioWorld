import math
import unittest

from physioworld.perturbations import AdditiveNoise, Missingness
from physioworld.physiology import SignalStream


class StreamTests(unittest.TestCase):
    def test_metadata_and_deterministic_noise(self):
        stream = SignalStream.from_samples([1, 2, 3], 1.0, "mV", "ECG", subject_id="s1")
        a = AdditiveNoise(0.1, seed=4).apply(stream)
        b = AdditiveNoise(0.1, seed=4).apply(stream)
        self.assertEqual(a.values, b.values)
        self.assertEqual(a.metadata.subject_id, "s1")

    def test_missingness_is_declared_not_recovered(self):
        stream = SignalStream.from_samples([1, 2, 3, 4], 1.0, "mV", "PPG")
        result = Missingness(1.0, block_length=2, seed=1).apply(stream)
        self.assertEqual(result.metadata.missing, (True, True, True, True))
        self.assertTrue(all(math.isnan(row[0]) for row in result.values))


if __name__ == "__main__":
    unittest.main()

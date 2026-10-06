"""Reference checks for the owned NIRS semantic boundary."""

import math
import unittest

from physioworld.core import EvidenceClass
from physioworld.instrumentation.nirs import (
    HemoglobinSignal,
    IntensitySignal,
    NIRSQuantity,
    OpticalDensitySignal,
    filter_optical_density,
    motion_correct,
    optical_density,
    raw_proxy_experiment,
    simulate_motion,
)
from physioworld.physiology.stream import SignalStream


class NIRSReferenceTests(unittest.TestCase):
    def test_exact_optical_density_references(self):
        intensity = IntensitySignal.from_samples([1.0, math.e**-1, 0.5, 2.0], 1)
        optical = optical_density(intensity)

        self.assertEqual(optical.quantity, NIRSQuantity.OPTICAL_DENSITY)
        self.assertEqual(optical.stream.metadata.physical_quantity, "optical_density")
        self.assertEqual(optical.stream.metadata.channels[0].unit, "1")
        self.assertEqual(optical.stream.channel(0)[0], 0.0)
        self.assertAlmostEqual(optical.stream.channel(0)[1], 1.0, places=15)
        self.assertAlmostEqual(optical.stream.channel(0)[2], math.log(2), places=15)
        self.assertAlmostEqual(optical.stream.channel(0)[3], -math.log(2), places=15)
        self.assertEqual(optical.stream.metadata.provenance.notes["normalization"], None)
        self.assertEqual(optical.stream.metadata.provenance.notes["reference_intensity"], None)

    def test_intensity_must_be_strictly_positive(self):
        for values in ([0.0, 1.0], [-1.0, 1.0]):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "positive"):
                IntensitySignal.from_samples(values, 1)

    def test_semantic_types_prevent_hbt_promotion(self):
        intensity = IntensitySignal.from_samples([1.0, math.e**-1, 1.0, math.e**-2], 1)
        optical = optical_density(intensity)

        self.assertNotEqual(optical.quantity, NIRSQuantity.HBT)
        with self.assertRaisesRegex(TypeError, "OpticalDensitySignal"):
            simulate_motion(intensity, [1], 0.1, seed=4)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "OpticalDensitySignal"):
            motion_correct(intensity, [1])  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "relabeling is forbidden"):
            OpticalDensitySignal(
                SignalStream.from_samples(
                    [0.0, 1.0],
                    1,
                    "1",
                    "NIRS",
                    physical_quantity="HbT",
                )
            )

    def test_hbt_requires_external_reconstruction_definition(self):
        stream = SignalStream.from_samples(
            [1.0, 2.0],
            1,
            "umol/L",
            "NIRS",
            physical_quantity="HbT",
        )
        with self.assertRaisesRegex(ValueError, "source definition"):
            HemoglobinSignal(stream, NIRSQuantity.HBT, "")
        with self.assertRaisesRegex(ValueError, "provenance"):
            HemoglobinSignal(stream, NIRSQuantity.HBT, "external modified Beer-Lambert")

    def test_declared_motion_and_filter_helpers_are_seeded_and_mathematical(self):
        optical = optical_density(IntensitySignal.from_samples([1.0] * 5, 1))
        simulated_a = simulate_motion(optical, [1, 3], 0.25, seed=17)
        simulated_b = simulate_motion(optical, [1, 3], 0.25, seed=17)
        self.assertEqual(simulated_a.stream.values, simulated_b.stream.values)
        self.assertEqual(simulated_a.stream.values[0], (0.0,))
        self.assertEqual(simulated_a.stream.metadata.provenance.seed, 17)
        self.assertEqual(
            simulated_a.stream.metadata.provenance.notes["evidence_class"], "SIMULATION"
        )

        corrected = motion_correct(
            OpticalDensitySignal(
                SignalStream.from_samples(
                    [0.0, 2.0, 8.0, 10.0], 1, "1", "NIRS", physical_quantity="optical_density"
                )
            ),
            [1, 2],
        )
        self.assertEqual(corrected.stream.channel(0)[0], 0.0)
        self.assertAlmostEqual(corrected.stream.channel(0)[1], 10 / 3, places=14)
        self.assertAlmostEqual(corrected.stream.channel(0)[2], 20 / 3, places=14)
        self.assertEqual(corrected.stream.channel(0)[3], 10.0)

        filtered = filter_optical_density(optical, 2)
        self.assertEqual(filtered.stream.channel(0), (0.0,) * 5)
        self.assertEqual(filtered.stream.metadata.provenance.notes["stochastic"], False)

    def test_raw_proxy_is_explicitly_od_and_not_published_or_clinical_validation(self):
        values = [math.exp(-1.0)] * 20 + [math.exp(-2.0)] * 30
        intensity = IntensitySignal.from_samples(values, 1)
        result = raw_proxy_experiment(
            intensity,
            event_time_s=20,
            baseline_window_s=(-20, 0),
            response_window_s=(0, 30),
            seed=23,
        )

        self.assertEqual(result.result_kind, "raw_optical_density_proxy")
        self.assertEqual(result.quantity, NIRSQuantity.OPTICAL_DENSITY)
        self.assertEqual(result.evidence_class, EvidenceClass.EMPIRICAL_ESTIMATE)
        self.assertAlmostEqual(result.baseline_mean_od, 1.0, places=15)
        self.assertAlmostEqual(result.response_mean_od, 2.0, places=15)
        self.assertAlmostEqual(result.delta_od, 1.0, places=15)
        self.assertEqual(result.preprocessing_gate, "OPEN_INCOMPLETE")
        self.assertFalse(result.published_reproduction)
        self.assertFalse(result.clinically_validated)
        self.assertIn("No motion correction", " ".join(result.limitations))
        self.assertIn("clinical validation", " ".join(result.limitations))


if __name__ == "__main__":
    unittest.main()

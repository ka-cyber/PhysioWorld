import unittest

from physioworld.capability import CapabilityRegion, compute_pir, support_gap
from physioworld.compute import ComputeEngine, ComputeNode, ModelProfile
from physioworld.resources import (
    CapacitorResource,
    ResourceState,
    canonical_current_limit,
    euler_capacitor_voltage,
)
from physioworld.tasklaws import binary_forecasting_certificate, total_variation


class ScientificEngineTests(unittest.TestCase):
    def test_support_optimizes_policy_sets_independently(self):
        full = CapabilityRegion({"p1": (1.0, 0.0), "p2": (0.0, 2.0)}, "H")
        reduced = CapabilityRegion({"q1": (0.7, 0.0), "q2": (0.0, 1.9)}, "Z")
        result = support_gap(full, reduced, (1.0, 1.0))
        self.assertEqual(result["full_policy"], "p2")
        self.assertEqual(result["reduced_policy"], "q2")
        self.assertAlmostEqual(result["gap"], 0.1)

    def test_forecasting_bound_requires_assumptions(self):
        with self.assertRaises(ValueError):
            binary_forecasting_certificate([0.9], [0.8])
        result = binary_forecasting_certificate(
            [0.9, 0.1],
            [0.8, 0.2],
            assumptions=(
                "binary_outcome",
                "same_target",
                "same_population",
                "representation_is_measurable",
            ),
        )
        self.assertTrue(result["valid"])
        self.assertGreaterEqual(result["gap"], 0)

    def test_capacitor_analytical_limit_and_transition(self):
        current = canonical_current_limit(2.65, 1.85, 0.2, 8.0, 10.0)
        self.assertAlmostEqual(current, 0.8 / 58.0, places=12)
        model = CapacitorResource(0.2, 8.0, v_min=1.85)
        transition = model.step(ResourceState(2.65), current, 10.0)
        self.assertAlmostEqual(transition.terminal_voltage, 1.85, places=12)
        self.assertTrue(transition.feasible)
        self.assertAlmostEqual(
            transition.next.voltage, euler_capacitor_voltage(2.65, current, 0.2, 10.0), places=8
        )

    def test_pir_decomposition(self):
        result = compute_pir(1.0, 0.8, 0.7, 0.0, references={"task": "demo"})
        self.assertAlmostEqual(result.information_loss + result.policy_loss, result.total_loss)
        self.assertAlmostEqual(result.pir, 0.8)

    def test_distribution_metric(self):
        self.assertAlmostEqual(total_variation((0.5, 0.5), (1.0, 0.0)), 0.5)

    def test_compute_energy_is_tagged(self):
        result = ComputeEngine(
            {"local": ComputeNode("local", "local", 10.0, 1024, 2.0)},
            {"model": ModelProfile("model", 10, 20)},
        ).execute("local", "model")
        self.assertEqual(result.energy.status.value, "SIMULATED")


if __name__ == "__main__":
    unittest.main()

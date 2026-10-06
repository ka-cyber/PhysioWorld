"""Independent finite-vector checks for capability PIR extensions."""

import unittest
from dataclasses import replace

from physioworld.capability import (
    CapabilityPreservationRegion,
    CapabilityRegion,
    compute_pir,
    finite_causal_pir,
    finite_policy_pir,
    finite_support_gap,
    information_loss,
    optimistic_value,
    pir_identity,
    policy_loss,
    robust_value,
    total_loss,
)
from physioworld.capability.pir import capability_preservation_region
from physioworld.capability.values import optimal_task_value
from physioworld.core import (
    CausalStatus,
    EstimandMismatch,
    EvidenceClass,
    Metric,
    MetricContext,
)


def value_context(information: str, *, utility: str = "finite utility") -> MetricContext:
    return MetricContext(
        "policy_value",
        "synthetic finite population",
        "fixed finite evaluation",
        (),
        "declared-safe-policies",
        "initial state",
        4.0,
        ("task-a", "task-b"),
        "fixed resource",
        "none_declared",
        metric_type="task_value",
        utility=utility,
        population="synthetic population",
        causal_status=CausalStatus.MODEL_INTERVENTIONAL,
        evidence_class=EvidenceClass.SIMULATION,
        communication_regime="declared channel",
        information_condition=information,
        estimator="finite enumeration",
        time_unit="s",
    )


def finite_regions() -> tuple[CapabilityRegion, CapabilityRegion]:
    ideal_context = value_context("ideal history")
    actual_context = value_context("actual representation")
    return (
        CapabilityRegion(
            {"ideal-a": (8.0, 2.0), "ideal-b": (5.0, 6.0), "ideal-c": (4.0, 4.0)},
            "ideal",
            task_labels=("task-a", "task-b"),
            context=ideal_context,
        ),
        CapabilityRegion(
            {"actual-x": (6.0, 1.0), "actual-y": (4.0, 5.0), "actual-z": (3.0, 3.0)},
            "actual",
            task_labels=("task-a", "task-b"),
            context=actual_context,
        ),
    )


def direct_support(
    policy_values: dict[str, tuple[float, ...]], direction: tuple[float, ...]
) -> tuple[float, str]:
    """Independent reference implementation, intentionally not CapabilityRegion.support."""
    scored = {
        policy: sum(weight * value for weight, value in zip(direction, vector))
        for policy, vector in policy_values.items()
    }
    policy = max(scored, key=scored.__getitem__)
    return scored[policy], policy


class CapabilityPIRTests(unittest.TestCase):
    def test_explicit_losses_and_pir_identity_under_common_references(self):
        ideal, actual = finite_regions()
        direction = (0.5, 1.0)
        ideal_values = {
            policy: sum(q * value for q, value in zip(direction, vector))
            for policy, vector in ideal.policy_values.items()
        }
        actual_values = {
            policy: sum(q * value for q, value in zip(direction, vector))
            for policy, vector in actual.policy_values.items()
        }
        ideal_optimum = optimal_task_value(ideal_values, context=ideal.context)
        actual_optimum = optimal_task_value(actual_values, context=actual.context)
        deployed = Metric(4.0, actual.context, "deployed-controller-value")
        minimum = Metric(-2.0, actual.context, "minimum-reference-value")

        self.assertEqual((ideal_optimum.value, actual_optimum.value), (8.5, 7.0))
        self.assertEqual(information_loss(ideal_optimum, actual_optimum), 1.5)
        self.assertEqual(policy_loss(actual_optimum, deployed), 3.0)
        self.assertEqual(total_loss(ideal_optimum, actual_optimum, deployed), 4.5)

        result = compute_pir(
            ideal_optimum.value,
            actual_optimum.value,
            deployed.value,
            minimum.value,
            value_contexts={
                "ideal": ideal.context,
                "actual_optimal": actual.context,
                "actual_policy": actual.context,
                "minimum": actual.context,
            },
            information_conditions=("ideal history", "actual representation"),
        )
        self.assertEqual(
            (result.information_loss, result.policy_loss, result.total_loss), (1.5, 3, 4.5)
        )
        self.assertAlmostEqual(result.pir, 6 / 7)
        self.assertAlmostEqual(pir_identity(result), result.pir)
        self.assertAlmostEqual(
            1.0 - result.information_loss / (result.ideal_value - result.minimum_value), 6 / 7
        )

        with self.assertRaises(EstimandMismatch):
            policy_loss(
                actual_optimum,
                Metric(4.0, replace(actual.context, utility="different utility"), "bad-reference"),
            )

    def test_finite_policy_pir_independently_enumerates_each_policy_image(self):
        ideal, actual = finite_regions()
        direction = (0.5, 1.0)
        minimum = Metric(-2.0, actual.context, "minimum-reference-value")
        result = finite_policy_pir(
            ideal,
            actual,
            direction,
            deployed_policy_id="actual-x",
            minimum=minimum,
        )

        expected_ideal, expected_ideal_policy = direct_support(ideal.policy_values, direction)
        expected_actual, expected_actual_policy = direct_support(actual.policy_values, direction)
        self.assertEqual((expected_ideal, expected_ideal_policy), (8.5, "ideal-b"))
        self.assertEqual((expected_actual, expected_actual_policy), (7.0, "actual-y"))
        self.assertEqual(result.references["ideal_policy"], expected_ideal_policy)
        self.assertEqual(result.references["actual_optimal_policy"], expected_actual_policy)
        self.assertEqual(result.actual_policy_value, 4.0)
        self.assertEqual(result.total_loss, 4.5)
        self.assertAlmostEqual(result.pir, 6 / 7)

        causal_result = finite_causal_pir(
            ideal,
            actual,
            direction,
            deployed_policy_id="actual-x",
            minimum=minimum,
        )
        self.assertEqual(
            causal_result.metric_context.causal_status, CausalStatus.MODEL_INTERVENTIONAL
        )
        self.assertIn(
            "finite supplied interventional policy images only", causal_result.limitations
        )

    def test_preservation_region_uses_independent_support_gap_diagnostic(self):
        ideal, actual = finite_regions()
        directions = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 1.0))
        region = capability_preservation_region(ideal, actual, directions)
        self.assertIsInstance(region, CapabilityPreservationRegion)

        expected = []
        for direction in directions:
            full_value, full_policy = direct_support(dict(ideal.policy_values), direction)
            reduced_value, reduced_policy = direct_support(dict(actual.policy_values), direction)
            expected.append((full_value, reduced_value, full_policy, reduced_policy))
        observed = tuple(
            (item.full_support, item.reduced_support, item.full_policy, item.reduced_policy)
            for item in region.diagnostics
        )
        self.assertEqual(observed, tuple(expected))
        self.assertEqual(finite_support_gap(ideal, actual, directions[0]), region.diagnostics[0])
        self.assertEqual(CapabilityPreservationRegion.from_json(region.to_json()), region)
        self.assertEqual(region.preserved_directions, ((-1.0, 1.0),))
        self.assertEqual(region.maximum_absolute_gap, 2.0)
        self.assertFalse(region.preserves_sampled_directions)
        self.assertTrue(region.contains((-1.0, 1.0)))
        self.assertFalse(region.contains((1.0, 0.0)))

        tolerant = capability_preservation_region(ideal, actual, directions, tolerance=1.0)
        self.assertEqual(len(tolerant.preserved_directions), 2)

    def test_robust_and_optimistic_values_keep_their_optimizer_order(self):
        policy_kernel_values = {"common-safe": (4.0, 0.0), "switching": (0.0, 3.0)}
        direct_robust = max(min(values) for values in policy_kernel_values.values())
        direct_optimistic = min(max(values) for values in zip(*policy_kernel_values.values()))
        self.assertEqual((direct_robust, direct_optimistic), (0.0, 3.0))
        self.assertEqual(robust_value(policy_kernel_values), direct_robust)
        self.assertEqual(optimistic_value(policy_kernel_values), direct_optimistic)


if __name__ == "__main__":
    unittest.main()

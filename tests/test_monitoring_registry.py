"""Scientific verification for event-time monitoring and model governance.

All observations and model descriptors in this file are deterministic synthetic
fixtures.  Passing these tests verifies software contracts, not physiological
or clinical validity.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from physioworld.core import (
    CausalStatus,
    ContextOfUse,
    DeprecationPolicy,
    DeprecationStatus,
    EvidenceClass,
    MetricContext,
    ModelDescriptor,
    ModelFamily,
    ModelGovernanceError,
    ModelRegistry,
    ModelType,
    OperatingEnvelope,
    SemanticVersion,
    VersionedRecord,
)
from physioworld.monitoring import (
    AdaptationMode,
    AdaptationPolicy,
    CapabilityValuePoint,
    ChronologyError,
    DriftMonitorConfig,
    EventObservation,
    FutureDataError,
    TaskValuePoint,
    WindowedDriftMonitor,
    compile_adaptation_split,
    eligible_fit_events,
    fit_pod,
    ks_statistic,
    scan_performance_drift,
    track_capability_value,
    track_task_value,
)


def metric_context(**changes: object) -> MetricContext:
    base = MetricContext(
        estimand_name="forecast_mae",
        data_domain="synthetic-monitoring",
        evaluation_regime="prospective",
        conditioning=("event_time",),
        policy_class="frozen_forecaster",
        reference_state="training-v1",
        time_horizon=10.0,
        task_family=("forecast",),
        resource_state="unconstrained",
        uncertainty_treatment="none_declared",
        metric_type="task_value",
        utility="negative_absolute_error",
        population="synthetic",
        unit="utility",
        causal_status=CausalStatus.PREDICTIVE,
        evidence_class=EvidenceClass.EMPIRICAL_ESTIMATE,
        estimator="mean_task_value",
    )
    return replace(base, **changes)


def event(
    record_id: str,
    time: float,
    value: float,
    *,
    label: float | None = None,
    label_available: float | None = None,
    input_available: float | None = None,
) -> EventObservation:
    return EventObservation(
        record_id,
        f"subject-{record_id}",
        time,
        {"signal": value},
        prediction=value,
        label=label,
        input_available_s=input_available,
        label_available_s=label_available,
        prediction_time_s=time,
        clock_basis="study-clock",
    )


class DriftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = DriftMonitorConfig(
            training_cutoff_s=5.0,
            clock_basis="study-clock",
            window_size_s=5.0,
            step_s=5.0,
            alpha=0.05,
            minimum_reference_n=2,
            minimum_current_n=2,
            performance_threshold=0.2,
            evaluation_as_of_s=30.0,
        )
        self.reference = tuple(event(f"r{i}", i * 0.2, float(i % 2)) for i in range(20))

    def test_ks_ties_are_evaluated_at_unique_values(self) -> None:
        self.assertEqual(ks_statistic((0, 0, 1, 1), (0, 1, 1, 1)), 0.25)

    def test_identical_distribution_is_a_false_drift_control(self) -> None:
        monitor = WindowedDriftMonitor(self.reference[:5], self.config)
        current = tuple(event(f"c{i}", 10.0 + i, float(i % 2)) for i in range(20))
        diagnostics = monitor.scan_covariate(current, "signal")
        self.assertTrue(diagnostics)
        self.assertTrue(
            all(not result.drift_detected for result in diagnostics if result.evaluated)
        )
        self.assertEqual(diagnostics, monitor.scan_covariate(tuple(reversed(current)), "signal"))

    def test_distribution_shift_is_distinct_from_concept_and_performance(self) -> None:
        monitor = WindowedDriftMonitor(self.reference, self.config)
        shifted = tuple(event(f"c{i}", 10.0 + i, 10.0) for i in range(20))
        covariate = monitor.scan_covariate(shifted, "signal")
        self.assertTrue(any(result.drift_detected for result in covariate))

        reference_labels = tuple(
            event(f"lr{i}", float(i), float(i % 2), label=float(i % 2), label_available=5.0)
            for i in range(5)
        )
        current_labels = tuple(
            event(f"lc{i}", 10.0 + i, 0.1, label=float((i + 1) % 2), label_available=20.0)
            for i in range(5)
        )
        concept_monitor = WindowedDriftMonitor(reference_labels, self.config)
        concept = concept_monitor.scan_concept(current_labels, "signal", (0.0, 0.5, 1.0))
        self.assertTrue(concept.drift_detected)
        self.assertIn("not a causal", concept.interpretation)

    def test_no_future_reference_or_current_events_and_availability_are_allowed(self) -> None:
        with self.assertRaises(FutureDataError):
            WindowedDriftMonitor((event("future", 6.0, 0.0),), self.config)
        monitor = WindowedDriftMonitor(self.reference[:2], self.config)
        with self.assertRaises(FutureDataError):
            monitor.scan_covariate((event("past", 5.0, 0.0),), "signal")
        with self.assertRaises(FutureDataError):
            monitor.scan_covariate((event("late", 10.0, 0.0, input_available=11.0),), "signal")

    def test_task_values_are_chronological_and_firewall_is_exact(self) -> None:
        context = metric_context()
        ordered = tuple(
            TaskValuePoint(float(i), 1.0, context, f"v{i}", available_s=float(i)) for i in range(4)
        )
        self.assertEqual(track_task_value(ordered, context), ordered)
        with self.assertRaises(ChronologyError):
            track_task_value((ordered[1], ordered[0]), context)
        with self.assertRaises(ValueError):
            track_task_value(
                (ordered[0], replace(ordered[1], context=replace(context, utility="other"))),
                context,
            )
        current = tuple(
            TaskValuePoint(10.0 + i, 0.0, context, f"c{i}", available_s=20.0) for i in range(4)
        )
        reports = scan_performance_drift(ordered, current, context, self.config, as_of_s=30.0)
        self.assertTrue(any(report.drift_detected for report in reports))
        with self.assertRaises(FutureDataError):
            scan_performance_drift(ordered, current, context, self.config, as_of_s=15.0)

    def test_capability_series_uses_its_own_metric_context(self) -> None:
        context = metric_context(metric_type="capability", estimand_name="support_function")
        points = tuple(CapabilityValuePoint(float(i), 1.0, context, f"cap-{i}") for i in range(3))
        self.assertEqual(track_capability_value(points, context), points)


class AdaptationTests(unittest.TestCase):
    def test_split_compiler_semantics_guard_fit_and_future_labels(self) -> None:
        policy = AdaptationPolicy("policy-v1", 5.0, "study-clock", AdaptationMode.DRIFT_TRIGGERED)
        events = tuple(
            event(f"e{i}", float(i), float(i), label=float(i), label_available=float(i))
            for i in (0, 1, 10)
        )
        compiled = compile_adaptation_split(
            (*events[:2], replace(events[2], label=None, label_available_s=None)), policy
        )
        self.assertEqual(compiled.assignments, {"e0": "train", "e1": "train", "e10": "test"})
        fit = eligible_fit_events(events, policy)
        self.assertEqual(tuple(item.record_id for item in fit), ("e0", "e1"))
        late_label = replace(events[1], label_available_s=6.0)
        with self.assertRaises(ValueError):
            eligible_fit_events((events[0], late_label, events[2]), policy)
        with self.assertRaises(FutureDataError):
            policy.validate_schedule((5.0,))

    def test_periodic_schedule_is_explicit(self) -> None:
        policy = AdaptationPolicy("periodic-v1", 10.0, "study-clock", AdaptationMode.PERIODIC, 5.0)
        self.assertEqual(policy.validate_schedule((15.0, 20.0)), (15.0, 20.0))
        with self.assertRaises(ValueError):
            policy.validate_schedule((16.0,))


class RegistryTests(unittest.TestCase):
    def descriptor(
        self, version: str, *, output: str = "float", algorithm: str = "linear-v1"
    ) -> ModelDescriptor:
        return ModelDescriptor(
            model_id="heart-rate-forecaster",
            family=ModelFamily.STATISTICAL,
            model_type=ModelType.FORECASTER,
            version=SemanticVersion.parse(version),
            equations_or_algorithm=algorithm,
            assumptions=("stationary within an evaluation window",),
            parameters={"regularization": 0.1},
            input_schema={"heart_rate": "float[bpm]"},
            output_schema={"forecast": output},
            operating_envelope=OperatingEnvelope(
                "hr-envelope", "synthetic fixture", {"sampling_rate_hz": (1.0, 10.0)}
            ),
            context_of_use=ContextOfUse(
                "forecast-use", "research forecast", "synthetic", "offline", ("forecast",)
            ),
            evidence_class=EvidenceClass.NUMERICAL_VERIFICATION,
            evidence_refs=("tests/test_monitoring_registry.py",),
            affected_claims=("claim-forecast",),
            affected_tests=("test_registry_versioning",),
            affected_benchmarks=("benchmark-forecast-v1",),
        )

    def test_descriptor_hash_envelope_and_roundtrip_are_deterministic(self) -> None:
        descriptor = self.descriptor("1.0.0")
        self.assertEqual(descriptor.content_hash(), self.descriptor("1.0.0").content_hash())
        self.assertTrue(descriptor.operating_envelope.contains({"sampling_rate_hz": 5.0}))
        self.assertEqual(ModelDescriptor.from_json(descriptor.to_json()), descriptor)
        self.assertIsInstance(descriptor, VersionedRecord)

    def test_registry_semver_compatibility_and_impact_graph(self) -> None:
        registry = ModelRegistry()
        first = registry.register(self.descriptor("1.0.0"))
        self.assertFalse(first.results_invalidated)
        second = registry.register(self.descriptor("1.1.0", algorithm="linear-v2"))
        self.assertTrue(second.api_compatible)
        self.assertTrue(second.results_invalidated)
        self.assertEqual(second.affected_claims, ("claim-forecast",))
        self.assertIn(
            "claim:claim-forecast", second.evidence_graph.affected_by(second.current_hash)
        )
        self.assertEqual(
            registry.latest("heart-rate-forecaster").descriptor.version,
            SemanticVersion.parse("1.1.0"),
        )
        self.assertEqual(ModelRegistry.from_json(registry.to_json()), registry)
        with self.assertRaises(ModelGovernanceError):
            registry.register(self.descriptor("1.2.0", output="probability"))
        major = registry.register(self.descriptor("2.0.0", output="probability"))
        self.assertFalse(major.api_compatible)
        retired = self.descriptor("3.0.0")
        retired = replace(
            retired,
            deprecation=DeprecationPolicy(
                DeprecationStatus.RETIRED, "superseded by a validated release"
            ),
        )
        registry.register(retired)
        with self.assertRaises(ModelGovernanceError):
            registry.require_active("heart-rate-forecaster")

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional scientific NumPy dependency")
    def test_pod_is_real_projection_with_declared_reference_error(self) -> None:
        model = fit_pod(
            ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            1,
            reference_id="snapshots-v1",
            error_tolerance=1.0,
        )
        coordinates = model.project((1.0, 0.0))
        reconstruction = model.reconstruct(coordinates)
        self.assertEqual(len(coordinates), 1)
        self.assertEqual(len(reconstruction), 2)
        self.assertTrue(model.criterion_satisfied)
        self.assertEqual(type(model).from_json(model.to_json()), model)


class ExampleTests(unittest.TestCase):
    def test_actual_monitoring_example_runs(self) -> None:
        example = Path(__file__).parents[1] / "examples" / "drift_monitoring.py"
        result = subprocess.run(
            [sys.executable, str(example)], check=True, capture_output=True, text=True
        )
        payload = json.loads(result.stdout)
        self.assertIn("covariate", payload)
        self.assertIn("performance", payload)
        self.assertGreater(payload["covariate"]["windows"], 0)


if __name__ == "__main__":
    unittest.main()

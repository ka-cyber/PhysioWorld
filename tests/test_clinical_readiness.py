import unittest

from physioworld.clinical import (
    AlarmWorkload,
    BinaryUtility,
    ClinicalValidationProtocol,
    CohortOrigin,
    EndpointObservation,
    EvaluationCohort,
    FrozenArtifacts,
    LockedClinicalProtocol,
    protocol_from_config,
    validation_readiness,
)
from physioworld.clinical.evaluation import evaluate_independent_cohort


def explicit_protocol_config() -> dict[str, object]:
    return {
        "intended_use": "prospective arrhythmia alert research evaluation",
        "population": "adults with a prespecified ambulatory ECG indication",
        "data_provenance": "independent registry release 2026-01",
        "inclusion_exclusion": "include eligible adults; exclude records without follow-up",
        "reference_standard": "blinded adjudication by two electrophysiologists",
        "endpoint": "incident arrhythmia within the ten-second horizon",
        "analysis_population": "all enrolled eligible subjects with complete follow-up",
        "split": "subject-level held-out test cohort",
        "missing_data": "primary analysis excludes missing endpoint; sensitivity analysis is prespecified",
        "model": {"name": "frozen-alert-model", "version": "2026.01"},
        "threshold": 0.5,
        "resource": {"device": "edge-v1", "power_mode": "nominal"},
        "channel": {"sampling": "250 Hz", "transport": "BLE"},
        "dataset_version": "registry-2026-01",
    }


def external_evaluation_fixture() -> tuple[
    LockedClinicalProtocol, EvaluationCohort, tuple[EndpointObservation, ...]
]:
    artifacts = FrozenArtifacts(*(["a" * 64] * 5))
    workload = AlarmWorkload(
        "synthetic-alarm",
        "researcher",
        "synthetic future event",
        10.0,
        0.5,
        BinaryUtility(4.0, 0.0, -2.0, -8.0, 0.5),
        "synthetic fixed resources",
        ("no intervention recommendations",),
        3.0,
    )
    protocol = LockedClinicalProtocol(
        "synthetic-protocol",
        workload,
        artifacts,
        "synthetic population",
        "all synthetic subjects",
        "known synthetic labels",
        "complete cases declared",
        ("fit",),
        ("development",),
        100.0,
        90.0,
        prospective=True,
        bootstrap_repetitions=200,
    )
    cohort = EvaluationCohort(
        "independent-human-cohort",
        CohortOrigin.HUMAN_OBSERVATIONAL,
        "independent registry",
        "registry-2026-01",
        ("registry-provenance",),
    )
    rows = tuple(
        EndpointObservation(
            f"eval-{i}",
            y,
            p,
            200.0,
            199.0,
            210.0,
            200.0,
            3800.0,
            150.0,
            artifacts.model_hash,
        )
        for i, (y, p) in enumerate(((1, 0.9), (1, 0.1), (0, 0.8), (0, 0.2)))
    )
    return protocol, cohort, rows


class ClinicalProtocolReadinessTests(unittest.TestCase):
    def test_protocol_requires_every_clinical_field(self):
        config = explicit_protocol_config()
        del config["reference_standard"]
        with self.assertRaisesRegex(ValueError, "reference_standard"):
            protocol_from_config(config)

    def test_protocol_rejects_placeholder_defaults_and_empty_regimes(self):
        for field, value in (
            ("population", "declared dataset population"),
            ("data_provenance", "unspecified"),
            ("resource", {}),
            ("channel", {"mode": "TBD"}),
        ):
            config = explicit_protocol_config()
            config[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                protocol_from_config(config)

    def test_readiness_labels_do_not_conflate_simulation_and_validation(self):
        simulation = validation_readiness("synthetic")
        external = validation_readiness("dataset")
        self.assertEqual(simulation["status"], "READY_FOR_SIMULATION")
        self.assertEqual(simulation["evaluation_scope"], "SIMULATION")
        self.assertEqual(external["evaluation_scope"], "EXTERNAL_PREDICTIVE_EVALUATION")
        self.assertEqual(external["clinical_validation_status"], "NOT_ESTABLISHED")

    def test_direct_protocol_validation_rejects_placeholder(self):
        protocol = ClinicalValidationProtocol(
            "research evaluation",
            "declared dataset population",
            "registry release",
            "eligible subjects",
            "expert adjudication",
            "arrhythmia",
            "eligible subjects",
            "subject test",
            "complete cases",
            "frozen model",
            "0.5",
            "edge-v1",
            "BLE",
            "2026-01",
        )
        with self.assertRaisesRegex(ValueError, "intended_use"):
            protocol.validate()

    def test_external_cohort_is_predictive_evaluation_without_clinical_validation(self):
        protocol, cohort, rows = external_evaluation_fixture()
        result = evaluate_independent_cohort(
            protocol,
            cohort,
            rows,
            expected_protocol_hash=protocol.frozen_hash,
            applied_artifacts=protocol.artifacts,
        )
        self.assertEqual(result.status, "EXTERNAL_PREDICTIVE_EVALUATION")
        self.assertEqual(result.evaluation_scope, "EXTERNAL_PREDICTIVE_EVALUATION")
        self.assertEqual(result.external_predictive_evaluation_status, "COMPLETED")
        self.assertEqual(result.clinical_validation_status, "NOT_ESTABLISHED")
        self.assertIn("clinical validation is not established", " ".join(result.limitations))


if __name__ == "__main__":
    unittest.main()

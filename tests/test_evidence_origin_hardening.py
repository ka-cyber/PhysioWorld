"""Regression tests for evidence origin and estimator evidence propagation."""

import copy
import unittest

from physioworld.core import EvidenceClass
from physioworld.evidence_origin import has_verified_artifact_metadata
from physioworld.experiments import ExperimentCompiler, ExperimentValidationError
from physioworld.representations import EmpiricalTaskLawRepresentation
from physioworld.tasklaws import TaskDefinition, TaskKind, conditional_task_law
from physioworld.verticals import default_configuration, run_vertical


def replay_configuration(*, evidence_origin=None, artifact_metadata=None):
    config = copy.deepcopy(default_configuration("cardiovascular", seed=19))
    config["name"] = "evidence_origin_replay"
    config["dataset"] = {
        "id": "three_atom_replay",
        "origin": "REPLAY",
        "source": "inline analytic fixture",
        "modality": "telemetry",
        "channel": "mean_pressure_mmhg",
        "unit": "mmHg",
        "time_unit": "s",
        "samples": [90.0, 85.0, 75.0],
        "timestamps_s": [0.0, 1.0, 2.0],
    }
    if evidence_origin is not None:
        config["dataset"]["evidence_origin"] = evidence_origin
    if artifact_metadata is not None:
        config["dataset"]["artifact_metadata"] = artifact_metadata
    config["task"].update(horizon_s=0.0, threshold=80.0)
    config["policy"].update(thresholds=[80.0])
    config["perturbations"] = []
    config["split"] = {"strategy": "replay_fixed"}
    config["channel"].update(
        loss_probability=0.0,
        latency_s=0.0,
        jitter_s=0.0,
        deadline_s=1.0,
        packet_payload_bytes=64,
    )
    config["resource"].update(
        model="capacitor_p0",
        capacitance_f=1.0,
        esr_ohm=0.0,
        leakage_siemens=0.0,
        initial_voltage_v=3.0,
        v_min=1.0,
        max_current_a=1.0,
        load_current_a=0.001,
        step_duration_s=0.001,
    )
    config["compute"].update(
        operations_per_sample=100.0,
        operations_per_second=1e6,
        energy_per_operation_j=0.0,
        network_latency_s=0.0,
    )
    return config


ARTIFACT = {
    "artifact_id": "observed-replay-001",
    "sha256": "0" * 64,
    "verified": True,
}


class EvidenceOriginHardeningTests(unittest.TestCase):
    def test_synthetic_vertical_and_task_laws_keep_simulation_evidence(self):
        result = run_vertical("cardiovascular", seed=4).to_dict()
        self.assertEqual(result["evidence_class"], EvidenceClass.SIMULATION.value)
        self.assertEqual(
            result["outputs"]["semantics"]["task_law"]["evidence_class"],
            EvidenceClass.SIMULATION.value,
        )
        self.assertEqual(
            result["outputs"]["semantics"]["task_law"]["evidence_metadata"]["evidence_origin"],
            "SYNTHETIC_MODEL",
        )

    def test_inline_analytic_replay_is_numerical_verification(self):
        result = run_vertical(replay_configuration()).to_dict()
        self.assertEqual(result["evidence_class"], EvidenceClass.NUMERICAL_VERIFICATION.value)
        self.assertEqual(
            result["provenance"]["source"]["evidence_origin"], "INLINE_ANALYTIC_REPLAY"
        )
        self.assertEqual(
            result["outputs"]["semantics"]["metric_context"]["evidence_class"],
            EvidenceClass.NUMERICAL_VERIFICATION.value,
        )
        self.assertNotEqual(
            result["outputs"]["semantics"]["task_law"]["evidence_class"],
            EvidenceClass.EMPIRICAL_ESTIMATE.value,
        )

    def test_observed_replay_boundary_is_explicit_and_artifact_backed(self):
        compiler = ExperimentCompiler()
        ambiguous = replay_configuration()
        ambiguous["dataset"]["source"] = "recorded cohort replay"
        with self.assertRaisesRegex(
            ExperimentValidationError, "explicitly declare evidence_origin"
        ):
            compiler.compile(ambiguous)

        observed = replay_configuration(evidence_origin="OBSERVED_REPLAY")
        with self.assertRaisesRegex(ExperimentValidationError, "verified artifact metadata"):
            compiler.compile(observed)

        observed["dataset"]["artifact_metadata"] = ARTIFACT
        self.assertTrue(has_verified_artifact_metadata(observed["dataset"]))
        result = run_vertical(observed).to_dict()
        self.assertEqual(result["evidence_class"], EvidenceClass.EMPIRICAL_ESTIMATE.value)
        self.assertEqual(result["provenance"]["source"]["artifact_metadata"], ARTIFACT)

    def test_external_and_human_origins_require_verified_artifact_metadata(self):
        compiler = ExperimentCompiler()
        for origin in ("EXTERNAL_DATASET", "HUMAN_OBSERVATION"):
            with self.subTest(origin=origin):
                config = replay_configuration(evidence_origin=origin)
                with self.assertRaisesRegex(
                    ExperimentValidationError, "verified artifact metadata"
                ):
                    compiler.compile(config)

    def test_task_law_estimators_do_not_default_to_empirical(self):
        task = TaskDefinition("synthetic_task", TaskKind.PREDICTION, 0.0, "Y", "squared_error")
        law = conditional_task_law((0.0, 1.0), ("a", "b"), task)
        self.assertEqual(law.evidence_class, EvidenceClass.NUMERICAL_VERIFICATION)
        self.assertEqual(
            EmpiricalTaskLawRepresentation("synthetic_task")
            .fit(("a", "b"), (0.0, 1.0))
            .transform("a")
            .metadata["evidence_class"],
            EvidenceClass.NUMERICAL_VERIFICATION.value,
        )

    def test_external_task_law_estimator_requires_and_propagates_artifact_metadata(self):
        metadata = {"artifact_metadata": ARTIFACT}
        model = EmpiricalTaskLawRepresentation(
            "observed_task",
            evidence_class=EvidenceClass.EXTERNAL_VALIDATION,
            evidence_metadata=metadata,
        ).fit(("a", "b"), (0.0, 1.0))
        self.assertEqual(
            model.transform("a").metadata["evidence_class"],
            EvidenceClass.EXTERNAL_VALIDATION.value,
        )
        self.assertEqual(model.evaluate(("a",), (0.0,))["evidence_metadata"], metadata)
        with self.assertRaisesRegex(ValueError, "verified artifact metadata"):
            EmpiricalTaskLawRepresentation(
                "unverified_task", evidence_class=EvidenceClass.CLINICAL_EVIDENCE
            )


if __name__ == "__main__":
    unittest.main()

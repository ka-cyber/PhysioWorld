"""Focused tests for source-frozen synthetic cross-regime transfer."""

import unittest

from physioworld.core import EvidenceClass, Task, TaskKind
from physioworld.experiments.transfer import (
    Regime,
    RegimeDataset,
    RegimeSequence,
    TransferPartition,
    TransferProtocol,
    evaluate_transfer,
    freeze_transfer,
    train_transfer,
)

SOURCE = Regime("source", "low-noise synthetic regime")
TARGET = Regime("target", "shifted synthetic regime")
TASK = Task(
    "event_forecast",
    TaskKind.CLASSIFICATION,
    1.0,
    "binary event at horizon",
    "brier",
    threshold=0.5,
    resource_regime="simulation_cpu",
    reference_condition="synthetic baseline",
)


def sequence(sequence_id, level, outcome):
    """Deterministic history with a future label kept outside the input prefix."""
    return RegimeSequence(
        sequence_id,
        (level, level + 0.01, level + 0.02, level + 0.03, level + 9.0),
        (0.0, 1.0, 2.0, 3.0, 4.0),
        3.0,
        outcome,
    )


def dataset(regime, partition, rows):
    return RegimeDataset(regime, partition, tuple(rows))


class CrossRegimeTransferTests(unittest.TestCase):
    def setUp(self):
        self.protocol = TransferProtocol(
            SOURCE,
            TARGET,
            TASK,
            history_window=4,
            pca_components=2,
            kernel_bandwidth=0.25,
            decision_threshold=0.5,
            calibration_bins=4,
        )
        self.source_train = dataset(
            SOURCE,
            TransferPartition.SOURCE_TRAIN,
            (
                sequence("train-0", 0.0, 0),
                sequence("train-1", 0.1, 0),
                sequence("train-2", 1.0, 1),
                sequence("train-3", 1.1, 1),
            ),
        )
        self.source_test = dataset(
            SOURCE,
            TransferPartition.SOURCE_TEST,
            (sequence("source-test-0", 0.05, 0), sequence("source-test-1", 1.05, 1)),
        )
        self.target_test = dataset(
            TARGET,
            TransferPartition.TARGET_TEST,
            (sequence("target-test-0", 0.05, 1), sequence("target-test-1", 1.05, 0)),
        )

    def test_train_freeze_evaluate_is_source_only_and_typed(self):
        trained = train_transfer(self.protocol, self.source_train)
        frozen = freeze_transfer(trained)
        result = evaluate_transfer(frozen, self.source_test, self.target_test)

        self.assertEqual(frozen.hashes.source_training, self.source_train.content_hash)
        self.assertEqual(len(frozen.hashes.frozen_artifact), 64)
        self.assertNotEqual(frozen.hashes.frozen_artifact, frozen.hashes.source_training)
        self.assertEqual(result.hashes, frozen.hashes)
        self.assertEqual(result.source.partition, TransferPartition.SOURCE_TEST)
        self.assertEqual(result.target.partition, TransferPartition.TARGET_TEST)
        self.assertEqual(result.source.sequence_ids, ("source-test-0", "source-test-1"))
        self.assertEqual(result.target.sequence_ids, ("target-test-0", "target-test-1"))
        self.assertEqual(result.evidence_class, EvidenceClass.SIMULATION)
        self.assertEqual(result.causal_status.value, "predictive")
        self.assertFalse(result.causal_identification)
        self.assertIn("does not identify causal effects", result.causal_scope)
        self.assertNotEqual(result.source.brier_score, result.target.brier_score)
        self.assertEqual(type(result).from_json(result.to_json()), result)

    def test_available_prefix_excludes_future_synthetic_sample(self):
        trained = train_transfer(self.protocol, self.source_train)
        encoded = trained.representation.transform_at(
            self.source_train.sequences[0].history,
            self.source_train.sequences[0].timestamps,
            self.source_train.sequences[0].prediction_time,
        )
        self.assertEqual(encoded.metadata["available_samples"], 4)
        self.assertEqual(encoded.metadata["excluded_future_samples"], 1)

    def test_test_sets_cannot_change_frozen_artifact_or_tune_protocol(self):
        frozen = freeze_transfer(train_transfer(self.protocol, self.source_train))
        original_hashes = frozen.hashes
        altered_target = dataset(
            TARGET,
            TransferPartition.TARGET_TEST,
            (sequence("new-target-0", 0.05, 0), sequence("new-target-1", 1.05, 1)),
        )
        result = evaluate_transfer(frozen, self.source_test, altered_target)

        self.assertEqual(frozen.hashes, original_hashes)
        self.assertEqual(result.hashes, original_hashes)
        self.assertEqual(frozen.trained.protocol, self.protocol)
        with self.assertRaises(AttributeError):
            frozen.trained.protocol = self.protocol

    def test_boundaries_reject_wrong_partitions_and_cross_partition_reuse(self):
        with self.assertRaises(ValueError):
            train_transfer(self.protocol, self.source_test)
        trained = train_transfer(self.protocol, self.source_train)
        with self.assertRaises(ValueError):
            evaluate_transfer(
                freeze_transfer(trained),
                self.source_train,
                self.target_test,
            )
        with self.assertRaises(ValueError):
            evaluate_transfer(
                freeze_transfer(trained),
                self.source_test,
                dataset(
                    TARGET,
                    TransferPartition.TARGET_TEST,
                    (sequence("source-test-0", 0.05, 0),),
                ),
            )


if __name__ == "__main__":
    unittest.main()

"""Frozen, source-only training and held-out predictive transport in simulation.

The boundaries are ``train_transfer(source_train) -> freeze_transfer(trained)
-> evaluate_transfer(frozen, source_test, target_test)``. Neither training nor
freezing accepts a test set. History length, PCA rank, kernel bandwidth, decision
threshold and calibration bins are declared in the protocol before training.
Evaluation has no fitting, recalibration or selection step.

Each sequence is an independent synthetic unit. Source/target score differences
describe predictive transport across the supplied regimes, not treatment effects,
causal identification, or population-level transport guarantees.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Sequence

from physioworld.core.contracts import (
    CausalStatus,
    EvidenceClass,
    MetricContext,
    Task,
    TaskKind,
    VersionedRecord,
    finite,
)
from physioworld.representations import (
    BaseRepresentation,
    FiniteHistoryRepresentation,
    PCARepresentation,
    RepresentationResult,
    RepresentationSpec,
)
from physioworld.tasklaws import expected_calibration_error, kernel_conditional_distribution


def _hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    for value in values:
        finite(value, name)
    return tuple(float(value) for value in values)


class TransferPartition(str, Enum):
    SOURCE_TRAIN = "source_train"
    SOURCE_TEST = "source_test"
    TARGET_TEST = "target_test"


@dataclass(frozen=True)
class Regime(VersionedRecord):
    regime_id: str
    description: str

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) and v.strip() for v in (self.regime_id, self.description)):
            raise ValueError("regimes require an ID and a description")


@dataclass(frozen=True)
class RegimeSequence(VersionedRecord):
    """One independent synthetic sequence with an explicitly available prefix.

    Times share the protocol's time unit. The binary outcome refers to the task's
    declared horizon; it is never an input feature. Samples after prediction_time
    may be retained for audit, but are excluded by the representation API.
    """

    sequence_id: str
    history: tuple[float, ...]
    timestamps: tuple[float, ...]
    prediction_time: float
    outcome: int

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequences require an independent sequence_id")
        object.__setattr__(self, "history", _vector(self.history, "history"))
        object.__setattr__(self, "timestamps", _vector(self.timestamps, "timestamp"))
        finite(self.prediction_time, "prediction_time")
        object.__setattr__(self, "prediction_time", float(self.prediction_time))
        if not self.history or len(self.history) != len(self.timestamps):
            raise ValueError("history and timestamps must have equal nonzero lengths")
        if any(b <= a for a, b in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError("timestamps must be strictly increasing")
        if self.timestamps[0] > self.prediction_time:
            raise ValueError("sequence has no observations available at prediction_time")
        if type(self.outcome) is not int or self.outcome not in (0, 1):
            raise ValueError("transfer outcomes must be binary integers")


@dataclass(frozen=True)
class RegimeDataset(VersionedRecord):
    regime: Regime
    partition: TransferPartition
    sequences: tuple[RegimeSequence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "partition", TransferPartition(self.partition))
        object.__setattr__(self, "sequences", tuple(self.sequences))
        if not isinstance(self.regime, Regime) or not self.sequences:
            raise ValueError("a regime dataset requires a regime and sequences")
        if any(not isinstance(row, RegimeSequence) for row in self.sequences):
            raise ValueError("datasets require typed RegimeSequence rows")
        if len(set(self.sequence_ids)) != len(self.sequences):
            raise ValueError("sequence IDs must be unique within a partition")

    @property
    def sequence_ids(self) -> tuple[str, ...]:
        return tuple(row.sequence_id for row in self.sequences)

    @property
    def content_hash(self) -> str:
        return _hash(self.to_dict())


@dataclass(frozen=True)
class TransferProtocol(VersionedRecord):
    """Predeclared binary predictive experiment; no test-driven hyperparameters."""

    source_regime: Regime
    target_regime: Regime
    task: Task
    history_window: int = 4
    pca_components: int | None = None
    kernel_bandwidth: float = 1.0
    decision_threshold: float = 0.5
    calibration_bins: int = 10
    evidence_class: EvidenceClass = EvidenceClass.SIMULATION

    def __post_init__(self) -> None:
        if not isinstance(self.source_regime, Regime) or not isinstance(self.target_regime, Regime):
            raise ValueError("source and target must be typed regimes")
        if self.source_regime.regime_id == self.target_regime.regime_id:
            raise ValueError("cross-regime transfer requires distinct regime IDs")
        if not isinstance(self.task, Task):
            raise ValueError("transfer requires a typed Task")
        self.task.validate()
        if self.task.kind not in {
            TaskKind.PREDICTION,
            TaskKind.CLASSIFICATION,
            TaskKind.FORECASTING,
        }:
            raise ValueError("predictive transport cannot evaluate causal or decision tasks")
        if self.task.action_semantics is not None or self.task.intervention_semantics is not None:
            raise ValueError("predictive transport cannot carry action/intervention semantics")
        if self.task.loss_or_utility != "brier" or self.task.aggregation_rule != "mean":
            raise ValueError("transfer uses the mean binary Brier loss")
        _positive_int(self.history_window, "history_window")
        _positive_int(self.calibration_bins, "calibration_bins")
        if self.pca_components is not None:
            _positive_int(self.pca_components, "pca_components")
            if self.pca_components > self.history_window:
                raise ValueError("PCA components cannot exceed history_window")
        finite(self.kernel_bandwidth, "kernel_bandwidth")
        finite(self.decision_threshold, "decision_threshold")
        if self.kernel_bandwidth <= 0 or not 0 < self.decision_threshold < 1:
            raise ValueError("bandwidth must be positive and decision_threshold must lie in (0, 1)")
        if self.task.threshold is not None and self.task.threshold != self.decision_threshold:
            raise ValueError("task and protocol thresholds must match")
        object.__setattr__(self, "kernel_bandwidth", float(self.kernel_bandwidth))
        object.__setattr__(self, "decision_threshold", float(self.decision_threshold))
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        if self.evidence_class != EvidenceClass.SIMULATION:
            raise ValueError("synthetic transfer evidence must be SIMULATION")


@dataclass(frozen=True)
class FrozenPCAState(VersionedRecord):
    """Immutable source-fitted PCA parameters, without a mutable model reference."""

    mean: tuple[float, ...]
    components: tuple[tuple[float, ...], ...]
    explained_variance: tuple[float, ...]
    rank: int
    fit_samples: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _vector(self.mean, "PCA mean"))
        object.__setattr__(
            self, "components", tuple(_vector(row, "PCA component") for row in self.components)
        )
        object.__setattr__(
            self, "explained_variance", _vector(self.explained_variance, "PCA variance")
        )
        _positive_int(self.fit_samples, "fit_samples")
        if (
            self.fit_samples < 2
            or not self.mean
            or not self.components
            or len(self.components) > len(self.mean)
            or any(len(row) != len(self.mean) for row in self.components)
            or len(self.explained_variance) != len(self.components)
            or any(value < 0 for value in self.explained_variance)
            or type(self.rank) is not int
            or not 0 <= self.rank <= len(self.mean)
        ):
            raise ValueError("invalid frozen PCA state")

    def restore(self) -> PCARepresentation:
        """Hydrate an ephemeral transform-only model using existing PCA semantics.

        No fit is performed. Callers can mutate this local model without changing
        the snapshot used for either regime's evaluation.
        """
        model = PCARepresentation(len(self.components))
        model.mean = self.mean
        model.components_ = self.components
        model.explained_variance_ = self.explained_variance
        model.rank_ = self.rank
        model.n_components_ = len(self.components)
        model.n_samples_ = self.fit_samples
        model.spec = replace(model.spec, dimension=len(self.components))
        return model


@dataclass(frozen=True)
class FrozenHistoryRepresentation(BaseRepresentation[Sequence[float]], VersionedRecord):
    history_window: int
    pca: FrozenPCAState | None = None
    spec: RepresentationSpec = field(init=False)

    def __post_init__(self) -> None:
        _positive_int(self.history_window, "history_window")
        if self.pca is not None and (
            not isinstance(self.pca, FrozenPCAState) or len(self.pca.mean) != self.history_window
        ):
            raise ValueError("PCA state must match the history window")
        object.__setattr__(
            self,
            "spec",
            self.pca.restore().spec
            if self.pca is not None
            else FiniteHistoryRepresentation(self.history_window).spec,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize fitted parameters, excluding the non-record runtime spec."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": type(self).__name__,
            "history_window": self.history_window,
            "pca": self.pca.to_dict() if self.pca is not None else None,
        }

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        result = FiniteHistoryRepresentation(self.history_window).transform(history)
        return self.pca.restore().transform(result.values) if self.pca is not None else result


def _encode(
    representation: FrozenHistoryRepresentation, dataset: RegimeDataset
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        representation.transform_at(row.history, row.timestamps, row.prediction_time).values
        for row in dataset.sequences
    )


@dataclass(frozen=True)
class TrainedTransfer(VersionedRecord):
    protocol: TransferProtocol
    source_training: RegimeDataset
    representation: FrozenHistoryRepresentation
    training_features: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, TransferProtocol):
            raise ValueError("training requires a typed TransferProtocol")
        if not isinstance(self.source_training, RegimeDataset) or (
            self.source_training.partition != TransferPartition.SOURCE_TRAIN
            or self.source_training.regime != self.protocol.source_regime
        ):
            raise ValueError("training accepts only the declared source_train partition")
        if not isinstance(self.representation, FrozenHistoryRepresentation):
            raise ValueError("training requires an immutable representation snapshot")
        if self.representation.history_window != self.protocol.history_window:
            raise ValueError("representation does not match the predeclared window")
        pca = self.representation.pca
        if (None if pca is None else len(pca.components)) != self.protocol.pca_components:
            raise ValueError("representation does not match the predeclared PCA rank")
        if pca is not None and pca.fit_samples != len(self.source_training.sequences):
            raise ValueError("PCA fit sample count must match source training")
        object.__setattr__(
            self,
            "training_features",
            tuple(_vector(row, "training feature") for row in self.training_features),
        )
        if self.training_features != _encode(self.representation, self.source_training):
            raise ValueError("training features must be the source-only available-prefix encoding")

    @property
    def task_law_state(self) -> dict[str, object]:
        return {
            "estimator": "gaussian_kernel_conditional_empirical_law",
            "task": self.protocol.task.to_dict(),
            "bandwidth": self.protocol.kernel_bandwidth,
            "features": self.training_features,
            "outcomes": tuple(row.outcome for row in self.source_training.sequences),
        }


def train_transfer(protocol: TransferProtocol, source_train: RegimeDataset) -> TrainedTransfer:
    """Train on source sequences only; all choices come from the supplied protocol."""
    if not isinstance(protocol, TransferProtocol) or not isinstance(source_train, RegimeDataset):
        raise ValueError("training requires a typed protocol and dataset")
    if (
        source_train.partition != TransferPartition.SOURCE_TRAIN
        or source_train.regime != protocol.source_regime
    ):
        raise ValueError("training accepts only the declared source_train partition")
    representation = FrozenHistoryRepresentation(protocol.history_window)
    features = _encode(representation, source_train)
    if protocol.pca_components is not None:
        model = PCARepresentation(protocol.pca_components).fit(features)
        assert model.mean is not None
        representation = FrozenHistoryRepresentation(
            protocol.history_window,
            FrozenPCAState(
                model.mean,
                model.components_,
                model.explained_variance_,
                model.rank_,
                model.n_samples_,
            ),
        )
        features = _encode(representation, source_train)
    return TrainedTransfer(protocol, source_train, representation, features)


@dataclass(frozen=True)
class TransferHashes(VersionedRecord):
    protocol: str
    source_training: str
    representation: str
    task_law: str
    frozen_artifact: str


def _hashes(trained: TrainedTransfer) -> TransferHashes:
    return TransferHashes(
        _hash(trained.protocol.to_dict()),
        trained.source_training.content_hash,
        _hash(trained.representation.to_dict()),
        _hash(trained.task_law_state),
        _hash(trained.to_dict()),
    )


@dataclass(frozen=True)
class FrozenTransfer(VersionedRecord):
    trained: TrainedTransfer
    hashes: TransferHashes

    def __post_init__(self) -> None:
        if not isinstance(self.trained, TrainedTransfer) or self.hashes != _hashes(self.trained):
            raise ValueError("frozen transfer hashes do not match the source-trained state")


def freeze_transfer(trained: TrainedTransfer) -> FrozenTransfer:
    """Seal the complete source-trained state; this boundary has no test-set input."""
    if not isinstance(trained, TrainedTransfer):
        raise ValueError("freeze requires a TrainedTransfer")
    return FrozenTransfer(trained, _hashes(trained))


@dataclass(frozen=True)
class RegimeEvaluation(VersionedRecord):
    regime: Regime
    partition: TransferPartition
    dataset_hash: str
    sequence_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    outcomes: tuple[int, ...]
    brier_score: float
    accuracy: float
    calibration_error: float
    context: MetricContext


@dataclass(frozen=True)
class TransportDiagnostics(VersionedRecord):
    """Target-minus-source held-out diagnostics; these are not causal contrasts."""

    brier_delta: float
    accuracy_delta: float
    calibration_delta: float
    scope: str = "predictive transport on supplied synthetic sequences"


@dataclass(frozen=True)
class CrossRegimeTransferResult(VersionedRecord):
    task: Task
    hashes: TransferHashes
    source: RegimeEvaluation
    target: RegimeEvaluation
    transport: TransportDiagnostics
    evidence_class: EvidenceClass = EvidenceClass.SIMULATION
    causal_status: CausalStatus = CausalStatus.PREDICTIVE
    causal_identification: bool = False
    causal_scope: str = "Not assessed; predictive transport does not identify causal effects."

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        object.__setattr__(self, "causal_status", CausalStatus(self.causal_status))
        if (
            self.evidence_class != EvidenceClass.SIMULATION
            or self.causal_status != CausalStatus.PREDICTIVE
            or self.causal_identification is not False
        ):
            raise ValueError("synthetic predictive transport cannot be promoted to causal evidence")

    @property
    def content_hash(self) -> str:
        return _hash(self.to_dict())


def _evaluate(frozen: FrozenTransfer, dataset: RegimeDataset) -> RegimeEvaluation:
    trained, protocol = frozen.trained, frozen.trained.protocol
    outcomes = tuple(row.outcome for row in dataset.sequences)
    laws = kernel_conditional_distribution(
        tuple(row.outcome for row in trained.source_training.sequences),
        trained.training_features,
        _encode(trained.representation, dataset),
        bandwidth=protocol.kernel_bandwidth,
    )
    probabilities = tuple(
        math.fsum(p for value, p in zip(law.support, law.probabilities) if value == 1)
        for law in laws
    )
    context = MetricContext(
        estimand_name="held_out_mean_binary_brier_loss",
        data_domain="deterministic_synthetic_sequences",
        evaluation_regime=dataset.regime.regime_id,
        conditioning=("available_history_prefix",),
        policy_class="fixed_probability_threshold",
        reference_state=frozen.hashes.frozen_artifact,
        time_horizon=protocol.task.horizon,
        task_family=(protocol.task.task_id,),
        resource_state=protocol.task.resource_regime,
        uncertainty_treatment="point_estimates; no transport guarantee",
        metric_type="forecast_loss",
        utility="binary_brier_loss",
        population="supplied_independent_synthetic_sequences",
        causal_status=CausalStatus.PREDICTIVE,
        evidence_class=EvidenceClass.SIMULATION,
        information_condition="source-trained frozen history representation",
        estimator="source_only_gaussian_kernel_conditional_empirical_law",
        time_unit=protocol.task.time_unit,
    )
    return RegimeEvaluation(
        dataset.regime,
        dataset.partition,
        dataset.content_hash,
        dataset.sequence_ids,
        probabilities,
        outcomes,
        math.fsum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(outcomes),
        sum((p >= protocol.decision_threshold) == y for p, y in zip(probabilities, outcomes))
        / len(outcomes),
        expected_calibration_error(probabilities, outcomes, bins=protocol.calibration_bins),
        context,
    )


def evaluate_transfer(
    frozen: FrozenTransfer, source_test: RegimeDataset, target_test: RegimeDataset
) -> CrossRegimeTransferResult:
    """Score the same frozen artifact in both regimes, without test-set tuning.

    Partition identities are global independent sequence IDs. Reusing a training
    unit in either holdout, or a unit across the two holdouts, is rejected.
    """
    if not isinstance(frozen, FrozenTransfer):
        raise ValueError("evaluation requires freeze_transfer before accessing test sets")
    protocol = frozen.trained.protocol
    seen = set(frozen.trained.source_training.sequence_ids)
    for dataset, partition, regime in (
        (source_test, TransferPartition.SOURCE_TEST, protocol.source_regime),
        (target_test, TransferPartition.TARGET_TEST, protocol.target_regime),
    ):
        if not isinstance(dataset, RegimeDataset) or (
            dataset.partition != partition or dataset.regime != regime
        ):
            raise ValueError(f"evaluation requires the declared {partition.value} regime")
        if seen.intersection(dataset.sequence_ids):
            raise ValueError("independent sequence leakage across train/test or regime partitions")
        seen.update(dataset.sequence_ids)
    source, target = _evaluate(frozen, source_test), _evaluate(frozen, target_test)
    return CrossRegimeTransferResult(
        protocol.task,
        frozen.hashes,
        source,
        target,
        TransportDiagnostics(
            target.brier_score - source.brier_score,
            target.accuracy - source.accuracy,
            target.calibration_error - source.calibration_error,
        ),
    )

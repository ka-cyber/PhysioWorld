"""Executable, explicitly synthetic Level 3 experiments.

The functions in this module are small numerical experiments, not replicas of
the unavailable source studies.  Every result therefore carries a synthetic
origin, a declared estimand, and the controls used to falsify the local
calculation.  In particular, predictive scores below are calculated from
held-out synthetic subjects; they are never formula-valued placeholders for
the source tables.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from physioworld.capability import (
    CapabilityRegion,
    condition_on_resource,
    finite_policy_pir,
    finite_support_gap,
)
from physioworld.capability.values import (
    InformationAvailabilityDesign,
    InformationDesignKind,
    estimate_piv,
)
from physioworld.core import (
    CausalStatus,
    EvidenceClass,
    Metric,
    MetricContext,
    TaskKind,
)
from physioworld.core.evidence import Assumption, AssumptionSet, AssumptionStatus
from physioworld.instrumentation.nirs import IntensitySignal, raw_proxy_experiment
from physioworld.tasklaws import TaskDefinition, clustered_bootstrap, conditional_task_law

_DIRECTIONS = ("BFV-R", "BFV-L", "BP", "Joint")
_HORIZONS = (10, 20, 30)
_DENSE_HISTORY_LENGTHS = tuple(range(81))
_EPSILONS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20)
_PRIMARY_EPSILON = 0.08
_TRAIN_SUBJECT_COUNT = 10
_SUBJECT_COUNT = 16
_POINT_COUNT = 120


# These are deliberately finite synthetic analogues.  The labels are retained
# to make comparisons easy to audit, but none is asserted to be the source
# paper's representation.
SYNTHETIC_CANDIDATE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "Z1S": {
        "synthetic_name": "Z1S-synthetic-full-history",
        "definition": "ordered 80-sample history vector immediately before prediction",
        "feature_family": "finite_full_history",
        "dimension": 80,
        "memory_values": 80,
        "complexity_rank": 5,
    },
    "ZA15": {
        "synthetic_name": "ZA15-synthetic-recent-history",
        "definition": "ordered 15-sample history vector immediately before prediction",
        "feature_family": "finite_history",
        "dimension": 15,
        "memory_values": 15,
        "complexity_rank": 2,
    },
    "ZB15": {
        "synthetic_name": "ZB15-synthetic-summary",
        "definition": "mean, last value, least-squares slope, and standard deviation over 15 samples",
        "feature_family": "handcrafted_summary",
        "dimension": 4,
        "memory_values": 15,
        "complexity_rank": 1,
    },
    "ZC15": {
        "synthetic_name": "ZC15-synthetic-multiscale-summary",
        "definition": "mean, last value, slope, and standard deviation at 5, 15, and 30 samples",
        "feature_family": "multiscale_summary",
        "dimension": 12,
        "memory_values": 30,
        "complexity_rank": 3,
    },
    "ZMS": {
        "synthetic_name": "ZMS-synthetic-multiscale-trend",
        "definition": "mean and slope at 5, 15, 30, and 60 samples plus the last value",
        "feature_family": "multiscale_summary",
        "dimension": 9,
        "memory_values": 60,
        "complexity_rank": 4,
    },
}


def _auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _rmse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("predictions and targets must be equally sized and nonempty")
    return math.sqrt(
        sum(
            (float(prediction) - float(target)) ** 2
            for prediction, target in zip(predictions, targets)
        )
        / len(predictions)
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _synthetic_subjects(
    seed: int,
    *,
    subject_count: int = _SUBJECT_COUNT,
    point_count: int = _POINT_COUNT,
    regime_shift: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    """Generate subject-clustered, temporally correlated synthetic signals."""
    rng = random.Random(seed)
    subjects = []
    for subject_index in range(subject_count):
        phase = rng.uniform(-math.pi, math.pi)
        baseline = 70.0 + rng.gauss(0.0, 2.5) + regime_shift
        level = rng.gauss(0.0, 0.5)
        signal = []
        for time_index in range(point_count):
            level = 0.94 * level + rng.gauss(0.0, 0.18)
            slow = 2.7 * math.sin(time_index / 11.0 + phase)
            medium = 0.9 * math.sin(time_index / 3.7 - phase / 2.0)
            signal.append(
                baseline
                + slow
                + medium
                + 0.35 * level
                + 0.25 * math.cos(time_index / 2.0 + subject_index)
            )
        subjects.append(
            {
                "subject_id": f"synthetic-subject-{subject_index:02d}",
                "signal": tuple(signal),
                "regime": "shifted" if regime_shift else "reference",
            }
        )
    return tuple(subjects)


def _history(signal: Sequence[float], time_index: int, length: int) -> tuple[float, ...]:
    if length < 0:
        raise ValueError("history length must be nonnegative")
    if length == 0:
        return ()
    return tuple(signal[max(0, time_index - length) : time_index])


def _slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_x = (len(values) - 1) / 2.0
    mean_y = _mean(values)
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    return (
        sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values)) / denominator
    )


def _spread(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(_mean([(value - mean) ** 2 for value in values]))


def _summary(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    return (_mean(values), values[-1], _slope(values), _spread(values))


def _candidate_features(candidate: str, history: Sequence[float]) -> tuple[float, ...]:
    if candidate not in SYNTHETIC_CANDIDATE_DEFINITIONS:
        raise KeyError(candidate)
    if candidate == "Z1S":
        return tuple(history)
    if candidate == "ZA15":
        return tuple(history[-15:])
    if candidate == "ZB15":
        return _summary(history[-15:])
    if candidate == "ZC15":
        return tuple(value for scale in (5, 15, 30) for value in _summary(history[-scale:]))
    multiscale: list[float] = []
    for scale in (5, 15, 30, 60):
        summary = _summary(history[-scale:])
        multiscale.extend((summary[0], summary[2]))
    return (*multiscale, history[-1] if history else 0.0)


def _feature_scalar(feature: Sequence[float]) -> float:
    """One declared scalar used only to form finite conditional-law strata."""
    return _mean(feature) if feature else 0.0


def _knn_predict(
    train_features: Sequence[Sequence[float]],
    train_targets: Sequence[float],
    query: Sequence[float],
    *,
    neighbors: int = 7,
) -> float:
    if len(train_features) != len(train_targets) or not train_targets:
        raise ValueError("a nearest-neighbour predictor needs aligned training data")
    if not query:
        return _mean(train_targets)
    if any(len(feature) != len(query) for feature in train_features):
        raise ValueError("nearest-neighbour feature dimensions do not match")
    distances = sorted(
        (
            sum((left - right) ** 2 for left, right in zip(feature, query)) / len(query),
            target,
        )
        for feature, target in zip(train_features, train_targets)
    )[: min(neighbors, len(train_targets))]
    weights = tuple(1.0 / (distance + 1e-8) for distance, _ in distances)
    return sum(weight * target for weight, (_, target) in zip(weights, distances)) / sum(weights)


def _target(signal: Sequence[float], time_index: int, horizon: int, direction: str) -> float:
    future = tuple(signal[time_index + 1 : time_index + horizon + 1])
    if len(future) != horizon:
        raise ValueError("target horizon extends beyond the synthetic recording")
    future_mean = _mean(future)
    future_trend = (future[-1] - future[0]) / max(1, horizon - 1)
    future_variability = _spread(future)
    if direction == "BFV-R":
        return future_mean
    if direction == "BFV-L":
        return future_mean + 5.0 * future_trend
    if direction == "BP":
        return future_mean + 0.7 * future_variability
    if direction == "Joint":
        return (
            0.55 * future_mean
            + 0.30 * (future_mean + 5.0 * future_trend)
            + 0.15 * future_variability
        )
    raise KeyError(direction)


def _examples(
    subjects: Sequence[Mapping[str, Any]],
    *,
    history_length: int,
    horizon: int,
    direction: str,
    step: int,
) -> list[dict[str, Any]]:
    rows = []
    for subject in subjects:
        signal = subject["signal"]
        for time_index in range(80, len(signal) - horizon, step):
            history = _history(signal, time_index, history_length)
            rows.append(
                {
                    "subject_id": subject["subject_id"],
                    "time_index": time_index,
                    "history": history,
                    "target": _target(signal, time_index, horizon, direction),
                    "full_scalar": _feature_scalar(_history(signal, time_index, 80)),
                }
            )
    if not rows:
        raise ValueError("synthetic design produced no evaluable examples")
    return rows


def _split_subjects(
    subjects: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if len(subjects) < 4:
        raise ValueError("synthetic train/test split needs at least four subjects")
    return tuple(subjects[:_TRAIN_SUBJECT_COUNT]), tuple(subjects[_TRAIN_SUBJECT_COUNT:])


def _bin_edges(values: Sequence[float], bins: int = 4) -> tuple[float, ...]:
    if not values:
        raise ValueError("cannot make strata from empty values")
    ordered = sorted(values)
    return tuple(
        ordered[min(len(ordered) - 1, (len(ordered) * index) // bins)] for index in range(1, bins)
    )


def _bin(value: float, edges: Sequence[float]) -> int:
    return sum(value > edge for edge in edges)


def _conditional_discrepancy(
    observations: Sequence[Mapping[str, Any]], *, value_key: str = "target"
) -> tuple[float, dict[str, float]]:
    """Compare held-out full-history and representation conditional means."""
    if not observations:
        raise ValueError("conditional discrepancy needs observations")
    by_cell: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_cell[observation["cell"]].append(observation)
    cell_values: dict[str, float] = {}
    for cell, cell_observations in by_cell.items():
        full_groups: dict[int, list[float]] = defaultdict(list)
        representation_groups: dict[int, list[float]] = defaultdict(list)
        for observation in cell_observations:
            full_groups[observation["full_group"]].append(observation[value_key])
            representation_groups[observation["representation_group"]].append(
                observation[value_key]
            )
        full_means = {group: _mean(values) for group, values in full_groups.items()}
        representation_means = {
            group: _mean(values) for group, values in representation_groups.items()
        }
        cell_values[cell] = _mean(
            [
                abs(
                    full_means[item["full_group"]]
                    - representation_means[item["representation_group"]]
                )
                for item in cell_observations
            ]
        )
    return max(cell_values.values()), cell_values


def _battery_evaluation(seed: int) -> dict[str, Any]:
    subjects = _synthetic_subjects(seed)
    train_subjects, test_subjects = _split_subjects(subjects)
    cells: list[dict[str, Any]] = []
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in SYNTHETIC_CANDIDATE_DEFINITIONS:
        for direction in _DIRECTIONS:
            for horizon in _HORIZONS:
                train = _examples(
                    train_subjects,
                    history_length=80,
                    horizon=horizon,
                    direction=direction,
                    step=2,
                )
                test = _examples(
                    test_subjects,
                    history_length=80,
                    horizon=horizon,
                    direction=direction,
                    step=2,
                )
                train_features = [_candidate_features(candidate, row["history"]) for row in train]
                test_features = [_candidate_features(candidate, row["history"]) for row in test]
                train_scalars = [_feature_scalar(feature) for feature in train_features]
                test_scalars = [_feature_scalar(feature) for feature in test_features]
                full_train_scalars = [row["full_scalar"] for row in train]
                full_test_scalars = [row["full_scalar"] for row in test]
                representation_edges = _bin_edges(train_scalars)
                full_edges = _bin_edges(full_train_scalars)
                predictions = [
                    _knn_predict(train_features, [row["target"] for row in train], feature)
                    for feature in test_features
                ]
                targets = [row["target"] for row in test]
                threshold = _mean(sorted(row["target"] for row in train)[len(train) // 2 :])
                labels = [int(target >= threshold) for target in targets]
                permutation = list(labels)
                random.Random(seed + horizon * 101 + len(direction) * 17).shuffle(permutation)
                cell_id = f"{direction}@{horizon}s"
                for row, full_scalar, representation_scalar, label in zip(
                    test, full_test_scalars, test_scalars, labels
                ):
                    observations[candidate].append(
                        {
                            "cell": cell_id,
                            "subject_id": row["subject_id"],
                            "target": row["target"],
                            "task_outcome": label,
                            "full_group": _bin(full_scalar, full_edges),
                            "representation_group": _bin(
                                representation_scalar, representation_edges
                            ),
                        }
                    )
                cell_auc = _auc(labels, predictions)
                permuted_auc = _auc(permutation, predictions)
                if cell_auc is None or permuted_auc is None:
                    raise ValueError(f"synthetic task cell {cell_id} has one outcome class")
                cells.append(
                    {
                        "representation": candidate,
                        "synthetic_representation": SYNTHETIC_CANDIDATE_DEFINITIONS[candidate][
                            "synthetic_name"
                        ],
                        "direction": direction,
                        "horizon_s": horizon,
                        "intrinsic_task_value": cell_auc,
                        "held_out_auc": cell_auc,
                        "permuted_label_auc": permuted_auc,
                        "prediction_method": "training-subject 7-nearest-neighbour regression",
                        "train_subject_count": len(train_subjects),
                        "test_subject_count": len(test_subjects),
                        "prediction_count": len(predictions),
                        "predictions": predictions,
                        "targets": targets,
                        "labels": labels,
                        "evidence_class": EvidenceClass.SIMULATION.value,
                    }
                )
    summaries = {}
    for candidate in SYNTHETIC_CANDIDATE_DEFINITIONS:
        values = [item["held_out_auc"] for item in cells if item["representation"] == candidate]
        discrepancy, per_cell = _conditional_discrepancy(
            observations[candidate], value_key="task_outcome"
        )
        summaries[candidate] = {
            "mean": _mean(values),
            "worst_cell": min(values),
            "cell_count": len(values),
            "task_law_discrepancy": discrepancy,
            "task_law_discrepancy_by_cell": per_cell,
            "complexity_rank": SYNTHETIC_CANDIDATE_DEFINITIONS[candidate]["complexity_rank"],
        }
    return {
        "subjects": subjects,
        "train_subjects": train_subjects,
        "test_subjects": test_subjects,
        "cells": cells,
        "summaries": summaries,
        "observations": observations,
    }


def run_temporal_history_sweep(
    lengths: Sequence[int] = _DENSE_HISTORY_LENGTHS,
    *,
    seed: int = 101,
) -> dict[str, Any]:
    """Evaluate actual held-out forecasts for every history length from 0 to 80."""
    if not lengths or any(isinstance(length, bool) or length < 0 for length in lengths):
        raise ValueError("history lengths must be nonnegative")
    subjects = _synthetic_subjects(seed)
    train_subjects, test_subjects = _split_subjects(subjects)
    rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    shuffle_rng = random.Random(seed + 9001)
    task = TaskDefinition("one_step_mean", TaskKind.FORECASTING, 1.0, "pressure", "rmse")
    for length in lengths:
        train = _examples(
            train_subjects,
            history_length=length,
            horizon=1,
            direction="BFV-R",
            step=3,
        )
        test = _examples(
            test_subjects,
            history_length=length,
            horizon=1,
            direction="BFV-R",
            step=3,
        )
        train_features = [row["history"] for row in train]
        test_features = [row["history"] for row in test]
        train_targets = [row["target"] for row in train]
        targets = [row["target"] for row in test]
        predictions = [
            _knn_predict(train_features, train_targets, feature) for feature in test_features
        ]
        shuffled_features = []
        for feature in test_features:
            shuffled = list(feature)
            shuffle_rng.shuffle(shuffled)
            shuffled_features.append(tuple(shuffled))
        shuffled_predictions = [
            _knn_predict(train_features, train_targets, feature) for feature in shuffled_features
        ]
        threshold = _mean(sorted(train_targets)[len(train_targets) // 2 :])
        labels = [int(target >= threshold) for target in targets]
        full_scalars = [row["full_scalar"] for row in test]
        representation_scalars = [_feature_scalar(feature) for feature in test_features]
        train_full_scalars = [row["full_scalar"] for row in train]
        train_prediction_scalars = [_feature_scalar(feature) for feature in train_features]
        law = conditional_task_law(
            targets,
            tuple(
                _bin(value, _bin_edges(train_prediction_scalars))
                for value in representation_scalars
            ),
            task,
            history_groups=tuple(
                _bin(value, _bin_edges(train_full_scalars)) for value in full_scalars
            ),
            evidence_class=EvidenceClass.SIMULATION,
            evidence_metadata={"source_artifact": "native.synthetic.temporal_history.v2"},
        )
        shuffle_law = conditional_task_law(
            targets,
            tuple(
                _bin(_feature_scalar(feature), _bin_edges(train_prediction_scalars))
                for feature in shuffled_features
            ),
            task,
            history_groups=tuple(
                _bin(value, _bin_edges(train_full_scalars)) for value in full_scalars
            ),
            evidence_class=EvidenceClass.SIMULATION,
            evidence_metadata={"source_artifact": "native.synthetic.temporal_history.v2.shuffle"},
        )
        row = {
            "history_length": length,
            "history_sample_count": len(test_features[0]),
            "prediction_rmse": _rmse(predictions, targets),
            "task_auc": _auc(labels, predictions),
            "task_law_discrepancy": law.discrepancy(),
            "predictions": predictions,
            "targets": targets,
            "prediction_method": "training-subject 7-nearest-neighbour regression",
            "compute_cost": float(len(train) * len(test) * max(1, len(test_features[0]))),
            "latency_samples": len(test_features[0]),
            "memory_values": len(test_features[0]),
            "resource_cost": float(
                len(train) * len(test) * max(1, len(test_features[0]))
                + 0.25 * len(test_features[0])
            ),
            "evidence_class": EvidenceClass.SIMULATION.value,
        }
        shuffled_row = {
            "history_length": length,
            "prediction_rmse": _rmse(shuffled_predictions, targets),
            "task_auc": _auc(labels, shuffled_predictions),
            "task_law_discrepancy": shuffle_law.discrepancy(),
            "predictions": shuffled_predictions,
            "targets": targets,
            "control": "within-test-window temporal order shuffle; model fit remains training-only",
        }
        rows.append(row)
        shuffle_rows.append(shuffled_row)
    by_length = {row["history_length"]: row for row in rows}
    return {
        "experiment_id": "temporal_history_sweep",
        "status": "PASS",
        "data_origin": "DETERMINISTIC_SYNTHETIC_PHYSIOLOGY",
        "data_fidelity": "SYNTHETIC",
        "evidence_class": EvidenceClass.SIMULATION.value,
        "estimand": "held_out_one_step_prediction_RMSE_and_declared_task_law_by_history_length",
        "seed": seed,
        "history_lengths": list(lengths),
        "rows": rows,
        "train_subject_ids": [subject["subject_id"] for subject in train_subjects],
        "test_subject_ids": [subject["subject_id"] for subject in test_subjects],
        "leakage_control": {
            "subject_disjoint_train_test": not {
                subject["subject_id"] for subject in train_subjects
            }.intersection(subject["subject_id"] for subject in test_subjects),
            "future_samples_in_features": False,
            "no_history_definition": "empty history; prediction is the training-target baseline",
            "no_history_uses_last_sample": False,
            "no_history_prediction_is_training_baseline": all(
                abs(
                    prediction
                    - _mean(
                        [
                            row["target"]
                            for row in _examples(
                                train_subjects,
                                history_length=0,
                                horizon=1,
                                direction="BFV-R",
                                step=3,
                            )
                        ]
                    )
                )
                < 1e-12
                for prediction in by_length[0]["predictions"]
            )
            if 0 in by_length
            else None,
        },
        "temporal_shuffle_control": {
            "executed": True,
            "definition": "randomly permute each held-out history window while retaining its target and the training fit",
            "rows": shuffle_rows,
            "mean_rmse": _mean([row["prediction_rmse"] for row in shuffle_rows]),
            "mean_auc": _mean(
                [row["task_auc"] for row in shuffle_rows if row["task_auc"] is not None]
            ),
            "zero_length_is_unchanged": by_length.get(0, {}).get("prediction_rmse")
            == next(row["prediction_rmse"] for row in shuffle_rows if row["history_length"] == 0),
        },
        "negative_controls": {
            "temporal_shuffle": "EXECUTED",
            "future_information": "NOT_USED",
            "universal_memory_constant": "FORBIDDEN_INTERPRETATION",
        },
        "limitations": [
            "Synthetic subject process and finite train/test split; no external cohort reproduction.",
            "History length is a declared sample-count window in this fixture, not a physiological memory constant.",
            "The shuffle is a predictive negative control, not a causal intervention.",
        ],
    }


def run_task_battery(*, seed: int = 102) -> dict[str, Any]:
    """Run actual held-out predictions for five declared synthetic candidates."""
    evaluation = _battery_evaluation(seed)
    return {
        "experiment_id": "task_law_battery",
        "status": "PASS",
        "data_origin": "DETERMINISTIC_SYNTHETIC_TASK_BATTERY",
        "data_fidelity": "SYNTHETIC",
        "evidence_class": EvidenceClass.SIMULATION.value,
        "estimand": "held_out_AUC_by_synthetic_representation_future_direction_and_horizon",
        "seed": seed,
        "cell_count": len(evaluation["cells"]),
        "task_cell_count": len(_DIRECTIONS) * len(_HORIZONS),
        "directions": list(_DIRECTIONS),
        "horizons_s": list(_HORIZONS),
        "cells": evaluation["cells"],
        "summaries": evaluation["summaries"],
        "candidate_definitions": SYNTHETIC_CANDIDATE_DEFINITIONS,
        "train_subject_ids": [subject["subject_id"] for subject in evaluation["train_subjects"]],
        "test_subject_ids": [subject["subject_id"] for subject in evaluation["test_subjects"]],
        "leakage_control": {
            "subject_disjoint_train_test": True,
            "future_samples_in_features": False,
            "predictions_are_executed": True,
            "label_permutation_control": {
                "executed": True,
                "metric": "AUC after deterministic held-out-label permutation",
            },
        },
        "objectives": {
            "mean_task_value": "mean of the 12 held-out cell AUCs",
            "worst_cell_task_value": "minimum of the 12 held-out cell AUCs",
            "task_law_discrepancy": "maximum held-out conditional-mean discrepancy for the declared binary task outcome over the 12 cells",
        },
        "negative_controls": {
            "mean_vs_worst_cell": "SEPARATE_OBJECTIVES",
            "label_permutation": "EXECUTED_PER_CELL",
        },
        "reference_comparison": "INACTIVE_EXTERNAL_REFERENCE_REQUIRES_SOURCE_CORPUS",
        "limitations": [
            "All five labels name finite synthetic analogues; they are not authoritative mappings to source representations.",
            "Held-out AUC is predictive task value in this fixture, not causal, clinical, or external validation evidence.",
        ],
    }


def run_task_law_discrepancy(*, seed: int = 104) -> dict[str, Any]:
    """Estimate task-law discrepancy and apply gates chosen before execution."""
    evaluation = _battery_evaluation(seed)
    candidates: dict[str, dict[str, Any]] = {}
    for candidate, candidate_observations in evaluation["observations"].items():
        point, per_cell = _conditional_discrepancy(candidate_observations, value_key="task_outcome")
        clusters = [observation["subject_id"] for observation in candidate_observations]

        def estimate_discrepancy(
            indices: tuple[int, ...], observations: list[dict[str, Any]] = candidate_observations
        ) -> float:
            return float(
                _conditional_discrepancy(
                    [observations[index] for index in indices], value_key="task_outcome"
                )[0]
            )

        bootstrap = clustered_bootstrap(
            len(candidate_observations),
            estimate_discrepancy,
            clusters=clusters,
            repetitions=199,
            seed=seed + len(candidate),
            evidence_class=EvidenceClass.SIMULATION,
            evidence_metadata={"source_artifact": "native.synthetic.cluster_bootstrap.v2"},
        )
        gates = {
            str(epsilon): {
                "point_gate": point <= epsilon,
                "upper_bound_gate": bootstrap.upper <= epsilon,
                "pass": point <= epsilon and bootstrap.upper <= epsilon,
            }
            for epsilon in _EPSILONS
        }
        candidates[candidate] = {
            "point_estimate": point,
            "per_cell": per_cell,
            "bootstrap": bootstrap.to_dict(),
            "bootstrap_estimand": "subject-cluster percentile interval with refit of conditional means",
            "epsilon_gates": gates,
            "primary_epsilon": _PRIMARY_EPSILON,
        }
    feasible = [
        candidate
        for candidate, result in candidates.items()
        if result["epsilon_gates"][str(_PRIMARY_EPSILON)]["pass"]
    ]
    best = min(
        feasible,
        key=lambda candidate: SYNTHETIC_CANDIDATE_DEFINITIONS[candidate]["complexity_rank"],
        default=None,
    )
    return {
        "experiment_id": "task_law_discrepancy",
        "status": "PASS",
        "data_origin": "DETERMINISTIC_SYNTHETIC_TASK_BATTERY",
        "data_fidelity": "SYNTHETIC",
        "evidence_class": EvidenceClass.SIMULATION.value,
        "estimand": "D_hat_T_sup_task_mean_absolute_conditional_expectation_difference",
        "seed": seed,
        "candidate_results": candidates,
        "primary_epsilon": _PRIMARY_EPSILON,
        "epsilon_sensitivity_grid": list(_EPSILONS),
        "gate_rule": "point estimate and subject-cluster 95% upper percentile bound must both be <= predeclared epsilon",
        "best_complexity_candidate_at_primary_epsilon": best,
        "bootstrap_is_not_primary_point_estimate": True,
        "cluster_unit": "synthetic subject",
        "leakage_control": "all conditional means and bootstrap replicates use held-out observations; training subjects never enter the test rows",
        "limitations": [
            "The tolerance grid and primary epsilon are synthetic protocol choices, not source-study thresholds.",
            "The interval quantifies this synthetic subject-cluster resampling procedure and is not a causal or population confidence interval.",
        ],
    }


def run_subject_bootstrap_uncertainty(*, seed: int = 105) -> dict[str, Any]:
    """Expose the subject-cluster uncertainty calculation as its own runner."""
    result = run_task_law_discrepancy(seed=seed)
    return {
        "experiment_id": "subject_bootstrap_uncertainty",
        "status": "PASS",
        "data_origin": "DETERMINISTIC_SYNTHETIC_TASK_BATTERY",
        "data_fidelity": "SYNTHETIC",
        "evidence_class": EvidenceClass.SIMULATION.value,
        "estimand": "clustered_subject_bootstrap_interval_for_task_law_discrepancy",
        "seed": seed,
        "resampling_unit": "subject",
        "windows_resampled_within_subject": True,
        "subjects_resampled_across_replicates": True,
        "candidate_intervals": {
            candidate: result["candidate_results"][candidate]["bootstrap"]
            for candidate in SYNTHETIC_CANDIDATE_DEFINITIONS
        },
        "primary_epsilon": result["primary_epsilon"],
        "limitations": [
            "Synthetic uncertainty only; no source cohort or clinical confidence statement is made."
        ],
    }


def run_representation_lattice(*, seed: int = 103) -> dict[str, Any]:
    """Compare synthetic representation families without a mean-AUC-only ranking."""
    evaluation = _battery_evaluation(seed)
    aliases = {
        "instantaneous-synthetic": "ZB15",
        "finite-history-synthetic": "ZA15",
        "multiscale-synthetic": "ZMS",
        "handcrafted-statistical-synthetic": "ZB15",
        "full-history-synthetic": "Z1S",
        "task-specific-synthetic": "ZC15",
        "resource-aware-synthetic": "ZA15",
    }
    rows = []
    for name, candidate in aliases.items():
        summary = evaluation["summaries"][candidate]
        values = [
            item["held_out_auc"]
            for item in evaluation["cells"]
            if item["representation"] == candidate
        ]
        rows.append(
            {
                "representation": name,
                "synthetic_candidate": candidate,
                "definition": SYNTHETIC_CANDIDATE_DEFINITIONS[candidate]["definition"],
                "dimensionality": SYNTHETIC_CANDIDATE_DEFINITIONS[candidate]["dimension"],
                "memory_values": SYNTHETIC_CANDIDATE_DEFINITIONS[candidate]["memory_values"],
                "parameters": 0,
                "training_cost": float(len(evaluation["train_subjects"]) * len(values)),
                "inference_cost": float(
                    len(values) * SYNTHETIC_CANDIDATE_DEFINITIONS[candidate]["dimension"]
                ),
                "modality_count": 1,
                "energy_model": "declared synthetic cost proportional to feature dimension",
                "task_law_discrepancy": summary["task_law_discrepancy"],
                "predictive_performance_mean_auc": summary["mean"],
                "predictive_performance_worst_auc": summary["worst_cell"],
                "decision_value": _mean([2.0 * value - 1.0 for value in values]),
                "policy_value": summary["worst_cell"],
                "uncertainty": "subject-cluster interval available from run_subject_bootstrap_uncertainty",
                "cross_regime_transfer": "not estimated by this finite lattice runner",
                "causal_status": CausalStatus.PREDICTIVE.value,
            }
        )
    return {
        "experiment_id": "representation_lattice",
        "status": "PASS",
        "data_origin": "DETERMINISTIC_SYNTHETIC_REPRESENTATION_LATTICE",
        "data_fidelity": "SYNTHETIC",
        "evidence_class": EvidenceClass.SIMULATION.value,
        "estimand": "joint_task_law_decision_policy_and_resource_comparison_across_synthetic_representations",
        "seed": seed,
        "rows": rows,
        "ranking_rule": "no single ranking; report mean AUC, worst-cell AUC, discrepancy, utility, and resource fields separately",
        "causal_firewall": "predictive scores do not identify causal sufficiency",
        "limitations": [
            "Several named lattice families are finite synthetic analogues backed by the declared candidate definitions.",
            "Cross-regime transfer is outside this finite lattice runner and is not silently inferred.",
        ],
    }


def run_support_function_experiment(*, seed: int = 113) -> dict[str, Any]:
    """Check independently optimized vector supports and finite enumeration."""
    full = CapabilityRegion(
        {"f0": (0.92, 0.20), "f1": (0.20, 0.92), "balanced": (0.64, 0.64)},
        "full_history",
    )
    reduced = CapabilityRegion(
        {"r0": (0.80, 0.30), "r1": (0.30, 0.80)},
        "compact_representation",
    )
    directions = tuple(
        (math.cos(2 * math.pi * index / 16), math.sin(2 * math.pi * index / 16))
        for index in range(16)
    )

    def direct_support(
        region: CapabilityRegion, direction: tuple[float, float]
    ) -> tuple[float, str]:
        scored = {
            policy: sum(weight * value for weight, value in zip(direction, vector))
            for policy, vector in region.policy_values.items()
        }
        policy = max(scored, key=scored.__getitem__)
        return scored[policy], policy

    diagnostics: list[dict[str, Any]] = []
    independent_errors: list[float] = []
    for direction in directions:
        module_gap = finite_support_gap(full, reduced, direction)
        full_value, full_policy = direct_support(full, direction)
        reduced_value, reduced_policy = direct_support(reduced, direction)
        independent_errors.append(
            max(
                abs(module_gap.full_support - full_value),
                abs(module_gap.reduced_support - reduced_value),
                abs(module_gap.gap - (full_value - reduced_value)),
            )
        )
        diagnostics.append(
            {
                "direction": list(direction),
                "full_support": full_value,
                "reduced_support": reduced_value,
                "gap": full_value - reduced_value,
                "full_policy": full_policy,
                "reduced_policy": reduced_policy,
                "enumerated_full_policy_scores": {
                    policy: sum(q * value for q, value in zip(direction, vector))
                    for policy, vector in full.policy_values.items()
                },
                "enumerated_reduced_policy_scores": {
                    policy: sum(q * value for q, value in zip(direction, vector))
                    for policy, vector in reduced.policy_values.items()
                },
            }
        )
    coordinatewise = full.coordinatewise_max()
    return {
        "experiment_id": "support_function_correction",
        "status": "PASS",
        "data_origin": "FINITE_SYNTHETIC_POLICY_IMAGE",
        "data_fidelity": "MATHEMATICAL",
        "evidence_class": EvidenceClass.NUMERICAL_VERIFICATION.value,
        "estimand": "independently_optimized_support_gap_by_utility_direction",
        "seed": seed,
        "directions": [list(direction) for direction in directions],
        "diagnostics": diagnostics,
        "support_gaps": [diagnostic["gap"] for diagnostic in diagnostics],
        "coordinatewise_maximum": list(coordinatewise),
        "coordinatewise_maximum_jointly_attainable": full.jointly_attainable(coordinatewise),
        "independent_reference": {
            "method": "fresh direct enumeration of every policy vector for each direction",
            "maximum_absolute_gap": max(abs(item["gap"]) for item in diagnostics),
            "maximum_module_vs_reference_error": max(independent_errors),
            "matches_module_support": max(independent_errors) <= 1e-12,
        },
        "limitations": [
            "Finite policy image and sampled directions; no universal capability dominance claim."
        ],
    }


def _synthetic_value_context(
    information_condition: str, resource_state: str = "unconstrained"
) -> MetricContext:
    return MetricContext(
        estimand_name="policy_value",
        data_domain="synthetic finite policy population",
        evaluation_regime="finite policy replay",
        conditioning=("same_initial_state",),
        policy_class="declared_synthetic_policies",
        reference_state="synthetic_reference_state",
        time_horizon=1.0,
        task_family=("task-a", "task-b"),
        resource_state=resource_state,
        uncertainty_treatment="none_declared",
        metric_type="task_value",
        utility="weighted synthetic task utility",
        population="synthetic_units",
        causal_status=CausalStatus.MODEL_INTERVENTIONAL,
        evidence_class=EvidenceClass.SIMULATION,
        communication_regime="none",
        information_condition=information_condition,
        estimator="finite_policy_enumeration",
        time_unit="s",
    )


def run_resource_conditioned_capability(*, seed: int = 112) -> dict[str, Any]:
    """Evaluate finite policy images after applying explicit resource feasibility."""
    evaluation = _battery_evaluation(seed)
    intrinsic = {
        candidate: (
            _mean(
                [
                    item["held_out_auc"]
                    for item in evaluation["cells"]
                    if item["representation"] == candidate and item["direction"] == "BFV-R"
                ]
            ),
            min(
                item["held_out_auc"]
                for item in evaluation["cells"]
                if item["representation"] == candidate and item["direction"] == "BP"
            ),
        )
        for candidate in SYNTHETIC_CANDIDATE_DEFINITIONS
    }
    context = _synthetic_value_context("intrinsic_synthetic_information")
    region = CapabilityRegion(
        intrinsic,
        "synthetic_intrinsic_policy_image",
        task_labels=("BFV-R mean AUC", "BP worst-horizon AUC"),
        context=context,
    )
    costs = {
        candidate: float(definition["memory_values"] + 2 * definition["dimension"])
        for candidate, definition in SYNTHETIC_CANDIDATE_DEFINITIONS.items()
    }
    budgets = {"ample": 220.0, "edge": 60.0, "tiny": 30.0}
    conditions = {}
    for resource_condition, budget in budgets.items():
        feasible = [candidate for candidate, cost in costs.items() if cost <= budget]
        conditioned = condition_on_resource(region, feasible, resource_condition)
        conditioned_region = conditioned.at(resource_condition)
        supports = {
            "BFV-R": conditioned_region.support((1.0, 0.0)),
            "BP": conditioned_region.support((0.0, 1.0)),
            "balanced": conditioned_region.support((0.5, 0.5)),
        }
        conditions[resource_condition] = {
            "budget": budget,
            "feasible_policies": feasible,
            "policy_costs": {candidate: costs[candidate] for candidate in feasible},
            "region": conditioned_region.to_dict(),
            "independently_optimized_supports": {
                name: {"value": value, "policy": policy}
                for name, (value, policy) in supports.items()
            },
        }
    return {
        "experiment_id": "resource_conditioned_capability",
        "status": "PASS",
        "data_origin": "DETERMINISTIC_SYNTHETIC_TASK_BATTERY",
        "data_fidelity": "SYNTHETIC",
        "evidence_class": EvidenceClass.SIMULATION.value,
        "estimand": "capability_region_conditioned_on_declared_resource_feasibility",
        "seed": seed,
        "intrinsic_estimand": "finite held-out policy-task vectors before resource filtering",
        "resource_conditioned_estimand": "same vectors restricted to policies feasible under each declared budget",
        "intrinsic_policy_values": intrinsic,
        "policy_cost_model": "memory_values + 2 * feature_dimension synthetic cost units",
        "resource_conditions": conditions,
        "resource_is_not_intrinsic_information": True,
        "limitations": [
            "Resource budgets and costs are synthetic declarations; they are not energy measurements from source hardware.",
            "The capability image contains five finite candidate policies and does not support a universal representation ordering.",
        ],
    }


def _piv_assumptions() -> AssumptionSet:
    names = (
        "paired_units",
        "same_initial_reference",
        "information_availability_intervention",
        "common_utility",
        "comparable_policy_classes",
        "no_future_information",
    )
    return AssumptionSet(
        tuple(
            Assumption(
                name,
                f"Synthetic paired design satisfies {name}.",
                AssumptionStatus.VERIFIED,
                evidence=("synthetic_design",),
            )
            for name in names
        )
    )


def run_piv_synthetic(*, seed: int = 116) -> dict[str, Any]:
    """Execute a guarded paired model-replay PIV calculation with cluster uncertainty."""
    common: dict[str, Any] = dict(
        estimand_name="policy_value",
        data_domain="synthetic paired units",
        evaluation_regime="paired model replay",
        conditioning=("same_initial_state",),
        policy_class="fixed_policy",
        reference_state="synthetic_baseline",
        time_horizon=1.0,
        task_family=("utility",),
        resource_state="fixed",
        uncertainty_treatment="none_declared",
        metric_type="task_value",
        utility="paired downstream utility",
        population="synthetic_units",
        unit="utility",
        causal_status=CausalStatus.MODEL_INTERVENTIONAL,
        evidence_class=EvidenceClass.SIMULATION,
        communication_regime="none",
        estimator="paired_mean_utility",
    )
    baseline = MetricContext(information_condition="baseline", **common)
    augmented = MetricContext(information_condition="augmented", **common)
    pair_ids = tuple(f"pair-{index}" for index in range(8))
    design = InformationAvailabilityDesign(
        "synthetic_piv",
        "additional_history",
        InformationDesignKind.PAIRED_MODEL_REPLAY,
        baseline,
        augmented,
        pair_ids,
        _piv_assumptions(),
        "native.synthetic.piv.v2",
    )
    baseline_values = (0.40, 0.45, 0.50, 0.55) * 2
    augmented_values = (0.55, 0.60, 0.55, 0.70) * 2
    clusters = ("subject-0", "subject-0", "subject-1", "subject-1") * 2
    estimate = estimate_piv(
        baseline_values,
        augmented_values,
        design=design,
        baseline_pair_ids=pair_ids,
        augmented_pair_ids=pair_ids,
        clusters=clusters,
        bootstrap_repetitions=199,
        seed=seed,
    )
    return {
        "experiment_id": "piv_counterfactual_synthetic",
        "status": "PASS",
        "data_origin": "KNOWN_SYNTHETIC_PAIRED_MODEL",
        "data_fidelity": "SYNTHETIC",
        "evidence_class": EvidenceClass.SIMULATION.value,
        "estimand": "paired_downstream_utility_difference_under_information_intervention",
        "seed": seed,
        "incremental_value": estimate.incremental_value,
        "paired_differences": list(estimate.paired_differences),
        "design": design.to_dict(),
        "uncertainty": estimate.uncertainty.to_dict() if estimate.uncertainty else None,
        "guard_checks": {
            "paired_ids_aligned": True,
            "distinct_information_conditions": baseline.information_condition
            != augmented.information_condition,
            "downstream_utility_not_confidence": True,
            "model_replay_retains_simulation_evidence": estimate.evidence_class
            == EvidenceClass.SIMULATION,
            "unpaired_observational_contrast_used": False,
        },
        "negative_controls": {"unpaired_observational_contrast": "NOT_PIV"},
        "limitations": [
            "Model-replay PIV is a synthetic information intervention and is not empirical causal evidence.",
            "The clustered interval reflects four synthetic subjects repeated across paired rows.",
        ],
    }


def run_pir_loss_decomposition(*, seed: int = 117) -> dict[str, Any]:
    """Account for information loss, controller loss, total loss, and PIR separately."""
    ideal_context = _synthetic_value_context("full_history")
    actual_context = _synthetic_value_context("compact_representation")
    ideal = CapabilityRegion(
        {
            "ideal-level": (0.90, 0.70),
            "ideal-pressure": (0.68, 0.91),
            "ideal-balanced": (0.78, 0.80),
        },
        "synthetic_full_history_policy_image",
        task_labels=("task-a", "task-b"),
        context=ideal_context,
    )
    actual = CapabilityRegion(
        {
            "actual-level": (0.80, 0.63),
            "actual-pressure": (0.62, 0.79),
            "actual-balanced": (0.74, 0.72),
        },
        "synthetic_compact_policy_image",
        task_labels=("task-a", "task-b"),
        context=actual_context,
    )
    direction = (0.6, 0.4)
    minimum = Metric(0.40, actual_context, "native.synthetic.pir.minimum.v1")
    result = finite_policy_pir(
        ideal,
        actual,
        direction,
        deployed_policy_id="actual-pressure",
        minimum=minimum,
    )
    denominator = result.ideal_value - result.minimum_value
    return {
        "experiment_id": "pir_loss_decomposition",
        "status": "PASS",
        "data_origin": "FINITE_SYNTHETIC_POLICY_IMAGE",
        "data_fidelity": "SYNTHETIC",
        "evidence_class": EvidenceClass.NUMERICAL_VERIFICATION.value,
        "estimand": "L_info_L_policy_L_total_and_PIR_under_common_reference",
        "seed": seed,
        "direction": list(direction),
        "ideal_optimal_policy": result.references["ideal_policy"],
        "actual_optimal_policy": result.references["actual_optimal_policy"],
        "deployed_policy": result.references["deployed_policy"],
        "ideal_value": result.ideal_value,
        "actual_optimal_value": result.actual_optimal_value,
        "actual_policy_value": result.actual_policy_value,
        "minimum_value": result.minimum_value,
        "L_info": result.information_loss,
        "L_policy": result.policy_loss,
        "L_total": result.total_loss,
        "PIR": result.pir,
        "PIR_identity": None if denominator == 0 else 1.0 - result.information_loss / denominator,
        "loss_identity_holds": abs(
            result.total_loss - (result.information_loss + result.policy_loss)
        )
        <= 1e-12,
        "controller_value_is_not_actual_optimum": result.actual_policy_value
        < result.actual_optimal_value,
        "common_reference_checked": result.reference_checked,
        "limitations": [
            "Finite synthetic policy images and model-interventional values only; this is accounting verification, not empirical causal evidence."
        ],
    }


def run_nirs_proxy_synthetic(*, seed: int = 109) -> dict[str, Any]:
    """Execute OD semantics on synthetic intensity without promoting it to HbT."""
    intensity = IntensitySignal.from_samples(
        [math.exp(-1.0)] * 20 + [math.exp(-2.0)] * 30,
        1.0,
    )
    result = raw_proxy_experiment(
        intensity,
        event_time_s=20.0,
        baseline_window_s=(-20.0, 0.0),
        response_window_s=(0.0, 30.0),
        evidence_class=EvidenceClass.SIMULATION,
        seed=seed,
    )
    return {
        "experiment_id": "nirs_raw_proxy",
        "status": "PASS",
        "data_origin": "DETERMINISTIC_SYNTHETIC_INTENSITY",
        "data_fidelity": "SYNTHETIC",
        "evidence_class": result.evidence_class.value,
        "quantity": result.quantity.value,
        "delta_optical_density": result.delta_od,
        "direction": result.direction,
        "preprocessing_gate": result.preprocessing_gate,
        "hemoglobin_label": "FORBIDDEN_WITHOUT_SOURCE_DEFINED_RECONSTRUCTION",
        "limitations": list(result.limitations),
    }


__all__ = [
    "SYNTHETIC_CANDIDATE_DEFINITIONS",
    "run_nirs_proxy_synthetic",
    "run_piv_synthetic",
    "run_pir_loss_decomposition",
    "run_representation_lattice",
    "run_resource_conditioned_capability",
    "run_subject_bootstrap_uncertainty",
    "run_support_function_experiment",
    "run_task_battery",
    "run_task_law_discrepancy",
    "run_temporal_history_sweep",
]

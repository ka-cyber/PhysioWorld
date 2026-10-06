"""Empirical conditional task distributions, with explicit estimands and support."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Mapping

from physioworld.core.contracts import EvidenceClass
from physioworld.evidence_origin import require_verified_artifact_metadata

from ._numeric import matrix, positive_int, vector
from .base import BaseRepresentation, RepresentationResult
from .information import discrete_key
from .standard import _spec


class EmpiricalTaskLawRepresentation(BaseRepresentation):
    """Empirical probability mass function for a scalar numeric task outcome.

    Exact discrete conditioning or uniform k-nearest-neighbor conditioning is
    declared in the constructor. The entire training outcome support is kept;
    conditional means and quantiles are separate functionals of this law.
    Unknown exact keys raise. Training and evaluation must be split by caller;
    optional acquisition timestamps reject samples beyond a training cutoff.
    No calibration or population-coverage guarantee is inferred from a fit.
    """

    def __init__(
        self,
        task_id: str,
        *,
        neighbors: int | None = None,
        evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION,
        evidence_metadata: Mapping[str, Any] | None = None,
    ):
        if not task_id:
            raise ValueError("task_id is required")
        if neighbors is not None:
            positive_int(neighbors, "neighbors")
        self.task_id, self.neighbors = task_id, neighbors
        self.evidence_class = EvidenceClass(evidence_class)
        self.evidence_metadata = dict(evidence_metadata or {})
        require_verified_artifact_metadata(self.evidence_class, self.evidence_metadata)
        self.spec = _spec(
            "empirical_conditional_task_law", None, training="training_partition_only", task=task_id
        )
        self._samples: tuple = ()

    def fit(
        self,
        samples: Sequence[Any],
        outcomes: Sequence[float],
        *,
        acquisition_times: Sequence[float] | None = None,
        training_cutoff: float | None = None,
    ) -> "EmpiricalTaskLawRepresentation":
        y = vector(outcomes, len(samples))
        if not y:
            raise ValueError("samples and outcomes must be nonempty")
        if (acquisition_times is None) != (training_cutoff is None):
            raise ValueError("acquisition_times and training_cutoff must be supplied together")
        if acquisition_times is not None:
            times = vector(acquisition_times, len(y))
            if training_cutoff is None:
                raise ValueError("acquisition_times and training_cutoff must be supplied together")
            cutoff = vector((training_cutoff,))[0]
            if any(t > cutoff for t in times):
                raise ValueError("future-information injection: training sample after cutoff")
        if self.neighbors is None:
            x = tuple(discrete_key(v) for v in samples)
        else:
            x = matrix(samples)
            if self.neighbors > len(x):
                raise ValueError("neighbors cannot exceed training sample count")
        self._samples, self._outcomes = x, y
        self.support_ = tuple(sorted(set(y)))
        self._support_index = {v: i for i, v in enumerate(self.support_)}
        self.spec = _spec(
            "empirical_conditional_task_law",
            len(self.support_),
            training="training_partition_only",
            task=self.task_id,
        )
        return self

    def _members(self, history: Any, eligible: Sequence[int] | None = None) -> tuple[int, ...]:
        if not self._samples:
            raise RuntimeError("fit EmpiricalTaskLawRepresentation before prediction")
        indices = tuple(range(len(self._samples))) if eligible is None else tuple(eligible)
        if self.neighbors is None:
            key = discrete_key(history)
            members = tuple(i for i in indices if self._samples[i] == key)
            if not members:
                raise ValueError("unseen conditioning key")
            return members
        x = vector(history, len(self._samples[0]))
        if len(indices) < self.neighbors:
            raise ValueError("not enough eligible neighbor samples")
        ranked = sorted(
            indices, key=lambda i: (math.hypot(*(a - b for a, b in zip(self._samples[i], x))), i)
        )
        return tuple(ranked[: self.neighbors])

    def transform(self, history: Any) -> RepresentationResult:
        members = self._members(history)
        counts = [0] * len(self.support_)
        for i in members:
            counts[self._support_index[self._outcomes[i]]] += 1
        return RepresentationResult(
            tuple(count / len(members) for count in counts),
            self.spec,
            {
                "support": self.support_,
                "estimand": "conditional_probability_mass_function",
                "estimator": "empirical_exact_key"
                if self.neighbors is None
                else "uniform_knn_euclidean",
                "conditioning_count": len(members),
                "evidence_class": self.evidence_class.value,
                "evidence_metadata": self.evidence_metadata,
                "causal_identification": False,
                "calibrated": False,
            },
        )

    def predict_proba(self, history: Any) -> tuple[float, ...]:
        return self.transform(history).values

    def expectation(self, history: Any, function: Callable[[float], float] = float) -> float:
        values = vector(tuple(function(v) for v in self.support_))
        return math.fsum(p * v for p, v in zip(self.predict_proba(history), values))

    def quantile(self, history: Any, level: float) -> float:
        if not math.isfinite(level) or not 0 <= level <= 1:
            raise ValueError("quantile level must lie in [0, 1]")
        cumulative = 0.0
        for value, p in zip(self.support_, self.predict_proba(history)):
            cumulative += p
            if p > 0 and cumulative >= level:
                return value
        return self.support_[-1]

    def interval(self, history: Any, coverage: float = 0.9) -> tuple[float, float]:
        if not math.isfinite(coverage) or not 0 < coverage < 1:
            raise ValueError("coverage must lie in (0, 1)")
        return self.quantile(history, (1 - coverage) / 2), self.quantile(
            history, (1 + coverage) / 2
        )

    def evaluate(
        self,
        samples: Sequence[Any],
        outcomes: Sequence[float],
        *,
        coverage: float = 0.9,
        conditioning_groups: Sequence[Any] | None = None,
        calibration_bins: int = 10,
    ) -> dict:
        """Held-out Brier/log scores and interval coverage by declared groups."""
        positive_int(calibration_bins, "calibration_bins")
        y = vector(outcomes, len(samples))
        if not y or (conditioning_groups is not None and len(conditioning_groups) != len(y)):
            raise ValueError("evaluation arrays must have equal nonzero lengths")
        brier, logs, hits, probabilities = [], [], [], []
        groups = (
            tuple(discrete_key(g) for g in conditioning_groups)
            if conditioning_groups is not None
            else ("all",) * len(y)
        )
        for x, target in zip(samples, y):
            p = self.predict_proba(x)
            probabilities.append(p)
            brier.append(
                math.fsum((prob - float(v == target)) ** 2 for prob, v in zip(p, self.support_))
                + float(target not in self._support_index)
            )
            mass = p[self._support_index[target]] if target in self._support_index else 0.0
            logs.append(-math.log(mass) if mass else math.inf)
            low, high = self.interval(x, coverage)
            hits.append(low <= target <= high)
        grouped = {
            g: math.fsum(float(hits[i]) for i in range(len(y)) if groups[i] == g)
            / sum(v == g for v in groups)
            for g in dict.fromkeys(groups)
        }
        calibration = []
        for j, value in enumerate(self.support_):
            error = 0.0
            for bucket in range(calibration_bins):
                members = tuple(
                    i
                    for i, p in enumerate(probabilities)
                    if min(calibration_bins - 1, int(p[j] * calibration_bins)) == bucket
                )
                if members:
                    confidence = math.fsum(probabilities[i][j] for i in members) / len(members)
                    frequency = sum(y[i] == value for i in members) / len(members)
                    error += len(members) / len(y) * abs(confidence - frequency)
            calibration.append(error)
        return {
            "brier": math.fsum(brier) / len(y),
            "log_score": math.fsum(logs) / len(y),
            "interval_coverage": sum(hits) / len(y),
            "conditional_coverage": grouped,
            "classwise_calibration_error": tuple(calibration),
            "mean_classwise_calibration_error": math.fsum(calibration) / len(calibration),
            "nominal_coverage": coverage,
            "evidence_class": self.evidence_class.value,
            "evidence_metadata": self.evidence_metadata,
            "estimand": "held_out_probabilistic_forecast_scores_and_coverage",
            "population": "supplied_evaluation_samples",
            "unit": "probability/log_nats",
            "uncertainty": "point_estimates",
            "calibration_guarantee": False,
        }

    def bootstrap_expectation(
        self,
        history: Any,
        subjects: Sequence[Any],
        *,
        draws: int = 200,
        seed: int = 0,
        function: Callable[[float], float] = float,
    ) -> dict:
        """Resample whole subjects, preserving within-subject observations.

        Replicates without the query's conditioning support are explicitly None.
        No interval is reported if any replicate is unsupported. This prevents
        silently conditioning a claimed confidence interval on key survival.
        """
        positive_int(draws, "draws")
        self._members(history)
        if len(subjects) != len(self._samples):
            raise ValueError("subject labels must match training samples")
        grouped: dict[Hashable, list[int]] = {}
        for i, subject in enumerate(subjects):
            grouped.setdefault(discrete_key(subject), []).append(i)
        if len(grouped) < 2:
            raise ValueError("cluster bootstrap requires at least two subjects")
        ids = tuple(grouped)
        rng = random.Random(seed)
        replicates: list[float | None] = []
        for _ in range(draws):
            eligible = tuple(i for _ in ids for i in grouped[rng.choice(ids)])
            try:
                members = self._members(history, eligible)
            except ValueError:
                replicates.append(None)
                continue
            values = vector(tuple(function(self._outcomes[i]) for i in members))
            replicates.append(math.fsum(v / len(values) for v in values))
        missing = sum(v is None for v in replicates)
        ordered = sorted(v for v in replicates if v is not None)
        interval = (
            None
            if missing
            else (ordered[int(0.025 * (draws - 1))], ordered[math.ceil(0.975 * (draws - 1))])
        )
        return {
            "replicates": tuple(replicates),
            "percentile_interval_95": interval,
            "unsupported_replicates": missing,
            "method": "whole_subject_cluster_bootstrap",
            "seed": seed,
            "draws": draws,
            "evidence_class": self.evidence_class.value,
            "evidence_metadata": self.evidence_metadata,
            "estimand": "conditional_task_function_expectation",
            "assumptions": ("independent exchangeable subjects", "fixed fitted conditioning rule"),
            "coverage_guarantee": False,
        }

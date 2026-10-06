"""Subject-resampling inference, proper scores, and honest finite-sample intervals."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("finite values and probability in [0, 1] required")
    ordered = sorted(values)
    position = probability * (len(values) - 1)
    low = int(position)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[min(low + 1, len(ordered) - 1)] * fraction


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    interval: tuple[float, float]
    standard_error: float
    independent_units: int
    repetitions: int
    alpha: float
    seed: int
    method: str = "subject-cluster percentile bootstrap"
    assumptions: tuple[str, ...] = ("independent exchangeable subjects",)


def cluster_bootstrap(
    rows: Sequence[T],
    subject_ids: Sequence[str],
    statistic: Callable[[Sequence[T]], float],
    *,
    repetitions: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Resample whole subjects with replacement, retaining every window and duplicate.

    The statistic defines row- versus subject-weighting. Statistics should not deduplicate
    repeated subject IDs. Refitting models inside the statistic includes fit uncertainty.
    """
    if (
        not rows
        or len(rows) != len(subject_ids)
        or repetitions < 2
        or not 0 < alpha < 1
        or any(not s for s in subject_ids)
    ):
        raise ValueError("invalid cluster bootstrap request")
    groups: dict[str, list[T]] = {}
    for row, subject in zip(rows, subject_ids):
        groups.setdefault(subject, []).append(row)
    clusters = list(groups.values())
    if len(clusters) < 2:
        raise ValueError("subject uncertainty requires at least two independent subjects")
    rng = random.Random(seed)
    estimates = [
        statistic([row for _ in clusters for row in clusters[rng.randrange(len(clusters))]])
        for _ in range(repetitions)
    ]
    point = statistic(rows)
    if not math.isfinite(point) or any(not math.isfinite(v) for v in estimates):
        raise ValueError("bootstrap statistic is undefined for a resampled cohort")
    mean = sum(estimates) / repetitions
    se = math.sqrt(sum((v - mean) ** 2 for v in estimates) / (repetitions - 1))
    return BootstrapResult(
        point,
        (quantile(estimates, alpha / 2), quantile(estimates, 1 - alpha / 2)),
        se,
        len(clusters),
        repetitions,
        alpha,
        seed,
    )


def _binary_inputs(labels: Sequence[int], probabilities: Sequence[float]) -> None:
    if (
        not labels
        or len(labels) != len(probabilities)
        or any(y not in (0, 1) for y in labels)
        or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
    ):
        raise ValueError("binary labels and finite probabilities in [0, 1] required")


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    _binary_inputs(labels, probabilities)
    return sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels)


def log_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Mean negative log likelihood; confidently wrong forecasts have infinite loss."""
    _binary_inputs(labels, probabilities)
    terms = [p if y else 1 - p for y, p in zip(labels, probabilities)]
    return math.inf if 0 in terms else -sum(math.log(p) for p in terms) / len(terms)


def gaussian_log_score(
    observations: Sequence[float], means: Sequence[float], standard_deviations: Sequence[float]
) -> float:
    if (
        not observations
        or not len(observations) == len(means) == len(standard_deviations)
        or any(not math.isfinite(v) for v in (*observations, *means, *standard_deviations))
        or any(s <= 0 for s in standard_deviations)
    ):
        raise ValueError("finite aligned Gaussian forecasts with positive scale required")
    return sum(
        math.log(s * math.sqrt(2 * math.pi)) + (y - m) ** 2 / (2 * s * s)
        for y, m, s in zip(observations, means, standard_deviations)
    ) / len(observations)


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_prediction: float | None
    observed_frequency: float | None


def calibration_bins(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 10
) -> tuple[CalibrationBin, ...]:
    """Descriptive reliability bins, not a proper score or a calibration certificate."""
    _binary_inputs(labels, probabilities)
    if bins < 1:
        raise ValueError("bins must be positive")
    grouped: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for y, p in zip(labels, probabilities):
        grouped[min(int(p * bins), bins - 1)].append((y, p))
    return tuple(
        CalibrationBin(
            i / bins,
            (i + 1) / bins,
            len(rows),
            sum(p for _, p in rows) / len(rows) if rows else None,
            sum(y for y, _ in rows) / len(rows) if rows else None,
        )
        for i, rows in enumerate(grouped)
    )


@dataclass(frozen=True)
class SplitConformal:
    radius: float
    alpha: float
    calibration_subject_ids: tuple[str, ...]
    training_subject_ids: tuple[str, ...]
    scope: str = "simultaneous windows of a new exchangeable subject via maximum residual"

    @classmethod
    def fit(
        cls,
        residuals: Sequence[float],
        subject_ids: Sequence[str],
        *,
        training_subject_ids: Sequence[str],
        alpha: float = 0.05,
    ) -> "SplitConformal":
        if (
            not residuals
            or len(residuals) != len(subject_ids)
            or not 0 < alpha < 1
            or not training_subject_ids
            or any(not s for s in subject_ids)
            or any(not math.isfinite(r) for r in residuals)
        ):
            raise ValueError("invalid conformal calibration request")
        if set(subject_ids) & set(training_subject_ids):
            raise ValueError("training/calibration subject leakage")
        scores: dict[str, float] = {}
        for residual, subject in zip(residuals, subject_ids):
            scores[subject] = max(scores.get(subject, 0.0), abs(residual))
        ordered = sorted(scores.values())
        rank = math.ceil((len(ordered) + 1) * (1 - alpha))
        radius = ordered[rank - 1] if rank <= len(ordered) else math.inf
        return cls(radius, alpha, tuple(scores), tuple(training_subject_ids))

    def prediction_interval(self, prediction: float, *, subject_id: str) -> tuple[float, float]:
        if not math.isfinite(prediction) or not subject_id:
            raise ValueError("finite prediction and subject ID required")
        if subject_id in (*self.training_subject_ids, *self.calibration_subject_ids):
            raise ValueError("conformal evaluation subject leakage")
        return prediction - self.radius, prediction + self.radius


@dataclass(frozen=True)
class NormalPosterior:
    mean: float
    variance: float
    observation_variance: float
    assumptions: tuple[str, ...] = ("independent normal observations", "known observation variance")

    def credible_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        return _normal_interval(self.mean, self.variance, alpha)

    def prediction_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        return _normal_interval(self.mean, self.variance + self.observation_variance, alpha)


def _normal_interval(mean: float, variance: float, alpha: float) -> tuple[float, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    radius = NormalDist().inv_cdf(1 - alpha / 2) * math.sqrt(variance)
    return mean - radius, mean + radius


def normal_normal_posterior(
    values: Sequence[float],
    *,
    prior_mean: float,
    prior_variance: float,
    observation_variance: float,
) -> NormalPosterior:
    if (
        not values
        or any(
            not math.isfinite(v)
            for v in (*values, prior_mean, prior_variance, observation_variance)
        )
        or prior_variance <= 0
        or observation_variance <= 0
    ):
        raise ValueError("finite observations and positive known variances required")
    precision = 1 / prior_variance + len(values) / observation_variance
    mean = (prior_mean / prior_variance + sum(values) / observation_variance) / precision
    return NormalPosterior(mean, 1 / precision, observation_variance)


def propagate_samples(
    parameter_samples: Sequence[T], model: Callable[[T], float], *, alpha: float = 0.05
) -> tuple[float, float]:
    """Push forward caller-supplied samples; a quantile range, not automatically a CI."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    values = [model(sample) for sample in parameter_samples]
    return quantile(values, alpha / 2), quantile(values, 1 - alpha / 2)

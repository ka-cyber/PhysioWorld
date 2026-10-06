"""Dependency-free signal summaries and genuine dense linear representations."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from ._numeric import dot, matrix, positive_int, symmetric_eigen, vector
from .base import BaseRepresentation, RepresentationResult, RepresentationSpec


def _spec(
    name: str,
    dimension: int | None,
    history: str = "provided",
    training: str = "none",
    task: str = "none",
) -> RepresentationSpec:
    return RepresentationSpec(
        name,
        "observations",
        history,
        dimension,
        training,
        task,
        "predictive_only; nonanticipative_given_input",
        "method-dependent",
        "method-dependent",
    )


@dataclass(frozen=True)
class RawSignalRepresentation(BaseRepresentation[Sequence[float]]):
    name: str = "raw_signal"
    spec: RepresentationSpec = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec", _spec(self.name, None))

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        return RepresentationResult(vector(history), self.spec)


@dataclass(frozen=True)
class InstantaneousStateRepresentation(BaseRepresentation[Sequence[float]]):
    """Latest scalar observation, without an implicit history or imputation."""

    spec: RepresentationSpec = field(default_factory=lambda: _spec("instantaneous_state", 1))

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        if not len(history):
            raise ValueError("instantaneous state requires an observation")
        return RepresentationResult(vector(history[-1:]), self.spec)


@dataclass(frozen=True)
class FiniteHistoryRepresentation(BaseRepresentation[Sequence[float]]):
    window: int
    pad_value: float = 0.0
    name: str = "finite_history"
    spec: RepresentationSpec = field(init=False)

    def __post_init__(self) -> None:
        positive_int(self.window, "window")
        vector((self.pad_value,))
        object.__setattr__(self, "spec", _spec(self.name, self.window, f"last_{self.window}"))

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        values = vector(history[-self.window :])
        padded = self.window - len(values)
        return RepresentationResult(
            (float(self.pad_value),) * padded + values,
            self.spec,
            {"padded_samples": padded, "pad_value": self.pad_value},
        )


class PredictiveHistoryRepresentation(FiniteHistoryRepresentation):
    """Migration target for the former PredictiveStateProxy; a history, not a PSR."""

    def __init__(self, dimension: int, pad_value: float = 0.0):
        super().__init__(dimension, pad_value, "predictive_history")


@dataclass(frozen=True)
class MultiscaleHistoryRepresentation(BaseRepresentation[Sequence[float]]):
    windows: tuple[int, ...]
    name: str = "multiscale_history"
    statistics: tuple[str, ...] = ("mean",)
    spec: RepresentationSpec = field(init=False)

    def __post_init__(self) -> None:
        if not self.windows or not self.statistics:
            raise ValueError("windows and statistics must be nonempty")
        for w in self.windows:
            positive_int(w, "window")
        if len(set(self.windows)) != len(self.windows):
            raise ValueError("windows must be distinct")
        if any(s not in {"mean", "std", "trend", "last"} for s in self.statistics):
            raise ValueError("supported statistics: mean, std, trend, last")
        object.__setattr__(
            self,
            "spec",
            _spec(self.name, len(self.windows) * len(self.statistics), str(self.windows)),
        )

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        x = vector(history)
        if not x:
            raise ValueError("multiscale summaries require observations")
        output: list[float] = []
        counts: list[int] = []
        for w in self.windows:
            chunk = x[-w:]
            n = len(chunk)
            mean = math.fsum(v / n for v in chunk)
            centered = tuple(v - mean for v in chunk)
            index = tuple(i - (n - 1) / 2 for i in range(n))
            denom = dot(index, index)
            stats = {
                "mean": mean,
                "std": math.sqrt(dot(centered, centered) / n),
                "last": chunk[-1],
                "trend": dot(index, centered) / denom if denom else 0.0,
            }
            output.extend(stats[s] for s in self.statistics)
            counts.append(n)
        return RepresentationResult(
            tuple(output),
            self.spec,
            {
                "sample_counts": tuple(counts),
                "statistics": self.statistics,
                "trend_unit": "observation_unit/sample_index",
            },
        )


@dataclass(frozen=True)
class HandcraftedPhysiologyRepresentation(BaseRepresentation[Sequence[float]]):
    """Generic statistics, not a validated physiological biomarker estimator."""

    name: str = "handcrafted_physiology"
    spec: RepresentationSpec = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec", _spec(self.name, 6))

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        x = vector(history)
        if not x:
            raise ValueError("statistical summaries require observations")
        mean = math.fsum(v / len(x) for v in x)
        std = math.hypot(*(v - mean for v in x)) / math.sqrt(len(x))
        rms = math.hypot(*x) / math.sqrt(len(x))
        crossings = sum((a < 0 < b) or (b < 0 < a) for a, b in zip(x, x[1:]))
        return RepresentationResult(
            (mean, std, rms, min(x), max(x), float(crossings)),
            self.spec,
            {"variance_denominator": len(x), "evidence_class": "DEFINITION"},
        )


class PCARepresentation(BaseRepresentation[Sequence[float]]):
    """Centered sample-covariance PCA solved with cyclic Jacobi rotations.

    ``components`` is a rank cap; ``variance_threshold`` chooses the smallest
    rank meeting a cumulative variance fraction within that cap. Whitening
    uses sample variance (n-1). Zero-variance components cannot be whitened.
    The uncentered option is explicitly a second-moment decomposition.
    """

    def __init__(
        self,
        components: int | None = None,
        *,
        center: bool = True,
        whiten: bool = False,
        variance_threshold: float | None = None,
        tolerance: float = 1e-13,
        max_sweeps: int = 100,
        rank_tolerance: float = 1e-12,
    ):
        if components is not None:
            positive_int(components, "components")
        if variance_threshold is not None and not (0 < variance_threshold <= 1):
            raise ValueError("variance_threshold must lie in (0, 1]")
        if not math.isfinite(rank_tolerance) or rank_tolerance <= 0:
            raise ValueError("rank_tolerance must be finite and positive")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        positive_int(max_sweeps, "max_sweeps")
        self.components = components
        self.center, self.whiten = center, whiten
        self.variance_threshold = variance_threshold
        self.tolerance, self.max_sweeps = tolerance, max_sweeps
        self.rank_tolerance = rank_tolerance
        self.mean: tuple[float, ...] | None = None
        self.components_: tuple[tuple[float, ...], ...] = ()
        self.explained_variance_: tuple[float, ...] = ()
        self.explained_variance_ratio_: tuple[float, ...] = ()
        self.eigenvalues_: tuple[float, ...] = ()
        self.rank_ = 0
        self.n_components_ = 0
        self.spec = _spec("pca", components, training="fit_training_partition_only")

    def fit(self, samples: Sequence[Sequence[float]]) -> "PCARepresentation":
        x = matrix(samples, min_rows=2)
        n, d = len(x), len(x[0])
        cap = self.components or d
        if cap > d:
            raise ValueError("components cannot exceed input dimension")
        # Offset centering retains small variation on a large common baseline.
        mean = (
            tuple(x[0][j] + math.fsum((row[j] - x[0][j]) / n for row in x) for j in range(d))
            if self.center
            else (0.0,) * d
        )
        centered = tuple(tuple(row[j] - mean[j] for j in range(d)) for row in x)
        scale = max(abs(v) for row in centered for v in row)
        scaled = tuple(tuple(v / scale for v in row) for row in centered) if scale else centered
        denom = n - 1 if self.center else n
        covariance = tuple(
            tuple(math.fsum(row[i] * row[j] for row in scaled) / denom for j in range(d))
            for i in range(d)
        )
        eig, basis = symmetric_eigen(covariance, self.tolerance, self.max_sweeps)
        cutoff = max(eig[0], 0.0) * self.rank_tolerance
        if min(eig) < -max(eig[0], 1e-300) * max(self.tolerance * d, self.rank_tolerance):
            raise ArithmeticError("computed covariance is not positive semidefinite")
        rank = sum(v > cutoff for v in eig)
        total = math.fsum(max(v, 0.0) for v in eig)
        ratios = tuple(max(v, 0.0) / total if total else 0.0 for v in eig)
        k = cap
        if self.variance_threshold is not None:
            if not total:
                raise ValueError("variance rank selection requires nonzero variance")
            k = next(
                (
                    i + 1
                    for i in range(d)
                    if math.fsum(ratios[: i + 1]) >= self.variance_threshold - 1e-14
                ),
                d,
            )
            if k > cap:
                raise ValueError("component cap cannot meet the variance threshold")
        if self.whiten and k > rank:
            raise ValueError("cannot whiten zero-variance components; select rank_ components")
        variances = tuple((max(v, 0.0) * scale) * scale for v in eig)
        if any(not math.isfinite(v) for v in variances):
            raise ValueError("covariance exceeds floating-point range; rescale inputs explicitly")
        if any(v > cutoff and var == 0 for v, var in zip(eig, variances)):
            raise ValueError("covariance underflow; rescale inputs explicitly")
        self.mean, self.components_ = mean, basis[:k]
        self.eigenvalues_ = variances
        self.explained_variance_ = variances[:k]
        self.explained_variance_ratio_ = ratios[:k]
        self.rank_, self.n_components_, self.n_samples_ = rank, k, n
        self.spec = _spec("pca", k, training="fit_training_partition_only")
        return self

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        if self.mean is None:
            raise RuntimeError("fit PCARepresentation before transform")
        x = vector(history, len(self.mean))
        values = tuple(
            dot(component, tuple(v - m for v, m in zip(x, self.mean)))
            for component in self.components_
        )
        if self.whiten:
            values = tuple(v / math.sqrt(var) for v, var in zip(values, self.explained_variance_))
        return RepresentationResult(
            vector(values),
            self.spec,
            {
                "center": self.center,
                "whiten": self.whiten,
                "rank": self.rank_,
                "explained_variance": self.explained_variance_,
                "solver": "cyclic_jacobi",
                "evidence_class": "NUMERICAL_VERIFICATION",
                "fit_samples": self.n_samples_,
            },
        )

    def inverse_transform(
        self, latent: Sequence[float] | RepresentationResult
    ) -> tuple[float, ...]:
        if self.mean is None:
            raise RuntimeError("fit PCARepresentation before inverse_transform")
        z = vector(
            latent.values if isinstance(latent, RepresentationResult) else latent,
            self.n_components_,
        )
        if self.whiten:
            z = tuple(v * math.sqrt(var) for v, var in zip(z, self.explained_variance_))
        return vector(
            tuple(
                m + math.fsum(z[j] * self.components_[j][i] for j in range(self.n_components_))
                for i, m in enumerate(self.mean)
            )
        )


@dataclass
class RandomProjectionRepresentation(BaseRepresentation[Sequence[float]]):
    dimension: int
    input_dimension: int
    seed: int = 0
    name: str = "random_projection"
    matrix: tuple[tuple[float, ...], ...] = field(init=False)
    spec: RepresentationSpec = field(init=False)

    def __post_init__(self) -> None:
        positive_int(self.dimension, "dimension")
        positive_int(self.input_dimension, "input_dimension")
        self.spec = _spec(self.name, self.dimension, training="seeded_random_matrix")
        rng = random.Random(self.seed)
        scale = 1.0 / math.sqrt(self.dimension)
        self.matrix = tuple(
            tuple(rng.gauss(0.0, scale) for _ in range(self.input_dimension))
            for _ in range(self.dimension)
        )

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        x = vector(history, self.input_dimension)
        return RepresentationResult(
            vector(tuple(dot(row, x) for row in self.matrix)),
            self.spec,
            {"seed": self.seed, "distribution": "N(0, 1/dimension)"},
        )


@dataclass(frozen=True)
class TaskSpecificRepresentation(BaseRepresentation[Sequence[float]]):
    """Explicit task-selected features, not a learned or sufficient-state claim."""

    task_id: str
    feature_indices: tuple[int, ...]
    name: str = "task_specific"
    spec: RepresentationSpec = field(init=False)

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or not self.feature_indices
            or any(
                isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.feature_indices
            )
            or len(set(self.feature_indices)) != len(self.feature_indices)
        ):
            raise ValueError("task_id and distinct nonnegative feature indices are required")
        object.__setattr__(
            self,
            "spec",
            _spec(f"{self.name}:{self.task_id}", len(self.feature_indices), task=self.task_id),
        )

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        x = vector(history)
        if max(self.feature_indices) >= len(x):
            raise ValueError("selected feature is unavailable")
        return RepresentationResult(
            tuple(x[i] for i in self.feature_indices),
            self.spec,
            {"selection": "declared_indices", "sufficiency_claim": False},
        )

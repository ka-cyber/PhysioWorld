"""Fitted stochastic latent and joint multimodal representations."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from ._numeric import dot, matrix, positive_int, vector
from .base import BaseRepresentation, RepresentationResult
from .information import InformationBottleneckRepresentation, SupervisedBottleneckRepresentation
from .predictive import PredictiveStateRepresentation
from .standard import PCARepresentation, PredictiveHistoryRepresentation, _spec


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    e = math.exp(logit)
    return e / (1.0 + e)


def _binary_loss(logit: float, label: int) -> float:
    return max(logit, 0.0) - label * logit + math.log1p(math.exp(-abs(logit)))


class VariationalInformationBottleneck(BaseRepresentation[Sequence[float]]):
    """Supervised Gaussian VIB with a stochastic, unit-covariance linear encoder.

    q(z|x)=N(Wx+b,I), r(z)=N(0,I), p(y=1|z)=sigmoid(v'z+c).
    Minimize E_q[BCE(y,p(y|z))] + beta KL(q(z|x)||r(z)), averaged
    over training examples. Reparameterized Monte Carlo gradients optimize the
    expected stochastic reconstruction loss, not the loss at the encoder mean.
    Unit covariance is an explicit restricted variational family; the mean KL
    is an upper bound on I(X;Z), not an MI estimate. CPU reference optimizer.
    """

    def __init__(
        self,
        input_dimension: int,
        latent_dimension: int = 2,
        beta: float = 1e-3,
        learning_rate: float = 0.05,
        epochs: int = 100,
        seed: int = 0,
        *,
        samples_per_observation: int = 16,
        tolerance: float = 1e-6,
    ):
        for name, value in (
            ("input_dimension", input_dimension),
            ("latent_dimension", latent_dimension),
            ("epochs", epochs),
            ("samples_per_observation", samples_per_observation),
        ):
            positive_int(value, name)
        if (
            not math.isfinite(beta)
            or beta < 0
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("beta must be finite/nonnegative and learning_rate finite/positive")
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite/positive")
        self.input_dimension, self.latent_dimension = input_dimension, latent_dimension
        self.beta, self.learning_rate, self.epochs, self.seed = beta, learning_rate, epochs, seed
        self.samples_per_observation, self.tolerance = samples_per_observation, tolerance
        self.spec = _spec(
            "variational_information_bottleneck",
            latent_dimension,
            training="fit_training_partition_only",
            task="binary_supervised",
        )
        self._weights: list[list[float]] | None = None
        self.training_loss: tuple[float, ...] = ()
        self.converged_ = False

    def _mean(self, row: Sequence[float]) -> tuple[float, ...]:
        if self._weights is None:
            raise RuntimeError("fit VariationalInformationBottleneck before transform")
        x = vector(row, self.input_dimension)
        return vector(tuple(b + dot(w, x) for b, w in zip(self._bias, self._weights)))

    def _objective(self, x: Sequence[Sequence[float]], y: Sequence[int], noise: Sequence) -> float:
        losses = []
        for row, target, draws in zip(x, y, noise):
            mu = self._mean(row)
            bce = math.fsum(
                _binary_loss(
                    self._classifier_bias
                    + dot(self._classifier, tuple(m + e for m, e in zip(mu, eps))),
                    target,
                )
                for eps in draws
            ) / len(draws)
            losses.append(bce + 0.5 * self.beta * dot(mu, mu))
        value = math.fsum(losses) / len(x)
        if not math.isfinite(value):
            raise ArithmeticError("VIB objective diverged; rescale inputs or reduce learning_rate")
        return value

    def fit(
        self, samples: Sequence[Sequence[float]], labels: Sequence[int]
    ) -> "VariationalInformationBottleneck":
        x = matrix(samples)
        if (
            len(x) != len(labels)
            or len(x[0]) != self.input_dimension
            or any(y not in (0, 1) for y in labels)
        ):
            raise ValueError("matching sample dimensions and binary labels are required")
        rng = random.Random(self.seed)
        self._weights = [
            [rng.gauss(0.0, 0.05) for _ in range(self.input_dimension)]
            for _ in range(self.latent_dimension)
        ]
        self._bias = [0.0] * self.latent_dimension
        self._classifier = [rng.gauss(0.0, 0.05) for _ in range(self.latent_dimension)]
        self._classifier_bias = 0.0
        eval_rng = random.Random(self.seed + 1_000_003)
        evaluation_noise = tuple(
            tuple(
                tuple(eval_rng.gauss(0, 1) for _ in range(self.latent_dimension))
                for _ in range(self.samples_per_observation)
            )
            for _ in x
        )
        history = [self._objective(x, labels, evaluation_noise)]
        self.converged_ = False
        for epoch in range(1, self.epochs + 1):
            gw = [[0.0] * self.input_dimension for _ in range(self.latent_dimension)]
            gb, gv, gc = [0.0] * self.latent_dimension, [0.0] * self.latent_dimension, 0.0
            for row, label in zip(x, labels):
                mu = self._mean(row)
                dmu = [self.beta * m for m in mu]
                for _ in range(self.samples_per_observation):
                    z = tuple(m + rng.gauss(0, 1) for m in mu)
                    logit = self._classifier_bias + dot(self._classifier, z)
                    if not math.isfinite(logit):
                        raise ArithmeticError("VIB gradient diverged; rescale inputs")
                    error = (_sigmoid(logit) - label) / self.samples_per_observation
                    gc += error
                    for j in range(self.latent_dimension):
                        gv[j] += error * z[j]
                        dmu[j] += error * self._classifier[j]
                for j in range(self.latent_dimension):
                    gb[j] += dmu[j]
                    for i in range(self.input_dimension):
                        gw[j][i] += dmu[j] * row[i]
            scale = 1.0 / len(x)
            all_gradients = tuple(g * scale for row in gw for g in row) + tuple(
                g * scale for g in gb + gv + [gc]
            )
            vector(all_gradients)
            self.gradient_norm_ = math.hypot(*all_gradients)
            step = self.learning_rate * scale
            for j in range(self.latent_dimension):
                for i in range(self.input_dimension):
                    self._weights[j][i] -= step * gw[j][i]
                self._bias[j] -= step * gb[j]
                self._classifier[j] -= step * gv[j]
            self._classifier_bias -= step * gc
            history.append(self._objective(x, labels, evaluation_noise))
            if self.gradient_norm_ <= self.tolerance:
                self.converged_ = True
                break
        self.n_iter_, self.training_loss = epoch, tuple(history)
        self.rate_upper_bound_ = math.fsum(
            0.5 * dot(self._mean(row), self._mean(row)) for row in x
        ) / len(x)
        self.metadata_ = {
            "objective": "E_q[binary_cross_entropy] + beta * KL(q(z|x)||N(0,I))",
            "objective_value": history[-1],
            "estimator": "reparameterized_monte_carlo",
            "optimization": "full_batch_stochastic_gradient_descent",
            "beta": self.beta,
            "regularization": "analytic_gaussian_KL",
            "stochasticity": "unit_covariance_gaussian",
            "rate_upper_bound_nats": self.rate_upper_bound_,
            "mutual_information_estimate": False,
            "converged": self.converged_,
            "iterations": epoch,
            "gradient_norm": self.gradient_norm_,
            "convergence_criterion": "monte_carlo_gradient_norm",
            "tolerance": self.tolerance,
            "seed": self.seed,
            "samples_per_observation": self.samples_per_observation,
            "evidence_class": "NUMERICAL_VERIFICATION",
            "causal_identification": False,
        }
        return self

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        mu = self._mean(history)
        return RepresentationResult(
            mu,
            self.spec,
            {
                **self.metadata_,
                "returned_statistic": "encoder_mean",
                "encoder_variance": (1.0,) * len(mu),
            },
        )

    def sample(self, history: Sequence[float], *, seed: int) -> RepresentationResult:
        mu, rng = self._mean(history), random.Random(seed)
        return RepresentationResult(
            tuple(m + rng.gauss(0, 1) for m in mu),
            self.spec,
            {**self.metadata_, "returned_statistic": "encoder_sample", "sample_seed": seed},
        )

    def predict_proba(
        self, history: Sequence[float], *, draws: int = 256, seed: int = 0
    ) -> tuple[float, float]:
        positive_int(draws, "draws")
        mu, rng = self._mean(history), random.Random(seed)
        p = (
            math.fsum(
                _sigmoid(
                    self._classifier_bias
                    + dot(self._classifier, tuple(m + rng.gauss(0, 1) for m in mu))
                )
                for _ in range(draws)
            )
            / draws
        )
        return 1.0 - p, p


class LearnedLatentRepresentation(BaseRepresentation[Sequence[float]]):
    """Learned linear least-squares latent state (PCA reconstruction objective).

    A distinct fitted model, not a feature alias or a claim of dynamic state
    identification. Learns the optimal rank-k centered orthogonal linear
    reconstruction subspace on the supplied training observations.
    """

    def __init__(self, input_dimension: int, latent_dimension: int = 2):
        positive_int(input_dimension, "input_dimension")
        positive_int(latent_dimension, "latent_dimension")
        if latent_dimension > input_dimension:
            raise ValueError("latent dimension cannot exceed input dimension")
        self.input_dimension, self.latent_dimension = input_dimension, latent_dimension
        self.encoder_ = PCARepresentation(latent_dimension)
        self.spec = _spec(
            "learned_linear_latent", latent_dimension, training="fit_reconstruction_loss"
        )

    def fit(self, samples: Sequence[Sequence[float]]) -> "LearnedLatentRepresentation":
        x = matrix(samples, min_rows=2)
        if len(x[0]) != self.input_dimension:
            raise ValueError("input dimension mismatch")
        self.encoder_.fit(x)
        residuals = tuple(
            tuple(
                a - b
                for a, b in zip(row, self.encoder_.inverse_transform(self.encoder_.transform(row)))
            )
            for row in x
        )
        self.reconstruction_loss_ = math.fsum(dot(row, row) / len(x) for row in residuals)
        return self

    def transform(self, history: Sequence[float]) -> RepresentationResult:
        result = self.encoder_.transform(history)
        return RepresentationResult(
            result.values,
            self.spec,
            {
                **result.metadata,
                "objective": "mean_squared_euclidean_reconstruction",
                "objective_value": self.reconstruction_loss_,
                "optimization": "covariance_eigendecomposition",
                "dynamic_state_identification": False,
            },
        )

    def inverse_transform(
        self, latent: Sequence[float] | RepresentationResult
    ) -> tuple[float, ...]:
        return self.encoder_.inverse_transform(latent)


class SharedMultimodalRepresentation(BaseRepresentation[Mapping[str, Sequence[float]]]):
    """Fitted joint-PCA latent over aligned, optionally standardized modalities.

    Shared means a single jointly fitted latent space and joint reconstruction;
    it does not assert that every factor is common to all modalities (CCA or a
    common-cause model). Each row is an explicitly aligned mapping of modality
    names to numeric vectors. No alignment, channel imputation or resampling is
    inferred. Inverse transform returns each modality in its original units.
    """

    def __init__(self, modalities: tuple[str, ...], dimension: int, *, standardize: bool = True):
        positive_int(dimension, "dimension")
        if (
            not modalities
            or any(not m for m in modalities)
            or len(set(modalities)) != len(modalities)
        ):
            raise ValueError("distinct nonempty modality names required")
        self.modalities, self.dimension, self.standardize = (
            tuple(modalities),
            dimension,
            standardize,
        )
        self.encoder_ = PCARepresentation(dimension)
        self.spec = _spec(
            "shared_multimodal_joint_pca", dimension, training="fit_aligned_joint_covariance"
        )
        self.widths_: tuple[int, ...] = ()

    def _flatten(self, sample: Mapping[str, Sequence[float]], *, fitted: bool) -> tuple[float, ...]:
        if not isinstance(sample, Mapping) or set(sample) != set(self.modalities):
            raise ValueError("channel mismatch: all declared modality names must match exactly")
        chunks = tuple(vector(sample[m]) for m in self.modalities)
        widths = tuple(len(c) for c in chunks)
        if any(w == 0 for w in widths) or (fitted and widths != self.widths_):
            raise ValueError("modality feature dimensions are invalid")
        return tuple(v for chunk in chunks for v in chunk)

    def fit(
        self, samples: Sequence[Mapping[str, Sequence[float]]]
    ) -> "SharedMultimodalRepresentation":
        x = matrix(tuple(self._flatten(sample, fitted=False) for sample in samples), min_rows=2)
        widths = tuple(len(samples[0][m]) for m in self.modalities)
        if any(tuple(len(sample[m]) for m in self.modalities) != widths for sample in samples):
            raise ValueError("modality dimensions must be stable across training samples")
        n, d = len(x), len(x[0])
        means = tuple(x[0][j] + math.fsum((row[j] - x[0][j]) / n for row in x) for j in range(d))
        scales = (
            tuple(
                math.sqrt(math.fsum((row[j] - means[j]) ** 2 / (n - 1) for row in x))
                for j in range(d)
            )
            if self.standardize
            else (1.0,) * d
        )
        scales = tuple(s if s > 0 else 1.0 for s in scales)
        normalized = tuple(tuple((v - m) / s for v, m, s in zip(row, means, scales)) for row in x)
        self.encoder_.fit(normalized)
        self.widths_, self.means_, self.scales_ = widths, means, scales
        return self

    def transform(self, history: Mapping[str, Sequence[float]]) -> RepresentationResult:
        if not self.widths_:
            raise RuntimeError("fit SharedMultimodalRepresentation before transform")
        x = self._flatten(history, fitted=True)
        result = self.encoder_.transform(
            tuple((v - m) / s for v, m, s in zip(x, self.means_, self.scales_))
        )
        return RepresentationResult(
            result.values,
            self.spec,
            {
                **result.metadata,
                "modalities": self.modalities,
                "standardize": self.standardize,
                "alignment": "caller_supplied",
                "shared_factor_identification": False,
            },
        )

    def inverse_transform(
        self, latent: Sequence[float] | RepresentationResult
    ) -> dict[str, tuple[float, ...]]:
        if not self.widths_:
            raise RuntimeError("fit SharedMultimodalRepresentation before inverse_transform")
        normalized = self.encoder_.inverse_transform(latent)
        flat = tuple(v * s + m for v, s, m in zip(normalized, self.scales_, self.means_))
        result, offset = {}, 0
        for modality, width in zip(self.modalities, self.widths_):
            result[modality] = flat[offset : offset + width]
            offset += width
        return result


__all__ = [
    "InformationBottleneckRepresentation",
    "SupervisedBottleneckRepresentation",
    "PredictiveStateRepresentation",
    "PredictiveHistoryRepresentation",
    "SharedMultimodalRepresentation",
    "VariationalInformationBottleneck",
    "LearnedLatentRepresentation",
]

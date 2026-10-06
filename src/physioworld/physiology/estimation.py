"""Validated state-space estimation with optional NumPy.

Kalman uses Joseph covariance updates; EKF uses declared analytic Jacobians;
RTS is a fixed-interval smoother and therefore explicitly uses future data.
Bootstrap particle filtering uses stable log weights and systematic resampling.
Noise matrices are covariances, never standard deviations. No clinical or
mechanistic identifiability is claimed by a successful numerical fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


def _np():
    from physioworld.io.optional import require

    return require("numpy", "state estimation", "scientific")


def _vector(value, name, size=None):
    np = _np()
    array = np.asarray(value, dtype=float)
    if (
        array.ndim != 1
        or (size is not None and array.shape != (size,))
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite vector" + (f" of length {size}" if size else ""))
    return array.copy()


def _matrix(value, name, shape):
    np = _np()
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix of shape {shape}")
    return array.copy()


def _covariance(value, name, size, *, positive=False):
    np = _np()
    array = _matrix(value, name, (size, size))
    if not np.allclose(array, array.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(array)
    if eigenvalues.min() < -1e-10 * max(1.0, np.linalg.norm(array)):
        raise ValueError(f"{name} must be positive semidefinite")
    if positive and eigenvalues.min() <= 0:
        raise ValueError(f"{name} must be positive definite")
    return (array + array.T) / 2


@dataclass(frozen=True)
class EstimatorMetadata:
    process_model: str
    measurement_model: str
    prior: Mapping[str, Any]
    assumptions: tuple[str, ...]
    uncertainty: str
    calibration: Mapping[str, Any] = field(default_factory=dict)
    observability: Mapping[str, Any] = field(default_factory=lambda: {"status": "not_established"})

    def __post_init__(self):
        if (
            not self.process_model
            or not self.measurement_model
            or not self.prior
            or not self.assumptions
            or not self.uncertainty
        ):
            raise ValueError(
                "estimator metadata must declare models, prior, assumptions, and uncertainty"
            )


@dataclass(frozen=True)
class GaussianState:
    mean: Any
    covariance: Any

    def __post_init__(self):
        mean = _vector(self.mean, "state mean")
        if not len(mean):
            raise ValueError("state dimension must be positive")
        covariance = _covariance(self.covariance, "state covariance", len(mean))
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "covariance": self.covariance.tolist()}


@dataclass(frozen=True)
class FilterResult:
    filtered: tuple[GaussianState, ...]
    predicted: tuple[GaussianState, ...]
    transitions: tuple[Any, ...]
    log_likelihood: float
    metadata: EstimatorMetadata
    timestamps: tuple[float, ...] | None = None


def observability_matrix(transition, observation):
    np = _np()
    f = np.asarray(transition, dtype=float)
    if f.ndim != 2 or f.shape[0] != f.shape[1] or f.shape[0] == 0:
        raise ValueError("transition must be nonempty square")
    f = _matrix(f, "transition", f.shape)
    h = np.asarray(observation, dtype=float)
    if h.ndim != 2 or h.shape[0] == 0:
        raise ValueError("observation must be a nonempty matrix")
    h = _matrix(h, "observation", (h.shape[0], f.shape[0]))
    matrix = np.vstack([h @ np.linalg.matrix_power(f, k) for k in range(len(f))])
    rank = int(np.linalg.matrix_rank(matrix))
    return {
        "matrix": matrix,
        "rank": rank,
        "state_dimension": len(f),
        "observable": rank == len(f),
        "scope": "linear time-invariant state observability; not parameter identifiability",
    }


class _GaussianFilterBase:
    """Shared validated Gaussian update machinery for linear and nonlinear filters."""

    def __init__(
        self,
        transition,
        observation,
        process_covariance,
        measurement_covariance,
        prior: GaussianState,
        *,
        metadata: EstimatorMetadata | None = None,
    ):
        np = _np()
        n = len(prior.mean)
        h = np.asarray(observation, dtype=float)
        if h.ndim != 2 or not h.shape[0]:
            raise ValueError("measurement model must be a nonempty matrix")
        self.transition = _matrix(transition, "transition", (n, n))
        self.observation = _matrix(h, "observation", (h.shape[0], n))
        self.process_covariance = _covariance(process_covariance, "process covariance", n)
        self.measurement_covariance = _covariance(
            measurement_covariance, "measurement covariance", h.shape[0], positive=True
        )
        self.state = prior
        diag = observability_matrix(self.transition, self.observation)
        self.metadata = metadata or EstimatorMetadata(
            "x'=F x + N(0,Q)",
            "z=H x + N(0,R)",
            prior.to_dict(),
            (
                "linear dynamics",
                "independent Gaussian process/measurement noise",
                "discrete declared steps",
            ),
            "posterior covariance",
            observability={k: v for k, v in diag.items() if k != "matrix"},
        )
        self.last_log_likelihood = 0.0

    def _update(self, measurement, expected, h, r, missing=None):
        np = _np()
        m = h.shape[0]
        if measurement is None:
            self.last_log_likelihood = 0.0
            return self.state
        z = np.asarray(measurement, dtype=float)
        if z.shape != (m,) or np.any(np.isinf(z)):
            raise ValueError(
                f"measurement must have length {m}, with finite values or missing NaNs"
            )
        mask = ~np.isnan(z)
        if missing is not None:
            gaps = np.asarray(missing)
            if gaps.shape != (m,) or gaps.dtype.kind != "b":
                raise ValueError("measurement missing mask must be a boolean vector")
            mask &= ~gaps
        if not np.any(mask):
            self.last_log_likelihood = 0.0
            return self.state
        h, r = h[mask, :], r[np.ix_(mask, mask)]
        innovation = z[mask] - expected[mask]
        p = self.state.covariance
        s = h @ p @ h.T + r
        gain = np.linalg.solve(s, h @ p).T
        identity = np.eye(len(p)) - gain @ h
        covariance = identity @ p @ identity.T + gain @ r @ gain.T
        self.state = GaussianState(self.state.mean + gain @ innovation, covariance)
        _, logdet = np.linalg.slogdet(s)
        self.last_log_likelihood = float(
            -0.5
            * (
                len(innovation) * math.log(2 * math.pi)
                + logdet
                + innovation @ np.linalg.solve(s, innovation)
            )
        )
        return self.state

    def update_fused(self, measurements, observation_matrices, joint_covariance) -> GaussianState:
        """Joint multimodal update; full covariance retains cross-sensor correlations."""
        np = _np()
        if not measurements or len(measurements) != len(observation_matrices):
            raise ValueError("fusion requires aligned measurements and observation matrices")
        h = np.vstack(observation_matrices)
        z = np.concatenate([np.asarray(m, dtype=float) for m in measurements])
        h = _matrix(h, "fused observation", (len(z), len(self.state.mean)))
        r = _covariance(joint_covariance, "fused measurement covariance", len(z), positive=True)
        return self._update(z, h @ self.state.mean, h, r)


class KalmanFilter(_GaussianFilterBase):
    """Discrete linear Gaussian model x'=F x+w, z=H x+v.

    ``filter`` predicts then updates for each observation (including the first).
    Missing ``None`` or NaN components skip only those measurement components.
    Times, when supplied, are explicit seconds; fixed F/Q still represent one
    discrete transition per sample, not an inferred continuous-time model.
    """

    def predict(self, *, transition=None, process_covariance=None, control=None) -> GaussianState:
        n = len(self.state.mean)
        f = self.transition if transition is None else _matrix(transition, "transition", (n, n))
        q = (
            self.process_covariance
            if process_covariance is None
            else _covariance(process_covariance, "process covariance", n)
        )
        mean = f @ self.state.mean
        if control is not None:
            mean = mean + _vector(control, "state-space control increment", n)
        self.state = GaussianState(mean, f @ self.state.covariance @ f.T + q)
        return self.state

    def update(
        self, measurement, *, observation=None, measurement_covariance=None, missing=None
    ) -> GaussianState:
        if observation is None:
            h = self.observation
        else:
            shape = _np().asarray(observation).shape
            if len(shape) != 2 or not shape[0]:
                raise ValueError("observation must be a nonempty matrix")
            h = _matrix(observation, "observation", (shape[0], len(self.state.mean)))
        r = (
            self.measurement_covariance
            if measurement_covariance is None
            else _covariance(
                measurement_covariance, "measurement covariance", h.shape[0], positive=True
            )
        )
        if r.shape != (h.shape[0], h.shape[0]):
            raise ValueError("measurement covariance and observation dimension mismatch")
        return self._update(measurement, h @ self.state.mean, h, r, missing)

    def filter(
        self, measurements: Sequence[Any], *, timestamps: Sequence[float] | None = None
    ) -> FilterResult:
        if timestamps is not None:
            if (
                len(timestamps) != len(measurements)
                or any(not math.isfinite(t) for t in timestamps)
                or any(b <= a for a, b in zip(timestamps, timestamps[1:]))
            ):
                raise ValueError(
                    "filter timestamps must be finite, strictly increasing, and aligned"
                )
        filtered, predicted, transitions, likelihood = [], [], [], 0.0
        for measurement in measurements:
            predicted.append(self.predict())
            transitions.append(self.transition.copy())
            filtered.append(self.update(measurement))
            likelihood += self.last_log_likelihood
        return FilterResult(
            tuple(filtered),
            tuple(predicted),
            tuple(transitions),
            likelihood,
            self.metadata,
            tuple(timestamps) if timestamps is not None else None,
        )


class ExtendedKalmanFilter(_GaussianFilterBase):
    """Nonlinear Gaussian approximation using explicit process/measurement Jacobians."""

    def __init__(
        self,
        process: Callable,
        measurement: Callable,
        process_jacobian: Callable,
        measurement_jacobian: Callable,
        process_covariance,
        measurement_covariance,
        prior: GaussianState,
        *,
        metadata: EstimatorMetadata,
    ):
        np = _np()
        n = len(prior.mean)
        expected = _vector(measurement(prior.mean), "measurement model output")
        h = _matrix(measurement_jacobian(prior.mean), "measurement Jacobian", (len(expected), n))
        super().__init__(
            np.eye(n), h, process_covariance, measurement_covariance, prior, metadata=metadata
        )
        self.process, self.measurement = process, measurement
        self.process_jacobian, self.measurement_jacobian = process_jacobian, measurement_jacobian

    def predict(self, *, dt: float, process_covariance=None) -> GaussianState:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("EKF dt must be finite positive seconds")
        n = len(self.state.mean)
        f = _matrix(self.process_jacobian(self.state.mean, dt), "process Jacobian", (n, n))
        mean = _vector(self.process(self.state.mean, dt), "process output", n)
        q = (
            self.process_covariance
            if process_covariance is None
            else _covariance(process_covariance, "process covariance", n)
        )
        self.transition = f
        self.state = GaussianState(mean, f @ self.state.covariance @ f.T + q)
        return self.state

    def update(self, measurement, *, missing=None, measurement_covariance=None) -> GaussianState:
        expected = _vector(
            self.measurement(self.state.mean), "measurement output", self.observation.shape[0]
        )
        h = _matrix(
            self.measurement_jacobian(self.state.mean),
            "measurement Jacobian",
            self.observation.shape,
        )
        r = (
            self.measurement_covariance
            if measurement_covariance is None
            else _covariance(
                measurement_covariance, "measurement covariance", len(expected), positive=True
            )
        )
        return self._update(measurement, expected, h, r, missing)

    def update_nonlinear_fused(
        self, measurements, measurement_functions, measurement_jacobians, joint_covariance
    ) -> GaussianState:
        """Joint nonlinear multimodal EKF update with declared Jacobians/correlations."""
        np = _np()
        if not measurements or not (
            len(measurements) == len(measurement_functions) == len(measurement_jacobians)
        ):
            raise ValueError(
                "nonlinear fusion requires aligned measurements, functions, and Jacobians"
            )
        expected, jacobians = [], []
        for function, jacobian in zip(measurement_functions, measurement_jacobians):
            output = _vector(function(self.state.mean), "fused nonlinear measurement output")
            expected.append(output)
            jacobians.append(
                _matrix(
                    jacobian(self.state.mean),
                    "fused measurement Jacobian",
                    (len(output), len(self.state.mean)),
                )
            )
        expected, h = np.concatenate(expected), np.vstack(jacobians)
        r = _covariance(
            joint_covariance, "fused measurement covariance", len(expected), positive=True
        )
        return self._update(np.concatenate(measurements), expected, h, r)

    def filter(
        self, measurements: Sequence[Any], *, timestamps: Sequence[float], initial_time_s: float
    ) -> FilterResult:
        if (
            len(measurements) != len(timestamps)
            or not math.isfinite(initial_time_s)
            or any(not math.isfinite(time) for time in timestamps)
            or any(b <= a for a, b in zip(timestamps, timestamps[1:]))
            or (timestamps and timestamps[0] <= initial_time_s)
        ):
            raise ValueError("EKF requires aligned explicit timestamps and initial_time_s")
        filtered, predicted, transitions, likelihood = [], [], [], 0.0
        previous = initial_time_s
        for measurement, time in zip(measurements, timestamps):
            predicted.append(self.predict(dt=time - previous))
            transitions.append(self.transition.copy())
            filtered.append(self.update(measurement))
            likelihood += self.last_log_likelihood
            previous = time
        return FilterResult(
            tuple(filtered),
            tuple(predicted),
            tuple(transitions),
            likelihood,
            self.metadata,
            tuple(timestamps),
        )


def rts_smooth(result: FilterResult) -> tuple[GaussianState, ...]:
    """Rauch-Tung-Striebel fixed-interval Gaussian smoother (acausal)."""
    np = _np()
    if not (len(result.filtered) == len(result.predicted) == len(result.transitions)):
        raise ValueError("smoother trace dimensions mismatch")
    states = list(result.filtered)
    for k in range(len(states) - 2, -1, -1):
        p = result.filtered[k].covariance
        predicted = result.predicted[k + 1]
        f = result.transitions[k + 1]
        # Degenerate process covariance is allowed; pinv is the generalized RTS map.
        gain = p @ f.T @ np.linalg.pinv(predicted.covariance, hermitian=True)
        states[k] = GaussianState(
            result.filtered[k].mean + gain @ (states[k + 1].mean - predicted.mean),
            p + gain @ (states[k + 1].covariance - predicted.covariance) @ gain.T,
        )
    return tuple(states)


def fuse_gaussians(states: Sequence[GaussianState], *, assume_independent: bool) -> GaussianState:
    """Product of independent Gaussian state evidence; never infer independence."""
    np = _np()
    if not assume_independent or not states:
        raise ValueError("Gaussian fusion requires explicit independent evidence")
    n = len(states[0].mean)
    precision, information = np.zeros((n, n)), np.zeros(n)
    for state in states:
        covariance = _covariance(state.covariance, "fusion covariance", n, positive=True)
        inverse = np.linalg.solve(covariance, np.eye(n))
        precision += inverse
        information += inverse @ _vector(state.mean, "fusion mean", n)
    covariance = np.linalg.solve(precision, np.eye(n))
    return GaussianState(covariance @ information, covariance)


@dataclass(frozen=True)
class ParticleState:
    particles: Any
    weights: Any

    def __post_init__(self):
        np = _np()
        particles = np.asarray(self.particles, dtype=float)
        if particles.ndim != 2 or 0 in particles.shape or not np.all(np.isfinite(particles)):
            raise ValueError("particles must be a finite nonempty (particle,state) matrix")
        weights = _vector(self.weights, "particle weights", len(particles))
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=1e-10, atol=1e-12):
            raise ValueError("particle weights must be non-negative and sum to one")
        particles = particles.copy()
        particles.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "weights", weights)

    @property
    def effective_sample_size(self):
        return float(1 / (self.weights @ self.weights))

    def gaussian_summary(self) -> GaussianState:
        mean = self.weights @ self.particles
        centered = self.particles - mean
        return GaussianState(mean, (centered * self.weights[:, None]).T @ centered)


@dataclass(frozen=True)
class ParticleFilterResult:
    states: tuple[ParticleState, ...]  # posterior BEFORE resampling
    metadata: EstimatorMetadata
    log_likelihood: float


class ParticleFilter:
    """Bootstrap/sequential-importance-resampling filter.

    process(particles, dt, rng) samples the declared transition kernel;
    log_likelihood(z_observed, particles, observed_mask) returns one log density
    per particle, marginalizing missing dimensions. Resampling is systematic.
    """

    def __init__(
        self,
        process: Callable,
        log_likelihood: Callable,
        prior: ParticleState,
        *,
        seed: int,
        metadata: EstimatorMetadata,
        resample_threshold: float = 0.5,
    ):
        if not math.isfinite(resample_threshold) or not 0 <= resample_threshold <= 1:
            raise ValueError("particle resample threshold must be in [0,1]")
        if not callable(process) or not callable(log_likelihood):
            raise ValueError("particle process and log_likelihood must be callable")
        self.process, self.log_likelihood = process, log_likelihood
        self.state, self.metadata = prior, metadata
        self.rng = _np().random.default_rng(seed)
        self.resample_threshold = resample_threshold
        self.last_log_likelihood = 0.0

    def predict(self, *, dt: float) -> ParticleState:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("particle dt must be finite positive seconds")
        values = self.process(self.state.particles.copy(), dt, self.rng)
        if _np().asarray(values).shape != self.state.particles.shape:
            raise ValueError("particle process must retain particle/state dimensions")
        self.state = ParticleState(values, self.state.weights)
        return self.state

    def update(self, measurement, *, missing=None) -> ParticleState:
        np = _np()
        self.last_log_likelihood = 0.0
        if measurement is None:
            return self.state
        z = np.asarray(measurement, dtype=float)
        if z.ndim != 1 or np.any(np.isinf(z)):
            raise ValueError(
                "particle measurements must be a vector with finite or missing NaN values"
            )
        mask = ~np.isnan(z)
        if missing is not None:
            gaps = np.asarray(missing)
            if gaps.shape != z.shape or gaps.dtype.kind != "b":
                raise ValueError("particle missing mask must be boolean and aligned")
            mask &= ~gaps
        if not mask.any():
            return self.state
        logs = np.asarray(self.log_likelihood(z[mask], self.state.particles, mask), dtype=float)
        if (
            logs.shape != self.state.weights.shape
            or np.any(np.isnan(logs))
            or np.any(np.isposinf(logs))
        ):
            raise ValueError(
                "particle likelihood must return finite or -inf log densities per particle"
            )
        with np.errstate(divide="ignore"):
            logs = logs + np.log(self.state.weights)
        maximum = logs.max()
        if not np.isfinite(maximum):
            raise ValueError("particle observation has zero support under the declared model")
        weights = np.exp(logs - maximum)
        total = weights.sum()
        self.last_log_likelihood = float(maximum + np.log(total))
        self.state = ParticleState(self.state.particles, weights / total)
        return self.state

    def resample(self) -> ParticleState:
        np = _np()
        count = len(self.state.weights)
        positions = (self.rng.random() + np.arange(count)) / count
        cumulative = np.cumsum(self.state.weights)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, positions)
        self.state = ParticleState(self.state.particles[indices], np.full(count, 1 / count))
        return self.state

    def filter(self, measurements, *, timestamps, initial_time_s) -> ParticleFilterResult:
        if len(measurements) != len(timestamps) or not math.isfinite(initial_time_s):
            raise ValueError(
                "particle filter requires aligned timestamps and explicit initial time"
            )
        states, likelihood, previous = [], 0.0, initial_time_s
        for measurement, time in zip(measurements, timestamps):
            self.predict(dt=time - previous)
            posterior = self.update(measurement)
            states.append(posterior)
            likelihood += self.last_log_likelihood
            if posterior.effective_sample_size < self.resample_threshold * len(posterior.weights):
                self.resample()
            previous = time
        return ParticleFilterResult(tuple(states), self.metadata, likelihood)


def particle_smooth(
    result: ParticleFilterResult, transition_log_density: Callable, *, timestamps: Sequence[float]
) -> tuple[ParticleState, ...]:
    """Forward-filter backward-smoothing weights, O(T*N^2).

    transition_log_density(previous_particles, next_particles, dt) returns a
    (N_previous,N_next) matrix. No transition density is fabricated from samples.
    """
    np = _np()
    if (
        len(timestamps) != len(result.states)
        or any(not math.isfinite(t) for t in timestamps)
        or any(b <= a for a, b in zip(timestamps, timestamps[1:]))
    ):
        raise ValueError("particle smoother requires aligned strictly increasing finite timestamps")
    states = list(result.states)
    for k in range(len(states) - 2, -1, -1):
        current, following = result.states[k], result.states[k + 1]
        logs = np.asarray(
            transition_log_density(
                current.particles, following.particles, timestamps[k + 1] - timestamps[k]
            ),
            dtype=float,
        )
        if (
            logs.shape != (len(current.weights), len(following.weights))
            or np.any(np.isnan(logs))
            or np.any(np.isposinf(logs))
        ):
            raise ValueError("particle smoother transition density has invalid shape/values")
        with np.errstate(divide="ignore"):
            joint = logs + np.log(current.weights)[:, None]
        maximum = joint.max(axis=0)
        if not np.all(np.isfinite(maximum)):
            raise ValueError("backward smoother transition has zero support")
        kernel = np.exp(joint - maximum)
        kernel /= kernel.sum(axis=0)
        weights = kernel @ states[k + 1].weights
        states[k] = ParticleState(current.particles, weights / weights.sum())
    return tuple(states)


def fit_affine_calibration(
    inputs,
    outputs,
    *,
    input_unit: str,
    output_unit: str,
    input_data: Sequence[str],
    checksum: str,
    operating_conditions: Mapping[str, Any],
):
    """OLS sensor calibration with covariance and excitation/rank diagnostics."""
    np = _np()
    from .time import CalibrationMetadata

    x, y = _vector(inputs, "calibration inputs"), _vector(outputs, "calibration outputs")
    if (
        len(x) != len(y)
        or len(x) < 3
        or not input_unit
        or not output_unit
        or not input_data
        or not checksum
    ):
        raise ValueError("calibration requires >=3 aligned points, explicit units/data/checksum")
    design = np.column_stack((x, np.ones(len(x))))
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("calibration gain/offset not identifiable without input excitation")
    parameters, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ parameters
    variance = float(residual @ residual / (len(x) - 2))
    covariance = variance * np.linalg.inv(design.T @ design)
    return CalibrationMetadata(
        float(parameters[0]),
        float(parameters[1]),
        method="ordinary_least_squares",
        input_unit=input_unit,
        output_unit=output_unit,
        input_data=tuple(input_data),
        checksum=checksum,
        status="FITTED",
        operating_conditions=operating_conditions,
        uncertainty={
            "parameter_covariance": covariance.tolist(),
            "output_variance": variance,
            "degrees_of_freedom": len(x) - 2,
            "design_rank": 2,
            "design_condition_number": float(np.linalg.cond(design)),
            "scope": "affine sensor parameter identifiability under declared OLS assumptions",
        },
    )

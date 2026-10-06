"""Finite-alphabet information bottleneck and rate-distortion optimizers.

All information quantities use natural logarithms (nats). Empirical results
describe the supplied training distribution, not a population or causal law.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence
from typing import Any

from ._numeric import matrix, positive_int, probability, softmax
from .base import BaseRepresentation, RepresentationResult
from .standard import _spec


def discrete_key(value: Any) -> Hashable:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(discrete_key(v) for v in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("discrete symbols must be finite")
    if not isinstance(value, Hashable):
        raise ValueError("discrete symbols must be hashable or finite sequences")
    return value


def _rate(px: Sequence[float], q: Sequence[Sequence[float]], pz: Sequence[float]) -> float:
    value = math.fsum(
        px[x] * p * (math.log(p) - math.log(pz[z]))
        for x, row in enumerate(q)
        for z, p in enumerate(row)
        if p > 0
    )
    if value < -1e-10:
        raise ArithmeticError("negative mutual information beyond roundoff")
    return max(value, 0.0)


def _draw(row: Sequence[float], rng: random.Random) -> int:
    u, cumulative = rng.random(), 0.0
    for i, p in enumerate(row):
        cumulative += p
        if u < cumulative:
            return i
    return len(row) - 1


class InformationBottleneckRepresentation(BaseRepresentation[Any]):
    """Discrete supervised IB: minimize I(X;Z) - beta I(Z;Y).

    Alternating Blahut-Arimoto updates use
    q(z|x) proportional to p(z) exp(-beta KL(p(y|x)||p(y|z))).
    This nonconvex problem has local solutions: seeded multi-start optimization
    selects the best computed objective, without a global-optimality claim.
    ``dimension`` is the latent alphabet size; transform returns q(z|x), not a
    deterministic feature prefix. No discretization or unknown-key fallback is
    performed. Use ``fit_joint`` for a declared finite population model.
    """

    def __init__(
        self,
        dimension: int,
        task_id: str = "supervised",
        *,
        beta: float = 2.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-9,
        n_init: int = 4,
        seed: int = 0,
    ):
        positive_int(dimension, "dimension")
        positive_int(max_iterations, "max_iterations")
        positive_int(n_init, "n_init")
        if not math.isfinite(beta) or beta < 0 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("beta must be finite/nonnegative and tolerance finite/positive")
        if not task_id:
            raise ValueError("task_id is required")
        self.dimension, self.task_id, self.beta = dimension, task_id, beta
        self.max_iterations, self.tolerance = max_iterations, tolerance
        self.n_init, self.seed = n_init, seed
        self.spec = _spec(
            "information_bottleneck", dimension, training="fit_discrete_joint", task=task_id
        )
        self.encoder_: tuple[tuple[float, ...], ...] = ()
        self.objective_history_: tuple[float, ...] = ()
        self.converged_ = False

    def fit(
        self, samples: Sequence[Any], labels: Sequence[Any]
    ) -> "InformationBottleneckRepresentation":
        if len(samples) != len(labels) or not len(samples):
            raise ValueError("samples and labels require equal nonzero lengths")
        xs = tuple(discrete_key(x) for x in samples)
        ys = tuple(discrete_key(y) for y in labels)
        x_symbols, y_symbols = tuple(dict.fromkeys(xs)), tuple(dict.fromkeys(ys))
        xi, yi = {x: i for i, x in enumerate(x_symbols)}, {y: i for i, y in enumerate(y_symbols)}
        joint = [[0.0] * len(y_symbols) for _ in x_symbols]
        for x, y in zip(xs, ys):
            joint[xi[x]][yi[y]] += 1.0 / len(xs)
        return self.fit_joint(
            joint,
            x_symbols=x_symbols,
            y_symbols=y_symbols,
            estimator="empirical_exact_discrete_joint",
        )

    def fit_joint(
        self,
        joint: Sequence[Sequence[float]],
        *,
        x_symbols: Sequence[Any] | None = None,
        y_symbols: Sequence[Any] | None = None,
        estimator: str = "declared_discrete_joint",
    ) -> "InformationBottleneckRepresentation":
        pxy = matrix(joint)
        if min(v for row in pxy for v in row) < 0 or not math.isclose(
            math.fsum(v for row in pxy for v in row), 1.0, abs_tol=1e-10
        ):
            raise ValueError("joint probabilities must be nonnegative and sum to one")
        total = math.fsum(v for row in pxy for v in row)
        pxy = tuple(tuple(v / total for v in row) for row in pxy)
        nx, ny = len(pxy), len(pxy[0])
        xs = tuple(discrete_key(v) for v in (range(nx) if x_symbols is None else x_symbols))
        ys = tuple(discrete_key(v) for v in (range(ny) if y_symbols is None else y_symbols))
        if len(xs) != nx or len(ys) != ny or len(set(xs)) != nx or len(set(ys)) != ny:
            raise ValueError("symbols must be distinct and match joint dimensions")
        px = tuple(math.fsum(row) for row in pxy)
        if min(px) <= 0:
            raise ValueError("source symbols must have positive probability")
        py = tuple(math.fsum(row[y] for row in pxy) for y in range(ny))
        pyx = tuple(tuple(v / px[x] for v in row) for x, row in enumerate(pxy))

        def evaluate(q: Sequence[Sequence[float]]) -> tuple:
            pz = tuple(math.fsum(px[x] * q[x][z] for x in range(nx)) for z in range(self.dimension))
            pyz = tuple(
                tuple(
                    math.fsum(pxy[x][y] * q[x][z] for x in range(nx)) / pz[z] if pz[z] else py[y]
                    for y in range(ny)
                )
                for z in range(self.dimension)
            )
            rate = _rate(px, q, pz)
            relevance = math.fsum(
                pz[z] * p * (math.log(p) - math.log(py[y]))
                for z, row in enumerate(pyz)
                for y, p in enumerate(row)
                if p > 0
            )
            relevance = max(relevance, 0.0)
            return rate - self.beta * relevance, rate, relevance, pz, pyz

        best = None
        for start in range(self.n_init):
            rng = random.Random(self.seed + start)
            q = tuple(
                probability(tuple(v / math.fsum(row) for v in row))
                for row in (
                    [rng.uniform(0.1, 1.0) for _ in range(self.dimension)] for _ in range(nx)
                )
            )
            stats = evaluate(q)
            trace = [stats[0]]
            converged = False
            for iteration in range(1, self.max_iterations + 1):
                pz, pyz = stats[3:]
                updated = []
                for cond in pyx:
                    divergences = tuple(
                        math.fsum(
                            p * (math.log(p) - math.log(model[y])) if model[y] > 0 else math.inf
                            for y, p in enumerate(cond)
                            if p > 0
                        )
                        for model in pyz
                    )
                    logs = tuple(
                        math.log(pz[z]) - (self.beta * divergences[z] if self.beta else 0.0)
                        if pz[z] > 0
                        else -math.inf
                        for z in range(self.dimension)
                    )
                    updated.append(softmax(logs))
                new = tuple(updated)
                change = max(abs(a - b) for old, row in zip(q, new) for a, b in zip(old, row))
                new_stats = evaluate(new)
                if new_stats[0] > stats[0] + 1e-10 * max(1.0, abs(stats[0])):
                    raise ArithmeticError("IB objective increased beyond numerical tolerance")
                q, stats = new, new_stats
                trace.append(stats[0])
                if change <= self.tolerance:
                    converged = True
                    break
            candidate = (stats[0], q, stats, tuple(trace), converged, iteration, start)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        _, self.encoder_, stats, self.objective_history_, self.converged_, self.n_iter_, start = (
            best
        )
        self.objective_, self.rate_, self.relevance_, self.latent_probability_, self.decoder_ = (
            stats
        )
        self.x_symbols_, self.y_symbols_ = xs, ys
        self._index = {x: i for i, x in enumerate(xs)}
        self.metadata_ = {
            "objective": "I(X;Z) - beta * I(Z;Y)",
            "objective_value": self.objective_,
            "beta": self.beta,
            "rate_nats": self.rate_,
            "relevance_nats": self.relevance_,
            "estimator": estimator,
            "optimization": "alternating_blahut_arimoto_multistart",
            "regularization": "information_rate",
            "stochasticity": "categorical_encoder",
            "seed": self.seed,
            "selected_start": start,
            "n_init": self.n_init,
            "converged": self.converged_,
            "iterations": self.n_iter_,
            "convergence_criterion": "max_encoder_change",
            "tolerance": self.tolerance,
            "optimality": "local_stationary_solution",
            "units": "nats",
            "evidence_class": "NUMERICAL_VERIFICATION",
            "causal_claim": False,
        }
        return self

    def transform(self, history: Any) -> RepresentationResult:
        if not self.encoder_:
            raise RuntimeError("fit InformationBottleneckRepresentation before transform")
        key = discrete_key(history)
        if key not in self._index:
            raise ValueError("unseen source symbol; no implicit discretization or fallback")
        return RepresentationResult(
            self.encoder_[self._index[key]], self.spec, dict(self.metadata_)
        )

    def sample(self, history: Any, *, seed: int) -> int:
        return _draw(self.transform(history).values, random.Random(seed))

    def predict_proba(self, history: Any) -> tuple[float, ...]:
        q = self.transform(history).values
        return tuple(
            math.fsum(q[z] * self.decoder_[z][y] for z in range(self.dimension))
            for y in range(len(self.y_symbols_))
        )


class SupervisedBottleneckRepresentation(InformationBottleneckRepresentation):
    """Compatibility class with genuine discrete supervised IB optimization."""

    def __init__(self, dimension: int, task_id: str, **kwargs: Any):
        super().__init__(dimension, task_id, **kwargs)
        self.spec = _spec(
            "supervised_information_bottleneck",
            dimension,
            training="fit_discrete_joint",
            task=task_id,
        )


class RateDistortionRepresentation(BaseRepresentation[Any]):
    """Finite task-aware rate/distortion: min I(X;Z) + beta E[d(X,Z)].

    The caller supplies a nonnegative distortion matrix whose columns are
    declared task reproduction symbols/actions. This is not a universal task
    utility: costs and their units belong to the chosen task.
    """

    def __init__(
        self,
        beta: float = 1.0,
        *,
        task_id: str = "declared_distortion",
        max_iterations: int = 2000,
        tolerance: float = 1e-10,
    ):
        if not math.isfinite(beta) or beta < 0 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("beta must be finite/nonnegative and tolerance finite/positive")
        positive_int(max_iterations, "max_iterations")
        self.beta, self.task_id = beta, task_id
        self.max_iterations, self.tolerance = max_iterations, tolerance
        self.encoder_: tuple[tuple[float, ...], ...] = ()
        self.spec = _spec("rate_distortion", None, training="fit_declared_source", task=task_id)

    def fit(
        self,
        source_probabilities: Sequence[float],
        distortions: Sequence[Sequence[float]],
        *,
        source_symbols: Sequence[Any] | None = None,
    ) -> "RateDistortionRepresentation":
        px, d = probability(source_probabilities), matrix(distortions)
        if len(d) != len(px) or min(px) <= 0 or min(v for row in d for v in row) < 0:
            raise ValueError(
                "positive source probabilities and nonnegative matching distortions required"
            )
        xs = tuple(
            discrete_key(v) for v in (range(len(px)) if source_symbols is None else source_symbols)
        )
        if len(xs) != len(px) or len(set(xs)) != len(xs):
            raise ValueError("distinct source symbols must match source probabilities")
        k = len(d[0])
        pz = (1.0 / k,) * k
        trace = []
        self.converged_ = False
        for iteration in range(1, self.max_iterations + 1):
            q = tuple(
                softmax(
                    tuple(
                        math.log(pz[z]) - self.beta * row[z] if pz[z] else -math.inf
                        for z in range(k)
                    )
                )
                for row in d
            )
            new_pz = tuple(math.fsum(px[x] * q[x][z] for x in range(len(px))) for z in range(k))
            rate = _rate(px, q, new_pz)
            distortion = math.fsum(
                px[x] * q[x][z] * d[x][z] for x in range(len(px)) for z in range(k)
            )
            trace.append(rate + self.beta * distortion)
            change = max(abs(a - b) for a, b in zip(pz, new_pz))
            pz = new_pz
            if change <= self.tolerance:
                self.converged_ = True
                break
        self.encoder_, self.rate_, self.distortion_ = q, rate, distortion
        self.latent_probability_, self.n_iter_ = pz, iteration
        self.objective_history_, self.objective_ = tuple(trace), trace[-1]
        self._index = {x: i for i, x in enumerate(xs)}
        self.spec = _spec("rate_distortion", k, training="fit_declared_source", task=self.task_id)
        self.metadata_ = {
            "objective": "I(X;Z) + beta * E[task_distortion]",
            "beta": self.beta,
            "rate_nats": rate,
            "distortion": distortion,
            "objective_value": self.objective_,
            "optimization": "blahut_arimoto",
            "estimator": "declared_finite_source",
            "stochasticity": "categorical_encoder",
            "converged": self.converged_,
            "iterations": iteration,
            "units": "rate:nats; distortion:caller_declared",
            "evidence_class": "NUMERICAL_VERIFICATION",
        }
        return self

    def transform(self, history: Any) -> RepresentationResult:
        if not self.encoder_:
            raise RuntimeError("fit RateDistortionRepresentation before transform")
        key = discrete_key(history)
        if key not in self._index:
            raise ValueError("unseen source symbol")
        return RepresentationResult(
            self.encoder_[self._index[key]], self.spec, dict(self.metadata_)
        )

    def sample(self, history: Any, *, seed: int) -> int:
        return _draw(self.transform(history).values, random.Random(seed))

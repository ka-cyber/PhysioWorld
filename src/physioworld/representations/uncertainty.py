"""Analytic uncertainty propagation through declared fitted linear maps."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from ._numeric import dot, psd_matrix, symmetric_eigen, vector
from .base import BaseRepresentation, RepresentationResult
from .standard import PCARepresentation, RandomProjectionRepresentation, _spec


class UncertaintyAwareRepresentation(BaseRepresentation[Sequence[float]]):
    """Propagate mean and full input covariance as A Sigma A', exactly.

    Supports PCA and Gaussian random projection; assumes the supplied covariance
    describes input uncertainty. It does not calibrate that covariance or turn
    a parameter range into a confidence interval. Values optionally include
    marginal standard deviations; correlated uncertainty is kept in metadata.
    """

    def __init__(
        self,
        representation: PCARepresentation | RandomProjectionRepresentation,
        *,
        append_std: bool = True,
    ):
        if not isinstance(representation, (PCARepresentation, RandomProjectionRepresentation)):
            raise ValueError("uncertainty propagation requires a supported explicit linear map")
        self.representation, self.append_std = representation, append_std
        d = representation.spec.dimension
        self.spec = _spec(
            "uncertainty_aware_linear",
            d * 2 if d and append_std else d,
            training="wrapped_linear_map",
        )

    def transform(
        self, history: Sequence[float], covariance: Sequence[Sequence[float]] | None = None
    ) -> RepresentationResult:
        if covariance is None:
            raise ValueError("explicit input covariance is required")
        x = vector(history)
        cov = psd_matrix(covariance, len(x))
        result = self.representation.transform(x)
        if isinstance(self.representation, PCARepresentation):
            a = self.representation.components_
            if self.representation.whiten:
                a = tuple(
                    tuple(v / math.sqrt(var) for v in row)
                    for row, var in zip(a, self.representation.explained_variance_)
                )
        else:
            a = self.representation.matrix
        output_cov = tuple(
            tuple(dot(row, tuple(dot(c, other) for c in cov)) for other in a) for row in a
        )
        std = tuple(math.sqrt(max(output_cov[i][i], 0.0)) for i in range(len(a)))
        values = result.values + std if self.append_std else result.values
        self.spec = _spec("uncertainty_aware_linear", len(values), training="wrapped_linear_map")
        return RepresentationResult(
            vector(values),
            self.spec,
            {
                "mean": result.values,
                "covariance": output_cov,
                "std": std,
                "propagation": "A Sigma A_transpose",
                "covariance_source": "caller_declared",
                "calibrated": False,
                "evidence_class": "MODEL",
                "wrapped_method": result.spec.name,
            },
        )

    def sample(
        self, history: Sequence[float], covariance: Sequence[Sequence[float]], *, seed: int
    ) -> tuple[float, ...]:
        result = self.transform(history, covariance)
        eigenvalues, basis = symmetric_eigen(result.metadata["covariance"])
        rng = random.Random(seed)
        z = tuple(rng.gauss(0, 1) * math.sqrt(max(v, 0)) for v in eigenvalues)
        return tuple(
            mean + math.fsum(z[j] * basis[j][i] for j in range(len(z)))
            for i, mean in enumerate(result.metadata["mean"])
        )

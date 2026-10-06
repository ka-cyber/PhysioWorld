"""Optional, genuine Proper Orthogonal Decomposition (POD).

POD here is the snapshot-matrix SVD method: snapshots are centered, a singular
value decomposition is computed, and the retained left/right singular vectors
define projection and reconstruction.  It is not a feature alias or a
truncation label.  NumPy is optional because the core package has no required
scientific dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from physioworld.core.contracts import VersionedRecord, finite


class PODDependencyError(ImportError):
    """Raised when POD is requested without the optional NumPy dependency."""


@dataclass(frozen=True)
class PODModel(VersionedRecord):
    """A fitted centered POD basis with an explicit reference error criterion."""

    reference_id: str
    input_dimension: int
    retained_modes: int
    mean: tuple[float, ...]
    modes: tuple[tuple[float, ...], ...]
    singular_values: tuple[float, ...]
    explained_variance_ratio: tuple[float, ...]
    reconstruction_rmse: float
    error_tolerance: float | None = None
    error_criterion: str = "reference snapshot reconstruction RMSE"

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise ValueError("POD requires a reference identifier")
        if self.input_dimension < 1 or self.retained_modes < 1:
            raise ValueError("POD dimensions must be positive")
        if len(self.mean) != self.input_dimension or len(self.modes) != self.retained_modes:
            raise ValueError("POD basis dimensions do not match metadata")
        if any(len(mode) != self.input_dimension for mode in self.modes):
            raise ValueError("POD modes have inconsistent dimensions")
        if len(self.singular_values) != self.retained_modes:
            raise ValueError("POD singular-value count does not match modes")
        if len(self.explained_variance_ratio) != self.retained_modes:
            raise ValueError("POD explained-variance count does not match modes")
        for value in (*self.mean, *self.singular_values, *self.explained_variance_ratio):
            finite(value, "POD parameter")
        finite(self.reconstruction_rmse, "POD reconstruction RMSE")
        if self.reconstruction_rmse < 0:
            raise ValueError("POD reconstruction RMSE cannot be negative")
        if self.error_tolerance is not None:
            finite(self.error_tolerance, "POD error tolerance")
            if self.error_tolerance < 0:
                raise ValueError("POD error tolerance cannot be negative")

    @property
    def criterion_satisfied(self) -> bool:
        return self.error_tolerance is None or self.reconstruction_rmse <= self.error_tolerance

    def project(self, snapshot: Sequence[float]) -> tuple[float, ...]:
        if len(snapshot) != self.input_dimension:
            raise ValueError("snapshot dimension does not match POD input dimension")
        centered = [float(value) - mean for value, mean in zip(snapshot, self.mean)]
        return tuple(
            sum(mode[index] * centered[index] for index in range(self.input_dimension))
            for mode in self.modes
        )

    def reconstruct(self, coordinates: Sequence[float]) -> tuple[float, ...]:
        if len(coordinates) != self.retained_modes:
            raise ValueError("coordinate dimension does not match retained POD modes")
        return tuple(
            mean
            + sum(
                coordinates[mode_index] * self.modes[mode_index][index]
                for mode_index in range(self.retained_modes)
            )
            for index, mean in enumerate(self.mean)
        )


def fit_pod(
    snapshots: Sequence[Sequence[float]],
    retained_modes: int,
    *,
    reference_id: str,
    error_tolerance: float | None = None,
) -> PODModel:
    """Fit POD by SVD and record reconstruction error on the supplied snapshots."""

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise  # A broken installed package is not an absent dependency.
        raise PODDependencyError("fit_pod requires optional dependency 'numpy'") from exc
    if not snapshots:
        raise ValueError("POD requires at least one snapshot")
    if retained_modes < 1:
        raise ValueError("retained_modes must be positive")
    matrix = np.asarray(snapshots, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("snapshots must be a nonempty rectangular matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("POD snapshots must be finite")
    if retained_modes > min(matrix.shape):
        raise ValueError("retained_modes cannot exceed min(number of snapshots, dimension)")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    modes = right_vectors[:retained_modes]
    coordinates = centered @ modes.T
    reconstruction = coordinates @ modes + mean
    error = float(np.sqrt(np.mean((matrix - reconstruction) ** 2)))
    total = float(np.sum(singular_values**2))
    ratios = tuple(
        float(value**2 / total) if total else 0.0 for value in singular_values[:retained_modes]
    )
    return PODModel(
        reference_id,
        int(matrix.shape[1]),
        retained_modes,
        tuple(float(value) for value in mean),
        tuple(tuple(float(value) for value in row) for row in modes),
        tuple(float(value) for value in singular_values[:retained_modes]),
        ratios,
        error,
        error_tolerance,
    )


__all__ = ["PODDependencyError", "PODModel", "fit_pod"]

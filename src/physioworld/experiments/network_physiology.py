"""Independent, undirected association analysis of supplied multichannel samples.

Input is an insertion-ordered mapping from channel names to aligned, uniformly
sampled real sequences. No alignment, resampling, detrending, missing-value
imputation, or external-data acquisition is performed. Pearson matrices retain
the sign; mutual information (MI) is a finite-sample, equal-width histogram
estimate in nats, sensitive to bin count and sampling bias. MI diagonals are
the discretized channel entropies, not normalized ones.

Phase surrogates independently randomize each channel's positive-frequency
phases, retaining the DC and even-length Nyquist terms and conjugate symmetry.
They preserve the periodogram and circular autocorrelation, but not generally
the amplitude distribution. Their null is appropriate only under the assumed
stationary, uniformly sampled setting; it is not a general independence test.

All results describe associations in the supplied samples, not causal links,
identified physiological mechanisms, or external validation. The implementation
requires only the standard library; Fourier transforms use radix-2 recursion
with a direct DFT fallback for odd sizes (quadratic in the worst case).
"""

from __future__ import annotations

import cmath
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real

Matrix = tuple[tuple[float, ...], ...]
Adjacency = tuple[tuple[bool, ...], ...]
Samples = Mapping[str, Sequence[float]]

__all__ = [
    "AssociationResult",
    "NetworkPhysiologyResult",
    "TopologySummary",
    "analyze_network_physiology",
    "mutual_information_matrix",
    "pearson_matrix",
    "phase_randomized_surrogate",
    "topology_summary",
]


@dataclass(frozen=True)
class TopologySummary:
    """Binary simple-graph summaries; isolated nodes have clustering zero."""

    n_nodes: int
    n_edges: int
    density: float
    degrees: tuple[int, ...]
    mean_degree: float
    connected_components: tuple[tuple[int, ...], ...]
    clustering_coefficients: tuple[float, ...]
    mean_clustering: float
    transitivity: float
    global_efficiency: float


@dataclass(frozen=True)
class AssociationResult:
    """Matrices share the parent result's channel order; p-value diagonals are 1."""

    metric: str
    units: str
    matrix: Matrix
    surrogate_mean: Matrix
    p_values: Matrix
    adjusted_p_values: Matrix
    adjacency: Adjacency
    topology: TopologySummary


@dataclass(frozen=True)
class NetworkPhysiologyResult:
    channel_names: tuple[str, ...]
    sample_count: int
    pearson: AssociationResult
    mutual_information: AssociationResult
    bins: int
    n_surrogates: int
    seed: int
    alpha: float
    pearson_threshold: float
    mi_threshold: float
    status: str = field(default="ASSOCIATION_NOT_CAUSALITY", init=False)
    input_scope: str = field(default="SUPPLIED_SAMPLES_ONLY", init=False)
    causal_claim: bool = field(default=False, init=False)
    external_validation: bool = field(default=False, init=False)
    multiple_testing: str = field(default="BONFERRONI_WITHIN_EACH_METRIC", init=False)
    mi_estimator: str = field(default="EQUAL_WIDTH_HISTOGRAM_PLUGIN", init=False)
    surrogate_method: str = field(default="INDEPENDENT_CHANNEL_PHASE_RANDOMIZATION", init=False)


def _integer(value: int, name: str, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


def _columns(samples: Samples) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    if not isinstance(samples, Mapping) or len(samples) < 2:
        raise ValueError("samples require a mapping with at least two channels")
    names = tuple(samples)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("channel names must be nonempty strings")
    columns = []
    for name in names:
        try:
            values = tuple(samples[name])
        except TypeError as exc:
            raise ValueError("channels must be real sample sequences") from exc
        for value in values:
            _finite(value, f"channel {name}")
        columns.append(tuple(float(value) for value in values))
    if len(columns[0]) < 3 or any(len(row) != len(columns[0]) for row in columns):
        raise ValueError("channels require equal lengths of at least three samples")
    return names, tuple(columns)


def _scaled(values: tuple[float, ...]) -> tuple[tuple[float, ...], float]:
    scale = max(abs(value) for value in values) or 1.0
    return tuple(value / scale for value in values), scale


def _pearson(columns: tuple[tuple[float, ...], ...]) -> Matrix:
    normalized = []
    for column in columns:
        values, _ = _scaled(column)
        mean = math.fsum(values) / len(values)
        centered = tuple(value - mean for value in values)
        norm = math.sqrt(math.fsum(value * value for value in centered))
        if norm == 0:
            raise ValueError("Pearson association is undefined for constant channels")
        normalized.append(tuple(value / norm for value in centered))
    n = len(columns)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        result[i][i] = 1.0
        for j in range(i + 1, n):
            value = math.fsum(a * b for a, b in zip(normalized[i], normalized[j]))
            result[i][j] = result[j][i] = max(-1.0, min(1.0, value))
    return tuple(tuple(row) for row in result)


def pearson_matrix(samples: Samples) -> Matrix:
    """Signed zero-lag correlations in mapping order; reject constant channels."""
    return _pearson(_columns(samples)[1])


def _mutual_information(columns: tuple[tuple[float, ...], ...], bins: int) -> Matrix:
    labels = []
    marginals = []
    for column in columns:
        values, _ = _scaled(column)
        low, high = min(values), max(values)
        symbols = tuple(
            min(bins - 1, int(bins * ((value - low) / (high - low)))) if high > low else 0
            for value in values
        )
        counts: dict[int, int] = {}
        for symbol in symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        labels.append(symbols)
        marginals.append(counts)
    n, count = len(columns), len(columns[0])
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            joint: dict[tuple[int, int], int] = {}
            for pair in zip(labels[i], labels[j]):
                joint[pair] = joint.get(pair, 0) + 1
            value = math.fsum(
                (frequency / count)
                * math.log((frequency * count) / (marginals[i][a] * marginals[j][b]))
                for (a, b), frequency in joint.items()
            )
            result[i][j] = result[j][i] = max(0.0, value)
    return tuple(tuple(row) for row in result)


def mutual_information_matrix(samples: Samples, *, bins: int = 8) -> Matrix:
    """Equal-width plug-in MI in nats; each channel's range defines its bins.

    The maximum goes in the final bin; interior boundaries go in the upper
    bin. Constant channels occupy one bin and have zero MI and entropy. The
    same estimator (with ranges recomputed) is used for every surrogate.
    """
    _integer(bins, "bins", 2)
    return _mutual_information(_columns(samples)[1], bins)


def _fft(values: Sequence[complex]) -> tuple[complex, ...]:
    n = len(values)
    if n == 1:
        return (values[0],)
    if n % 2:
        return tuple(
            sum(value * cmath.exp(-2j * math.pi * k * t / n) for t, value in enumerate(values))
            for k in range(n)
        )
    even, odd = _fft(values[::2]), _fft(values[1::2])
    terms = tuple(cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2))
    return tuple(even[k] + terms[k] for k in range(n // 2)) + tuple(
        even[k] - terms[k] for k in range(n // 2)
    )


def _spectra(
    columns: tuple[tuple[float, ...], ...],
) -> tuple[tuple[tuple[complex, ...], float], ...]:
    result = []
    for column in columns:
        values, scale = _scaled(column)
        result.append((_fft(tuple(complex(value) for value in values)), scale))
    return tuple(result)


def _randomize(
    spectra: tuple[tuple[tuple[complex, ...], float], ...], rng: random.Random
) -> tuple[tuple[float, ...], ...]:
    columns = []
    for spectrum, scale in spectra:
        n = len(spectrum)
        randomized = list(spectrum)
        randomized[0] = complex(spectrum[0].real)
        if n % 2 == 0:
            randomized[n // 2] = complex(spectrum[n // 2].real)
        for k in range(1, (n + 1) // 2):
            randomized[k] = abs(spectrum[k]) * cmath.exp(1j * rng.uniform(0.0, math.tau))
            randomized[n - k] = randomized[k].conjugate()
        inverse = _fft(tuple(value.conjugate() for value in randomized))
        columns.append(tuple((value.real / n) * scale for value in inverse))
    return tuple(columns)


def phase_randomized_surrogate(samples: Samples, *, seed: int = 0) -> dict[str, tuple[float, ...]]:
    """One reproducible real surrogate with independent phases per channel.

    The local RNG does not alter global random state. Preserved DC/Nyquist
    terms mean that some signals have no randomizable spectral content.
    """
    _integer(seed, "seed")
    names, columns = _columns(samples)
    randomized = _randomize(_spectra(columns), random.Random(int(seed)))
    return dict(zip(names, randomized))


def topology_summary(adjacency: Sequence[Sequence[bool]]) -> TopologySummary:
    """Summarize a symmetric binary adjacency matrix with no self-loops.

    Components contain sorted node indices. Global efficiency averages inverse
    unweighted shortest-path distances over all ordered distinct node pairs;
    unreachable pairs contribute zero. Transitivity is closed/all centered
    triplets; all zero-denominator summaries are defined as zero.
    """
    rows = tuple(tuple(row) for row in adjacency)
    n = len(rows)
    if not n or any(len(row) != n for row in rows):
        raise ValueError("adjacency must be a nonempty square matrix")
    if any(not isinstance(v, Integral) or v not in (0, 1) for row in rows for v in row):
        raise ValueError("adjacency must be binary")
    if any(rows[i][i] or any(rows[i][j] != rows[j][i] for j in range(n)) for i in range(n)):
        raise ValueError("adjacency must be symmetric with no self-loops")
    neighbors = tuple(tuple(j for j in range(n) if rows[i][j]) for i in range(n))
    degrees = tuple(len(row) for row in neighbors)
    links = tuple(
        sum(rows[a][b] for k, a in enumerate(row) for b in row[k + 1 :]) for row in neighbors
    )
    triplets = tuple(degree * (degree - 1) // 2 for degree in degrees)
    clustering = tuple(link / total if total else 0.0 for link, total in zip(links, triplets))
    components = []
    visited: set[int] = set()
    inverse_distances: list[float] = []
    for source in range(n):
        distances = {source: 0}
        queue = [source]
        for node in queue:
            for neighbor in neighbors[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        inverse_distances.extend(1.0 / d for d in distances.values() if d)
        if source not in visited:
            components.append(tuple(sorted(distances)))
            visited.update(distances)
    n_edges = sum(degrees) // 2
    pairs = n * (n - 1)
    return TopologySummary(
        n_nodes=n,
        n_edges=n_edges,
        density=2 * n_edges / pairs if pairs else 0.0,
        degrees=degrees,
        mean_degree=sum(degrees) / n,
        connected_components=tuple(components),
        clustering_coefficients=clustering,
        mean_clustering=math.fsum(clustering) / n,
        transitivity=sum(links) / sum(triplets) if sum(triplets) else 0.0,
        global_efficiency=math.fsum(inverse_distances) / pairs if pairs else 0.0,
    )


def _association_result(
    observed: Matrix, null: Sequence[Matrix], metric: str, threshold: float, alpha: float
) -> AssociationResult:
    n = len(observed)
    tests = n * (n - 1) // 2
    means = tuple(
        tuple(math.fsum(matrix[i][j] for matrix in null) / len(null) for j in range(n))
        for i in range(n)
    )
    p_values = [[1.0] * n for _ in range(n)]
    adjusted = [[1.0] * n for _ in range(n)]
    adjacency = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            statistic = abs(observed[i][j]) if metric == "pearson" else observed[i][j]
            tolerance = 1e-12 * max(1.0, statistic)
            exceedances = sum(
                (abs(matrix[i][j]) if metric == "pearson" else matrix[i][j])
                >= statistic - tolerance
                for matrix in null
            )
            p_value = (1 + exceedances) / (1 + len(null))
            corrected = min(1.0, tests * p_value)
            p_values[i][j] = p_values[j][i] = p_value
            adjusted[i][j] = adjusted[j][i] = corrected
            adjacency[i][j] = adjacency[j][i] = corrected <= alpha and statistic > threshold
    graph = tuple(tuple(row) for row in adjacency)
    return AssociationResult(
        metric=metric,
        units="dimensionless" if metric == "pearson" else "nats",
        matrix=observed,
        surrogate_mean=means,
        p_values=tuple(tuple(row) for row in p_values),
        adjusted_p_values=tuple(tuple(row) for row in adjusted),
        adjacency=graph,
        topology=topology_summary(graph),
    )


def analyze_network_physiology(
    samples: Samples,
    *,
    bins: int = 8,
    n_surrogates: int = 99,
    seed: int = 0,
    alpha: float = 0.05,
    pearson_threshold: float = 0.0,
    mi_threshold: float = 0.0,
) -> NetworkPhysiologyResult:
    """Compute associations, phase-surrogate controls, and binary graph summaries.

    Pearson uses |r| for a two-sided control; MI uses its nonnegative value.
    Monte Carlo p = (1 + number of null statistics >= observed) / (B + 1),
    counting differences within 1e-12 * max(1, observed) as numerical ties.
    P-values are Bonferroni-adjusted over distinct undirected pairs separately
    for each metric, not jointly across the two metrics. An edge requires
    adjusted p <= alpha and a statistic strictly above its effect threshold.
    No self-loops are included. With B surrogates, raw p cannot be below
    1/(B+1), so small B may preclude any significant edges.

    Inputs are not mutated. Constant channels are rejected because Pearson is
    undefined. The returned status always states ASSOCIATION_NOT_CAUSALITY.
    """
    _integer(bins, "bins", 2)
    _integer(n_surrogates, "n_surrogates", 1)
    _integer(seed, "seed")
    for name, value in (
        ("alpha", alpha),
        ("pearson_threshold", pearson_threshold),
        ("mi_threshold", mi_threshold),
    ):
        _finite(value, name)
    if not 0 < alpha <= 1 or not 0 <= pearson_threshold <= 1 or mi_threshold < 0:
        raise ValueError("require alpha in (0,1], Pearson threshold in [0,1], MI threshold >= 0")
    names, columns = _columns(samples)
    pearson, mi = _pearson(columns), _mutual_information(columns, bins)
    spectra = _spectra(columns)
    rng = random.Random(int(seed))
    pearson_null, mi_null = [], []
    for _ in range(n_surrogates):
        surrogate = _randomize(spectra, rng)
        pearson_null.append(_pearson(surrogate))
        mi_null.append(_mutual_information(surrogate, bins))
    return NetworkPhysiologyResult(
        channel_names=names,
        sample_count=len(columns[0]),
        pearson=_association_result(pearson, pearson_null, "pearson", pearson_threshold, alpha),
        mutual_information=_association_result(
            mi, mi_null, "mutual_information", mi_threshold, alpha
        ),
        bins=int(bins),
        n_surrogates=int(n_surrogates),
        seed=int(seed),
        alpha=float(alpha),
        pearson_threshold=float(pearson_threshold),
        mi_threshold=float(mi_threshold),
    )

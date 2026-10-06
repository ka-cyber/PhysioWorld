"""Small dense reference linear algebra; no optional backend or fallback."""

from __future__ import annotations

import math
from typing import Sequence


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def vector(values: Sequence[float], width: int | None = None) -> tuple[float, ...]:
    result = tuple(float(v) for v in values)
    if width is not None and len(result) != width:
        raise ValueError(f"expected {width} values, received {len(result)}")
    if any(not math.isfinite(v) for v in result):
        raise ValueError("values must be finite; missing values require explicit preprocessing")
    return result


def matrix(rows: Sequence[Sequence[float]], min_rows: int = 1) -> tuple[tuple[float, ...], ...]:
    result = tuple(vector(row) for row in rows)
    if len(result) < min_rows or not result[0]:
        raise ValueError(f"requires at least {min_rows} nonempty rows")
    if any(len(row) != len(result[0]) for row in result):
        raise ValueError("samples must be rectangular")
    return result


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def probability(values: Sequence[float], width: int | None = None) -> tuple[float, ...]:
    row = vector(values, width)
    if not row or min(row) < 0 or not math.isclose(math.fsum(row), 1.0, abs_tol=1e-10):
        raise ValueError("probabilities must be nonnegative and sum to one")
    total = math.fsum(row)
    return tuple(v / total for v in row)


def softmax(log_weights: Sequence[float]) -> tuple[float, ...]:
    peak = max(log_weights)
    if not math.isfinite(peak):
        raise ValueError("no finite-probability codeword supports this source")
    weights = tuple(math.exp(v - peak) for v in log_weights)
    total = math.fsum(weights)
    return tuple(v / total for v in weights)


def symmetric_eigen(
    rows: Sequence[Sequence[float]],
    tolerance: float = 1e-13,
    max_sweeps: int = 100,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Cyclic Jacobi eigensolver, descending eigenvalues and row eigenvectors.

    Scaling protects rotation arithmetic. Convergence failure is explicit.
    Eigenvector signs are canonical; a repeated eigenspace has no unique basis.
    """
    a0 = matrix(rows)
    n = len(a0)
    if len(a0[0]) != n:
        raise ValueError("eigensolver requires a square matrix")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    positive_int(max_sweeps, "max_sweeps")
    scale = max(abs(v) for row in a0 for v in row)
    if any(
        abs(a0[i][j] - a0[j][i]) > tolerance * max(scale, 1e-300)
        for i in range(n)
        for j in range(i)
    ):
        raise ValueError("matrix must be symmetric")
    v = [[float(i == j) for j in range(n)] for i in range(n)]
    if scale == 0:
        return (0.0,) * n, tuple(tuple(row) for row in v)
    a = [[entry / scale for entry in row] for row in a0]
    for _ in range(max_sweeps):
        off = max((abs(a[i][j]) for i in range(n) for j in range(i)), default=0.0)
        if off <= tolerance:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= tolerance / n:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = math.copysign(1.0, tau) / (abs(tau) + math.hypot(1.0, tau))
                c = 1.0 / math.hypot(1.0, t)
                s = t * c
                a[p][p] -= t * apq
                a[q][q] += t * apq
                a[p][q] = a[q][p] = 0.0
                for k in range(n):
                    if k != p and k != q:
                        x, y = a[k][p], a[k][q]
                        a[k][p] = a[p][k] = c * x - s * y
                        a[k][q] = a[q][k] = s * x + c * y
                    x, y = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * x - s * y, s * x + c * y
    if max((abs(a[i][j]) for i in range(n) for j in range(i)), default=0.0) > tolerance:
        raise RuntimeError("Jacobi eigensolver did not converge")
    order = sorted(range(n), key=lambda i: (-a[i][i], i))
    components = []
    for j in order:
        col = tuple(v[i][j] for i in range(n))
        pivot = max(range(n), key=lambda i: abs(col[i]))
        sign = -1.0 if col[pivot] < 0 else 1.0
        components.append(tuple(sign * x for x in col))
    return tuple(a[i][i] * scale for i in order), tuple(components)


def psd_matrix(rows: Sequence[Sequence[float]], width: int) -> tuple[tuple[float, ...], ...]:
    result = matrix(rows)
    if len(result) != width or len(result[0]) != width:
        raise ValueError("covariance has wrong dimensions")
    eigenvalues, _ = symmetric_eigen(result)
    scale = max(abs(v) for v in eigenvalues)
    if min(eigenvalues) < -1e-11 * max(scale, 1e-300):
        raise ValueError("covariance must be positive semidefinite")
    return result

"""Small dense matrix exponential, requiring only Python's standard library.

Scaling/squaring uses a Taylor series at infinity norm <= 1/2. The truncation
tail is controlled before squaring; floating-point roundoff is not a rigorous
interval bound. Independent SciPy/DOP853 checks qualify the numerical method.
"""

import math


def matvec(a, x):
    return tuple(math.fsum(v * w for v, w in zip(row, x)) for row in a)


def norm(a):
    return max((math.fsum(abs(v) for v in row) for row in a), default=0.0)


def multiply(a, b):
    columns = tuple(zip(*b))
    return tuple(tuple(math.fsum(x * y for x, y in zip(row, col)) for col in columns) for row in a)


def expm(a):
    n = len(a)
    if not n or any(len(row) != n for row in a):
        raise ValueError("matrix must be nonempty and square")
    if not all(math.isfinite(v) for row in a for v in row):
        raise ValueError("matrix must be finite")
    size = norm(a)
    scaling = max(0, math.ceil(math.log2(size / 0.5))) if size else 0
    divisor = 2.0**scaling
    scaled = tuple(tuple(v / divisor for v in row) for row in a)
    result = tuple(tuple(float(i == j) for j in range(n)) for i in range(n))
    term = result
    for k in range(1, 100):
        term = tuple(tuple(v / k for v in row) for row in multiply(term, scaled))
        result = tuple(tuple(x + y for x, y in zip(r, t)) for r, t in zip(result, term))
        # The remaining terms are bounded by a geometric majorant.
        if norm(term) * 0.5 / (k + 1) / (1 - 0.5 / (k + 2)) <= 2e-17 / divisor:
            break
    else:
        raise ArithmeticError("matrix exponential failed to converge")
    for _ in range(scaling):
        result = multiply(result, result)
    if not all(math.isfinite(v) for row in result for v in row):
        raise ArithmeticError("matrix exponential overflow")
    return result


def affine_transition(a, b, x, duration):
    n = len(x)
    augmented = tuple(
        tuple(a[i][j] * duration for j in range(n)) + (b[i] * duration,) for i in range(n)
    )
    augmented += ((0.0,) * (n + 1),)
    return matvec(expm(augmented), (*x, 1.0))[:n]


def integrate(function, duration, tolerance=1e-12):
    """Adaptive Simpson integral with an explicit error estimate."""
    if duration == 0:
        return 0.0, 0.0
    f0, fm, f1 = function(0.0), function(duration / 2), function(duration)
    whole = duration * (f0 + 4 * fm + f1) / 6

    def recurse(lo, hi, left, mid, right, estimate, tol, depth):
        center = (lo + hi) / 2
        quarter, three_quarter = function((lo + center) / 2), function((center + hi) / 2)
        first = (center - lo) * (left + 4 * quarter + mid) / 6
        second = (hi - center) * (mid + 4 * three_quarter + right) / 6
        error = abs(first + second - estimate) / 15
        if error <= tol:
            return first + second + (first + second - estimate) / 15, error
        if depth == 0:
            raise ArithmeticError("energy quadrature tolerance unresolved")
        a, ea = recurse(lo, center, left, quarter, mid, first, tol / 2, depth - 1)
        b, eb = recurse(center, hi, mid, three_quarter, right, second, tol / 2, depth - 1)
        return a + b, ea + eb

    return recurse(0.0, duration, f0, fm, f1, whole, tolerance, 24)

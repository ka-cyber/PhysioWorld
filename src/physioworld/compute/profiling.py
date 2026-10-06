"""Actual callable profiling, separate from operation-count execution simulation."""

import math
import platform
import statistics
import time
import tracemalloc
from dataclasses import dataclass


@dataclass(frozen=True)
class CallableProfile:
    operation_id: str
    durations_s: tuple[float, ...]
    mean_duration_s: float
    sample_standard_deviation_s: float
    peak_python_allocations_bytes: int
    python_version: str
    platform: str
    evidence: str = "EMPIRICAL_ESTIMATE"
    memory_scope: str = "tracemalloc Python allocations, not total device/OS memory"
    timing_scope: str = (
        "local perf_counter wall time; supplied callable/input and current machine only"
    )


def profile_callable(operation, *, operation_id, repetitions=10, warmup=1):
    if (
        not operation_id
        or not isinstance(repetitions, int)
        or repetitions < 1
        or not isinstance(warmup, int)
        or warmup < 0
    ):
        raise ValueError("profiling requires an operation ID and valid repetition/warmup counts")
    if tracemalloc.is_tracing():
        raise RuntimeError("profiling requires an independent tracemalloc session")
    for _ in range(warmup):
        operation()
    durations = []
    tracemalloc.start()
    try:
        for _ in range(repetitions):
            start = time.perf_counter()
            operation()
            durations.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if not all(math.isfinite(v) and v >= 0 for v in durations):
        raise ArithmeticError("profiling clock returned invalid elapsed time")
    return CallableProfile(
        operation_id,
        tuple(durations),
        statistics.mean(durations),
        statistics.stdev(durations) if repetitions > 1 else 0.0,
        peak,
        platform.python_version(),
        platform.platform(),
    )

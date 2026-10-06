"""Byte/symbol quantization and semantic reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from ._validation import finite, integer
from .models import Packet, ScientificContext, SemanticStatistics


@dataclass(frozen=True)
class UniformQuantizer:
    low: float
    high: float
    levels: int

    def __post_init__(self) -> None:
        finite(self.low, "low")
        finite(self.high, "high")
        integer(self.levels, "levels", 2)
        if self.high <= self.low or self.levels > 256:
            raise ValueError("invalid quantizer")
        finite(self.high - self.low, "quantizer range")

    def encode(self, values: Sequence[float]) -> bytes:
        scale = (self.levels - 1) / (self.high - self.low)
        finite(scale, "quantizer scale")
        return bytes(
            round((max(self.low, min(self.high, finite(value, "value"))) - self.low) * scale)
            for value in values
        )

    def decode(self, symbols: bytes) -> tuple[float, ...]:
        if not isinstance(symbols, bytes) or any(symbol >= self.levels for symbol in symbols):
            raise ValueError("symbols are outside the quantizer alphabet")
        return tuple(
            self.low + symbol / (self.levels - 1) * (self.high - self.low) for symbol in symbols
        )


def mean_squared_distortion(expected: Sequence[float], observed: Sequence[float]) -> float:
    if len(expected) != len(observed) or not expected:
        raise ValueError("sequences must have equal non-zero length")
    for value in (*expected, *observed):
        finite(value, "distortion input")
    try:
        result = sum((a - b) ** 2 for a, b in zip(expected, observed)) / len(expected)
    except OverflowError as exc:
        raise ValueError("distortion overflow") from exc
    return finite(result, "distortion", 0)


def task_law_preservation(full_metric: float, degraded_metric: float) -> float:
    """Legacy absolute scalar difference; not a full conditional task-law estimator.

    Use evaluate_semantics with declared task conditional expectations to
    calculate the finite task-law discrepancy sup_tau E|m_H - m_Z|.
    """
    return finite(
        abs(finite(full_metric, "full metric") - finite(degraded_metric, "degraded metric")),
        "metric difference",
        0,
    )


def evaluate_semantics(
    reference: Sequence[float],
    observed: Sequence[float | None],
    *,
    context: ScientificContext,
    reference_task_laws: Sequence[Sequence[float]] | None = None,
    observed_task_laws: Sequence[Sequence[float]] | None = None,
    reference_task_value: float | None = None,
    task_value: float | None = None,
    capability_preservation: float | None = None,
) -> SemanticStatistics:
    """Evaluate declared downstream quantities independently of packet statistics.

    Task-law arrays are [task][evaluation sample] conditional expectations of
    declared task functions. Discrepancy is max_task mean_sample absolute gap,
    with equal sample weights. These inputs are not estimated by transport.
    Reconstruction is unavailable if any coordinate is unavailable: no zero
    imputation and no silent conditioning on the successfully delivered subset.
    Task value and capability quantities come from the caller's task/policy
    evaluator, never from decode probability or reconstruction distortion.
    """
    if not isinstance(context, ScientificContext):
        raise ValueError("a scientific context is required")
    if not reference or len(reference) != len(observed):
        raise ValueError("semantic vectors must have equal non-zero length")
    for reference_value in reference:
        finite(reference_value, "reference")
    for observed_value in observed:
        if observed_value is not None:
            finite(observed_value, "observed")
    observed_values: list[float] = []
    for observed_value in observed:
        if observed_value is not None:
            finite(observed_value, "observed")
            observed_values.append(observed_value)
    available = len(observed_values)
    distortion = (
        mean_squared_distortion(reference, tuple(observed_values))
        if available == len(observed)
        else None
    )
    discrepancy = None
    if (reference_task_laws is None) != (observed_task_laws is None):
        raise ValueError("both task-law arrays must be provided")
    if reference_task_laws is not None:
        if observed_task_laws is None:
            raise ValueError("both task-law arrays must be provided")
        if len(reference_task_laws) != len(context.task_family) or (
            len(reference_task_laws) != len(observed_task_laws)
        ):
            raise ValueError("task-law arrays must match the declared task family")
        gaps = []
        sample_count = len(reference_task_laws[0])
        if not sample_count:
            raise ValueError("task-law evaluation needs non-empty samples")
        for expected, actual in zip(reference_task_laws, observed_task_laws):
            if len(expected) != sample_count or len(actual) != sample_count:
                raise ValueError("task-law sample dimensions must agree")
            gap = (
                sum(
                    abs(finite(a, "reference task law") - finite(b, "observed task law"))
                    for a, b in zip(expected, actual)
                )
                / sample_count
            )
            gaps.append(finite(gap, "task-law gap", 0))
        discrepancy = max(gaps)
    for name, metric_value in (
        ("task_value", task_value),
        ("reference_task_value", reference_task_value),
        ("capability_preservation", capability_preservation),
    ):
        if metric_value is not None:
            finite(metric_value, name)
    return SemanticStatistics(
        context,
        available / len(observed),
        distortion,
        discrepancy,
        task_value,
        reference_task_value,
        capability_preservation,
    )


def permute_payloads(packets: Sequence[Packet], permutation: Sequence[int]) -> tuple[Packet, ...]:
    """Permanent semantic negative control: valid frames, incorrect role assignments.

    The role/sequence/timestamp metadata stay attached to the original positions;
    only payloads are permuted before coding. New lengths/checksums are valid, so
    perfect transport and decoding cannot detect this semantic misassignment.
    """
    indices = tuple(permutation)
    for index in indices:
        integer(index, "permutation index")
    if sorted(indices) != list(range(len(packets))):
        raise ValueError("permutation must contain each source position exactly once")
    if len({packet.sequence for packet in packets}) != len(packets):
        raise ValueError("source sequence IDs must be unique")
    result = []
    for packet, index in zip(packets, indices):
        payload = packets[index].payload
        header = packet.header
        if header is None:
            raise ValueError("packet header invariant was not established")
        result.append(
            replace(packet, payload=payload, header=replace(header, payload_length=len(payload)))
        )
    return tuple(result)

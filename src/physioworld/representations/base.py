"""Representation contracts and explicit metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, cast

from ._numeric import vector

InputT = TypeVar("InputT", contravariant=True)


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    input: str
    history: str
    dimension: int | None
    training_dependence: str
    task_dependence: str
    causal_status: str
    computational_cost: str
    memory_cost: str
    approximation: str | None = None


@dataclass(frozen=True)
class RepresentationResult:
    values: tuple[float, ...]
    spec: RepresentationSpec
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRepresentation(Generic[InputT]):
    spec: RepresentationSpec

    def transform(self, history: InputT) -> RepresentationResult:
        raise NotImplementedError

    def __call__(self, history: InputT) -> RepresentationResult:
        return self.transform(history)

    def transform_at(
        self,
        history: Sequence[float],
        timestamps: Sequence[float],
        prediction_time: float,
    ) -> RepresentationResult:
        """Transform the available prefix using explicit acquisition-time semantics.

        Timestamps are strictly increasing, in the caller's declared common unit.
        Future samples are excluded before calling the representation.
        """
        times = vector(timestamps, len(history))
        cutoff = vector((prediction_time,))[0]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("timestamps must be strictly increasing")
        count = sum(t <= cutoff for t in times)
        result = self.transform(cast(InputT, history[:count]))
        return RepresentationResult(
            result.values,
            result.spec,
            {
                **result.metadata,
                "prediction_time": cutoff,
                "available_samples": count,
                "excluded_future_samples": len(history) - count,
                "latest_acquisition_time": times[count - 1] if count else None,
            },
        )

"""Deterministic replay of real or synthetic streams."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from physioworld.core import WorldState

from .stream import SignalStream


@dataclass
class PhysiologicalWorld:
    streams: Mapping[str, SignalStream]
    state: WorldState = field(default_factory=WorldState)
    mode: str = "replay"
    ground_truth: Mapping[str, Any] = field(default_factory=dict)
    _replay_cursor: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.streams:
            raise ValueError("world requires at least one stream")
        if self.mode not in {"replay", "synthetic", "hybrid"}:
            raise ValueError("world mode must be replay, synthetic, or hybrid")

    @classmethod
    def synthetic(
        cls, streams: Mapping[str, SignalStream], ground_truth: Mapping[str, Any]
    ) -> "PhysiologicalWorld":
        return cls(streams, mode="synthetic", ground_truth=dict(ground_truth))

    @classmethod
    def hybrid(
        cls, streams: Mapping[str, SignalStream], ground_truth: Mapping[str, Any] | None = None
    ) -> "PhysiologicalWorld":
        return cls(streams, mode="hybrid", ground_truth=dict(ground_truth or {}))

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(self.streams)

    def timeline(self) -> tuple[float, ...]:
        return tuple(sorted({t for stream in self.streams.values() for t in stream.timestamps}))

    def replay(self, *, resume: bool = False) -> Iterator[WorldState]:
        """Yield snapshots without merging modality values into one state."""
        if not resume:
            self._replay_cursor = 0
        for time in self.timeline()[self._replay_cursor :]:
            observations = {
                name: stream.values[stream.timestamps.index(time)]
                for name, stream in self.streams.items()
                if time in stream.timestamps
            }
            next_state = self.state.snapshot()
            next_state.time = time
            next_state.physiological = observations
            next_state.history = tuple((*next_state.history, (time, observations)))
            self.state = next_state
            self._replay_cursor += 1
            yield next_state.snapshot()

    def replay_chunks(self, max_samples: int = 256) -> Iterator[tuple[WorldState, ...]]:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        chunk = []
        for snapshot in self.replay(resume=True):
            chunk.append(snapshot)
            if len(chunk) == max_samples:
                yield tuple(chunk)
                chunk = []
        if chunk:
            yield tuple(chunk)

    def fork(self) -> PhysiologicalWorld:
        """Independent replay cursor/state; immutable source streams are retained."""
        return copy.deepcopy(self)

    @staticmethod
    def physiological(*args: Any, **kwargs: Any):
        """Create the coupled simulation engine without changing stream-world construction."""
        from .engine import PhysiologyEngine

        return PhysiologyEngine(*args, **kwargs)

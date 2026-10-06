"""Typed, replayable calculation DAG on native SignalStream objects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Callable

from physioworld.core import Provenance
from physioworld.physiology.stream import SignalStream

from . import analysis, physiology


class ResultKind(str, Enum):
    SIGNAL = "signal"
    SPECTRUM = "spectrum"
    SPECTROGRAM = "spectrogram"
    EVENTS = "events"
    CYCLES = "cycles"
    MEASUREMENTS = "measurements"


@dataclass(frozen=True)
class CalculationNode:
    id: str
    operation: str
    inputs: tuple[str, ...]
    parameters: dict[str, Any]
    output_kind: ResultKind


_S = ResultKind.SIGNAL
_E = ResultKind.EVENTS
_M = ResultKind.MEASUREMENTS
# function, input kinds, output kind. Registry is local signal operations, not plugins.
_OPS: dict[str, tuple[Callable[..., Any], tuple[ResultKind, ...], ResultKind]] = {
    "add": (lambda a, b: analysis.arithmetic(a, b, "add"), (_S, _S), _S),
    "subtract": (lambda a, b: analysis.arithmetic(a, b, "subtract"), (_S, _S), _S),
    "multiply": (lambda a, b: analysis.arithmetic(a, b, "multiply"), (_S, _S), _S),
    "divide": (lambda a, b: analysis.arithmetic(a, b, "divide"), (_S, _S), _S),
    "convert_units": (analysis.convert, (_S,), _S),
    "smooth": (analysis.smooth, (_S,), _S),
    "filter": (analysis.filter_signal, (_S,), _S),
    "derivative": (analysis.derivative, (_S,), _S),
    "integral": (analysis.integral, (_S,), _S),
    "resample": (analysis.resample, (_S,), _S),
    "time_shift": (analysis.time_shift, (_S,), _S),
    "transform": (analysis.explicit_transform, (_S,), _S),
    "fft": (lambda s, **kw: analysis.spectrum(s, "fft", **kw), (_S,), ResultKind.SPECTRUM),
    "psd": (lambda s, **kw: analysis.spectrum(s, "psd", **kw), (_S,), ResultKind.SPECTRUM),
    "spectrogram": (analysis.spectrogram, (_S,), ResultKind.SPECTROGRAM),
    "events": (analysis.detect_events, (_S,), _E),
    "cycles": (analysis.cycles, (_E,), ResultKind.CYCLES),
    "autocorrelation": (analysis.autocorrelation, (_S,), _S),
    "template_match": (analysis.template_match, (_S,), _S),
    "event_average": (analysis.event_average, (_S, _E), _S),
    "statistics": (analysis.statistics_extract, (_S,), _M),
    "quality": (analysis.quality, (_S,), _M),
    "ecg_peaks": (physiology.ecg_peaks, (_S,), _E),
    "rr": (physiology.rr_intervals, (_E,), _S),
    "hrv": (physiology.hrv, (_S,), _M),
    "blood_pressure": (physiology.blood_pressure, (_S, _E), _M),
    "respiration": (physiology.respiration, (_S,), _M),
    "emg": (physiology.emg, (_S,), _M),
    "eeg": (physiology.eeg, (_S,), _M),
    "pressure_volume": (physiology.pressure_volume_work, (_S, _S), _M),
}


def _digest(value: Any) -> str:
    def default(obj: Any) -> Any:
        if callable(obj):
            return f"{obj.__module__}.{obj.__qualname__}"
        return str(obj)

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=default, allow_nan=True).encode()
    ).hexdigest()


class CalculationGraph:
    def __init__(self) -> None:
        self.sources: dict[str, SignalStream] = {}
        self.nodes: dict[str, CalculationNode] = {}
        self.history: list[Provenance] = []

    def source(self, node_id: str, stream: SignalStream) -> str:
        self._unique(node_id)
        self.sources[node_id] = stream
        return node_id

    def add(self, node_id: str, operation: str, *inputs: str, **parameters: Any) -> str:
        self._unique(node_id)
        if operation not in _OPS:
            raise ValueError(f"unknown calculation {operation!r}")
        _, expected, output = _OPS[operation]
        if len(inputs) != len(expected):
            raise TypeError(f"{operation} requires {len(expected)} inputs")
        self.nodes[node_id] = CalculationNode(
            node_id, operation, tuple(inputs), dict(parameters), output
        )
        return node_id

    def _unique(self, node_id: str) -> None:
        if not node_id or node_id in self.sources or node_id in self.nodes:
            raise ValueError(f"duplicate/empty node id {node_id!r}")

    def validate(self) -> tuple[str, ...]:
        done, active, order = set(self.sources), set(), []

        def visit(name: str) -> None:
            if name in active:
                raise ValueError(f"calculation graph cycle at {name}")
            if name in done:
                return
            if name not in self.nodes:
                raise KeyError(f"unknown graph input {name}")
            active.add(name)
            node = self.nodes[name]
            for parent, expected in zip(node.inputs, _OPS[node.operation][1]):
                visit(parent)
                actual = _S if parent in self.sources else self.nodes[parent].output_kind
                if actual != expected:
                    raise TypeError(f"{name}: {parent} is {actual}, expected {expected}")
            active.remove(name)
            done.add(name)
            order.append(name)

        for name in self.nodes:
            visit(name)
        return tuple(order)

    def evaluate(self, node_id: str) -> Any:
        self.validate()
        cache: dict[str, Any] = dict(self.sources)
        fingerprints = {
            name: _digest(
                {
                    "timestamps": s.timestamps,
                    "values": s.values,
                    "metadata": asdict(s.metadata),
                    "annotations": asdict(s)["annotations"],
                }
            )
            for name, s in self.sources.items()
        }

        def evaluate(name: str) -> Any:
            if name in cache:
                return cache[name]
            node = self.nodes[name]
            inputs = [evaluate(parent) for parent in node.inputs]
            notes = {
                "node_id": name,
                "parameters": {
                    k: (f"{v.__module__}.{v.__qualname__}" if callable(v) else v)
                    for k, v in node.parameters.items()
                },
                "parent_hashes": [fingerprints[p] for p in node.inputs],
            }
            provenance = Provenance(
                "physioworld.CalculationGraph", node.operation, parent_ids=node.inputs, notes=notes
            )
            result = _OPS[node.operation][0](*inputs, **node.parameters)
            original = (
                result.metadata.provenance
                if isinstance(result, SignalStream)
                else getattr(result, "provenance", None)
            )
            if original is not None:
                notes["algorithm"] = dict(original.notes)
            notes["operation"] = node.operation
            if isinstance(result, SignalStream):
                result = replace(result, metadata=replace(result.metadata, provenance=provenance))
            elif hasattr(result, "provenance"):
                result = replace(result, provenance=provenance)
            fingerprints[name] = _digest({"operation": node.operation, "lineage": notes})
            self.history.append(provenance)
            cache[name] = result
            return result

        if node_id not in self.sources and node_id not in self.nodes:
            raise KeyError(node_id)
        return evaluate(node_id)

    def description(self) -> dict[str, Any]:
        return {"sources": list(self.sources), "nodes": [asdict(n) for n in self.nodes.values()]}

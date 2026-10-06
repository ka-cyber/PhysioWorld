"""Fast benchmark that exercises independent core layers."""

from __future__ import annotations

import time
from typing import Any

from physioworld.communication import RLNCDecoder, RLNCEncoder
from physioworld.resources import CapacitorResource, ResourceState


def run_smoke_benchmark() -> dict[str, Any]:
    start = time.perf_counter()
    symbols = (b"physio", b"world!", b"core!!")
    codewords = RLNCEncoder(symbols, seed=3).encode(3)
    decoder = RLNCDecoder(3, 6)
    for word in reversed(codewords):
        decoder.add(word)
    decoded = decoder.decode()
    model = CapacitorResource(0.2, 8.0, v_min=1.85)
    transition = model.step(ResourceState(2.65), 0.01, 10.0)
    return {
        "duration_s": time.perf_counter() - start,
        "rlnc_correct": decoded == symbols,
        "resource_feasible": transition.feasible,
        "resource_terminal_voltage": transition.terminal_voltage,
    }

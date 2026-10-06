"""Seeded representation negative controls with replayable provenance."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from ._numeric import matrix


@dataclass(frozen=True)
class RepresentationNegativeControl:
    values: tuple[Any, ...]
    control: str
    seed: int
    permutation: tuple[int, ...] | None
    evidence_class: str = "SIMULATION"


def permutation_control(
    values: Sequence[Any], *, control: str, seed: int = 0
) -> RepresentationNegativeControl:
    """Permute labels, row semantics or temporal order; leave values intact.

    Apply to a chosen pipeline edge before refitting/evaluating its scientific
    estimand. A permutation is not guaranteed to destroy all dependence (e.g.
    constant values); measure the control's effect rather than assuming it.
    """
    if control not in {"label_permutation", "semantic_permutation", "temporal_shuffle"}:
        raise ValueError("unknown permutation negative control")
    if len(values) < 2:
        raise ValueError("permutation controls require at least two items")
    indices = list(range(len(values)))
    random.Random(seed).shuffle(indices)
    return RepresentationNegativeControl(
        tuple(values[i] for i in indices), control, seed, tuple(indices)
    )


def randomized_representation_control(
    samples: Sequence[Sequence[float]], *, seed: int = 0
) -> RepresentationNegativeControl:
    """Shape-preserving independent N(0,1) features, unrelated to source rows."""
    x, rng = matrix(samples), random.Random(seed)
    values = tuple(tuple(rng.gauss(0, 1) for _ in row) for row in x)
    return RepresentationNegativeControl(values, "randomized_representation", seed, None)

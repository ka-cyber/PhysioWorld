"""Reproducible finite-grid design, corners, phase boundaries and sensitivities."""

import math
from dataclasses import dataclass
from itertools import product
from typing import Mapping


@dataclass(frozen=True)
class DesignEvaluation:
    capability: tuple[float, ...]
    policy_id: str
    feasible: bool = True
    reason: str | None = None

    def __post_init__(self):
        if not self.capability or not all(math.isfinite(v) for v in self.capability):
            raise ValueError("design capability must be finite and nonempty")
        if not self.policy_id:
            raise ValueError("policy identity is required")


@dataclass(frozen=True)
class PhasePoint:
    coordinates: Mapping[str, float]
    evaluation: DesignEvaluation
    satisfies: bool
    region: str


@dataclass(frozen=True)
class PhaseDiagram:
    points: tuple[PhasePoint, ...]
    boundary_edges: tuple[tuple[int, int], ...]
    policy_swings: tuple[tuple[int, int], ...]
    scope: str = "declared finite grid; boundaries are brackets, not universal laws"


def phase_diagram(axes, evaluate, requirement):
    if not axes or any(
        not values or len(set(values)) != len(values) or not all(math.isfinite(v) for v in values)
        for values in axes.values()
    ):
        raise ValueError("axes must be finite, unique, nonempty grids")
    if not requirement or not all(math.isfinite(v) for v in requirement):
        raise ValueError("requirement must be finite")
    names = tuple(axes)
    grids = tuple(tuple(sorted(axes[k])) for k in names)
    indices = tuple(product(*(range(len(g)) for g in grids)))
    lookup, points = {index: i for i, index in enumerate(indices)}, []
    for index in indices:
        coordinates = {name: grids[j][index[j]] for j, name in enumerate(names)}
        result = evaluate(coordinates)
        if len(result.capability) != len(requirement):
            raise ValueError("task requirement dimension mismatch")
        meets = result.feasible and all(x >= y for x, y in zip(result.capability, requirement))
        region = (
            "sufficient"
            if meets
            else "resource_constrained"
            if result.feasible
            else "capability_collapse"
        )
        points.append(PhasePoint(coordinates, result, meets, region))
    boundaries, swings = [], []
    for i, index in enumerate(indices):
        for axis in range(len(names)):
            neighbor = tuple(v + (axis == j) for j, v in enumerate(index))
            if neighbor in lookup:
                k = lookup[neighbor]
                if points[i].region != points[k].region:
                    boundaries.append((i, k))
                if points[i].evaluation.policy_id != points[k].evaluation.policy_id:
                    swings.append((i, k))
    return PhaseDiagram(tuple(points), tuple(boundaries), tuple(swings))


def resource_frontier(diagram, *, resource_axes=None):
    """Minimal feasible grid budgets, minimizing each named resource axis."""
    resource_axes = tuple(resource_axes or diagram.points[0].coordinates)
    feasible = tuple(p for p in diagram.points if p.satisfies)
    return tuple(
        p
        for p in feasible
        if not any(
            all(q.coordinates[k] <= p.coordinates[k] for k in resource_axes)
            and any(q.coordinates[k] < p.coordinates[k] for k in resource_axes)
            for q in feasible
        )
    )


def design_search(candidates, evaluate, objective):
    """Exact enumeration on a declared candidate set; no global-optimum claim."""
    results = tuple((candidate, evaluate(candidate)) for candidate in candidates)
    scored = tuple(
        (candidate, result, objective(candidate, result))
        for candidate, result in results
        if result.feasible
    )
    if not scored or not all(math.isfinite(score) for _, _, score in scored):
        raise ValueError("design search requires feasible candidates and finite scores")
    return max(scored, key=lambda item: item[2]), results


@dataclass(frozen=True)
class CornerAnalysis:
    corners: tuple[Mapping[str, float], ...]
    values: tuple[float, ...]
    worst_index: int
    scope: str = "enumerated corners only; interior robustness requires additional justification"

    @property
    def worst_value(self):
        return self.values[self.worst_index]


def corner_analysis(bounds, evaluate):
    if not bounds or any(
        len(v) != 2 or not all(math.isfinite(x) for x in v) or v[0] > v[1] for v in bounds.values()
    ):
        raise ValueError("finite ordered parameter bounds required")
    names = tuple(sorted(bounds))
    corners = tuple(
        dict(zip(names, values))
        for values in product(*(tuple(dict.fromkeys(bounds[k])) for k in names))
    )
    values = tuple(evaluate(corner) for corner in corners)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("corner values must be finite")
    return CornerAnalysis(corners, values, min(range(len(values)), key=values.__getitem__))


@dataclass(frozen=True)
class ShadowValue:
    axis: str
    backward_difference: float | None
    forward_difference: float
    step: float
    interpretation: str = "finite differences; differentiability is not asserted"


def shadow_value(value, resources, axis, step, *, nonnegative=True):
    if (
        axis not in resources
        or not math.isfinite(step)
        or step <= 0
        or not all(math.isfinite(v) for v in resources.values())
        or nonnegative
        and resources[axis] < 0
    ):
        raise ValueError("invalid shadow-value coordinate/step")
    x = resources[axis]
    base = value(dict(resources))
    forward = value({**resources, axis: x + step})
    backward = None if nonnegative and x < step else value({**resources, axis: x - step})
    if not all(
        math.isfinite(v) for v in (base, forward) + (() if backward is None else (backward,))
    ):
        raise ValueError("shadow evaluation must be finite")
    return ShadowValue(
        axis, None if backward is None else (base - backward) / step, (forward - base) / step, step
    )


def separability_interaction(v00, v10, v01, v11):
    """2x2 information/resource factorial interaction, at a declared policy regime."""
    if not all(math.isfinite(v) for v in (v00, v10, v01, v11)):
        raise ValueError("factorial values must be finite")
    return v11 - v10 - v01 + v00

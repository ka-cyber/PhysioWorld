"""Capability-region visualization boundary with optional matplotlib."""

from __future__ import annotations

from typing import Any

from .region import CapabilityRegion


def region_points(region: CapabilityRegion) -> tuple[tuple[float, ...], ...]:
    return tuple(region.policy_values.values())


def plot_capability(
    region: CapabilityRegion, *, ax: Any = None, dimensions: tuple[int, int] = (0, 1)
) -> Any:
    if region.dimension < 2:
        raise ValueError("capability plot requires at least two dimensions")
    if ax is None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError as exc:
            raise RuntimeError("plotting requires the 'plot' optional dependency") from exc
        _, ax = plt.subplots()
    x, y = dimensions
    points = region_points(region)
    ax.scatter([point[x] for point in points], [point[y] for point in points])
    ax.set_xlabel(f"capability[{x}]")
    ax.set_ylabel(f"capability[{y}]")
    ax.set_title(region.representation)
    return ax

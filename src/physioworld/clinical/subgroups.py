"""Declared-metadata subgroup analysis without inferred sensitive attributes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class SubgroupResult:
    attribute: str
    groups: Mapping[str, float]
    counts: Mapping[str, int]
    note: str = "Groups use available legitimate metadata only."


def subgroup_metric(
    subject_ids: Sequence[str],
    metadata: Mapping[str, Mapping[str, str]],
    values: Sequence[float],
    *,
    attribute: str,
    metric: Callable[[Sequence[float]], float] = lambda values: sum(values) / len(values),
) -> SubgroupResult:
    if len(subject_ids) != len(values) or not subject_ids:
        raise ValueError("subject_ids and values must have equal non-zero length")
    grouped: dict[str, list[float]] = {}
    for subject_id, value in zip(subject_ids, values):
        if subject_id not in metadata or attribute not in metadata[subject_id]:
            continue
        grouped.setdefault(str(metadata[subject_id][attribute]), []).append(float(value))
    if not grouped:
        raise ValueError("declared subgroup metadata is unavailable; no attributes may be inferred")
    return SubgroupResult(
        attribute,
        {group: metric(items) for group, items in grouped.items()},
        {group: len(items) for group, items in grouped.items()},
    )

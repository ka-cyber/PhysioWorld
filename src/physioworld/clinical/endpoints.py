"""Clinical/functional endpoint metrics distinct from technical metrics."""

from __future__ import annotations

import math
from typing import Sequence


def binary_confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    if len(y_true) != len(y_pred) or not y_true or any(y not in (0, 1) for y in (*y_true, *y_pred)):
        raise ValueError("labels must have equal non-zero length")
    return {
        "tp": sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred)),
        "tn": sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred)),
        "fp": sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred)),
        "fn": sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred)),
    }


def binary_endpoints(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | None]:
    counts = binary_confusion(y_true, y_pred)
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "ppv": tp / (tp + fp) if tp + fp else None,
        "event_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def false_alarms_per_hour(
    y_true: Sequence[int], y_pred: Sequence[int], duration_hours: float
) -> float:
    if not math.isfinite(duration_hours) or duration_hours <= 0:
        raise ValueError("duration and labels are invalid")
    false_alarms = binary_confusion(y_true, y_pred)["fp"]
    return false_alarms / duration_hours


def lead_time_seconds(
    event_times: Sequence[float], prediction_times: Sequence[float]
) -> float | None:
    if not event_times or not prediction_times:
        return None
    if any(not math.isfinite(t) for t in (*event_times, *prediction_times)):
        raise ValueError("event and prediction times must be finite")
    return min(
        (
            event - prediction
            for event in event_times
            for prediction in prediction_times
            if prediction <= event
        ),
        default=None,
    )


def epoch_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("epoch labels must have equal non-zero length")
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

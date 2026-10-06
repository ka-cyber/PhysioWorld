"""One-to-one event matching and nonparametric survival endpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EventAlarmMetrics:
    events: int
    detected: int
    false_alarms: int
    recall: float | None
    precision: float | None
    false_alarms_per_hour: float
    lead_times: tuple[float, ...]


def event_alarm_metrics(
    event_times: Sequence[float],
    alarm_times: Sequence[float],
    *,
    lookback: float,
    exposure_hours: float,
) -> EventAlarmMetrics:
    """Evaluate one subject; each alarm/event can be matched at most once."""
    if (
        not math.isfinite(lookback)
        or lookback < 0
        or not math.isfinite(exposure_hours)
        or exposure_hours <= 0
        or any(not math.isfinite(t) for t in (*event_times, *alarm_times))
    ):
        raise ValueError("finite event times, lookback, and positive exposure required")
    unmatched = sorted(alarm_times)
    leads = []
    for event in sorted(event_times):
        eligible = [i for i, alarm in enumerate(unmatched) if event - lookback <= alarm <= event]
        if eligible:
            # Earliest qualifying alert reports the first available warning.
            alarm = unmatched.pop(eligible[0])
            leads.append(event - alarm)
    detected = len(leads)
    return EventAlarmMetrics(
        len(event_times),
        detected,
        len(unmatched),
        detected / len(event_times) if event_times else None,
        detected / len(alarm_times) if alarm_times else None,
        len(unmatched) / exposure_hours,
        tuple(leads),
    )


@dataclass(frozen=True)
class SurvivalPoint:
    time: float
    at_risk: int
    events: int
    censored: int
    survival: float
    cumulative_hazard: float


def kaplan_meier(
    durations: Sequence[float], observed_events: Sequence[int]
) -> tuple[SurvivalPoint, ...]:
    """Kaplan–Meier survival and Nelson–Aalen hazard; independent censoring assumed."""
    if (
        not durations
        or len(durations) != len(observed_events)
        or any(not math.isfinite(t) or t < 0 for t in durations)
        or any(e not in (0, 1) for e in observed_events)
    ):
        raise ValueError("valid durations and censoring indicators required")
    risk, survival, hazard = len(durations), 1.0, 0.0
    result = []
    for time in sorted(set(durations)):
        events = sum(t == time and e == 1 for t, e in zip(durations, observed_events))
        censored = sum(t == time and e == 0 for t, e in zip(durations, observed_events))
        survival *= 1 - events / risk
        hazard += events / risk
        result.append(SurvivalPoint(time, risk, events, censored, survival, hazard))
        risk -= events + censored
    return tuple(result)

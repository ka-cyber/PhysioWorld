"""Fitted model discrepancy is separate from measurement noise and numerical error."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from physioworld.uncertainty import SplitConformal


@dataclass(frozen=True)
class DiscrepancyModel:
    kind: str
    intercept: float
    slope: float
    fit_subject_ids: tuple[str, ...]
    source: str
    task: str
    state_range: tuple[float, float] | None

    def predict(self, model_output: float, *, state: float | None = None) -> float:
        if not math.isfinite(model_output):
            raise ValueError("finite model output required")
        if self.kind == "state_dependent":
            if (
                state is None
                or not math.isfinite(state)
                or self.state_range is None
                or not self.state_range[0] <= state <= self.state_range[1]
            ):
                raise ValueError("state-dependent discrepancy outside fitted support")
            return model_output + self.intercept + self.slope * state
        if self.kind == "multiplicative":
            return model_output * self.intercept
        return model_output + self.intercept

    def calibrate_intervals(
        self,
        model_outputs: Sequence[float],
        observations: Sequence[float],
        subject_ids: Sequence[str],
        *,
        states: Sequence[float] | None = None,
        alpha: float = 0.05,
    ) -> SplitConformal:
        if not len(model_outputs) == len(observations) == len(subject_ids):
            raise ValueError("aligned held-out discrepancy calibration data required")
        if states is not None and len(states) != len(observations):
            raise ValueError("aligned states required")
        corrected = [
            self.predict(output, state=states[i] if states is not None else None)
            for i, output in enumerate(model_outputs)
        ]
        return SplitConformal.fit(
            [y - p for y, p in zip(observations, corrected)],
            subject_ids,
            training_subject_ids=self.fit_subject_ids,
            alpha=alpha,
        )


def fit_discrepancy(
    model_outputs: Sequence[float],
    observations: Sequence[float],
    subject_ids: Sequence[str],
    *,
    kind: str = "additive",
    source: str,
    task: str,
    states: Sequence[float] | None = None,
) -> DiscrepancyModel:
    """OLS additive offset, multiplicative scale, or linear state-dependent residual.

    This is a descriptive fit; correlated windows do not yield independent fit CIs.
    Prediction interval calibration uses disjoint subjects through calibrate_intervals.
    """
    if (
        not observations
        or not len(model_outputs) == len(observations) == len(subject_ids)
        or any(not math.isfinite(v) for v in (*model_outputs, *observations))
        or any(not s for s in subject_ids)
        or not source
        or not task
        or kind not in ("additive", "multiplicative", "state_dependent")
    ):
        raise ValueError("explicit finite discrepancy data, subjects, source, and task required")
    residuals = [y - p for y, p in zip(observations, model_outputs)]
    intercept = sum(residuals) / len(residuals)
    slope, state_range = 0.0, None
    if kind == "multiplicative":
        denominator = sum(p * p for p in model_outputs)
        if denominator == 0:
            raise ValueError("multiplicative discrepancy is unidentified at zero model output")
        intercept = sum(p * y for p, y in zip(model_outputs, observations)) / denominator
    if kind == "state_dependent":
        if (
            states is None
            or len(states) != len(observations)
            or any(not math.isfinite(s) for s in states)
        ):
            raise ValueError("finite states required")
        mean = sum(states) / len(states)
        denominator = sum((s - mean) ** 2 for s in states)
        if denominator == 0:
            raise ValueError("state-dependent discrepancy requires state variation")
        slope = sum((s - mean) * r for s, r in zip(states, residuals)) / denominator
        intercept -= slope * mean
        state_range = min(states), max(states)
    return DiscrepancyModel(
        kind, intercept, slope, tuple(dict.fromkeys(subject_ids)), source, task, state_range
    )

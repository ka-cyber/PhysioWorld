"""Adaptive sensing and information-aware communication actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from physioworld.resources.budgets import MultiResourceBudget

from .base import Policy


@dataclass(frozen=True)
class SensingAction:
    operation: str
    modality: str | None = None
    sampling_rate_hz: float | None = None
    fidelity: str | None = None
    representation: str | None = None
    duration_s: float = 0.0
    resolution_bits: int | None = None
    active: bool = True

    def __post_init__(self):
        if (
            not self.operation
            or not math.isfinite(self.duration_s)
            or self.duration_s < 0
            or self.sampling_rate_hz is not None
            and (not math.isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0)
            or self.resolution_bits is not None
            and self.resolution_bits < 1
        ):
            raise ValueError("invalid sensing action")


@dataclass(frozen=True)
class FixedAcquisition(Policy):
    selected: SensingAction
    name: str = "fixed_acquisition"

    def action(self, state: object, available_actions: Sequence[SensingAction]) -> SensingAction:
        if self.selected not in available_actions:
            raise ValueError("fixed acquisition action is not admissible")
        return self.selected


@dataclass(frozen=True)
class QualityAwareAcquisition(Policy):
    quality_key: str = "sqi"
    name: str = "quality_aware_acquisition"

    def action(
        self, state: Mapping[str, float], available_actions: Sequence[SensingAction]
    ) -> SensingAction:
        if not available_actions:
            raise ValueError("no acquisition actions")
        # A quality policy may use only declared current-state information.
        quality = state.get(self.quality_key, 0.0)

        def score(candidate):
            value = (
                quality.get(candidate.modality, 0.0)
                if isinstance(quality, Mapping)
                else float(quality)
            )
            if not math.isfinite(value):
                raise ValueError("quality scores must be finite")
            return (candidate.operation == "sample", value)

        return max(available_actions, key=score)


@dataclass(frozen=True)
class PIVAwareAcquisition(Policy):
    piv: Mapping[str, float]
    name: str = "piv_aware_acquisition"

    def action(self, state: object, available_actions: Sequence[SensingAction]) -> SensingAction:
        return max(
            available_actions,
            key=lambda candidate: self.piv.get(candidate.modality or "", float("-inf")),
        )


@dataclass(frozen=True)
class JointPIVUncertaintyPolicy(Policy):
    piv: Mapping[str, float]
    uncertainty_penalty: Mapping[str, float]
    name: str = "joint_piv_uncertainty"

    def action(self, state: object, available_actions: Sequence[SensingAction]) -> SensingAction:
        return max(
            available_actions,
            key=lambda candidate: (
                self.piv.get(candidate.modality or "", 0.0)
                - self.uncertainty_penalty.get(candidate.modality or "", 0.0)
            ),
        )


@dataclass(frozen=True)
class OracleAcquisition(Policy):
    utility: Mapping[SensingAction, float]
    name: str = "oracle_acquisition"

    def action(self, state: object, available_actions: Sequence[SensingAction]) -> SensingAction:
        return max(
            available_actions, key=lambda candidate: self.utility.get(candidate, float("-inf"))
        )


@dataclass(frozen=True)
class CrossLayerAction:
    sensing: SensingAction
    compression_fraction: float = 1.0
    coding_rate: float = 1.0
    transmit: bool = True
    compute_location: str = "local"
    inference_id: str = "full"

    def __post_init__(self):
        if (
            not math.isfinite(self.compression_fraction)
            or not 0 < self.compression_fraction <= 1
            or not math.isfinite(self.coding_rate)
            or not 0 < self.coding_rate <= 1
            or not self.compute_location
            or not self.inference_id
        ):
            raise ValueError("invalid cross-layer action")


@dataclass(frozen=True)
class CrossLayerObjective:
    objective_id: str
    utility: Callable[[Mapping[str, Any], CrossLayerAction], float]
    penalties: Mapping[str, float]
    uncertainty_weight: float = 0.0

    def __post_init__(self):
        if (
            not self.objective_id
            or not math.isfinite(self.uncertainty_weight)
            or self.uncertainty_weight < 0
            or not all(math.isfinite(v) and v >= 0 for v in self.penalties.values())
        ):
            raise ValueError("explicit finite cross-layer objective required")


@dataclass(frozen=True)
class JointCrossLayerPolicy(Policy):
    objective: CrossLayerObjective
    cost: Callable[[Mapping[str, Any], CrossLayerAction], Mapping[str, float]]
    uncertainty: Callable[[Mapping[str, Any], CrossLayerAction], float]
    safe: Callable[[Mapping[str, Any], CrossLayerAction], bool]
    name: str = "joint_cross_layer"

    def action(self, state, available_actions):
        budget = state.get("budget")
        if not isinstance(budget, MultiResourceBudget):
            raise TypeError("joint policy requires a typed multi-resource budget")
        candidates = []
        for action in available_actions:
            cost = self.cost(state, action)
            if not budget.permits(cost) or not self.safe(state, action):
                continue
            utility = self.objective.utility(state, action)
            uncertainty = self.uncertainty(state, action)
            if not math.isfinite(utility) or not math.isfinite(uncertainty) or uncertainty < 0:
                raise ValueError("utility and uncertainty must be finite; uncertainty nonnegative")
            score = (
                utility
                - math.fsum(self.objective.penalties.get(k, 0.0) * v for k, v in cost.items())
                - self.objective.uncertainty_weight * uncertainty
            )
            candidates.append((score, action))
        if not candidates:
            raise RuntimeError("no safe budget-feasible cross-layer action")
        return max(candidates, key=lambda pair: pair[0])[1]

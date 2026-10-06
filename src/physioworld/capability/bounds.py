"""Conditional finite/infinite-horizon bounds with explicit TV convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from physioworld.core.contracts import EvidenceClass, VersionedRecord, finite
from physioworld.core.evidence import AssumptionSet
from physioworld.core.flow import FiniteKernel


@dataclass(frozen=True)
class ValueTransferBound(VersionedRecord):
    bound: float
    stage_bounds: tuple[float, ...]
    tv_convention: str
    assumptions: AssumptionSet
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION


def finite_kernel_error_bound(
    reward_errors: Sequence[float],
    transition_tv_errors: Sequence[float],
    continuation_absolute_bounds: Sequence[float],
    *,
    assumptions: AssumptionSet,
    terminal_error: float = 0.0,
) -> ValueTransferBound:
    """Delta_t <= eps_r,t + 2*M_{t+1}*TV_t + Delta_{t+1}, TV=half L1."""
    assumptions.require(
        (
            "common_feasible_actions",
            "uniform_reward_error",
            "uniform_transition_tv",
            "bounded_continuation_value",
        )
    )
    if len(reward_errors) == 0 or not (
        len(reward_errors) == len(transition_tv_errors) == len(continuation_absolute_bounds)
    ):
        raise ValueError("finite-horizon approximation arrays must have equal nonzero length")
    for value in (
        *reward_errors,
        *transition_tv_errors,
        *continuation_absolute_bounds,
        terminal_error,
    ):
        finite(value, "approximation bound")
        if value < 0:
            raise ValueError("approximation bounds cannot be negative")
    if any(value > 1 for value in transition_tv_errors):
        raise ValueError("half-L1 TV errors must lie in [0,1]")
    reward_errors = tuple(float(value) for value in reward_errors)
    transition_tv_errors = tuple(float(value) for value in transition_tv_errors)
    continuation_absolute_bounds = tuple(float(value) for value in continuation_absolute_bounds)
    error, stages = float(terminal_error), [float(terminal_error)]
    for reward, tv, bound in reversed(
        tuple(zip(reward_errors, transition_tv_errors, continuation_absolute_bounds))
    ):
        error = reward + 2 * bound * tv + error
        stages.append(error)
    return ValueTransferBound(error, tuple(reversed(stages)), "TV=0.5*L1", assumptions)


def bellman_contraction_bound(gamma: float, sup_norm_difference: float) -> float:
    """Bound ||TV-TW||inf for the same discounted Bellman operator/action sets."""
    finite(gamma, "discount")
    finite(sup_norm_difference, "sup norm difference")
    if not 0 <= gamma < 1 or sup_norm_difference < 0:
        raise ValueError("discount must lie in [0,1), norm difference must be nonnegative")
    return gamma * sup_norm_difference


@dataclass(frozen=True)
class DistributionStabilityBound(VersionedRecord):
    finite_time_bound: float
    stationary_bound: float | None
    steps: int
    contraction: float
    kernel_tv_error: float
    estimand: str = "distribution_total_variation"
    tv_convention: str = "TV=0.5*L1"


def dobrushin_stability_bound(
    reference: FiniteKernel, perturbed: FiniteKernel, *, steps: int, initial_tv_error: float = 0.0
) -> DistributionStabilityBound:
    """e_next <= alpha e + eps; no discounted-value or safety-probability substitution."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    finite(initial_tv_error, "initial TV error")
    if not 0 <= initial_tv_error <= 1:
        raise ValueError("initial TV error must lie in [0,1]")
    error = reference.uniform_tv_error(perturbed)
    alpha = reference.dobrushin_coefficient()
    bound = initial_tv_error
    for _ in range(steps):
        bound = min(1.0, alpha * bound + error)
    stationary = min(1.0, error / (1 - alpha)) if alpha < 1 else None
    return DistributionStabilityBound(bound, stationary, steps, alpha, error)

"""Declared finite-policy robustness and capability/policy-emulation relations."""

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .models import ResourceState


@dataclass(frozen=True)
class ReachabilityResult:
    reachable: bool
    states: tuple[ResourceState, ...]
    violating_step: int | None = None
    reason: str | None = None


def reachability(model, initial, currents, duration):
    state, states = initial, [initial]
    for index, current in enumerate(currents):
        transition = model.step(state, current, duration)
        states.append(transition.next)
        if not transition.feasible:
            return ReachabilityResult(False, tuple(states), index, transition.reason)
        state = transition.next
    return ReachabilityResult(True, tuple(states))


@dataclass(frozen=True)
class RobustPolicyResult:
    value: float
    policy: Any
    worst_parameters: tuple[Any, ...]
    policy_worst_values: Mapping[Any, float]
    clairvoyant_upper_bound: float
    policy_weights: Mapping[Any, float]
    formulation: str = "sup_pi inf_theta"


def optimize_robust_policy(values, *, feasible=None, randomized=False):
    """Optimize one policy against all columns of a policy-by-parameter table.

    Rows are complete nonanticipative policies, not actions chosen after
    seeing theta. A false feasibility entry excludes that entire policy.
    Randomization, when requested, is chosen *before* theta and uses an LP
    (optional SciPy). Nature cannot observe the random draw in that formulation.
    """
    if not isinstance(values, Mapping) or not values:
        raise ValueError("provide a nonempty policy -> parameter -> value table")
    policies = tuple(values)
    if not all(isinstance(values[p], Mapping) and values[p] for p in policies):
        raise ValueError("flat independently optimized values are not a robust policy problem")
    parameters = tuple(values[policies[0]])
    if any(set(values[p]) != set(parameters) for p in policies):
        raise ValueError("all policies must share the same uncertainty set")
    if not all(math.isfinite(v) for row in values.values() for v in row.values()):
        raise ValueError("policy values must be finite")
    admissible = tuple(
        p for p in policies if feasible is None or all(feasible[p][theta] for theta in parameters)
    )
    if not admissible:
        raise ValueError("no policy is safe throughout the uncertainty set")
    worst = {p: min(values[p].values()) for p in admissible}
    clairvoyant = min(max(values[p][theta] for p in admissible) for theta in parameters)
    if not randomized:
        selected = max(admissible, key=lambda policy: worst[policy])
        value = worst[selected]
        weights = {selected: 1.0}
    else:
        try:
            from scipy.optimize import linprog
        except ImportError as error:
            raise ImportError("randomized robust policies require scipy") from error
        result = linprog(
            [0.0] * len(admissible) + [-1.0],
            A_ub=[[-values[p][theta] for p in admissible] + [1.0] for theta in parameters],
            b_ub=[0.0] * len(parameters),
            A_eq=[[1.0] * len(admissible) + [0.0]],
            b_eq=[1.0],
            bounds=[(0.0, 1.0)] * len(admissible) + [(None, None)],
            method="highs",
        )
        if not result.success:
            raise ArithmeticError(result.message)
        weights = {p: float(w) for p, w in zip(admissible, result.x[:-1]) if w > 0}
        mass = math.fsum(weights.values())
        weights = {p: w / mass for p, w in weights.items()}
        value, selected = float(result.x[-1]), "randomized_mixture"
    column_values = {t: math.fsum(weights[p] * values[p][t] for p in weights) for t in parameters}
    value = min(column_values.values())
    worst_parameters = tuple(t for t in parameters if abs(column_values[t] - value) <= 1e-9)
    return RobustPolicyResult(value, selected, worst_parameters, worst, clairvoyant, weights)


def robust_value(values, order="sup_inf", *, feasible=None, randomized=False):
    if order != "sup_inf":
        raise ValueError("robust_value is common-policy sup_inf; clairvoyance is an upper bound")
    return optimize_robust_policy(values, feasible=feasible, randomized=randomized).value


@dataclass(frozen=True)
class ResourceCapability:
    """Finite attainable vectors under one task/objective/resource declaration."""

    policy_values: Mapping[Any, tuple[float, ...]]
    task_ids: tuple[str, ...]
    regime_id: str
    horizon: int
    uncertainty_id: str = "nominal"

    def __post_init__(self):
        if not self.task_ids or not self.regime_id or self.horizon < 1 or not self.policy_values:
            raise ValueError("capability needs tasks, regime, horizon, and policies")
        if any(
            len(v) != len(self.task_ids) or not all(math.isfinite(x) for x in v)
            for v in self.policy_values.values()
        ):
            raise ValueError("invalid attainable capability vectors")


def resource_dominates(
    a, b, *, relation="capability_inclusion", emulation=None, strict=False, tolerance=1e-12
):
    """Finite deterministic relation, scoped to declared task requirement sets.

    capability_inclusion compares downward closures of *attainable joint
    vectors*. policy_emulation additionally requires an explicit b->a policy
    mapping. This does not assert universal physical dominance or infer joint
    attainability from coordinatewise maxima. Convex mixtures are not implicit.
    """
    if not isinstance(a, ResourceCapability) or not isinstance(b, ResourceCapability):
        raise TypeError("dominance requires declared capabilities, not numeric resource profiles")
    if (a.task_ids, a.regime_id, a.horizon, a.uncertainty_id) != (
        b.task_ids,
        b.regime_id,
        b.horizon,
        b.uncertainty_id,
    ):
        raise ValueError("capability estimands must match")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid dominance tolerance")

    def covers(left, right):
        return all(x + tolerance >= y for x, y in zip(left, right))

    if relation == "policy_emulation":
        if emulation is None or set(emulation) != set(b.policy_values):
            raise ValueError("provide an emulating policy for every target policy")
        inclusion = all(
            emulation[p] in a.policy_values and covers(a.policy_values[emulation[p]], v)
            for p, v in b.policy_values.items()
        )
    elif relation == "capability_inclusion":
        inclusion = all(
            any(covers(x, y) for x in a.policy_values.values()) for y in b.policy_values.values()
        )
    else:
        raise ValueError("unknown scientific dominance relation")
    reverse = all(
        any(covers(y, x) for y in b.policy_values.values()) for x in a.policy_values.values()
    )
    return inclusion and (not strict or not reverse)


def deliverable_current(model, initial, duration, *, upper=None, tolerance_a=1e-10):
    """Bisection for a passive circuit's constant-current safety envelope."""
    if (
        duration < 0
        or not math.isfinite(duration)
        or not math.isfinite(tolerance_a)
        or tolerance_a <= 0
    ):
        raise ValueError("invalid current search")
    upper = model.max_current_a if upper is None else upper
    if not math.isfinite(upper) or upper < 0:
        raise ValueError("search upper bound must be finite and nonnegative")

    def feasible(current):
        return model.step(initial, current, duration).feasible

    # Feasibility need not be monotone from zero: a small discharge can be
    # required to leave an over-voltage state. Locate a feasible interval first,
    # then refine its upper boundary. This never silently reports zero for a
    # valid positive deliverable current.
    if feasible(upper):
        return upper
    grid = 256
    feasible_current = None
    previous = 0.0
    previous_ok = feasible(previous)
    if previous_ok:
        feasible_current = previous
    for index in range(1, grid + 1):
        current = upper * index / grid
        current_ok = feasible(current)
        if current_ok:
            feasible_current = current
        elif previous_ok and feasible_current is not None:
            break
        previous, previous_ok = current, current_ok
    if feasible_current is None:
        raise ValueError("no feasible constant-current operating point in the declared envelope")
    low = feasible_current
    high = upper
    while high - low > tolerance_a:
        mid = (low + high) / 2
        if feasible(mid):
            low = mid
        else:
            high = mid
    return low


@dataclass(frozen=True)
class ResourceScheduleSegment:
    duration_s: float
    load_current_a: float
    charging_current_a: float = 0.0
    operation_id: str = "resource_segment"

    def __post_init__(self):
        if not self.operation_id or not all(
            math.isfinite(v) and v >= 0
            for v in (self.duration_s, self.load_current_a, self.charging_current_a)
        ):
            raise ValueError("invalid physical charge/discharge schedule segment")


def physical_schedule(model, initial, segments):
    """Contiguous SI-time segments; first unsafe trajectory halts scheduling."""
    states, state, transitions = [initial], initial, []
    for index, segment in enumerate(segments):
        transition = model.step(
            state,
            segment.load_current_a,
            segment.duration_s,
            charging_current=segment.charging_current_a,
        )
        states.append(transition.next)
        transitions.append(transition)
        if not transition.feasible:
            return ReachabilityResult(False, tuple(states), index, transition.reason), tuple(
                transitions
            )
        state = transition.next
    return ReachabilityResult(True, tuple(states)), tuple(transitions)

"""Finite-kernel bounds with TV = one half of the L1 probability distance."""

import math


def validate_kernel(kernel):
    n = len(kernel)
    if not n or any(len(row) != n for row in kernel):
        raise ValueError("kernel must be nonempty and square")
    for row in kernel:
        if not all(math.isfinite(v) and v >= 0 for v in row) or abs(math.fsum(row) - 1) > 1e-12:
            raise ValueError("kernel rows must be probability distributions")


def total_variation(p, q):
    if not p or len(p) != len(q):
        raise ValueError("probability dimensions must match")
    for values in (p, q):
        if (
            not all(math.isfinite(v) and v >= 0 for v in values)
            or abs(math.fsum(values) - 1) > 1e-12
        ):
            raise ValueError("invalid probability distribution")
    return 0.5 * math.fsum(abs(x - y) for x, y in zip(p, q))


def dobrushin_coefficient(kernel):
    """delta(P) = max_{x,y} TV(P(x,.),P(y,.))."""
    validate_kernel(kernel)
    return max(total_variation(p, q) for p in kernel for q in kernel)


def kernel_perturbation(p, q):
    validate_kernel(p)
    validate_kernel(q)
    if len(p) != len(q):
        raise ValueError("kernels must share the same state space")
    return max(total_variation(x, y) for x, y in zip(p, q))


def finite_value_transfer_bound(
    reward_errors,
    transition_tv_errors,
    next_value_absolute_bounds,
    *,
    gamma=1.0,
    terminal_error=0.0,
):
    """Delta_t <= er_t + gamma*(2*M_{t+1}*eP_t + Delta_{t+1}).

    Same states, actions, safety/admissibility, reward interpretation and
    terminal horizon; |Vhat_{t+1}|<=M. Applies to fixed policies or optimal
    values with shared action sets. No bound for changing feasible actions.
    """
    if not len(reward_errors) == len(transition_tv_errors) == len(next_value_absolute_bounds):
        raise ValueError("time-varying error arrays must align")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("invalid discount")
    if not all(
        math.isfinite(v) and v >= 0
        for v in (
            *reward_errors,
            *transition_tv_errors,
            *next_value_absolute_bounds,
            terminal_error,
        )
    ):
        raise ValueError("error and value bounds must be finite and nonnegative")
    if any(v > 1 for v in transition_tv_errors):
        raise ValueError("TV cannot exceed one")
    delta, bounds = terminal_error, [terminal_error]
    for er, ep, m in reversed(
        tuple(zip(reward_errors, transition_tv_errors, next_value_absolute_bounds))
    ):
        delta = er + gamma * (2 * m * ep + delta)
        bounds.append(delta)
    return tuple(reversed(bounds))


def discounted_value_transfer_bound(
    reward_error, transition_tv_error, reward_absolute_bound, gamma
):
    """Shared-action discounted simulation bound; not an ergodic bound."""
    if not all(
        math.isfinite(v) for v in (reward_error, transition_tv_error, reward_absolute_bound, gamma)
    ):
        raise ValueError("bounds must be finite")
    if (
        reward_error < 0
        or reward_absolute_bound < 0
        or not 0 <= transition_tv_error <= 1
        or not 0 <= gamma < 1
    ):
        raise ValueError("invalid discounted bounds")
    return (
        reward_error / (1 - gamma)
        + 2 * gamma * reward_absolute_bound * transition_tv_error / (1 - gamma) ** 2
    )


def distribution_error_bounds(delta, epsilon, steps, *, initial_tv=0.0):
    """d_{t+1} <= delta*d_t+epsilon; also bounds event-probability errors.

    delta is a uniform coefficient for the declared fixed/controlled kernels;
    an optimal-policy switch is not automatically covered.
    """
    if (
        not all(math.isfinite(v) and 0 <= v <= 1 for v in (delta, epsilon, initial_tv))
        or not isinstance(steps, int)
        or steps < 0
    ):
        raise ValueError("invalid distributional stability bounds")
    result = [initial_tv]
    for _ in range(steps):
        result.append(min(1.0, delta * result[-1] + epsilon))
    return tuple(result)


def stationary_distribution_error_bound(delta, epsilon):
    """Finite Markov kernels with delta(P)<1: TV(mu_P,mu_Q)<=eps/(1-delta).

    Q has an invariant distribution (automatic on finite state spaces); all
    such invariant distributions satisfy the bound. This is not a discounted
    value, average reward, or hardware safety certificate.
    """
    if (
        not math.isfinite(delta)
        or not 0 <= delta < 1
        or not math.isfinite(epsilon)
        or not 0 <= epsilon <= 1
    ):
        raise ValueError("strict contraction and a valid TV perturbation required")
    return min(1.0, epsilon / (1 - delta))


def stationary_average_reward_error_bound(delta, epsilon, reward_span, *, reward_error=0.0):
    """Fixed-kernel stationary average reward, distinct from discounted control.

    One reward has declared span max(r)-min(r); pointwise cross-model reward
    error is bounded by reward_error. No optimized-policy sensitivity claim.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (reward_span, reward_error)):
        raise ValueError("finite nonnegative reward span/error required")
    return reward_error + reward_span * stationary_distribution_error_bound(delta, epsilon)

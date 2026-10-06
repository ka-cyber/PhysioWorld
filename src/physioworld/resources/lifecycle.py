"""Optional phenomenological thermal/throughput-ESR lifecycle model.

Capacitance stays fixed; degradation increases ESR with cumulative absolute
port charge. This is a declared empirical law, not an electrochemical ageing
theory. The coupled nonlinear extension uses DOP853 with caller-visible error
tolerances; the base LTI electrical model retains its exponential transition.
"""

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from .calibration import ParameterStatus
from .models import EnergyBalance, ResourceEvent, ResourceState, ResourceTransition


@dataclass(frozen=True)
class LifecycleLaw:
    thermal_capacitance_j_k: float
    thermal_resistance_k_w: float
    ambient_temperature_c: float
    maximum_temperature_c: float
    esr_growth_per_coulomb: float
    esr_temperature_coefficient_per_k: float
    reference_temperature_c: float
    status: ParameterStatus
    source: str

    def __post_init__(self):
        if (
            not self.source
            or not isinstance(self.status, ParameterStatus)
            or self.status == ParameterStatus.UNAVAILABLE
            or not all(
                math.isfinite(v) for k, v in self.__dict__.items() if k not in ("status", "source")
            )
            or min(self.thermal_capacitance_j_k, self.thermal_resistance_k_w) <= 0
            or self.esr_growth_per_coulomb < 0
            or min(
                self.ambient_temperature_c, self.maximum_temperature_c, self.reference_temperature_c
            )
            < -273.15
        ):
            raise ValueError(
                "lifecycle law requires finite valid parameters and declared provenance"
            )


@dataclass(frozen=True)
class LifecycleState:
    electrical: ResourceState
    temperature_c: float
    throughput_coulombs: float = 0.0

    def __post_init__(self):
        if (
            not math.isfinite(self.temperature_c)
            or self.temperature_c < -273.15
            or not math.isfinite(self.throughput_coulombs)
            or self.throughput_coulombs < 0
        ):
            raise ValueError("invalid lifecycle state")


@dataclass(frozen=True)
class LifecycleTransition:
    electrical: ResourceTransition
    next: LifecycleState
    thermal_energy_change_j: float
    cooling_work_j: float
    thermal_residual_j: float
    esr_ohm: float
    rtol: float
    atol: float


class _LifecycleEvent:
    """A scipy event function carrying the attributes scipy reads."""

    terminal = True
    direction = -1

    def __init__(self, function: Callable[[float, Sequence[float]], float]) -> None:
        self._function = function

    def __call__(self, time: float, state: Sequence[float]) -> float:
        return self._function(time, state)


class LifecycleResource:
    def __init__(self, resource, law):
        if resource.converter is not None:
            raise ValueError("lifecycle extension currently requires a direct electrical port")
        self.resource, self.law = resource, law

    def step(self, state, current, duration, *, charging_current=0.0, rtol=1e-11, atol=1e-12):
        if (
            not all(math.isfinite(v) for v in (current, duration, charging_current, rtol, atol))
            or min(current, duration, charging_current) < 0
            or min(rtol, atol) <= 0
        ):
            raise ValueError("invalid lifecycle controls/tolerances")
        from scipy.integrate import solve_ivp

        model, law = self.resource, self.law
        x0 = model._vector(state.electrical)
        n, net = len(x0), current - charging_current

        def resistance(temp, throughput):
            factor = (1 + law.esr_growth_per_coulomb * throughput) * (
                1 + law.esr_temperature_coefficient_per_k * (temp - law.reference_temperature_c)
            )
            if factor <= 0:
                raise ValueError("ESR law outside its positive-resistance operating domain")
            return model.esr_ohm * factor

        def derivative(t, y):
            x, temp, throughput = y[:n], y[n], y[n + 1]
            r = resistance(temp, throughput)
            internal = x[0] - sum(x[1:])
            a, b = model._matrices(net)
            dx = [sum(v * w for v, w in zip(row, x)) + source for row, source in zip(a, b)]
            esr, leakage, polarization = (
                r * net**2,
                model._leakage_loss(x),
                model._polarization_loss(x),
            )
            cooling = (temp - law.ambient_temperature_c) / law.thermal_resistance_k_w
            return [
                *dx,
                (esr + leakage + polarization - cooling) / law.thermal_capacitance_j_k,
                abs(net),
                (internal - r * net) * net,
                esr,
                leakage,
                polarization,
                cooling,
            ]

        def undervoltage(t, y):
            return (
                y[0]
                - sum(y[1:n])
                - resistance(y[n], y[n + 1]) * net
                - model.v_min
                - model.safety_margin_v
            )

        def overvoltage(t, y):
            return model.v_max - (y[0] - sum(y[1:n]) - resistance(y[n], y[n + 1]) * net)

        def overheating(t, y):
            return law.maximum_temperature_c - y[n]

        functions = tuple(
            _LifecycleEvent(function) for function in (undervoltage, overvoltage, overheating)
        )
        initial = [*x0, state.temperature_c, state.throughput_coulombs, 0.0, 0.0, 0.0, 0.0, 0.0]
        reason = None
        events: tuple[ResourceEvent, ...] = ()
        if current > model.max_current_a or charging_current > model.max_charge_current_a:
            reason = "current envelope exceeded"
        if any(f(0.0, initial) < 0 for f in functions):
            reason = reason or "initial lifecycle state outside safety bounds"
        if reason or duration == 0:
            final, elapsed = initial, 0.0
        else:
            solution = solve_ivp(
                derivative,
                (0, duration),
                initial,
                method="DOP853",
                rtol=rtol,
                atol=atol,
                events=functions,
                max_step=duration / 32,
            )
            if not solution.success:
                raise ArithmeticError(solution.message)
            final, elapsed = solution.y[:, -1], float(solution.t[-1])
            terminal = final[0] - sum(final[1:n]) - resistance(final[n], final[n + 1]) * net
            thresholds = (
                model.v_min + model.safety_margin_v,
                model.v_max,
                law.maximum_temperature_c,
            )
            values = (terminal, terminal, final[n])
            events = tuple(
                ResourceEvent(
                    kind, state.electrical.time + float(times[0]), float(value), threshold
                )
                for kind, value, threshold, times in zip(
                    ("undervoltage", "overvoltage", "overheating"),
                    values,
                    thresholds,
                    solution.t_events,
                )
                if len(times)
            )
            if events:
                reason = events[0].kind
        x = tuple(map(float, final[:n]))
        electrical = ResourceState(
            x[0],
            x[1:],
            model.capacitance_f * x[0],
            state.electrical.time + elapsed,
            current,
            charging_current,
            dict(state.electrical.metadata),
            state.electrical.converter_ready,
        )
        r = resistance(final[n], final[n + 1])
        work, esr, leakage, polarization, cooling = map(float, final[n + 2 :])
        balance = EnergyBalance(
            model._stored(x) - model._stored(x0),
            work,
            esr,
            leakage,
            polarization,
            integration_error_j=None,
        )
        transition = ResourceTransition(
            state.electrical,
            electrical,
            elapsed,
            x[0] - sum(x[1:]) - r * net,
            reason is None,
            reason,
            events,
            balance,
            "DOP853_lifecycle",
        )
        thermal_change = law.thermal_capacitance_j_k * (final[n] - state.temperature_c)
        return LifecycleTransition(
            transition,
            LifecycleState(electrical, float(final[n]), float(final[n + 1])),
            float(thermal_change),
            cooling,
            float(thermal_change - esr - leakage - polarization + cooling),
            float(r),
            rtol,
            atol,
        )

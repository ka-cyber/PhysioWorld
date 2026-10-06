"""Passive P0/P1 and Foster P2 circuits with SI units and signed port work.

Foster topology: series bulk C, ESR, and parallel R_i/C_i polarization
elements (C_i=tau_i/R_i). Leakage shunts the *internal terminal* v-sum(x_i).
Thus j=I_load-I_charge+g*(v-sum(x_i)), v'=-j/C,
x_i'=(R_i*j-x_i)/tau_i. Terminal leakage and resistive loads couple every
state. Under ideal imposed current and zero leakage the circuit correctly
reduces to independent branch equations; this is a physical special case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from physioworld.core import EnergyStatus, EnergyValue

from .linalg import affine_transition, integrate, matvec, norm
from .models import EnergyBalance, ResourceEvent, ResourceState, ResourceTransition


def _finite(*values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError("physical parameters and controls must be finite")


def canonical_current_limit(v0, vmin, capacitance, esr, duration):
    """No-leak constant-current bound, I=(v0-vmin)/(T/C+ESR).

    The prompt names this P0 although its declared ESR is 8 ohm; a strictly
    ideal, ESR-free P0 instead gives C*(v0-vmin)/T.
    """
    _finite(v0, vmin, capacitance, esr, duration)
    if capacitance <= 0 or duration < 0 or esr < 0 or v0 < vmin:
        raise ValueError("invalid capacitor limit parameters")
    denominator = duration / capacitance + esr
    if denominator == 0:
        raise ValueError("zero duration and ESR do not define a finite current limit")
    return (v0 - vmin) / denominator


@dataclass(frozen=True)
class ConverterConstraints:
    efficiency: float = 1.0
    minimum_input_v: float = 0.0
    maximum_input_v: float = 1e6
    maximum_current_a: float = 1e6
    maximum_output_power_w: float = 1e6
    startup_current_a: float = 0.0
    startup_duration_s: float = 0.0

    def __post_init__(self):
        _finite(*self.__dict__.values())
        if (
            not 0 < self.efficiency <= 1
            or self.minimum_input_v < 0
            or self.maximum_input_v < self.minimum_input_v
            or min(
                self.maximum_current_a,
                self.maximum_output_power_w,
                self.startup_current_a,
                self.startup_duration_s,
            )
            < 0
        ):
            raise ValueError("invalid converter constraints")


@dataclass(frozen=True)
class CapacitorResource:
    capacitance_f: float
    esr_ohm: float = 0.0
    leakage_siemens: float = 0.0
    v_min: float = 0.0
    v_max: float = 1e6
    model_name: str = "capacitor_p0"
    max_current_a: float = 1e6
    max_charge_current_a: float = 1e6
    converter: ConverterConstraints | None = None
    safety_margin_v: float = 0.0

    def __post_init__(self):
        _finite(
            self.capacitance_f,
            self.esr_ohm,
            self.leakage_siemens,
            self.v_min,
            self.v_max,
            self.max_current_a,
            self.max_charge_current_a,
            self.safety_margin_v,
        )
        if (
            self.capacitance_f <= 0
            or min(
                self.esr_ohm,
                self.leakage_siemens,
                self.v_min,
                self.max_current_a,
                self.max_charge_current_a,
                self.safety_margin_v,
            )
            < 0
            or self.v_min + self.safety_margin_v > self.v_max
        ):
            raise ValueError("invalid resource parameters")
        if self.model_name == "capacitor_p0" and (self.esr_ohm or self.leakage_siemens):
            object.__setattr__(self, "model_name", "capacitor_p1")

    def _vector(self, state):
        if state.internal_states:
            raise ValueError("P0/P1 does not accept polarization states")
        return (state.voltage,)

    def _matrices(self, net_current, load_resistance=None):
        conductance = self.leakage_siemens
        if load_resistance is not None:
            conductance += 1 / (load_resistance + self.esr_ohm)
        return ((-conductance / self.capacitance_f,),), (-net_current / self.capacitance_f,)

    def _stored(self, vector):
        return 0.5 * self.capacitance_f * vector[0] ** 2

    def energy(self, state):
        return EnergyValue(
            self._stored(self._vector(state)), EnergyStatus.MODEL_DERIVED, self.model_name
        )

    def charge(self, state):
        return self.capacitance_f * state.voltage

    def theoretical_energy(self, voltage):
        _finite(voltage)
        return EnergyValue(
            0.5 * self.capacitance_f * voltage**2, EnergyStatus.THEORETICAL, self.model_name
        )

    def terminal_voltage(self, state, current):
        _finite(current)
        vector = self._vector(state)
        return vector[0] - sum(vector[1:]) - current * self.esr_ohm

    def feasible(self, state, current):
        terminal = self.terminal_voltage(state, current)
        lower = (
            0.0
            if current < 0 or self.converter and not state.converter_ready and current == 0
            else self.v_min + self.safety_margin_v
        )
        upper = self.v_max
        if self.converter:
            if current > 0 and not state.converter_ready:
                return False
            if state.converter_ready and current >= 0:
                lower = max(lower, self.converter.minimum_input_v)
            upper = min(upper, self.converter.maximum_input_v)
            if current > 0 and (
                current > self.converter.maximum_current_a
                or self.converter.efficiency * current * terminal
                > self.converter.maximum_output_power_w
            ):
                return False
        return (
            lower <= terminal <= upper
            and -self.max_charge_current_a <= current <= self.max_current_a
            and state.voltage >= 0
        )

    def constant_current_limit(self, v0, duration):
        _finite(v0, duration)
        if duration < 0 or v0 < self.v_min + self.safety_margin_v:
            raise ValueError("invalid initial voltage or duration")
        current_cap = self.max_current_a
        if self.converter is not None:
            current_cap = min(current_cap, self.converter.maximum_current_a)
            # This is a conservative input-power envelope: terminal voltage is
            # no greater than the starting voltage during constant discharge.
            current_cap = min(
                current_cap,
                self.converter.maximum_output_power_w
                / (self.converter.efficiency * max(v0, self.converter.minimum_input_v, 1e-300)),
            )
            target_min = max(self.v_min + self.safety_margin_v, self.converter.minimum_input_v)
        else:
            target_min = self.v_min + self.safety_margin_v
        if not self.leakage_siemens:
            return min(
                current_cap,
                canonical_current_limit(v0, target_min, self.capacitance_f, self.esr_ohm, duration),
            )
        g = self.leakage_siemens
        decay = math.exp(-g * duration / self.capacitance_f)
        denominator = -math.expm1(-g * duration / self.capacitance_f) / g + self.esr_ohm
        if not denominator:
            raise ValueError("zero duration and ESR do not define a finite limit")
        return min(current_cap, max(0.0, (v0 * decay - target_min) / denominator))

    def step(self, state, current, duration, *, charging_current=0.0):
        _finite(current, duration, charging_current)
        if min(current, duration, charging_current) < 0:
            raise ValueError("use nonnegative discharge/charging controls")
        if self.converter and not state.converter_ready and current > 0:
            return self._blocked_converter(state)
        return self._step(state, current, duration, charging_current)

    def _blocked_converter(self, state):
        return ResourceTransition(
            state,
            state,
            0.0,
            self.terminal_voltage(state, 0.0),
            False,
            "converter startup required",
            (ResourceEvent("converter_not_ready", state.time, 0.0, 1.0),),
            EnergyBalance(0, 0, 0, 0),
            "blocked_converter",
        )

    def step_load_resistance(self, state, load_resistance, duration, *, charging_current=0.0):
        _finite(load_resistance)
        if load_resistance <= 0:
            raise ValueError("load resistance must be positive")
        _finite(duration, charging_current)
        if min(duration, charging_current) < 0:
            raise ValueError("duration and charging current must be nonnegative")
        if self.converter and not state.converter_ready:
            return self._blocked_converter(state)
        return self._step(state, 0.0, duration, charging_current, load_resistance)

    def _step(self, state, current, duration, charging_current, resistance=None, *, startup=False):
        _finite(current, duration, charging_current)
        if min(duration, current, charging_current) < 0:
            raise ValueError("use nonnegative discharge/charging controls")
        x0 = self._vector(state)
        if state.charge_coulombs is not None and not math.isclose(
            state.charge_coulombs, self.capacitance_f * state.voltage, rel_tol=2e-12, abs_tol=2e-12
        ):
            raise ValueError("charge must agree with bulk capacitance and voltage")
        net_forcing = (
            current - charging_current
            if resistance is None
            else -charging_current * resistance / (resistance + self.esr_ohm)
        )
        a, b = self._matrices(net_forcing, resistance)
        cache = {0.0: x0}

        def vector(t):
            if t not in cache:
                cache[t] = affine_transition(a, b, x0, t)
            return cache[t]

        def port(t):
            x = vector(t)
            internal = x[0] - sum(x[1:])
            load = (
                current
                if resistance is None
                else (internal + self.esr_ohm * charging_current) / (resistance + self.esr_ohm)
            )
            net = load - charging_current
            return internal - self.esr_ohm * net, net, load

        final = vector(duration)
        terminal, _, load = port(duration)
        next_state = ResourceState(
            final[0],
            tuple(final[1:]),
            self.capacitance_f * final[0],
            state.time + duration,
            load,
            charging_current,
            dict(state.metadata),
            state.converter_ready,
        )
        lower, upper = self.v_min + self.safety_margin_v, self.v_max
        if (
            not startup
            and current == 0
            and (charging_current > 0 or self.converter and not state.converter_ready)
        ):
            # Vmin is an operational discharge cutoff, not a prohibition on
            # replenishing an off/depleted store. Upper voltage remains hard.
            lower = 0.0
        if resistance is not None:
            upper = min(upper, self.max_current_a * resistance)
        if self.converter:
            if state.converter_ready or startup:
                lower = max(lower, self.converter.minimum_input_v)
            upper = min(upper, self.converter.maximum_input_v)
            if current > 0 and not startup:
                upper = min(
                    upper,
                    self.converter.maximum_output_power_w / (self.converter.efficiency * current),
                )
            if resistance is not None:
                upper = min(
                    upper,
                    self.converter.maximum_current_a * resistance,
                    math.sqrt(
                        self.converter.maximum_output_power_w
                        * resistance
                        / self.converter.efficiency
                    ),
                )
        events = []
        reason = "initial bulk charge is negative" if state.voltage < 0 else None
        if current > self.max_current_a or charging_current > self.max_charge_current_a:
            reason = "current envelope exceeded"
            if current > self.max_current_a:
                events.append(
                    ResourceEvent(
                        "discharge_current_limit", state.time, current, self.max_current_a
                    )
                )
            if charging_current > self.max_charge_current_a:
                events.append(
                    ResourceEvent(
                        "charge_current_limit",
                        state.time,
                        charging_current,
                        self.max_charge_current_a,
                    )
                )
        # Continuous-time safety, including interior excursions: subdivide using
        # a conservative |y''| bound and the linear-interpolation remainder.
        output = (1.0, *(-1.0 for _ in x0[1:]))
        factor = 1.0 if resistance is None else resistance / (resistance + self.esr_ohm)
        output = tuple(factor * c for c in output)
        row = tuple(math.fsum(output[i] * a[i][j] for i in range(len(x0))) for j in range(len(x0)))
        tol = 2e-12

        def safe_interval(lo, hi, depth=28):
            left, right = port(lo)[0], port(hi)[0]
            slopes = tuple(v + w for v, w in zip(matvec(a, vector(lo)), b))
            curvature = math.fsum(abs(v) for v in row) * max(map(abs, slopes))
            exponent = norm(a) * (hi - lo)
            curvature = curvature * math.exp(exponent) if exponent < 700 else float("inf")
            remainder = curvature * (hi - lo) ** 2 / 8
            if (
                min(left, right) - remainder >= lower - tol
                and max(left, right) + remainder <= upper + tol
            ):
                return None
            if left < lower - tol or left > upper + tol:
                return (lo, hi, left, "undervoltage" if left < lower else "overvoltage")
            if depth == 0 or hi - lo <= 1e-9:
                if right < lower - tol or right > upper + tol:
                    return (lo, hi, right, "undervoltage" if right < lower else "overvoltage")
                return (lo, hi, min(left, right), "safety_enclosure_unresolved")
            mid = (lo + hi) / 2
            return safe_interval(lo, mid, depth - 1) or safe_interval(mid, hi, depth - 1)

        violation = safe_interval(0.0, duration)
        if violation:
            lo, hi, value, kind = violation
            events.append(
                ResourceEvent(
                    kind, state.time + lo, value, lower if kind != "overvoltage" else upper, hi - lo
                )
            )
            reason = reason or kind
        if final[0] < -tol:
            reason = reason or "bulk charge exhausted"
        if self.converter:
            # Constant-current converter path: output power follows input
            # voltage. Nonlinear constant-power discharge is not implied.
            if current > self.converter.maximum_current_a:
                reason = reason or "converter current limit"
                events.append(
                    ResourceEvent(
                        "converter_current_limit",
                        state.time,
                        current,
                        self.converter.maximum_current_a,
                    )
                )
        work, ew = integrate(lambda t: port(t)[0] * port(t)[1], duration)
        esr, ee = integrate(lambda t: self.esr_ohm * port(t)[1] ** 2, duration)
        leakage, el = integrate(lambda t: self._leakage_loss(vector(t)), duration)
        polarization, ep = integrate(lambda t: self._polarization_loss(vector(t)), duration)
        load_work, _ = integrate(lambda t: port(t)[0] * port(t)[2], duration)
        charging_work, _ = integrate(lambda t: port(t)[0] * charging_current, duration)
        balance = EnergyBalance(
            self._stored(final) - self._stored(x0),
            work,
            esr,
            leakage,
            polarization,
            ew + ee + el + ep,
        )
        if startup:
            next_state = replace(next_state, converter_ready=reason is None)
        return ResourceTransition(
            state,
            next_state,
            duration,
            terminal,
            reason is None,
            reason,
            tuple(events),
            balance,
            "controlled_matrix_exponential",
            load_work,
            charging_work,
            None
            if self.converter is None or reason
            else 0.0
            if startup
            else self.converter.efficiency * load_work,
            None
            if self.converter is None or reason
            else load_work
            if startup
            else (1 - self.converter.efficiency) * load_work,
        )

    def _polarization_loss(self, vector):
        return 0.0

    def _leakage_loss(self, vector):
        return self.leakage_siemens * vector[0] ** 2

    def startup(self, state):
        if self.converter is None:
            raise ValueError("startup requires explicit converter constraints")
        return self._step(
            state,
            self.converter.startup_current_a,
            self.converter.startup_duration_s,
            0.0,
            startup=True,
        )


@dataclass(frozen=True)
class FosterResource(CapacitorResource):
    time_constants_s: tuple[float, ...] = ()
    branch_resistances_ohm: tuple[float, ...] = ()
    model_name: str = "capacitor_p2_foster"
    leakage_topology: str = "internal_terminal"

    def __post_init__(self):
        super().__post_init__()
        if not self.time_constants_s or len(self.time_constants_s) != len(
            self.branch_resistances_ohm
        ):
            raise ValueError("P2 requires at least one matched polarization branch")
        _finite(*self.time_constants_s, *self.branch_resistances_ohm)
        if min(*self.time_constants_s, *self.branch_resistances_ohm) <= 0:
            raise ValueError("Foster time constants and resistances must be positive")
        if self.leakage_topology not in ("internal_terminal", "bulk_capacitor"):
            raise ValueError("declare internal_terminal or bulk_capacitor leakage topology")

    def _vector(self, state):
        old = state.internal_states or (0.0,) * len(self.time_constants_s)
        if len(old) != len(self.time_constants_s):
            raise ValueError("internal-state dimension does not match circuit")
        return (state.voltage, *old)

    def _matrices(self, net_current, load_resistance=None):
        g = self.leakage_siemens if self.leakage_topology == "internal_terminal" else 0.0
        if load_resistance is not None:
            g += 1 / (load_resistance + self.esr_ohm)
        coefficients = (
            -1 / self.capacitance_f,
            *(r / t for r, t in zip(self.branch_resistances_ohm, self.time_constants_s)),
        )
        c = (1.0, *(-1.0 for _ in self.time_constants_s))
        a = tuple(
            tuple(
                k * g * value - (1 / self.time_constants_s[i - 1] if i > 0 and i == j else 0.0)
                for j, value in enumerate(c)
            )
            for i, k in enumerate(coefficients)
        )
        if self.leakage_topology == "bulk_capacitor":
            a = ((a[0][0] - self.leakage_siemens / self.capacitance_f, *a[0][1:]), *a[1:])
        return a, tuple(k * net_current for k in coefficients)

    def _stored(self, vector):
        return 0.5 * self.capacitance_f * vector[0] ** 2 + math.fsum(
            0.5 * t / r * x**2
            for x, t, r in zip(vector[1:], self.time_constants_s, self.branch_resistances_ohm)
        )

    def _polarization_loss(self, vector):
        return math.fsum(x**2 / r for x, r in zip(vector[1:], self.branch_resistances_ohm))

    def _leakage_loss(self, vector):
        voltage = (
            vector[0] - sum(vector[1:])
            if self.leakage_topology == "internal_terminal"
            else vector[0]
        )
        return self.leakage_siemens * voltage**2

    def constant_current_limit(self, v0, duration):
        raise ValueError("P2 limits depend on internal states; use deliverable_current")

    def _derivative(self, vector, current, charging_current, load_resistance=None):
        forcing = (
            current - charging_current
            if load_resistance is None
            else -charging_current * load_resistance / (load_resistance + self.esr_ohm)
        )
        a, b = self._matrices(forcing, load_resistance)
        return tuple(v + w for v, w in zip(matvec(a, vector), b))

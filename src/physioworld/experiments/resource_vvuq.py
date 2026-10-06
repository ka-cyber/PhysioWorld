"""Dependency-free resource verification experiments with independent circuit laws.

``run_resource_vvuq`` returns a finite JSON-ready report without writing artifacts.
The P1 inputs come from the packaged golden fixture. The complete synthetic P2
circuit comes from ``test_resource_control_qualification``; it is not a fitted
reconstruction of the prompt's incompletely specified P2 reference numbers.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import replace
from importlib.resources import files
from typing import Sequence

from physioworld.resources import CapacitorResource, FosterResource, ResourceState
from physioworld.verification import rk4_integrate


def _comparison(observed: float, reference: float, tolerance: float) -> dict:
    residual = observed - reference
    return {
        "observed": observed,
        "reference": reference,
        "residual": residual,
        "absolute_tolerance": tolerance,
        "passed": math.isfinite(residual) and abs(residual) <= tolerance,
    }


def _stored(model: FosterResource, state: Sequence[float]) -> float:
    return model.capacitance_f * state[0] ** 2 / 2 + math.fsum(
        tau * x * x / (2 * resistance)
        for x, tau, resistance in zip(
            state[1:], model.time_constants_s, model.branch_resistances_ohm
        )
    )


def _foster_reference(model, initial, current, charge, duration, steps, *, wrong_sign=False):
    """Integrate Kirchhoff equations and four work/loss terms, without kernel helpers.

    Only the internal-terminal leakage topology and imposed-current control used
    in this experiment are supported. No production matrices, derivative, stored
    energy implementation, quadrature, or matrix exponential enter this solver.
    """
    dimension = 1 + len(model.time_constants_s)

    def rhs(time, values):
        voltage, branches = values[0], values[1:dimension]
        internal = voltage - math.fsum(branches)
        external = current - charge
        circuit_current = external + model.leakage_siemens * internal
        derivatives = (
            (1 if wrong_sign else -1) * circuit_current / model.capacitance_f,
            *(
                (resistance * circuit_current - x) / tau
                for x, resistance, tau in zip(
                    branches, model.branch_resistances_ohm, model.time_constants_s
                )
            ),
        )
        terminal = internal - model.esr_ohm * external
        return (
            *derivatives,
            terminal * external,
            model.esr_ohm * external**2,
            model.leakage_siemens * internal**2,
            math.fsum(x * x / r for x, r in zip(branches, model.branch_resistances_ohm)),
        )

    return rk4_integrate(
        rhs,
        (initial.voltage, *initial.internal_states, 0.0, 0.0, 0.0, 0.0),
        duration=duration,
        steps=steps,
    )


def run_resource_vvuq(*, steps: Sequence[int] = (16, 32, 64), corner_fraction: float = 0.1) -> dict:
    """Verify P0/P1 analytics, coupled P2 convergence, safety, and 128 corners.

    ``steps`` must contain at least three increasing positive integer counts.
    Corner extrema are a finite parameter envelope, never a confidence interval.
    The reported PASS is scoped to these circuits, controls, and tested steps.
    """
    steps = tuple(steps)
    if (
        len(steps) < 3
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in steps)
        or any(a >= b for a, b in zip(steps, steps[1:]))
    ):
        raise ValueError("at least three increasing positive integer step counts are required")
    if not math.isfinite(corner_fraction) or not 0 < corner_fraction < 1:
        raise ValueError("corner_fraction must be finite and in (0, 1)")

    payload = (
        files("physioworld")
        .joinpath("corpus")
        .joinpath("golden")
        .joinpath("resource_depletion.json")
        .read_bytes()
    )
    fixture = json.loads(payload)
    c, esr = fixture["capacitance_f"], fixture["esr_ohm"]
    v0, vmin, duration, current = (
        fixture[key] for key in ("v0", "vmin", "duration_s", "current_a")
    )
    p0 = CapacitorResource(c, v_min=vmin)
    p1 = CapacitorResource(c, esr, v_min=vmin)
    transition = p1.step(ResourceState(v0), current, duration)
    analytic = {
        "p0_current_limit": _comparison(
            p0.constant_current_limit(v0, duration), c * (v0 - vmin) / duration, 1e-12
        ),
        "p1_current_limit": _comparison(
            p1.constant_current_limit(v0, duration), (v0 - vmin) / (duration / c + esr), 1e-12
        ),
        "p1_terminal_voltage": _comparison(
            transition.terminal_voltage, v0 - current * (duration / c + esr), 1e-12
        ),
        "p1_energy_balance": _comparison(transition.energy_balance.residual_j, 0.0, 1e-11),
    }
    # Leakage is explicitly synthetic, not inferred from the quoted P1 current.
    g = 0.001
    leaky = CapacitorResource(c, esr, g, v_min=vmin)
    decay = math.exp(-g * duration / c)
    voltage = v0 * decay + current * math.expm1(-g * duration / c) / g
    analytic["p1_leakage_voltage"] = _comparison(
        leaky.step(ResourceState(v0), current, duration).next.voltage, voltage, 1e-12
    )
    analytic["p1_leakage_current_limit"] = _comparison(
        leaky.constant_current_limit(v0, duration),
        (v0 * decay - vmin) / (-math.expm1(-g * duration / c) / g + esr),
        1e-12,
    )

    # Existing complete two-branch qualification circuit; leakage couples states.
    model = FosterResource(
        0.2,
        2.0,
        0.001,
        v_min=0.5,
        time_constants_s=(0.2, 2.0),
        branch_resistances_ohm=(1.0, 4.0),
    )
    initial = ResourceState(3.0, (0.1, 0.2))
    load, charge, horizon = 0.02, 0.004, 0.7
    exact = model.step(initial, load, horizon, charging_current=charge)
    exact_state = (exact.next.voltage, *exact.next.internal_states)
    solutions = [_foster_reference(model, initial, load, charge, horizon, n) for n in steps]
    errors = [max(abs(a - b) for a, b in zip(s[:3], exact_state)) for s in solutions]
    orders = [
        math.log(a / b) / math.log(n2 / n1) if a > 0 and b > 0 else None
        for a, b, n1, n2 in zip(errors, errors[1:], steps, steps[1:])
    ]
    reference = solutions[-1]
    independent_residual = (
        _stored(model, reference[:3])
        - _stored(model, (initial.voltage, *initial.internal_states))
        + math.fsum(reference[3:])
    )
    balance = exact.energy_balance
    p2_checks = {
        "terminal_voltage": _comparison(
            exact.terminal_voltage,
            reference[0] - sum(reference[1:3]) - model.esr_ohm * (load - charge),
            2e-8,
        ),
        "production_energy_balance": _comparison(balance.residual_j, 0.0, 1e-11),
        "independent_energy_balance": _comparison(independent_residual, 0.0, 2e-8),
    }
    for name, actual, expected in zip(
        ("terminal_work", "esr_loss", "leakage_loss", "polarization_loss"),
        (
            balance.terminal_work_j,
            balance.esr_loss_j,
            balance.leakage_loss_j,
            balance.polarization_loss_j,
        ),
        reference[3:],
    ):
        p2_checks[name] = _comparison(actual, expected, 2e-8)
    convergence_passed = (
        all(b < a for a, b in zip(errors, errors[1:]))
        and all(order is not None and order >= 3.5 for order in orders)
        and errors[-1] <= 2e-8
    )

    parameter_names = (
        "capacitance_f",
        "esr_ohm",
        "leakage_siemens",
        "tau_0_s",
        "tau_1_s",
        "r_0_ohm",
        "r_1_ohm",
    )
    nominal = (0.2, 2.0, 0.001, 0.2, 2.0, 1.0, 4.0)

    def circuit(parameters):
        return replace(
            model,
            capacitance_f=parameters[0],
            esr_ohm=parameters[1],
            leakage_siemens=parameters[2],
            time_constants_s=tuple(parameters[3:5]),
            branch_resistances_ohm=tuple(parameters[5:7]),
        )

    def terminal(parameters, *, independent):
        candidate = circuit(parameters)
        if independent:
            state = _foster_reference(candidate, initial, load, charge, horizon, steps[-1])
            return state[0] - sum(state[1:3]) - candidate.esr_ohm * (load - charge)
        return candidate.step(initial, load, horizon, charging_current=charge).terminal_voltage

    corners = []
    for signs in itertools.product((-1, 1), repeat=7):
        parameters = tuple(x * (1 + sign * corner_fraction) for x, sign in zip(nominal, signs))
        candidate = circuit(parameters)
        # The nominal case above cross-checks the production exponential. For
        # the 128-point uncertainty envelope, use the independent integrator
        # alone so the sweep remains a genuinely independent propagation study
        # rather than 128 repeated production safety enclosures.
        independent = _foster_reference(candidate, initial, load, charge, horizon, steps[-1])
        fine = _foster_reference(candidate, initial, load, charge, horizon, steps[-1] * 2)
        terminal_voltage = (
            independent[0] - sum(independent[1:3]) - candidate.esr_ohm * (load - charge)
        )
        fine_terminal = fine[0] - sum(fine[1:3]) - candidate.esr_ohm * (load - charge)
        comparison = _comparison(terminal_voltage, fine_terminal, 1e-7)
        independent_residual = (
            _stored(candidate, independent[:3])
            - _stored(candidate, (initial.voltage, *initial.internal_states))
            + math.fsum(independent[3:])
        )
        feasible = candidate.v_min <= terminal_voltage <= candidate.v_max
        corners.append(
            {
                "parameters": dict(zip(parameter_names, parameters)),
                "terminal_voltage": comparison,
                "energy_residual_j": independent_residual,
                "feasible": feasible,
                "positive_bulk_voltage": independent[0] > 0,
                "nonnegative_losses": min(independent[4:]) >= 0,
                "passed": comparison["passed"]
                and feasible
                and independent[0] > 0
                and abs(independent_residual) <= 2e-8,
            }
        )
        corners[-1]["passed"] &= corners[-1]["nonnegative_losses"]
    sensitivities = {}
    for index, (name, value) in enumerate(zip(parameter_names, nominal)):
        delta = value * 1e-3
        low, high = list(nominal), list(nominal)
        low[index] -= delta
        high[index] += delta
        observed = (terminal(high, independent=False) - terminal(low, independent=False)) / (
            2 * delta
        )
        expected = (terminal(high, independent=True) - terminal(low, independent=True)) / (
            2 * delta
        )
        sensitivities[name] = {
            "perturbation": delta,
            **_comparison(observed, expected, 1e-5 + abs(expected) * 1e-5),
        }

    unsafe_current = 0.02
    unsafe = p1.step(ResourceState(v0), unsafe_current, duration)
    crossing = c * (v0 - esr * unsafe_current - vmin) / unsafe_current
    event = next((event for event in unsafe.events if event.kind == "undervoltage"), None)
    safety = {
        "analytic_crossing_s": crossing,
        "event_time_s": None if event is None else event.time,
        "time_error_s": None if event is None else event.time_error_s,
        "feasible": unsafe.feasible,
        "passed": not unsafe.feasible and event is not None and abs(event.time - crossing) <= 1e-7,
    }
    wrong = _foster_reference(model, initial, load, charge, horizon, steps[-1], wrong_sign=True)
    sign_error = max(abs(a - b) for a, b in zip(wrong[:3], exact_state))
    omitted_loss_residual = independent_residual - reference[5]

    equal_model = replace(model, leakage_siemens=0.0, v_min=2.4)
    polarized = ResourceState(2.65, (0.3, 0.0))
    equivalent_voltage = math.sqrt(2.65**2 + (0.2 / 1.0) * 0.3**2 / model.capacitance_f)
    relaxed = ResourceState(equivalent_voltage, (0.0, 0.0))
    left = equal_model.step(polarized, 0.02, 0.01)
    right = equal_model.step(relaxed, 0.02, 0.01)
    energy_gap = _stored(equal_model, (polarized.voltage, *polarized.internal_states)) - _stored(
        equal_model, (relaxed.voltage, *relaxed.internal_states)
    )
    controls = {
        "wrong_discharge_sign": {"state_error_v": sign_error, "passed": sign_error > 0.01},
        "omitted_leakage_loss": {
            "energy_residual_j": omitted_loss_residual,
            "passed": abs(omitted_loss_residual) > 1e-4,
        },
        "undervoltage_crossing": safety,
        "equal_energy_different_feasibility": {
            "initial_energy_gap_j": energy_gap,
            "polarized_terminal_v": left.terminal_voltage,
            "relaxed_terminal_v": right.terminal_voltage,
            "polarized_feasible": left.feasible,
            "relaxed_feasible": right.feasible,
            "passed": abs(energy_gap) <= 1e-12 and not left.feasible and right.feasible,
        },
    }
    passed = (
        all(check["passed"] for check in (*analytic.values(), *p2_checks.values()))
        and convergence_passed
        and exact.feasible
        and all(row["passed"] for row in corners)
        and all(check["passed"] for check in sensitivities.values())
        and all(control["passed"] for control in controls.values())
    )
    return {
        "experiment_id": "resource_vvuq",
        "status": "PASS" if passed else "FAIL",
        "evidence_class": "NUMERICAL_VERIFICATION",
        "data_origin": "GOLDEN_FIXTURE_AND_COMPLETE_SYNTHETIC_CIRCUIT",
        "data_fidelity": "MATHEMATICAL",
        "clinical_claim": False,
        "fixture": {
            "name": "resource_depletion.json",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "inputs": fixture,
        },
        "context_of_use": "constant-current capacitor/Foster circuit verification at declared SI inputs",
        "analytic_references": {"method": "closed_form_P0_P1_Kirchhoff_laws", "checks": analytic},
        "p2": {
            "parameters": dict(zip(parameter_names, nominal)),
            "initial_state": [initial.voltage, *initial.internal_states],
            "load_current_a": load,
            "charging_current_a": charge,
            "duration_s": horizon,
            "reference_method": "independent_Kirchhoff_RK4_with_work_and_loss_states",
            "production_state": list(exact_state),
            "independent_state": list(reference[:3]),
            "checks": p2_checks,
        },
        "convergence": {
            "steps": list(steps),
            "step_sizes_s": [horizon / n for n in steps],
            "maximum_state_errors_v": errors,
            "observed_orders": orders,
            "minimum_order": 3.5,
            "absolute_tolerance_v": 2e-8,
            "passed": convergence_passed,
        },
        "uncertainty": {
            "method": "seven_coordinate_rectangular_corner_enumeration",
            "fraction": corner_fraction,
            "corner_count": len(corners),
            "corners": corners,
            "terminal_voltage_range_v": [
                min(row["terminal_voltage"]["observed"] for row in corners),
                max(row["terminal_voltage"]["observed"] for row in corners),
            ],
            "interpretation": "finite tested corner range; not a statistical interval or an interior extremum proof",
        },
        "sensitivity": sensitivities,
        "negative_controls": controls,
        "limitations": [
            "Circuit-model numerical verification only; no hardware or clinical validation.",
            "The prompt's ESR-bearing P0 label is a P1/no-leak baseline; ideal P0 here has zero ESR.",
            "Quoted P2 currents and energy reductions lack complete original inputs and are not reproduced.",
        ],
    }

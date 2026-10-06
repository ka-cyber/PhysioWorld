"""Independent analytic, differential, quantifier, timing and control qualification.

Fixtures here are explicitly synthetic unless a complete source parameter
vector is identified. These checks are software/numerical verification, not
hardware validation or mathematical proofs.
"""

import json
import math
import unittest
from dataclasses import replace

from physioworld.compute import (
    ApproximateVariant,
    ComputeEngine,
    ComputeNode,
    ModelProfile,
    profile_callable,
)
from physioworld.policies import (
    ContextualBandit,
    CrossLayerAction,
    CrossLayerObjective,
    DecisionContext,
    DynamicProgrammingOracle,
    FiniteMDP,
    FixedPolicy,
    JointCrossLayerPolicy,
    MPCPolicy,
    NonanticipativePolicy,
    QLearningPolicy,
    RulePolicy,
    SafePolicyFilter,
    SensingAction,
    StochasticPolicy,
    TabularModelBasedPolicy,
    TimedInformation,
    discounted_value_iteration,
    evaluate_finite_policy,
    select_chance_constrained_policy,
    viability_kernel,
)
from physioworld.resources import (
    ApproximationBudget,
    CapacitorResource,
    ConverterConstraints,
    DesignEvaluation,
    FosterResource,
    LifecycleLaw,
    LifecycleResource,
    LifecycleState,
    MultiResourceBudget,
    ParameterStatus,
    PhysicalParameter,
    ResourceCapability,
    ResourceScheduleSegment,
    ResourceState,
    SafetyEnvelope,
    calibrate_resource,
    canonical_current_limit,
    canonical_fixture_status,
    corner_analysis,
    deliverable_current,
    design_search,
    discounted_value_transfer_bound,
    distribution_error_bounds,
    dobrushin_coefficient,
    evaluate_harvest_schedule,
    finite_value_transfer_bound,
    kernel_perturbation,
    optimize_robust_policy,
    phase_diagram,
    physical_schedule,
    resource_dominates,
    resource_frontier,
    robust_value,
    schedule_dominates,
    separability_interaction,
    shadow_value,
    stationary_average_reward_error_bound,
    stationary_distribution_error_bound,
    theorem_registry,
)
from physioworld.resources.linalg import expm

try:
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.linalg import expm as scipy_expm
    from scipy.optimize import brentq
except ImportError:
    np = None


class ResourcePhysicsQualification(unittest.TestCase):
    def test_p0_p1_analytic_baselines(self):
        p0 = CapacitorResource(0.2, v_min=1.85)
        self.assertAlmostEqual(p0.constant_current_limit(2.65, 10), 0.016)
        self.assertAlmostEqual(canonical_current_limit(2.65, 1.85, 0.2, 8, 10), 0.8 / 58)
        p1 = CapacitorResource(0.2, 8, 0.001, v_min=1.85)
        current = p1.constant_current_limit(2.65, 10)
        expected = (2.65 + current / 0.001) * math.exp(-0.001 * 10 / 0.2) - current / 0.001
        transition = p1.step(ResourceState(2.65), current, 10)
        self.assertAlmostEqual(transition.next.voltage, expected, places=12)
        self.assertAlmostEqual(transition.terminal_voltage, 1.85, places=12)
        self.assertTrue(transition.feasible)
        self.assertLess(abs(transition.energy_balance.residual_j), 2e-12)

    def test_ideal_current_foster_analytic_special_case(self):
        model = FosterResource(
            0.2, 0.5, time_constants_s=(0.1, 2.0), branch_resistances_ohm=(2.0, 3.0)
        )
        initial = ResourceState(3.0, (0.03, -0.02))
        transition = model.step(initial, 0.04, 1.3, charging_current=0.01)
        self.assertAlmostEqual(transition.next.voltage, 3 - 0.03 * 1.3 / 0.2, places=13)
        for old, r, tau, actual in zip(
            initial.internal_states,
            model.branch_resistances_ohm,
            model.time_constants_s,
            transition.next.internal_states,
        ):
            self.assertAlmostEqual(
                actual, r * 0.03 + (old - r * 0.03) * math.exp(-1.3 / tau), places=12
            )
        self.assertLess(abs(transition.energy_balance.residual_j), 3e-12)

    def test_bulk_shunt_foster_preserves_correct_legacy_circuit(self):
        model = FosterResource(
            0.2,
            0.5,
            0.01,
            time_constants_s=(0.3,),
            branch_resistances_ohm=(2,),
            leakage_topology="bulk_capacitor",
        )
        initial = ResourceState(3, (0.1,))
        transition = model.step(initial, 0.02, 0.7)
        voltage = (3 + 0.02 / 0.01) * math.exp(-0.01 * 0.7 / 0.2) - 0.02 / 0.01
        branch = 0.04 + (0.1 - 0.04) * math.exp(-0.7 / 0.3)
        self.assertAlmostEqual(transition.next.voltage, voltage, places=12)
        self.assertAlmostEqual(transition.next.internal_states[0], branch, places=12)
        self.assertLess(abs(transition.energy_balance.residual_j), 3e-12)

    @unittest.skipIf(np is None, "independent differential qualification requires numpy/scipy")
    def test_coupled_p2_dop853_and_energy(self):
        # Complete synthetic circuit, not the unavailable quoted canonical P2.
        c, esr, g = 0.3, 1.2, 0.015
        taus, resistors = (0.03, 0.7, 8.0), (0.8, 2.2, 4.0)
        model = FosterResource(
            c, esr, g, v_min=0.5, time_constants_s=taus, branch_resistances_ohm=resistors
        )
        initial = ResourceState(3.0, (0.01, 0.03, -0.02))
        load, charge, duration = 0.02, 0.007, 2.0

        def independent(t, y):
            v, branches = y[0], y[1:4]
            internal = v - sum(branches)
            external = load - charge
            circuit_current = external + g * internal
            dx = [-circuit_current / c] + [
                (r * circuit_current - x) / tau for r, x, tau in zip(resistors, branches, taus)
            ]
            terminal = internal - esr * external
            losses = (
                esr * external**2
                + g * internal**2
                + sum(x * x / r for x, r in zip(branches, resistors))
            )
            return dx + [terminal * external, losses]

        solution = solve_ivp(
            independent,
            (0, duration),
            [initial.voltage, *initial.internal_states, 0, 0],
            method="DOP853",
            rtol=2.3e-14,
            atol=2e-15,
        )
        transition = model.step(initial, load, duration, charging_current=charge)
        actual = np.array([transition.next.voltage, *transition.next.internal_states])
        self.assertLess(np.max(np.abs(actual - solution.y[:4, -1])), 3e-12)
        e0 = c * initial.voltage**2 / 2 + sum(
            tau / r * x * x / 2 for tau, r, x in zip(taus, resistors, initial.internal_states)
        )
        e1 = c * actual[0] ** 2 / 2 + sum(
            tau / r * x * x / 2 for tau, r, x in zip(taus, resistors, actual[1:])
        )
        self.assertLess(abs(e1 - e0 + sum(solution.y[4:, -1])), 3e-12)
        self.assertLess(abs(transition.energy_balance.residual_j), 3e-12)
        # A branch perturbation propagates to bulk and another branch during step.
        shifted = model.step(
            ResourceState(3.0, (0.04, 0.03, -0.02)), load, duration, charging_current=charge
        )
        self.assertGreater(abs(shifted.next.voltage - transition.next.voltage), 1e-7)
        self.assertGreater(
            abs(shifted.next.internal_states[1] - transition.next.internal_states[1]), 1e-7
        )

    @unittest.skipIf(np is None, "independent matrix check requires scipy")
    def test_matrix_exponential_and_resistive_coupling(self):
        matrix = ((-20.0, 2.0, 0.0), (3.0, -4.0, 1.0), (0.0, 2.0, -1.0))
        self.assertLess(np.max(np.abs(np.asarray(expm(matrix)) - scipy_expm(matrix))), 2e-14)
        model = FosterResource(
            0.2, 2, 0.001, time_constants_s=(0.2, 2.0), branch_resistances_ohm=(1.0, 4.0)
        )
        initial, rl, charge, duration = ResourceState(3.0, (0.1, 0.2)), 100.0, 0.004, 0.7

        def physical(t, x):
            load = (x[0] - x[1] - x[2] + 2 * charge) / (rl + 2)
            j = load - charge + 0.001 * (x[0] - x[1] - x[2])
            return (-j / 0.2, (j - x[1]) / 0.2, (4 * j - x[2]) / 2)

        ode = solve_ivp(
            physical, (0, duration), (3, 0.1, 0.2), method="DOP853", rtol=2.3e-14, atol=2e-15
        )
        transition = model.step_load_resistance(initial, rl, duration, charging_current=charge)
        self.assertLess(
            np.max(
                np.abs(
                    np.array([transition.next.voltage, *transition.next.internal_states])
                    - ode.y[:, -1]
                )
            ),
            2e-12,
        )
        self.assertAlmostEqual(
            transition.next.load_current * rl, transition.terminal_voltage, places=12
        )
        self.assertLess(abs(transition.energy_balance.residual_j), 3e-12)

    def test_charge_discharge_cycles_and_startup(self):
        model = CapacitorResource(0.4, 1, v_min=1, v_max=4)
        initial = ResourceState(2.0)
        discharge = model.step(initial, 0.05, 2)
        recharge = model.step(discharge.next, 0, 2, charging_current=0.05)
        self.assertAlmostEqual(recharge.next.voltage, initial.voltage, places=13)
        self.assertGreater(discharge.energy_balance.terminal_work_j, 0)
        self.assertLess(recharge.energy_balance.terminal_work_j, 0)
        converter = ConverterConstraints(0.9, 1.8, 4, 0.1, 0.2, 0.09, 0.2)
        startup_model = CapacitorResource(0.2, 8, v_min=1.8, converter=converter)
        self.assertFalse(startup_model.startup(ResourceState(2.0)).feasible)
        self.assertFalse(startup_model.step(ResourceState(3.5), 0.1, 0.01).feasible)
        feasible = startup_model.step(ResourceState(3), 0.04, 0.1)
        self.assertTrue(feasible.feasible)
        self.assertAlmostEqual(
            feasible.converter_output_energy_j, 0.9 * feasible.load_work_j, places=13
        )
        self.assertAlmostEqual(
            feasible.converter_loss_j + feasible.converter_output_energy_j,
            feasible.load_work_j,
            places=13,
        )
        cold = ResourceState(3, converter_ready=False)
        self.assertFalse(startup_model.step(cold, 0.04, 0.1).feasible)
        self.assertFalse(startup_model.feasible(cold, 0.04))
        started = startup_model.startup(cold)
        self.assertTrue(started.feasible)
        self.assertTrue(started.next.converter_ready)
        self.assertEqual(started.converter_output_energy_j, 0)
        self.assertAlmostEqual(started.converter_loss_j, started.load_work_j)
        self.assertTrue(startup_model.step(started.next, 0.04, 0.1).feasible)
        depleted = ResourceState(1.5, converter_ready=False)
        harvested = startup_model.step(depleted, 0, 3, charging_current=0.1)
        self.assertTrue(harvested.feasible)
        self.assertFalse(harvested.next.converter_ready)
        self.assertTrue(startup_model.startup(harvested.next).feasible)

    @unittest.skipIf(np is None, "root-reference check requires scipy")
    def test_interior_safety_excursion_and_event_timing(self):
        model = FosterResource(
            0.2, 0.5, v_min=1, v_max=2.7, time_constants_s=(0.1,), branch_resistances_ohm=(2.0,)
        )
        result = model.step(ResourceState(3.0, (0.8,)), 0.04, 3.0)
        self.assertLess(result.terminal_voltage, 2.7)  # endpoint-only check would pass
        self.assertFalse(result.feasible)
        reference = brentq(
            lambda t: 3 - 0.2 * t - (0.08 + 0.72 * math.exp(-10 * t)) - 0.02 - 2.7, 0, 0.3
        )
        self.assertLess(abs(result.events[0].time - reference), 2e-8)
        self.assertLess(result.events[0].time_error_s, 2e-8)

    def test_deliverable_current_and_safety_margin(self):
        model = CapacitorResource(0.2, 8, v_min=1.85, max_current_a=0.05)
        current = deliverable_current(model, ResourceState(2.65), 10, upper=0.05, tolerance_a=1e-9)
        self.assertLess(abs(current - 0.8 / 58), 1e-9)
        margin_model = replace(model, safety_margin_v=0.05)
        self.assertLess(margin_model.constant_current_limit(2.65, 10), current)

    def test_deliverable_current_finds_positive_discharge_from_overvoltage(self):
        model = CapacitorResource(1.0, 1.0, v_min=1.0, v_max=3.0, max_current_a=1.0)
        current = deliverable_current(model, ResourceState(3.1), 0.1, upper=1.0)
        self.assertGreater(current, 0.0)

    @unittest.skipIf(np is None, "nonlinear lifecycle requires scipy")
    def test_lifecycle_thermal_degradation_and_energy(self):
        law = LifecycleLaw(
            10,
            2,
            25,
            40,
            0.3,
            0.01,
            25,
            ParameterStatus.SYNTHETIC,
            "qualification synthetic throughput-ESR law",
        )
        wrapper = LifecycleResource(CapacitorResource(2, 1, 0.002, v_min=0.1), law)
        state = LifecycleState(ResourceState(3), 25)
        result = wrapper.step(state, 0.1, 2)
        self.assertTrue(result.electrical.feasible)
        self.assertAlmostEqual(result.next.throughput_coulombs, 0.2, places=12)
        self.assertGreater(result.esr_ohm, 1)
        self.assertGreater(result.next.temperature_c, 25)
        self.assertLess(abs(result.thermal_residual_j), 2e-11)
        self.assertLess(abs(result.electrical.energy_balance.residual_j), 2e-11)
        hot = LifecycleResource(
            CapacitorResource(2, 1, v_min=0.1), replace(law, maximum_temperature_c=25.001)
        )
        stopped = hot.step(state, 0.1, 2)
        self.assertFalse(stopped.electrical.feasible)
        self.assertLess(stopped.electrical.duration, 2)
        self.assertEqual(stopped.electrical.events[0].kind, "overheating")

    def test_finite_parameters_and_dimensions(self):
        for value in (float("nan"), float("inf"), -1.0, 0.0):
            with self.assertRaises(ValueError):
                CapacitorResource(value)
        with self.assertRaises(ValueError):
            CapacitorResource(1, v_max=float("inf"))
        with self.assertRaises(ValueError):
            FosterResource(1)
        with self.assertRaises(ValueError):
            FosterResource(1, time_constants_s=(1,), branch_resistances_ohm=(1,)).step(
                ResourceState(2, (1, 2)), 0, 1
            )
        with self.assertRaises(ValueError):
            ResourceState(float("nan"))
        with self.assertRaises(ValueError):
            CapacitorResource(1).step(ResourceState(2), 0.1, -1)
        with self.assertRaises(ValueError):
            SafetyEnvelope(max_current=float("nan"))


class ResourceControlQualification(unittest.TestCase):
    def test_robust_crossing_policies(self):
        values = {
            "left": {"cold": 10.0, "hot": 0.0},
            "right": {"cold": 0.0, "hot": 10.0},
            "balanced": {"cold": 4.0, "hot": 4.0},
        }
        result = optimize_robust_policy(values)
        self.assertEqual(result.policy, "balanced")
        self.assertEqual(robust_value(values), 4)
        self.assertEqual(result.clairvoyant_upper_bound, 10)
        feasible = {p: {t: p != "balanced" for t in ("cold", "hot")} for p in values}
        self.assertEqual(robust_value(values, feasible=feasible), 0)
        with self.assertRaises(ValueError):
            robust_value([10, 10])
        with self.assertRaises(ValueError):
            robust_value(values, order="inf_sup")

    @unittest.skipIf(np is None, "randomized robust LP requires scipy")
    def test_randomized_common_policy(self):
        values = {"left": {"cold": 10, "hot": 0}, "right": {"cold": 0, "hot": 10}}
        result = optimize_robust_policy(values, randomized=True)
        self.assertAlmostEqual(result.value, 5)
        self.assertAlmostEqual(result.policy_weights["left"], 0.5)

    def test_joint_capability_and_emulation(self):
        a = ResourceCapability({"left": (1, 0), "right": (0, 1)}, ("a", "b"), "common", 2)
        b = ResourceCapability({"joint": (0.6, 0.6)}, ("a", "b"), "common", 2)
        self.assertFalse(resource_dominates(a, b))
        strong = ResourceCapability({"all": (1, 1)}, ("a", "b"), "common", 2)
        self.assertTrue(resource_dominates(strong, a, strict=True))
        self.assertTrue(
            resource_dominates(
                strong, a, relation="policy_emulation", emulation={"left": "all", "right": "all"}
            )
        )
        with self.assertRaises(TypeError):
            resource_dominates((3, 2), (2, 1))
        with self.assertRaises(ValueError):
            resource_dominates(strong, replace(a, horizon=3))

    def test_harvest_timing(self):
        early, late = (1.0, 0.0), (0.0, 1.0)
        self.assertEqual(sum(early), sum(late))
        self.assertEqual(evaluate_harvest_schedule(early, (1, 0), (10, 0)).total_value, 10)
        self.assertEqual(evaluate_harvest_schedule(late, (1, 0), (10, 0)).total_value, 0)
        self.assertTrue(schedule_dominates(early, late))
        self.assertFalse(schedule_dominates(late, early))
        # The prefix criterion is deliberately limited to lossless storage.
        self.assertEqual(
            evaluate_harvest_schedule(early, (0, 1), (0, 10), retention=0.5).total_value, 0
        )
        self.assertEqual(
            evaluate_harvest_schedule(late, (0, 1), (0, 10), retention=0.5).total_value, 10
        )

    def test_physical_schedule_harvest_time(self):
        model = CapacitorResource(1, v_min=1.8, v_max=3)
        initial = ResourceState(1.85)
        supply = ResourceScheduleSegment(1, 0, 0.2, "charge")
        demand = ResourceScheduleSegment(1, 0.1, 0, "deadline_task")
        early, transitions = physical_schedule(model, initial, (supply, demand))
        late, _ = physical_schedule(model, initial, (demand, supply))
        self.assertTrue(early.reachable)
        self.assertFalse(late.reachable)
        self.assertEqual(late.violating_step, 0)
        self.assertEqual(early.states[-1].time, 2)
        self.assertLess(abs(sum(t.energy_balance.residual_j for t in transitions)), 2e-12)

    def test_equal_energy_different_physical_capability(self):
        state = ResourceState(2.0)
        low, high = CapacitorResource(1, 0.1, v_min=1.8), CapacitorResource(1, 10, v_min=1.8)
        self.assertEqual(low.energy(state).value_joules, high.energy(state).value_joules)
        self.assertTrue(low.step(state, 0.1, 0.1).feasible)
        self.assertFalse(high.step(state, 0.1, 0.1).feasible)
        p2 = FosterResource(0.2, 0.5, v_min=1.8, time_constants_s=(1,), branch_resistances_ohm=(2,))
        ready, polarized = ResourceState(2, (0,)), ResourceState(math.sqrt(3.9), (0.2,))
        self.assertAlmostEqual(
            p2.energy(ready).value_joules, p2.energy(polarized).value_joules, places=13
        )
        self.assertTrue(p2.step(ready, 0.01, 1).feasible)
        self.assertFalse(p2.step(polarized, 0.01, 1).feasible)

    def test_budget_reservations_and_approximation_sources(self):
        budget = MultiResourceBudget(
            {"energy": 2, "bandwidth": 10, "privacy": 1},
            {"energy": "J", "bandwidth": "byte", "privacy": "epsilon"},
            {"energy": 1},
        )
        self.assertFalse(budget.permits({"energy": 1.1}))
        self.assertFalse(budget.permits({"bandwidth": 11}))
        self.assertEqual(budget.consume({"energy": 0.5}).available["energy"], 1.5)
        with self.assertRaises(ValueError):
            budget.permits({"unknown": 0})
        approximation = ApproximationBudget(
            0.01,
            0.02,
            0.03,
            0,
            {
                "phys": "circuit error transferred to utility",
                "grid": "grid error transferred to utility",
                "transition": "kernel TV transferred via value bound",
                "stat": "no estimator used",
            },
            target_estimand="absolute_policy_value_error",
            unit="utility",
            composition_assumptions=(
                "all terms bound the same task/objective error",
                "triangle inequality applies",
            ),
        )
        self.assertAlmostEqual(approximation.upper_bound, 0.06)

    def test_phase_frontier_shadow(self):
        def evaluate(r):
            if r["energy"] < 1:
                return DesignEvaluation((0,), "off", False, "insufficient startup")
            return DesignEvaluation(
                (min(r["energy"], r["bandwidth"]),), "high" if r["energy"] >= 2 else "low"
            )

        diagram = phase_diagram({"energy": (0, 1, 2, 3), "bandwidth": (1, 2, 3)}, evaluate, (2,))
        frontier = resource_frontier(diagram)
        self.assertEqual([p.coordinates for p in frontier], [{"energy": 2, "bandwidth": 2}])
        self.assertTrue(diagram.boundary_edges)
        self.assertTrue(diagram.policy_swings)
        # A discrete activation threshold has unequal directional differences.
        shadow = shadow_value(lambda r: float(r["energy"] >= 2), {"energy": 2}, "energy", 0.1)
        self.assertAlmostEqual(shadow.backward_difference, 10)
        self.assertEqual(shadow.forward_difference, 0)
        self.assertEqual(separability_interaction(0, 1, 2, 5), 2)
        best, _ = design_search(
            (1, 2, 3),
            lambda c: DesignEvaluation((c,), str(c)),
            lambda c, result: result.capability[0] - 0.2 * c * c,
        )
        self.assertEqual(best[0], 2)

    def test_128_synthetic_corners_reproducible(self):
        bounds = {
            "v0": (2.5, 2.7),
            "vmin": (1.7, 1.9),
            "c": (0.18, 0.22),
            "esr": (7, 9),
            "g": (0.0005, 0.001),
            "duration": (9, 11),
            "margin": (0, 0.025),
        }

        def evaluate(theta):
            model = CapacitorResource(
                theta["c"],
                theta["esr"],
                theta["g"],
                v_min=theta["vmin"],
                safety_margin_v=theta["margin"],
            )
            return model.constant_current_limit(theta["v0"], theta["duration"])

        first, second = (
            corner_analysis(bounds, evaluate),
            corner_analysis(dict(reversed(tuple(bounds.items()))), evaluate),
        )
        self.assertEqual(len(first.corners), 128)
        self.assertEqual(first, second)
        self.assertEqual(
            first.corners[first.worst_index],
            {
                "c": 0.18,
                "duration": 11,
                "esr": 9,
                "g": 0.001,
                "margin": 0.025,
                "v0": 2.5,
                "vmin": 1.9,
            },
        )

    def test_calibration_provenance_and_fixture_availability(self):
        fixtures = canonical_fixture_status()
        self.assertEqual(fixtures["declared_no_leak_esr_baseline"]["status"], "available")
        for key in (
            "p1_quoted_current",
            "p2_nominal_and_128_corners",
            "tailored_uniform_margin_energy",
        ):
            self.assertEqual(fixtures[key]["status"], "unavailable")
            self.assertNotIn("current_a", fixtures[key])
        with self.assertRaises(ValueError):
            PhysicalParameter("c", 1.0, "F", ParameterStatus.UNAVAILABLE, "missing original data")

    @unittest.skipIf(np is None, "bounded least squares requires scipy")
    def test_synthetic_calibration_adapter_local_identifiability(self):
        observations = [
            {
                "time_s": t,
                "current_a": 0.02,
                "initial_voltage_v": 3,
                "voltage_v": 3 - 0.02 * t / 0.3,
            }
            for t in range(1, 8)
        ]
        result = calibrate_resource(
            observations,
            lambda p, r: r["initial_voltage_v"] - r["current_a"] * r["time_s"] / p["c"],
            {"c": 0.2},
            {"c": (0.1, 0.5)},
            units={"c": "F"},
            source="synthetic P0 test",
            operating_conditions={"temperature": "25 degC assumed"},
            data_origin="synthetic",
        )
        self.assertAlmostEqual(result.parameters[0].value, 0.3, places=7)
        self.assertEqual(result.parameters[0].status, ParameterStatus.SYNTHETIC)
        self.assertEqual(result.local_jacobian_rank, 1)
        self.assertEqual(len(result.input_checksum_sha256), 64)


class PolicyComputeQualification(unittest.TestCase):
    def test_actual_profiling_is_not_simulated_cost(self):
        calls = []
        profile = profile_callable(
            lambda: calls.append(sum(range(100))),
            operation_id="synthetic_sum",
            repetitions=3,
            warmup=1,
        )
        self.assertEqual(len(calls), 4)
        self.assertEqual(profile.evidence, "EMPIRICAL_ESTIMATE")
        self.assertEqual(len(profile.durations_s), 3)
        self.assertGreaterEqual(profile.mean_duration_s, 0)
        self.assertGreater(profile.peak_python_allocations_bytes, 0)

    def test_markov_quotient_preserves_values(self):
        from physioworld.policies.abstraction import (
            check_markov_quotient,
            quotient_mdp,
            reachable_sets,
        )

        mdp = FiniteMDP(
            (0, 1, 2),
            {s: ("a",) for s in (0, 1, 2)},
            lambda s, a: {0: 0.4, 1: 0.6} if s < 2 else {2: 1},
            lambda s, a: 1 if s < 2 else 2,
            3,
        )
        partition = {0: "left", 1: "left", 2: "right"}
        self.assertTrue(check_markov_quotient(mdp, partition).valid)
        full, quotient = (
            DynamicProgrammingOracle(mdp),
            DynamicProgrammingOracle(quotient_mdp(mdp, partition)),
        )
        for state in mdp.states:
            self.assertEqual(full.value(0, state), quotient.value(0, partition[state]))
        self.assertEqual(reachable_sets(mdp, (0,))[1], frozenset((0, 1)))
        bad = replace(mdp, reward=lambda s, a: s)
        self.assertFalse(check_markov_quotient(bad, partition).valid)

    def test_dp_analytic_and_validation(self):
        mdp = FiniteMDP(
            (0,),
            {0: ("a", "b")},
            lambda s, a: {0: 1.0},
            lambda s, a: 2 if a == "a" else 1,
            3,
            gamma=0.5,
            terminal_reward=lambda s: 4,
        )
        oracle = DynamicProgrammingOracle(mdp)
        self.assertAlmostEqual(oracle.value(0, 0), 2 + 1 + 0.5 + 0.5)
        infinite = discounted_value_iteration(mdp, tolerance=1e-11)
        self.assertTrue(infinite.converged)
        self.assertLess(abs(infinite.values[0] - 4), infinite.value_error_bound + 1e-14)
        truncated = discounted_value_iteration(mdp, max_iterations=1, tolerance=1e-15)
        self.assertFalse(truncated.converged)
        for kernel in ({0: 0.9}, {0: -1}, {1: 1}, {0: float("nan")}):
            with self.assertRaises(ValueError):
                replace(mdp, transition=lambda s, a, k=kernel: k).validate()
        with self.assertRaises(ValueError):
            replace(mdp, reward=lambda s, a: float("inf")).validate()
        with self.assertRaises(ValueError):
            discounted_value_iteration(mdp, gamma=1)

    def test_viability_and_exact_safety_probability(self):
        mdp = FiniteMDP(
            ("safe", "bad"),
            {"safe": ("wait", "risk"), "bad": ("wait",)},
            lambda s, a: {"safe": 0.8, "bad": 0.2} if s == "safe" and a == "risk" else {s: 1},
            lambda s, a: 10 if a == "risk" else 0,
            3,
            safe_states=frozenset({"safe"}),
        )
        oracle = DynamicProgrammingOracle(mdp)
        self.assertEqual(oracle.action_at(0, "safe"), "wait")
        self.assertEqual(viability_kernel(mdp), frozenset({"safe"}))
        evaluation = evaluate_finite_policy(
            mdp, lambda t, s: {"risk" if s == "safe" else "wait": 1}
        )
        self.assertAlmostEqual(evaluation.safety_violation_probabilities[(0, "safe")], 1 - 0.8**3)
        self.assertAlmostEqual(evaluation.values[(0, "safe")], 10 * (1 + 0.8 + 0.8**2))
        safe_evaluation = evaluate_finite_policy(mdp, lambda t, s: {"wait": 1})
        evaluations = {"risk": evaluation, "wait": safe_evaluation}
        self.assertEqual(select_chance_constrained_policy(evaluations, "safe", 0.1)[0], "wait")
        self.assertEqual(select_chance_constrained_policy(evaluations, "safe", 0.5)[0], "risk")
        with self.assertRaises(ValueError):
            select_chance_constrained_policy(
                {
                    **evaluations,
                    "other": evaluate_finite_policy(
                        replace(mdp, horizon=2), lambda t, s: {"wait": 1}
                    ),
                },
                "safe",
                0.5,
            )
        forced = replace(mdp, actions={"safe": ("risk",), "bad": ("wait",)})
        self.assertNotIn("safe", DynamicProgrammingOracle(forced).viability_sets[0])

    def test_time_varying_dp_and_true_mpc(self):
        mdp = FiniteMDP(
            (0,),
            {0: ("spend", "wait")},
            lambda s, a: {0: 1},
            lambda s, a: 0,
            2,
            time_reward=lambda t, s, a: (10 if t == 1 else 1) if a == "spend" else 0,
            admissible=lambda s, a, t: t != 0 or a == "wait",
        )
        oracle = DynamicProgrammingOracle(mdp)
        self.assertEqual(oracle.action_at(0, 0), "wait")
        self.assertEqual(oracle.value(0, 0), 10)
        mpc = MPCPolicy(
            lambda s, a: 5 if a == "cash" else 10 if s == 1 else 0,
            horizon=2,
            transition=lambda s, a: {1 if a == "invest" else 2: 1},
            future_actions=lambda s: ("collect",),
        )
        self.assertEqual(mpc.action(0, ("cash", "invest")), "invest")
        with self.assertRaises(ValueError):
            MPCPolicy(lambda s, a: 1, horizon=2).action(0, ("a",))

    def test_nonanticipation_stochastic_and_explicit_fallback(self):
        context = DecisionContext(
            1,
            (TimedInformation("present", 3, 0, 0.5), TimedInformation("future", 99, 0, 2)),
            frozenset({"present", "future"}),
        )
        policy = NonanticipativePolicy(RulePolicy(lambda s, a: "leak" if "future" in s else "safe"))
        self.assertEqual(policy.action(context, ("safe", "leak")), "safe")
        filtered = SafePolicyFilter(FixedPolicy("risk"), lambda s, a: a == "safe", "safe")
        self.assertEqual(filtered.action({}, ("risk", "safe")), "safe")
        with self.assertRaises(RuntimeError):
            filtered.action({}, ("risk",))
        first = StochasticPolicy(lambda s, a: {"a": 0.4, "b": 0.6}, seed=11)
        second = StochasticPolicy(lambda s, a: {"a": 0.4, "b": 0.6}, seed=11)
        self.assertEqual(
            [first.action({}, ("a", "b")) for _ in range(30)],
            [second.action({}, ("a", "b")) for _ in range(30)],
        )

    def test_actual_bandit_and_model_based_learning(self):
        bandit = ContextualBandit(("left", "right"), epsilon=0.2, seed=4)
        for t in range(1000):
            context = t % 2
            action = bandit.action(context, bandit.actions)
            reward = float(action == ("left" if context == 0 else "right"))
            bandit.update(context, action, reward)
        bandit.epsilon = 0
        self.assertEqual(bandit.action(0, bandit.actions), "left")
        self.assertEqual(bandit.action(1, bandit.actions), "right")
        model = TabularModelBasedPolicy(
            (0, 1), {0: ("stay", "go"), 1: ("collect",)}, 3, epsilon=0, offline=True
        )
        model.fit_offline([(0, "stay", 1, 0), (0, "go", 0, 1), (1, "collect", 3, 1)] * 5)
        self.assertEqual(model.action(0, ("stay", "go")), "go")
        self.assertEqual(model.plan().value(0, 0), 6)
        q = QLearningPolicy(alpha=1, gamma=0.9)
        q.q[(1, "future")] = 100
        q.update(0, "a", 2, 1, ("future",), terminal=True)
        self.assertEqual(q.q[(0, "a")], 2)

    def test_crosslayer_budget_policy_swing(self):
        local = CrossLayerAction(
            SensingAction("sample", "PPG", 10, duration_s=1, resolution_bits=8), transmit=False
        )
        remote = CrossLayerAction(
            SensingAction("sample", "ECG", 100, duration_s=1, resolution_bits=16),
            compression_fraction=0.5,
            coding_rate=0.8,
            compute_location="edge",
        )
        policy = JointCrossLayerPolicy(
            CrossLayerObjective(
                "task-value-energy-latency",
                lambda s, a: 3 if a == remote else 1,
                {"energy": 0.1, "bandwidth": 0.01},
            ),
            lambda s, a: (
                {"energy": 2, "bandwidth": 10} if a == remote else {"energy": 0.5, "bandwidth": 0}
            ),
            lambda s, a: 0.1,
            lambda s, a: True,
        )
        rich = {
            "budget": MultiResourceBudget(
                {"energy": 3, "bandwidth": 20}, {"energy": "J", "bandwidth": "byte"}
            )
        }
        poor = {
            "budget": MultiResourceBudget(
                {"energy": 1, "bandwidth": 20}, {"energy": "J", "bandwidth": "byte"}
            )
        }
        self.assertEqual(policy.action(rich, (local, remote)), remote)
        self.assertEqual(policy.action(poor, (local, remote)), local)

    def test_fifo_compute_cost_thermal_and_placement(self):
        local = ComputeNode("local", "local", 10, 1024, 2, requester_wait_power_w=0.1)
        edge = ComputeNode(
            "edge",
            "edge",
            100,
            1024,
            3,
            network_latency_s=0.1,
            network_bandwidth_bytes_s=100,
            communication_energy_per_byte_j=0.01,
        )
        model = ModelProfile(
            "model",
            100,
            20,
            input_bytes=10,
            variants={"early": ApproximateVariant(0.5, 0.2, "synthetic task-loss bound")},
        )
        engine = ComputeEngine({"local": local, "edge": edge}, {"model": model})
        quote = engine.execute("local", "model")
        self.assertEqual(quote.execution_time_s, 2)
        self.assertEqual(quote.energy_j, 4)
        self.assertEqual(engine.choose((("local", "model"), ("edge", "model"))).node_id, "edge")
        self.assertEqual(
            engine.choose(
                (("local", "model"), ("edge", "model")),
                objective=lambda r: -r.communication_energy_j,
            ).node_id,
            "local",
        )
        first = engine.submit("local", "model", arrival_time_s=0)
        second = engine.submit("local", "model", arrival_time_s=0.5)
        self.assertEqual(first.completion_time_s, 2)
        self.assertEqual(second.queue_wait_s, 1.5)
        self.assertEqual(second.completion_time_s, 4)
        self.assertAlmostEqual(second.waiting_energy_j, 0.15)
        early = engine.execute("edge", "model", variant="early")
        self.assertAlmostEqual(early.service_time_s, 0.1)
        self.assertAlmostEqual(early.network_latency_s, 0.2)
        self.assertAlmostEqual(early.communication_energy_j, 0.1)
        self.assertEqual(early.task_loss_bound, 0.2)
        thermal = ComputeNode(
            "hot", "local", 10, 1024, 10, thermal_capacitance_j_k=1, max_temperature_c=26
        )
        hot = ComputeEngine({"hot": thermal}, {"model": model}).execute("hot", "model")
        self.assertFalse(hot.feasible)
        self.assertEqual(hot.reason, "thermal limit")
        self.assertAlmostEqual(hot.final_temperature_c, 35 - 10 * math.exp(-2), places=12)
        self.assertFalse(engine.execute("edge", "model", deadline_s=0.3).feasible)

    def test_compute_failure_seed_and_invalid_cost(self):
        node = ComputeNode("local", "local", 10, 100, 1, failure_probability=0.5)
        model = ModelProfile("m", 20, 1)
        a, b = (
            ComputeEngine({"local": node}, {"m": model}, 5),
            ComputeEngine({"local": node}, {"m": model}, 5),
        )
        self.assertEqual(
            [a.submit("local", "m").succeeded for _ in range(20)],
            [b.submit("local", "m").succeeded for _ in range(20)],
        )
        for ops in (-1, float("nan")):
            with self.assertRaises(ValueError):
                a.execute("local", "m", input_operations=ops)
        with self.assertRaises(ValueError):
            ComputeNode("local", "local", 1, 1, 1, queue=2)
        with self.assertRaises(ValueError):
            ModelProfile("m", 1.5, 1)

    @unittest.skipIf(np is None, "exact linear reference requires numpy")
    def test_kernel_bounds_against_exact_values(self):
        p, q = ((0.8, 0.2), (0.3, 0.7)), ((0.75, 0.25), (0.35, 0.65))
        delta, error = dobrushin_coefficient(p), kernel_perturbation(p, q)
        self.assertAlmostEqual(delta, 0.5)
        self.assertAlmostEqual(error, 0.05)
        mu_p, mu_q = np.array([1.0, 0]), np.array([1.0, 0])
        bounds = distribution_error_bounds(delta, error, 10)
        for step in range(11):
            self.assertLessEqual(0.5 * sum(abs(mu_p - mu_q)), bounds[step] + 1e-14)
            mu_p, mu_q = mu_p @ p, mu_q @ q
        stationary_p, stationary_q = np.array([0.6, 0.4]), np.array([7 / 12, 5 / 12])
        self.assertLessEqual(
            0.5 * sum(abs(stationary_p - stationary_q)),
            stationary_distribution_error_bound(delta, error),
        )
        self.assertLessEqual(
            abs(stationary_p @ np.array([1, -1]) - stationary_q @ np.array([1, -1])),
            stationary_average_reward_error_bound(delta, error, 2),
        )
        reward, gamma = np.array([1.0, -1.0]), 0.9
        vp = np.linalg.solve(np.eye(2) - gamma * np.array(p), reward)
        vq = np.linalg.solve(np.eye(2) - gamma * np.array(q), reward)
        self.assertLessEqual(max(abs(vp - vq)), discounted_value_transfer_bound(0, error, 1, gamma))
        vpf, vqf = np.zeros(2), np.zeros(2)
        for _ in range(4):
            vpf, vqf = reward + np.array(p) @ vpf, reward + np.array(q) @ vqf
        finite = finite_value_transfer_bound((0,) * 4, (error,) * 4, (3, 2, 1, 0))
        self.assertLessEqual(max(abs(vpf - vqf)), finite[0])
        # Saturates the factor 2 for signed bounded f under half-L1 TV.
        self.assertEqual(finite_value_transfer_bound((0,), (1,), (1,))[0], 2)
        with self.assertRaises(ValueError):
            stationary_distribution_error_bound(1, 0.01)

    def test_machine_readable_registry_honesty(self):
        registry = theorem_registry()
        self.assertGreaterEqual(len(registry), 14)
        json.dumps(registry, allow_nan=False)
        for identifier, claim in registry.items():
            self.assertTrue(claim["assumptions"], identifier)
            self.assertTrue(claim["scope"], identifier)
            self.assertEqual(claim["proof_status"], "analytic_proof_sketch")
            self.assertNotIn("tests pass", claim["proof"])


if __name__ == "__main__":
    unittest.main()

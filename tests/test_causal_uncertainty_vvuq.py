"""Independent synthetic causal, clustered inference, and numerical reference checks."""

import math
import unittest
from dataclasses import replace

from physioworld.causal import (
    MANDATORY_ASSUMPTIONS,
    AssumptionStatus,
    CausalAssumption,
    EvaluationPolicy,
    IdentificationError,
    IdentificationRecord,
    KernelUncertaintySet,
    LoggedStep,
    LoggedTrajectory,
    OutcomeNuisance,
    SafePolicyCertificate,
    SupportRecord,
    TargetTrial,
    TimedCovariate,
    causal_placebo,
    empirical_g_formula,
    evaluate_policy,
    finite_robust_causal_value,
    hidden_confounding_bounds,
    positivity_diagnostic,
)
from physioworld.clinical import (
    AlarmWorkload,
    BinaryUtility,
    CohortOrigin,
    EndpointObservation,
    EvaluationCohort,
    FrozenArtifacts,
    LockedClinicalProtocol,
    average_precision,
    binary_endpoints,
    evaluate_independent_cohort,
    event_alarm_metrics,
    kaplan_meier,
    temporal_performance,
)
from physioworld.uncertainty import (
    RobustObjective,
    SplitConformal,
    brier_score,
    cluster_bootstrap,
    log_score,
    normal_normal_posterior,
)
from physioworld.verification import (
    ContextOfUse,
    CounterexampleRegistry,
    CredibilityCase,
    DesignCandidate,
    EnvelopeBound,
    FidelityLevel,
    Mechanism,
    ReferenceCase,
    ReferenceRegistry,
    Requirement,
    Tolerance,
    ValidationEvidence,
    compare_mechanisms,
    compare_validation_data,
    conservation_check,
    convergence_study,
    expected_information_gain,
    finite_design_search,
    fit_discrepancy,
    parameter_sensitivity,
    permutation_diagnostic,
    predictive_causal_counterexample,
    rk4_integrate,
)


def identification(method, *, follow_up=4.0):
    return IdentificationRecord(
        "expected cumulative synthetic reward",
        "binary treatment",
        TargetTrial(
            "baseline acquisition",
            "all synthetic subjects",
            ("0", "1"),
            ("always-1", "always-0", "dynamic"),
            "synthetic reward",
            follow_up,
            "complete follow-up; no censoring",
        ),
        tuple(
            CausalAssumption(
                name, AssumptionStatus.ASSUMED, "declared known synthetic structural model"
            )
            for name in MANDATORY_ASSUMPTIONS
        ),
        tuple(
            SupportRecord("synthetic finite design", a, 0.5, 40, "synthetic fixture")
            for a in ("0", "1")
        ),
        method,
        "synthetic finite population",
        "known-two-stage-SCM",
        "subject bootstrap",
        "binary odds sensitivity",
        ("synthetic only",),
    )


def constant_policy(action):
    return EvaluationPolicy(
        f"always-{action}", lambda h: {"0": 1.0 - int(action), "1": float(action)}
    )


def finite_two_stage_population():
    """Exactly integrate a known confounded finite DGP using integer frequency counts.

    X~Bernoulli(.5), P(A0=1|X)=.2+.6X; S1=X+A0;
    P(A1=1|S1)=.25 if S1=0 else .75; R0=X+A0; R1=S1+2A1.
    Analytic always-treat value is 5; never-treat value is 1.
    """
    rows = []
    for x in (0, 1):
        p0 = 0.8 if x else 0.2
        for a0 in (0, 1):
            state = x + a0
            p1 = 0.75 if state else 0.25
            for a1 in (0, 1):
                frequency = round(80 * 0.5 * (p0 if a0 else 1 - p0) * (p1 if a1 else 1 - p1))
                for _ in range(frequency):
                    steps = (
                        LoggedStep(
                            str(x),
                            (TimedCovariate("X", float(x), 0.0),),
                            str(a0),
                            (("0", 1 - p0), ("1", p0)),
                            float(x + a0),
                            0.0,
                            1.0,
                            ("0", "1"),
                        ),
                        LoggedStep(
                            str(state),
                            (TimedCovariate("S1", float(state), 2.0),),
                            str(a1),
                            (("0", 1 - p1), ("1", p1)),
                            float(state + 2 * a1),
                            2.0,
                            3.0,
                            ("0", "1"),
                        ),
                    )
                    rows.append(LoggedTrajectory(f"synthetic-{len(rows)}", 0.0, 0.0, steps))
    return tuple(rows)


def oracle_q(history, action):
    if history.index == 0:
        return 2 + 2 * int(history.state) + 2 * int(action)
    return int(history.state) + 2 * int(action)


class GuardedCausalTests(unittest.TestCase):
    def setUp(self):
        self.rows = finite_two_stage_population()
        self.policy = constant_policy("1")

    def test_known_longitudinal_ipw_and_discrete_g_formula(self):
        self.assertEqual(len(self.rows), 80)
        for method in ("ipw", "empirical_g_formula"):
            run = evaluate_policy if method == "ipw" else empirical_g_formula
            for action, truth in (("1", 5.0), ("0", 1.0)):
                result = run(self.rows, constant_policy(action), identification(method))
                self.assertAlmostEqual(result.value, truth)
                self.assertEqual(result.claim_status, "IDENTIFIED_UNDER_EXPLICIT_ASSUMPTIONS")

    def test_nonanticipative_dynamic_regime(self):
        def probabilities(history):
            action = int(history.index == 1 and int(history.state) > 0)
            return {"0": 1.0 - action, "1": float(action)}

        policy = EvaluationPolicy("dynamic", probabilities)
        result = evaluate_policy(self.rows, policy, identification("ipw"))
        self.assertAlmostEqual(result.value, 2.0)

    def test_sequential_dr_is_robust_to_either_nuisance_misspecification(self):
        distorted = tuple(
            replace(
                t,
                steps=tuple(
                    replace(s, behavior_probabilities=(("0", 0.5), ("1", 0.5))) for s in t.steps
                ),
            )
            for t in self.rows
        )
        oracle = OutcomeNuisance(oracle_q, "analytic Bellman oracle", external=True)
        correct_q = evaluate_policy(
            distorted, self.policy, identification("dr"), method="dr", nuisance=oracle
        )
        wrong_q = evaluate_policy(
            self.rows,
            self.policy,
            identification("dr"),
            method="dr",
            nuisance=OutcomeNuisance(lambda h, a: 0.0, "zero model", external=True),
        )
        self.assertAlmostEqual(correct_q.value, 5.0)
        self.assertAlmostEqual(wrong_q.value, 5.0)
        both_wrong = evaluate_policy(
            distorted,
            self.policy,
            identification("dr"),
            method="dr",
            nuisance=OutcomeNuisance(lambda h, a: 0.0, "zero model", external=True),
        )
        self.assertGreater(abs(both_wrong.value - 5.0), 0.5)

    def test_aipw_uses_control_action_propensity_and_g_computation(self):
        one_stage = tuple(
            replace(
                t,
                steps=(
                    replace(
                        t.steps[0], reward=3 * int(t.steps[0].action) + 2 * int(t.steps[0].state)
                    ),
                ),
            )
            for t in self.rows
        )
        nuisance = OutcomeNuisance(
            lambda h, a: 3 * int(a) + 2 * int(h.state),
            "known single-step response law",
            external=True,
        )
        for action, truth in (("0", 1.0), ("1", 4.0)):
            for method in ("aipw", "g_computation", "ipw"):
                result = evaluate_policy(
                    one_stage,
                    constant_policy(action),
                    identification(method),
                    method=method,
                    nuisance=nuisance if method != "ipw" else None,
                )
                self.assertAlmostEqual(result.value, truth)
        treated = [t.steps[0].reward for t in one_stage if t.steps[0].action == "1"]
        control = [t.steps[0].reward for t in one_stage if t.steps[0].action == "0"]
        observational_gap = sum(treated) / len(treated) - sum(control) / len(control)
        self.assertAlmostEqual(observational_gap, 4.2)
        self.assertNotAlmostEqual(observational_gap, 3.0)

    def test_identification_missing_uncheckable_violated_and_tested_block(self):
        record = identification("ipw")
        for status in (
            AssumptionStatus.UNCHECKABLE,
            AssumptionStatus.VIOLATED,
            AssumptionStatus.TESTED,
            AssumptionStatus.PLAUSIBLE,
        ):
            assumptions = tuple(
                replace(a, status=status) if a.name == "sequential_exchangeability" else a
                for a in record.assumptions
            )
            with self.assertRaisesRegex(IdentificationError, "sequential_exchangeability"):
                evaluate_policy(self.rows, self.policy, replace(record, assumptions=assumptions))
        with self.assertRaisesRegex(IdentificationError, "consistency"):
            evaluate_policy(
                self.rows, self.policy, replace(record, assumptions=record.assumptions[1:])
            )
        with self.assertRaisesRegex(ValueError, "evidence"):
            CausalAssumption("consistency", AssumptionStatus.VERIFIED, "unsupported assertion")

    def test_structural_support_cannot_be_overridden_by_assumptions(self):
        row = self.rows[0]
        bad = replace(row.steps[0], behavior_probabilities=(("0", 1.0), ("1", 0.0)))
        with self.assertRaisesRegex(IdentificationError, "positivity"):
            evaluate_policy(
                (replace(row, steps=(bad, row.steps[1])),), self.policy, identification("ipw")
            )
        with self.assertRaisesRegex(IdentificationError, "unsafe"):
            unsafe = replace(row.steps[0], safe_actions=("0",))
            evaluate_policy(
                (replace(row, steps=(unsafe, row.steps[1])),), self.policy, identification("ipw")
            )
        with self.assertRaisesRegex(IdentificationError, "no declared observed support"):
            evaluate_policy(
                self.rows, self.policy, replace(identification("ipw"), observed_support=())
            )

    def test_future_information_eligibility_and_censoring_guards(self):
        row = self.rows[0]
        future = replace(row.steps[0], covariates=(TimedCovariate("future-Y", 1.0, 10.0),))
        variants = (
            replace(row, steps=(future, row.steps[1])),
            replace(row, eligible_at=1.0),
            replace(row, censored=True),
            replace(row, steps=(row.steps[0], replace(row.steps[1], decision_time=0.5))),
        )
        for variant in variants:
            with self.assertRaises(IdentificationError):
                evaluate_policy((variant,), self.policy, identification("ipw"))

    def test_nuisance_subject_leakage_and_estimator_mismatch(self):
        with self.assertRaisesRegex(IdentificationError, "leakage"):
            evaluate_policy(
                self.rows,
                self.policy,
                identification("dr"),
                method="dr",
                nuisance=OutcomeNuisance(oracle_q, "fitted", (self.rows[0].subject_id,)),
            )
        with self.assertRaisesRegex(IdentificationError, "estimator"):
            evaluate_policy(self.rows, self.policy, identification("dr"))

    def test_empirical_g_formula_rejects_missing_counterfactual_cell(self):
        selected = tuple(
            t for t in self.rows if t.steps[0].state != "0" or t.steps[0].action == "0"
        )
        with self.assertRaisesRegex(IdentificationError, "cell"):
            empirical_g_formula(selected, self.policy, identification("empirical_g_formula"))

    def test_sensitivity_gamma_one_and_placebo(self):
        point = evaluate_policy(self.rows, self.policy, identification("ipw")).value
        bounds = hidden_confounding_bounds(self.rows, self.policy, identification("ipw"), gamma=1.0)
        wide = hidden_confounding_bounds(self.rows, self.policy, identification("ipw"), gamma=2.0)
        self.assertAlmostEqual(bounds.lower, point)
        self.assertAlmostEqual(bounds.upper, point)
        self.assertLess(wide.lower, point)
        self.assertGreater(wide.upper, point)
        placebo = causal_placebo(
            self.rows,
            self.policy,
            identification("ipw"),
            [(float(t.steps[0].state), 0.0) for t in self.rows],
        )
        self.assertAlmostEqual(placebo, 0.0)
        misspecified = tuple(
            replace(
                t,
                steps=tuple(
                    replace(s, behavior_probabilities=(("0", 0.5), ("1", 0.5))) for s in t.steps
                ),
            )
            for t in self.rows
        )
        failed_placebo = causal_placebo(
            misspecified,
            self.policy,
            identification("ipw"),
            [(float(t.steps[0].state), 0.0) for t in misspecified],
        )
        self.assertAlmostEqual(failed_placebo, 0.3)
        with self.assertRaisesRegex(IdentificationError, "precede"):
            causal_placebo(
                self.rows, self.policy, identification("ipw"), [(1.0, 5.0)] * len(self.rows)
            )

    def test_probability_normalization_and_nan_diagnostics(self):
        malformed = EvaluationPolicy("always-1", lambda h: {"0": 0.0, "1": 0.7})
        with self.assertRaisesRegex(ValueError, "normalized"):
            evaluate_policy(self.rows, malformed, identification("ipw"))
        self.assertFalse(positivity_diagnostic((math.nan, 0.5)).valid)

    def test_resource_indexed_robust_value_requires_safety_and_uncertainty_evidence(self):
        regime = "synthetic fixed resource state"
        kernels = KernelUncertaintySet(
            ("kernel-0", "kernel-1"),
            regime,
            0.95,
            AssumptionStatus.VERIFIED,
            ("synthetic-set-definition",),
        )
        certificates = tuple(
            SafePolicyCertificate(
                f"always-{a}",
                regime,
                AssumptionStatus.VERIFIED,
                ("synthetic-action-safety-enumeration",),
            )
            for a in (0, 1)
        )
        estimates = {}
        for kernel in kernels.kernel_ids:
            rows = (
                self.rows
                if kernel == "kernel-0"
                else tuple(
                    replace(
                        t,
                        steps=tuple(
                            replace(s, reward=s.reward - 3 * int(s.action)) for s in t.steps
                        ),
                    )
                    for t in self.rows
                )
            )
            for action in ("0", "1"):
                record = replace(identification("ipw"), model_id=kernel, resource_regime=regime)
                estimates[(f"always-{action}", kernel)] = evaluate_policy(
                    rows, constant_policy(action), record
                )
        result = finite_robust_causal_value(
            estimates,
            safety_certificates=certificates,
            uncertainty_set=kernels,
            resource_regime=regime,
        )
        self.assertAlmostEqual(result.value, 1.0)
        self.assertEqual(result.selected_policy_id, "always-0")
        with self.assertRaisesRegex(IdentificationError, "safety-certified"):
            finite_robust_causal_value(
                estimates,
                safety_certificates=(replace(certificates[0], status=AssumptionStatus.ASSUMED),),
                uncertainty_set=kernels,
                resource_regime=regime,
            )
        with self.assertRaisesRegex(IdentificationError, "uncertainty set"):
            finite_robust_causal_value(
                estimates,
                safety_certificates=certificates,
                uncertainty_set=replace(kernels, evidence_refs=()),
                resource_regime=regime,
            )
        incompatible = dict(estimates)
        incompatible[("always-0", "kernel-0")] = replace(
            incompatible[("always-0", "kernel-0")], discount=0.5
        )
        with self.assertRaisesRegex(IdentificationError, "incompatible"):
            finite_robust_causal_value(
                incompatible,
                safety_certificates=certificates,
                uncertainty_set=kernels,
                resource_regime=regime,
            )


class ClusteredUncertaintyTests(unittest.TestCase):
    def test_cluster_bootstrap_does_not_count_windows_as_subjects(self):
        values = [float(i) for i in range(8)]
        subjects = [f"subject-{i}" for i in range(8)]

        def statistic(rows):
            return sum(rows) / len(rows)

        original = cluster_bootstrap(values, subjects, statistic, repetitions=400, seed=11)
        duplicated = cluster_bootstrap(
            [v for v in values for _ in range(50)],
            [s for s in subjects for _ in range(50)],
            statistic,
            repetitions=400,
            seed=11,
        )
        self.assertEqual(original.independent_units, 8)
        self.assertAlmostEqual(original.standard_error, duplicated.standard_error)
        self.assertEqual(original.interval, duplicated.interval)
        with self.assertRaisesRegex(ValueError, "two independent"):
            cluster_bootstrap([1.0, 2.0], ["a", "a"], statistic)

    def test_proper_scores_and_no_confidence_clipping(self):
        labels = [1] * 20 + [0] * 80
        self.assertAlmostEqual(brier_score(labels, [0.2] * 100), 0.16)
        self.assertLess(brier_score(labels, [0.2] * 100), brier_score(labels, [0.8] * 100))
        self.assertLess(log_score(labels, [0.2] * 100), log_score(labels, [0.8] * 100))
        self.assertEqual(log_score((1,), (0.0,)), math.inf)

    def test_conformal_finite_sample_order_and_subject_maxima(self):
        subjects = [f"cal-{i}" for i in range(9)]
        finite = SplitConformal.fit(
            range(1, 10), subjects, training_subject_ids=("fit",), alpha=0.1
        )
        self.assertEqual(finite.prediction_interval(20.0, subject_id="new"), (11.0, 29.0))
        infinite = SplitConformal.fit(
            range(1, 10), subjects, training_subject_ids=("fit",), alpha=0.05
        )
        self.assertEqual(infinite.radius, math.inf)
        maxima = SplitConformal.fit(
            [1.0, 100.0, 2.0, 3.0], ["a", "a", "b", "c"], training_subject_ids=("fit",), alpha=0.25
        )
        self.assertEqual(maxima.radius, 100.0)
        with self.assertRaisesRegex(ValueError, "leakage"):
            finite.prediction_interval(20.0, subject_id="cal-0")
        with self.assertRaisesRegex(ValueError, "leakage"):
            SplitConformal.fit([1.0], ["fit"], training_subject_ids=("fit",))

    def test_normal_conjugate_posterior_analytic_reference(self):
        posterior = normal_normal_posterior(
            (1.0, 3.0), prior_mean=0.0, prior_variance=1.0, observation_variance=1.0
        )
        self.assertAlmostEqual(posterior.mean, 4 / 3)
        self.assertAlmostEqual(posterior.variance, 1 / 3)
        credible = posterior.credible_interval()
        predictive = posterior.prediction_interval()
        self.assertGreater(predictive[1] - predictive[0], credible[1] - credible[0])

    def test_common_policy_robust_minimax_has_correct_quantifier_order(self):
        objective = RobustObjective(((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(objective.sup_pi_inf_kernel(), 0.0)
        self.assertEqual(objective.inf_kernel_sup_pi(), 1.0)
        self.assertEqual(RobustObjective((1.0, 2.0)).inf_kernel_sup_pi(), 1.0)


def clinical_fixture():
    artifacts = FrozenArtifacts(*(["a" * 64] * 5))
    workload = AlarmWorkload(
        "synthetic-alarm",
        "researcher",
        "synthetic future event",
        10.0,
        0.5,
        BinaryUtility(4.0, 0.0, -2.0, -8.0, 0.5),
        "synthetic fixed resources",
        ("no intervention recommendations",),
        3.0,
    )
    protocol = LockedClinicalProtocol(
        "synthetic-protocol",
        workload,
        artifacts,
        "synthetic population",
        "all synthetic subjects",
        "known synthetic labels",
        "complete cases declared",
        ("fit",),
        ("development",),
        100.0,
        90.0,
        prospective=True,
        bootstrap_repetitions=200,
    )
    cohort = EvaluationCohort(
        "independent-synthetic",
        CohortOrigin.SYNTHETIC,
        "test fixture",
        "1",
        ("synthetic-fixture-definition",),
    )
    observations = tuple(
        EndpointObservation(
            f"eval-{i}",
            y,
            p,
            200.0,
            199.0,
            210.0,
            200.0,
            3800.0,
            150.0,
            artifacts.model_hash,
            "group-a" if i < 2 else "group-b",
        )
        for i, (y, p) in enumerate(((1, 0.9), (1, 0.1), (0, 0.8), (0, 0.2)))
    )
    return protocol, cohort, observations


class ClinicalEvaluationTests(unittest.TestCase):
    def evaluate(self, protocol, cohort, rows):
        return evaluate_independent_cohort(
            protocol,
            cohort,
            rows,
            expected_protocol_hash=protocol.frozen_hash,
            applied_artifacts=protocol.artifacts,
        )

    def test_locked_workload_metrics_subject_ci_and_no_clinical_claim(self):
        protocol, cohort, rows = clinical_fixture()
        result = self.evaluate(protocol, cohort, rows)
        self.assertEqual(result.status, "SYNTHETIC_PROTOCOL_EVALUATION")
        self.assertEqual(result.metrics["sensitivity"], 0.5)
        self.assertEqual(result.metrics["precision"], 0.5)
        self.assertEqual(result.metrics["false_alarms_per_hour"], 0.25)
        self.assertEqual(result.metrics["mean_utility"], -1.75)
        self.assertEqual(result.primary_interval.independent_units, 4)
        self.assertIn("group-a", result.subgroup_metrics)

    def test_frozen_task_threshold_artifact_and_seed_hash(self):
        protocol, cohort, rows = clinical_fixture()
        changed = replace(protocol, workload=replace(protocol.workload, threshold=0.8))
        self.assertNotEqual(protocol.frozen_hash, changed.frozen_hash)
        self.assertNotEqual(protocol.frozen_hash, replace(protocol, seed=1).frozen_hash)
        with self.assertRaisesRegex(ValueError, "lock mismatch"):
            evaluate_independent_cohort(
                changed,
                cohort,
                rows,
                expected_protocol_hash=protocol.frozen_hash,
                applied_artifacts=protocol.artifacts,
            )
        with self.assertRaisesRegex(ValueError, "lock mismatch"):
            evaluate_independent_cohort(
                protocol,
                cohort,
                rows,
                expected_protocol_hash=protocol.frozen_hash,
                applied_artifacts=replace(protocol.artifacts, model_hash="b" * 64),
            )

    def test_independence_prospective_and_future_information_guards(self):
        protocol, cohort, rows = clinical_fixture()
        variants = (
            replace(rows[0], subject_id="fit"),
            replace(rows[0], feature_available_at=201.0),
            replace(rows[0], outcome_ascertained_at=205.0),
            replace(rows[0], enrolled_at=50.0),
        )
        for row in variants:
            with self.assertRaises(ValueError):
                self.evaluate(protocol, cohort, (row,) + rows[1:])
        with self.assertRaisesRegex(ValueError, "development cohorts"):
            self.evaluate(protocol, replace(cohort, cohort_id="development"), rows)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.evaluate(protocol, cohort, rows + (replace(rows[0], probability=0.4),))

    def test_rare_event_ties_undefined_denominators_and_survival(self):
        self.assertAlmostEqual(average_precision((1, 0, 1, 0), (0.8, 0.8, 0.2, 0.2)), 0.5)
        self.assertIsNone(average_precision((0, 0), (0.2, 0.1)))
        self.assertIsNone(binary_endpoints((0, 0), (0, 0))["sensitivity"])
        metrics = event_alarm_metrics(
            (10.0, 20.0), (8.0, 9.0, 18.0, 25.0), lookback=3.0, exposure_hours=1.0
        )
        self.assertEqual(metrics.detected, 2)
        self.assertEqual(metrics.false_alarms, 2)
        self.assertEqual(metrics.lead_times, (2.0, 2.0))
        survival = kaplan_meier((1.0, 1.0, 2.0), (1, 0, 1))
        self.assertAlmostEqual(survival[0].survival, 2 / 3)
        self.assertEqual(survival[-1].survival, 0.0)
        self.assertEqual(survival[1].at_risk, 1)

    def test_frozen_temporal_performance_slices(self):
        protocol, _, rows = clinical_fixture()
        slices = temporal_performance(rows, protocol.workload, (0.0, 100.0, 300.0))
        self.assertIsNone(slices[0])
        self.assertEqual(slices[1]["windows"], 4.0)


class NumericalVerificationTests(unittest.TestCase):
    def test_actual_convergence_against_analytic_exponential(self):
        def solver(h):
            return rk4_integrate(lambda t, y: (-y[0],), (1.0,), duration=1.0, steps=round(1 / h))

        result = convergence_study(
            solver,
            lambda: (math.exp(-1.0),),
            (0.25, 0.125, 0.0625),
            tolerance=Tolerance(1e-7),
            reference_method="analytic exp(-t)",
            minimum_order=3.9,
        )
        self.assertEqual(result.status, "NUMERICALLY_VERIFIED_AT_TESTED_STEPS")
        self.assertTrue(all(order > 4.0 for order in result.observed_orders))
        bad = convergence_study(
            lambda h: (0.5,),
            lambda: (math.exp(-1),),
            (0.25, 0.125, 0.0625),
            tolerance=Tolerance(1e-7),
            reference_method="analytic exp(-t)",
        )
        self.assertEqual(bad.status, "FAILED")

    def test_coupled_compartment_conservation_and_independent_solution(self):
        final = rk4_integrate(
            lambda t, y: (-2 * y[0] + y[1], 2 * y[0] - y[1]), (1.0, 0.0), duration=2.0, steps=200
        )
        expected_first = 1 / 3 + 2 / 3 * math.exp(-6)
        self.assertAlmostEqual(final[0], expected_first, places=9)
        self.assertAlmostEqual(final[1], 1 - expected_first, places=9)
        check = conservation_check(
            1.0,
            sum(final),
            0.0,
            tolerance=Tolerance(1e-13),
            quantity="compartment amount",
            unit="mol",
        )
        self.assertTrue(check.passed)
        self.assertFalse(
            conservation_check(
                1.0, 1.01, 0.0, tolerance=Tolerance(1e-6), quantity="amount", unit="mol"
            ).passed
        )

    def test_actual_parameter_sensitivity_and_corner_range(self):
        study = parameter_sensitivity(
            lambda x: x[0] ** 2 + 3 * x[1],
            (2.0, 1.0),
            (0.01, 0.01),
            bounds=((1.0, 3.0), (0.0, 2.0)),
        )
        self.assertAlmostEqual(study.gradients[0], 4.0)
        self.assertAlmostEqual(study.gradients[1], 3.0)
        self.assertEqual((study.minimum, study.maximum), (1.0, 15.0))
        self.assertEqual(len(study.corner_values), 4)

    def registry(self, offset):
        registry = ReferenceRegistry()
        registry.add_case(
            ReferenceCase(
                "exp-decay",
                "mathematical decay",
                (1.0,),
                lambda inputs: (math.exp(-inputs[0]) + offset[0],),
                lambda inputs: (1 / math.exp(inputs[0]),),
                "independent reciprocal exponential",
                Tolerance(1e-12),
                ("decay-requirement",),
            )
        )
        registry.add_requirement(Requirement("decay-requirement", "decay accuracy", ("exp-decay",)))
        return registry

    def test_credibility_fresh_checks_and_context_specific_validation(self):
        offset = [0.0]
        registry = self.registry(offset)
        context = ContextOfUse(
            "decay-use",
            "mathematical decay analysis",
            "numerical error",
            "low",
            (("time", 1.0, "s"),),
        )
        evidence = ValidationEvidence(
            "analytic-support",
            "decay-use",
            "analytic decay law",
            "THEOREM",
            "mathematical reference, no human data",
            (EnvelopeBound("time", 0.0, 2.0, "s"),),
            (),
            verification_test_ids=("exp-decay",),
        )
        case = CredibilityCase(
            context,
            ("decay-requirement",),
            (evidence,),
            ("analytic-uncertainty",),
            ("zero-discrepancy-model",),
            ("decay-parameter-study",),
            ("mathematical use only",),
        )
        good = case.assess(registry, software_version="test-version")
        self.assertEqual(good.status, "SUPPORTED_WITHIN_DECLARED_CONTEXT")
        offset[0] = 0.1
        bad = case.assess(registry, software_version="test-version")
        self.assertEqual(bad.status, "CONTEXT_CLAIM_UNSUPPORTED")
        self.assertNotEqual(good.verification.run_id, bad.verification.run_id)
        offset[0] = 0.0
        outside = replace(case, context=replace(context, coordinates=(("time", 3.0, "s"),)))
        self.assertEqual(
            outside.assess(registry, software_version="test").status, "CONTEXT_CLAIM_UNSUPPORTED"
        )
        high_risk = replace(case, context=replace(context, risk_class="high"))
        self.assertIn(
            "high-consequence",
            " ".join(high_risk.assess(registry, software_version="test").reasons),
        )
        human_use = replace(
            case, context=replace(context, fidelity=FidelityLevel.HUMAN_OBSERVATIONAL)
        )
        self.assertEqual(
            human_use.assess(registry, software_version="test").status, "CONTEXT_CLAIM_UNSUPPORTED"
        )
        inflated = replace(evidence, fidelity=FidelityLevel.CLINICAL)
        self.assertEqual(
            replace(human_use, validation=(inflated,))
            .assess(registry, software_version="test")
            .status,
            "CONTEXT_CLAIM_UNSUPPORTED",
        )

    def test_reference_failures_shape_and_independence(self):
        def function(x):
            return x

        with self.assertRaises(ValueError):
            ReferenceCase(
                "self-check",
                "context",
                (1.0,),
                function,
                function,
                "same function",
                Tolerance(0.0),
                ("r",),
            )
        registry = ReferenceRegistry()
        registry.add_case(
            ReferenceCase(
                "shape",
                "context",
                (1.0,),
                lambda x: (),
                lambda x: x,
                "independent",
                Tolerance(0.0),
                ("r",),
            )
        )
        self.assertFalse(registry.run(software_version="test").results[0].passed)

    def test_model_validation_actual_residuals_cluster_interval_and_leakage(self):
        kwargs = dict(
            training_subject_ids=("fit",),
            source="synthetic analytic data",
            evidence_class="SIMULATION",
            model_artifact_hash="a" * 64,
            software_version="test",
            unit="mol",
            tolerance=Tolerance(0.2),
            repetitions=100,
        )
        result = compare_validation_data(
            (1.1, 1.9, 3.1), (1.0, 2.0, 3.0), ("a", "b", "c"), **kwargs
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.rmse, 0.1)
        self.assertEqual(result.mean_absolute_error_interval.independent_units, 3)
        failed = compare_validation_data(
            (1.1, 1.9, 4.0), (1.0, 2.0, 3.0), ("a", "b", "c"), **kwargs
        )
        self.assertFalse(failed.passed)
        self.assertNotEqual(result.data_hash, failed.data_hash)
        with self.assertRaisesRegex(ValueError, "leakage"):
            compare_validation_data((1.1, 1.9, 3.1), (1.0, 2.0, 3.0), ("fit", "b", "c"), **kwargs)

    def test_counterexample_requirement_reference_links_and_actual_run_receipt(self):
        template = predictive_causal_counterexample()
        registry = ReferenceRegistry()
        with self.assertRaisesRegex(ValueError, "registered"):
            registry.add_falsification(template)

        def interventional_gap(parameters):
            confounded_mean = parameters[0]
            # Y=A has effect 1; Y=U has the same mean under either intervention.
            return ((1.0 - confounded_mean) - (0.0 - confounded_mean),)

        registry.add_case(
            ReferenceCase(
                "synthetic-scm-identification",
                "synthetic SCM",
                (0.5,),
                interventional_gap,
                lambda x: (1.0,),
                "analytic contrast of two SCM intervention laws",
                Tolerance(0.0),
                ("causal-boundary",),
            )
        )
        registry.add_requirement(
            Requirement(
                "causal-boundary",
                "prediction does not identify interventions",
                ("synthetic-scm-identification",),
                (template.template_id,),
            )
        )
        registry.add_falsification(template)
        run = registry.run(software_version="test")
        self.assertTrue(run.results[0].passed)
        self.assertEqual(run.falsifications[0].status, "FALSIFIED")
        self.assertEqual(len(run.falsifications[0].evaluations), 3)

    def test_discrepancy_separate_fit_and_calibration_subjects(self):
        additive = fit_discrepancy(
            (1.0, 2.0),
            (3.0, 4.0),
            ("fit-a", "fit-b"),
            source="synthetic",
            task="synthetic",
            kind="additive",
        )
        self.assertEqual(additive.predict(5.0), 7.0)
        scale = fit_discrepancy(
            (1.0, 2.0),
            (2.0, 4.0),
            ("a", "b"),
            source="synthetic",
            task="synthetic",
            kind="multiplicative",
        )
        self.assertEqual(scale.predict(3.0), 6.0)
        state = fit_discrepancy(
            (1.0, 1.0, 1.0),
            (3.0, 5.0, 7.0),
            ("a", "b", "c"),
            source="synthetic",
            task="synthetic",
            kind="state_dependent",
            states=(0.0, 1.0, 2.0),
        )
        self.assertEqual(state.predict(1.0, state=1.0), 5.0)
        with self.assertRaisesRegex(ValueError, "support"):
            state.predict(1.0, state=3.0)
        with self.assertRaisesRegex(ValueError, "leakage"):
            additive.calibrate_intervals((1.0,), (3.0,), ("fit-a",))


class FalsificationAndDesignTests(unittest.TestCase):
    def test_predictive_not_causal_counterexample_is_computed(self):
        registry = CounterexampleRegistry()
        registry.register(predictive_causal_counterexample())
        result = registry.run()[0]
        self.assertEqual(result.status, "FALSIFIED")
        self.assertEqual(len(result.counterexamples), 3)
        self.assertEqual(result.counterexamples[0].observed, 1.0)
        self.assertEqual(result.counterexamples[0].requirement_ids, ("causal-boundary",))

    def test_mechanism_discrimination_finite_design_and_adaptive_posterior(self):
        flat = DesignCandidate("uninformative", "measurement", (), 1.0, True)
        informative = DesignCandidate("informative", "perturbation", (), 2.0, True)
        unsafe = DesignCandidate("unsafe", "action", (), 1.0, False)
        mechanisms = (
            Mechanism(
                "A",
                lambda c: (0.5, 0.5) if c.candidate_id == "uninformative" else (0.9, 0.1),
                "synthetic mechanism A",
            ),
            Mechanism(
                "B",
                lambda c: (0.5, 0.5) if c.candidate_id == "uninformative" else (0.1, 0.9),
                "synthetic mechanism B",
            ),
        )

        def objective(c):
            return expected_information_gain(mechanisms, (0.5, 0.5), c)

        self.assertEqual(objective(flat), 0.0)
        result = finite_design_search(
            (flat, informative, unsafe),
            budget=2.0,
            objective=objective,
            objective_name="expected information gain",
        )
        self.assertEqual(result.selected.candidate_id, "informative")
        self.assertEqual(len(result.scores), 2)
        self.assertEqual(result.rejected_candidates, ("unsafe",))
        compared = compare_mechanisms(mechanisms, (0.5, 0.5), ((informative, 0),))
        self.assertAlmostEqual(compared.posterior_probabilities[0], 0.9)
        posterior_gain = expected_information_gain(
            mechanisms, compared.posterior_probabilities, informative
        )
        self.assertLess(posterior_gain, objective(informative))
        cheap = finite_design_search(
            (flat, informative), budget=1.0, objective=objective, objective_name="information gain"
        )
        self.assertEqual(cheap.selected.candidate_id, "uninformative")

    def test_negative_control_can_destroy_order_dependent_semantics(self):
        values = tuple(range(20))

        def score(rows):
            return sum(int(a == b) for a, b in zip(rows, values)) / len(rows)

        diagnostic = permutation_diagnostic(values, score, kind="semantic_permutation", seed=2)
        self.assertEqual(diagnostic.original_statistic, 1.0)
        self.assertLess(diagnostic.control_statistic, 0.3)
        self.assertEqual(diagnostic.status, "DIAGNOSTIC_NOT_IDENTIFICATION")


if __name__ == "__main__":
    unittest.main()

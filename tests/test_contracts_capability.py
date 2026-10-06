"""Analytic, counterexample, serialization and adversarial scientific contracts.

All fixtures below are synthetic finite constructions. Passing these tests is
software/numerical verification, never empirical or clinical validation.
"""

import math
import unittest
from dataclasses import replace

try:
    import numpy as np
except ImportError:
    np = None

from physioworld.capability import (
    CapabilityRegion,
    InformationAvailabilityDesign,
    InformationDesignKind,
    PIRResult,
    PolicyRelation,
    PolicyRelationKind,
    RobustCapabilityRegion,
    StochasticCapabilityRegion,
    StrategicMeasure,
    bellman_contraction_bound,
    blackwell_dominates,
    compute_pir,
    condition_on_resource,
    dobrushin_stability_bound,
    estimate_piv,
    finite_kernel_error_bound,
    lower_tail_cvar,
    optimal_task_value,
    optimistic_value,
    representation_deficiency,
    resource_dominates,
    robust_value,
    support_gap,
)
from physioworld.capability.values import PIV_REQUIREMENTS, PIVEstimate
from physioworld.core import (
    Assumption,
    AssumptionSet,
    AssumptionStatus,
    CausalStatus,
    Certificate,
    Claim,
    ContextOfUse,
    EstimandFirewall,
    EstimandMismatch,
    EvidenceClass,
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeKind,
    FiniteKernel,
    FlowEdge,
    FlowEdgeType,
    FlowNode,
    FlowNodeType,
    IdentificationRecord,
    InformationFlowGraph,
    LocalGuarantee,
    Metric,
    MetricContext,
    OperatingEnvelope,
    ProofStatus,
    ScientificPipelineCertificate,
    Task,
    TaskKind,
    WorldState,
    compile_task,
    scientific_claim_registry,
    theorem_registry,
)
from physioworld.core.evidence import CAUSAL_REQUIREMENTS
from physioworld.tasklaws import (
    BootstrapResult,
    DiscreteDistribution,
    TaskDefinition,
    TaskLaw,
    binary_forecasting_certificate,
    clustered_bootstrap,
    conditional_coverage,
    conditional_task_law,
    empirical_conditional_distribution,
    expected_calibration_error,
    kernel_conditional_distribution,
    task_function_discrepancy,
    total_task_discrepancy,
    total_variation,
    wasserstein_1d,
)


def context(**changes):
    base = MetricContext(
        estimand_name="policy_value",
        data_domain="synthetic-finite-world-v1",
        evaluation_regime="resource_conditioned",
        conditioning=("time_zero",),
        policy_class="all_nonanticipative_policies",
        reference_state="initial-state-v1",
        time_horizon=10.0,
        task_family=("task",),
        resource_state="resource-v1",
        uncertainty_treatment="fixed_model",
        utility="utility-v1",
        population="synthetic",
        unit="utility",
        evidence_class=EvidenceClass.SIMULATION,
        causal_status=CausalStatus.MODEL_INTERVENTIONAL,
        information_condition="G",
        estimator="finite_enumeration",
        communication_regime="perfect",
    )
    return replace(base, **changes)


def verified(*names):
    return AssumptionSet(
        tuple(
            Assumption(
                name,
                f"Declared synthetic fixture condition: {name}",
                AssumptionStatus.VERIFIED,
                evidence=(f"analytic-fixture:{name}",),
            )
            for name in names
        )
    )


def task():
    return TaskDefinition("task", TaskKind.FORECASTING, 10.0, "outcome", "absolute_error")


def kernel(input_space, output_space, matrix):
    return FiniteKernel(input_space, output_space, ("0", "1"), ("0", "1"), matrix)


def model_design():
    return InformationAvailabilityDesign(
        "paired-model-v1",
        "modality-M",
        InformationDesignKind.PAIRED_MODEL_REPLAY,
        context(information_condition="G"),
        context(information_condition="G+M"),
        ("unit-a", "unit-b"),
        verified(*PIV_REQUIREMENTS),
        "synthetic-paired-model-fixture",
    )


class ContractTests(unittest.TestCase):
    def test_task_context_metric_roundtrip_and_legacy_task_alias(self):
        declared = replace(
            task(),
            evaluation_population="synthetic",
            resource_regime="r",
            reference_condition="initial",
            uncertainty_method="cluster_bootstrap",
        )
        self.assertIs(TaskDefinition, Task)
        self.assertEqual(Task.from_json(declared.to_json()), declared)
        compiled = compile_task(
            declared,
            metric_type="task_law_discrepancy",
            estimand_name="sup_task_mean_absolute_conditional_expectation_difference",
            data_domain="synthetic",
            evaluation_regime="intrinsic",
            policy_class="predictor",
        )
        self.assertEqual(compiled.time_horizon, 10)
        self.assertEqual(compiled.task_family, ("task",))
        self.assertEqual(MetricContext.from_json(compiled.to_json()), compiled)
        metric = Metric(0.1, compiled, "analytic-fixture")
        self.assertEqual(Metric.from_json(metric.to_json()), metric)

    def test_firewall_checks_every_context_field(self):
        base = context()
        changes = {
            "estimand_name": "auc",
            "data_domain": "external",
            "evaluation_regime": "intrinsic",
            "conditioning": ("channel_delivered",),
            "policy_class": "different_policy_class",
            "reference_state": "other_initial",
            "time_horizon": 20.0,
            "task_family": ("other_task",),
            "resource_state": "resource-v2",
            "uncertainty_treatment": "statistical_ci",
            "metric_type": "packet_delivery",
            "utility": "other_utility",
            "population": "other_population",
            "unit": "J",
            "causal_status": CausalStatus.PREDICTIVE,
            "evidence_class": EvidenceClass.EMPIRICAL_ESTIMATE,
            "communication_regime": "erasure",
            "information_condition": "H",
            "estimator": "different_estimator",
            "time_unit": "ms",
        }
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaisesRegex(EstimandMismatch, field):
                EstimandFirewall.compare(base, replace(base, **{field: value}))
        EstimandFirewall.compare(base, MetricContext.from_json(base.to_json()))

    def test_undefined_reference_metadata_cannot_pass_a_common_reference_check(self):
        for name in (
            "utility",
            "population",
            "reference_state",
            "policy_class",
            "resource_state",
            "estimator",
        ):
            incomplete = context(**{name: "unspecified"})
            with self.subTest(field=name), self.assertRaisesRegex(EstimandMismatch, "unspecified"):
                EstimandFirewall.common_reference(incomplete, incomplete)
        with self.assertRaisesRegex(ValueError, "utility"):
            context(utility=None)

    def test_intrinsic_channel_resource_and_evidence_cannot_merge(self):
        intrinsic = context(
            metric_type="auc", estimand_name="ranking_auc", evaluation_regime="intrinsic"
        )
        for changed in (
            replace(intrinsic, evaluation_regime="channel_conditioned"),
            replace(intrinsic, evaluation_regime="resource_conditioned"),
            replace(intrinsic, evidence_class=EvidenceClass.EMPIRICAL_ESTIMATE),
        ):
            with self.assertRaises(EstimandMismatch):
                EstimandFirewall.compare(intrinsic, changed)
        with self.assertRaises(EstimandMismatch):
            EstimandFirewall.require_metric(context(metric_type="packet_delivery"), "pir", "pir")

    def test_task_compiler_causal_and_decision_boundary(self):
        intervention = replace(task(), kind=TaskKind.INTERVENTIONAL_OUTCOME)
        with self.assertRaises(EstimandMismatch):
            compile_task(
                intervention,
                metric_type="auc",
                estimand_name="auc",
                data_domain="synthetic",
                evaluation_regime="intrinsic",
                policy_class="predictor",
            )
        decision = replace(task(), kind=TaskKind.DECISION)
        with self.assertRaises(EstimandMismatch):
            compile_task(
                decision,
                metric_type="correlation",
                estimand_name="policy_value",
                data_domain="synthetic",
                evaluation_regime="intrinsic",
                policy_class="controller",
            )

    def test_version_unknown_field_wrong_type_and_nonfinite_rejected(self):
        payload = context().to_dict()
        for altered in (
            {**payload, "schema_version": "2.0.0"},
            {**payload, "unrecognized": "field"},
            {**payload, "time_horizon": "10 seconds"},
            {**payload, "record_type": "Task"},
        ):
            with self.subTest(payload=altered), self.assertRaises((ValueError, TypeError)):
                MetricContext.from_dict(altered)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Metric(value, context(), "synthetic")

    def test_world_snapshot_isolates_all_nested_layers_and_aliases(self):
        shared = {"samples": [1, 2]}
        world = WorldState(
            physiological={"nested": shared},
            history=[shared],
            resource={"diffusion": [2, 3]},
            wireless={"queue": [shared]},
            compute={"cache": {"x": [4]}},
        )
        snapshot = world.snapshot()
        shared["samples"].append(99)
        world.resource["diffusion"][0] = -1
        world.compute["cache"]["x"].append(5)
        self.assertEqual(snapshot.physiological["nested"]["samples"], [1, 2])
        self.assertEqual(snapshot.resource["diffusion"], [2, 3])
        self.assertEqual(snapshot.compute["cache"]["x"], [4])
        self.assertIs(snapshot.physiological["nested"], snapshot.history[0])
        self.assertIs(snapshot.history[0], snapshot.wireless["queue"][0])


class EvidenceTests(unittest.TestCase):
    def certificate(self, assumptions, high_confidence=True):
        return Certificate(
            "cert",
            "claim",
            "fixture guarantee",
            EvidenceClass.NUMERICAL_VERIFICATION,
            assumptions,
            OperatingEnvelope("toy", "synthetic finite envelope", {"x": (0, 1)}),
            ContextOfUse("toy-use", "verification", "synthetic", "finite world", ("toy",)),
            ("analytic-fixture",),
            high_confidence,
            () if high_confidence else ("conditional on unresolved assumptions",),
        )

    def test_certificate_guards_all_unsupported_mandatory_statuses(self):
        for status in (
            AssumptionStatus.VIOLATED,
            AssumptionStatus.UNCHECKABLE,
            AssumptionStatus.PLAUSIBLE,
            AssumptionStatus.NOT_APPLICABLE,
        ):
            assumptions = AssumptionSet(
                (Assumption("a", "condition", status, rationale="fixture blocker"),)
            )
            with (
                self.subTest(status=status),
                self.assertRaisesRegex(ValueError, f"a: {status.value}"),
            ):
                self.certificate(assumptions)
        with self.assertRaisesRegex(ValueError, "missing mandatory"):
            AssumptionSet().require(("positivity",))

    def test_verified_needs_artifact_and_certificate_roundtrip(self):
        with self.assertRaisesRegex(ValueError, "requires evidence"):
            Assumption("a", "condition", AssumptionStatus.VERIFIED)
        cert = self.certificate(verified("a"))
        self.assertEqual(Certificate.from_json(cert.to_json()), cert)
        payload = cert.to_dict()
        payload["assumptions"]["assumptions"][0]["status"] = "VIOLATED"
        with self.assertRaisesRegex(ValueError, "VIOLATED"):
            Certificate.from_dict(payload)

    def test_conditional_certificate_does_not_upgrade_uncheckable(self):
        assumptions = AssumptionSet((Assumption("a", "condition"),))
        cert = self.certificate(assumptions, high_confidence=False)
        self.assertFalse(cert.high_confidence)
        self.assertEqual(cert.assumptions.assumptions[0].status, AssumptionStatus.UNCHECKABLE)

    def test_envelope_requires_all_declared_conditions(self):
        envelope = OperatingEnvelope("operating", "toy", {"x": (0, 1)}, {"domain": ("synthetic",)})
        self.assertTrue(envelope.contains({"x": 0.5, "domain": "synthetic"}))
        for conditions in (
            {"x": 0.5},
            {"x": 2, "domain": "synthetic"},
            {"x": math.nan, "domain": "synthetic"},
            {"x": 0.5, "domain": "clinical"},
        ):
            self.assertFalse(envelope.contains(conditions))

    def test_evidence_graph_roundtrip_cycle_dangling_and_impact(self):
        nodes = tuple(
            EvidenceNode(name, kind, f"fixture:{name}", EvidenceClass.NUMERICAL_VERIFICATION)
            for name, kind in (
                ("model", EvidenceNodeKind.MODEL),
                ("test", EvidenceNodeKind.TEST),
                ("claim", EvidenceNodeKind.CLAIM),
            )
        )
        graph = EvidenceGraph(nodes, (EvidenceEdge("model", "test"), EvidenceEdge("test", "claim")))
        self.assertEqual(graph.affected_by("model"), ("claim", "test"))
        self.assertEqual(EvidenceGraph.from_json(graph.to_json()), graph)
        dependency_graph = EvidenceGraph(nodes, (EvidenceEdge("claim", "model", "depends_on"),))
        self.assertEqual(dependency_graph.affected_by("model"), ("claim",))
        with self.assertRaisesRegex(ValueError, "cycle"):
            EvidenceGraph(
                nodes,
                (
                    EvidenceEdge("model", "claim", "supports"),
                    EvidenceEdge("model", "claim", "depends_on"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "cycle"):
            EvidenceGraph(nodes, (*graph.edges, EvidenceEdge("claim", "model")))
        with self.assertRaisesRegex(ValueError, "dangling"):
            EvidenceGraph(nodes, (EvidenceEdge("missing", "claim"),))
        with self.assertRaisesRegex(ValueError, "not a theorem proof"):
            EvidenceNode("test", EvidenceNodeKind.TEST, "fixture", EvidenceClass.THEOREM)

    def test_theorem_and_claim_registries_have_honest_status_and_roundtrip(self):
        theorems = theorem_registry()
        self.assertEqual(len(theorems), 14)
        self.assertTrue(
            all(
                record.proof_status in {ProofStatus.UNPROVED, ProofStatus.COUNTEREXAMPLE}
                for record in theorems
            )
        )
        self.assertTrue(any(record.validation_tests for record in theorems))
        claims = scientific_claim_registry()
        self.assertEqual(len(claims), 9)
        for claim in claims:
            self.assertEqual(claim.evidence_class, EvidenceClass.INTERPRETATION)
            self.assertEqual(Claim.from_json(claim.to_json()), claim)

    def test_causal_identification_cannot_emit_without_required_assumptions(self):
        fields = dict(
            identification_id="id",
            estimand="causal_value",
            treatment="information",
            outcome="utility",
            time_zero="0",
            eligibility="synthetic units",
            treatment_strategies=("G", "G+M"),
            follow_up="10 s",
            censoring="none",
            observed_support="both synthetic branches",
            estimator="fixture enumeration",
            uncertainty="fixed model",
            sensitivity="fixture",
            limitations=("synthetic",),
            status=CausalStatus.IDENTIFIED,
        )
        with self.assertRaisesRegex(ValueError, "missing mandatory"):
            IdentificationRecord(assumptions=verified("temporal_ordering"), **fields)
        record = IdentificationRecord(assumptions=verified(*CAUSAL_REQUIREMENTS), **fields)
        self.assertEqual(IdentificationRecord.from_json(record.to_json()), record)

    def test_numerical_certificate_cannot_be_upgraded_to_theorem_by_relabeling(self):
        certificate = self.certificate(verified("a"))
        with self.assertRaisesRegex(ValueError, "proof artifact/status"):
            replace(certificate, evidence_class=EvidenceClass.THEOREM)


class TaskLawTests(unittest.TestCase):
    @unittest.skipIf(np is None, "optional NumPy not installed")
    def test_numpy_arrays_scalars_and_unsigned_subtraction_preserve_estimands(self):
        p = np.array((0, 1), dtype=np.uint8)
        q = np.array((1, 0), dtype=np.uint8)
        self.assertEqual(total_variation(p, q), 1)
        self.assertEqual(wasserstein_1d(p, q, np.array((0, 1))), 1)
        law = conditional_task_law(
            np.array((0, 2, 1, 1), dtype=np.int64), np.array((0, 0, 1, 1)), task()
        )
        self.assertEqual(TaskLaw.from_json(law.to_json()), law)
        probabilities = np.array((0.9, 0.1))
        self.assertTrue(
            binary_forecasting_certificate(
                probabilities,
                np.array((0.5, 0.5)),
                assumptions=(
                    "binary_outcome",
                    "same_target",
                    "same_population",
                    "representation_is_measurable",
                ),
            )["valid"]
        )
        region = CapabilityRegion({"p": np.array((1, 2), dtype=np.int64)}, "numpy")
        self.assertEqual(region.support(np.array((1, 1))), (3, "p"))
        self.assertEqual(CapabilityRegion.from_json(region.to_json()), region)
        self.assertEqual(
            lower_tail_cvar(np.array((0, 10)), np.array((0.2, 0.8)), tail_probability=0.5), 6
        )

    def test_empirical_laws_retain_distinct_distributions_with_equal_means(self):
        law = conditional_task_law((0, 2, 1, 1), ("spread", "spread", "point", "point"), task())
        spread, point = law.distribution("spread"), law.distribution("point")
        self.assertEqual(spread.expectation(), point.expectation())
        self.assertEqual(spread.support, (0.0, 2.0))
        self.assertEqual(spread.probabilities, (0.5, 0.5))
        self.assertEqual(point.support, (1.0,))
        self.assertNotEqual(spread.cdf(1), point.cdf(1))
        self.assertEqual(spread.quantile(0.75), 2)
        self.assertEqual(TaskLaw.from_json(law.to_json()), law)

    def test_task_function_discrepancy_catches_information_hidden_by_equal_means(self):
        spread = DiscreteDistribution((0, 2), (0.5, 0.5))
        point = DiscreteDistribution((1,), (1,))
        result = task_function_discrepancy(
            (spread,), (point,), {"mean": lambda y: y, "squared": lambda y: y * y}
        )
        self.assertEqual(result.discrepancies, {"mean": 0, "squared": 1})
        self.assertEqual(result.worst_task, 1)
        self.assertEqual(result.average_task, 0.5)

    def test_conditional_history_expectations_match_task_function_estimand(self):
        law = conditional_task_law(
            (0, 2, 1, 1),
            ("z",) * 4,
            task(),
            history_groups=("h1", "h1", "h2", "h2"),
            task_function=lambda y: y * y,
            task_function_name="squared",
        )
        self.assertEqual(law.full_information_values, (2, 2, 1, 1))
        self.assertEqual(law.representation_values, (1.5,) * 4)
        self.assertEqual(law.discrepancy(), 0.5)
        with self.assertRaisesRegex(ValueError, "task_function_name"):
            conditional_task_law((0, 1), ("z", "z"), task(), task_function=lambda y: y * y)

    def test_gaussian_conditional_law_has_analytic_weights(self):
        laws = kernel_conditional_distribution((0, 1), ((0,), (2,)), ((0,), (1,)), bandwidth=1)
        self.assertAlmostEqual(laws[0].probabilities[1], math.exp(-2) / (1 + math.exp(-2)))
        self.assertEqual(laws[1].probabilities, (0.5, 0.5))
        with self.assertRaises(ValueError):
            kernel_conditional_distribution((0, 1), ((0,), (2,)), ((1,),), bandwidth=0)

    def test_cluster_bootstrap_refits_and_preserves_whole_clusters(self):
        seen = []
        values = (0, 0, 10, 10)

        def estimator(indices):
            self.assertEqual(indices.count(0), indices.count(1))
            self.assertEqual(indices.count(2), indices.count(3))
            seen.append(indices)
            return sum(values[index] for index in indices) / len(indices)

        result = clustered_bootstrap(
            4, estimator, clusters=("a", "a", "b", "b"), repetitions=100, seed=5
        )
        self.assertEqual(len(seen), 101)
        self.assertEqual(result.point_estimate, 5)
        self.assertEqual(set(result.replicates), {0, 5, 10})
        self.assertEqual(result.resampling_unit, "cluster")
        self.assertEqual(BootstrapResult.from_json(result.to_json()), result)
        self.assertEqual(
            result.to_json(),
            clustered_bootstrap(
                4, estimator, clusters=("a", "a", "b", "b"), repetitions=100, seed=5
            ).to_json(),
        )

    def test_bootstrap_point_and_bootstrap_mean_are_separate(self):
        result = clustered_bootstrap(2, lambda indices: max(indices), repetitions=100, seed=1)
        self.assertEqual(result.point_estimate, 1)
        self.assertLess(result.bootstrap_mean, result.point_estimate)

    def test_task_family_duplicate_and_context_substitution_rejected(self):
        law = conditional_task_law((0, 1), ("z", "z"), task())
        with self.assertRaisesRegex(ValueError, "unique"):
            total_task_discrepancy((law, law))
        with self.assertRaises(EstimandMismatch):
            conditional_task_law((0, 1), ("z", "z"), task(), context=context(metric_type="auc"))

    def test_invalid_distributions_nonfinite_and_unsorted_support_rejected(self):
        for p, q in (((0.2, 0.2), (0.5, 0.5)), ((math.nan, 1), (0, 1)), ((-1, 2), (0, 1))):
            with self.subTest(p=p), self.assertRaises(ValueError):
                total_variation(p, q)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            wasserstein_1d((0.5, 0.5), (1, 0), (1, 0))
        with self.assertRaises(ValueError):
            empirical_conditional_distribution((math.nan,), ("z",))
        with self.assertRaises(ValueError):
            expected_calibration_error((1.1,), (1,))

    def test_binary_typed_certificate_is_guarded_not_a_causal_claim(self):
        names = (
            "binary_outcome",
            "same_target",
            "same_population",
            "representation_is_measurable",
            "correct_conditional_laws",
        )
        result = binary_forecasting_certificate(
            (0.9, 0.1), (0.5, 0.5), assumptions=verified(*names)
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result["high_confidence"])
        self.assertAlmostEqual(result["gap"], 0.4)
        self.assertAlmostEqual(result["D_tau"], 0.4)
        unverified = AssumptionSet(tuple(Assumption(name, name) for name in names))
        with self.assertRaisesRegex(ValueError, "UNCHECKABLE"):
            binary_forecasting_certificate((0.9,), (0.5,), assumptions=unverified)
        with self.assertRaisesRegex(ValueError, "not an intervention"):
            binary_forecasting_certificate(
                (0.9,),
                (0.5,),
                assumptions=verified(*names),
                task=replace(task(), kind=TaskKind.INTERVENTIONAL_OUTCOME),
            )

    def test_conditional_coverage_is_stratified(self):
        self.assertEqual(
            conditional_coverage((0, 0, 0), (1, 1, 1), (0.5, 2, 0), ("a", "a", "b")),
            {"a": 0.5, "b": 1},
        )


class CapabilityTests(unittest.TestCase):
    def test_independent_support_not_same_policy_difference(self):
        full = CapabilityRegion({"a": (10,), "b": (9,)}, "H")
        reduced = CapabilityRegion({"a": (0,), "b": (9,)}, "Z")
        result = support_gap(full, reduced, (1,))
        self.assertEqual(result["gap"], 1)
        self.assertEqual((result["full_policy"], result["reduced_policy"]), ("a", "b"))
        self.assertNotEqual(
            result["gap"],
            max(full.policy_values[p][0] - reduced.policy_values[p][0] for p in full.policy_values),
        )

    def test_coordinate_maxima_not_joint_and_downward_closure(self):
        region = CapabilityRegion({"a": (1, 0), "b": (0, 1)}, "crossing")
        self.assertEqual(region.coordinatewise_max(), (1, 1))
        self.assertFalse(region.jointly_attainable((1, 1)))
        self.assertFalse(region.jointly_attainable((0.5, 0.5)))
        self.assertTrue(region.requirement_region().contains((0.8, -1)))
        self.assertFalse(region.jointly_attainable((0.8, -1), downward_closure=False))

    def test_randomized_convexification_is_explicit_with_a_mixture_witness(self):
        region = CapabilityRegion({"a": (1, 0), "b": (0, 1)}, "crossing", randomized_policies=True)
        witness = region.attainability((0.5, 0.5), downward_closure=False)
        self.assertTrue(witness.attainable)
        self.assertTrue(witness.randomized)
        self.assertEqual(witness.policy_weights, {"a": 0.5, "b": 0.5})
        self.assertFalse(region.jointly_attainable((1, 1)))
        self.assertEqual(region.support((1, 1))[0], 1)
        self.assertEqual(CapabilityRegion.from_json(region.to_json()), region)

    def test_three_policy_randomized_joint_requirement(self):
        region = CapabilityRegion({"a": (1, 0, 0), "b": (0, 1, 0), "c": (0, 0, 1)}, "simplex", True)
        witness = region.attainability((1 / 3,) * 3, downward_closure=False)
        self.assertTrue(witness.attainable)
        self.assertAlmostEqual(sum(witness.policy_weights.values()), 1)
        self.assertFalse(region.jointly_attainable((0.34,) * 3))

    def test_dominance_checks_full_randomized_hull_in_nonconvex_requirement_region(self):
        hull = CapabilityRegion({"a": (1, 0), "b": (0, 1)}, "mixed", True)
        endpoints_only = CapabilityRegion({"a": (1, 0), "b": (0, 1)}, "deterministic")
        self.assertFalse(endpoints_only.dominates(hull))  # same vertices, uncovered mixed policies
        covered_by_two_policies = CapabilityRegion({"a": (1, 0.6), "b": (0.6, 1)}, "cover")
        self.assertTrue(covered_by_two_policies.dominates(hull))
        self.assertFalse(covered_by_two_policies.jointly_attainable(hull.coordinatewise_max()))

    def test_pareto_mean_worst_and_convex_dominance(self):
        region = CapabilityRegion(
            {"mean": (1, 0.4), "balanced": (0.6, 0.6), "dominated": (0.1, 0.1)}, "tasks"
        )
        self.assertEqual(set(region.pareto_front().policy_values), {"mean", "balanced"})
        self.assertGreater(sum(region.policy_values["mean"]), sum(region.policy_values["balanced"]))
        self.assertLess(region.worst_task("mean"), region.worst_task("balanced"))
        convex = CapabilityRegion({"a": (1, 0), "b": (0, 1), "c": (0.3, 0.3)}, "convex", True)
        self.assertEqual(set(convex.pareto_front().policy_values), {"a", "b"})
        self.assertEqual(
            CapabilityRegion({"only": (1, 1)}, "single", True).pareto_front().dimension, 2
        )

    def test_common_policy_robustness_differs_from_oracle(self):
        values = {"a": (4, 0), "b": (0, 3)}
        self.assertEqual(robust_value(values), 0)
        with self.assertRaises(ValueError):
            robust_value(values, order="inf_sup")
        self.assertEqual(optimistic_value(values), 3)
        region = RobustCapabilityRegion(
            {p: tuple((v,) for v in row) for p, row in values.items()}, "toy", ("k1", "k2")
        )
        self.assertEqual(region.optimistic_support((1,)), 3)  # not max over all kernels (=4)
        self.assertEqual(RobustCapabilityRegion.from_json(region.to_json()), region)
        with self.assertRaisesRegex(ValueError, "mapping"):
            robust_value((4, 3))
        with self.assertRaisesRegex(ValueError, "aligned"):
            RobustCapabilityRegion({"a": ((1,),), "b": ((1,), (2,))}, "bad")

    def test_equal_total_harvest_does_not_preserve_deadline_capability(self):
        before_deadline, after_deadline = (1, 0), (0, 1)
        self.assertEqual(sum(before_deadline), sum(after_deadline))
        regions = [
            CapabilityRegion(
                {"skip": (0,), **({"act_at_deadline": (1,)} if harvest[0] >= 1 else {})},
                "declared-deadline-task",
            )
            for harvest in (before_deadline, after_deadline)
        ]
        self.assertEqual(regions[0].support((1,))[0], 1)
        self.assertEqual(regions[1].support((1,))[0], 0)

    def test_robust_direction_is_not_coordinatewise_worst_scalarization(self):
        region = RobustCapabilityRegion({"a": ((1, 0), (0, 1))}, "toy")
        self.assertEqual(region.support((1, 1))[0], 1)
        self.assertFalse(region.requirement_region().contains((0.1, 0.1)))

    def test_weighted_cvar_fractional_tail_and_common_policy(self):
        self.assertEqual(lower_tail_cvar((0, 10), (0.2, 0.8), tail_probability=0.5), 6)
        region = StochasticCapabilityRegion(
            {"risky": ((0,), (10,)), "safe": ((4,), (4,))}, (0.2, 0.8), "risk"
        )
        self.assertEqual(region.cvar_support((1,), tail_probability=0.2), (4, "safe"))
        self.assertEqual(region.cvar_support((1,), tail_probability=0.5), (6, "risky"))
        self.assertEqual(StochasticCapabilityRegion.from_json(region.to_json()), region)

    def test_chance_constraint_is_joint_not_coordinatewise(self):
        region = StochasticCapabilityRegion(
            {"cross": ((1, 0), (0, 1)), "safe": ((0.5, 0.5), (0.5, 0.5))}, (0.5, 0.5), "chance"
        )
        result = region.chance_capability((0.5, 0.5), minimum_probability=0.5)
        self.assertEqual(result.policy_probabilities, {"cross": 0, "safe": 1})
        self.assertEqual(result.feasible_policies, ("safe",))

    def test_resource_dominance_requires_policy_relation_not_resource_coordinates(self):
        strong = CapabilityRegion({"p": (1, 1)}, "resource-A")
        weak = CapabilityRegion({"q": (0.5, 0.6)}, "resource-B")
        relation = PolicyRelation(
            "resource-A",
            "resource-B",
            PolicyRelationKind.POLICY_EMULATION,
            {"q": "p"},
            verified("feasible_policy_emulation"),
            "synthetic-witness",
        )
        self.assertTrue(resource_dominates(strong, weak, relation=relation, strict=True))
        self.assertFalse(
            resource_dominates(
                CapabilityRegion({"p": (1, 0)}, "resource-A"), weak, relation=relation
            )
        )
        with self.assertRaisesRegex(ValueError, "policy capability"):
            resource_dominates((10, 20), (1, 2), relation=relation)
        with self.assertRaisesRegex(ValueError, "every dominated"):
            resource_dominates(strong, weak, relation=replace(relation, policy_map={}))

    def test_contextual_support_and_resource_constraints_preserve_semantics(self):
        region = CapabilityRegion(
            {"p": (1,), "q": (0.5,)}, "H", context=context(information_condition="H")
        )
        reduced = CapabilityRegion({"z": (0.6,)}, "Z", context=context(information_condition="Z"))
        self.assertAlmostEqual(support_gap(region, reduced, (1,))["gap"], 0.4)
        with self.assertRaises(EstimandMismatch):
            support_gap(
                region,
                replace(reduced, context=context(utility="other", information_condition="Z")),
                (1,),
            )
        constrained = condition_on_resource(region, ("q",), "tight-budget").at("tight-budget")
        self.assertEqual(constrained.context.resource_state, "tight-budget")
        self.assertEqual(constrained.support((1,)), (0.5, "q"))
        with self.assertRaisesRegex(ValueError, "unknown"):
            condition_on_resource(region, ("missing",), "r")

    def test_empty_resource_feasible_set_retains_task_dimension(self):
        region = CapabilityRegion({"p": (1, 1)}, "toy")
        empty = condition_on_resource(region, (), "no-feasible-policy").at("no-feasible-policy")
        self.assertEqual(empty.dimension, 2)
        self.assertEqual(empty.support((1, 1)), (-math.inf, None))
        self.assertFalse(empty.requirement_region().contains((0, 0)))
        self.assertEqual(empty.requirement_region().support((-1, 1)), (-math.inf, None))
        self.assertEqual(CapabilityRegion.from_json(empty.to_json()), empty)

    def test_blackwell_garbling_and_strategic_measure(self):
        full = kernel("world", "full", ((1, 0), (0, 1)))
        garbling = kernel("full", "reduced", ((0.5, 0.5), (0.5, 0.5)))
        reduced = kernel("world", "reduced", ((0.5, 0.5), (0.5, 0.5)))
        self.assertTrue(blackwell_dominates(full, reduced, garbling))
        self.assertFalse(
            blackwell_dominates(full, kernel("world", "reduced", ((1, 0), (0, 1))), garbling)
        )
        measure = StrategicMeasure(
            "p", (("s", "good"), ("s", "bad")), (0.75, 0.25), "H", ("a",), "initial"
        )
        self.assertEqual(measure.value({("s", "good"): 1, ("s", "bad"): -1}), 0.5)
        self.assertEqual(StrategicMeasure.from_json(measure.to_json()), measure)

    def test_adversarial_nonfinite_dimensions_and_negative_requirement_support(self):
        with self.assertRaises(ValueError):
            CapabilityRegion({"p": (math.nan,)}, "bad")
        region = CapabilityRegion({"p": (1, 1)}, "toy")
        with self.assertRaises(ValueError):
            region.support((1,))
        with self.assertRaisesRegex(ValueError, "unbounded"):
            region.requirement_region().support((-1, 1))
        with self.assertRaises(ValueError):
            RobustCapabilityRegion({"p": ((1, 2),)}, "toy").support((1,))


class ValueTests(unittest.TestCase):
    @unittest.skipIf(np is None, "optional NumPy not installed")
    def test_unsigned_value_inputs_cannot_wrap_counterfactual_losses(self):
        design = model_design()
        result = estimate_piv(
            np.array((10, 1), dtype=np.uint8),
            np.array((1, 10), dtype=np.uint8),
            design=design,
            baseline_pair_ids=design.pair_ids,
            augmented_pair_ids=design.pair_ids,
        )
        self.assertEqual(result.paired_differences, (-9, 9))
        self.assertEqual(result.incremental_value, 0)
        signed_pir = compute_pir(
            np.uint8(1),
            np.uint8(2),
            np.uint8(1),
            np.uint8(0),
            references={"task": "signed-synthetic-comparison"},
        )
        self.assertEqual(signed_pir.information_loss, -1)
        self.assertEqual(signed_pir.pir, 2)  # not clipped; no nesting assumption was asserted

    def test_paired_counterfactual_information_piv_and_roundtrip(self):
        design = model_design()
        result = estimate_piv(
            (1, 3),
            (2, 2),
            design=design,
            baseline_pair_ids=design.pair_ids,
            augmented_pair_ids=design.pair_ids,
            bootstrap_repetitions=100,
            seed=1,
        )
        self.assertEqual(result.incremental_value, 0)
        self.assertEqual(result.paired_differences, (1, -1))
        self.assertEqual(result.evidence_class, EvidenceClass.SIMULATION)
        self.assertEqual(result.context.metric_type, "piv")
        self.assertEqual(PIVEstimate.from_json(result.to_json()), result)

    def test_piv_rejects_unpaired_data_and_incompatible_reference(self):
        design = model_design()
        with self.assertRaisesRegex(ValueError, "not aligned"):
            estimate_piv(
                (1, 3),
                (2, 2),
                design=design,
                baseline_pair_ids=design.pair_ids,
                augmented_pair_ids=tuple(reversed(design.pair_ids)),
            )
        for field, value in (
            ("utility", "other"),
            ("resource_state", "other"),
            ("reference_state", "other"),
            ("time_horizon", 20),
        ):
            with self.subTest(field=field), self.assertRaises(EstimandMismatch):
                replace(
                    design, augmented_context=replace(design.augmented_context, **{field: value})
                )
        with self.assertRaisesRegex(ValueError, "UNCHECKABLE"):
            replace(
                design,
                assumptions=AssumptionSet(
                    tuple(Assumption(name, name) for name in PIV_REQUIREMENTS)
                ),
            )

    def test_piv_does_not_upgrade_simulation_or_observation_to_identification(self):
        design = model_design()
        with self.assertRaisesRegex(ValueError, "identification record"):
            replace(design, kind=InformationDesignKind.IDENTIFIED_PAIRED_COUNTERFACTUAL)
        with self.assertRaisesRegex(ValueError, "MODEL or SIMULATION"):
            replace(
                design,
                baseline_context=replace(
                    design.baseline_context, evidence_class=EvidenceClass.EMPIRICAL_ESTIMATE
                ),
                augmented_context=replace(
                    design.augmented_context, evidence_class=EvidenceClass.EMPIRICAL_ESTIMATE
                ),
            )

    def test_pir_common_reference_decomposition_and_dimensionless_context(self):
        ideal = context(information_condition="H")
        actual = context(information_condition="Z")
        contexts = {
            "ideal": ideal,
            "actual_optimal": actual,
            "actual_policy": actual,
            "minimum": actual,
        }
        result = compute_pir(
            1, 0.8, 0.7, 0, value_contexts=contexts, information_conditions=("H", "Z")
        )
        self.assertAlmostEqual(result.pir, 0.8)
        self.assertAlmostEqual(result.information_loss + result.policy_loss, result.total_loss)
        self.assertTrue(result.reference_checked)
        self.assertEqual(result.metric_context.metric_type, "pir")
        self.assertEqual(result.metric_context.unit, "1")
        self.assertEqual(PIRResult.from_json(result.to_json()), result)
        altered = {**contexts, "actual_policy": replace(actual, communication_regime="erasure")}
        with self.assertRaises(EstimandMismatch):
            compute_pir(1, 0.8, 0.7, 0, value_contexts=altered, information_conditions=("H", "Z"))

    def test_pir_reference_zero_denominator_and_policy_optimum_checks(self):
        result = compute_pir(0, 0, 0, 0, references={"task": "legacy-synthetic"})
        self.assertIsNone(result.pir)
        self.assertFalse(result.reference_checked)
        self.assertTrue(result.limitations)
        with self.assertRaisesRegex(ValueError, "common references"):
            compute_pir(1, 0.8, 0.7, 0)
        with self.assertRaisesRegex(ValueError, "exceeds.*optimum"):
            compute_pir(1, 0.8, 0.9, 0, references={"task": "toy"})
        with self.assertRaisesRegex(ValueError, "minimum reference"):
            compute_pir(1, 0.8, 0.7, 0.9, references={"task": "toy"})

    def test_serialized_value_outputs_reject_forged_algebra_and_evidence(self):
        pir = compute_pir(1, 0.8, 0.7, 0, references={"task": "toy"})
        payload = pir.to_dict()
        payload["information_loss"] = 0.01
        with self.assertRaisesRegex(ValueError, "inconsistent loss"):
            PIRResult.from_dict(payload)
        design = model_design()
        piv = estimate_piv(
            (1, 3),
            (2, 2),
            design=design,
            baseline_pair_ids=design.pair_ids,
            augmented_pair_ids=design.pair_ids,
        )
        with self.assertRaisesRegex(ValueError, "paired utility differences"):
            replace(piv, incremental_value=10)
        with self.assertRaisesRegex(ValueError, "upgrade"):
            replace(piv, evidence_class=EvidenceClass.CLINICAL_EVIDENCE)
        with self.assertRaisesRegex(ValueError, "context/estimator"):
            replace(piv, context=replace(piv.context, utility="unrelated utility"))

    def test_optimal_constrained_task_value_and_deficiency(self):
        full = optimal_task_value({"a": 1, "b": 0.6}, context=context(information_condition="H"))
        reduced = optimal_task_value(
            {"c": 0.8, "d": 0.5}, context=context(information_condition="Z")
        )
        deficiency = representation_deficiency(full, reduced)
        self.assertAlmostEqual(deficiency.value, 0.2)
        self.assertEqual(deficiency.context.metric_type, "representation_deficiency")
        self.assertEqual(type(deficiency).from_json(deficiency.to_json()), deficiency)
        constrained = optimal_task_value(
            {"c": 0.8, "d": 0.5}, context=context(), feasible_policies=("d",)
        )
        self.assertEqual(constrained.value, 0.5)


class FlowAndBoundTests(unittest.TestCase):
    def graph(self):
        nodes = (
            FlowNode("h", FlowNodeType.HISTORY, "H"),
            FlowNode("z", FlowNodeType.REPRESENTATION, "Z"),
            FlowNode("p", FlowNodeType.PREDICTION, "P"),
        )
        edges = (
            FlowEdge(
                "compress",
                "h",
                "z",
                FlowEdgeType.COMPRESSION,
                "finite stochastic encoder",
                kernel("H", "Z", ((1, 0), (0, 1))),
                kernel("H", "Z", ((0.9, 0.1), (0.1, 0.9))),
                local_guarantee=LocalGuarantee("uniform_kernel_tv", 0.1, "H"),
            ),
            FlowEdge(
                "infer",
                "z",
                "p",
                FlowEdgeType.INFERENCE,
                "constant forecast kernel",
                kernel("Z", "P", ((0.5, 0.5), (0.5, 0.5))),
                kernel("Z", "P", ((0.5, 0.5), (0.5, 0.5))),
                local_guarantee=LocalGuarantee("uniform_kernel_tv", 0, "Z"),
            ),
        )
        return InformationFlowGraph(nodes, edges)

    def test_real_composition_uses_contraction_not_blanket_sum(self):
        graph = self.graph()
        cert = graph.certify_pipeline(
            ("compress", "infer"),
            certificate_id="pipeline",
            validity_domain=OperatingEnvelope("finite", "synthetic finite kernels"),
        )
        self.assertEqual(cert.resulting_guarantee.bound, 0)
        self.assertEqual(sum(g.bound for g in cert.local_guarantees), 0.1)
        self.assertIn("Dobrushin", cert.composition_rule)
        self.assertEqual(ScientificPipelineCertificate.from_json(cert.to_json()), cert)
        self.assertEqual(InformationFlowGraph.from_json(graph.to_json()), graph)
        payload = cert.to_dict()
        payload["resulting_guarantee"]["bound"] = 0.1
        with self.assertRaisesRegex(ValueError, "inconsistent composition"):
            ScientificPipelineCertificate.from_dict(payload)

    def test_pipeline_rejects_mixed_losses_unconnected_path_and_understated_guarantee(self):
        graph = self.graph()
        kwargs = dict(certificate_id="bad", validity_domain=OperatingEnvelope("finite", "toy"))
        with self.assertRaisesRegex(ValueError, "connected"):
            graph.certify_pipeline(("infer", "compress"), **kwargs)
        for guarantee in (
            LocalGuarantee("auc_loss", 0.1, "H"),
            LocalGuarantee("uniform_kernel_tv", 0.01, "H"),
        ):
            changed = replace(
                graph, edges=(replace(graph.edges[0], local_guarantee=guarantee), graph.edges[1])
            )
            with self.assertRaises(ValueError):
                changed.certify_pipeline(("compress", "infer"), **kwargs)

    def test_pipeline_guards_assumptions_and_dag_validity(self):
        graph = self.graph()
        blocked_edge = replace(
            graph.edges[0],
            assumptions=AssumptionSet(
                (Assumption("kernel_valid", "valid", AssumptionStatus.VIOLATED),)
            ),
        )
        with self.assertRaisesRegex(ValueError, "VIOLATED"):
            replace(graph, edges=(blocked_edge, graph.edges[1])).certify_pipeline(
                ("compress",), certificate_id="bad", validity_domain=OperatingEnvelope("toy", "toy")
            )
        with self.assertRaisesRegex(ValueError, "cycle"):
            replace(
                graph,
                edges=(
                    *graph.edges,
                    FlowEdge("back", "p", "h", FlowEdgeType.PREPROCESSING, "bad feedback"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "dangling"):
            replace(graph, edges=(replace(graph.edges[0], target="missing"),))
        with self.assertRaisesRegex(ValueError, "sensing"):
            replace(graph, edges=(replace(graph.edges[0], kind=FlowEdgeType.SENSING),))

    def test_kernel_validation_and_dobrushin_analytic_bound(self):
        reference = kernel("S", "S", ((0.75, 0.25), (0.25, 0.75)))
        perturbed = kernel("S", "S", ((0.8, 0.2), (0.3, 0.7)))
        self.assertEqual(reference.dobrushin_coefficient(), 0.5)
        self.assertAlmostEqual(reference.uniform_tv_error(perturbed), 0.05)
        result = dobrushin_stability_bound(reference, perturbed, steps=3)
        self.assertAlmostEqual(result.finite_time_bound, 0.0875)
        self.assertAlmostEqual(result.stationary_bound, 0.1)
        actual, nominal = (1, 0), (1, 0)
        for _ in range(3):
            actual = tuple(
                sum(actual[i] * perturbed.probabilities[i][j] for i in range(2)) for j in range(2)
            )
            nominal = tuple(
                sum(nominal[i] * reference.probabilities[i][j] for i in range(2)) for j in range(2)
            )
        self.assertLessEqual(total_variation(actual, nominal), result.finite_time_bound + 1e-12)
        identity = kernel("S", "S", ((1, 0), (0, 1)))
        self.assertIsNone(dobrushin_stability_bound(identity, identity, steps=3).stationary_bound)
        with self.assertRaisesRegex(ValueError, "probability"):
            kernel("S", "S", ((1, 1), (0, 1)))

    def test_finite_value_transfer_coefficient_matches_half_l1_tv(self):
        assumptions = verified(
            "common_feasible_actions",
            "uniform_reward_error",
            "uniform_transition_tv",
            "bounded_continuation_value",
        )
        bound = finite_kernel_error_bound((0,), (0.1,), (2,), assumptions=assumptions)
        # p=(1,0), q=(.9,.1), continuation=(2,-2): actual expectation error=.4.
        self.assertAlmostEqual(bound.bound, 0.4)
        self.assertEqual(bound.tv_convention, "TV=0.5*L1")
        accumulated = finite_kernel_error_bound(
            (0.1, 0.2), (0.05, 0), (2, 0), assumptions=assumptions
        )
        self.assertAlmostEqual(accumulated.bound, 0.5)
        with self.assertRaisesRegex(ValueError, "missing mandatory"):
            finite_kernel_error_bound((0,), (0,), (1,), assumptions=AssumptionSet())

    def test_bellman_contraction_is_a_discounted_value_bound(self):
        self.assertEqual(bellman_contraction_bound(0.9, 2), 1.8)
        for gamma in (-0.1, 1, math.nan):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                bellman_contraction_bound(gamma, 2)


if __name__ == "__main__":
    unittest.main()

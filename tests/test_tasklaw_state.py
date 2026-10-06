"""Focused tests for exact finite task-law state quotients."""

import unittest

from physioworld.core.contracts import EvidenceClass, Task, TaskKind
from physioworld.tasklaws.models import ConditionalCell, DiscreteDistribution, TaskLaw
from physioworld.tasklaws.state import (
    SufficiencyKind,
    TaskLawState,
    predictive_intervention_counterexample,
)


def make_law(task_id: str, cells: dict[str, DiscreteDistribution], *, kind=TaskKind.PREDICTION):
    task = Task(
        task_id,
        kind,
        1.0,
        "Y",
        "squared_error",
        intervention_semantics="declared_do(A)"
        if kind == TaskKind.INTERVENTIONAL_OUTCOME
        else None,
    )
    conditional = tuple(ConditionalCell(history, law, 1) for history, law in cells.items())
    expectations = tuple(law.expectation() for law in cells.values())
    return TaskLaw(
        task,
        expectations,
        expectations,
        tuple(cells),
        conditional_distributions=conditional,
        history_distributions=conditional,
        estimator="test_declared_finite_laws",
        evidence_class=EvidenceClass.MODEL,
    )


class TaskLawStateTests(unittest.TestCase):
    def test_coarsest_grouping_preserves_complete_laws_not_only_means(self):
        laws = make_law(
            "outcome",
            {
                "h0": DiscreteDistribution((0.0, 2.0), (0.5, 0.5)),
                "h1": DiscreteDistribution((2.0, 0.0), (0.5, 0.5)),
                "h2": DiscreteDistribution((1.0,), (1.0,)),
            },
        )
        state = TaskLawState((laws,))

        self.assertEqual(state.classes, (("h0", "h1"), ("h2",)))
        self.assertEqual(state.class_index("h0"), state.class_index("h1"))
        self.assertNotEqual(state.class_index("h0"), state.class_index("h2"))
        self.assertEqual(
            state.distribution("h1", "outcome"), laws.history_distributions[0].distribution
        )
        with self.assertRaises(KeyError):
            state.class_index("unseen")

    def test_diagnostic_has_declared_estimand_and_exact_finite_evidence(self):
        laws = make_law(
            "outcome",
            {
                "h0": DiscreteDistribution((0.0,), (1.0,)),
                "h1": DiscreteDistribution((0.0,), (1.0,)),
                "h2": DiscreteDistribution((1.0,), (1.0,)),
            },
        )
        state = TaskLawState((laws,))
        sufficient = state.diagnose({"h0": "left", "h1": "left", "h2": "right"})
        insufficient = state.diagnose({history: "one_state" for history in state.histories})

        self.assertTrue(sufficient.sufficient)
        self.assertTrue(sufficient.predictive_sufficient)
        self.assertEqual(sufficient.max_total_variation, 0.0)
        self.assertEqual(sufficient.minimal_classes, 2)
        self.assertEqual(sufficient.evidence_class, EvidenceClass.NUMERICAL_VERIFICATION)
        self.assertEqual(sufficient.source_evidence_classes, (EvidenceClass.MODEL,))
        self.assertEqual(
            sufficient.estimand,
            "max_within_state_task_law_total_variation_over_declared_finite_histories",
        )
        self.assertFalse(insufficient.sufficient)
        self.assertEqual(insufficient.max_total_variation, 1.0)
        self.assertEqual(insufficient.violations[0].task_id, "outcome")
        self.assertEqual(insufficient.violations[0].state_label, "one_state")

    def test_intervention_diagnostic_is_distinct_and_requires_declared_regime_semantics(self):
        predictive, intervention = predictive_intervention_counterexample()

        self.assertEqual(predictive.kind, SufficiencyKind.PREDICTIVE)
        self.assertEqual(intervention.kind, SufficiencyKind.INTERVENTION)
        self.assertEqual(predictive.classes, ((0, 1),))
        self.assertEqual(intervention.classes, ((0,), (1,)))
        self.assertTrue(predictive.diagnose({0: "same", 1: "same"}).predictive_sufficient)
        intervention_result = intervention.diagnose({0: "same", 1: "same"})
        self.assertFalse(intervention_result.sufficient)
        self.assertIsNone(intervention_result.predictive_sufficient)
        self.assertTrue(intervention_result.intervention_sufficient is False)
        self.assertEqual(intervention_result.max_total_variation, 1.0)
        self.assertEqual(intervention_result.violations[0].task_id, "do_A1")
        self.assertEqual(intervention_result.source_evidence_classes, (EvidenceClass.MODEL,) * 2)

    def test_missing_history_laws_and_wrong_endpoint_kind_are_rejected(self):
        predictive = make_law(
            "outcome",
            {"h0": DiscreteDistribution((0.0,), (1.0,))},
        )
        intervention = make_law(
            "do_outcome",
            {"h0": DiscreteDistribution((0.0,), (1.0,))},
            kind=TaskKind.INTERVENTIONAL_OUTCOME,
        )
        with self.assertRaises(ValueError):
            TaskLawState((intervention,))
        no_history = TaskLaw(
            predictive.task,
            (0.0,),
            (0.0,),
            conditioning_labels=("h0",),
            evidence_class=EvidenceClass.MODEL,
        )
        with self.assertRaises(ValueError):
            TaskLawState((no_history,))


if __name__ == "__main__":
    unittest.main()

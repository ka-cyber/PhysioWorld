"""Exact task-law states on a declared finite history domain.

For supplied kernels Q_tau(. | h), define h ~ h' exactly when every declared
task's probability law agrees. The resulting quotient is the coarsest
deterministic state preserving these kernels: a candidate z(h) is sufficient
iff each of its fibers lies inside one quotient class. Support order and
zero-mass atoms do not affect equality; positive masses are compared exactly,
without rounding, binning, or tolerance chaining.

Predictive kernels and intervention kernels define separate equivalence
relations. Predictive sufficiency leaves intervention sufficiency unassessed.
Even an intervention diagnostic only checks supplied kernels; it neither
identifies them from observations nor certifies their causal correctness.
This is not a recursive/Markov state construction or an unseen-history claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

from physioworld.core.contracts import CausalStatus, EvidenceClass, TaskKind, VersionedRecord

from .metrics import total_variation
from .models import ConditionalCell, DiscreteDistribution, TaskDefinition, TaskLaw


class SufficiencyKind(str, Enum):
    PREDICTIVE = "predictive_task_law"
    INTERVENTION = "intervention_task_law"


@dataclass(frozen=True)
class SufficiencyViolation(VersionedRecord):
    state_label: Any
    left_history: Any
    right_history: Any
    task_id: str
    total_variation: float


@dataclass(frozen=True)
class TaskLawSufficiencyDiagnostic(VersionedRecord):
    """Exhaustive within-state comparisons, with no history-frequency weighting."""

    kind: SufficiencyKind
    task_ids: tuple[str, ...]
    minimal_classes: int
    candidate_classes: tuple[tuple[Any, ...], ...]
    max_total_variation: float
    violations: tuple[SufficiencyViolation, ...]
    source_evidence_classes: tuple[EvidenceClass, ...]
    estimand: str = "max_within_state_task_law_total_variation_over_declared_finite_histories"
    evidence_class: EvidenceClass = EvidenceClass.NUMERICAL_VERIFICATION
    scope: str = "supplied_finite_histories_and_declared_task_laws"
    causal_identification: bool = False

    @property
    def sufficient(self) -> bool:
        # Exact equality, independently of floating-point distance arithmetic.
        return not self.violations

    @property
    def predictive_sufficient(self) -> bool | None:
        return self.sufficient if self.kind == SufficiencyKind.PREDICTIVE else None

    @property
    def intervention_sufficient(self) -> bool | None:
        return self.sufficient if self.kind == SufficiencyKind.INTERVENTION else None


def _signature(distribution: DiscreteDistribution) -> tuple[tuple[float, float], ...]:
    return tuple(
        sorted(
            (value, mass)
            for value, mass in zip(distribution.support, distribution.probabilities)
            if mass > 0
        )
    )


class TaskLawState:
    """Minimal deterministic quotient of full ``TaskLaw.history_distributions``.

    Every task must supply exactly one distribution for each history on the
    same nonempty finite domain. Histories and candidate labels use Python
    hash/equality semantics. Class numbering follows first appearance in the
    first task's history cells; task/cell reordering changes no equivalence.

    Expectation arrays, representation distributions, task functions, and cell
    counts are not substitutes for these full laws and do not determine the
    quotient. Intervention tasks require explicit intervention semantics, with
    one task per declared intervention/regime. No missing regimes are inferred.
    Source evidence remains attached to ``task_laws`` and diagnostics; numerical
    verification of equality does not validate the supplied kernels.
    """

    estimand = "coarsest_deterministic_quotient_preserving_declared_conditional_task_distributions"
    evidence_class = EvidenceClass.DEFINITION

    def __init__(
        self,
        task_laws: Sequence[TaskLaw],
        *,
        kind: SufficiencyKind = SufficiencyKind.PREDICTIVE,
    ) -> None:
        self.kind = SufficiencyKind(kind)
        self.task_laws = tuple(task_laws)
        if not self.task_laws:
            raise ValueError("a nonempty declared task family is required")
        self.task_ids = tuple(law.task.task_id for law in self.task_laws)
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task IDs must be unique")

        self._distributions: dict[str, dict[Any, DiscreteDistribution]] = {}
        predictive_kinds = {
            TaskKind.PREDICTION,
            TaskKind.CLASSIFICATION,
            TaskKind.DETECTION,
            TaskKind.FORECASTING,
            TaskKind.ESTIMATION,
            TaskKind.RANKING,
            TaskKind.EVENT_TIME,
        }
        intervention_kinds = {TaskKind.INTERVENTIONAL_OUTCOME, TaskKind.SEQUENTIAL_INTERVENTION}
        for law in self.task_laws:
            law.task.validate()
            allowed = (
                predictive_kinds if self.kind == SufficiencyKind.PREDICTIVE else intervention_kinds
            )
            if law.task.kind not in allowed:
                raise ValueError(f"task {law.task.task_id} is not a {self.kind.value} endpoint")
            if self.kind == SufficiencyKind.INTERVENTION and (
                not isinstance(law.task.intervention_semantics, str)
                or not law.task.intervention_semantics.strip()
            ):
                raise ValueError("intervention tasks require declared intervention semantics")
            if law.context is not None:
                predictive_context = law.context.causal_status in {
                    CausalStatus.PREDICTIVE,
                    CausalStatus.NOT_APPLICABLE,
                }
                if predictive_context != (self.kind == SufficiencyKind.PREDICTIVE):
                    raise ValueError("task-law causal context conflicts with sufficiency kind")
            if not law.history_distributions:
                raise ValueError(
                    "full history distributions are required; means alone are insufficient"
                )
            cells: dict[Any, DiscreteDistribution] = {}
            for cell in law.history_distributions:
                if cell.group in cells:
                    raise ValueError("each task must have exactly one law per history")
                cells[cell.group] = cell.distribution
            self._distributions[law.task.task_id] = cells

        self.histories = tuple(self._distributions[self.task_ids[0]])
        domain = set(self.histories)
        if any(set(cells) != domain for cells in self._distributions.values()):
            raise ValueError("all tasks must share exactly the same finite history domain")
        self._signatures = {
            history: tuple(_signature(self._distributions[task][history]) for task in self.task_ids)
            for history in self.histories
        }
        grouped: dict[tuple, list[Any]] = {}
        for history, signature in self._signatures.items():
            grouped.setdefault(signature, []).append(history)
        self.classes = tuple(tuple(histories) for histories in grouped.values())
        self._class_index = {
            history: index for index, histories in enumerate(self.classes) for history in histories
        }

    def class_index(self, history: Any) -> int:
        """State identity on the declared domain; unknown histories raise KeyError."""
        return self._class_index[history]

    def distribution(self, history: Any, task_id: str) -> DiscreteDistribution:
        """Recover a representative full law using only the quotient state."""
        representative = self.classes[self.class_index(history)][0]
        return self._distributions[task_id][representative]

    def diagnose(self, representation: Mapping[Any, Any]) -> TaskLawSufficiencyDiagnostic:
        """Check an explicitly supplied deterministic state map on the whole domain.

        The estimand is max_{h,h': z(h)=z(h')} max_tau TV(Q_tau|h, Q_tau|h').
        Empty within-state pair sets have discrepancy zero. All unequal pairs
        are retained as witnesses; sufficiency itself uses exact signatures,
        not a distance threshold. Runtime/output can be quadratic in histories.
        """
        if set(representation) != set(self.histories):
            raise ValueError("representation must cover exactly the declared history domain")
        grouped: dict[Any, list[Any]] = {}
        for history in self.histories:
            label = representation[history]
            try:
                hash(label)
            except TypeError as error:
                raise ValueError("state labels must be hashable") from error
            if label != label:
                raise ValueError("state labels must have reflexive equality")
            grouped.setdefault(label, []).append(history)

        violations = []
        maximum = 0.0
        for label, histories in grouped.items():
            for left, right in combinations(histories, 2):
                for index, task_id in enumerate(self.task_ids):
                    p, q = self._signatures[left][index], self._signatures[right][index]
                    if p == q:
                        continue
                    p_mass, q_mass = dict(p), dict(q)
                    support = sorted(set(p_mass) | set(q_mass))
                    distance = total_variation(
                        tuple(p_mass.get(atom, 0.0) for atom in support),
                        tuple(q_mass.get(atom, 0.0) for atom in support),
                    )
                    maximum = max(maximum, distance)
                    violations.append(SufficiencyViolation(label, left, right, task_id, distance))
        return TaskLawSufficiencyDiagnostic(
            kind=self.kind,
            task_ids=self.task_ids,
            minimal_classes=len(self.classes),
            candidate_classes=tuple(tuple(histories) for histories in grouped.values()),
            max_total_variation=maximum,
            violations=tuple(violations),
            source_evidence_classes=tuple(
                EvidenceClass(law.evidence_class) for law in self.task_laws
            ),
        )


def predictive_intervention_counterexample() -> tuple[TaskLawState, TaskLawState]:
    """Return predictive/intervention quotients for the finite model Y = A * H.

    H is either 0 or 1, and the observational policy always takes A = 0. Both
    histories thus predict Y = 0, but under do(A = 1), Y equals H. The predictive
    quotient has one class; preserving both declared intervention kernels needs
    two. These are MODEL laws, not observational causal identification: A = 1
    has no observational support. Histories here are the two pre-action H atoms.
    """

    def law(task_id: str, outcomes: tuple[float, float], intervention: str | None) -> TaskLaw:
        cells = tuple(
            ConditionalCell(history, DiscreteDistribution((outcome,), (1.0,)), 1)
            for history, outcome in enumerate(outcomes)
        )
        return TaskLaw(
            task=TaskDefinition(
                task_id,
                TaskKind.INTERVENTIONAL_OUTCOME if intervention else TaskKind.PREDICTION,
                1.0,
                "Y",
                "squared_error",
                intervention_semantics=intervention,
                reference_condition="observed A=0",
            ),
            full_information_values=outcomes,
            representation_values=outcomes,
            conditioning_labels=(0, 1),
            conditional_distributions=cells,
            history_distributions=cells,
            assumptions=("finite_structural_model_Y=A*H", "observational_policy_A=0"),
            estimator="declared_finite_model_kernels",
            evidence_class=EvidenceClass.MODEL,
        )

    predictive = TaskLawState((law("observed_Y", (0.0, 0.0), None),))
    intervention = TaskLawState(
        (law("do_A0", (0.0, 0.0), "do(A=0)"), law("do_A1", (0.0, 1.0), "do(A=1)")),
        kind=SufficiencyKind.INTERVENTION,
    )
    return predictive, intervention

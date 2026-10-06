"""Workload specifications are executable contracts, not clinical claims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from physioworld.tasklaws import TaskDefinition, TaskKind


@dataclass(frozen=True)
class ClinicalWorkload:
    workload_id: str
    title: str
    dataset_ids: tuple[str, ...]
    task: TaskDefinition
    ground_truth: str
    evaluation_protocol: str
    clinical_metric: str
    resource_model: str
    uncertainty_model: str
    failure_analysis: tuple[str, ...]
    clinical_validity: str = "research evaluation only; no clinical validity implied"

    def describe(self) -> Mapping[str, Any]:
        return {
            "workload_id": self.workload_id,
            "title": self.title,
            "datasets": self.dataset_ids,
            "task": self.task.task_id,
            "metric": self.clinical_metric,
            "clinical_validity": self.clinical_validity,
        }


def default_workloads() -> tuple[ClinicalWorkload, ...]:
    return (
        ClinicalWorkload(
            "arrhythmia_monitoring",
            "Arrhythmia monitoring",
            ("mitdb", "afdb"),
            TaskDefinition(
                "arrhythmia", TaskKind.CLASSIFICATION, 0.0, "rhythm_annotation", "macro_f1"
            ),
            "provider annotations",
            "subject-level split and temporal holdout",
            "macro-F1 and AUROC",
            "declared resource model",
            "subject bootstrap and calibration",
            ("annotation ambiguity", "class imbalance", "motion and missing leads"),
        ),
        ClinicalWorkload(
            "hypotension_forecasting",
            "Hypotension forecasting/detection",
            ("mimic_waveform",),
            TaskDefinition(
                "hypotension",
                TaskKind.FORECASTING,
                300.0,
                "future_MAP_below_threshold",
                "utility",
                clinical_relevance="clinical deterioration research",
            ),
            "provider MAP and event definition",
            "patient-level temporal evaluation",
            "AUROC, AUPRC, calibration, lead time",
            "declared resource model",
            "bootstrap and shift analysis",
            ("positivity", "irregular sampling", "intervention confounding"),
        ),
        ClinicalWorkload(
            "deterioration_detection",
            "Physiological deterioration detection",
            ("mimic_waveform", "physionet_challenge"),
            TaskDefinition(
                "deterioration", TaskKind.DETECTION, 3600.0, "future_deterioration", "utility"
            ),
            "dataset-specific outcome",
            "patient and temporal holdout",
            "AUPRC, alarm burden, calibration",
            "declared resource model",
            "epistemic and shift uncertainty",
            ("label timing", "care-process confounding"),
        ),
        ClinicalWorkload(
            "sleep_state",
            "Sleep-state analysis",
            ("sleep_edf",),
            TaskDefinition("sleep_state", TaskKind.CLASSIFICATION, 30.0, "sleep_stage", "macro_f1"),
            "scored epochs",
            "subject-independent epoch evaluation",
            "macro-F1, Cohen kappa, calibration",
            "declared resource model",
            "subject bootstrap",
            ("scorer disagreement", "stage imbalance"),
        ),
        ClinicalWorkload(
            "mobility_activity",
            "Mobility/activity analysis",
            ("pamap2",),
            TaskDefinition("activity", TaskKind.CLASSIFICATION, 1.0, "activity_label", "macro_f1"),
            "subject annotations",
            "subject-independent split",
            "macro-F1 and balanced accuracy",
            "declared resource model",
            "subject uncertainty",
            ("transition boundaries", "device placement"),
        ),
        ClinicalWorkload(
            "rehabilitation_assessment",
            "Rehabilitation/mobility assessment",
            ("gaitpdb",),
            TaskDefinition(
                "rehabilitation",
                TaskKind.ESTIMATION,
                1.0,
                "gait_functional_score",
                "absolute_error",
            ),
            "provider gait annotations",
            "participant-level split",
            "MAE and rank correlation",
            "declared resource model",
            "participant bootstrap",
            ("small cohort", "disease-specific generalization"),
        ),
        ClinicalWorkload(
            "multimodal_monitoring",
            "Multimodal physiological monitoring",
            ("wesad", "mimic_waveform"),
            TaskDefinition(
                "multimodal_monitoring", TaskKind.DECISION, 60.0, "downstream_utility", "utility"
            ),
            "task-specific labels",
            "missing-modality and temporal evaluation",
            "task utility and alarm burden",
            "resource and compute constrained",
            "channel, resource, and model uncertainty",
            ("missing-not-at-random", "modality confounding"),
        ),
    )

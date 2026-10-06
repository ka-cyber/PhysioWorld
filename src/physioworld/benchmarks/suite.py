"""Frozen benchmark-family declarations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .corpus import run_corpus_campaign
from .smoke import run_smoke_benchmark


@dataclass(frozen=True)
class BenchmarkDefinition:
    benchmark_id: str
    problem: str
    dataset: str
    configuration: str
    baselines: tuple[str, ...]
    metric: tuple[str, ...]
    expected_tolerance: str
    runtime_class: str


def default_benchmarks() -> tuple[BenchmarkDefinition, ...]:
    return tuple(
        BenchmarkDefinition(*values)
        for values in (
            (
                "representation",
                "history/representation task-law preservation",
                "golden_synthetic",
                "configs/benchmark_representation.json",
                ("raw", "finite_history", "random_projection"),
                ("D_tau", "D_T"),
                "analytical fixture exact",
                "small",
            ),
            (
                "task-law",
                "conditional-law discrepancy and calibration",
                "golden_synthetic",
                "configs/benchmark_tasklaw.json",
                ("empirical_mean",),
                ("D_tau", "TV", "ECE"),
                "analytical fixture exact",
                "small",
            ),
            (
                "communication",
                "erasure/recovery/task-law preservation",
                "golden_synthetic",
                "configs/benchmark_communication.json",
                ("uncoded", "RLNC"),
                ("packet_delivery", "symbol_recovery", "distortion"),
                "round-trip exact",
                "small",
            ),
            (
                "resource",
                "P0/P1/P2 safety and dynamics",
                "golden_synthetic",
                "configs/benchmark_resource.json",
                ("P0", "P1", "P2"),
                ("terminal_voltage", "energy", "viability"),
                "independent integrator tolerance",
                "small",
            ),
            (
                "policy",
                "adaptive action value under constraints",
                "golden_synthetic",
                "configs/benchmark_policy.json",
                ("fixed", "greedy", "DP_oracle"),
                ("value", "regret", "violations"),
                "exact DP on finite MDP",
                "small",
            ),
            (
                "clinical",
                "workload readiness and endpoint schema",
                "mitdb",
                "configs/arrhythmia_adaptive.json",
                ("fixed", "quality_aware"),
                ("sensitivity", "PPV", "false_alarms_per_hour"),
                "protocol-defined",
                "dataset-dependent",
            ),
            (
                "causal",
                "SCM intervention identification",
                "golden_synthetic_scm",
                "configs/benchmark_causal.json",
                ("g_formula", "IPW", "DR"),
                ("ATE", "positivity"),
                "known SCM effect",
                "small",
            ),
            (
                "robustness",
                "shift/channel/resource uncertainty",
                "golden_synthetic",
                "configs/benchmark_robustness.json",
                ("nominal", "robust"),
                ("worst_case_value", "adaptation_value"),
                "scenario-defined",
                "small",
            ),
        )
    )


def run_benchmark_family(
    family: str | None = None,
    *,
    configuration: str | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    if family == "native_level3_synthetic":
        if output is None:
            raise ValueError("native_level3_synthetic requires a fresh --output directory")
        return run_corpus_campaign(output, configuration=configuration)
    definitions = default_benchmarks()
    selected = tuple(item for item in definitions if family is None or item.benchmark_id == family)
    if family and not selected:
        raise KeyError(family)
    result: dict[str, Any] = {"definitions": [asdict(item) for item in selected]}
    if family is None or family == "resource":
        result["smoke"] = run_smoke_benchmark()
    return result

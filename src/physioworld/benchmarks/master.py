"""One-command scientific regression orchestration.

The master suite is an evidence bundle, not a claim that unavailable external
cohorts have been reproduced. Core synthetic/numerical kernels execute locally;
external and historical experiments receive explicit blocked receipts with the
source and acquisition boundary preserved.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from physioworld.corpus import load_experiment_registry, load_registry
from physioworld.experiments import (
    analyze_network_physiology,
    run_causal_vvuq,
    run_communication_vvuq,
    run_nirs_proxy_synthetic,
    run_pir_loss_decomposition,
    run_piv_synthetic,
    run_representation_lattice,
    run_resource_conditioned_capability,
    run_resource_vvuq,
    run_subject_bootstrap_uncertainty,
    run_support_function_experiment,
    run_task_battery,
    run_task_law_discrepancy,
    run_temporal_history_sweep,
)
from physioworld.experiments.manager import _check_output_directory
from physioworld.provenance import prepare_run_directory, write_output
from physioworld.version import __version__

from .corpus import run_corpus_campaign

SUITE_VERSION = "1.0.0"
PROFILES = ("fast", "full", "independent-reference", "release")


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _case_index(corpus_result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {case["case_id"]: case for case in corpus_result["cases"]}


def _result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Retain diagnostic scalars in receipts; full results live in artifacts."""
    keys = (
        "experiment_id",
        "status",
        "evidence_class",
        "data_origin",
        "data_fidelity",
        "estimand",
        "seed",
        "cell_count",
        "incremental_value",
        "replay_equal",
        "preprocessing_gate",
        "quantity",
        "direction",
    )
    return {key: result[key] for key in keys if key in result}


def _receipt(
    spec: Any,
    *,
    status: str,
    result: Mapping[str, Any] | None = None,
    reason: str | None = None,
    result_artifact: str | None = None,
) -> dict[str, Any]:
    value = {
        "experiment": spec.to_dict(),
        "status": status,
        "evidence_class": spec.evidence_class,
        "estimand": spec.estimand,
        "seed": spec.seed,
        "falsification_conditions": list(spec.falsification_conditions),
    }
    if result is not None:
        value["result"] = _result_summary(result)
        value["result_hash"] = _hash(result)
        if result_artifact is not None:
            value["result_artifact"] = result_artifact
    if reason is not None:
        value["reason"] = reason
    return value


def _suite_result(profile: str, corpus_result: Mapping[str, Any]) -> dict[str, Any]:
    """Execute independent kernel checks and build registry-linked receipts."""
    if profile not in PROFILES:
        raise ValueError(f"unsupported scientific suite profile: {profile}")
    case_index = _case_index(corpus_result)
    corpus_summary = {
        key: corpus_result[key]
        for key in (
            "status",
            "corpus_id",
            "actual_fidelity",
            "case_count",
            "failed_cases",
            "configuration_hash",
            "manifest_hash",
        )
        if key in corpus_result
    }
    fast = profile == "fast"
    reference = profile in {"independent-reference", "release"}
    causal = run_causal_vvuq(counts=(32, 64, 128) if fast else (64, 256, 1024), seed=7)
    communication = run_communication_vvuq(seed=17, trials=16 if fast else 64)
    resource = run_resource_vvuq(steps=(8, 16, 32) if fast else (16, 32, 64))
    temporal = run_temporal_history_sweep(seed=101)
    battery = run_task_battery(seed=102)
    support = run_support_function_experiment(seed=113)
    piv = run_piv_synthetic(seed=116)
    pir = run_pir_loss_decomposition(seed=117)
    nirs = run_nirs_proxy_synthetic(seed=109)
    lattice = run_representation_lattice(seed=103)
    resource_capability = run_resource_conditioned_capability(seed=112)
    task_law = run_task_law_discrepancy(seed=104)
    bootstrap = run_subject_bootstrap_uncertainty(seed=105)
    network = analyze_network_physiology(
        {
            "flow": tuple(float(index) for index in range(16)),
            "pressure": tuple(float((index * index + 3 * index) % 11) for index in range(16)),
            "autonomic": tuple(float((index % 4) - (index // 4)) for index in range(16)),
        },
        bins=4,
        n_surrogates=3,
        seed=110,
        alpha=0.05,
        pearson_threshold=0.2,
        mi_threshold=0.05,
    )
    kernel_results = {
        "causal_identification_synthetic": causal,
        "communication_negative_control": communication,
        "ra_upit_physics": resource,
        "temporal_history_sweep": temporal,
        "task_law_battery": battery,
        "support_function_correction": support,
        "piv_counterfactual_synthetic": piv,
        "pir_loss_decomposition": pir,
        "nirs_raw_proxy": nirs,
        "representation_lattice": lattice,
        "resource_conditioned_capability": resource_capability,
        "task_law_discrepancy": task_law,
        "subject_bootstrap_uncertainty": bootstrap,
        "network_physiology": asdict(network),
    }
    case_mapping = {
        "temporal_history_sweep": "representation_temporal",
        "task_law_battery": "tasklaw_transition",
        "task_law_discrepancy": "tasklaw_transition",
        "representation_lattice": "representation_temporal",
        "resource_conditioned_capability": "policy_capability",
        "support_function_correction": "policy_capability",
        "pir_loss_decomposition": "policy_capability",
    }
    receipts = []
    for spec in load_experiment_registry():
        if spec.experiment_id == "master_scientific_suite":
            continue
        if spec.experiment_id in kernel_results:
            result = kernel_results[spec.experiment_id]
            receipts.append(
                _receipt(
                    spec,
                    status="FAIL"
                    if result.get("status") in {"FAIL", "FAILED_REFERENCE"}
                    else "PASS",
                    result=result,
                    result_artifact=f"kernels/{spec.experiment_id}.json",
                )
            )
        elif spec.experiment_id in case_mapping:
            case = case_index[case_mapping[spec.experiment_id]]
            receipts.append(
                _receipt(
                    spec,
                    status="PASS" if case["status"] != "FAILED_REFERENCE" else "FAIL",
                    result=case,
                    result_artifact=f"corpus/cases/{case['case_id']}/result.json",
                )
            )
        elif spec.is_external:
            receipts.append(
                _receipt(
                    spec,
                    status="BLOCKED_EXTERNAL_INPUT",
                    reason="The declared source records, raw data, or historical controller artifacts are not supplied.",
                )
            )
        else:
            receipts.append(
                _receipt(spec, status="REGISTERED_NOT_RUN", reason="No profile runner is bound.")
            )
    blocked = [
        item["experiment"]["id"] for item in receipts if item["status"] == "BLOCKED_EXTERNAL_INPUT"
    ]
    failures = [item["experiment"]["id"] for item in receipts if item["status"] == "FAIL"]
    status = "PASS" if not failures and (profile != "release" or not blocked) else "INCOMPLETE"
    if failures:
        status = "FAIL"
    return {
        "suite_version": SUITE_VERSION,
        "status": status,
        "profile": profile,
        "software_version": __version__,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "independent_reference_mode": reference,
        "corpus": corpus_summary,
        "kernel_checks": {
            "causal": {
                "status": causal.get("status"),
                "evidence_class": causal.get("evidence_class"),
            },
            "communication": {
                "status": communication.get("status"),
                "evidence_class": communication.get("evidence_class"),
            },
            "resource": {
                "status": resource.get("status"),
                "evidence_class": resource.get("evidence_class"),
            },
        },
        "experiment_receipts": receipts,
        "blocked_experiments": blocked,
        "failed_experiments": failures,
        "claim_mapping": {
            "transport_success_is_not_semantic_success": ["communication_negative_control"],
            "predictive_sufficiency_is_not_causal_sufficiency": [
                "causal_identification_synthetic",
                "piv_counterfactual_synthetic",
            ],
            "coordinatewise_optima_need_not_be_jointly_attainable": ["support_function_correction"],
            "p2_requires_coupled_states": ["ra_upit_physics"],
        },
        "uncertainty": {
            "primary_empirical_intervals": "BLOCKED_EXTERNAL_INPUT",
            "synthetic_uncertainty": "reported by each kernel result; not a population confidence claim",
            "cluster_unit": "subject where subject-level data are available",
        },
        "operating_envelopes": {
            "communication": "declared finite seeded packet/channel fixtures",
            "resource": "declared P0/P1 and synthetic coupled Foster circuit inputs",
            "causal": "known-ground-truth synthetic SCM only",
            "physiology": "synthetic vertical examples; no clinical operating envelope",
        },
        "reproducibility": {
            "level": 4 if reference else 3,
            "seed_policy": "experiment registry seed plus explicit kernel seeds",
            "configuration_hash": _hash({"profile": profile, "software_version": __version__}),
        },
        "execution_details": kernel_results,
    }


def run_scientific_suite(
    output: str | Path, *, profile: str = "fast", configuration: str | Path | None = None
) -> dict[str, Any]:
    """Run a profile and write a compact, hashed evidence bundle.

    ``release`` intentionally returns ``INCOMPLETE`` while required external
    inputs remain unavailable. This is a qualification result, not a failure
    suppression mechanism.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    destination = _check_output_directory(output)
    registry = load_registry()
    root = prepare_run_directory(destination)
    corpus_path = root / "corpus"
    corpus = run_corpus_campaign(corpus_path, configuration=configuration)
    result = _suite_result(profile, corpus)
    execution_details = result.pop("execution_details")
    result["source_registry"] = registry
    result["experiment_registry"] = [spec.to_dict() for spec in load_experiment_registry()]
    result["evidence_bundle_hash"] = _hash(result)
    write_output(
        root,
        "suite_result.json",
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    write_output(
        root,
        "experiment_registry.json",
        json.dumps(result["experiment_registry"], indent=2, sort_keys=True) + "\n",
    )
    write_output(
        root, "source_registry.json", json.dumps(registry, indent=2, sort_keys=True) + "\n"
    )
    for experiment_id, details in execution_details.items():
        write_output(
            root,
            f"kernels/{experiment_id}.json",
            json.dumps(details, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    checksums = {
        "suite_result.json": hashlib.sha256((root / "suite_result.json").read_bytes()).hexdigest(),
        "experiment_registry.json": hashlib.sha256(
            (root / "experiment_registry.json").read_bytes()
        ).hexdigest(),
        "source_registry.json": hashlib.sha256(
            (root / "source_registry.json").read_bytes()
        ).hexdigest(),
    }
    write_output(
        root,
        "checksums.sha256",
        "\n".join(f"{value}  {name}" for name, value in checksums.items()) + "\n",
    )
    return result


__all__ = ["PROFILES", "run_scientific_suite"]

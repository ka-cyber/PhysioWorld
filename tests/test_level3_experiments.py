"""Meaningful execution checks for the explicitly synthetic Level 3 runners."""

from __future__ import annotations

import math

from physioworld.experiments import level3


def _direct_auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    return sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    ) / (len(positives) * len(negatives))


def test_temporal_sweep_is_dense_held_out_and_actually_shuffled() -> None:
    result = level3.run_temporal_history_sweep()

    assert result["history_lengths"] == list(range(81))
    assert len(result["rows"]) == 81
    assert result["leakage_control"]["subject_disjoint_train_test"]
    assert not result["leakage_control"]["future_samples_in_features"]
    assert result["leakage_control"]["no_history_uses_last_sample"] is False
    assert result["leakage_control"]["no_history_prediction_is_training_baseline"]
    assert result["rows"][0]["history_sample_count"] == 0
    assert result["rows"][0]["memory_values"] == 0
    assert result["rows"][-1]["history_sample_count"] == 80

    control = result["temporal_shuffle_control"]
    assert control["executed"]
    assert len(control["rows"]) == 81
    assert any(
        abs(row["prediction_rmse"] - shuffled["prediction_rmse"]) > 1e-12
        for row, shuffled in zip(result["rows"][1:], control["rows"][1:])
    )
    assert control["zero_length_is_unchanged"]


def test_task_battery_has_actual_predictions_for_all_declared_cells() -> None:
    result = level3.run_task_battery()

    assert result["task_cell_count"] == 12
    assert result["cell_count"] == 5 * 12
    assert set(result["directions"]) == {"BFV-R", "BFV-L", "BP", "Joint"}
    assert set(result["horizons_s"]) == {10, 20, 30}
    assert result["leakage_control"]["predictions_are_executed"]
    assert result["leakage_control"]["future_samples_in_features"] is False

    definitions = result["candidate_definitions"]
    assert set(definitions) == {"Z1S", "ZA15", "ZB15", "ZC15", "ZMS"}
    assert all(
        definition["synthetic_name"].startswith(f"{candidate}-synthetic-")
        for candidate, definition in definitions.items()
    )
    assert all("definition" in definition for definition in definitions.values())

    first = result["cells"][0]
    assert len(first["predictions"]) == len(first["targets"]) == len(first["labels"])
    assert first["prediction_count"] == len(first["predictions"])
    assert math.isclose(first["held_out_auc"], _direct_auc(first["labels"], first["predictions"]))
    assert any(
        abs(cell["held_out_auc"] - cell["permuted_label_auc"]) > 1e-12 for cell in result["cells"]
    )


def test_task_law_uses_subject_bootstrap_and_predeclared_gates() -> None:
    result = level3.run_task_law_discrepancy()

    assert result["primary_epsilon"] == 0.08
    assert result["epsilon_sensitivity_grid"] == [0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]
    assert result["bootstrap_is_not_primary_point_estimate"]
    assert result["cluster_unit"] == "synthetic subject"
    for candidate, candidate_result in result["candidate_results"].items():
        bootstrap = candidate_result["bootstrap"]
        assert math.isfinite(candidate_result["point_estimate"])
        assert bootstrap["point_estimate"] == candidate_result["point_estimate"]
        assert bootstrap["resampling_unit"] == "cluster"
        assert len(bootstrap["replicates"]) == 199
        assert set(candidate_result["epsilon_gates"]) == {
            "0.02",
            "0.04",
            "0.06",
            "0.08",
            "0.1",
            "0.15",
            "0.2",
        }
    assert result["candidate_results"]["Z1S"]["point_estimate"] == 0.0


def test_support_and_resource_experiments_use_independent_finite_images() -> None:
    support = level3.run_support_function_experiment()
    assert support["independent_reference"]["matches_module_support"]
    assert support["independent_reference"]["maximum_module_vs_reference_error"] <= 1e-12
    assert support["coordinatewise_maximum_jointly_attainable"] is False
    assert len(support["diagnostics"]) == 16
    assert all(item["full_policy"] and item["reduced_policy"] for item in support["diagnostics"])

    resource = level3.run_resource_conditioned_capability()
    assert resource["resource_is_not_intrinsic_information"]
    feasible = {
        name: tuple(condition["feasible_policies"])
        for name, condition in resource["resource_conditions"].items()
    }
    assert len(set(feasible.values())) > 1
    assert feasible["tiny"] == ("ZB15",)
    assert resource["intrinsic_estimand"] != resource["resource_conditioned_estimand"]


def test_pir_piv_and_nirs_keep_their_guards_and_semantics() -> None:
    piv = level3.run_piv_synthetic()
    assert piv["guard_checks"]["paired_ids_aligned"]
    assert piv["guard_checks"]["distinct_information_conditions"]
    assert piv["guard_checks"]["unpaired_observational_contrast_used"] is False
    assert piv["uncertainty"]["resampling_unit"] == "cluster"
    assert math.isclose(piv["incremental_value"], 0.125)

    pir = level3.run_pir_loss_decomposition()
    assert pir["loss_identity_holds"]
    assert math.isclose(pir["L_total"], pir["L_info"] + pir["L_policy"])
    assert pir["controller_value_is_not_actual_optimum"]
    assert pir["common_reference_checked"]

    nirs = level3.run_nirs_proxy_synthetic()
    assert nirs["quantity"] == "optical_density"
    assert nirs["hemoglobin_label"].startswith("FORBIDDEN")


def test_level3_extra_runners_are_exported_without_package_init_changes() -> None:
    expected = {
        "run_pir_loss_decomposition",
        "run_representation_lattice",
        "run_resource_conditioned_capability",
        "run_subject_bootstrap_uncertainty",
        "run_task_law_discrepancy",
    }
    assert expected.issubset(set(level3.__all__))

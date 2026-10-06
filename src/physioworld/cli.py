"""Command-line interface for deterministic experiments and self-audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from . import qualification
from .benchmarks import (
    PROFILES,
    flagship_status,
    run_benchmark_family,
    run_scientific_suite,
    run_smoke_benchmark,
)
from .benchmarks.corpus import CORPUS_ID, validate_corpus_configuration
from .clinical.protocol import protocol_from_config, validation_readiness
from .datasets import LocalFileAdapter, default_registry
from .experiments import ExperimentCompiler, ExperimentPlan, ExperimentValidationError
from .experiments.compiler import load_schema, schema_names
from .experiments.manager import _check_output_directory
from .io import load_config
from .perturbations import (
    ADCSaturation,
    AdditiveNoise,
    BaselineWander,
    CalibrationFailure,
    Clipping,
    DistributionShift,
    ElectrodeArtifact,
    EMGContamination,
    GainError,
    Missingness,
    MotionArtifact,
    OffsetError,
    Quantization,
    SensorDrift,
)
from .physiology import PhysiologicalWorld, SignalStream, validate_stream
from .plugins import BuiltInVerticalPlugin, PluginRegistry, load_plugin
from .provenance import (
    ManifestVerificationError,
    compare_scientific_outputs,
    configuration_snapshot,
    create_manifest,
    load_manifest,
    prepare_run_directory,
    verify_manifest,
    write_json_output,
    write_report,
)
from .version import __version__
from .verticals import run_vertical, vertical_names


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _write_run(
    plan: ExperimentPlan,
    output: str,
    artifacts: Mapping[str, tuple[str, Any]],
    *,
    scientific_outputs: tuple[str, ...],
    question: str,
    hypothesis: str,
    estimand: str,
    data: str,
    results: Mapping[str, Any],
    limitations: list[str],
    status: str = "COMPLETE",
    datasets: tuple[Mapping[str, Any], ...] = (),
    input_hashes: Mapping[str, str] | None = None,
) -> Path:
    """Persist a compiled execution in a fresh, exclusively owned directory."""
    outputs = {key: name for key, (name, _) in artifacts.items()}
    outputs["compiled_plan"] = "compiled_plan.json"
    # Snapshot validation (including credential checks) precedes any artifact.
    manifest = create_manifest(
        Path(output).name,
        plan.configuration,
        seed=plan.configuration["seed"],
        outputs=outputs,
        input_hashes=input_hashes,
        datasets=datasets,
        status=status,
        scientific_outputs=scientific_outputs,
    )
    target = prepare_run_directory(output)
    for name, value in artifacts.values():
        write_json_output(target, name, value)
    write_json_output(target, "compiled_plan.json", plan.to_dict())
    manifest.write(target)
    write_report(
        target,
        question=question,
        hypothesis=hypothesis,
        estimand=estimand,
        data=data,
        method=json.dumps(plan.configuration, sort_keys=True, allow_nan=False),
        results=results,
        limitations=limitations,
    )
    return target


def _simulate_plan(plan: ExperimentPlan, output: str) -> dict[str, Any]:
    config = plan.configuration
    seed = config["seed"]
    stream = SignalStream.from_samples(
        config["samples"],
        config.get("sample_rate"),
        config["unit"],
        config["modality"],
        timestamps=config.get("timestamps"),
        record_id=config.get("record_id"),
        subject_id=config.get("subject_id"),
    )
    perturbation_classes = {
        "additive_noise": AdditiveNoise,
        "baseline_wander": BaselineWander,
        "clipping": Clipping,
        "missingness": Missingness,
        "quantization": Quantization,
        "sensor_drift": SensorDrift,
        "motion_artifact": MotionArtifact,
        "electrode_artifact": ElectrodeArtifact,
        "emg_contamination": EMGContamination,
        "adc_saturation": ADCSaturation,
        "distribution_shift": DistributionShift,
        "gain_error": GainError,
        "offset_error": OffsetError,
        "calibration_failure": CalibrationFailure,
    }
    transformed = stream
    applied = []
    for declaration in config.get("perturbations", []):
        declaration = dict(declaration)
        kind = declaration.pop("type")
        if kind not in perturbation_classes:
            raise ValueError(f"unsupported perturbation: {kind}")
        declaration.setdefault("seed", seed)
        try:
            transformed = perturbation_classes[kind](**declaration).apply(transformed)
        except (TypeError, ValueError) as exc:
            raise ExperimentValidationError(f"perturbation {kind!r}: {exc}") from exc
        applied.append(kind)
    world = PhysiologicalWorld({transformed.metadata.modality: transformed})
    summary = {
        "configuration_hash": plan.configuration_hash,
        "version": __version__,
        "seed": seed,
        "stream": transformed.summary(),
        "applied_perturbations": applied,
        "timeline_length": len(world.timeline()),
        "data_quality": validate_stream(transformed).to_dict(),
        "metrics": {},
    }
    metrics = {
        "mean": sum(row[0] for row in transformed.values) / transformed.n_samples,
        "rms": (sum(row[0] ** 2 for row in transformed.values) / transformed.n_samples) ** 0.5,
    }
    summary["metrics"] = {name: metrics[name] for name in config["metrics"]}
    # Strict JSON failure must also precede directory creation (e.g. missing samples).
    json.dumps(summary, allow_nan=False)
    _write_run(
        plan,
        output,
        {"summary": ("summary.json", summary)},
        scientific_outputs=("summary",),
        question="What does the declared deterministic physiological replay produce?",
        hypothesis="The configured stream and perturbations are reproducible under the declared seed.",
        estimand="Configured stream summary and downstream-independent smoke metrics.",
        data="Declared inline synthetic samples; time in seconds and sample rate in Hz.",
        results=summary,
        limitations=[
            "This synthetic replay is not clinical evidence.",
            "No unavailable dataset is downloaded.",
        ],
    )
    return summary


def simulate(config_path: str, output: str) -> dict[str, Any]:
    plan = ExperimentCompiler().compile_file(config_path)
    if plan.configuration_kind == "dataset_readiness" or plan.data_mode != "synthetic":
        raise ExperimentValidationError(
            "simulate requires synthetic samples or a SYNTHETIC vertical; use run for dataset readiness or replay for REPLAY inputs"
        )
    return _execute_plan(plan, output)


def replay(config_path: str, output: str) -> dict[str, Any]:
    plan = ExperimentCompiler().compile_file(config_path)
    if plan.configuration_kind != "synthetic_samples" and plan.data_mode != "replay":
        raise ExperimentValidationError(
            "replay requires a REPLAY vertical or the legacy inline sample declaration"
        )
    return _execute_plan(plan, output)


def run_vertical_experiment(config: Mapping[str, Any], output: str) -> dict[str, Any]:
    plan = ExperimentCompiler().compile(config)
    if plan.configuration_kind != "vertical":
        raise ExperimentValidationError(
            "run_vertical_experiment requires a complete vertical declaration"
        )
    return _execute_plan(plan, output)


def _run_vertical_plan(plan: ExperimentPlan, output: str) -> dict[str, Any]:
    result = run_vertical(plan.configuration).to_dict()
    _write_run(
        plan,
        output,
        {"result": ("result.json", result)},
        scientific_outputs=("result",),
        question="How do the declared information, transport, resource and policy conditions affect the task?",
        hypothesis="The complete declared vertical executes deterministically under its seed.",
        estimand=result["estimand"],
        data=json.dumps(plan.configuration["dataset"], sort_keys=True, allow_nan=False),
        results=result,
        limitations=list(result["provenance"]["limitations"]),
    )
    return result


def run_declared_experiment(config_path: str, output: str) -> dict[str, Any]:
    """Run a synthetic config or record a data-dependent configuration safely."""
    return _execute_plan(ExperimentCompiler().compile_file(config_path), output)


def _execute_plan(plan: ExperimentPlan, output: str) -> dict[str, Any]:
    _check_output_directory(output)
    if plan.configuration_kind == "synthetic_samples":
        return _simulate_plan(plan, output)
    if plan.configuration_kind == "vertical":
        return _run_vertical_plan(plan, output)
    config = plan.configuration
    result = {
        "status": "CONFIGURATION_ONLY",
        "message": "No local samples supplied; acquire the declared dataset through its adapter before execution.",
        "dataset": config.get("dataset"),
        "task": config.get("task"),
    }
    _write_run(
        plan,
        output,
        {"metrics": ("metrics.json", result)},
        scientific_outputs=(),
        status="PARTIAL",
        datasets=({"dataset_id": config["dataset"], "status": "not_acquired"},),
        question="Can the declared workload be executed with available inputs?",
        hypothesis="The configuration is valid but requires its declared dataset.",
        estimand="Configuration readiness, not clinical performance.",
        data=str(config["dataset"]),
        results=result,
        limitations=["No dataset was acquired.", "No clinical metric was produced."],
    )
    return result


def validate_config(config_path: str, *, family: str | None = None) -> dict[str, Any]:
    if family is not None:
        if family != CORPUS_ID:
            raise ExperimentValidationError(f"unsupported validation family: {family!r}")
        configuration = validate_corpus_configuration(load_config(config_path))
        return {
            "status": "PASS",
            "family": family,
            "schema_name": "corpus.schema.json",
            "configuration": configuration,
            "execution_status": "NOT_EXECUTED",
        }
    plan = ExperimentCompiler().compile_file(config_path)
    return {"status": "PASS", **plan.to_dict()}


def clinical_validate(config_path: str, output: str) -> dict[str, Any]:
    plan = ExperimentCompiler().compile_file(config_path)
    _check_output_directory(output)
    readiness = validation_readiness(plan.data_mode)
    protocol = protocol_from_config(plan.configuration)
    result = {
        **readiness,
        "experiment_id": plan.experiment_id,
        "configuration_hash": plan.configuration_hash,
        "clinical_protocol_hash": protocol.frozen_hash,
        "performance": None,
        "claim_policy": "research validation readiness only; no medical-device or clinical-validity claim",
    }
    artifacts = {
        "validation": ("validation_result.json", result),
        "protocol": ("clinical_protocol.json", protocol.to_dict()),
    }
    input_hashes = {
        "configuration": plan.configuration_hash,
        "clinical_protocol": protocol.frozen_hash,
    }
    _write_run(
        plan,
        output,
        artifacts,
        scientific_outputs=(),
        status="PARTIAL",
        input_hashes=input_hashes,
        datasets=(
            {"dataset_id": plan.configuration.get("dataset"), "status": readiness["data_status"]},
        ),
        question="What simulation, replay, or external predictive evaluation readiness is declared?",
        hypothesis="Clinical protocol fields must be explicitly frozen before external predictive evaluation.",
        estimand="Execution/protocol readiness, not clinical performance or clinical validation.",
        data=str(plan.configuration.get("dataset")),
        results=result,
        limitations=[
            "No post-hoc test-set tuning is permitted.",
            "No predictive evaluation was performed.",
            "Readiness does not establish clinical validation, clinical benefit, or safety.",
        ],
    )
    return result


def doctor(*, check_tests: bool = False) -> dict[str, Any]:
    result = qualification.doctor()
    if check_tests:
        result = _add_checkout_test_result(result)
    return result


def _add_checkout_test_result(result: dict[str, Any]) -> dict[str, Any]:
    """Run source-checkout tests only when the caller explicitly requests them."""
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    checks = list(result["checks"])
    checks.append(
        {
            "check": "checkout_tests",
            "location": "tests",
            "status": "PASS" if completed.returncode == 0 else "ERROR",
            "message": "source-checkout test suite passed"
            if completed.returncode == 0
            else completed.stderr[-2000:],
        }
    )
    result = {**result, "checks": checks}
    if completed.returncode != 0:
        result["status"] = "ERROR"
    result["source_tests"] = {
        "status": "PASS" if completed.returncode == 0 else "ERROR",
        "reason": "explicit checkout test request",
    }
    return result


def qualify_install(*, check_tests: bool = False) -> dict[str, Any]:
    """Delegate installed-package qualification; checkout tests are opt-in."""
    result = qualification.qualify_install()
    if check_tests:
        result = _add_checkout_test_result(result)
    return result


def qualify_plugin_command(reference: str) -> dict[str, Any]:
    if reference == "builtin:verticals":
        plugin = BuiltInVerticalPlugin()
    else:
        plugin = load_plugin(reference)
    registry = PluginRegistry()
    registry.register(plugin)
    result = registry.qualify(plugin.metadata.plugin_id, tests=(lambda value: value.metadata,))
    return result.to_dict()


def reproduce_manifest(manifest_path: str, output: str) -> dict[str, Any]:
    original = verify_manifest(manifest_path)
    configuration = configuration_snapshot(original.manifest)
    plan = ExperimentCompiler().compile(configuration, experiment_id=original.manifest.run_id)
    if plan.configuration_kind == "dataset_readiness":
        raise ExperimentValidationError(
            "reproduce cannot execute a dataset-backed manifest without its declared dataset adapter and records"
        )
    result = _execute_plan(plan, output)
    reproduced = verify_manifest(output)
    comparison = compare_scientific_outputs(original, reproduced)
    report_result = {
        "status": "REPRODUCED",
        "original_manifest": original.to_dict(),
        "reproduced_manifest": reproduced.to_dict(),
        "scientific_output_comparison": comparison.to_dict(),
    }
    write_report(
        output,
        question="Does the verified run reproduce the recorded scientific outputs exactly?",
        hypothesis="The verified configuration and runtime reproduce every selected scientific output byte-for-byte.",
        estimand="Exact SHA-256 equality of the manifest-selected scientific output artifacts.",
        data=f"Original manifest: {manifest_path}",
        method="Verify original manifest, execute its canonical configuration, verify reproduced manifest, compare selected output hashes.",
        results=report_result,
        limitations=["Exact artifact equality does not establish scientific or clinical validity."],
    )
    return {"status": "REPRODUCED", "result": result, "comparison": comparison.to_dict()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physioworld", description="Auditable physiological systems computing"
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor")
    doctor_parser.add_argument(
        "--checkout-tests",
        action="store_true",
        help="explicitly run the source-checkout unittest suite after installed-package diagnostics",
    )
    datasets = commands.add_parser("datasets")
    datasets.add_argument("action", choices=("list", "validate"))
    datasets.add_argument("--dataset", default=None)
    datasets.add_argument("--path", default=None)
    for name in ("simulate", "replay", "run"):
        command = commands.add_parser(name)
        command.add_argument("config")
        command.add_argument("--output", default=f".physioworld/{name}")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--family", default=None)
    benchmark.add_argument("--config", default=None)
    benchmark.add_argument("--output", default=None)
    suite = commands.add_parser(
        "scientific-suite",
        help="run the registry-driven Level 3 scientific regression/evidence suite",
    )
    suite.add_argument("--profile", choices=PROFILES, default="fast")
    suite.add_argument("--config", default=None, help="optional native corpus configuration")
    suite.add_argument("--output", default=".physioworld/scientific-suite")
    validate = commands.add_parser("validate")
    validate.add_argument("config")
    validate.add_argument(
        "--family", choices=(CORPUS_ID,), help="validate a benchmark corpus declaration"
    )
    clinical = commands.add_parser("clinical-validate")
    clinical.add_argument("config")
    clinical.add_argument("--output", default=".physioworld/clinical_validation")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--full", type=float, nargs="+")
    evaluate.add_argument("--representation", type=float, nargs="+")
    evaluate.add_argument(
        "--assumption",
        action="append",
        required=True,
        help="one explicit assumption identifier; repeat for each declared assumption",
    )
    reproduce = commands.add_parser("reproduce")
    reproduce.add_argument(
        "analysis", help="a result_manifest.json path or a built-in flagship analysis name"
    )
    reproduce.add_argument("--output", default=".physioworld/reproduced")
    provenance = commands.add_parser("provenance")
    provenance.add_argument("directory")
    report = commands.add_parser("report")
    report.add_argument("directory")
    qualify = commands.add_parser("qualify-install")
    qualify.add_argument(
        "--checkout-tests",
        action="store_true",
        help="explicitly run the source-checkout unittest suite after installed-package qualification",
    )
    plugin = commands.add_parser("qualify-plugin")
    plugin.add_argument("reference", nargs="?", default="builtin:verticals")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("target", choices=("verticals", "package", "manifest"))
    inspect.add_argument("value", nargs="?")
    schema = commands.add_parser("schema")
    schema.add_argument("name", nargs="?", default="all")
    commands.add_parser("version")
    return parser


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        result = doctor(check_tests=args.checkout_tests)
        _json_print(result)
        return 0 if result["status"] != "ERROR" else 1
    if args.command == "datasets":
        registry = default_registry()
        if args.action == "list":
            _json_print([spec.to_dict() for spec in registry.list()])
        else:
            specs = [registry.get(args.dataset)] if args.dataset else list(registry.list())
            _json_print([LocalFileAdapter(spec, args.path).validate() for spec in specs])
        return 0
    if args.command == "simulate":
        _json_print(simulate(args.config, args.output))
        return 0
    if args.command == "replay":
        _json_print(replay(args.config, args.output))
        return 0
    if args.command == "run":
        _json_print(run_declared_experiment(args.config, args.output))
        return 0
    if args.command == "validate":
        _json_print(validate_config(args.config, family=args.family))
        return 0
    if args.command == "clinical-validate":
        _json_print(clinical_validate(args.config, args.output))
        return 0
    if args.command == "benchmark":
        if args.family == CORPUS_ID:
            output = args.output or ".physioworld/native_level3_synthetic"
            _json_print(run_benchmark_family(args.family, configuration=args.config, output=output))
        else:
            if args.config is not None or args.output is not None:
                raise ExperimentValidationError(
                    "benchmark --config and --output require --family native_level3_synthetic; "
                    "legacy benchmark families do not consume these options"
                )
            _json_print(run_benchmark_family(args.family) if args.family else run_smoke_benchmark())
        return 0
    if args.command == "scientific-suite":
        result = run_scientific_suite(
            args.output,
            profile=args.profile,
            configuration=args.config,
        )
        _json_print(
            {
                "status": result["status"],
                "profile": result["profile"],
                "evidence_bundle_hash": result["evidence_bundle_hash"],
                "blocked_experiments": result["blocked_experiments"],
                "failed_experiments": result["failed_experiments"],
                "output": args.output,
            }
        )
        return 0 if result["status"] == "PASS" else 1
    if args.command == "evaluate":
        from .tasklaws import binary_forecasting_certificate

        if args.full is None or args.representation is None:
            raise ExperimentValidationError("evaluate requires --full and --representation")
        _json_print(
            binary_forecasting_certificate(
                args.full, args.representation, assumptions=tuple(args.assumption)
            )
        )
        return 0
    if args.command == "reproduce":
        if Path(args.analysis).is_file():
            _json_print(reproduce_manifest(args.analysis, args.output))
        else:
            matches = [item for item in flagship_status() if item["analysis"] == args.analysis]
            if not matches:
                raise ExperimentValidationError(
                    f"reproduce analysis not found: {args.analysis!r}; provide a result_manifest.json path"
                )
            _json_print(matches[0])
        return 0
    if args.command == "provenance":
        _json_print(load_manifest(args.directory).to_dict())
        return 0
    if args.command == "report":
        path = Path(args.directory) / "report.md"
        if not path.exists():
            raise ExperimentValidationError(f"report not found: {path}")
        print(path.read_text(encoding="utf-8"))
        return 0
    if args.command == "qualify-install":
        result = qualify_install(check_tests=args.checkout_tests)
        _json_print(result)
        return 0 if result["status"] == "PASS" else 1
    if args.command == "qualify-plugin":
        result = qualify_plugin_command(args.reference)
        _json_print(result)
        return 0 if result["passed"] else 1
    if args.command == "inspect":
        if args.target == "verticals":
            _json_print({"verticals": vertical_names()})
        elif args.target == "package":
            _json_print({"version": __version__, "package": "physioworld"})
        else:
            if not args.value:
                raise ExperimentValidationError("inspect manifest requires a path")
            _json_print(load_manifest(args.value).to_dict())
        return 0
    if args.command == "schema":
        names = schema_names() if args.name == "all" else [args.name]
        output = []
        for name in names:
            try:
                output.append({"name": name, "schema": load_schema(name)})
            except ExperimentValidationError as exc:
                raise ExperimentValidationError(f"schema not found: {name}") from exc
        _json_print(output)
        return 0
    if args.command == "version":
        print(__version__)
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with stable failure classes and nonzero exit codes."""
    try:
        return _main(argv)
    except ManifestVerificationError as exc:
        print(f"physioworld: integrity error: {exc}", file=sys.stderr)
        return 3
    except (ExperimentValidationError, ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"physioworld: execution error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

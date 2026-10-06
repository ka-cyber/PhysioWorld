"""Installed-package diagnostics and executable, independent reference checks.

These routines need only the installed package and the standard library. They
do not infer source-checkout, release, clinical, or hardware assurance from an
installation. Optional capabilities are inspected without importing their deps.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import pkgutil
import platform
import sys
from importlib.resources import files
from typing import Any, Callable

from .version import __version__

__all__ = ["doctor", "qualify_install"]

# Distribution names and import names differ for these declared package extras.
_CAPABILITIES = {
    "scientific": (("numpy", "numpy"), ("scipy", "scipy")),
    "plot": (("matplotlib", "matplotlib"),),
    "wfdb": (("wfdb", "wfdb"),),
    "yaml": (("PyYAML", "yaml"),),
    "schema": (("jsonschema", "jsonschema"),),
    "arrow": (("pyarrow", "pyarrow"),),
    "hdf5": (("h5py", "h5py"), ("numpy", "numpy")),
    "nwb": (("pynwb", "pynwb"),),
    "dicom": (("pydicom", "pydicom"),),
    "serial": (("pyserial", "serial"),),
    "fmi": (("fmpy", "fmpy"),),
}
_REQUIRED_SCHEMAS = (
    "experiment.schema.json",
    "world.schema.json",
    "vertical.schema.json",
    "result_manifest.schema.json",
    "protocol.schema.json",
    "corpus.schema.json",
    "corpus_result.schema.json",
)
_REQUIRED_GOLDEN_FIXTURES = (
    "normal_physiology.json",
    "artifact.json",
    "missing_modality.json",
    "packet_loss.json",
    "resource_depletion.json",
    "task_transition.json",
    "causal_intervention.json",
)
_VERTICALS = ("cardiovascular", "neurophysiology", "icu")
_ESTIMAND = "finite_population_empirical_task_law_and_independently_optimized_policy_utility"


def _json_ready(value: Any) -> Any:
    """Keep failed nonfinite observations visible without emitting invalid JSON."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _error(check: str, location: str, exc: Exception) -> dict[str, Any]:
    return {
        "check": check,
        "location": location,
        "status": "ERROR",
        "message": f"{type(exc).__name__}: {exc}",
    }


def _import_checks() -> list[dict[str, Any]]:
    """Discover every installed module, retaining failures in sibling packages."""
    checks: list[dict[str, Any]] = []
    pending = ["physioworld"]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            module = importlib.import_module(name)
            checks.append({"check": "import", "location": name, "status": "PASS"})
        except Exception as exc:
            checks.append(_error("import", name, exc))
            continue
        if hasattr(module, "__path__"):
            try:
                pending.extend(
                    sorted(
                        entry.name for entry in pkgutil.iter_modules(module.__path__, name + ".")
                    )
                )
            except Exception as exc:
                checks.append(_error("module_discovery", name, exc))
    return sorted(checks, key=lambda item: (item["location"], item["check"]))


def _optional_checks() -> list[dict[str, Any]]:
    checks = []
    for group, dependencies in _CAPABILITIES.items():
        packages = []
        for distribution, module in dependencies:
            version = None
            message = None
            try:
                version = importlib.metadata.version(distribution)
                discoverable = importlib.util.find_spec(module) is not None
            except importlib.metadata.PackageNotFoundError:
                discoverable = False
            except Exception as exc:
                discoverable = False
                message = f"{type(exc).__name__}: {exc}"
            record = {
                "distribution": distribution,
                "module": module,
                "version": version,
                "available": discoverable,
            }
            if message:
                record["message"] = message
            packages.append(record)
        missing = [item["distribution"] for item in packages if not item["available"]]
        checks.append(
            {
                "check": "optional_dependency",
                "location": group,
                "status": "WARNING" if missing else "PASS",
                "required": False,
                "packages": packages,
                "missing": missing,
                "message": f"Install physioworld[{group}] to enable this capability."
                if missing
                else "Installed metadata and import discovery available; dependency code was not executed.",
            }
        )
    return checks


def _schema_structure(node: Any, root: dict[str, Any], documents: dict[str, Any]) -> None:
    """Check the bundled structural vocabulary and resolve refs without network I/O.

    This is deliberately a bounded integrity check, not full Draft meta-schema
    validation. The optional schema capability supplies a general validator.
    """
    if type(node) is bool:
        return
    if not isinstance(node, dict):
        raise ValueError("schema nodes must be objects or booleans")
    if "type" in node:
        kinds = node["type"] if isinstance(node["type"], list) else [node["type"]]
        allowed = ("object", "array", "string", "number", "integer", "boolean", "null")
        if not kinds or any(kind not in allowed for kind in kinds):
            raise ValueError(f"invalid schema type: {node['type']!r}")
    if "required" in node:
        required = node["required"]
        if (
            not isinstance(required, list)
            or any(not isinstance(key, str) for key in required)
            or len(set(required)) != len(required)
        ):
            raise ValueError("schema required must be an array of distinct strings")
    if "$ref" in node:
        if not isinstance(node["$ref"], str):
            raise ValueError("schema $ref must be a string")
        name, _, fragment = node["$ref"].partition("#")
        target = documents[name] if name else root
        if fragment and not fragment.startswith("/"):
            raise ValueError("bundled references must use JSON pointers")
        for part in fragment[1:].split("/") if fragment else ():
            target = target[part.replace("~1", "/").replace("~0", "~")]
        if not isinstance(target, (dict, bool)):
            raise ValueError("schema reference does not resolve to a schema")
    for keyword in ("properties", "$defs", "patternProperties", "dependentSchemas"):
        if keyword in node:
            if not isinstance(node[keyword], dict):
                raise ValueError(f"schema {keyword} must be an object")
            for child in node[keyword].values():
                _schema_structure(child, root, documents)
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if keyword in node:
            if not isinstance(node[keyword], list) or not node[keyword]:
                raise ValueError(f"schema {keyword} must be a nonempty array")
            for child in node[keyword]:
                _schema_structure(child, root, documents)
    for keyword in ("items", "additionalProperties", "not", "if", "then", "else", "contains"):
        if keyword in node:
            _schema_structure(node[keyword], root, documents)


def _schema_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    try:
        compiler = importlib.import_module("physioworld.experiments.compiler")
        names = compiler.schema_names()
        missing = sorted(set(_REQUIRED_SCHEMAS) - set(names))
        checks = [
            {
                "check": "bundled_schemas",
                "location": "physioworld.schemas",
                "status": "ERROR" if missing else "PASS",
                "observed": list(names),
                "reference": list(_REQUIRED_SCHEMAS),
                "missing": missing,
            }
        ]
    except Exception as exc:
        return [_error("bundled_schemas", "physioworld.schemas", exc)]
    documents = {}
    for name in names:
        try:
            # load_schema parses the actual installed resource, including in a wheel.
            document = compiler.load_schema(name)
            if not isinstance(document, dict) or document.get("$schema") != (
                "https://json-schema.org/draft/2020-12/schema"
            ):
                raise ValueError("bundled schema must be an object declaring Draft 2020-12")
            json.dumps(document, allow_nan=False)
            documents[name] = document
        except Exception as exc:
            checks.append(_error("schema_json", name, exc))
    for name, document in documents.items():
        try:
            _schema_structure(document, document, documents)
            encoded = json.dumps(document, sort_keys=True, allow_nan=False).encode("utf-8")
            checks.append(
                {
                    "check": "schema_json",
                    "location": name,
                    "status": "PASS",
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "observed": {"draft": document["$schema"], "json_bytes": len(encoded)},
                    "method": "resource JSON parsing, finite JSON, bounded structure and references",
                }
            )
        except Exception as exc:
            checks.append(_error("schema_json", name, exc))
    return checks


def _golden_fixture_checks() -> list[dict[str, Any]]:
    """Read every required installed fixture, retaining missing/corrupt resources."""
    location = "physioworld.corpus.golden"
    try:
        resources = files("physioworld").joinpath("corpus").joinpath("golden")
        names = sorted(
            entry.name
            for entry in resources.iterdir()
            if entry.is_file() and entry.name.endswith(".json")
        )
    except Exception as exc:
        return [_error("bundled_golden_fixtures", location, exc)]
    missing = sorted(set(_REQUIRED_GOLDEN_FIXTURES) - set(names))
    checks: list[dict[str, Any]] = [
        {
            "check": "bundled_golden_fixtures",
            "location": location,
            "status": "ERROR" if missing else "PASS",
            "observed": names,
            "reference": list(_REQUIRED_GOLDEN_FIXTURES),
            "missing": missing,
        }
    ]
    for seed, name in enumerate(_REQUIRED_GOLDEN_FIXTURES, start=1):
        try:
            payload = resources.joinpath(name).read_bytes()
            document = json.loads(payload)
            json.dumps(document, allow_nan=False)
            if not isinstance(document, dict) or document.get("fixture") != name.removesuffix(
                ".json"
            ):
                raise ValueError("golden fixture must be an object identifying its resource name")
            if type(document.get("seed")) is not int or document["seed"] != seed:
                raise ValueError(f"golden fixture must declare reviewed seed {seed}")
            checks.append(
                {
                    "check": "golden_fixture_json",
                    "location": name,
                    "status": "PASS",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "observed": {
                        "fixture": document["fixture"],
                        "seed": seed,
                        "json_bytes": len(payload),
                    },
                    "method": "installed resource JSON parsing, finite JSON, fixture identity and seed",
                }
            )
        except Exception as exc:
            checks.append(_error("golden_fixture_json", name, exc))
    return checks


def _report(checks: list[dict[str, Any]], *, qualification: bool) -> dict[str, Any]:
    status = "ERROR" if any(item["status"] == "ERROR" for item in checks) else "PASS"
    if (
        not qualification
        and status == "PASS"
        and any(item["status"] == "WARNING" for item in checks)
    ):
        status = "WARNING"
    return _json_ready(
        {
            "status": status,
            "version": __version__,
            "checks": checks,
            "evidence_scope": "installed_package_numerical_verification"
            if qualification
            else "installed_package_diagnostics",
            "source_tests": {
                "status": "NOT_RUN",
                "reason": "Source-checkout testing is a separate explicit operation.",
            },
        }
    )


def doctor() -> dict[str, Any]:
    """Inspect interpreter, all installed modules, resources and optional extras.

    Missing optional extras yield WARNING; required import/resource failures or
    an interpreter outside Python 3.10 through 3.14 yield ERROR. No checkout or
    subprocess test discovery is performed.
    """
    compatible = (3, 10) <= sys.version_info[:2] < (3, 15)
    checks = [
        {
            "check": "interpreter",
            "location": "python",
            "status": "PASS" if compatible else "ERROR",
            "observed": platform.python_version(),
            "reference": ">=3.10,<3.15",
        }
    ]
    checks.extend(_import_checks())
    checks.extend(_schema_checks())
    checks.extend(_golden_fixture_checks())
    checks.extend(_optional_checks())
    return _report(checks, qualification=False)


def _comparison(quantity: str, observed: Any, reference: float, tolerance: float) -> dict[str, Any]:
    valid = type(observed) in (int, float) and math.isfinite(observed)
    residual = abs(observed - reference) if valid else None
    return {
        "quantity": quantity,
        "observed": observed,
        "reference": reference,
        "residual": residual,
        "tolerance": tolerance,
        "passed": valid and residual is not None and residual <= tolerance,
    }


def _measured(comparisons: list[dict[str, Any]], **observations: Any) -> dict[str, Any]:
    return {
        "status": "PASS" if all(item["passed"] for item in comparisons) else "ERROR",
        "observed": observations,
        "comparisons": comparisons,
    }


def _capacitor_reference() -> dict[str, Any]:
    from .resources import CapacitorResource, ResourceState, canonical_current_limit

    # Reference derived here from V(t)=V0-I*t/C-I*ESR, not from a package solver.
    current = 0.8 / 58.0
    resource = CapacitorResource(0.2, 8.0, v_min=1.85, v_max=3.3)
    transition = resource.step(ResourceState(2.65), current, 10.0)
    bulk = 2.65 - current * 10.0 / 0.2
    terminal = bulk - current * 8.0
    comparisons = [
        _comparison(
            "canonical_current_a",
            canonical_current_limit(2.65, 1.85, 0.2, 8.0, 10.0),
            current,
            1e-14,
        ),
        _comparison(
            "resource_current_limit_a", resource.constant_current_limit(2.65, 10.0), current, 1e-14
        ),
        _comparison("bulk_voltage_v", transition.next.voltage, bulk, 2e-12),
        _comparison("terminal_voltage_v", transition.terminal_voltage, terminal, 2e-12),
        _comparison("energy_balance_residual_j", transition.energy_balance.residual_j, 0.0, 2e-12),
    ]
    return _measured(
        comparisons,
        current_a=current,
        duration_s=10.0,
        reference_equation="V_terminal = V0 - I*T/C - I*ESR",
    )


def _pca_reference() -> dict[str, Any]:
    from .representations import PCARepresentation

    samples = ((12.0, -2.0), (11.0, -1.0), (8.0, -4.0), (9.0, -5.0))
    model = PCARepresentation(components=2).fit(samples)
    if model.mean is None:
        raise RuntimeError("PCA reference did not expose a fitted mean")
    # Centered rows are ±(2,1), ±(1,2). Covariance has genuine off-diagonal 8/3.
    # Its eigenpairs are (6, (1,1)/sqrt(2)), (2/3, (1,-1)/sqrt(2)).
    direction = model.components_[0]
    reference = (1 / math.sqrt(2), 1 / math.sqrt(2))
    direction_error = min(
        math.dist(direction, reference), math.dist(direction, tuple(-v for v in reference))
    )
    projection = model.transform(samples[0])
    reconstruction_error = max(
        math.dist(row, model.inverse_transform(model.transform(row))) for row in samples
    )
    comparisons = [
        _comparison("mean_0", model.mean[0], 10.0, 1e-12),
        _comparison("mean_1", model.mean[1], -3.0, 1e-12),
        _comparison("leading_eigenvalue", model.eigenvalues_[0], 6.0, 2e-12),
        _comparison("second_eigenvalue", model.eigenvalues_[1], 2 / 3, 2e-12),
        _comparison("sign_invariant_eigenvector_error", direction_error, 0.0, 2e-12),
        _comparison(
            "absolute_leading_projection", abs(projection.values[0]), 3 / math.sqrt(2), 2e-12
        ),
        _comparison("maximum_reconstruction_error", reconstruction_error, 0.0, 2e-12),
    ]
    return _measured(
        comparisons,
        principal_direction=direction,
        reference_direction=reference,
        sample_covariance=((10 / 3, 8 / 3), (8 / 3, 10 / 3)),
    )


def _reference_gf_product(left: int, right: int) -> int:
    # Polynomial multiplication followed by long division, independent of GF256.mul.
    product = 0
    for bit in range(8):
        if right & (1 << bit):
            product ^= left << bit
    for bit in range(14, 7, -1):
        if product & (1 << bit):
            product ^= 0x11D << (bit - 8)
    return product


def _rlnc_reference() -> dict[str, Any]:
    from .communication.rlnc import Codeword, RLNCDecoder, RLNCEncoder

    symbols = (
        bytes((0, 1, 127, 128, 255)),
        bytes((255, 33, 0, 200, 7)),
        bytes((13, 90, 240, 2, 129)),
    )
    first = RLNCEncoder(symbols, seed=19).encode(9)
    second = RLNCEncoder(symbols, seed=19).encode(9)
    mismatches = 0
    for word in first:
        if len(word.coefficients) != 3 or len(word.payload) != 5:
            raise ValueError("RLNC codeword has incorrect dimensions")
        expected = bytearray(5)
        for coefficient, symbol in zip(word.coefficients, symbols):
            for index, value in enumerate(symbol):
                expected[index] ^= _reference_gf_product(coefficient, value)
        mismatches += sum(a != b for a, b in zip(word.payload, expected))
    decoder = RLNCDecoder(3, 5)
    # Discard every systematic word: exercise field elimination, not identity copying.
    for word in first[3:]:
        decoder.add(Codeword.deserialize(word.serialize()))
    decoded = decoder.decode()
    comparisons = [
        _comparison("independent_encoded_byte_mismatches", mismatches, 0.0, 0.0),
        _comparison("repair_only_rank", decoder.rank, 3.0, 0.0),
        _comparison(
            "decoded_payload_mismatches", sum(a != b for a, b in zip(decoded, symbols)), 0.0, 0.0
        ),
        _comparison("decoded_symbol_count", len(decoded), 3.0, 0.0),
        _comparison(
            "seeded_codeword_mismatches", sum(a != b for a, b in zip(first, second)), 0.0, 0.0
        ),
        _comparison("emitted_codeword_count", len(first), 9.0, 0.0),
        _comparison("replayed_codeword_count", len(second), 9.0, 0.0),
    ]
    return _measured(
        comparisons,
        source_payload_hex=[word.hex() for word in symbols],
        decoded_payload_hex=[word.hex() for word in decoded],
        repair_coefficients_hex=[word.coefficients.hex() for word in first[3:]],
        reference_field_polynomial="0x11D",
    )


def _vertical_observations(result: Any, name: str, evidence: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("vertical to_dict() must return an object")
    json.dumps(result, allow_nan=False)
    if result.get("estimand") != _ESTIMAND or result.get("evidence_class") != evidence:
        raise ValueError(
            "vertical must identify the finite empirical task-law method and evidence class"
        )
    outputs = result["outputs"]
    if outputs["vertical_id"] != name or result["seed"] != 19:
        raise ValueError("vertical identity/seed disagrees with the executed declaration")
    if outputs["causal_gate"]["status"] != "UNIDENTIFIED":
        raise ValueError("simulation/replay must retain an UNIDENTIFIED causal gate")
    if outputs["resource"]["energy_status"] != "MODEL_DERIVED":
        raise ValueError("vertical energy must identify its model-derived status")
    evaluation, transport, semantics = (
        outputs["evaluation"],
        outputs["transport"],
        outputs["semantics"],
    )
    total, evaluated, sent = (
        evaluation["total_atoms"],
        evaluation["evaluated_atoms"],
        transport["sent_packets"],
    )
    if not (
        type(total) is int
        and type(evaluated) is int
        and 0 < evaluated <= total
        and type(sent) is int
        and sent > 0
    ):
        raise ValueError("vertical must execute a nonempty finite population and actual transport")
    available, discrepancy = semantics["available_fraction"], semantics["task_law_discrepancy"]
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
        for v in (available, discrepancy)
    ):
        raise ValueError("availability and task-law discrepancy must be finite probabilities")
    return {
        "total_atoms": total,
        "evaluated_atoms": evaluated,
        "sent_packets": sent,
        "available_fraction": available,
        "task_law_discrepancy": discrepancy,
        "estimand": result["estimand"],
        "energy_status": outputs["resource"]["energy_status"],
        "causal_status": outputs["causal_gate"]["status"],
    }


def _vertical_replay(name: str) -> dict[str, Any]:
    from .verticals import run_vertical

    first = run_vertical(name, seed=19).to_dict()
    second = run_vertical(name, seed=19).to_dict()
    observed = _vertical_observations(first, name, "SIMULATION")
    _vertical_observations(second, name, "SIMULATION")
    return _measured(
        [_comparison("paired_result_mismatch", int(first != second), 0.0, 0.0)], **observed
    )


def _replay_configuration(loss_probability: float) -> dict[str, Any]:
    from .verticals import default_configuration

    config = default_configuration("cardiovascular", seed=19)
    config["name"] = "installation_qualification_replay"
    config["dataset"] = {
        "id": "qualification_three_atoms",
        "origin": "REPLAY",
        "source": "inline analytic fixture",
        "modality": "telemetry",
        "channel": "mean_pressure_mmhg",
        "unit": "mmHg",
        "time_unit": "s",
        "samples": [90.0, 85.0, 75.0],
        "timestamps_s": [0.0, 1.0, 2.0],
    }
    config["task"].update(horizon_s=0.0, threshold=80.0)
    config["policy"]["thresholds"] = [80.0]
    config["representation"] = {
        "name": "finite_history",
        "window": 3,
        "window_unit": "samples",
        "pad_value": 0.0,
    }
    config["perturbations"] = []
    config["split"] = {"strategy": "replay_fixed"}
    config["channel"].update(
        loss_probability=loss_probability,
        latency_s=0.0,
        jitter_s=0.0,
        deadline_s=1.0,
        semantic_permutation=False,
        packet_payload_bytes=64,
    )
    config["resource"].update(
        model="capacitor_p0",
        capacitance_f=1.0,
        esr_ohm=0.0,
        leakage_siemens=0.0,
        initial_voltage_v=3.0,
        v_min=1.0,
        max_current_a=1.0,
        load_current_a=0.001,
        step_duration_s=0.001,
    )
    config["compute"].update(
        operations_per_sample=100.0,
        operations_per_second=1e6,
        energy_per_operation_j=0.0,
        network_latency_s=0.0,
    )
    return config


def _analytic_replay(loss_probability: float) -> dict[str, Any]:
    from .verticals import run_vertical

    first = run_vertical(_replay_configuration(loss_probability)).to_dict()
    second = run_vertical(_replay_configuration(loss_probability)).to_dict()
    observed = _vertical_observations(first, "cardiovascular", "NUMERICAL_VERIFICATION")
    _vertical_observations(second, "cardiovascular", "NUMERICAL_VERIFICATION")
    outputs = first["outputs"]
    labels = [row["endpoint"] for row in outputs["rows"]]
    lost = loss_probability == 1.0
    # Atomic full histories give p=(1,1,0). One unavailable group gives q=2/3.
    # E|p-q|=(1/3+1/3+2/3)/3=4/9, independently of packet delivery metadata.
    comparisons = [
        _comparison("paired_result_mismatch", int(first != second), 0.0, 0.0),
        _comparison("total_atoms", observed["total_atoms"], 3.0, 0.0),
        _comparison("evaluated_atoms", observed["evaluated_atoms"], 3.0, 0.0),
        _comparison("endpoint_label_mismatch", int(labels != [1, 1, 0]), 0.0, 0.0),
        _comparison(
            "available_fraction", observed["available_fraction"], 0.0 if lost else 1.0, 0.0
        ),
        _comparison(
            "task_law_discrepancy", observed["task_law_discrepancy"], 4 / 9 if lost else 0.0, 2e-12
        ),
        _comparison(
            "endpoint_task_value",
            outputs["capability"]["endpoint_task_value"],
            0.0 if lost else 1.0,
            2e-12,
        ),
        _comparison(
            "intrinsic_task_law_discrepancy",
            outputs["semantics"]["intrinsic_task_law_discrepancy"],
            0.0,
            2e-12,
        ),
    ]
    if lost:
        comparisons.append(
            _comparison(
                "nonabstention_actions",
                sum(action != "abstain" for action in outputs["capability"]["selected_actions"]),
                0.0,
                0.0,
            )
        )
    return _measured(
        comparisons,
        **observed,
        loss_probability=loss_probability,
        endpoint_labels=labels,
        reference_conditional_probabilities=[2 / 3] * 3 if lost else [1.0, 1.0, 0.0],
    )


def _execute(check: str, location: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return {"check": check, "location": location, **function()}
    except Exception as exc:
        return _error(check, location, exc)


def qualify_install() -> dict[str, Any]:
    """Execute installed kernels against analytic/byte references and replay.

    All required diagnostics and numerical checks must pass. Absent optional
    extras remain explicit warnings but do not fail core qualification. Each
    executed comparison records observation, reference, residual and tolerance.
    """
    checks: list[dict[str, Any]] = list(doctor()["checks"])
    for location, function in (
        ("capacitor", _capacitor_reference),
        ("pca", _pca_reference),
        ("rlnc", _rlnc_reference),
    ):
        checks.append(_execute("numerical_reference", location, function))
    for name in _VERTICALS:

        def vertical_function(name: str = name) -> dict[str, Any]:
            return _vertical_replay(name)

        checks.append(_execute("vertical_replay", name, vertical_function))
    for loss, location in ((0.0, "full_availability"), (1.0, "complete_packet_loss")):

        def replay_function(loss: float = loss) -> dict[str, Any]:
            return _analytic_replay(loss)

        checks.append(_execute("analytic_replay_reference", location, replay_function))
    return _report(checks, qualification=True)

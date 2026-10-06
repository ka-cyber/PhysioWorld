"""Validate shipped schemas, their bundled copies, and declared examples."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_configuration(configuration, documents, *, validator_for=None, registry=None) -> str:
    """Validate a declaration with its execution family's configuration contract."""
    from physioworld.experiments import ExperimentCompiler

    if "corpus_id" in configuration:
        from physioworld.benchmarks.corpus import _validate_document

        schema_name = "corpus.schema.json"
        canonical = _validate_document(configuration, documents[schema_name])
        kind = "corpus"
    else:
        plan = ExperimentCompiler().compile(configuration)
        schema_name = plan.schema_name
        canonical = plan.configuration
        kind = plan.configuration_kind
    if validator_for is not None:
        schema = documents[schema_name]
        validator_for(schema)(schema, registry=registry).validate(canonical)
        if kind != "corpus":
            # The experiment entry schema covers only its own named branches.
            entry_schema = documents["experiment.schema.json"]
            validator_for(entry_schema)(entry_schema, registry=registry).validate(configuration)
    return kind


def main() -> int:
    failures = []
    schema_root = ROOT / "schemas"
    schemas = sorted(schema_root.rglob("*.json"))
    if not schemas:
        raise SystemExit("no schemas found")
    from physioworld.experiments.compiler import load_schema, schema_names
    from physioworld.io import load_config

    try:
        from jsonschema.exceptions import SchemaError, ValidationError
        from jsonschema.validators import validator_for
        from referencing import Registry, Resource
    except ImportError:
        validator_for = None
        ValidationError = ValueError
        SchemaError = ValueError
        print("Optional jsonschema unavailable; checking bundled parity and runtime validation")

    documents = {}
    for path in schemas:
        name = path.relative_to(schema_root).as_posix()
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if validator_for is not None:
                validator_for(value).check_schema(value)
            bundled = ROOT / "src" / "physioworld" / "schemas" / name
            if not bundled.is_file() or bundled.read_bytes() != path.read_bytes():
                failures.append(f"{name}: bundled schema missing or differs")
            if load_schema(name) != value:
                failures.append(f"{name}: installed schema resource differs")
            documents[name] = value
        except (OSError, ValueError, SchemaError, ValidationError) as exc:
            failures.append(f"{name}: {exc}")
    if set(schema_names()) != {path.relative_to(schema_root).as_posix() for path in schemas}:
        failures.append("source and installed schema names differ")
    branches = documents.get("experiment.schema.json", {}).get("$defs", {})
    for name in ("dataset_readiness", "synthetic_samples", "vertical"):
        if name not in branches:
            failures.append(f"experiment.schema.json: missing named branch {name}")

    registry = None
    if validator_for is not None:
        registry = Registry().with_resources(
            (document["$id"], Resource.from_contents(document)) for document in documents.values()
        )

    checked = {"synthetic_samples": 0, "vertical": 0, "dataset_readiness": 0, "corpus": 0}
    for directory in ("examples", "configs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            try:
                configuration = load_config(path)
                kind = check_configuration(
                    configuration, documents, validator_for=validator_for, registry=registry
                )
                checked[kind] += 1
            except (OSError, ValueError, RuntimeError, ValidationError) as exc:
                failures.append(f"{path.relative_to(ROOT).as_posix()}: {exc}")
    for failure in failures:
        print(failure)
    print(f"Checked {len(schemas)} schemas; configuration-only validation by branch: {checked}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

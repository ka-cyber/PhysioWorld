# Versioned scientific contracts

A persistent experiment/result must record the software, task, representation,
resource model and schema versions used to interpret it. Dataset versions,
preprocessing, split, utility, horizon, reference and uncertainty are part of the
scientific contract, not incidental run options.

## Schema validation

JSON Schema source documents have individual links so they are also available
as assets in the offline HTML documentation:

- [Experiment declarations](../schemas/experiment.schema.json)
- [Synthetic sample worlds](../schemas/world.schema.json)
- [Scientific verticals](../schemas/vertical.schema.json)
- [Declarative protocols](../schemas/protocol.schema.json)
- [Run manifests](../schemas/result_manifest.schema.json)
- [Corpus configuration](../schemas/corpus.schema.json)
- [Corpus results and manifests](../schemas/corpus_result.schema.json)

Their runtime copies are bundled under `physioworld/schemas`. The schema's
declared draft determines validation behavior. Resource files must match the
reviewed source schemas.

```bash
uv run --no-sync python tools/check_schemas.py
physioworld schema --help
physioworld validate examples/minimal_world.json
```

Schema validity establishes a data contract, not scientific correctness.
Compilation also checks semantic requirements such as units, time, task and
data mode. Required scientific fields must not be filled with hidden defaults.

## Compatibility

Public API changes require a documented compatibility impact. Prefer deprecation
warnings before removals. A schema/model version change must describe affected
claims, tests, fixtures and whether previous results remain comparable.
Round trips preserve semantic equality; migrations are explicit and recorded in
provenance. Never reinterpret an old schema as a newer scientific meaning merely
because its JSON parses.

Serialization must reject unsafe object construction, nonfinite values where
invalid, path traversal and unsupported versions. Study-specific thresholds,
fallback policy, causal assumptions, imputation and resampling are explicit.

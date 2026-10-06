# Reproducibility qualification

An experiment manifest records the configuration, seed, software/environment,
dataset identity, declared transformations and input/output hashes. Compare
deterministic scientific outputs independently of timestamps and runtime metadata.

```bash
uv sync --frozen --all-extras
uv run --no-sync physioworld run examples/minimal_world.json --output .physioworld/reference
uv run --no-sync physioworld reproduce .physioworld/reference/result_manifest.json --output .physioworld/replayed
uv run --no-sync physioworld provenance .physioworld/replayed
```

## Qualification levels

| Level | Evidence required |
|---|---|
| 0 — script | Same machine/environment executes the declared script |
| 1 — environment | Exact dependency lock and runtime identity are preserved |
| 2 — artifact | Frozen configuration and legally usable dataset/reference hashes |
| 3 — replay | Seed/execution semantics reproduce deterministic scientific outputs |
| 4 — independent | Independent implementation/reference agrees within declared tolerances |
| 5 — cross-platform | Supported OS/runtime matrix meets scientific comparison criteria |

Record the highest evidenced level for the specific run, not the project as a
whole. A lock alone does not establish levels 2–5. Seed equality alone does not
prove cross-platform floating-point equality.

## Data and benchmark qualification

Provider permissions and versions apply to every dataset. The
[registry](../dataset-registry.md) is metadata, not raw data. Credentialed
PhysioNet/MIMIC resources require user-side authorization; absence of records is
not a zero-valued result and must not activate synthetic fallback.

Benchmarks must record hardware, runtime/dependency versions, task, horizon,
policy class, split, solver tolerances, resource/channel regime, seeds and fixture
hashes. Timing changes are performance observations, not physiological effect sizes.

## Release binding

The [release manifest generator](../../tools/release_manifest.py) binds the
observed commit/dirty state, runtime, installed dependencies, lock, source,
schemas, benchmark definitions and dataset manifest inventory. Evidence files
are hashed. No successful acceptance result or data acquisition is inferred.
See [release qualification](release-qualification.md).

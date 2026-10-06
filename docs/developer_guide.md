# Developer guide

Install uv 0.12.23, then `uv sync --frozen --all-extras`. Core types remain
dependency-light; optional imports belong at the selected capability boundary.
Scientific records should be immutable/serializable where practical.

## Extending a component

1. State the mathematical object, input/output units, clock and estimand.
2. Declare assumptions, parameter origin, evidence class and operating envelope.
3. Define a small analytic reference and an independent comparison.
4. Implement against the existing typed API/adapter boundary.
5. Include provenance and explicit missing-dependency/failure semantics.
6. Update the public reference, schema/version contract and affected evidence.

Plugin installation is separate from qualification. Python plugins run in the
user's process; use external isolation for untrusted code. Native libraries,
devices and external models need additional runtime/version/calibration evidence.

## Local quality checks

```bash
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check src tests tools scripts
uv run --no-sync ruff format --check src tests tools scripts
uv run --no-sync mypy src/physioworld tools
uv run --no-sync python tools/check_schemas.py
python tools/docs.py check
uv run --no-sync bandit -r src tools scripts -ll
uv run --no-sync python tools/check_security.py
uv run --no-sync python tools/check_licenses.py
```

Source review, typing and regression scope cover the package, not just a selected
green subset. Do not suppress checks to certify an unqualified release.
See [contracts](contracts.md), [Verification Manual](quality/verification-manual.md)
and [contribution policy](../CONTRIBUTING.md).

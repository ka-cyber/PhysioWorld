# Contributing to PhysioWorld

Contributions should preserve the separation between scientific estimands,
data adapters, and execution infrastructure. Before implementing a major
module:

1. write the mathematical specification and assumptions;
2. add an analytical example and invariants;
3. implement unit and numerical tests;
4. record provenance and limitations;
5. run `physioworld doctor` and the test suite.

Do not add credentials, patient identifiers, restricted raw datasets, or
unverified clinical claims. New dataset adapters must document license,
access class, citation, source URL, version, subject unit, preprocessing, and
known limitations.

Use typed public APIs, small composable objects, deterministic seeds, and
backward-compatible changes where possible.

Use `uv sync --frozen --all-extras` for the reviewed development environment.
Run the commands in the [developer guide](docs/developer_guide.md), including
schema, docs, typing, lint/format and security checks. Install pre-commit with
`uv run --no-sync pre-commit install`; hook revisions are immutable references.

Scientific method changes identify affected claims, assumptions, tests,
benchmark references and result compatibility. Public API removals need a
deprecation period and documented migration; persistent scientific contracts
retain schema/model versions. Do not preserve stale numerical results after a
kernel or estimand changes.

Keep source control limited to product code, maintained public documentation and
small curated reproducibility fixtures. Runtime outputs, raw logs, personal
notes and internal task lists belong outside the public source tree. Report
vulnerabilities through [SECURITY.md](SECURITY.md).

"""Validate generated CycloneDX 1.5 JSON against its strict schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonStrictValidator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sbom", type=Path)
    args = parser.parse_args()
    error = JsonStrictValidator(SchemaVersion.V1_5).validate_str(
        args.sbom.read_text(encoding="utf-8")
    )
    if error is not None:
        print(error)
        return 1
    print("CycloneDX 1.5 strict schema validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

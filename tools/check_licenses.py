"""Audit installed distribution license evidence without inferring legal approval."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = []
    missing = []
    for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name = dist.metadata["Name"] or "unnamed"
        expression = (
            dist.metadata["License-Expression"] if "License-Expression" in dist.metadata else None
        )
        classifiers = [
            item
            for item in dist.metadata.get_all("Classifier", [])
            if item.startswith("License ::")
        ]
        legacy = (dist.metadata["License"] if "License" in dist.metadata else "").strip()
        license_files = [
            str(path)
            for path in dist.files or []
            if any(word in str(path).upper() for word in ("LICENSE", "COPYING", "NOTICE"))
        ]
        evidence = bool(expression or classifiers or legacy not in {"", "UNKNOWN"} or license_files)
        row = {
            "name": name,
            "version": dist.version,
            "expression": expression,
            "classifiers": classifiers,
            "license_files": license_files,
            "legacy_license_present": legacy not in {"", "UNKNOWN"},
            "evidence_status": "PRESENT" if evidence else "MISSING",
        }
        rows.append(row)
        if not evidence:
            missing.append(name)
    result = {
        "audit_scope": "installed distribution metadata and license/notice file inventory",
        "legal_compatibility": "requires release-specific human review",
        "packages": rows,
        "missing": missing,
    }
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
    for name in missing:
        print(f"Missing license evidence: {name}")
    print(f"License metadata checked: {len(rows)} distributions, {len(missing)} missing")
    return bool(missing)


if __name__ == "__main__":
    raise SystemExit(main())

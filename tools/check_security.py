"""Fail on detected credential literals; never print the matched secrets."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    result = subprocess.run(
        [
            "detect-secrets",
            "scan",
            "--all-files",
            "--no-verify",
            "--exclude-files",
            r"(^|/)(\.git|\.venv|\.physioworld|outputs|results|run-artifacts|build|dist|site|__pycache__|\.mypy_cache|\.ruff_cache|\.pytest_cache|\.hypothesis|\.cache)/|docs/corpus_inventory\.json$|uv\.lock$|\.egg-info/|requirements/.*\.txt$",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        print("Secret scanner failed to execute", file=sys.stderr)
        return result.returncode
    findings = json.loads(result.stdout).get("results", {})
    for filename, entries in findings.items():
        for entry in entries:
            print(f"{filename}:{entry['line_number']}: {entry['type']} requires review")
    print(f"Secret scan: {sum(map(len, findings.values()))} findings")
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())

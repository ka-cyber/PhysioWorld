"""Check committed hash-pinned pip exports against frozen uv resolution."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "dev": ("dev",),
    "scientific": (
        "scientific",
        "wfdb",
        "yaml",
        "plot",
        "schema",
        "arrow",
        "hdf5",
        "nwb",
        "dicom",
        "serial",
        "fmi",
    ),
}


def normalized(text: str) -> str:
    return "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def main() -> int:
    failures = []
    for profile, extras in PROFILES.items():
        command = ["uv", "export", "--frozen", "--no-default-groups", "--no-emit-project"]
        for extra in extras:
            command.extend(("--extra", extra))
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode:
            print(result.stderr)
            return result.returncode
        path = ROOT / "requirements" / f"requirements-{profile}.txt"
        if normalized(result.stdout) != normalized(path.read_text(encoding="utf-8")):
            failures.append(path.name)
    for name in failures:
        print(f"Frozen lock export differs: {name}")
    if not failures:
        print("Frozen uv exports match committed versions, markers and hashes")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())

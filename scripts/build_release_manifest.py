"""Compatibility entry point for tools/release_manifest.py."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tools" / "release_manifest.py"),
        run_name="__main__",
    )

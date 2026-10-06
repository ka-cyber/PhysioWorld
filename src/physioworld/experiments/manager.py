"""Configuration-first experiment manager.

The manager delegates scientific computation to injected callables and owns
only configuration, seed, and output provenance. This prevents notebook code
from becoming an untracked execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Mapping

from physioworld.provenance import create_manifest, prepare_run_directory, write_json_output
from physioworld.provenance.manifest import _checked_path

from .compiler import ExperimentCompiler, ExperimentValidationError


def _check_output_directory(path: str | Path) -> Path:
    """Check run ownership before computation, without creating any directories."""
    target = _checked_path(path)
    for parent in target.parents:
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError(f"run output parent is not a directory: {parent}")
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"run output is not a directory: {target}")
        if any(target.iterdir()):
            raise FileExistsError(f"run output directory must be empty: {target}")
    return target


def _manager_run_id(name: Any) -> str:
    if (
        not isinstance(name, str)
        or not name.strip()
        or name in (".", "..")
        or any(character in name for character in ("/", "\\", "\x00"))
        or PureWindowsPath(name).drive
    ):
        raise ExperimentValidationError("manager name must be a non-empty, safe path component")
    return name


def _configuration_filename_stem(config_path: str | Path) -> str:
    """Extract a filename stem without letting the host parse a foreign drive."""
    raw = str(config_path).replace("\\", "/")
    filename = raw.rsplit("/", 1)[-1]
    return filename.rsplit(".", 1)[0] if "." in filename else filename


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    output_directory: Path
    result: Mapping[str, Any]


class ExperimentManager:
    def __init__(self, output_root: str | Path = "results"):
        self.output_root = Path(output_root)

    def run(
        self, config_path: str | Path, function: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ) -> ExperimentRun:
        plan = ExperimentCompiler().compile_file(config_path)
        config = plan.configuration
        run_id = _manager_run_id(config.get("name", _configuration_filename_stem(config_path)))
        output = _check_output_directory(self.output_root / run_id)
        manifest = create_manifest(
            run_id,
            config,
            seed=config.get("seed"),
            outputs={"metrics": "metrics.json", "compiled_plan": "compiled_plan.json"},
            input_hashes={"configuration": plan.configuration_hash},
        )
        result = dict(function(config))
        output = prepare_run_directory(output)
        write_json_output(output, "metrics.json", result)
        write_json_output(output, "compiled_plan.json", plan.to_dict())
        manifest.write(output)
        return ExperimentRun(run_id, output, result)

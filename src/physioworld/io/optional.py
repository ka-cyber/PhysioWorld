"""Lazy optional dependencies with actionable, format-specific errors."""

from importlib import import_module


class MissingOptionalDependency(ImportError):
    def __init__(self, format_name: str, package: str, extra: str | None = None):
        install = f"pip install 'physioworld[{extra}]'" if extra else f"pip install {package}"
        super().__init__(
            f"{format_name} requires optional package '{package}'; install with `{install}`"
        )
        self.format_name, self.package, self.extra = format_name, package, extra


def require(package: str, format_name: str, extra: str | None = None):
    try:
        return import_module(package)
    except ModuleNotFoundError as exc:
        if exc.name == package or package.startswith(str(exc.name) + "."):
            raise MissingOptionalDependency(format_name, package.split(".")[0], extra) from exc
        raise  # A broken installed package is not an absent extra.

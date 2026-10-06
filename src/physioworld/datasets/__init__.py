"""Dataset registry and access-controlled adapter interfaces."""

from .adapters import DatasetAdapter, LocalFileAdapter, LocalWFDBAdapter, local_checksum
from .registry import DatasetAccess, DatasetRegistry, DatasetSpec, default_registry
from .splits import (
    CompiledSplit,
    LeakageIssue,
    PreprocessingArtifact,
    SplitCompiler,
    SplitItem,
    SplitLeakageError,
    SplitSpec,
)

__all__ = [
    "DatasetAccess",
    "DatasetSpec",
    "DatasetRegistry",
    "default_registry",
    "DatasetAdapter",
    "LocalFileAdapter",
    "LocalWFDBAdapter",
    "local_checksum",
    "SplitItem",
    "SplitSpec",
    "SplitCompiler",
    "CompiledSplit",
    "PreprocessingArtifact",
    "SplitLeakageError",
    "LeakageIssue",
]

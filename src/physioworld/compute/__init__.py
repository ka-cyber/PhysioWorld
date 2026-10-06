"""Local/edge/cloud compute budgets and inference placement."""

from .models import ApproximateVariant, ComputeEngine, ComputeNode, ExecutionResult, ModelProfile
from .profiling import CallableProfile, profile_callable

__all__ = [
    "ApproximateVariant",
    "ComputeNode",
    "ModelProfile",
    "ExecutionResult",
    "ComputeEngine",
    "CallableProfile",
    "profile_callable",
]

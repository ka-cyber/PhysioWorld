"""Deterministic analytical and reproduction benchmark entry points."""

from .corpus import run_corpus_campaign
from .flagship import flagship_status
from .master import PROFILES, run_scientific_suite
from .smoke import run_smoke_benchmark
from .suite import BenchmarkDefinition, default_benchmarks, run_benchmark_family

__all__ = [
    "run_smoke_benchmark",
    "flagship_status",
    "BenchmarkDefinition",
    "default_benchmarks",
    "run_benchmark_family",
    "run_corpus_campaign",
    "PROFILES",
    "run_scientific_suite",
]

# Benchmark guide

Run `physioworld benchmark` for the deterministic analytical smoke benchmark.
It validates a GF(256) round trip and the canonical capacitor transition.
Large dataset benchmarks should proceed in order: smoke, analytical and
numerical validation, controlled sweep, then large run. Every sweep belongs
to a configuration and result manifest.

`physioworld benchmark --family resource` selects the declared family. Other
family definitions can be inspected through the installed CLI/API; distinguish
metadata-only definitions from executed metrics in the output.
The [native Level 3 corpus guide](corpus_guide.md) documents the executable
`native_level3_synthetic` family, its configuration at
`configs/corpus/benchmark_corpus_level3.json`, and its hashed synthetic/numerical
case outputs.
See [benchmark reference](quality/benchmark-reference.md) and
[benchmark protocol](../benchmarks/README.md). The release manifest hashes both
benchmark definitions and the source implementation, not mutable timing logs.

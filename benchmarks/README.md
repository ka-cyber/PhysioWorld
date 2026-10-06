# Benchmarks

Small, intentionally frozen benchmark definitions and analytical fixtures
belong here. Mutable timings/results belong in the user's output workspace;
raw or credentialed records do not belong in this directory.

```bash
physioworld benchmark
physioworld benchmark --family resource
```

The analytical smoke exercises GF(256) coding/recovery and a declared capacitor
transition. Family definitions live in
[suite.py](../src/physioworld/benchmarks/suite.py). A family definition is not
evidence that a dataset-dependent benchmark has been executed.

For every comparison record hardware, OS/Python/dependency versions, dataset and
split, seed, task/horizon/estimand, representation/policy, resource/channel regime,
solver tolerances, repetitions and fixture hashes. Do not compare different
regimes without labeling the difference. Timing is environment-dependent
performance evidence, not a physiological effect size or certification.

See [benchmark guide](../docs/benchmark_guide.md),
[analytical reference](../docs/quality/benchmark-reference.md) and
[release evidence](../release/README.md).

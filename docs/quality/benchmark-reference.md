# Analytical benchmark reference

## Deterministic smoke benchmark

Command:

```bash
physioworld benchmark
```

The benchmark exercises systematic GF(256) coding and decoding and the
canonical P1 capacitor transition. A current of 10 mA over 10 seconds from
2.65 V with `C=0.2 F`, `RESR=8 Ω` produces `Vterminal=2.07 V`, which is
feasible for `Vmin=1.85 V`. The exact safety-limit example is reported in the
[numerical qualification](numerical-qualification.md).

Runtime is environment metadata, not a scientific effect size. Large
benchmarks must record hardware, dependency versions, dataset identifiers,
split, seed, and resource/compute limits in their result manifest.

## Interpretation

The analytical fixture is distinct from the
[historical research comparisons](../scientific-context.md). Reproducing a
historical empirical claim requires its original source inputs and estimator;
copying its headline value into a benchmark is not reproduction.

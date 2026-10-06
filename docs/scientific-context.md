# Historical scientific context

The research context below preserves supplied corpus distinctions. These are
historical reference figures, **not newly generated release results**. The raw
cohorts, full estimators and policy tables required for independent reproduction
are not implied by their inclusion here. A claim promoted into release evidence
must identify its actual source artifacts, protocol, uncertainty and computation.
The [source ledger](SCIENTIFIC_SOURCE_LEDGER.yaml) records external source
identifiers, and the [reference ledger](REFERENCE_ARTIFACTS.yaml) contains the
distributed numerical extracts. Original documents and raw cohorts are not
included; see [known limitations](KNOWN_LIMITATIONS.md).

## Temporal history and external prediction

The declared forecasting exercise reported RMSE approximately 17.17 without
history, 12.70 with 5-unit history, 12.26 with 20-unit history, and 12.18 with
40-unit history: reductions about 26.0%, 28.6%, and 29.1%. Units/windows belong to
that protocol. The interpretation is task-specific temporal benefit with
diminishing returns, not a universal 5-second memory constant. The withdrawn
64-feature result is excluded from valid scientific headlines.

A separate 31-recording/subject external cohort had working diagnostic AUCs
approximately 0.61 instantaneous, 0.81 recent history and 0.83 level + trend.
Its scientific interpretation is predictive transport under the declared
protocol, not prospective clinical validation or causal intervention evidence.

## Representation battery

The battery combines BFV-R, BFV-L, BP and Joint tasks with 10, 20 and 30 s horizons.
`Z1S` is the full-history reference.

| Representation | Intrinsic task-balanced mean AUC | Worst-cell AUC |
|---|---|---|
| Z1S | 0.737206 | 0.531 |
| ZA15 | 0.767162 | 0.629 |
| ZB15 | 0.772869 | 0.578 |
| ZC15 | 0.770908 | 0.576 |
| ZMS | 0.768519 | 0.610 |

Highest average and strongest worst-task profile are distinct objectives.
The primary declared task-law point estimate `D_hat_T(ZMS)=0.086201874` does not
meet the illustrative tolerance 0.08. The separate bootstrap estimand is 0.080426
with 95% upper bound 0.114440; it must not replace the primary estimate.

## Transport and semantics

The supplied semantic negative control had mean communication discrepancy
approximately 0.057983 valid versus 0.527194 after semantic permutation, with
decode approximately 0.999346 in both. Worst-cell discrepancies were about
0.293563 and 0.996599. Decoding success therefore does not establish preserved
physiological task semantics.

## Resource-conditioned capability

The stored ZA15 versus Z1S comparison had 12/12 positive task gaps, mean ΔAUC
about 0.266544, range 0.185102–0.373086. Model energy was approximately 1.95 J
for ZA15 versus 7.8146596859 J for Z1S. Decode was about 1.0 versus 0.469895
(stored trials about 0.450–0.481).

These are attainable capabilities under a declared resource regime, not a
universal intrinsic information ordering. Intrinsic ZA15 BFV-L@10 AUC about
0.751885, resource-conditioned row AUC about 0.978033 and a separate R=8
aggregate mean gap about 0.131684 are different estimands.

Policy-optimized support gaps independently optimize H and Z policy sets.
The supplied recomputed directional gaps were:

```text
+0.00451, -0.00890, +0.00101, -0.00398, +0.00266, -0.00562,
+0.00330, -0.00647, +0.00363, -0.00246, +0.00204, -0.00698
```

Their range is −0.00890 to +0.00451 and maximum absolute gap 0.00890.
Any simultaneous uncertainty bound requires regeneration with the final
estimator and frozen protocol; superseded same-policy bounds are not authoritative.

## Resource numerical context

For the supplied synthetic family `(V0,Vmin,C,RESR,T)=(2.65,1.85,0.2,8,10)`,
the analytic zero-leakage current limit is `0.8/58 A ≈ 13.7931 mA`.
Additional model-specific reference targets were P1 about 13.39603 mA, P2 nominal
about 6.86462 mA and robust 10-s current about 6.113567 mA across 128 ±10% corners.
The P1 target includes a different declared model configuration from the
zero-leakage analytic bound; those quantities cannot be substituted.

| Safety margin | Tailored robust ready-state energy (J) | Uniform equal-voltage energy (J) |
|---|---|---|
| 0.000 | 0.529841 | 0.646043 |
| 0.025 | 0.549985 | 0.659806 |
| 0.050 | 0.571450 | 0.673713 |
| 0.100 | 0.618345 | 0.701963 |

At zero margin the supplied reduction is about 17.99%. Historical exact-transition
versus DOP853 error below 4.5e−15 V and energy residual below 8e−12 J require
the matching topology, parameters, solver and uncertainty set before they can be
claimed for a new implementation. These figures are numerical/model-derived,
not hardware measurements or statistical confidence intervals.

## Interpretation rule

Record the estimand name, domain, regime, conditioning, policy class, reference,
horizon, task family, resource state and uncertainty for every comparison.
Predictive, resource-conditioned, interventional and clinical claims remain
separate. See [scientific scope](scientific-scope.md) and
[scientific verification](quality/scientific-verification.md).

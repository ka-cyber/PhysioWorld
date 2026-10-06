# Architecture

PhysioWorld preserves the environment tuple `E=(S,R,H,A,Y,K,Π,C,I)` rather than
flattening physiology, information, physical resource and control state into an
uninterpretable vector. `Environment` and `WorldState` are core contracts;
`JointWorld` composes the scientific boundaries.

```text
dataset / physiological world
    → signals + time + events + provenance
    → perturbation / history / representation
    → task law / capability / PIR
    → communication / compute / resource
    → nonanticipative policy / endpoint
    → evidence / manifest / report
```

## Module boundaries

| Boundary | Modules and responsibility |
|---|---|
| Contracts | `core`: environment/state, declared constraints and epistemic origin |
| Observations | `physiology`, `datasets`, `io`: streams, acquisition/replay metadata and format adapters |
| Information | `representations`, `tasklaws`: history transforms and task-relative conditional objects |
| Capability | `capability`: feasible policy-value vectors, independent support optimization and PIR accounting |
| Delivery | `communication`, `compute`: transport/recovery and processing cost/placement |
| Physical feasibility | `resources`: dynamic resource state and explicit safety constraints |
| Decisions | `policies`, `causal`, `uncertainty`: policy evaluation, identification assumptions and uncertainty |
| Execution | `experiments`, `provenance`: compilation, configuration hashes, execution evidence and reports |
| Context of use | `clinical`: workload and frozen protocol objects |
| References | `benchmarks`: analytical smoke and family definitions |
| Laboratory | `instrumentation`, `devices`: calculation graphs, acquisition, calibration, recording and simulator/transport contracts |
| External models | `coupling`: explicit variable/unit/clock exchange, scheduled stepping and FMI boundaries |
| VVUQ | `verification`: numerical comparison, discrepancy, falsification and context-specific evidence |

Optional integrations import dependencies at their capability boundaries. Core
types remain usable without plotting, WFDB or numerical extension packages.
Dataset provider semantics map into canonical signals and provenance rather than
defining the scientific ontology. No restricted waveform is bundled.

## Scientific dependency direction

A representation has a declared history and training dependence. A task defines
the target estimand. A policy uses only its available information and admissible
actions. A capability comparison optimizes each feasible policy set separately.
Communication metrics describe transport and semantics separately. A resource
comparison requires a declared capability/policy-emulation relation.

Model and schema versions belong in persistent contracts. Algorithm changes
invalidate affected reference results until their evidence is regenerated;
version labels alone do not preserve scientific equivalence. See
[contracts](contracts.md), [API](api_reference.md) and
[model credibility](quality/model-credibility.md).

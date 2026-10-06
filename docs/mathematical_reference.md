# Mathematical reference

## Environment and history

`E=(S,R,H,A,Y,K,Π,C,I)` and `Z_t=φ(H_t)`. A representation must state its
history window, training dependence, task dependence, causal status, and cost.

## Task law

`gτ(Ht)=E[uτ(Hfuture)|Ht]`, `qτ(Zt)=E[gτ(Ht)|Zt]`,
`Dτ(Z)=E[|gτ-qτ|]`, and `DT(Z)=maxτ Dτ(Z)`. The empirical implementation
must identify its conditional object: `empirical_conditional_mean` estimates a
mean; `conditional_task_law` must expose the declared conditional law. Group
means cannot substitute for a distributional estimand.

## Binary certificate

`A_I*=1/2+E|P(Y=1|I)-1/2|` and, only for declared binary forecasting
assumptions, `0≤AH*-AZ*≤Dτ(Z)`. The implementation rejects intervention,
decision, and policy-value tasks.

## Capability

`CZ={V(π):π∈ΠZ}` and `hCZ(q)=supπ qᵀV(π)`. Full and reduced regions are
optimized independently. Coordinatewise maxima and average policy values are
available but never substituted for the region.

## PIR

`Linfo=Videal*-Vactual*`, `Lpolicy=Vactual*-Vactual(πA)`,
`Ltotal=Linfo+Lpolicy`, and
`PIR=(Vactual*-Vmin)/(Videal*-Vmin)`. `PIRContext` records task, horizon,
utility, policy, resource, communication, and reference.

## Resource

`E=1/2CV²`, `Vterminal=Vcap-IR_ESR`, and
`Imax=(V0-Vmin)/(T/C+RESR)` for the declared discharge convention. P2 Foster
branches retain explicit internal-state dynamics. The solver/input regime and
convergence checks determine numerical accuracy. Each energy
quantity is tagged by epistemic origin.

## Robustness

The APIs retain the distinction between `supπ infK V(π,K)` and
`infK supπ V(π,K)`. No minimax equality is assumed.

## Equation-to-evidence map

| Equation/object | Implementation | Assumption | Test/example | Evidence status |
|---|---|---|---|---|
| `h_CZ(q)=supπ qᵀV(π)` | `capability.support_gap` | Separate feasible policy sets, common direction and utility | Independent optimizer test; [Verification Manual](quality/verification-manual.md) PW-V-004 | Numerical verification, not empirical dominance |
| Binary accuracy bound | `tasklaws.binary_forecasting_certificate` | Binary outcome, same target/population, measurable representation | PW-V-005 | Conditional mathematical guarantee with computational checks |
| PIR and loss decomposition | `capability.compute_pir` | Common task, references, horizon, resource/policy regime | PW-V-006 | Reference-relative numerical accounting |
| Capacitor safety | `resources.canonical_current_limit` | Constant current, zero leakage, positive C, nonnegative ESR/time | [first calculation](getting_started.md), PW-V-001/002 | Model-derived numerical reference |
| P2 transition | `resources.FosterResource` | Declared coupled topology, input regime and parameters | PW-V-003; [numerical qualification](quality/numerical-qualification.md) | Differential solution verification within tested envelope |
| Finite Bellman optimum | `policies.DynamicProgrammingOracle` | Declared finite states/actions/kernel/horizon | PW-V-008 | Exact finite problem; no unrestricted physiology optimum |
| Interventional effect | `causal.BinaryTreatmentSCM` | Known synthetic structural equations | PW-V-010 | Synthetic causal model, not human identification |

PIV is `E[Vτ(G,M)−Vτ(G)]` for a declared information-availability comparison;
it is not confidence or a label correlation. A causal version additionally
requires identification. The approximation budget
`ε_total ≤ ε_phys + ε_grid + ε_transition + ε_stat` must identify each source
and its bounding convention; parameter corners are not automatically statistical
confidence bounds.

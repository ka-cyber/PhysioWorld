# Scientific specification

## Environment

The environment is `E = (S, R, H, A, Y, K, Π, C, I)`, where physiological
state `S`, resource state `R`, information/history state `H`, admissible
actions `A`, observations `Y`, transition/observation/resource kernel `K`,
nonanticipative policies `Π`, constraints `C`, and intervention semantics `I`
are distinct objects. `WorldState` stores these components rather than
flattening them into a single opaque vector.

## Task-law representation

For `Z_t = φ(H_t)`, a task law is represented by the conditional target or
utility distribution, not only a point score. For a predictive task:

`gτ(Ht) = E[uτ(Hfuture) | Ht]`

`qτ(Zt) = E[gτ(Ht) | Zt]`

`Dτ(Z) = E[|gτ(Ht) - qτ(Zt)|]` and `DT(Z) = maxτ Dτ(Z)`.

The binary forecasting certificate is implemented only under its declared
assumptions: `0 ≤ AH* - AZ* ≤ Dτ(Z)`. The software never applies that
certificate to intervention or control value.

## Capability

`C_Z = {V(π): π ∈ Π_Z}` and `h_CZ(q) = supπ qᵀV(π)` are first-class objects.
Support gaps optimize each policy set independently. A same-policy
comparison is not a support-function comparison.

## PIR

With explicit ideal, actual, minimum, horizon, task, utility, policy,
resource, communication, and reference conditions:

`L_info = Videal* - Vactual*`

`L_policy = Vactual* - Vactual(πA)`

`L_total = L_info + L_policy`

`PIR = (Vactual* - Vmin) / (Videal* - Vmin)`.

PIR is a derived accounting quantity and is never attached intrinsically to a
waveform.

## Resource state

The canonical capacitor uses `E = 1/2 C V²` and, for the discharge-current
convention, `Vterminal = Vcap - I R_ESR`. The exact constant-current safety
bound is `Imax = (V0 - Vmin)/(T/C + RESR)`. Values are tagged as measured,
model-derived, or simulated.

## Validation contract

Each module should have unit, analytical, numerical, and estimand tests.
Subject and temporal splitting, leakage checks, recovery semantics, resource
constraints, positivity, and optimizer order are explicit validation topics.

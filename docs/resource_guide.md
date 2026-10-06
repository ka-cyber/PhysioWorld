# Resource model guide

`ResourceState` includes terminal-relevant voltage, internal states, time,
load, and charging current. `CapacitorResource` supports the P0 ideal model
and P1 ESR/leakage; `FosterResource` adds P2 internal timescales. P0/P1 use
analytic constant-input transitions. P2 accuracy depends on its declared
coupled solver and input regime; verify the actual implementation and tolerances.

For discharge current `I`, terminal voltage is `Vcap - I RESR`. The canonical
zero-leakage bound is `Imax = (V0 - Vmin)/(T/C + RESR)`. Resource outputs use explicit
`MEASURED`, `MODEL-DERIVED`, or `SIMULATED` status tags.

Resource rankings must specify capability inclusion or policy emulation under
the same task and constraints. Componentwise energy/voltage ordering alone is
insufficient. Robust objectives use a common policy over the parameter set.
See [numerical qualification](quality/numerical-qualification.md).

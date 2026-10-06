# Numerical qualification

Numerical verification compares an implementation with an analytic or
independent computational reference. It is not a certification of physiological
or hardware validity. Test methods, parameters and tolerances are authoritative
in the versioned tests referenced below; record outcomes for the exact release.

## Capacitor reference

For constant discharge current, capacitance C, ESR and **zero leakage**:

```text
Vcap(T) = V0 - I T/C
Vterminal(T) = Vcap(T) - I RESR
Imax = (V0 - Vmin)/(T/C + RESR)
```

`canonical_current_limit(2.65, 1.85, 0.2, 8, 10)` gives
`0.8/58 = 0.01379310344827586 A`. This is a reference value evaluated from the
equation, not a measured device limit. At 10 mA the same zero-leakage model has
terminal voltage 2.07 V after 10 s.

Implementation: [capacitor.py](../../src/physioworld/resources/capacitor.py).
Independent Euler reference: [numerical.py](../../src/physioworld/resources/numerical.py).
The analytic current and terminal-voltage assertions in
[test_scientific_engines.py](../../tests/test_scientific_engines.py) use 12
decimal places; the independent capacitor integration comparison uses 8.
Those tolerances apply to that fixture and are not universal accuracy promises.

## Coupled internal resources

`FosterResource` maintains a vector of capacitor and polarization states.
`step_load_resistance()` couples internal voltage drops to current and therefore
to the evolution of the entire resource state. Constant-current and resistive-load
equations are different input regimes. Verify the declared regime before
comparing results.

The small branch test in
[test_safety_causal_and_p2.py](../../tests/test_safety_causal_and_p2.py) compares
the state transition with independent Euler integration and a 10,000-step
reference. It is a differential check at one configuration. Qualifying a new
operating envelope additionally needs timestep/convergence studies, energy balance,
threshold event checks, positivity and uncertainty corners. SciPy provides an
optional independent solver boundary for broader studies.

The expanded [resource qualification suite](../../tests/test_resource_control_qualification.py)
specifies a complete synthetic coupled circuit and independently integrates
its state and energy with SciPy DOP853 (`rtol=2.3e−14`, `atol=2e−15`). Its state
and energy checks require errors below 3e−12 for that fixture. A separate
controlled matrix exponential versus SciPy comparison requires error below
2e−14; a resistive-load state comparison requires error below 2e−12. These are
test acceptance rules, not reported execution residuals or guarantees over all
parameters. Interior safety-excursion timing, converter startup, thermal
degradation and robust crossing-policy checks extend the qualification scope.

Historical P2 current, robust ready-state energy and near-machine-precision
figures are [research context](../scientific-context.md). They may only become
release results after the exact model parameters, topology, uncertainty set,
solver and tolerances have been reproduced. Do not transfer a previous bound to
a changed model implementation.

## Discrete and probabilistic kernels

- GF(256) coding/decoding: source-symbol equality after full-rank recovery,
  permutation tests and explicit erasure failure in
  [communication tests](../../tests/test_communication_and_policies.py).
- Finite dynamic programming: compare action/value to exact enumeration in the
  declared finite MDP, not a claim of global physiological optimality.
- Distribution metrics: normalization, support, extreme probabilities and the
  exact total-variation convention belong in the estimand test contract.

For every numerical report include the test ID, input hash, software version,
reference method, tolerance, residual/error, parameter envelope and outcome.
See the [Verification Manual](verification-manual.md).

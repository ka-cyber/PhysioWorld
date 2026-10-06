# Scientific verification

## Estimand traceability

The scientific interfaces distinguish conditional task laws, conditional means,
distributional discrepancies, calibration, support-function gaps, PIR,
resource transitions, transport recovery, finite-MDP value and causal estimands.
Their definitions and code boundaries are mapped in the
[mathematical reference](../mathematical_reference.md).

## Analytical and negative-control requirements

- Binary optimal accuracy is `1/2 + E|p-1/2|`. A forecasting certificate requires
  binary outcomes, the same target/population and a measurable representation;
  it cannot certify intervention/control value.
- Capability support optimizes full and reduced policy sets independently.
  A same-policy comparison answers a different scientific question.
- PIR declares ideal, actual and minimum value references, task, utility,
  horizon, policy class, resource/channel regime and uncertainty.
- Resource energy carries epistemic origin. Constant-current capacitor safety
  is checked against an analytic reference; coupled states require independent
  numerical checks over the intended envelope.
- Communication recovery requires full finite-field rank. Packet delivery and
  semantic preservation are evaluated separately; erasures do not imply zeros.
- Causal interpretation requires identification assumptions and action support;
  predictive metrics do not establish those assumptions.

Tests specify these requirements in the
[Verification Manual](verification-manual.md). An asserted property or passing
fixture does not constitute an empirical external validation.

## Research context and study evidence

The historical temporal, representation, semantic-channel and resource figures
are retained as [scientific context](../scientific-context.md). Their data,
estimator and policy artifacts must be acquired and frozen before claiming an
independent reproduction. A reproduction report must distinguish unavailable
inputs, incompatible estimands and actual numerical disagreement.

Current clinical or external predictive results must identify the evaluated
cohort, endpoint, split, protocol and uncertainty. Model credibility is specific
to the [context of use](model-credibility.md).

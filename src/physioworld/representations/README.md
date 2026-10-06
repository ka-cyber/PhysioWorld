# Representation methods

These reference methods are implemented with Python's standard library. NumPy
is optional for differential verification, never a runtime fallback. Inputs must
be finite; missing-data preprocessing, discretization, time alignment and
training/evaluation splits are explicit caller responsibilities.

Each `transform` returns a `RepresentationResult(values, spec, metadata)`.
Information quantities are in **nats**. Predictive representations do not imply
causal sufficiency or clinical validation.

## Linear and history methods

```python
PCARepresentation(components=None, *, center=True, whiten=False,
                  variance_threshold=None, tolerance=1e-13, max_sweeps=100,
                  rank_tolerance=1e-12)
```

`fit(samples)` learns a centered sample-covariance eigenbasis via a scaled cyclic
Jacobi symmetric eigensolver. `transform(vector)` projects onto that fixed basis;
`inverse_transform(values_or_result)` reconstructs in the input coordinates.
Inspect `mean`, `components_`, `eigenvalues_`, `explained_variance_`,
`explained_variance_ratio_`, `rank_`, `n_components_` and `n_samples_`.
`components` remains the requested rank cap. A variance threshold chooses the
smallest rank meeting that fraction; a conflicting cap is rejected. Whitening
uses `n-1` sample variance and rejects selected numerically null directions.
`center=False` is explicitly an uncentered second-moment decomposition with
denominator `n`. Convergence failure and covariance overflow/underflow raise.
This dense reference solver targets small-to-moderate feature dimensions.

| API | Method |
| --- | --- |
| `RawSignalRepresentation()` | Validated identity on provided observations |
| `InstantaneousStateRepresentation()` | Last scalar observation; empty input raises |
| `FiniteHistoryRepresentation(window, pad_value=0.0)` | Last samples with declared left padding and padding count |
| `PredictiveHistoryRepresentation(dimension, pad_value=0.0)` | Honest finite-history predictive baseline |
| `MultiscaleHistoryRepresentation(windows, name="multiscale_history", statistics=("mean",))` | Trailing-window `mean`, population `std`, sample-index OLS `trend`, or `last` |
| `HandcraftedPhysiologyRepresentation()` | Generic mean, population std, RMS, min, max, sign crossings; not biomarker validation |
| `RandomProjectionRepresentation(dimension, input_dimension, seed=0)` | Seeded Gaussian matrix with variance `1/dimension`; strict input dimensions |
| `TaskSpecificRepresentation(task_id, feature_indices)` | Declared feature selection, with no learned/sufficiency claim |
| `LearnedLatentRepresentation(input_dimension, latent_dimension=2)` | Fitted orthogonal linear reconstruction latent, with `fit`, `transform`, `inverse_transform`, and `reconstruction_loss_` |

`transform_at(history, timestamps, prediction_time)` selects the acquired prefix
before transformation, with strictly increasing timestamps in a common declared
unit. It records excluded future samples. The method also works for PSR
action/observation histories. It is not automatic temporal alignment of
multimodal feature vectors. Padding is explicit only in the finite-history
methods; empty statistical windows and unavailable features raise.

## Discrete information bottleneck and task compression

```python
InformationBottleneckRepresentation(dimension, task_id="supervised", *, beta=2.0,
                                    max_iterations=1000, tolerance=1e-9,
                                    n_init=4, seed=0)
```

`fit(samples, labels)` estimates the exact discrete training joint; numeric
vectors are exact tuple keys, not silently binned. `fit_joint(joint, *,
x_symbols=None, y_symbols=None, estimator="declared_discrete_joint")` accepts a
declared model distribution. The actual optimization is alternating
Blahut–Arimoto for `I(X;Z) - beta I(Z;Y)`, using seeded multiple starts. The
problem is nonconvex: the selected solution is locally optimized, with no global
optimum claim. `dimension` means latent **alphabet cardinality**.

`transform(x)` returns the categorical encoder `q(z|x)`. `sample(x, *, seed)`
returns a latent symbol; `predict_proba(x)` marginalizes the learned label decoder
over that encoder. Unknown source keys raise. Inspect `encoder_`, `decoder_`,
`x_symbols_`, `y_symbols_`, `latent_probability_`, `rate_`, `relevance_`,
`objective_`, `objective_history_`, `converged_`, `n_iter_` and `metadata_`.
Metadata declares the estimator, objective, optimization, regularization,
stochasticity, seed, tolerance, convergence and evidence class.

`SupervisedBottleneckRepresentation(dimension, task_id, **kwargs)` provides the
same genuine supervised IB formulation under its existing public class name.

```python
RateDistortionRepresentation(beta=1.0, *, task_id="declared_distortion",
                             max_iterations=2000, tolerance=1e-10)
```

`fit(source_probabilities, distortions, *, source_symbols=None)` optimizes
`I(X;Z) + beta E[d(X,Z)]` using Blahut–Arimoto. Distortion columns are declared
task reproduction symbols/actions, with caller-declared cost units.
`transform(x)` returns its categorical encoder; `sample(x, *, seed)` samples it.
`rate_`, `distortion_`, `objective_history_` and `metadata_` expose the tradeoff.

## Stochastic variational bottleneck

```python
VariationalInformationBottleneck(input_dimension, latent_dimension=2, beta=1e-3,
                                learning_rate=0.05, epochs=100, seed=0, *,
                                samples_per_observation=16, tolerance=1e-6)
```

`fit(samples, binary_labels)` trains `q(z|x)=N(Wx+b,I)` and a logistic decoder
using reparameterized Monte Carlo batch gradients for
`E_q[BCE] + beta KL(q(z|x)||N(0,I))`. Unit covariance is the declared variational
family. `transform(x)` reports the encoder mean and its variance;
`sample(x, *, seed)` draws the actual stochastic latent;
`predict_proba(x, *, draws=256, seed=0)` averages stochastic decoded predictions.

`rate_upper_bound_` is mean analytic Gaussian KL, **not** measured MI.
`training_loss` includes the initial objective and post-update estimates using a
fixed evaluation noise bank; stochastic optimization need not decrease each
epoch. `converged_` reports the declared Monte Carlo gradient-norm criterion,
not guaranteed convergence simply because epochs finished.

Beta conventions differ: increasing **discrete IB** beta emphasizes label
relevance; increasing **VIB** beta emphasizes compression regularization;
increasing **rate/distortion** beta emphasizes lower distortion.

## Observable predictive states

```python
PredictiveStateRepresentation(transitions, emissions, initial, *,
                              observations=None, tests=None, rank_tolerance=1e-11)
```

Transitions and emissions are mappings from action to stochastic matrices. An
action transitions hidden state first, then emits the observation. Each future
test is a sequence of `(action, observation)` pairs; its probability conditions
on that fixed action sequence. Default tests are all one-step observable tests.
The matrix of test probabilities, augmented by the constant test, must have full
hidden-state column rank. Add multistep tests when necessary; insufficient test
families raise.

`initial_state()`, `transform(action_observation_history)`, and
`update(state, action, observation)` operate on test-probability states, using a
left-inverse belief realization and exact Bayes update under the declared model.
`predict_test(state, test)` predicts arbitrary future sequences;
`predict_observation(state, action)` predicts the next observation distribution.
`from_belief(belief)` and `belief_from_state(state)` permit model-consistency
round trips. Impossible observations and inconsistent states raise.
`predictive_sufficiency()` reports observability rank and left-inverse residual.
This is a finite controlled-HMM **model-based PSR**, not a learned empirical PSR
or a causal-identification result.

## Task laws, quotients, modalities and uncertainty

- `EmpiricalTaskLawRepresentation(task_id, *, neighbors=None)` fits scalar
  outcome probability masses via exact discrete keys or uniform Euclidean kNN.
  `fit(samples, outcomes, *, acquisition_times=None, training_cutoff=None)` can
  explicitly reject future training samples. `transform`/`predict_proba` return
  the full PMF on `support_`; `expectation(history, function=float)`,
  `quantile(history, level)` and `interval(history, coverage=0.9)` expose separate
  functionals. `evaluate(samples, outcomes, *, coverage=0.9,
  conditioning_groups=None, calibration_bins=10)` reports held-out proper scores,
  classwise calibration diagnostics and grouped interval coverage.
  `bootstrap_expectation(history, subjects, *, draws=200, seed=0, function=float)`
  resamples whole subjects; unsupported query replicates remain explicit and
  suppress the percentile interval. Coverage guarantees are not asserted.
- `QuotientRepresentation(classes)` encodes a declared finite equivalence
  relation. `class_index(history)` returns a class, and `inverse_class(index_or_result)`
  returns all members, not an invented unique inverse.
- `MinimalSufficientRepresentation(task_laws)` and
  `minimal_sufficient_representation(task_laws)` compute the exact coarsest
  finite deterministic quotient preserving every supplied task distribution.
  Input is `{state: {task_id: aligned_probability_vector}}`. `task_law` and
  `certificate` expose preservation and the finite-domain scope. Equal means
  with different distributions do not merge.
- `StateAbstractionRepresentation(transitions, rewards, *, states=None)` uses
  partition refinement for exact reward-respecting controlled probabilistic
  bisimulation. `transitions_` and `rewards_` are the actual quotient MDP.
  Approximate equality is not silently treated as exact equivalence.
- `SharedMultimodalRepresentation(modalities, dimension, *, standardize=True)`
  fits a joint PCA latent to aligned mappings of modality names to vectors.
  `inverse_transform` reconstructs each modality in its original units. Shared
  joint coordinates are not a claim of CCA/common-cause factor identification.
  Missing/mismatched channels and changing modality widths raise.
- `UncertaintyAwareRepresentation(representation, *, append_std=True)` wraps a
  PCA or random projection. `transform(history, covariance)` propagates full
  covariance by `A Sigma A'`, retaining correlation in metadata and optionally
  appending marginal standard deviations. `sample(history, covariance, *, seed)`
  draws a declared Gaussian uncertainty model. Supplied covariance must be
  symmetric PSD; its calibration is a caller assumption.

## Negative controls and migration

`permutation_control(values, *, control, seed=0)` supports `label_permutation`,
`semantic_permutation` and `temporal_shuffle`, recording the index permutation.
`randomized_representation_control(samples, *, seed=0)` supplies independent
shape-matched Gaussian features. Controls preserve replayable provenance, but
their effect must be measured: a permutation of constant labels has no effect.
Time-cutoff APIs make future-information injection testable. Strict modality
matching makes channel-mismatch injection testable.

- The former magnitude-sorting PCA behavior is removed. PCA now requires a
  rectangular training matrix with at least two samples and projects onto a
  learned fixed basis; its signs are meaningful, with canonical eigenvector
  signs. Repeated eigenspaces have no uniquely identifiable basis.
- The truncating `InformationBottleneckRepresentation` and
  `SupervisedBottleneckRepresentation` now require fitting and output categorical
  probabilities. Downstream consumers should deliberately sample or integrate
  this encoder rather than treating its probabilities as truncated features.
- `PredictiveStateProxy` is removed. Use `PredictiveHistoryRepresentation` for
  finite history. The former `PredictiveStateRepresentation(history_dimension,
  future_dimension).fit(...)` conditional-mean table has been replaced by the
  controlled-HMM constructor above. For empirical future outcome laws, use
  `EmpiricalTaskLawRepresentation`.
- `LearnedLatentRepresentation` is no longer an alias to VIB; it is a separately
  fitted linear reconstruction model. Use `VariationalInformationBottleneck`
  explicitly for binary supervised stochastic learning.
- Multimodal mean repetition is removed. Supply aligned named modality rows and
  fit the joint model before transforming.
- Random projection and task feature selection enforce exact availability;
  implicit padding/truncation and NaN dropping are removed. Finite-history
  padding remains an explicitly parameterized operation.

Run `examples/representation_demo.py` for a synthetic numerical demonstration.
The verification suite checks rotated covariance eigenvectors, signed
projections, inverse/whitening/rank, analytic information tradeoffs, stochastic
VIB objectives, observable prediction/update, quotient laws, clustered bootstrap,
negative controls, and future exclusion.

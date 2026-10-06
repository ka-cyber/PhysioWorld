# API reference

Public modules are grouped by scientific boundary:

- `physioworld.physiology`: `SignalStream`, `PhysiologicalWorld`, medical
  information state, and PIV.
- `physioworld.perturbations`: deterministic artifact, missingness, network,
  regime, resource, and shift generators.
- `physioworld.representations`: history, feature, PCA, task, and
  multimodal representations with `RepresentationSpec`.
- `physioworld.tasklaws`: task entities, binary certificate, distribution
  discrepancies, calibration, and task-family aggregation.
- `physioworld.capability`: policy-value regions, independent support
  optimization, and PIR decomposition.
- `physioworld.resources`: P0/P1/P2 state transitions, safety,
  reachability, robustness, and dominance.
- `physioworld.communication`: packets, explicit erasures, quantization,
  codecs, channels, and GF(256) RLNC.
- `physioworld.compute` and `physioworld.policies`: placement and common
  action interfaces, including a finite DP oracle.
- `physioworld.causal` and `physioworld.uncertainty`: estimands,
  assumptions, diagnostics, estimators, intervals, and robust order.

The CLI is registered as `physioworld` by `pyproject.toml`.

World and laboratory interfaces are also available through:

- `physioworld.physiology.engine` and `.scenario`: patient parameters,
  coupled physiological state, scenarios, events, data requests and serialization.
- `physioworld.physiology.estimation`: explicit process/measurement models,
  Kalman/extended and particle estimation with uncertainty.
- `physioworld.instrumentation`: calculation graphs, signal analysis,
  physiological measurement functions and recording organization.
- `physioworld.devices`: acquisition contracts, calibration, device simulator,
  transports and qualification procedures.
- `physioworld.coupling`: declared variable/time/unit boundaries, co-simulation
  master and FMI round-trip interfaces.
- `physioworld.io`: canonical JSON/tabular/clinical formats and optional
  WFDB, Arrow/Parquet, HDF5, NWB and DICOM adapters.
- `physioworld.verification`: numerical comparisons, discrepancy, falsification,
  evidence registry and context-specific validation.

Consult the concrete module docstrings and the versioned executable examples
before selecting a model fidelity or external format. The adapter's unit/time
requirements and runtime dependencies are part of its contract.

## Entry-point map

| Scientific object | Import boundary | Reference use |
|---|---|---|
| `SignalStream`, `PhysiologicalWorld` | `physioworld.physiology` | Typed observations and timeline |
| `ExperimentCompiler` | `physioworld.experiments` | Validate/compile a declarative experiment |
| `TaskDefinition`, `conditional_task_law` | `physioworld.tasklaws` | Task definition and declared conditional object |
| `CapabilityRegion`, `support_gap`, `compute_pir` | `physioworld.capability` | Independent support optimization and reference-relative accounting |
| `ResourceState`, `CapacitorResource`, `FosterResource` | `physioworld.resources` | Dynamic resource state and explicit safety regime |
| `Packet`, `ErasureChannel`, `RLNCEncoder`, `RLNCDecoder` | `physioworld.communication` | Packet semantics and finite-field recovery |
| `ComputeEngine` | `physioworld.compute` | Declared processing costs/placement |
| `FiniteMDP`, `DynamicProgrammingOracle` | `physioworld.policies` | Exact finite-horizon decision problem |
| `ClinicalValidationProtocol` | `physioworld.clinical` | Frozen validation protocol |

Use `help()` on the installed API for its precise signature. Package `__all__`
exports identify the intended import boundary. Concrete source modules remain
available for equation review, but callers should prefer these import boundaries.
The [Verification Manual](quality/verification-manual.md) links executable API
uses to reference cases. The [developer guide](developer_guide.md) describes
compatibility and evidence requirements.

## Level 3 scientific interfaces

These interfaces extend the task-first laboratory with typed scientific
contracts, finite reference calculations, and explicit evidence boundaries.
The native campaign is described in the [corpus guide](corpus_guide.md).

| Boundary | Public imports | Usage / evidence scope |
|---|---|---|
| Estimands and evidence | `physioworld.core`: `Task`, `MetricContext`, `Metric`, `EstimandFirewall`, `AssumptionSet`, `Claim`, `EvidenceGraph` | Preserve target, population, regime, reference, uncertainty, and causal status; reject incompatible metric comparisons |
| Information flow | `physioworld.core`: `FiniteKernel`, `InformationFlowGraph`, `LocalGuarantee`, `ScientificPipelineCertificate` | Explicit finite kernels and local guarantees; pipeline certificates depend on their declared assumptions |
| Conditional distributions | `physioworld.tasklaws`: `empirical_conditional_distribution`, `kernel_conditional_distribution`, `task_function_discrepancy`, `clustered_bootstrap`, `conditional_coverage` | Declare cells/kernel, task functions, independent resampling units, and coverage; a conditional mean is a different estimand |
| Capability and information value | `physioworld.capability`: `RequirementRegion`, `StochasticCapabilityRegion`, `StrategicMeasure`, `optimal_task_value`, `estimate_piv`, `representation_deficiency`, `finite_policy_pir`, `finite_causal_pir` | Finite policy/utility/reference-relative calculations; causal variants require identification assumptions |
| Policy and abstraction | `physioworld.policies`: `NonanticipativePolicy`, `SafePolicyFilter`, `evaluate_finite_policy`, `check_markov_quotient`, `quotient_mdp`, `viability_kernel` | Information timing, safety filtering, finite policy evaluation, and explicit quotient checks; no unrestricted optimality claim |
| Packet semantics | `physioworld.communication`: `PacketReceiver`, `ScientificContext`, `TransportStatistics`, `SemanticStatistics`, `evaluate_semantics`, `RLNCPacketEncoder`, `RLNCPacketDecoder` | Keep transport recovery separate from task semantics under declared timing, contexts, and missingness |
| Resource evidence | `physioworld.resources`: `LifecycleResource`, `MultiResourceBudget`, `physical_schedule`, `CalibrationArtifact`, `ParameterStatus`, `corner_analysis` | State, schedules, parameter origin, and finite uncertainty corners; model-derived energy is not a measurement |
| Clinical readiness | `physioworld.clinical`: `ClinicalValidationProtocol`, `protocol_from_config`, `validation_readiness` | Freeze explicit declarations and classify readiness without evaluating endpoints |
| Locked cohort evaluation | `physioworld.clinical`: `AlarmWorkload`, `FrozenArtifacts`, `LockedClinicalProtocol`, `EvaluationCohort`, `EndpointObservation`, `evaluate_independent_cohort` | Evaluate supplied, independently sourced observations against locked artifacts/protocol; external predictive evaluation does not establish clinical validation |
| Predictive transport | `physioworld.experiments`: `TransferProtocol`, `RegimeDataset`, `train_transfer`, `freeze_transfer`, `evaluate_transfer` | Train only on the source-training partition, freeze artifacts, then evaluate held-out source/target partitions; current transfer evidence is simulation-only |
| Network physiology | `physioworld.experiments`: `analyze_network_physiology`, `pearson_matrix`, `mutual_information_matrix`, `phase_randomized_surrogate` | Aligned, uniformly sampled supplied channels; undirected association and a declared surrogate null, not causal mechanisms |
| Monitoring | `physioworld.monitoring`: `WindowedDriftMonitor`, `EventTimeWindow`, `compile_adaptation_split`, `fit_pod` | Event-time and fit/evaluation boundaries; POD requires the scientific extra |
| Credibility and falsification | `physioworld.verification`: `ReferenceRegistry`, `VerificationRun`, `CredibilityCase`, `CounterexampleRegistry`, `permutation_control`, `finite_design_search` | Execute scoped comparisons and negative controls; credibility remains specific to the declared context of use |
| Native corpus | `physioworld.benchmarks`: `run_corpus_campaign`, `run_benchmark_family` | Package-bound synthetic/mathematical fixtures and hashed case artifacts |

For a complete readiness declaration, see the
[clinical guide](clinical_validation_guide.md). For typed laboratory command
execution, see [declarative protocols](protocols.md). Source extracts and
test-backed claim scopes are in the [source ledger](SCIENTIFIC_SOURCE_LEDGER.yaml)
and [claim ledger](CLAIM_LEDGER.yaml); [known limitations](KNOWN_LIMITATIONS.md)
state the external-data boundary.

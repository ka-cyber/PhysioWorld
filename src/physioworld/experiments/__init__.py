"""Declarative experiment execution helpers."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .compiler import ExperimentCompiler, ExperimentPlan, ExperimentValidationError
from .level3 import (
    run_nirs_proxy_synthetic,
    run_pir_loss_decomposition,
    run_piv_synthetic,
    run_representation_lattice,
    run_resource_conditioned_capability,
    run_subject_bootstrap_uncertainty,
    run_support_function_experiment,
    run_task_battery,
    run_task_law_discrepancy,
    run_temporal_history_sweep,
)
from .manager import ExperimentManager, ExperimentRun
from .network_physiology import (
    AssociationResult,
    NetworkPhysiologyResult,
    TopologySummary,
    analyze_network_physiology,
    mutual_information_matrix,
    pearson_matrix,
    phase_randomized_surrogate,
    topology_summary,
)
from .protocol import (
    Acquire,
    Baseline,
    Calibration,
    Computation,
    Condition,
    Decision,
    Event,
    ExecutionContext,
    ExecutionLimits,
    ExecutionResult,
    Export,
    Handler,
    HandlerContract,
    HandlerRegistry,
    If,
    Intervention,
    Loop,
    Measurement,
    Node,
    ParameterRef,
    ParameterSweep,
    PhysioScript,
    Protocol,
    ProtocolAbort,
    ProtocolBuilder,
    ProtocolError,
    ProtocolExecutionError,
    ProtocolHandler,
    ProtocolLimitError,
    ProtocolRuntime,
    ProtocolValidationError,
    Quantity,
    Reference,
    Resource,
    Schedule,
    ScheduleCase,
    Setup,
    SimulationClock,
    Stimulus,
    Stop,
    Wait,
    load_protocol_schema,
)
from .result import ExperimentResult
from .transfer import (
    CrossRegimeTransferResult,
    FrozenTransfer,
    Regime,
    RegimeDataset,
    RegimeEvaluation,
    RegimeSequence,
    TrainedTransfer,
    TransferHashes,
    TransferPartition,
    TransferProtocol,
    TransportDiagnostics,
    evaluate_transfer,
    freeze_transfer,
    train_transfer,
)

if TYPE_CHECKING:
    from .causal_vvuq import run_causal_vvuq as run_causal_vvuq
    from .communication_vvuq import run_communication_vvuq as run_communication_vvuq
    from .resource_vvuq import run_resource_vvuq as run_resource_vvuq


_VVUQ_MODULES = {
    "run_causal_vvuq": ".causal_vvuq",
    "run_communication_vvuq": ".communication_vvuq",
    "run_resource_vvuq": ".resource_vvuq",
}


def __getattr__(name: str) -> Any:
    # Keep reference machinery out of package initialization and import cycles.
    if name not in _VVUQ_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    runner = getattr(import_module(_VVUQ_MODULES[name], __name__), name)
    globals()[name] = runner
    return runner


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_VVUQ_MODULES))


__all__ = [
    "ExperimentManager",
    "ExperimentRun",
    "ExperimentCompiler",
    "ExperimentPlan",
    "ExperimentValidationError",
    "ExperimentResult",
    "AssociationResult",
    "NetworkPhysiologyResult",
    "TopologySummary",
    "analyze_network_physiology",
    "mutual_information_matrix",
    "pearson_matrix",
    "phase_randomized_surrogate",
    "topology_summary",
    "run_nirs_proxy_synthetic",
    "run_pir_loss_decomposition",
    "run_piv_synthetic",
    "run_representation_lattice",
    "run_resource_conditioned_capability",
    "run_subject_bootstrap_uncertainty",
    "run_support_function_experiment",
    "run_task_battery",
    "run_task_law_discrepancy",
    "run_temporal_history_sweep",
    "Regime",
    "RegimeSequence",
    "RegimeDataset",
    "TransferPartition",
    "TransferProtocol",
    "TrainedTransfer",
    "FrozenTransfer",
    "TransferHashes",
    "RegimeEvaluation",
    "TransportDiagnostics",
    "CrossRegimeTransferResult",
    "train_transfer",
    "freeze_transfer",
    "evaluate_transfer",
    "run_causal_vvuq",
    "run_communication_vvuq",
    "run_resource_vvuq",
    "Acquire",
    "Baseline",
    "Calibration",
    "Computation",
    "Condition",
    "Decision",
    "Event",
    "ExecutionContext",
    "ExecutionLimits",
    "ExecutionResult",
    "Export",
    "Handler",
    "HandlerContract",
    "HandlerRegistry",
    "If",
    "Intervention",
    "Loop",
    "Measurement",
    "Node",
    "ParameterRef",
    "ParameterSweep",
    "PhysioScript",
    "Protocol",
    "ProtocolAbort",
    "ProtocolBuilder",
    "ProtocolError",
    "ProtocolExecutionError",
    "ProtocolHandler",
    "ProtocolLimitError",
    "ProtocolRuntime",
    "ProtocolValidationError",
    "Quantity",
    "Reference",
    "Resource",
    "Schedule",
    "ScheduleCase",
    "Setup",
    "SimulationClock",
    "Stop",
    "Stimulus",
    "Wait",
    "load_protocol_schema",
]

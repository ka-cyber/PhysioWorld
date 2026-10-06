"""Renderer-independent inspection of recorded local experiment evidence.

``ExperimentObservation`` binds existing result, manifest, channel, protocol and
qualification objects to a selectable run. ``build_inspection`` projects their
records into a versioned JSON model; ``render_inspection`` returns terminal-safe
text or JSON without printing, executing experiments, or opening artifacts.

Scientific quantities come from the supplied objects. In particular, aggregate
transport statistics cannot supply packet attempts, and selected policy actions
apply only to fully followed-up rows. Missing traces are explicitly marked
NOT_RECORDED. Completion, integrity and qualification have separate meanings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

from physioworld.core.contracts import VersionedRecord, finite

if TYPE_CHECKING:
    from physioworld.communication.models import ChannelResult, Packet
    from physioworld.devices.qualification import QualificationReport
    from physioworld.experiments.protocol import ExecutionResult
    from physioworld.experiments.result import ExperimentResult
    from physioworld.plugins import PluginQualification
    from physioworld.provenance import ResultManifest, Verification


class TraceStatus(str, Enum):
    RECORDED = "RECORDED"
    NOT_RECORDED = "NOT_RECORDED"


def _freeze(value: Any) -> Any:
    if isinstance(value, (VersionedRecord, Enum)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("inspection mappings require string keys")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        finite(value, "inspection value")
        return value
    raise ValueError(f"inspection contains a non-JSON value: {type(value).__name__}")


class _InspectionRecord(VersionedRecord):
    def __post_init__(self) -> None:
        for item in fields(self):  # type: ignore[arg-type]
            object.__setattr__(self, item.name, _freeze(getattr(self, item.name)))


@dataclass(frozen=True)
class ExperimentSummary(_InspectionRecord):
    run_id: str
    result_id: str
    experiment_id: str
    seed: int
    evidence_class: str
    estimand: str
    run_status: str
    qualification_status: str


@dataclass(frozen=True)
class TraceRecord(_InspectionRecord):
    """An indexed source record; source is a path within the supplied object."""

    index: int
    row_index: int | None
    time_s: float | None
    source: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class TimelineEvent(_InspectionRecord):
    time_s: float
    kind: str
    row_index: int | None
    source: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class RepresentationSample(_InspectionRecord):
    row_index: int
    acquisition_time_s: float | None
    original: tuple[float, ...] | None
    quantized: tuple[float, ...] | None
    received: tuple[float, ...] | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RepresentationInspection(_InspectionRecord):
    declaration: Mapping[str, Any]
    trace_status: TraceStatus
    samples: tuple[RepresentationSample, ...]


@dataclass(frozen=True)
class PacketSnapshot(_InspectionRecord):
    sequence: int
    acquisition_time_s: float
    header: Mapping[str, Any]
    metadata: Mapping[str, Any]
    payload_hex: str


@dataclass(frozen=True)
class PacketAttempt(_InspectionRecord):
    sequence: int
    attempt: int
    departure_s: float
    arrival_s: float | None
    status: str
    channel_state: str | None
    duplicate: bool
    packet: PacketSnapshot | None


@dataclass(frozen=True)
class PacketTiming(_InspectionRecord):
    sequence: int
    delay_s: float | None
    arrival_s: float | None


@dataclass(frozen=True)
class PacketInspection(_InspectionRecord):
    trace_status: TraceStatus
    statistics: Mapping[str, Any]
    channel_statistics: Mapping[str, Any] | None
    sent: tuple[PacketSnapshot, ...]
    received: tuple[PacketSnapshot, ...]
    lost_sequences: tuple[int, ...]
    attempts: tuple[PacketAttempt, ...]
    timings: tuple[PacketTiming, ...]


@dataclass(frozen=True)
class ResourceInspection(_InspectionRecord):
    declaration: Mapping[str, Any]
    summary: Mapping[str, Any]
    trace_status: TraceStatus
    transitions: tuple[TraceRecord, ...]


@dataclass(frozen=True)
class PolicyDecision(_InspectionRecord):
    row_index: int
    acquisition_time_s: float | None
    decision_time_s: float | None
    available: bool | None
    endpoint_status: str | None
    endpoint: str | int | float | bool | None
    full_decision_statistic: float | None
    received_decision_statistic: float | None
    action: str | None
    fallback_action: str | None
    unavailability_reason: str | None


@dataclass(frozen=True)
class PolicyInspection(_InspectionRecord):
    declaration: Mapping[str, Any]
    summary: Mapping[str, Any]
    selected_policy: str | None
    trace_status: TraceStatus
    decisions: tuple[PolicyDecision, ...]


@dataclass(frozen=True)
class QualificationInspection(_InspectionRecord):
    source: str
    status: str
    passed: bool | None
    evidence_scope: str | None
    assurance_tier: str | None
    evidence_status: str | None
    checks: tuple[Mapping[str, Any], ...]
    report: Mapping[str, Any] | None


@dataclass(frozen=True)
class ProvenanceInspection(_InspectionRecord):
    result: Mapping[str, Any]
    manifest: Mapping[str, Any] | None
    integrity_status: str
    verification: Mapping[str, Any] | None
    protocol: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ExperimentInspection(_InspectionRecord):
    summary: ExperimentSummary
    scientific_context: Mapping[str, Any]
    outputs: Mapping[str, Any]
    timeline: tuple[TimelineEvent, ...]
    representations: RepresentationInspection
    packets: PacketInspection
    resources: ResourceInspection
    policy: PolicyInspection
    provenance: ProvenanceInspection
    qualification: QualificationInspection
    protocol: Mapping[str, Any] | None


@dataclass(frozen=True)
class LabInspection(_InspectionRecord):
    experiments: tuple[ExperimentSummary, ...]
    selected_run_id: str | None
    selected: ExperimentInspection | None


@dataclass(frozen=True)
class ExperimentObservation:
    """Explicit inputs for one local run; no discovery or qualification is run.

    Pass ``run_vertical(...).result`` or an ``ExperimentResult.from_json(...)``
    result. A manifest supplies the run ID unless an identical ID is explicit;
    without a manifest the result ID is the default. Explicit run IDs distinguish
    repeated executions of the same result. A ``Verification`` receipt records
    existing integrity evidence; a plain manifest is marked NOT_VERIFIED.

    Optional traces/reports are caller-bound to this run. Their original IDs,
    provenance and qualification scope remain inspectable.
    """

    result: ExperimentResult
    manifest: ResultManifest | Verification | None = None
    channel: ChannelResult | None = None
    protocol: ExecutionResult | None = None
    qualification: PluginQualification | QualificationReport | Mapping[str, Any] | None = None
    run_id: str | None = None


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _records(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array of records")
    return tuple(_mapping(item, name) for item in value)


def _time(value: Any, name: str) -> float | None:
    if value is not None:
        finite(value, name)
    return value


def _text(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def _values(value: Any, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a numeric array or null")
    for item in value:
        finite(item, name)
    return tuple(value)


def _number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    finite(value, name)
    return float(value)


def _scalar(value: Any, name: str) -> str | int | float | bool | None:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        finite(value, name)
        return value
    raise ValueError(f"{name} must be a JSON scalar or null")


def _qualification(value: Any) -> QualificationInspection:
    from physioworld.devices.qualification import QualificationReport
    from physioworld.plugins import PluginQualification

    if value is None:
        return QualificationInspection("none", "NOT_EVALUATED", None, None, None, None, (), None)
    if isinstance(value, PluginQualification):
        report = value.to_dict()
        source = "plugin"
        status = value.assurance_tier
    elif isinstance(value, QualificationReport):
        from dataclasses import asdict

        report = asdict(value)
        report["passed"] = value.passed
        source = "device"
        status = "PASS" if value.passed else "FAIL"
    elif isinstance(value, Mapping):
        report = dict(value)
        source = "report"
        status = report.get("status", report.get("assurance_tier", "NOT_EVALUATED"))
    else:
        raise TypeError("qualification requires an existing qualification report")
    if not isinstance(status, str) or not status:
        raise ValueError("qualification status must be a nonempty string")
    passed = report.get("passed")
    if passed is not None and type(passed) is not bool:
        raise ValueError("qualification passed must be a boolean or null")
    return QualificationInspection(
        source,
        status,
        passed,
        _text(report.get("evidence_scope"), "qualification evidence_scope"),
        _text(report.get("assurance_tier"), "qualification assurance_tier"),
        _text(report.get("evidence_status"), "qualification evidence_status"),
        _records(report.get("checks", ()), "qualification checks"),
        report,
    )


def _identity(
    observation: ExperimentObservation,
) -> tuple[ExperimentSummary, ResultManifest | None, Verification | None]:
    from physioworld.experiments.result import ExperimentResult
    from physioworld.provenance import ResultManifest, Verification

    if not isinstance(observation, ExperimentObservation):
        raise TypeError("inspection requires ExperimentObservation inputs")
    result = observation.result
    if not isinstance(result, ExperimentResult):
        raise TypeError("observation result must be an ExperimentResult")
    receipt = observation.manifest if isinstance(observation.manifest, Verification) else None
    manifest = receipt.manifest if receipt else observation.manifest
    if manifest is not None and not isinstance(manifest, ResultManifest):
        raise TypeError("observation manifest must be a ResultManifest or Verification")
    run_id = observation.run_id
    if manifest is not None:
        if run_id is not None and run_id != manifest.run_id:
            raise ValueError("run ID conflicts with manifest")
        if result.seed != manifest.seed or result.software_version != manifest.software_version:
            raise ValueError("result seed/software version conflicts with manifest")
        recorded_hash = result.provenance.get("configuration_hash")
        if recorded_hash is not None and recorded_hash != manifest.configuration_hash:
            raise ValueError("result configuration hash conflicts with manifest")
        if result.commit and manifest.git_commit and result.commit != manifest.git_commit:
            raise ValueError("result commit conflicts with manifest")
        run_id = manifest.run_id
    if run_id is None:
        run_id = result.result_id
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run ID must be a nonempty string")
    qualification = _qualification(observation.qualification)
    return (
        ExperimentSummary(
            run_id,
            result.result_id,
            result.experiment_id,
            result.seed,
            result.evidence_class.value,
            result.estimand,
            manifest.status if manifest else "UNKNOWN",
            qualification.status,
        ),
        manifest,
        receipt,
    )


def _packet(packet: Packet) -> PacketSnapshot:
    from dataclasses import asdict

    return PacketSnapshot(
        packet.sequence,
        packet.timestamp,
        asdict(packet.header) if packet.header is not None else {},
        packet.metadata,
        packet.payload.hex(),
    )


def _packets(channel: ChannelResult | None, statistics: Mapping[str, Any]) -> PacketInspection:
    if channel is None:
        return PacketInspection(TraceStatus.NOT_RECORDED, statistics, None, (), (), (), (), ())
    from dataclasses import asdict

    from physioworld.communication.models import ChannelResult

    if not isinstance(channel, ChannelResult):
        raise TypeError("observation channel must be a ChannelResult")
    # Use the domain object's transport calculation, never a renderer-specific metric.
    return PacketInspection(
        TraceStatus.RECORDED,
        statistics,
        asdict(channel.transport),
        tuple(_packet(packet) for packet in channel.sent),
        tuple(_packet(packet) for packet in channel.received),
        channel.lost_sequences,
        tuple(
            PacketAttempt(
                event.sequence,
                event.attempt,
                event.departure,
                event.arrival,
                event.status,
                event.channel_state,
                event.duplicate,
                _packet(event.packet) if event.packet is not None else None,
            )
            for event in channel.events
        ),
        tuple(
            PacketTiming(
                sequence, channel.delays.get(sequence), channel.arrival_times.get(sequence)
            )
            for sequence in sorted(set(channel.delays) | set(channel.arrival_times))
        ),
    )


def _policy(
    declaration: Mapping[str, Any],
    capability: Mapping[str, Any],
    rows: tuple[Mapping[str, Any], ...],
) -> PolicyInspection:
    actions = capability.get("selected_actions")
    evaluated = tuple(index for index, row in enumerate(rows) if row.get("endpoint") is not None)
    if actions is not None and (
        not isinstance(actions, (list, tuple)) or len(actions) != len(evaluated)
    ):
        raise ValueError("selected policy actions must align with fully followed-up rows")
    by_row = dict(zip(evaluated, actions)) if actions is not None else {}
    decisions = []
    for index, row in enumerate(rows):
        available = row.get("available")
        if available is not None and type(available) is not bool:
            raise ValueError("row available must be a boolean or null")
        decisions.append(
            PolicyDecision(
                index,
                _time(row.get("acquisition_time_s"), "acquisition time"),
                _time(row.get("decision_time_s"), "decision time"),
                available,
                _text(row.get("endpoint_status"), "endpoint status"),
                _scalar(row.get("endpoint"), "endpoint"),
                _number(row.get("full_decision_statistic"), "full decision statistic"),
                _number(row.get("received_decision_statistic"), "received decision statistic"),
                _text(by_row.get(index), "selected action"),
                _text(row.get("fallback_action"), "fallback action"),
                _text(row.get("unavailability_reason"), "unavailability reason"),
            )
        )
    return PolicyInspection(
        declaration,
        {key: value for key, value in capability.items() if key != "selected_actions"},
        _text(capability.get("selected_policy"), "selected policy"),
        TraceStatus.RECORDED if actions is not None else TraceStatus.NOT_RECORDED,
        tuple(decisions),
    )


def inspect_experiment(observation: ExperimentObservation) -> ExperimentInspection:
    """Snapshot one run and project source-linked traces without recomputing science."""
    summary, manifest, receipt = _identity(observation)
    result = observation.result
    payload = result.to_dict()
    outputs = payload.pop("outputs")
    rows = _records(outputs.get("rows", ()), "outputs.rows")
    timeline: list[TimelineEvent] = []
    samples = []
    for index, row in enumerate(rows):
        for field, kind in (
            ("acquisition_time_s", "acquisition"),
            ("decision_time_s", "decision"),
            ("deadline_s", "deadline"),
            ("endpoint_time_s", "endpoint"),
        ):
            time = _time(row.get(field), field)
            if time is not None:
                timeline.append(
                    TimelineEvent(time, kind, index, f"result.outputs.rows[{index}]", row)
                )
        if any(
            key in row
            for key in ("representation", "quantized_representation", "received_representation")
        ):
            samples.append(
                RepresentationSample(
                    index,
                    _time(row.get("acquisition_time_s"), "acquisition time"),
                    _values(row.get("representation"), "representation"),
                    _values(row.get("quantized_representation"), "quantized representation"),
                    _values(row.get("received_representation"), "received representation"),
                    _mapping(row.get("representation_metadata", {}), "representation metadata"),
                )
            )
    resource = _mapping(outputs.get("resource", {}), "outputs.resource")
    transitions = []
    for index, record in enumerate(
        _records(resource.get("transitions", ()), "resource transitions")
    ):
        row_index = record.get("row")
        if row_index is not None and (type(row_index) is not int or not 0 <= row_index < len(rows)):
            raise ValueError("resource transition row must refer to an existing result row")
        time = _time(record.get("start_s"), "resource start")
        source = f"result.outputs.resource.transitions[{index}]"
        transitions.append(TraceRecord(index, row_index, time, source, record))
        if time is not None:
            timeline.append(TimelineEvent(time, "resource_transition", row_index, source, record))
    packets = _packets(observation.channel, _mapping(outputs.get("transport", {}), "transport"))
    for index, attempt in enumerate(packets.attempts):
        source = f"channel.events[{index}]"
        timeline.append(
            TimelineEvent(attempt.departure_s, "packet_departure", None, source, attempt.to_dict())
        )
        if attempt.arrival_s is not None:
            timeline.append(
                TimelineEvent(attempt.arrival_s, "packet_arrival", None, source, attempt.to_dict())
            )
    protocol = None
    if observation.protocol is not None:
        from physioworld.experiments.protocol import ExecutionResult

        if not isinstance(observation.protocol, ExecutionResult):
            raise TypeError("observation protocol must be an ExecutionResult")
        protocol = observation.protocol.to_dict()
        for index, record in enumerate(_records(protocol["chronology"], "protocol chronology")):
            time = _time(record.get("time_s"), "protocol time")
            if time is not None:
                timeline.append(
                    TimelineEvent(time, "protocol", None, f"protocol.chronology[{index}]", record)
                )
    return ExperimentInspection(
        summary,
        payload,
        outputs,
        # Stable sorting preserves source order for simultaneous events; IDs are not clocks.
        tuple(sorted(timeline, key=lambda event: event.time_s)),
        RepresentationInspection(
            result.representation,
            TraceStatus.RECORDED if samples else TraceStatus.NOT_RECORDED,
            tuple(samples),
        ),
        packets,
        ResourceInspection(
            result.resource,
            {key: value for key, value in resource.items() if key != "transitions"},
            TraceStatus.RECORDED if "transitions" in resource else TraceStatus.NOT_RECORDED,
            tuple(transitions),
        ),
        _policy(result.policy, _mapping(outputs.get("capability", {}), "capability"), rows),
        ProvenanceInspection(
            result.provenance,
            manifest.to_dict() if manifest else None,
            "PASS" if receipt else "NOT_VERIFIED",
            receipt.to_dict() if receipt else None,
            _records(protocol["provenance"], "protocol provenance") if protocol else (),
        ),
        _qualification(observation.qualification),
        protocol,
    )


def inspect_result(
    result: ExperimentResult,
    *,
    manifest: ResultManifest | Verification | None = None,
    channel: ChannelResult | None = None,
    protocol: ExecutionResult | None = None,
    qualification: PluginQualification | QualificationReport | Mapping[str, Any] | None = None,
    run_id: str | None = None,
) -> ExperimentInspection:
    """Convenience projection for an existing result and optional trace objects."""
    return inspect_experiment(
        ExperimentObservation(
            result,
            manifest=manifest,
            channel=channel,
            protocol=protocol,
            qualification=qualification,
            run_id=run_id,
        )
    )


def build_inspection(
    observations: Iterable[ExperimentObservation],
    *,
    selected_run_id: str | None = None,
) -> LabInspection:
    """Build a deterministic catalog and inspect the selected run.

    Catalog order and the default selection are lexicographic by run ID. Unknown
    selections and duplicate IDs raise instead of silently selecting another run.
    An empty catalog has no selection. Only the selected run's detailed traces
    are projected; the catalog does not merge experiments or compare estimands.
    """
    catalog: dict[str, tuple[ExperimentSummary, ExperimentObservation]] = {}
    for observation in observations:
        summary, _, _ = _identity(observation)
        if summary.run_id in catalog:
            raise ValueError(f"duplicate run ID: {summary.run_id!r}")
        catalog[summary.run_id] = (summary, observation)
    if selected_run_id is None and catalog:
        selected_run_id = min(catalog)
    if selected_run_id is not None and selected_run_id not in catalog:
        raise ValueError(f"unknown selected run ID: {selected_run_id!r}")
    return LabInspection(
        tuple(catalog[key][0] for key in sorted(catalog)),
        selected_run_id,
        inspect_experiment(catalog[selected_run_id][1]) if selected_run_id is not None else None,
    )


def render_inspection(
    inspection: LabInspection | ExperimentInspection,
    *,
    format: Literal["text", "json"] = "text",
) -> str:
    """Return a complete CLI-safe report with escaped terminal controls.

    JSON is deterministic, ASCII-escaped and strict (no NaN/Infinity). Text uses
    the same data with fixed section headings; all researcher-supplied strings
    remain JSON-escaped, including ANSI escapes, newlines and bidi controls.
    The caller owns stdout or report-file handling.
    """
    if not isinstance(inspection, (LabInspection, ExperimentInspection)):
        raise TypeError("renderer requires a LabInspection or ExperimentInspection")
    if format not in ("text", "json"):
        raise ValueError("inspection format must be 'text' or 'json'")

    def encode(value: Any) -> str:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)

    payload = inspection.to_dict()
    if format == "json":
        return encode(payload) + "\n"
    lines = ["Local lab inspection", ""]
    if isinstance(inspection, LabInspection):
        lines.extend(["Experiments", encode(payload["experiments"]), ""])
        lines.append("Selected run: " + encode(payload["selected_run_id"]))
        selected = payload["selected"]
    else:
        selected = payload
    if selected is None:
        lines.append("No experiment selected.")
    else:
        for key, title in (
            ("summary", "Experiment"),
            ("scientific_context", "Scientific context"),
            ("timeline", "Timeline (seconds)"),
            ("representations", "Representations"),
            ("packets", "Packet trace"),
            ("resources", "Resource trace"),
            ("policy", "Policy trace"),
            ("provenance", "Provenance and integrity"),
            ("qualification", "Qualification"),
            ("protocol", "Protocol execution"),
            ("outputs", "Recorded scientific outputs"),
        ):
            lines.extend(["", title, encode(selected[key])])
    return "\n".join(lines) + "\n"


__all__ = [
    "ExperimentObservation",
    "ExperimentSummary",
    "TraceStatus",
    "TraceRecord",
    "TimelineEvent",
    "RepresentationSample",
    "RepresentationInspection",
    "PacketSnapshot",
    "PacketAttempt",
    "PacketTiming",
    "PacketInspection",
    "ResourceInspection",
    "PolicyDecision",
    "PolicyInspection",
    "QualificationInspection",
    "ProvenanceInspection",
    "ExperimentInspection",
    "LabInspection",
    "inspect_experiment",
    "inspect_result",
    "build_inspection",
    "render_inspection",
]

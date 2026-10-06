"""Deterministic split compilation with independent-unit and chronology guards.

All temporal quantities are seconds in an explicitly shared clock_basis. Label
times describe availability (not outcome occurrence); fitted artifacts identify
their complete training population. No data or assignments are silently removed.
Window bounds are inclusive acquisition bounds: sharing an endpoint sample
across partitions is leakage as well as sharing an interval.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from physioworld.io.canonical import pack
from physioworld.physiology.record import PhysiologicalRecord

AXES = {
    "subject": "subject_id",
    "session": "session_id",
    "record": "record_id",
    "site": "site_id",
    "device": "device_id",
}
Identity = str | tuple[str, str]


@dataclass(frozen=True)
class SplitItem:
    record_id: str
    subject_id: str
    session_id: str | None = None
    site_id: str | None = None
    device_id: str | None = None
    content_hash: str | None = None
    source_record_id: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    prediction_time_s: float | None = None
    input_available_s: float | None = None
    label_available_s: float | None = None
    clock_basis: str | None = None
    labels_used_as_features: bool = False

    def __post_init__(self):
        if not self.record_id or not self.subject_id:
            raise ValueError("split rows require verified record and subject identities")
        times = (
            self.start_s,
            self.end_s,
            self.prediction_time_s,
            self.input_available_s,
            self.label_available_s,
        )
        if any(t is not None and not math.isfinite(t) for t in times):
            raise ValueError("split times must be finite seconds")
        if self.start_s is not None and self.end_s is not None and self.start_s > self.end_s:
            raise ValueError("split window bounds reversed")
        if any(t is not None for t in times) and not self.clock_basis:
            raise ValueError("temporal split metadata requires an explicit shared clock_basis")

    @classmethod
    def from_record(cls, record: PhysiologicalRecord, **temporal):
        # Content identity intentionally excludes renamed record/subject/provenance.
        content = {
            name: {
                "timestamps": s.timestamps,
                "values": s.values,
                "channels": tuple((c.name, c.unit) for c in s.metadata.channels),
                "time_basis": s.metadata.clock.time_basis,
                "clock_domain": s.metadata.clock.clock_domain,
                "missing": s.metadata.missing,
                "missing_channels": s.metadata.missing_channels,
            }
            for name, s in record.streams.items()
        }
        if record.images:
            content["__imaging_pixels__"] = {
                name: image["pixels"] for name, image in record.images.items() if "pixels" in image
            }
        digest = (
            hashlib.sha256(
                json.dumps(pack(content), sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if record.streams
            else None
        )
        devices = {s.metadata.device_id for s in record.streams.values() if s.metadata.device_id}
        device = record.device_id or (next(iter(devices)) if len(devices) == 1 else None)
        sites = {s.metadata.site_id for s in record.streams.values() if s.metadata.site_id}
        site = record.site_id or (next(iter(sites)) if len(sites) == 1 else None)
        return cls(
            record.record_id,
            record.subject_id,
            record.session_id,
            site,
            device,
            content_hash=digest,
            **temporal,
        )


@dataclass(frozen=True)
class PreprocessingArtifact:
    name: str
    fit_record_ids: tuple[str, ...]
    available_at_s: float | None = None
    input_end_s: float | None = None
    clock_basis: str | None = None
    uses_labels: bool = False
    causal: bool = True

    def __post_init__(self):
        if not self.name:
            raise ValueError("preprocessing artifacts require a name")
        if any(
            t is not None and not math.isfinite(t) for t in (self.available_at_s, self.input_end_s)
        ):
            raise ValueError("preprocessing times must be finite seconds")
        if (
            self.available_at_s is not None or self.input_end_s is not None
        ) and not self.clock_basis:
            raise ValueError("preprocessing times require an explicit clock basis")


@dataclass(frozen=True)
class SplitSpec:
    independent_unit: str
    strategy: str = "group"  # group, temporal, prospective, domain
    protected_axes: tuple[str, ...] = ()
    train_fraction: float = 0.8
    seed: int = 0
    cutoff_s: float | None = None
    embargo_s: float = 0.0
    holdout_values: tuple[str, ...] = ()
    require_content_hash: bool = False
    domain_axis: str | None = None  # may differ from the estimand's independent unit
    require_temporal_availability: bool = False

    def __post_init__(self):
        if self.independent_unit not in AXES or any(a not in AXES for a in self.protected_axes):
            raise ValueError(
                "independent/protected axes must be subject/session/record/site/device"
            )
        if self.domain_axis is not None and self.domain_axis not in AXES:
            raise ValueError("domain_axis must be subject/session/record/site/device")
        if self.strategy not in {"group", "temporal", "prospective", "domain"}:
            raise ValueError("unsupported split strategy")
        if not math.isfinite(self.train_fraction) or not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0,1)")
        if not math.isfinite(self.embargo_s) or self.embargo_s < 0:
            raise ValueError("embargo must be finite non-negative seconds")
        if self.strategy in {"temporal", "prospective"} and (
            self.cutoff_s is None or not math.isfinite(self.cutoff_s)
        ):
            raise ValueError("temporal/prospective split requires finite cutoff_s")
        if self.strategy == "domain" and not self.holdout_values:
            raise ValueError("domain split requires explicit holdout_values")


@dataclass(frozen=True)
class LeakageIssue:
    code: str
    record_ids: tuple[str, ...]
    message: str


class SplitLeakageError(ValueError):
    def __init__(self, issues: Sequence[LeakageIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{i.code}: {i.message}" for i in issues))


@dataclass(frozen=True)
class CompiledSplit:
    assignments: Mapping[str, str]
    spec: SplitSpec
    configuration_hash: str
    independent_counts: Mapping[str, int] = field(default_factory=dict)
    content_audit: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self):
        return {
            "schema_version": "1.0.0",
            "assignments": dict(self.assignments),
            "spec": asdict(self.spec),
            "configuration_hash": self.configuration_hash,
            "independent_counts": dict(self.independent_counts),
            "content_audit": dict(self.content_audit),
        }


def _identity(item: SplitItem, axis: str) -> Identity:
    value: str | None = getattr(item, AXES[axis])
    if value is None or value == "":
        raise ValueError(f"missing {axis} identity for {item.record_id}")
    return (item.subject_id, value) if axis == "session" else value


class SplitCompiler:
    def compile_records(
        self,
        records: Sequence[PhysiologicalRecord],
        spec: SplitSpec,
        *,
        temporal_metadata: Mapping[str, Mapping[str, object]] | None = None,
        assignments: Mapping[str, str] | None = None,
        preprocessing: Sequence[PreprocessingArtifact] = (),
    ) -> CompiledSplit:
        temporal = temporal_metadata or {}
        if set(temporal) - {r.record_id for r in records}:
            raise ValueError("temporal metadata refers to unknown records")
        items = [
            SplitItem.from_record(record, **temporal.get(record.record_id, {}))
            for record in records
        ]
        return self.compile(items, spec, assignments=assignments, preprocessing=preprocessing)

    def compile(
        self,
        items: Sequence[SplitItem],
        spec: SplitSpec,
        *,
        assignments: Mapping[str, str] | None = None,
        preprocessing: Sequence[PreprocessingArtifact] = (),
    ) -> CompiledSplit:
        if not items:
            raise ValueError("split requires records")
        if len({i.record_id for i in items}) != len(items):
            raise SplitLeakageError(
                (LeakageIssue("duplicate_record", (), "record IDs must be unique"),)
            )
        if assignments is None:
            assignments = {}
            if spec.strategy == "group":
                # Union connected independent/protected units before randomization.
                parent = {i.record_id: i.record_id for i in items}

                def root(key):
                    while parent[key] != key:
                        parent[key] = parent[parent[key]]
                        key = parent[key]
                    return key

                for axis in (spec.independent_unit, *spec.protected_axes):
                    seen: dict[Identity, str] = {}
                    for item in items:
                        key = _identity(item, axis)
                        if key in seen:
                            a, b = root(item.record_id), root(seen[key])
                            parent[max(a, b)] = min(a, b)
                        else:
                            seen[key] = item.record_id
                clusters = sorted({root(i.record_id) for i in items})
                if len(clusters) < 2:
                    raise ValueError("at least two independent leakage-safe groups are required")
                random.Random(spec.seed).shuffle(clusters)
                n = max(1, min(len(clusters) - 1, int(len(clusters) * spec.train_fraction)))
                train = set(clusters[:n])
                assignments = {
                    i.record_id: "train" if root(i.record_id) in train else "test" for i in items
                }
            elif spec.strategy == "domain":
                assignments = {
                    i.record_id: "test"
                    if str(_identity(i, spec.domain_axis or spec.independent_unit))
                    in spec.holdout_values
                    else "train"
                    for i in items
                }
            else:
                cutoff = spec.cutoff_s
                if cutoff is None:
                    raise ValueError("temporal/prospective split requires finite cutoff_s")
                for item in items:
                    if item.start_s is None or item.end_s is None:
                        raise ValueError("temporal split requires start_s/end_s on every record")
                    if item.end_s <= cutoff - spec.embargo_s:
                        assignments[item.record_id] = "train"
                    elif item.start_s >= cutoff + spec.embargo_s:
                        assignments[item.record_id] = "test"
                    else:
                        raise SplitLeakageError(
                            (
                                LeakageIssue(
                                    "cutoff_overlap",
                                    (item.record_id,),
                                    "window crosses cutoff/embargo; declare exclusion upstream",
                                ),
                            )
                        )
        assignments = dict(assignments)
        issues = self.validate(items, spec, assignments, preprocessing=preprocessing)
        if issues:
            raise SplitLeakageError(issues)
        counts = {
            part: len(
                {
                    _identity(i, spec.independent_unit)
                    for i in items
                    if assignments[i.record_id] == part
                }
            )
            for part in set(assignments.values())
        }
        configuration = {
            "items": [asdict(i) for i in sorted(items, key=lambda i: i.record_id)],
            "spec": asdict(spec),
            "assignments": assignments,
            "preprocessing": [asdict(p) for p in preprocessing],
        }
        digest = hashlib.sha256(
            json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        hashed = sum(i.content_hash is not None for i in items)
        return CompiledSplit(
            assignments,
            spec,
            digest,
            counts,
            {
                "hashed_records": hashed,
                "total_records": len(items),
                "status": "complete" if hashed == len(items) else "identity_only_or_partial",
            },
        )

    def validate(self, items, spec, assignments, *, preprocessing=()) -> tuple[LeakageIssue, ...]:
        issues = []
        ids = {i.record_id for i in items}
        if set(assignments) != ids or any(
            v not in {"train", "validation", "test"} for v in assignments.values()
        ):
            return (
                LeakageIssue(
                    "invalid_assignments", (), "each record must appear once in a named partition"
                ),
            )
        if "train" not in assignments.values() or not (
            {"test", "validation"} & set(assignments.values())
        ):
            issues.append(
                LeakageIssue(
                    "empty_partition", (), "training and evaluation partitions are required"
                )
            )
        for axis in dict.fromkeys((spec.independent_unit, *spec.protected_axes)):
            seen: dict[Identity, str] = {}
            for item in items:
                key = _identity(item, axis)
                previous = seen.get(key)
                if previous and assignments[previous] != assignments[item.record_id]:
                    issues.append(
                        LeakageIssue(
                            f"{axis}_overlap",
                            (previous, item.record_id),
                            f"shared {axis} crosses partitions",
                        )
                    )
                seen[key] = item.record_id
        if spec.strategy == "domain":
            for item in items:
                expected = (
                    "test"
                    if str(_identity(item, spec.domain_axis or spec.independent_unit))
                    in spec.holdout_values
                    else "train"
                )
                if assignments[item.record_id] != expected:
                    issues.append(
                        LeakageIssue(
                            "domain_contamination",
                            (item.record_id,),
                            "assignment violates explicit domain holdout",
                        )
                    )
        hashes: dict[str, str] = {}
        for item in items:
            if spec.require_content_hash and not item.content_hash:
                issues.append(
                    LeakageIssue(
                        "missing_content_hash",
                        (item.record_id,),
                        "content duplicate audit requires a hash",
                    )
                )
            if item.content_hash:
                if item.content_hash in hashes:
                    issues.append(
                        LeakageIssue(
                            "duplicate_content",
                            (hashes[item.content_hash], item.record_id),
                            "duplicate samples under distinct IDs",
                        )
                    )
                hashes[item.content_hash] = item.record_id
            if item.labels_used_as_features:
                issues.append(
                    LeakageIssue("label_leakage", (item.record_id,), "target label used as feature")
                )
            if item.prediction_time_s is not None:
                for time in (item.end_s, item.input_available_s):
                    if time is not None and time > item.prediction_time_s:
                        issues.append(
                            LeakageIssue(
                                "future_input",
                                (item.record_id,),
                                "input acquired/available after prediction",
                            )
                        )
        timed = [i for i in items if i.start_s is not None and i.end_s is not None]
        if spec.strategy in {"temporal", "prospective"}:
            cutoff = spec.cutoff_s
            if cutoff is None:
                raise ValueError("temporal/prospective split requires finite cutoff_s")
            if len(timed) != len(items) or len({i.clock_basis for i in timed}) != 1:
                issues.append(
                    LeakageIssue(
                        "incomparable_clocks",
                        (),
                        "temporal partitions require one explicit comparable clock basis",
                    )
                )
            for item in timed:
                part = assignments[item.record_id]
                if (part == "train" and item.end_s > cutoff - spec.embargo_s) or (
                    part != "train" and item.start_s < cutoff + spec.embargo_s
                ):
                    issues.append(
                        LeakageIssue(
                            "future_partition",
                            (item.record_id,),
                            "partition violates cutoff/embargo",
                        )
                    )
                if (
                    part == "train"
                    and item.label_available_s is not None
                    and item.label_available_s > cutoff
                ):
                    issues.append(
                        LeakageIssue(
                            "future_label",
                            (item.record_id,),
                            "training label unavailable at cutoff",
                        )
                    )
                if spec.require_temporal_availability and (
                    item.prediction_time_s is None
                    or item.input_available_s is None
                    or item.label_available_s is None
                ):
                    issues.append(
                        LeakageIssue(
                            "missing_temporal_availability",
                            (item.record_id,),
                            "temporal/prospective supervised splits require prediction, input-availability, and label-availability times",
                        )
                    )
        # Interval overlaps are meaningful only within the same source recording/clock.
        windows: dict[tuple[str, str, str | None], list[SplitItem]] = {}
        for item in timed:
            window_key = (
                item.subject_id,
                item.source_record_id or item.record_id,
                item.clock_basis,
            )
            windows.setdefault(window_key, []).append(item)
        for cluster in windows.values():
            active: list[SplitItem] = []
            for item in sorted(cluster, key=lambda i: i.start_s if i.start_s is not None else 0.0):
                active = [
                    p
                    for p in active
                    if p.end_s is not None and item.start_s is not None and p.end_s >= item.start_s
                ]
                for prior_item in active:
                    if assignments[prior_item.record_id] != assignments[item.record_id]:
                        issues.append(
                            LeakageIssue(
                                "time_overlap",
                                (prior_item.record_id, item.record_id),
                                "source windows overlap across partitions",
                            )
                        )
                active.append(item)
        train = {rid for rid, part in assignments.items() if part == "train"}
        for artifact in preprocessing:
            contaminated = set(artifact.fit_record_ids) - train
            if contaminated:
                issues.append(
                    LeakageIssue(
                        "preprocessing_contamination",
                        tuple(sorted(contaminated)),
                        f"{artifact.name} fitted outside training",
                    )
                )
            if not artifact.causal:
                issues.append(
                    LeakageIssue(
                        "future_preprocessing", (), f"{artifact.name} uses noncausal inputs"
                    )
                )
            if spec.strategy in {"temporal", "prospective"}:
                cutoff = spec.cutoff_s
                if cutoff is None:
                    raise ValueError("temporal/prospective split requires finite cutoff_s")
                basis = {i.clock_basis for i in items}
                for time in (artifact.available_at_s, artifact.input_end_s):
                    if time is not None and (time > cutoff or artifact.clock_basis not in basis):
                        issues.append(
                            LeakageIssue(
                                "future_preprocessing",
                                (),
                                f"{artifact.name} fitted after cutoff/incomparable clock",
                            )
                        )
            for item in items:
                deadline = item.prediction_time_s
                if deadline is not None:
                    for time in (artifact.available_at_s, artifact.input_end_s):
                        if time is not None and (
                            artifact.clock_basis != item.clock_basis or time > deadline
                        ):
                            issues.append(
                                LeakageIssue(
                                    "future_preprocessing",
                                    (item.record_id,),
                                    f"{artifact.name} unavailable at prediction/incomparable clock",
                                )
                            )
            if artifact.uses_labels and not artifact.fit_record_ids:
                issues.append(
                    LeakageIssue(
                        "label_preprocessing",
                        (),
                        "supervised preprocessing requires declared training records",
                    )
                )
            if artifact.uses_labels and artifact.available_at_s is not None:
                for item in items:
                    if (
                        item.record_id in artifact.fit_record_ids
                        and item.label_available_s is not None
                        and (
                            item.clock_basis != artifact.clock_basis
                            or item.label_available_s > artifact.available_at_s
                        )
                    ):
                        issues.append(
                            LeakageIssue(
                                "future_label_preprocessing",
                                (item.record_id,),
                                "label unavailable when supervised preprocessing fitted",
                            )
                        )
        return tuple(issues)

"""Evidence-origin rules shared by executable sources and estimators.

Evidence classes describe how a result is supported; they are not a free-form
claim that can be upgraded by relabeling a fixture.  In particular, replayed
analytic fixtures are numerical verification, while observed/external sources
need a separately verified artifact receipt.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from physioworld.core.contracts import EvidenceClass

_ARTIFACT_EVIDENCE = frozenset(
    {
        EvidenceClass.EMPIRICAL_ESTIMATE,
        EvidenceClass.EXTERNAL_VALIDATION,
        EvidenceClass.CLINICAL_EVIDENCE,
    }
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_INLINE_MARKERS = ("analytic", "fixture", "synthetic")


def _artifact_record(metadata: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    artifact = metadata.get("artifact_metadata", metadata.get("artifact"))
    return artifact if isinstance(artifact, Mapping) else None


def has_verified_artifact_metadata(metadata: Mapping[str, Any] | None) -> bool:
    """Return whether metadata contains a portable, verified artifact receipt."""
    artifact = _artifact_record(metadata)
    if artifact is None:
        return False
    artifact_id = artifact.get("artifact_id", artifact.get("id"))
    digest = artifact.get("sha256")
    verification = artifact.get("verified") is True and (
        "verification_status" not in artifact or artifact.get("verification_status") == "VERIFIED"
    )
    return (
        isinstance(artifact_id, str)
        and bool(artifact_id.strip())
        and isinstance(digest, str)
        and bool(_SHA256.fullmatch(digest))
        and verification
    )


def require_verified_artifact_metadata(
    evidence_class: EvidenceClass, metadata: Mapping[str, Any] | None
) -> None:
    """Reject empirical, external, or clinical evidence without a receipt."""
    if evidence_class in _ARTIFACT_EVIDENCE and not has_verified_artifact_metadata(metadata):
        raise ValueError(f"{evidence_class.value} evidence requires verified artifact metadata")


def source_evidence_origin(dataset: Mapping[str, Any]) -> str:
    """Resolve a source's explicit evidence origin without inferring observation.

    Existing built-in replay declarations use descriptive inline fixture names,
    which remain numerical verification.  Other replay sources must opt into
    ``OBSERVED_REPLAY`` (or another external/human origin) explicitly, keeping
    the observed-replay extension boundary visible and auditable.
    """
    declared = dataset.get("evidence_origin")
    if dataset.get("origin") == "SYNTHETIC":
        if declared is not None:
            raise ValueError(
                "synthetic sources cannot declare an observed/external evidence_origin"
            )
        return "SYNTHETIC_MODEL"
    if dataset.get("origin") != "REPLAY":
        raise ValueError("unsupported dataset origin for evidence classification")
    if declared is not None:
        if not isinstance(declared, str) or not declared:
            raise ValueError("dataset.evidence_origin must be a non-empty string")
        return declared
    source = dataset.get("source")
    if isinstance(source, str):
        normalized = source.casefold()
        if any(marker in normalized for marker in _INLINE_MARKERS):
            return "INLINE_ANALYTIC_REPLAY"
    raise ValueError(
        "REPLAY source must explicitly declare evidence_origin; "
        "observed replay is a separate extension requiring artifact metadata"
    )


def source_evidence_class(
    dataset: Mapping[str, Any], *, evidence_metadata: Mapping[str, Any] | None = None
) -> tuple[str, EvidenceClass]:
    """Return the source origin and the highest evidence class it may carry."""
    origin = source_evidence_origin(dataset)
    if origin == "SYNTHETIC_MODEL":
        return origin, EvidenceClass.SIMULATION
    if origin == "INLINE_ANALYTIC_REPLAY":
        return origin, EvidenceClass.NUMERICAL_VERIFICATION
    if origin == "OBSERVED_REPLAY":
        evidence_class = EvidenceClass.EMPIRICAL_ESTIMATE
    elif origin == "EXTERNAL_DATASET":
        evidence_class = EvidenceClass.EXTERNAL_VALIDATION
    elif origin == "HUMAN_OBSERVATION":
        evidence_class = EvidenceClass.EMPIRICAL_ESTIMATE
    else:
        raise ValueError(f"unsupported evidence_origin {origin!r}")
    metadata = evidence_metadata if evidence_metadata is not None else dataset
    require_verified_artifact_metadata(evidence_class, metadata)
    return origin, evidence_class


def estimator_evidence_metadata(
    origin: str, evidence_class: EvidenceClass, metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build the small provenance payload propagated into task-law records."""
    result: dict[str, Any] = {"evidence_origin": origin}
    if metadata is not None and "artifact_metadata" in metadata:
        result["artifact_metadata"] = metadata["artifact_metadata"]
    require_verified_artifact_metadata(evidence_class, result)
    return result

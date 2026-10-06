"""Adapters enforce access boundaries and keep dataset logic out of core."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .registry import DatasetAccess, DatasetSpec


class DatasetAdapter:
    spec: DatasetSpec

    def validate(self, path: str | Path | None = None) -> dict[str, Any]:
        raise NotImplementedError


class LocalFileAdapter(DatasetAdapter):
    def __init__(self, spec: DatasetSpec, path: str | Path | None = None):
        self.spec = spec
        self.path = Path(path) if path is not None else None

    def validate(self, path: str | Path | None = None) -> dict[str, Any]:
        candidate = Path(path) if path is not None else self.path
        result = {
            "dataset_id": self.spec.dataset_id,
            "access": self.spec.access.value,
            "source_url": self.spec.source_url,
            "path_supplied": candidate is not None,
            "exists": candidate.exists() if candidate else False,
            "redistributable": False,  # Access class alone cannot grant redistribution.
            "status": "PRESENT" if candidate and candidate.exists() else "WARNING",
            "validation_scope": (
                "path presence only; schema, units, checksum, identity and leakage "
                "require a format-specific adapter and declared records"
            ),
        }
        if self.spec.access != DatasetAccess.OPEN:
            result["instruction"] = self.spec.acquisition
        return result


class LocalWFDBAdapter(LocalFileAdapter):
    """Authorized local loading. This adapter never asks for/stores credentials."""

    def validate(self, path: str | Path | None = None) -> dict[str, Any]:
        candidate = Path(path) if path is not None else self.path
        result = super().validate(candidate)
        header_exists = candidate is not None and Path(str(candidate) + ".hea").is_file()
        result.update(
            exists=header_exists,
            status="PASS" if header_exists else "WARNING",
            validation_scope="local WFDB header presence; read() validates samples/units/rate",
            expected_schema=dict(self.spec.expected_schema),
        )
        return result

    def read(
        self,
        *,
        subject_id: str,
        record_id: str,
        session_id: str | None = None,
        authorized: bool = False,
        annotation_extension: str | None = None,
        modality: str = "unknown",
    ):
        from physioworld.io.wfdb import WFDBAdapter

        if not subject_id or not record_id or self.path is None:
            raise ValueError("local WFDB requires path and verified subject/record identity")
        if self.spec.access != DatasetAccess.OPEN and not authorized:
            raise PermissionError(
                "restricted/credentialed dataset requires explicit local provider authorization"
            )
        if self.spec.version == "provider-defined":
            raise ValueError("bind a concrete DatasetSpec.version before reading samples")
        return WFDBAdapter().read(
            self.path,
            subject_id=subject_id,
            record_id=record_id,
            session_id=session_id,
            access_class=self.spec.access.value,
            authorized=authorized,
            annotation_extension=annotation_extension,
            modality=modality,
            source_dataset=self.spec.dataset_id,
            dataset_version=self.spec.version,
        )


def local_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

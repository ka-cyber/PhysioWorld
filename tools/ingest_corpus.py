"""Build a metadata-only, recursively hashed corpus inventory.

The public output is deliberately compact.  A detailed, path-level inventory is
written outside the repository by default.  Archives are inspected with
``zipfile``/``tarfile`` without extracting any member to disk; size, path, and
nested-archive limits are enforced before member bytes are read.

This tool uses only the Python standard library so that corpus provenance can be
checked before the optional PhysioWorld dependencies are installed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import stat
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Protocol

SCHEMA_VERSION = "1.0.0"
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_ENTRY_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_NESTED_ARCHIVE_DEPTH = 5
TEXT_SUFFIXES = {
    ".c",
    ".cff",
    ".cfg",
    ".csv",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sample",
    ".tag",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
EMPIRICAL_SUFFIXES = {
    ".arrow",
    ".atr",
    ".dat",
    ".edf",
    ".h5",
    ".hdf5",
    ".hea",
    ".mat",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pickle",
    ".rec",
    ".wfdb",
}
ARCHIVE_SUFFIXES = {".7z", ".gz", ".rar", ".tar", ".tgz", ".zip"}
PRIVATE_MARKERS = {
    "agent",
    "chatlog",
    "chatter",
    "private",
    "planning",
    "raw_log",
    "raw_logs",
    "transcript",
    "transcripts",
}
SKIP_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
}
GENERATED_DIRECTORY_NAMES = {
    ".cache",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
VERSION_RE = re.compile(
    r"(?i)\bversion\s*(?:[:=]|is|-)\s*[\"']?([0-9]+(?:\.[0-9]+){1,3}(?:[-+][A-Za-z0-9.-]+)?)"
)
DATE_RE = re.compile(r"\b(20[0-9]{2}-[01][0-9]-[0-3][0-9])\b")


def canonical(value: Any) -> bytes:
    """Serialize metadata deterministically for digest construction."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> tuple[str, int, str | None]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    return "", size, "FILE_TOO_LARGE"
                digest.update(chunk)
    except (OSError, ValueError) as exc:
        return "", size, type(exc).__name__
    return digest.hexdigest(), size, None


def relative_path(path: Path, root: Path) -> str:
    """Return a portable, root-relative logical path; never an absolute path."""

    return path.relative_to(root).as_posix()


def safe_archive_path(name: str) -> tuple[str | None, str | None]:
    """Normalize an archive member name and reject traversal or absolute paths."""

    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        return None, "ABSOLUTE_MEMBER_PATH"
    parts = [part for part in PurePosixPath(normalized).parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return None, "TRAVERSAL_MEMBER_PATH"
    result = "/".join(parts)
    return result or None, None


def is_private_path(logical_path: str) -> bool:
    parts = {part.lower() for part in PurePosixPath(logical_path).parts}
    name = PurePosixPath(logical_path).name.lower()
    return bool(parts & PRIVATE_MARKERS) or any(marker in name for marker in PRIVATE_MARKERS)


def is_generated_path(logical_path: str) -> bool:
    parts = {part.lower() for part in PurePosixPath(logical_path).parts}
    return (
        bool(parts & (GENERATED_DIRECTORY_NAMES | {".pyc"}))
        or any(part.endswith(".egg-info") for part in parts)
        or logical_path.endswith(".pyc")
    )


def classify_path(logical_path: str, *, archive: bool = False) -> str:
    lower = logical_path.lower()
    suffix = PurePosixPath(lower).suffix
    if is_private_path(logical_path):
        return "private_or_agent_material"
    if is_generated_path(logical_path) or suffix in {".db", ".pyc"} and ".cache/" in lower:
        return "generated_cache"
    if "/.git/hooks/" in f"/{lower}" or suffix == ".sample":
        return "version_control_template"
    if "datasets/golden/" in lower or "/corpus/golden/" in lower or "/test_data/" in lower:
        return "synthetic_fixture"
    if suffix in EMPIRICAL_SUFFIXES:
        return "empirical_data_candidate"
    if suffix == ".py":
        return "source_code"
    if suffix in {".md", ".rst"}:
        return "documentation"
    if suffix in {".json", ".yaml", ".yml", ".toml", ".cff", ".ini", ".cfg"}:
        if "registry" in lower or "ledger" in lower:
            return "project_registry_or_ledger"
        return "configuration_or_metadata"
    if suffix in TEXT_SUFFIXES:
        return "readable_text"
    if archive or suffix in ARCHIVE_SUFFIXES:
        return "archive"
    return "binary_or_unclassified"


def evidence_for_class(kind: str, logical_path: str) -> str:
    if kind == "synthetic_fixture":
        return "SIMULATION"
    if kind in {"source_code", "project_registry_or_ledger"}:
        return "SOFTWARE_VERIFICATION"
    if kind == "documentation":
        if "scientific-context" in logical_path or "prior_art" in logical_path:
            return "HISTORICAL_CONTEXT"
        return "EXTRACTED_SPECIFICATION"
    if kind == "empirical_data_candidate":
        return "EMPIRICAL_ESTIMATE"
    if kind in {"generated_cache", "version_control_template"}:
        return "NON_EVIDENCE_ARTIFACT"
    return "METADATA"


def text_metadata(data: bytes) -> dict[str, list[str]]:
    """Extract only dates/versions; never return source text."""

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return {"versions": [], "dates": []}
    versions = sorted(set(VERSION_RE.findall(text)))
    dates = sorted(set(DATE_RE.findall(text)))
    return {"versions": versions[:12], "dates": dates[:12]}


def read_text_metadata(
    path_or_stream: Path | BinaryIO, size: int | None = None
) -> dict[str, list[str]]:
    if size is not None and size > 2 * 1024 * 1024:
        return {"versions": [], "dates": []}
    try:
        if isinstance(path_or_stream, Path):
            data = path_or_stream.read_bytes()
        else:
            data = path_or_stream.read(2 * 1024 * 1024 + 1)
    except (OSError, ValueError):
        return {"versions": [], "dates": []}
    return text_metadata(data)


def tree_digest(records: Iterable[dict[str, Any]]) -> str:
    rows = []
    for record in records:
        rows.append(
            {
                "path": record.get("logical_path"),
                "sha256": record.get("sha256"),
                "bytes": record.get("bytes"),
                "status": record.get("hash_status"),
            }
        )
    rows.sort(key=lambda row: row["path"] or "")
    return sha256_bytes(canonical(rows))


class _BinaryReader(Protocol):
    def read(self, size: int = -1) -> bytes: ...


def _read_limited(handle: _BinaryReader, limit: int) -> tuple[bytes, str | None]:
    chunks: list[bytes] = []
    total = 0
    while chunk := handle.read(1024 * 1024):
        total += len(chunk)
        if total > limit:
            return b"", "READ_LIMIT_EXCEEDED"
        chunks.append(chunk)
    return b"".join(chunks), None


class ArchiveScanner:
    """Inspect archive members in memory and retain metadata, never extracted files."""

    def __init__(self) -> None:
        self.inspections: list[dict[str, Any]] = []
        self.entries: list[dict[str, Any]] = []
        self._archive_counter = 0

    def scan_file(self, path: Path, archive_id: str) -> dict[str, Any]:
        try:
            with zipfile.ZipFile(path) as archive:
                return self._scan_zip(archive, archive_id, depth=0, source="workspace")
        except (OSError, zipfile.BadZipFile) as exc:
            summary = self._failed_summary(archive_id, type(exc).__name__)
            self.inspections.append(summary)
            return summary

    def _failed_summary(self, logical_archive: str, status: str) -> dict[str, Any]:
        return {
            "logical_archive": logical_archive,
            "archive_kind": "zip",
            "depth": logical_archive.count("!") + 0,
            "status": status,
            "entry_count": 0,
            "entry_tree_sha256": sha256_bytes(b"[]"),
            "nested_archive_count": 0,
            "readable_source_count": 0,
            "empirical_candidate_count": 0,
            "entries": [],
        }

    def _scan_zip(
        self, archive: zipfile.ZipFile, logical_archive: str, *, depth: int, source: str
    ) -> dict[str, Any]:
        entry_records: list[dict[str, Any]] = []
        expanded = 0
        nested_count = 0
        for index, info in enumerate(archive.infolist()):
            member, path_error = safe_archive_path(info.filename)
            is_dir = info.is_dir() or info.filename.endswith(("/", "\\"))
            record: dict[str, Any] = {
                "archive": logical_archive,
                "index": index,
                "member_path": member or info.filename,
                "original_member_name": info.filename,
                "bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "is_directory": is_dir,
                "compression": info.compress_type,
                "hash_status": "NOT_READ",
                "sha256": None,
                "classification": classify_path(member or info.filename, archive=True),
                "nested_archive": None,
            }
            if path_error:
                record["hash_status"] = path_error
                entry_records.append(record)
                continue
            mode = (info.external_attr >> 16) & 0o170000
            if mode and stat.S_ISLNK(mode):
                record["hash_status"] = "UNSAFE_SYMLINK_MEMBER"
                entry_records.append(record)
                continue
            if is_dir:
                record["hash_status"] = "DIRECTORY"
                entry_records.append(record)
                continue
            if info.file_size > MAX_ARCHIVE_ENTRY_BYTES:
                record["hash_status"] = "ENTRY_TOO_LARGE"
                entry_records.append(record)
                continue
            if expanded + info.file_size > MAX_ARCHIVE_EXPANDED_BYTES:
                record["hash_status"] = "ARCHIVE_EXPANSION_LIMIT"
                entry_records.append(record)
                continue
            try:
                with archive.open(info, "r") as handle:
                    data, error = _read_limited(handle, MAX_ARCHIVE_ENTRY_BYTES)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                data, error = b"", type(exc).__name__
            if error:
                record["hash_status"] = error
                entry_records.append(record)
                continue
            expanded += len(data)
            record["bytes"] = len(data)
            record["sha256"] = sha256_bytes(data)
            record["hash_status"] = "HASHED"
            if record["classification"] not in {"generated_cache", "version_control_template"}:
                if record["classification"] in {"source_code", "documentation", "readable_text"}:
                    record["readable"] = True
            metadata = (
                text_metadata(data)
                if PurePosixPath(member or info.filename).suffix.lower() in TEXT_SUFFIXES
                else {}
            )
            if metadata:
                record["date_version"] = metadata
            if depth < MAX_NESTED_ARCHIVE_DEPTH and self._is_nested_archive(member or "", data):
                nested_count += 1
                nested_name = f"{logical_archive}!{member}"
                record["nested_archive"] = nested_name
                self._scan_nested(data, nested_name, depth + 1, source)
            entry_records.append(record)
        digest_rows = [
            {
                "index": entry["index"],
                "member_path": entry["member_path"],
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "hash_status": entry["hash_status"],
            }
            for entry in entry_records
        ]
        summary = {
            "logical_archive": logical_archive,
            "archive_kind": "zip",
            "depth": depth,
            "source": source,
            "status": "INSPECTED",
            "entry_count": len(entry_records),
            "expanded_bytes_read": expanded,
            "entry_tree_sha256": sha256_bytes(canonical(digest_rows)),
            "nested_archive_count": nested_count,
            "readable_source_count": sum(
                entry.get("classification") in {"source_code", "documentation", "readable_text"}
                and entry.get("hash_status") == "HASHED"
                for entry in entry_records
            ),
            "empirical_candidate_count": sum(
                entry.get("classification") == "empirical_data_candidate"
                and entry.get("hash_status") == "HASHED"
                for entry in entry_records
            ),
            "entries": entry_records,
        }
        self.inspections.append(summary)
        self.entries.extend(entry_records)
        return summary

    @staticmethod
    def _is_nested_archive(member: str, data: bytes) -> bool:
        suffix = PurePosixPath(member.lower()).suffix
        return suffix in ARCHIVE_SUFFIXES or data.startswith((b"PK\x03\x04", b"ustar"))

    def _scan_nested(self, data: bytes, logical_archive: str, depth: int, source: str) -> None:
        stream = io.BytesIO(data)
        try:
            if zipfile.is_zipfile(stream):
                with zipfile.ZipFile(stream) as archive:
                    self._scan_zip(archive, logical_archive, depth=depth, source=source)
                return
        except (OSError, RuntimeError, zipfile.BadZipFile):
            pass
        stream.seek(0)
        try:
            if tarfile.is_tarfile(stream):
                self._scan_tar(stream, logical_archive, depth=depth, source=source)
                return
        except (OSError, tarfile.TarError):
            pass
        if logical_archive.lower().endswith((".gz", ".tgz")):
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
                    decompressed, error = _read_limited(handle, MAX_ARCHIVE_EXPANDED_BYTES)
                if not error:
                    self._scan_nested(
                        decompressed, logical_archive + "!decompressed", depth, source
                    )
            except (OSError, EOFError):
                pass

    def _scan_tar(
        self, stream: io.BytesIO, logical_archive: str, *, depth: int, source: str
    ) -> None:
        entry_records: list[dict[str, Any]] = []
        expanded = 0
        try:
            with tarfile.open(fileobj=stream, mode="r:*") as archive:
                members = archive.getmembers()
                for index, info in enumerate(members):
                    member, path_error = safe_archive_path(info.name)
                    record: dict[str, Any] = {
                        "archive": logical_archive,
                        "index": index,
                        "member_path": member or info.name,
                        "original_member_name": info.name,
                        "bytes": info.size,
                        "compressed_bytes": None,
                        "is_directory": info.isdir(),
                        "compression": "tar",
                        "hash_status": "NOT_READ",
                        "sha256": None,
                        "classification": classify_path(member or info.name, archive=True),
                        "nested_archive": None,
                    }
                    if path_error:
                        record["hash_status"] = path_error
                    elif info.issym() or info.islnk() or not info.isfile():
                        record["hash_status"] = "UNSAFE_OR_NONREGULAR_MEMBER"
                    elif info.size > MAX_ARCHIVE_ENTRY_BYTES:
                        record["hash_status"] = "ENTRY_TOO_LARGE"
                    elif expanded + info.size > MAX_ARCHIVE_EXPANDED_BYTES:
                        record["hash_status"] = "ARCHIVE_EXPANSION_LIMIT"
                    else:
                        extracted = archive.extractfile(info)
                        data, error = (
                            _read_limited(extracted, MAX_ARCHIVE_ENTRY_BYTES)
                            if extracted
                            else (
                                b"",
                                "NO_MEMBER_STREAM",
                            )
                        )
                        if error:
                            record["hash_status"] = error
                        else:
                            expanded += len(data)
                            record["bytes"] = len(data)
                            record["sha256"] = sha256_bytes(data)
                            record["hash_status"] = "HASHED"
                            if depth < MAX_NESTED_ARCHIVE_DEPTH and self._is_nested_archive(
                                member or "", data
                            ):
                                nested_name = f"{logical_archive}!{member}"
                                record["nested_archive"] = nested_name
                                self._scan_nested(data, nested_name, depth + 1, source)
                    entry_records.append(record)
        except (OSError, tarfile.TarError) as exc:
            self.inspections.append(
                {
                    "logical_archive": logical_archive,
                    "archive_kind": "tar",
                    "depth": depth,
                    "source": source,
                    "status": type(exc).__name__,
                    "entry_count": len(entry_records),
                    "entry_tree_sha256": sha256_bytes(canonical(entry_records)),
                    "nested_archive_count": 0,
                    "entries": entry_records,
                }
            )
            self.entries.extend(entry_records)
            return
        digest_rows = [
            {
                "index": entry["index"],
                "member_path": entry["member_path"],
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "hash_status": entry["hash_status"],
            }
            for entry in entry_records
        ]
        self.inspections.append(
            {
                "logical_archive": logical_archive,
                "archive_kind": "tar",
                "depth": depth,
                "source": source,
                "status": "INSPECTED",
                "entry_count": len(entry_records),
                "expanded_bytes_read": expanded,
                "entry_tree_sha256": sha256_bytes(canonical(digest_rows)),
                "nested_archive_count": sum(
                    entry.get("nested_archive") is not None for entry in entry_records
                ),
                "readable_source_count": sum(
                    entry.get("classification") in {"source_code", "documentation", "readable_text"}
                    and entry.get("hash_status") == "HASHED"
                    for entry in entry_records
                ),
                "empirical_candidate_count": sum(
                    entry.get("classification") == "empirical_data_candidate"
                    and entry.get("hash_status") == "HASHED"
                    for entry in entry_records
                ),
                "entries": entry_records,
            }
        )
        self.entries.extend(entry_records)


def scan_workspace(workspace: Path, excluded_paths: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRECTORY_NAMES for part in path.parts):
            continue
        logical = relative_path(path, workspace)
        if logical in excluded_paths:
            continue
        kind = classify_path(logical)
        file_error: str | None
        if path.is_symlink():
            target = path.readlink().as_posix().encode()
            digest, size, file_error = sha256_bytes(target), len(target), "SYMLINK_TARGET_HASH"
        else:
            digest, size, file_error = sha256_file(path)
        record: dict[str, Any] = {
            "logical_path": logical,
            "bytes": size,
            "sha256": digest or None,
            "hash_status": "HASHED"
            if not file_error or file_error == "SYMLINK_TARGET_HASH"
            else file_error,
            "classification": kind,
            "evidence_class": evidence_for_class(kind, logical),
            "readable": False,
            "provenance_status": (
                "EXCLUDED_PRIVATE_METADATA"
                if kind == "private_or_agent_material"
                else "DIRECTLY_ACCESSIBLE"
            ),
        }
        if file_error == "SYMLINK_TARGET_HASH":
            record["readable"] = False
        elif kind in {
            "source_code",
            "documentation",
            "readable_text",
            "project_registry_or_ledger",
            "synthetic_fixture",
        }:
            metadata = read_text_metadata(path, size)
            record["readable"] = bool(metadata is not None)
            record["date_version"] = metadata
        records.append(record)
    return records


def records_for_prefix(records: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["logical_path"] == prefix
        or record["logical_path"].startswith(prefix.rstrip("/") + "/")
    ]


def summary_for_tree(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    selected = records_for_prefix(records, prefix)
    hash_basis = [
        record
        for record in selected
        if record["classification"]
        not in {"generated_cache", "version_control_template", "private_or_agent_material"}
    ]
    counts = Counter(record["classification"] for record in selected)
    return {
        "logical_root": prefix.rstrip("/") + "/",
        "tree_sha256": tree_digest(hash_basis),
        "complete_tree_sha256": tree_digest(selected),
        "file_count": len(selected),
        "tree_hash_file_count": len(hash_basis),
        "bytes": sum(record["bytes"] for record in selected),
        "classification_counts": dict(sorted(counts.items())),
        "tree_hash_exclusions": [
            "generated_cache",
            "version_control_template",
            "private_or_agent_material",
        ],
    }


def archive_top_level_summary(
    path: Path, workspace: Path, scanner: ArchiveScanner
) -> dict[str, Any]:
    logical = relative_path(path, workspace)
    digest, size, error = sha256_file(path)
    summary = scanner.scan_file(path, logical)
    return {
        "logical_path": logical,
        "sha256": digest or None,
        "bytes": size,
        "hash_status": "HASHED" if not error else error,
        "entry_count": summary.get("entry_count", 0),
        "entry_tree_sha256": summary.get("entry_tree_sha256"),
        "nested_archive_count": sum(
            inspection.get("logical_archive", "").startswith(logical + "!")
            for inspection in scanner.inspections
        ),
        "readable_source_count": summary.get("readable_source_count", 0),
        "empirical_candidate_count": summary.get("empirical_candidate_count", 0),
        "status": summary.get("status"),
    }


def duplicate_groups(
    file_records: list[dict[str, Any]], archive_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for record in file_records:
        if record.get("hash_status") == "HASHED" and record.get("sha256"):
            groups[record["sha256"]].append(
                {
                    "logical_path": record["logical_path"],
                    "kind": record["classification"],
                    "scope": "workspace",
                }
            )
    for record in archive_entries:
        if record.get("hash_status") == "HASHED" and record.get("sha256"):
            groups[record["sha256"]].append(
                {
                    "logical_path": f"{record['archive']}!{record['member_path']}",
                    "kind": record["classification"],
                    "scope": "archive",
                }
            )
    result = []
    for digest, members in sorted(groups.items()):
        unique = {(member["logical_path"], member["scope"]) for member in members}
        if len(unique) > 1:
            result.append({"sha256": digest, "member_count": len(unique), "members": members})
    return result


def source_record(
    source_id: str,
    title: str,
    artifact: str,
    role: str,
    authority: str,
    evidence_class: str,
    provenance_status: str,
    *,
    date_version: dict[str, Any] | None = None,
    supersedes: list[str] | None = None,
    data_links: list[str] | None = None,
    implementation_links: list[str] | None = None,
    experiment_links: list[str] | None = None,
    access_note: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": source_id,
        "title": title,
        "artifact": artifact,
        "role": role,
        "date_version": date_version or {"version": None, "dates": []},
        "authority": authority,
        "evidence_class": evidence_class,
        "provenance_status": provenance_status,
        "supersedes": supersedes or [],
        "data_links": data_links or [],
        "implementation_links": implementation_links or [],
        "experiment_links": experiment_links or [],
    }
    if access_note:
        record["access_note"] = access_note
    return record


def build_inventory(
    workspace: Path, project: Path, public_output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    public_logical = (
        relative_path(public_output, workspace) if public_output.is_relative_to(workspace) else ""
    )
    file_records = scan_workspace(workspace, {public_logical} if public_logical else set())
    scanner = ArchiveScanner()
    archive_paths = [path for path in sorted(workspace.glob("*.zip")) if path.is_file()]
    archive_summaries = [
        archive_top_level_summary(path, workspace, scanner) for path in archive_paths
    ]
    duplicates = duplicate_groups(file_records, scanner.entries)

    project_summary = summary_for_tree(file_records, relative_path(project, workspace))
    chat_summary = summary_for_tree(file_records, "chat") if (workspace / "chat").exists() else None
    docs_summary = summary_for_tree(file_records, f"{relative_path(project, workspace)}/docs")
    source_summary = summary_for_tree(file_records, f"{relative_path(project, workspace)}/src")
    golden_records = [
        record
        for record in file_records
        if record["classification"] == "synthetic_fixture"
        and record["logical_path"].startswith(relative_path(project, workspace) + "/")
    ]
    golden_summary = {
        "tree_sha256": tree_digest(golden_records),
        "file_count": len(golden_records),
        "logical_paths": [record["logical_path"] for record in golden_records],
    }

    top_level_files = [
        record
        for record in file_records
        if "/" not in record["logical_path"] and record["logical_path"] not in {""}
    ]
    root_file_summaries = [
        {
            "logical_path": record["logical_path"],
            "sha256": record["sha256"],
            "bytes": record["bytes"],
            "hash_status": record["hash_status"],
        }
        for record in top_level_files
    ]

    def file_meta(path: str) -> dict[str, Any]:
        for record in file_records:
            if record["logical_path"] == path:
                return record.get("date_version", {"versions": [], "dates": []})
        return {"versions": [], "dates": []}

    archive_by_path = {summary["logical_path"]: summary for summary in archive_summaries}
    current_path = relative_path(project, workspace)
    archive_ids = {
        "PhysioWorld_Final.zip": "ARCHIVE_PHYSIOWORLD_FINAL",
        "physio.zip": "ARCHIVE_PHYSIO",
        "physioworld1.zip": "ARCHIVE_PHYSIOWORLD1",
    }
    registry: list[dict[str, Any]] = []
    registry.append(
        source_record(
            "PROMPT2",
            "PhysioWorld complete master build specification",
            "prompt 2.md",
            "baseline architecture, scientific and release acceptance contract",
            "supplied build prompt",
            "DEFINITION",
            "DIRECTLY_ACCESSIBLE",
            date_version=file_meta("prompt 2.md"),
            implementation_links=["PHYSIOWORLD_CURRENT"],
            experiment_links=["MASTER_SCIENTIFIC_SUITE"],
        )
    )
    registry.append(
        source_record(
            "LEVEL3_PROMPT",
            "PhysioWorld Level 3 scientific completion and corpus experiment specification",
            "PHYSIOWORLD_LEVEL3_COMPLETE_CORPUS_EXPERIMENT_BUILD_PROMPT.md",
            "corpus ingestion, executable experiments, VVUQ, falsification and release extension",
            "supplied Level 3 build prompt",
            "DEFINITION",
            "DIRECTLY_ACCESSIBLE",
            date_version=file_meta("PHYSIOWORLD_LEVEL3_COMPLETE_CORPUS_EXPERIMENT_BUILD_PROMPT.md"),
            implementation_links=["PHYSIOWORLD_CURRENT"],
            experiment_links=["MASTER_SCIENTIFIC_SUITE", "ALL_LEVEL3_EXPERIMENTS"],
        )
    )
    registry.append(
        source_record(
            "PHYSIOWORLD_CURRENT",
            "Current PhysioWorld implementation tree",
            current_path + "/",
            "executable platform and implementation authority",
            "current workspace source tree",
            "SOFTWARE_VERIFICATION",
            "DIRECTLY_ACCESSIBLE",
            date_version={
                "version": "0.1.0",
                "dates": sorted(set(project_summary.get("dates", []))),
            },
            supersedes=["ARCHIVE_PHYSIOWORLD_FINAL", "ARCHIVE_PHYSIO", "ARCHIVE_PHYSIOWORLD1"],
            data_links=["GOLDEN_SYNTHETIC_FIXTURES", "EXTERNAL_DATASET_REGISTRY"],
            implementation_links=["PHYSIOWORLD_CURRENT"],
            experiment_links=["MASTER_SCIENTIFIC_SUITE"],
            access_note="tree_sha256 is the curated source digest; complete_tree_sha256 covers every hashed project file except this generated inventory.",
        )
    )
    registry.append(
        source_record(
            "PHYSIOWORLD_DOCS",
            "Project-owned documentation and scientific ledgers",
            current_path + "/docs/",
            "documentation, provenance ledgers, scope and interpretation boundaries",
            "current workspace documentation",
            "EXTRACTED_SPECIFICATION",
            "DIRECTLY_ACCESSIBLE",
            date_version={"version": None, "dates": []},
            data_links=["EXTERNAL_DATASET_REGISTRY"],
            implementation_links=["PHYSIOWORLD_CURRENT"],
            experiment_links=["MASTER_SCIENTIFIC_SUITE"],
            access_note="Documentation extracts do not become direct evidence for sources whose original bytes are absent.",
        )
    )
    registry.append(
        source_record(
            "PRIOR_RESEARCH_REPOSITORIES",
            "Accessible prior PhysioWorld research archives",
            "PhysioWorld_Final.zip; physio.zip; physioworld1.zip",
            "historical implementation snapshots; duplicate or stale files are not active evidence",
            "workspace archive inspection",
            "HISTORICAL_CONTEXT",
            "ACCESSIBLE_ARCHIVES",
            implementation_links=["PHYSIOWORLD_CURRENT"],
            experiment_links=["master_scientific_suite"],
            access_note="Archive member hashes and duplicate groups are recorded in the external detailed inventory.",
        )
    )
    registry.append(
        source_record(
            "GOLDEN_SYNTHETIC_FIXTURES",
            "Bundled deterministic golden fixtures",
            current_path
            + "/datasets/golden/ and "
            + current_path
            + "/src/physioworld/corpus/golden/",
            "synthetic and mathematical fixtures for local verification",
            "current workspace source tree",
            "SIMULATION",
            "DIRECTLY_ACCESSIBLE",
            data_links=[],
            implementation_links=["PHYSIOWORLD_CURRENT"],
            experiment_links=["MASTER_SCIENTIFIC_SUITE"],
            access_note="Fixtures are synthetic or mathematical; they are not human empirical observations.",
        )
    )
    registry.append(
        source_record(
            "EXTERNAL_DATASET_REGISTRY",
            "Project dataset access registry",
            current_path + "/datasets/registry.json",
            "metadata about authorized external dataset acquisition",
            "current workspace documentation",
            "METADATA",
            "DIRECTLY_ACCESSIBLE",
            date_version=file_meta(current_path + "/datasets/registry.json"),
            data_links=["UPIT_EMPIRICAL_RAW", "EXTERNAL_31_RECORDING_COHORT", "NIRS_EXTERNAL"],
            implementation_links=["PHYSIOWORLD_CURRENT"],
            access_note="Registry entries are access metadata and do not indicate that the datasets are present.",
        )
    )
    for archive_name, source_id in archive_ids.items():
        archive = archive_by_path.get(archive_name, {})
        registry.append(
            source_record(
                source_id,
                f"Accessible {archive_name} implementation snapshot",
                archive_name,
                "historical implementation archive; duplicate files are not independent evidence",
                "workspace archive inspection",
                "HISTORICAL_CONTEXT",
                "ACCESSIBLE_ARCHIVE",
                date_version={"version": "0.1.0", "dates": []},
                supersedes=[],
                implementation_links=["PHYSIOWORLD_CURRENT"],
                experiment_links=["MASTER_SCIENTIFIC_SUITE"],
                access_note=(
                    f"sha256={archive.get('sha256')}; entry_tree_sha256={archive.get('entry_tree_sha256')}"
                ),
            )
        )

    indirect_definitions = [
        (
            "PIR_MATH",
            "PIR mathematical foundation",
            "external:pir/mathematical-foundation",
            "normative definitions for PIV, representation deficiency, loss decomposition and PIR",
            "DEFINITION",
            [],
            ["PIR_LOSS_DECOMPOSITION", "PIV_COUNTERFACTUAL_SYNTHETIC"],
        ),
        (
            "PIR_CONTROLLER",
            "PIR cross-layer controller study",
            "external:pir/cross-layer-controller-study",
            "historical cross-layer allocation/controller benchmark",
            "EXTRACTED_SPECIFICATION",
            [],
            ["CROSS_LAYER_CONTROLLER_BENCHMARK"],
        ),
        (
            "UPIT_EMPIRICAL",
            "UPIT empirical/validation implementation",
            "external:upit/empirical-study",
            "historical orthostatic task-law and representation programme",
            "EXTRACTED_REFERENCE",
            ["UPIT_EMPIRICAL_RAW", "EXTERNAL_31_RECORDING_COHORT"],
            ["TEMPORAL_HISTORY_SWEEP", "TASK_LAW_BATTERY", "EXTERNAL_TRANSPORT"],
        ),
        (
            "UPIT_MATH",
            "UPIT mathematical framework",
            "external:upit/mathematical-framework",
            "task-law, capability and policy-relative information definitions",
            "DEFINITION",
            [],
            ["TASK_LAW_BATTERY", "SUPPORT_FUNCTION_CORRECTION"],
        ),
        (
            "RA_UPIT_GENERAL",
            "General RA-UPIT mathematical framework",
            "external:ra-upit/general-framework",
            "resource-conditioned capability and robust policy definitions",
            "DEFINITION",
            [],
            ["RA_UPIT_PHYSICS", "RESOURCE_CONDITIONED_CAPABILITY"],
        ),
        (
            "RA_UPIT_RESOURCE_MASTER",
            "RA-UPIT nanocapacitor master specification",
            "external:ra-upit/nanocapacitor-specification",
            "canonical P0/P1/P2 resource fixture and numerical references",
            "EXTRACTED_REFERENCE",
            [],
            ["RA_UPIT_PHYSICS"],
        ),
        (
            "COMMUNICATION_THEORY",
            "Communication and capability mathematical draft",
            "external:communication/capability-draft",
            "semantic-versus-transport distinction and coding requirements",
            "DEFINITION",
            [],
            ["COMMUNICATION_NEGATIVE_CONTROL"],
        ),
        (
            "DECISION_THEORY",
            "Prior decision-theoretic scientific draft",
            "external:decision-theory/prior-draft",
            "utility, policy and capability interpretation boundary",
            "EXTRACTED_SPECIFICATION",
            [],
            ["SUPPORT_FUNCTION_CORRECTION", "PIR_LOSS_DECOMPOSITION"],
        ),
        (
            "CURRENT_SCIENTIFIC_MAIN",
            "Current flagship main scientific document",
            "external:current-scientific-main",
            "current corrected scientific interpretation",
            "EXTRACTED_REFERENCE",
            ["UPIT_EMPIRICAL_RAW", "EXTERNAL_31_RECORDING_COHORT"],
            ["TEMPORAL_HISTORY_SWEEP", "TASK_LAW_BATTERY", "SUPPORT_FUNCTION_CORRECTION"],
        ),
        (
            "CURRENT_SCIENTIFIC_SUPPLEMENT",
            "Current flagship supplementary scientific document",
            "external:current-scientific-supplement",
            "supplementary methods, controls and source-specific analyses",
            "EXTRACTED_SPECIFICATION",
            ["NIRS_EXTERNAL", "NETWORK_EXTERNAL"],
            ["NIRS_RAW_PROXY", "NETWORK_PHYSIOLOGY"],
        ),
        (
            "PRIOR_CORRECTED_SCIENTIFIC_DOCUMENTS",
            "Previous corrected or reconciled scientific documents",
            "external:prior-corrected-scientific-documents",
            "historical source documents mentioned by the supplied corpus prompt",
            "HISTORICAL_CONTEXT",
            [],
            [],
        ),
        (
            "PRIOR_RESEARCH_AUDITS",
            "Prior research-audit files",
            "external:physioworld/prior-research-audits",
            "historical audit material mentioned by the supplied corpus prompt",
            "SOFTWARE_VERIFICATION",
            [],
            ["MASTER_SCIENTIFIC_SUITE"],
        ),
        (
            "LEVEL1_AUDIT",
            "PhysioWorld Level 1 audit",
            "external:physioworld/level1-audit",
            "prior implementation and scientific audit findings",
            "SOFTWARE_VERIFICATION",
            [],
            ["MASTER_SCIENTIFIC_SUITE"],
        ),
        (
            "LEVEL2_AUDIT",
            "PhysioWorld Level 2 post-build audit",
            "external:physioworld/level2-audit",
            "post-build audit mentioned by the supplied Level 3 prompt",
            "SOFTWARE_VERIFICATION",
            [],
            ["MASTER_SCIENTIFIC_SUITE"],
        ),
        (
            "UPIT_EMPIRICAL_RAW",
            "UPIT empirical raw cohort",
            "external:upit/empirical-study/raw-data",
            "raw empirical source data required for the historical UPIT analyses",
            "EMPIRICAL_ESTIMATE",
            [],
            ["TEMPORAL_HISTORY_SWEEP", "TASK_LAW_BATTERY"],
        ),
        (
            "EXTERNAL_31_RECORDING_COHORT",
            "31-recording/subject external cohort",
            "external:upit/31-recording-cohort",
            "independent predictive transport cohort mentioned by the supplied corpus",
            "EXTERNAL_VALIDATION",
            [],
            ["EXTERNAL_TRANSPORT", "CROSS_REGIME_TRANSFER"],
        ),
        (
            "NIRS_EXTERNAL",
            "NIRS source recordings",
            "external:nirs/source-recordings",
            "source recordings for the declared raw optical-density proxy analysis",
            "EMPIRICAL_ESTIMATE",
            [],
            ["NIRS_RAW_PROXY"],
        ),
        (
            "NETWORK_EXTERNAL",
            "Network physiology source recordings",
            "external:network-physiology/source-recordings",
            "source recordings for association and surrogate analyses",
            "EMPIRICAL_ESTIMATE",
            [],
            ["NETWORK_PHYSIOLOGY"],
        ),
        (
            "EXTERNAL_STRESS_DATASETS",
            "Wearable, CPAP, Mayo/LBNP and other external stress datasets",
            "external:declared-stress-datasets",
            "source-specific external datasets mentioned for transport or stress testing",
            "EXTERNAL_VALIDATION",
            ["WEARABLE_EXTERNAL", "CPAP_EXTERNAL", "MAYO_LBNP_EXTERNAL"],
            ["EXTERNAL_STRESS_FAMILY"],
        ),
    ]
    for (
        source_id,
        title,
        artifact,
        role,
        evidence,
        data_links,
        experiment_links,
    ) in indirect_definitions:
        registry.append(
            source_record(
                source_id,
                title,
                artifact,
                role,
                "mentioned by supplied prompts or project ledgers",
                evidence,
                "INDIRECTLY_DOCUMENTED",
                data_links=data_links,
                implementation_links=["PHYSIOWORLD_CURRENT"],
                experiment_links=experiment_links,
                access_note="Original bytes are not present in the accessible workspace; contents are not inferred.",
            )
        )
    for source_id, title, artifact, role in [
        (
            "WEARABLE_EXTERNAL",
            "Wearable external source data",
            "external:wearable/source-data",
            "source-specific wearable stress-test data",
        ),
        (
            "CPAP_EXTERNAL",
            "CPAP external source data",
            "external:cpap/source-data",
            "source-specific CPAP stress-test data",
        ),
        (
            "MAYO_LBNP_EXTERNAL",
            "Mayo/LBNP external source data",
            "external:mayo-lbnp/source-data",
            "source-specific dynamic validation data",
        ),
    ]:
        registry.append(
            source_record(
                source_id,
                title,
                artifact,
                role,
                "mentioned by supplied Level 3 prompt",
                "EMPIRICAL_ESTIMATE",
                "INDIRECTLY_DOCUMENTED",
                implementation_links=["PHYSIOWORLD_CURRENT"],
                experiment_links=["EXTERNAL_STRESS_FAMILY"],
                access_note="Mentioned source only; no source bytes were found.",
            )
        )

    inaccessible_empirical_mentions = [
        "UPIT_EMPIRICAL_RAW",
        "EXTERNAL_31_RECORDING_COHORT",
        "NIRS_EXTERNAL",
        "NETWORK_EXTERNAL",
        "WEARABLE_EXTERNAL",
        "CPAP_EXTERNAL",
        "MAYO_LBNP_EXTERNAL",
    ]
    empirical_candidates = [
        record
        for record in file_records + scanner.entries
        if record.get("classification") == "empirical_data_candidate"
        and record.get("hash_status") == "HASHED"
    ]
    real_empirical = [
        record
        for record in empirical_candidates
        if record.get("classification") == "empirical_data_candidate"
        and "golden" not in str(record.get("logical_path", "")).lower()
    ]
    # The current workspace has no candidate outside generated caches or synthetic fixtures;
    # keep this assertion data-driven rather than relying on a hard-coded file list.
    real_empirical_paths = [record.get("logical_path") for record in real_empirical]
    generated_cache_candidates = [
        record.get("logical_path")
        for record in file_records + scanner.entries
        if record.get("classification") in {"generated_cache", "version_control_template"}
        and PurePosixPath(str(record.get("logical_path", ""))).suffix.lower() in {".db", ".sample"}
    ]

    counts = {
        "workspace_files_hashed": sum(
            record.get("hash_status") == "HASHED" for record in file_records
        ),
        "workspace_files_scanned": len(file_records),
        "project_files_scanned": len(records_for_prefix(file_records, current_path)),
        "project_source_files_readable": sum(
            record.get("readable") is True
            for record in records_for_prefix(file_records, current_path)
            if record.get("classification")
            in {"source_code", "documentation", "readable_text", "project_registry_or_ledger"}
        ),
        "archive_inputs": len(archive_summaries),
        "archive_inspections_including_nested": len(scanner.inspections),
        "archive_entries_hashed": sum(
            entry.get("hash_status") == "HASHED" for entry in scanner.entries
        ),
        "nested_archives_found": sum(
            summary.get("nested_archive_count", 0) for summary in archive_summaries
        ),
        "exact_duplicate_groups": len(duplicates),
        "synthetic_fixture_files": len(golden_records),
        "empirical_data_candidates_hashed": len(empirical_candidates),
        "real_empirical_source_files_found": len(real_empirical_paths),
        "indirectly_documented_sources": sum(
            record["provenance_status"] == "INDIRECTLY_DOCUMENTED" for record in registry
        ),
    }

    public_duplicates = [
        {
            "sha256": group["sha256"],
            "member_count": group["member_count"],
            "representative_logical_paths": [
                member["logical_path"] for member in group["members"][:6]
            ],
        }
        for group in duplicates
        if not all(
            member["kind"] in {"generated_cache", "version_control_template"}
            for member in group["members"]
        )
    ][:8]

    public_inventory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inventory_type": "curated_corpus_lineage",
        "inventory_status": "ACCESSIBLE_WORKSPACE_METADATA_ONLY",
        "hash_algorithm": "SHA-256",
        "logical_path_policy": "All paths are workspace-relative or archive-logical; host paths and file contents are omitted.",
        "privacy_policy": {
            "metadata_only": True,
            "public_content_included": False,
            "agent_chatter_raw_logs_private_planning_excluded": True,
            "indirect_sources_not_invented": True,
        },
        "input_scope": {
            "workspace_root_label": "workspace-root",
            "project_root_label": current_path + "/",
            "vcs_metadata_scanned": False,
            "generated_public_inventory_excluded_from_project_tree_digest": public_logical,
            "generated_cache_and_vcs_template_files_excluded_from_curated_tree_digest": True,
            "complete_tree_sha256_includes_generated_files": True,
            "archives_inspected_without_extraction": True,
        },
        "counts": counts,
        "top_level_inputs": [
            *root_file_summaries,
            {"id": "PHYSIOWORLD_CURRENT", **project_summary},
            {"id": "PHYSIOWORLD_DOCS", **docs_summary},
            {"id": "PHYSIOWORLD_SRC", **source_summary},
            {"id": "GOLDEN_SYNTHETIC_FIXTURES", **golden_summary},
            *([{"id": "WORKSPACE_CHAT", **chat_summary}] if chat_summary else []),
        ],
        "archives": archive_summaries,
        "archive_entry_tree_policy": "Top-level archive SHA-256 and entry-tree SHA-256 are public; complete member metadata is externalized.",
        "duplicate_policy": {
            "same_content_is_one_logical_artifact": True,
            "duplicate_group_count": len(duplicates),
            "public_example_groups": public_duplicates,
            "detailed_groups_externalized": True,
        },
        "source_registry": registry,
        "lineage": [
            {"from": "PROMPT2", "relation": "specifies", "to": "PHYSIOWORLD_CURRENT"},
            {"from": "LEVEL3_PROMPT", "relation": "extends_and_requires", "to": "PROMPT2"},
            {
                "from": "LEVEL3_PROMPT",
                "relation": "requires_ingestion_of",
                "to": "PRIOR_RESEARCH_REPOSITORIES",
            },
            {
                "from": "PHYSIOWORLD_CURRENT",
                "relation": "implements",
                "to": "GOLDEN_SYNTHETIC_FIXTURES",
            },
            {
                "from": "PHYSIOWORLD_CURRENT",
                "relation": "supersedes_as_active_implementation",
                "to": "PRIOR_RESEARCH_REPOSITORIES",
            },
            {
                "from": "CURRENT_SCIENTIFIC_MAIN",
                "relation": "supersedes",
                "to": "PRIOR_CORRECTED_SCIENTIFIC_DOCUMENTS",
            },
            {"from": "LEVEL2_AUDIT", "relation": "supersedes", "to": "LEVEL1_AUDIT"},
            {
                "from": "CURRENT_SCIENTIFIC_MAIN",
                "relation": "requires_data",
                "to": "UPIT_EMPIRICAL_RAW",
            },
            {
                "from": "PHYSIOWORLD_CURRENT",
                "relation": "provides_synthetic_substitute_for",
                "to": "UPIT_EMPIRICAL_RAW",
            },
            {
                "from": "EXTERNAL_DATASET_REGISTRY",
                "relation": "describes_access_for",
                "to": "EXTERNAL_STRESS_DATASETS",
            },
        ],
        "empirical_data_findings": {
            "real_empirical_source_data_found": bool(real_empirical_paths),
            "accessible_real_empirical_logical_paths": real_empirical_paths,
            "indirectly_documented_empirical_sources": inaccessible_empirical_mentions,
            "synthetic_fixture_policy": "Bundled golden JSON fixtures are synthetic or mathematical and are not human evidence.",
            "non_empirical_binary_notes": {
                "generated_cache_candidates": generated_cache_candidates[:20],
                "interpretation": "Any .db entries found under mypy caches and .sample entries under VCS hooks are generated/tooling artifacts, not empirical source records.",
            },
        },
        "raw_inventory": {
            "logical_location": "external_raw_inventory/corpus_inventory_detailed.json",
            "contains_content": False,
            "contains_complete_file_and_entry_metadata": True,
        },
    }
    raw_inventory = {
        "schema_version": SCHEMA_VERSION,
        "inventory_status": "DETAILED_METADATA_ONLY",
        "counts": counts,
        "workspace_files": file_records,
        "archive_inspections": scanner.inspections,
        "archive_entries": scanner.entries,
        "duplicate_groups": duplicates,
        "public_inventory_sha256_excluded_until_written": True,
    }
    return public_inventory, raw_inventory


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve()
    defaults = {
        "workspace": here.parents[2],
        "project": here.parents[1],
        "public_output": here.parents[1] / "docs" / "corpus_inventory.json",
        "raw_output": Path(tempfile.gettempdir())
        / "physioworld"
        / "corpus_inventory_detailed.json",
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=defaults["workspace"])
    parser.add_argument("--project", type=Path, default=defaults["project"])
    parser.add_argument("--public-output", type=Path, default=defaults["public_output"])
    parser.add_argument("--raw-output", type=Path, default=defaults["raw_output"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    project = args.project.resolve()
    public_output = args.public_output.resolve()
    raw_output = args.raw_output.resolve()
    if not workspace.is_dir() or not project.is_dir():
        raise SystemExit("workspace and project must be existing directories")
    if not project.is_relative_to(workspace):
        raise SystemExit("project must be inside workspace")
    if public_output.is_relative_to(workspace) and public_output == raw_output:
        raise SystemExit("public and raw outputs must be different")
    if raw_output.is_relative_to(workspace):
        raise SystemExit("raw inventory must be outside the workspace source tree")
    public_inventory, raw_inventory = build_inventory(workspace, project, public_output)
    write_json(raw_output, raw_inventory)
    write_json(public_output, public_inventory)
    print(
        json.dumps(
            {
                "public_output": public_output.as_posix(),
                "raw_output": raw_output.as_posix(),
                **public_inventory["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

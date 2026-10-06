"""Governed public dataset catalog; contains no samples, secrets, or access grants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


class DatasetAccess(str, Enum):
    OPEN = "OPEN"
    RESTRICTED = "RESTRICTED"
    CREDENTIALED = "CREDENTIALED"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    title: str
    access: DatasetAccess
    source_url: str
    citation: str
    version: str
    subject_unit: str
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    acquisition: str
    preprocessing: str
    license: str
    expected_schema: Mapping[str, Any]
    units: Mapping[str, str]
    population: str
    intended_use: tuple[str, ...]
    forbidden_use: tuple[str, ...]
    split_rules: tuple[str, ...]
    leakage_risk: tuple[str, ...]
    credential_requirements: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    checksum: str | None = None
    checksum_policy: str = "SHA-256 on authorized local inputs; no raw data bundled"
    synthetic_ci_substitute: str = (
        "generated local WFDB/standard fixtures in test_data_interoperability.py"
    )
    license_status: str = "source_terms_required"

    def __post_init__(self) -> None:
        object.__setattr__(self, "access", DatasetAccess(self.access))
        required = (
            "dataset_id",
            "title",
            "source_url",
            "citation",
            "version",
            "subject_unit",
            "modalities",
            "tasks",
            "acquisition",
            "preprocessing",
            "license",
            "expected_schema",
            "units",
            "population",
            "intended_use",
            "split_rules",
            "leakage_risk",
            "checksum_policy",
        )
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError("dataset governance requires " + ", ".join(missing))
        if self.access != DatasetAccess.OPEN and not self.credential_requirements:
            raise ValueError("restricted datasets require credential requirements")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["access"] = self.access.value
        return result


@dataclass
class DatasetRegistry:
    specs: tuple[DatasetSpec, ...]

    def __post_init__(self) -> None:
        if len({s.dataset_id for s in self.specs}) != len(self.specs):
            raise ValueError("dataset identifiers must be unique")

    def get(self, dataset_id: str) -> DatasetSpec:
        for spec in self.specs:
            if spec.dataset_id == dataset_id:
                return spec
        raise KeyError(dataset_id)

    def list(self, access: DatasetAccess | None = None) -> tuple[DatasetSpec, ...]:
        return tuple(s for s in self.specs if access is None or s.access == access)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "1.0.0", "datasets": [s.to_dict() for s in self.specs]}


def default_registry() -> DatasetRegistry:
    # Broad collections remain explicitly provider-defined until a caller binds
    # an individual versioned release. Licensing pointers are not invented grants.
    rows = (
        (
            "mitdb",
            "MIT-BIH Arrhythmia Database",
            "mitdb",
            "1.0.0",
            "Moody and Mark, 2001",
            ("ECG",),
            ("arrhythmia",),
            "ambulatory arrhythmia cohort",
            "WFDB",
        ),
        (
            "afdb",
            "MIT-BIH Atrial Fibrillation Database",
            "afdb",
            "1.0.0",
            "Goldberger et al., 2000",
            ("ECG",),
            ("atrial_fibrillation",),
            "long-term atrial fibrillation recordings",
            "WFDB",
        ),
        (
            "physionet_challenge",
            "PhysioNet/CinC Challenge resources",
            "",
            "provider-defined",
            "Challenge-specific source citation required",
            ("multimodal",),
            ("challenge_task",),
            "challenge/year-specific cohort",
            "provider-defined",
        ),
        (
            "mimic_waveform",
            "MIMIC-IV Waveform Database",
            "mimic4wdb",
            "provider-defined",
            "Source release citation required",
            ("ECG", "PPG", "BP", "respiration"),
            ("deterioration", "hypotension"),
            "critical-care cohort",
            "WFDB",
        ),
        (
            "mimic_iv_ecg",
            "MIMIC-IV-ECG",
            "mimic-iv-ecg",
            "provider-defined",
            "Source release citation required",
            ("ECG",),
            ("diagnosis",),
            "clinical hospital ECG cohort",
            "WFDB",
        ),
        (
            "sleep_edf",
            "Sleep-EDF Expanded",
            "sleep-edfx",
            "1.0.0",
            "Kemp et al., 2000",
            ("EEG", "EOG", "EMG"),
            ("sleep_staging",),
            "source-defined sleep cohorts",
            "EDF",
        ),
        (
            "wesad",
            "WESAD",
            "https://archive.ics.uci.edu/dataset/465/wesad",
            "1.0",
            "Schmidt et al., 2018",
            ("ECG", "EDA", "temperature", "respiration", "acceleration"),
            ("stress",),
            "wearable stress study cohort",
            "provider-defined",
        ),
        (
            "pamap2",
            "PAMAP2 Physical Activity Monitoring",
            "https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring",
            "2.0",
            "Reiss and Stricker, 2012",
            ("IMU", "ECG", "temperature"),
            ("activity",),
            "physical activity study cohort",
            "text",
        ),
        (
            "gaitpdb",
            "Gait in Parkinson's Disease Database",
            "gaitpdb",
            "1.0.0",
            "Hausdorff et al., 2007",
            ("gait",),
            ("mobility", "rehabilitation"),
            "Parkinson's disease and control gait cohort",
            "text",
        ),
    )
    specs = []
    verified_licenses = {
        "mitdb": "Open Data Commons Attribution License v1.0",
        "sleep_edf": "Open Data Commons Attribution License v1.0",
        "gaitpdb": "Open Data Commons Attribution License v1.0",
    }
    source_units = {
        "mitdb": {"ECG": "mV"},
        "afdb": {"ECG": "mV"},
        "sleep_edf": {"EEG": "uV", "EOG": "uV", "EMG": "uV"},
        "gaitpdb": {"gait": "N"},
    }
    for dataset_id, title, location, version, citation, modalities, tasks, population, fmt in rows:
        source = (
            location
            if location.startswith("https:")
            else (
                f"https://physionet.org/content/{location}/"
                if location
                else "https://physionet.org/about/database/"
            )
        )
        access = (
            DatasetAccess.CREDENTIALED if dataset_id.startswith("mimic_") else DatasetAccess.OPEN
        )
        credentials = (
            ()
            if access == DatasetAccess.OPEN
            else (
                "Provider credentialing and training",
                "Provider data-use agreement",
                "Local authorized=True attestation",
            )
        )
        specs.append(
            DatasetSpec(
                dataset_id,
                title,
                access,
                source,
                citation,
                version,
                "subject",
                modalities,
                tasks,
                "Acquire the declared release directly from provider under its terms; supply a local path",
                "None applied by registry; preserve source units/timing/annotations",
                verified_licenses.get(
                    dataset_id,
                    f"Provider release terms at {source}; bind license before use/redistribution",
                ),
                {
                    "format": fmt,
                    "required": (
                        "record_id",
                        "source_subject_mapping",
                        "timestamps_or_rate",
                        "channel_units",
                    ),
                },
                source_units.get(
                    dataset_id, {m: "explicit source channel unit required" for m in modalities}
                ),
                population,
                tasks,
                ("Source-specific prohibitions apply; this manifest grants no additional rights",),
                (
                    "Independent subjects with verified source identity mapping",
                    "Train-only preprocessing",
                    "No overlapping windows or future inputs",
                ),
                (
                    "Repeated subjects/sessions",
                    "Copied records",
                    "Temporal and label preprocessing contamination",
                ),
                credentials,
                limitations=(
                    "Adapter availability is format-specific; no dataset is bundled or downloaded",
                ),
                license_status="verified_source_metadata"
                if dataset_id in verified_licenses
                else "source_terms_required",
            )
        )
    return DatasetRegistry(tuple(specs))

"""NWB TimeSeries and DICOM imaging metadata via their supported packages."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from physioworld.core import Provenance
from physioworld.physiology.record import PhysiologicalRecord
from physioworld.physiology.stream import SignalStream
from physioworld.physiology.time import CalibrationMetadata, ClockDomain, TimeMetadata

from .canonical import decode_json
from .containers import _join, _split
from .optional import require


class NWBAdapter:
    def write(self, obj, path: str | Path, *, session_start_time: datetime) -> None:
        nwb = require("pynwb", "NWB")
        np = require("numpy", "NWB", "scientific")
        if session_start_time.tzinfo is None:
            raise ValueError("NWB session_start_time requires timezone")
        header, streams = _split(obj)
        identifier = obj.metadata.record_id if isinstance(obj, SignalStream) else obj.record_id
        file = nwb.NWBFile(
            "PhysioWorld canonical interchange", identifier or "signal", session_start_time
        )
        mapping = {}
        for index, (name, stream) in enumerate(streams.items()):
            if len({c.unit for c in stream.metadata.channels}) != 1:
                raise ValueError(
                    "NWB TimeSeries requires homogeneous units; split heterogeneous channels explicitly"
                )
            series_name = f"stream_{index}"
            time_offset = (
                session_start_time.timestamp()
                if stream.metadata.clock.time_basis == "unix"
                else 0.0
            )
            mapping[series_name] = {"name": name, "time_offset_s": time_offset}
            file.add_acquisition(
                nwb.TimeSeries(
                    name=series_name,
                    data=np.array(stream.values, dtype=float).reshape(
                        stream.n_samples, stream.n_channels
                    ),
                    unit=stream.metadata.channels[0].unit,
                    timestamps=np.array(stream.timestamps, dtype=float) - time_offset,
                    description="Canonical clock in envelope; Unix time explicitly converted to NWB reference seconds",
                )
            )
        file.add_scratch(
            json.dumps({"header": header, "series": mapping}),
            name="physioworld_canonical",
            description="Versioned canonical metadata, missingness, and clock semantics",
        )
        with nwb.NWBHDF5IO(str(path), "w") as handle:
            handle.write(file)

    def read(
        self,
        path: str | Path,
        *,
        subject_id: str | None = None,
        record_id: str | None = None,
        session_id: str | None = None,
    ) -> PhysiologicalRecord | SignalStream:
        nwb = require("pynwb", "NWB")
        with nwb.NWBHDF5IO(str(path), "r") as handle:
            file = handle.read()
            if "physioworld_canonical" in file.scratch:
                raw = file.scratch["physioworld_canonical"].data
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                envelope = decode_json(str(raw))
                samples = {}
                for series_name, semantics in envelope["series"].items():
                    series = file.acquisition[series_name]
                    samples[semantics["name"]] = (
                        series.timestamps[:] + semantics["time_offset_s"],
                        series.data[:],
                    )
                return _join(envelope["header"], samples)
            subject = subject_id or (file.subject.subject_id if file.subject is not None else None)
            if not subject:
                raise ValueError("external NWB requires subject identity")
            rid = record_id or file.identifier
            streams = {}
            for name, series in file.acquisition.items():
                if not isinstance(series, nwb.TimeSeries):
                    raise ValueError(f"unsupported NWB acquisition {name}: not TimeSeries")
                values = series.data[:]
                if values.ndim > 2:
                    raise ValueError(
                        "NWB image/higher-dimensional series require a dedicated profile"
                    )
                if series.timestamps is not None:
                    times, rate = series.timestamps[:], series.rate
                elif series.rate is not None and series.starting_time is not None:
                    rate = float(series.rate)
                    if not math.isfinite(rate) or rate <= 0:
                        raise ValueError("NWB rate must be finite and positive")
                    times = [series.starting_time + i / rate for i in range(len(values))]
                else:
                    raise ValueError("NWB TimeSeries requires timestamps or starting_time and rate")
                conversion, offset = float(series.conversion), float(series.offset)
                if not math.isfinite(conversion) or not math.isfinite(offset):
                    raise ValueError("invalid NWB conversion/offset")
                values = values * conversion + offset
                rows = values.tolist()
                rows = [[v] for v in rows] if values.ndim == 1 else rows
                streams[name] = SignalStream.from_samples(
                    rows,
                    rate,
                    series.unit,
                    "NWB_timeseries",
                    timestamps=times,
                    subject_id=subject,
                    record_id=rid,
                    session_id=session_id,
                    clock=TimeMetadata(
                        ClockDomain.ACQUISITION, origin=file.timestamps_reference_time.isoformat()
                    ),
                    missing_channels=[tuple(not math.isfinite(v) for v in row) for row in rows],
                    calibration=CalibrationMetadata(
                        conversion,
                        offset,
                        method="NWB.TimeSeries",
                        output_unit=series.unit,
                        status="ASSUMED",
                        operating_conditions={"physical_samples_already_calibrated": True},
                    ),
                    provenance=Provenance(
                        "NWB",
                        "map_timeseries",
                        notes={
                            "conversion": conversion,
                            "offset": offset,
                            "description": series.description,
                            "identifier": file.identifier,
                        },
                    ),
                    attributes={
                        "unit_source": "NWB.TimeSeries.unit",
                        "session_description": file.session_description,
                    },
                )
            return PhysiologicalRecord(
                subject,
                rid,
                session_id,
                streams,
                provenance={
                    "standard": "NWB",
                    "session_start_time": file.session_start_time.isoformat(),
                },
            )


class DICOMAdapter:
    def read(
        self, path: str | Path, *, subject_id: str, include_pixels: bool = False
    ) -> PhysiologicalRecord:
        """Preserve DICOM public metadata; caller supplies pseudonymous subject identity.

        Pixel values (when explicitly requested) remain stored values with their
        rescale/window metadata; no unsupported physical-unit conversion occurs.
        Patient names/IDs, private tags, and raw binary elements are not copied.
        """
        pydicom = require("pydicom", "DICOM")
        data = pydicom.dcmread(str(path), stop_before_pixels=not include_pixels)
        if not subject_id:
            raise ValueError("DICOM requires explicit canonical subject identity")
        rid = str(getattr(data, "SOPInstanceUID", ""))
        if not rid:
            raise ValueError("DICOM SOPInstanceUID is required")
        keys = (
            "SOPClassUID",
            "SOPInstanceUID",
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Modality",
            "StudyDate",
            "StudyTime",
            "AcquisitionDateTime",
            "AcquisitionDate",
            "AcquisitionTime",
            "TimezoneOffsetFromUTC",
            "Rows",
            "Columns",
            "NumberOfFrames",
            "PixelSpacing",
            "ImageOrientationPatient",
            "ImagePositionPatient",
            "SliceThickness",
            "BitsAllocated",
            "BitsStored",
            "PixelRepresentation",
            "PhotometricInterpretation",
            "SamplesPerPixel",
            "RescaleSlope",
            "RescaleIntercept",
            "RescaleType",
            "WindowCenter",
            "WindowWidth",
        )
        image = {}
        for key in keys:
            if hasattr(data, key):
                value = getattr(data, key)
                if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
                    value = tuple(str(v) for v in value)
                else:
                    value = str(value)
                image[key] = value
        # Preserve absent timezone as unknown rather than treating local DICOM time as UTC.
        image["time_basis"] = "DICOM local/offset metadata; no inferred timezone"
        image["spatial_units"] = {
            "PixelSpacing": "mm",
            "SliceThickness": "mm",
            "ImagePositionPatient": "mm",
        }
        acquisition = str(getattr(data, "AcquisitionDateTime", ""))
        if acquisition and len(acquisition.split(".")[0].split("+")[0].split("-")[0]) >= 14:
            instant = pydicom.valuerep.DT(acquisition)
            offset = str(getattr(data, "TimezoneOffsetFromUTC", ""))
            if instant.tzinfo is None and offset:
                sign = -1 if offset[0] == "-" else 1
                instant = instant.replace(
                    tzinfo=timezone(
                        sign * timedelta(hours=int(offset[1:3]), minutes=int(offset[3:5]))
                    )
                )
            if instant.tzinfo is not None:
                image["acquisition_timestamp_s"] = instant.timestamp()
                image["time_basis"] = "unix"
                image["clock_domain"] = "acquisition"
        if include_pixels:
            image["pixels"] = data.pixel_array.tolist()
            image["pixel_units"] = str(getattr(data, "RescaleType", "stored_pixel_value"))
        return PhysiologicalRecord(
            subject_id,
            rid,
            str(getattr(data, "StudyInstanceUID", "")) or None,
            {},
            images={rid: image},
            metadata={"data_kind": "clinical_imaging"},
            provenance={
                "standard": "DICOM",
                "operation": "read_imaging_metadata",
                "pixels_included": include_pixels,
            },
        )

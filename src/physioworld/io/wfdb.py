"""Local WFDB physical samples, annotations, calibration, and access boundary."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from physioworld.core import Provenance
from physioworld.physiology.stream import Annotation, ChannelMetadata, SignalStream
from physioworld.physiology.time import CalibrationMetadata, TimeMetadata

from .optional import require


class WFDBAdapter:
    name = "wfdb"

    def read(
        self,
        record: str | Path,
        *,
        modality: str = "unknown",
        annotation_extension: str | None = None,
        subject_id: str | None = None,
        session_id: str | None = None,
        record_id: str | None = None,
        access_class: str = "OPEN",
        authorized: bool = False,
        source_dataset: str | None = None,
        dataset_version: str | None = None,
    ) -> SignalStream:
        if access_class not in {"OPEN", "RESTRICTED", "CREDENTIALED"}:
            raise ValueError("invalid WFDB access class")
        if access_class != "OPEN" and not authorized:
            raise PermissionError(
                "restricted/credentialed WFDB requires authorized=True after provider authorization"
            )
        candidate = Path(record)
        if "://" in str(record) or not Path(str(candidate) + ".hea").is_file():
            raise ValueError(
                "WFDBAdapter reads local record stems with a .hea file; no network/credentials accepted"
            )
        wfdb = require("wfdb", "WFDB", "wfdb")
        raw = wfdb.rdrecord(str(record), physical=True)
        signal = raw.p_signal
        if signal is None:
            raise ValueError(
                "WFDB record has no uniform physical samples (multi-segment expansion required)"
            )
        rate = raw.fs
        if rate is None or not math.isfinite(rate) or rate <= 0:
            raise ValueError("WFDB requires a finite positive declared sampling rate")
        names, units = raw.sig_name, raw.units
        if (
            not names
            or not units
            or len(names) != signal.shape[1]
            or len(units) != len(names)
            or any(not u for u in units)
        ):
            raise ValueError("WFDB requires complete channel names and units")
        channels = []
        for index, (name, unit) in enumerate(zip(names, units)):
            gain, baseline = raw.adc_gain[index], raw.baseline[index]
            if gain is None or not math.isfinite(gain) or gain == 0:
                raise ValueError("invalid WFDB ADC gain")
            cal = CalibrationMetadata(
                1 / gain,
                -baseline / gain,
                adc_bits=raw.adc_res[index] or None,
                method="WFDB_header",
                input_unit="ADC_count",
                output_unit=unit,
                status="ASSUMED",
                operating_conditions={"physical_samples_already_calibrated": True},
            )
            channels.append(ChannelMetadata(name, unit, calibration=cal))
        rid = record_id or candidate.name  # a record name is never promoted to subject identity
        rows = signal.tolist()
        stream = SignalStream.from_samples(
            rows,
            rate,
            units[0],
            modality,
            record_id=rid,
            subject_id=subject_id,
            session_id=session_id,
            channel_names=names,
            source_dataset=source_dataset,
            dataset_version=dataset_version,
            access_class=access_class,
            missing_channels=[tuple(not math.isfinite(v) for v in row) for row in rows],
            clock=TimeMetadata(origin=raw.base_datetime.isoformat() if raw.base_datetime else None),
            provenance=Provenance(
                "WFDB",
                "read_local_physical",
                notes={
                    "record_name": candidate.name,
                    "header_comments": tuple(raw.comments or ()),
                    "fs": rate,
                    "annotation_extension": annotation_extension,
                },
            ),
        )
        annotations: tuple[Annotation, ...] = ()
        if annotation_extension:
            annotation = wfdb.rdann(str(record), annotation_extension)
            annotations = tuple(
                Annotation(
                    float(sample) / rate,
                    label=symbol,
                    attributes={
                        "sample_index": int(sample),
                        "aux_note": annotation.aux_note[i],
                        "channel": int(annotation.chan[i]),
                        "subtype": int(annotation.subtype[i]),
                    },
                )
                for i, (sample, symbol) in enumerate(zip(annotation.sample, annotation.symbol))
            )
        return replace(
            stream,
            metadata=replace(stream.metadata, channels=tuple(channels)),
            annotations=annotations,
        )

"""Dependency-free semantic record interchange using synthetic local samples.

Run: python examples/data_interchange.py /your/output/directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

from physioworld.core import Provenance
from physioworld.io import CSVAdapter, JSONAdapter, NDJSONAdapter
from physioworld.physiology.record import PhysiologicalRecord, RecordEvent
from physioworld.physiology.stream import SignalStream
from physioworld.physiology.time import CalibrationMetadata, TimeMetadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stream = SignalStream.from_samples(
        [0.1, float("nan"), 0.3],
        None,
        "mV",
        "ECG",
        timestamps=[0.0, 0.12, 0.29],
        subject_id="synthetic-subject",
        record_id="synthetic-record",
        session_id="synthetic-session",
        device_id="synthetic-device",
        missing=[False, True, False],
        data_kind="simulation",
        clock=TimeMetadata("acquisition", origin="synthetic-relative-seconds"),
        calibration=CalibrationMetadata(method="synthetic_declared", output_unit="mV"),
        provenance=Provenance(
            "synthetic", "example", seed=0, notes={"evidence_class": "SIMULATION"}
        ),
    )
    record = PhysiologicalRecord(
        "synthetic-subject",
        "synthetic-record",
        "synthetic-session",
        {"ECG": stream},
        (RecordEvent(0.12, "synthetic dropout", clock=stream.metadata.clock),),
        provenance={"evidence_class": "SIMULATION", "source": "generated example"},
    )
    JSONAdapter().write(record, args.output / "record.json")
    CSVAdapter().write(stream, args.output / "signal.csv")
    NDJSONAdapter().write([record, record], args.output / "records.ndjson")
    assert record.semantic_equal(JSONAdapter().read(args.output / "record.json"))
    assert stream.semantic_equal(CSVAdapter().read(args.output / "signal.csv"))
    converted = stream.convert_units("V")
    assert converted.metadata.lineage[-1] == stream.metadata.provenance
    print(
        {
            "record": record.record_id,
            "samples": stream.n_samples,
            "schema_version": record.to_dict()["schema_version"],
            "semantic_roundtrip": True,
            "output": str(args.output),
        }
    )


if __name__ == "__main__":
    main()

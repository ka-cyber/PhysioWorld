"""Arrow/Parquet/HDF5 scientific tables with canonical metadata envelopes.

Samples are native numeric arrays/tables, not blobs masquerading as tables.
These readers require the PhysioWorld envelope rather than guessing the units
or clocks of arbitrary external tables.
"""

from __future__ import annotations

import json
from pathlib import Path

from physioworld.physiology.record import PhysiologicalRecord
from physioworld.physiology.stream import SignalStream

from .canonical import decode_json, from_dict, pack, to_dict
from .optional import require

KEY = b"physioworld.canonical"

SampleTable = dict[str, tuple[list[float], list[list[float]]]]


def _split(obj):
    payload = to_dict(obj)
    streams = {"": obj} if isinstance(obj, SignalStream) else obj.streams
    metadata = (
        {"": payload["data"]} if isinstance(obj, SignalStream) else payload["data"]["streams"]
    )
    for data in metadata.values():
        data["timestamps"], data["values"] = pack(()), pack(())
    return json.dumps(payload, allow_nan=False), streams


def _join(header, samples):
    payload = decode_json(header)
    metadata = (
        {"": payload["data"]} if payload["object_type"] == "signal" else payload["data"]["streams"]
    )
    if set(metadata) != set(samples):
        raise ValueError("container streams disagree with canonical metadata")
    for name, (times, rows) in samples.items():
        metadata[name]["timestamps"] = pack(tuple(float(t) for t in times))
        metadata[name]["values"] = pack(tuple(tuple(float(v) for v in row) for row in rows))
    return from_dict(payload)


class ArrowAdapter:
    def write(self, obj: SignalStream | PhysiologicalRecord, path: str | Path) -> None:
        pa = require("pyarrow", "Arrow")
        ipc = require("pyarrow.ipc", "Arrow")
        table = self.to_table(obj, pa)
        with pa.OSFile(str(path), "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    def read(self, path: str | Path):
        pa = require("pyarrow", "Arrow")
        ipc = require("pyarrow.ipc", "Arrow")
        with pa.memory_map(str(path), "r") as source:
            return self.from_table(ipc.open_file(source).read_all())

    @staticmethod
    def to_table(obj, pa=None):
        pa = pa or require("pyarrow", "Arrow")
        header, streams = _split(obj)
        names, times, values = [], [], []
        for name, stream in streams.items():
            names.extend([name] * stream.n_samples)
            times.extend(stream.timestamps)
            values.extend(stream.values)
        table = pa.table(
            {
                "stream": pa.array(names, type=pa.string()),
                "timestamp_s": pa.array(times, type=pa.float64()),
                "values": pa.array(values, type=pa.list_(pa.float64())),
            }
        )
        return table.replace_schema_metadata({KEY: header.encode("utf-8")})

    @staticmethod
    def from_table(table):
        metadata = table.schema.metadata or {}
        if KEY not in metadata:
            raise ValueError("Arrow/Parquet input requires canonical units/clock metadata")
        header = metadata[KEY].decode("utf-8")
        payload = decode_json(header)
        names = [""] if payload["object_type"] == "signal" else list(payload["data"]["streams"])
        samples: SampleTable = {name: ([], []) for name in names}
        if table.column_names != ["stream", "timestamp_s", "values"]:
            raise ValueError("unexpected canonical table columns")
        for row in table.to_pylist():
            if row["stream"] not in samples or row["timestamp_s"] is None or row["values"] is None:
                raise ValueError("malformed canonical table row")
            times, values = samples[row["stream"]]
            times.append(row["timestamp_s"])
            values.append(row["values"])
        return _join(header, samples)


class ParquetAdapter(ArrowAdapter):
    def write(self, obj, path: str | Path) -> None:
        pq = require("pyarrow.parquet", "Parquet")
        pq.write_table(self.to_table(obj), str(path))

    def read(self, path: str | Path):
        pq = require("pyarrow.parquet", "Parquet")
        return self.from_table(pq.read_table(str(path)))


class HDF5Adapter:
    def write(self, obj, path: str | Path) -> None:
        h5py = require("h5py", "HDF5")
        np = require("numpy", "HDF5", "scientific")
        header, streams = _split(obj)
        with h5py.File(str(path), "w") as handle:
            handle.attrs["physioworld.canonical"] = header
            for index, (name, stream) in enumerate(streams.items()):
                group = handle.create_group(f"stream_{index}")
                group.attrs["name"] = name
                group.create_dataset("timestamps_s", data=np.array(stream.timestamps, dtype=float))
                group.create_dataset(
                    "values",
                    data=np.array(stream.values, dtype=float).reshape(
                        stream.n_samples, stream.n_channels
                    ),
                )

    def read(self, path: str | Path):
        h5py = require("h5py", "HDF5")
        with h5py.File(str(path), "r") as handle:
            if "physioworld.canonical" not in handle.attrs:
                raise ValueError("HDF5 input requires canonical units/clock metadata")
            samples = {}
            for group in handle.values():
                name = group.attrs["name"]
                if name in samples:
                    raise ValueError("duplicate HDF5 stream")
                samples[name] = (group["timestamps_s"][:], group["values"][:])
            return _join(handle.attrs["physioworld.canonical"], samples)

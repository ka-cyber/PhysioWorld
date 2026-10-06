"""Small explicit affine conversion registry. Unknown UCUM codes are retained.

Retention is not conversion support: dimensionally incompatible/unknown codes
cannot be converted. No unit is inferred from a channel name.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from physioworld.core import Provenance
from physioworld.physiology.stream import SignalStream
from physioworld.physiology.time import CalibrationMetadata

# unit -> (dimension, scale to base, offset to base)
UNIT_REGISTRY = {
    "V": ("voltage", 1.0, 0.0),
    "mV": ("voltage", 1e-3, 0.0),
    "uV": ("voltage", 1e-6, 0.0),
    "Pa": ("pressure", 1.0, 0.0),
    "kPa": ("pressure", 1e3, 0.0),
    "mm[Hg]": ("pressure", 133.322387415, 0.0),
    "mmHg": ("pressure", 133.322387415, 0.0),
    "K": ("temperature", 1.0, 0.0),
    "Cel": ("temperature", 1.0, 273.15),
    "[degF]": ("temperature", 5 / 9, 255.3722222222222),
    "s": ("time", 1.0, 0.0),
    "ms": ("time", 1e-3, 0.0),
    "Hz": ("frequency", 1.0, 0.0),
    "/min": ("frequency", 1 / 60, 0.0),
    "1": ("dimensionless", 1.0, 0.0),
    "%": ("dimensionless", 0.01, 0.0),
    "m": ("length", 1.0, 0.0),
    "cm": ("length", 0.01, 0.0),
    "kg": ("mass", 1.0, 0.0),
    "g": ("mass", 0.001, 0.0),
}


def conversion(source: str, target: str) -> tuple[float, float]:
    if source not in UNIT_REGISTRY or target not in UNIT_REGISTRY:
        raise ValueError(f"unsupported unit conversion {source!r} -> {target!r}")
    dim, scale, offset = UNIT_REGISTRY[source]
    target_dim, target_scale, target_offset = UNIT_REGISTRY[target]
    if dim != target_dim:
        raise ValueError(f"incompatible units {source!r} and {target!r}")
    return scale / target_scale, (offset - target_offset) / target_scale


def convert_value(value: float, source: str, target: str) -> float:
    scale, offset = conversion(source, target)
    return scale * value + offset


def _calibration(
    cal: CalibrationMetadata | None, scale: float, offset: float, target: str
) -> CalibrationMetadata | None:
    if cal is None:
        return None
    uncertainty = dict(cal.uncertainty)
    if "output_std" in uncertainty:
        uncertainty["output_std"] *= abs(scale)
    if "output_variance" in uncertainty:
        uncertainty["output_variance"] *= scale**2
    if "parameter_covariance" in uncertainty:
        uncertainty["parameter_covariance"] = [
            [v * scale**2 for v in row] for row in uncertainty["parameter_covariance"]
        ]
    return replace(
        cal,
        gain=cal.gain * scale,
        offset=cal.offset * scale + offset,
        saturation_low=cal.saturation_low * scale + offset
        if cal.saturation_low is not None
        else None,
        saturation_high=cal.saturation_high * scale + offset
        if cal.saturation_high is not None
        else None,
        output_unit=target,
        uncertainty=uncertainty,
    )


def convert_stream_units(stream: SignalStream, units: str | Sequence[str]) -> SignalStream:
    targets = (units,) * stream.n_channels if isinstance(units, str) else tuple(units)
    if len(targets) != stream.n_channels:
        raise ValueError("one target unit per channel is required")
    maps = tuple(conversion(c.unit, u) for c, u in zip(stream.metadata.channels, targets))
    if stream.metadata.calibration and len(set(maps)) > 1:
        raise ValueError("heterogeneous conversion requires per-channel calibration")
    channels = tuple(
        replace(c, unit=u, calibration=_calibration(c.calibration, a, b, u))
        for c, u, (a, b) in zip(stream.metadata.channels, targets, maps)
    )
    values = tuple(tuple(a * x + b for x, (a, b) in zip(row, maps)) for row in stream.values)
    uncertainty = tuple(
        tuple(abs(a) * x for x, (a, _) in zip(row, maps)) for row in stream.metadata.uncertainty
    )
    cal = stream.metadata.calibration
    if cal and maps:
        cal = _calibration(cal, *maps[0], targets[0])
    md = replace(
        stream.metadata,
        channels=channels,
        calibration=cal,
        uncertainty=uncertainty,
        lineage=stream._lineage(),
        provenance=Provenance(
            "physioworld",
            "unit_conversion",
            notes={
                "source_units": tuple(c.unit for c in stream.metadata.channels),
                "target_units": targets,
                "registry": "physioworld.affine_units.v1",
            },
        ),
    )
    return replace(stream, values=values, metadata=md)

"""Dependency-free reference signal algorithms with explicit sample semantics.

No function imputes missing samples. Filters are causal (zero initial state for
IIR, partial startup windows for moving averages). Spectral windows are periodic;
PSD is one-sided density, normalized by fs * sum(window**2).
"""

from __future__ import annotations

import bisect
import cmath
import math
import statistics
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Sequence

from physioworld.core import Provenance
from physioworld.physiology.stream import ChannelMetadata, SignalStream

from .units import unit


def samples(stream: SignalStream, channel: int | str = 0) -> tuple[float, ...]:
    if len(stream.metadata.channels) != stream.n_channels:
        raise ValueError("channel metadata width must match signal width")
    values = stream.channel(channel)
    index = (
        channel
        if isinstance(channel, int)
        else [c.name for c in stream.metadata.channels].index(channel)
    )
    channel_missing = any(row[index] for row in stream.metadata.missing_channels)
    if (
        not values
        or channel_missing
        or any(stream.metadata.missing)
        or not all(map(math.isfinite, values))
    ):
        raise ValueError("analysis requires nonempty finite samples with no missing flags")
    if not all(map(math.isfinite, stream.timestamps)) or any(
        b <= a for a, b in zip(stream.timestamps, stream.timestamps[1:])
    ):
        raise ValueError("analysis requires finite, strictly increasing timestamps")
    return values


def uniform_rate(stream: SignalStream) -> float:
    samples(stream)
    fs = stream.sample_rate
    if fs is None or not math.isfinite(fs) or fs <= 0:
        raise ValueError("a uniform sample rate must be declared")
    if any(
        not math.isclose(b - a, 1 / fs, rel_tol=1e-6, abs_tol=1e-10)
        for a, b in zip(stream.timestamps, stream.timestamps[1:])
    ):
        raise ValueError("spectral/filter operation requires uniform samples; resample explicitly")
    return fs


def same_clock(left: SignalStream, right: SignalStream) -> bool:
    """Clock identity/map equality; uncertainty is evidence, not a clock identity."""
    fields = (
        "clock_domain",
        "origin",
        "wall_clock_start_s",
        "clock_id",
        "clock_offset_s",
        "sampling_drift_ppm",
    )
    return all(getattr(left.metadata.clock, f) == getattr(right.metadata.clock, f) for f in fields)


def derived(
    stream: SignalStream,
    rows: Sequence[Sequence[float]],
    operation: str,
    *,
    units: Sequence[str] | None = None,
    **parameters: Any,
) -> SignalStream:
    channels = stream.metadata.channels
    if units is not None:
        channels = tuple(
            replace(
                c,
                unit=u,
                unit_system="physioworld_multiplicative",
                calibration=None if c.unit != u else c.calibration,
            )
            for c, u in zip(channels, units)
        )
        if stream.metadata.calibration:
            parameters["input_calibration"] = asdict(stream.metadata.calibration)
    if stream.metadata.uncertainty:
        parameters["uncertainty_policy"] = "not_propagated_by_this_algorithm"
    parent = stream.metadata.provenance
    metadata = replace(
        stream.metadata,
        channels=channels,
        uncertainty=(),
        calibration=None if units is not None else stream.metadata.calibration,
        lineage=stream.metadata.lineage + ((parent,) if parent else ()),
        provenance=Provenance(
            "physioworld.instrumentation",
            operation,
            parent_ids=(stream.metadata.record_id or "anonymous",),
            notes=parameters,
        ),
    )
    return replace(stream, values=tuple(tuple(row) for row in rows), metadata=metadata)


def convert(stream: SignalStream, target_unit: str) -> SignalStream:
    factors = [unit(c.unit).factor_to(unit(target_unit)) for c in stream.metadata.channels]
    result = derived(
        stream,
        [[v * f for v, f in zip(row, factors)] for row in stream.values],
        "convert_units",
        units=[target_unit] * stream.n_channels,
        target=target_unit,
    )
    if stream.metadata.uncertainty:
        provenance = result.metadata.provenance
        if provenance is None:
            raise RuntimeError("derived unit conversion lost its provenance")
        provenance = replace(
            provenance,
            notes={"target": target_unit, "uncertainty_policy": "standard_deviation_scaled"},
        )
        result = replace(
            result,
            metadata=replace(
                result.metadata,
                provenance=provenance,
                uncertainty=tuple(
                    tuple(v * abs(f) for v, f in zip(row, factors))
                    for row in stream.metadata.uncertainty
                ),
            ),
        )
    return result


def arithmetic(left: SignalStream, right: SignalStream, operation: str) -> SignalStream:
    if (
        left.timestamps != right.timestamps
        or left.n_channels != right.n_channels
        or not same_clock(left, right)
    ):
        raise ValueError("arithmetic requires explicit clock/sample/channel alignment")
    for i in range(left.n_channels):
        samples(left, i)
        samples(right, i)
    units = []
    for a, b in zip(left.metadata.channels, right.metadata.channels):
        ua, ub = unit(a.unit), unit(b.unit)
        if operation in {"add", "subtract"}:
            if ua != ub:
                raise ValueError("add/subtract requires identical units; convert_units explicitly")
            units.append(a.unit)
        else:
            units.append(str(ua * ub if operation == "multiply" else ua / ub))
    ops = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b,
    }
    if operation not in ops:
        raise ValueError(operation)
    return derived(
        left,
        [[ops[operation](a, b) for a, b in zip(x, y)] for x, y in zip(left.values, right.values)],
        operation,
        units=units,
    )


def smooth(stream: SignalStream, window: int) -> SignalStream:
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive sample count")
    columns = [samples(stream, j) for j in range(stream.n_channels)]
    rows = [
        [sum(col[max(0, i - window + 1) : i + 1]) / min(i + 1, window) for col in columns]
        for i in range(stream.n_samples)
    ]
    return derived(stream, rows, "smooth", window=window, startup="partial", phase="causal")


def filter_signal(stream: SignalStream, cutoff_hz: float, kind: str = "lowpass") -> SignalStream:
    fs = uniform_rate(stream)
    if not 0 < cutoff_hz < fs / 2 or kind not in {"lowpass", "highpass"}:
        raise ValueError("one-pole filter needs 0 < cutoff < Nyquist and lowpass/highpass")
    # Bilinear-transform first-order Butterworth; frequency prewarping.
    k = math.tan(math.pi * cutoff_hz / fs)
    b0 = k / (1 + k) if kind == "lowpass" else 1 / (1 + k)
    b1 = b0 if kind == "lowpass" else -b0
    a1 = (k - 1) / (k + 1)
    columns = []
    for j in range(stream.n_channels):
        out, previous_x, previous_y = [], 0.0, 0.0
        for x in samples(stream, j):
            y = b0 * x + b1 * previous_x - a1 * previous_y
            out.append(y)
            previous_x, previous_y = x, y
        columns.append(out)
    return derived(
        stream,
        list(zip(*columns)),
        "filter",
        kind=kind,
        cutoff_hz=cutoff_hz,
        algorithm="bilinear_first_order_butterworth",
        initial_state="zero",
    )


def derivative(stream: SignalStream) -> SignalStream:
    if stream.n_samples < 2:
        raise ValueError("derivative needs at least two samples")
    columns = []
    t = stream.timestamps
    for j in range(stream.n_channels):
        x = samples(stream, j)
        # Three-point nonuniform central difference, first-order one-sided endpoints.
        out = [(x[1] - x[0]) / (t[1] - t[0])]
        for i in range(1, len(x) - 1):
            a, b = t[i] - t[i - 1], t[i + 1] - t[i]
            out.append(
                -b * x[i - 1] / (a * (a + b))
                + (b - a) * x[i] / (a * b)
                + a * x[i + 1] / (b * (a + b))
            )
        out.append((x[-1] - x[-2]) / (t[-1] - t[-2]))
        columns.append(out)
    return derived(
        stream,
        list(zip(*columns)),
        "derivative",
        units=[str(unit(c.unit) / unit("s")) for c in stream.metadata.channels],
        algorithm="nonuniform_three_point",
        endpoints="one_sided",
    )


def integral(stream: SignalStream, initial: float = 0.0) -> SignalStream:
    if not math.isfinite(initial):
        raise ValueError("initial integral must be finite and in the output unit")
    columns = []
    for j in range(stream.n_channels):
        x = samples(stream, j)
        out = [initial]
        for i in range(1, len(x)):
            out.append(
                out[-1]
                + (x[i] + x[i - 1]) * 0.5 * (stream.timestamps[i] - stream.timestamps[i - 1])
            )
        columns.append(out)
    return derived(
        stream,
        list(zip(*columns)),
        "integral",
        units=[str(unit(c.unit) * unit("s")) for c in stream.metadata.channels],
        algorithm="cumulative_trapezoid",
        initial=initial,
    )


def resample(stream: SignalStream, sample_rate: float, method: str = "linear") -> SignalStream:
    for j in range(stream.n_channels):
        samples(stream, j)
    if not math.isfinite(sample_rate) or sample_rate <= 0 or method not in {"linear", "nearest"}:
        raise ValueError("positive rate and linear/nearest method required")
    # No extrapolation, no automatic anti-alias filter. Downsampling is explicitly recorded.
    times = tuple(
        stream.timestamps[0] + i / sample_rate
        for i in range(math.floor(stream.duration * sample_rate + 1e-9) + 1)
    )
    rows = []
    for t in times:
        right = min(bisect.bisect_left(stream.timestamps, t), stream.n_samples - 1)
        left = max(0, right - 1)
        if right == left or method == "nearest":
            idx = left if t - stream.timestamps[left] <= stream.timestamps[right] - t else right
            rows.append(stream.values[idx])
        else:
            f = (t - stream.timestamps[left]) / (stream.timestamps[right] - stream.timestamps[left])
            rows.append(
                tuple(a + f * (b - a) for a, b in zip(stream.values[left], stream.values[right]))
            )
    metadata = replace(
        stream.metadata,
        quality=(),
        missing=(),
        missing_channels=(),
        uncertainty=(),
        sampling_semantics="regular",
        provenance=Provenance(
            "physioworld.instrumentation",
            "resample",
            notes={
                "method": method,
                "original_rate": stream.sample_rate,
                "rate": sample_rate,
                "anti_alias_filter": None,
                "uncertainty_policy": "not_propagated",
                "quality_policy": "not_interpolated",
            },
        ),
    )
    return replace(
        stream, timestamps=times, values=tuple(rows), sample_rate=sample_rate, metadata=metadata
    )


def time_shift(stream: SignalStream, seconds: float) -> SignalStream:
    if not math.isfinite(seconds) or (
        stream.annotations and min(a.onset for a in stream.annotations) + seconds < 0
    ):
        raise ValueError("shift must be finite and annotation times nonnegative")
    result = derived(stream, stream.values, "time_shift", seconds=seconds)
    return replace(
        result,
        timestamps=tuple(t + seconds for t in stream.timestamps),
        annotations=tuple(replace(a, onset=a.onset + seconds) for a in stream.annotations),
    )


def _fft(x: Sequence[complex]) -> tuple[complex, ...]:
    n = len(x)
    if n == 1:
        return (complex(x[0]),)
    if n & (n - 1):
        return tuple(
            sum(v * cmath.exp(-2j * math.pi * k * i / n) for i, v in enumerate(x)) for k in range(n)
        )
    even, odd = _fft(x[::2]), _fft(x[1::2])
    terms = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return tuple(even[k] + terms[k] for k in range(n // 2)) + tuple(
        even[k] - terms[k] for k in range(n // 2)
    )


@dataclass(frozen=True)
class Spectrum:
    frequencies: tuple[float, ...]
    values: tuple[complex | float, ...]
    unit: str
    sample_rate: float
    n_samples: int
    kind: str
    provenance: Provenance


def spectrum(
    stream: SignalStream,
    kind: str = "psd",
    channel: int = 0,
    window: str = "hann",
    detrend: bool = False,
) -> Spectrum:
    fs, x = uniform_rate(stream), samples(stream, channel)
    n = len(x)
    if n < 2 or kind not in {"fft", "psd"} or window not in {"hann", "rectangular"}:
        raise ValueError("FFT/PSD requires >=2 samples and hann/rectangular window")
    w = tuple(
        0.5 - 0.5 * math.cos(2 * math.pi * i / n) if window == "hann" else 1.0 for i in range(n)
    )
    mean = statistics.fmean(x) if detrend else 0.0
    bins = _fft(tuple((v - mean) * a for v, a in zip(x, w)))[: n // 2 + 1]
    doubles = [1 if k == 0 or (n % 2 == 0 and k == n // 2) else 2 for k in range(len(bins))]
    if kind == "fft":
        values = tuple(z * d / sum(w) for z, d in zip(bins, doubles))
        output_unit = stream.metadata.channels[channel].unit
    else:
        values = tuple(
            abs(z) ** 2 * d / (fs * sum(a * a for a in w)) for z, d in zip(bins, doubles)
        )
        output_unit = str(unit(stream.metadata.channels[channel].unit) ** 2 / unit("Hz"))
    return Spectrum(
        tuple(k * fs / n for k in range(len(bins))),
        values,
        output_unit,
        fs,
        n,
        kind,
        Provenance(
            "physioworld.instrumentation",
            kind,
            notes={
                "window": window,
                "detrend": detrend,
                "one_sided": True,
                "normalization": "coherent_amplitude" if kind == "fft" else "density",
            },
        ),
    )


@dataclass(frozen=True)
class Spectrogram:
    times: tuple[float, ...]
    spectra: tuple[Spectrum, ...]
    provenance: Provenance


def spectrogram(
    stream: SignalStream,
    window_samples: int,
    hop_samples: int,
    channel: int = 0,
    window: str = "hann",
    detrend: bool = False,
) -> Spectrogram:
    uniform_rate(stream)
    if window_samples < 2 or hop_samples < 1 or window_samples > stream.n_samples:
        raise ValueError("invalid spectrogram sample window/hop")
    spectra, times = [], []
    for start in range(0, stream.n_samples - window_samples + 1, hop_samples):
        sub = replace(
            stream,
            timestamps=stream.timestamps[start : start + window_samples],
            values=stream.values[start : start + window_samples],
            metadata=replace(
                stream.metadata,
                missing=(),
                quality=(),
                missing_channels=stream.metadata.missing_channels[start : start + window_samples],
                uncertainty=stream.metadata.uncertainty[start : start + window_samples],
            ),
        )
        spectra.append(spectrum(sub, "psd", channel, window, detrend))
        times.append((sub.timestamps[0] + sub.timestamps[-1]) / 2)
    return Spectrogram(
        tuple(times),
        tuple(spectra),
        Provenance(
            "physioworld.instrumentation",
            "spectrogram",
            notes={
                "window_samples": window_samples,
                "hop_samples": hop_samples,
                "time_reference": "sample_span_center",
                "padding": None,
            },
        ),
    )


@dataclass(frozen=True)
class EventSeries:
    times: tuple[float, ...]
    indices: tuple[int, ...]
    label: str
    provenance: Provenance


def detect_events(
    stream: SignalStream,
    threshold: float,
    refractory_s: float = 0.0,
    channel: int = 0,
    mode: str = "rising",
) -> EventSeries:
    x = samples(stream, channel)
    if not math.isfinite(threshold) or refractory_s < 0 or mode not in {"rising", "peaks"}:
        raise ValueError("invalid event detection parameters")
    indices: list[int] = []
    for i in range(1, len(x) - (1 if mode == "peaks" else 0)):
        hit = (
            x[i - 1] < threshold <= x[i]
            if mode == "rising"
            else (x[i] >= threshold and x[i] > x[i - 1] and x[i] >= x[i + 1])
        )
        if hit and (
            not indices or stream.timestamps[i] - stream.timestamps[indices[-1]] >= refractory_s
        ):
            indices.append(i)
    return EventSeries(
        tuple(stream.timestamps[i] for i in indices),
        tuple(indices),
        mode,
        Provenance(
            "physioworld.instrumentation",
            "detect_events",
            notes={"threshold": threshold, "refractory_s": refractory_s, "mode": mode},
        ),
    )


def cycles(events: EventSeries) -> tuple[tuple[float, float], ...]:
    return tuple(zip(events.times, events.times[1:]))


def autocorrelation(
    stream: SignalStream, max_lag: int, channel: int = 0, demean: bool = True
) -> SignalStream:
    fs, x = uniform_rate(stream), samples(stream, channel)
    if not 0 <= max_lag < len(x):
        raise ValueError("max_lag outside sample range")
    mean = statistics.fmean(x) if demean else 0
    y = [v - mean for v in x]
    energy = sum(v * v for v in y)
    if energy == 0:
        raise ValueError("zero-energy autocorrelation is undefined")
    values = [sum(y[i] * y[i + k] for i in range(len(y) - k)) / energy for k in range(max_lag + 1)]
    return SignalStream.from_samples(
        values,
        fs,
        "1",
        "autocorrelation",
        provenance=Provenance(
            "physioworld.instrumentation",
            "autocorrelation",
            notes={"demean": demean, "normalization": "biased_normalized", "axis": "lag_seconds"},
        ),
    )


def template_match(
    stream: SignalStream, template: Sequence[float], channel: int = 0
) -> SignalStream:
    x = samples(stream, channel)
    template = tuple(float(v) for v in template)
    n = len(template)
    if n < 2 or n > len(x) or not all(map(math.isfinite, template)):
        raise ValueError("template must have 2..N finite samples")
    mean = statistics.fmean(template)
    y = [v - mean for v in template]
    energy = sum(v * v for v in y)
    if energy == 0:
        raise ValueError("constant template has undefined normalized correlation")
    values = []
    for i in range(len(x) - n + 1):
        chunk = x[i : i + n]
        avg = statistics.fmean(chunk)
        den = math.sqrt(energy * sum((v - avg) ** 2 for v in chunk))
        values.append((sum((a - avg) * b for a, b in zip(chunk, y)) / den,) if den else (math.nan,))
    metadata = replace(
        stream.metadata,
        channels=(ChannelMetadata("template_score", "1"),),
        missing=tuple(not math.isfinite(v[0]) for v in values),
        quality=(),
        missing_channels=(),
        uncertainty=(),
        provenance=Provenance(
            "physioworld.instrumentation",
            "template_match",
            notes={"template": template, "timestamp_reference": "window_start"},
        ),
    )
    return replace(
        stream,
        timestamps=stream.timestamps[: len(values)],
        values=tuple(values),
        metadata=metadata,
        annotations=(),
    )


def event_average(
    stream: SignalStream,
    events: EventSeries,
    before_samples: int,
    after_samples: int,
    channel: int = 0,
) -> SignalStream:
    fs, x = uniform_rate(stream), samples(stream, channel)
    if before_samples < 0 or after_samples < 0:
        raise ValueError("sample extents must be nonnegative")
    chunks = []
    for t in events.times:
        i = bisect.bisect_left(stream.timestamps, t)
        if i >= len(x) or not math.isclose(stream.timestamps[i], t, abs_tol=1e-9):
            raise ValueError("averaging events must align to actual sample times")
        if i - before_samples < 0 or i + after_samples >= len(x):
            raise ValueError("event window exceeds recording; no implicit partial averaging")
        chunks.append(x[i - before_samples : i + after_samples + 1])
    if not chunks:
        raise ValueError("event averaging requires events")
    values = [statistics.fmean(v) for v in zip(*chunks)]
    return SignalStream.from_samples(
        values,
        fs,
        stream.metadata.channels[channel].unit,
        "event_average",
        timestamps=[(i - before_samples) / fs for i in range(len(values))],
        provenance=Provenance(
            "physioworld.instrumentation",
            "event_average",
            notes={"event_times": events.times, "count": len(chunks), "axis": "relative_seconds"},
        ),
    )


@dataclass(frozen=True)
class Measurements:
    values: dict[str, float]
    units: dict[str, str]
    provenance: Provenance
    limitations: tuple[str, ...] = ()


def statistics_extract(stream: SignalStream, channel: int = 0) -> Measurements:
    x = samples(stream, channel)
    u = stream.metadata.channels[channel].unit
    vals = {
        "mean": statistics.fmean(x),
        "min": min(x),
        "max": max(x),
        "rms": math.sqrt(statistics.fmean(v * v for v in x)),
        "std": statistics.pstdev(x),
        "peak_to_peak": max(x) - min(x),
    }
    return Measurements(
        vals,
        {k: u for k in vals},
        Provenance(
            "physioworld.instrumentation", "statistics", notes={"std": "population", "n": len(x)}
        ),
    )


def quality(stream: SignalStream, channel: int = 0) -> Measurements:
    x = stream.channel(channel)
    if not x:
        raise ValueError("quality requires samples")
    index = (
        channel
        if isinstance(channel, int)
        else [c.name for c in stream.metadata.channels].index(channel)
    )
    missing = tuple(
        a or b
        for a, b in zip(
            stream.metadata.missing or (False,) * len(x),
            tuple(row[index] for row in stream.metadata.missing_channels) or (False,) * len(x),
        )
    )
    valid = [v for v, m in zip(x, missing) if not m and math.isfinite(v)]
    vals = {
        "finite_fraction": sum(math.isfinite(v) for v in x) / len(x),
        "valid_fraction": len(valid) / len(x),
        "flagged_missing_fraction": sum(missing) / len(x),
    }
    if valid:
        vals["flat_fraction"] = sum(a == b for a, b in zip(valid, valid[1:])) / max(
            1, len(valid) - 1
        )
    return Measurements(
        vals,
        {k: "1" for k in vals},
        Provenance("physioworld.instrumentation", "quality"),
        ("Descriptive metrics, not diagnostic SQI.",),
    )


def explicit_transform(
    stream: SignalStream, function: Callable[[float], float], output_unit: str, name: str
) -> SignalStream:
    if not name:
        raise ValueError("custom transform must have a declared name")
    unit(output_unit)
    for j in range(stream.n_channels):
        samples(stream, j)
    rows = [[function(v) for v in row] for row in stream.values]
    if not all(math.isfinite(v) for row in rows for v in row):
        raise ValueError("explicit transform produced nonfinite output")
    return derived(
        stream, rows, "explicit_transform", units=[output_unit] * stream.n_channels, name=name
    )

"""Transparent physiology measurements, not validated diagnostic classifiers.

Thresholds and refractory periods are caller choices. Artifact removal, normal
beat labeling, calibration, filtering and RR interpolation are never implicit.
"""

from __future__ import annotations

import math
import statistics

from physioworld.core import Provenance
from physioworld.physiology.stream import SignalStream

from .analysis import EventSeries, Measurements, detect_events, samples, spectrum, uniform_rate
from .units import unit


def _real_spectral_value(value: complex | float) -> float:
    """Extract a real spectral scalar, rejecting unexpected complex leakage."""
    if isinstance(value, complex):
        if value.imag != 0:
            raise ValueError("power spectrum contains a non-real value")
        return value.real
    return value


def ecg_peaks(
    stream: SignalStream, threshold: float, refractory_s: float = 0.25, channel: int = 0
) -> EventSeries:
    unit(stream.metadata.channels[channel].unit).factor_to(unit("V"))
    uniform_rate(stream)
    result = detect_events(stream, threshold, refractory_s, channel, "peaks")
    return EventSeries(
        result.times,
        result.indices,
        "candidate_R_peaks",
        Provenance(
            "physioworld.instrumentation",
            "ecg_local_maxima",
            notes={
                "threshold": threshold,
                "refractory_s": refractory_s,
                "limits": "Positive local maxima only; no QRS morphology or artifact classifier.",
            },
        ),
    )


def rr_intervals(peaks: EventSeries) -> SignalStream:
    if len(peaks.times) < 2 or any(b <= a for a, b in zip(peaks.times, peaks.times[1:])):
        raise ValueError("RR requires at least two strictly ordered peaks")
    return SignalStream.from_samples(
        [b - a for a, b in zip(peaks.times, peaks.times[1:])],
        None,
        "s",
        "RR",
        timestamps=peaks.times[1:],
        provenance=Provenance(
            "physioworld.instrumentation",
            "rr_intervals",
            notes={
                "peak_times": peaks.times,
                "timestamp_reference": "second_peak",
                "normal_beats_assumed": False,
            },
        ),
    )


def hrv(rr: SignalStream, *, normal_intervals: bool = False) -> Measurements:
    if unit(rr.metadata.channels[0].unit) != unit("s"):
        raise ValueError("convert RR intervals explicitly to seconds")
    x = samples(rr)
    if len(x) < 2 or min(x) <= 0:
        raise ValueError("HRV needs at least two positive intervals")
    diffs = [b - a for a, b in zip(x, x[1:])]
    sd_label = "SDNN" if normal_intervals else "SD_RR"
    vals = {
        "mean_RR": statistics.fmean(x),
        "mean_HR": statistics.fmean(60 / v for v in x),
        sd_label: statistics.stdev(x),
        "RMSSD": math.sqrt(statistics.fmean(d * d for d in diffs)),
        "pNN50" if normal_intervals else "pRR50": sum(abs(d) > 0.05 for d in diffs) / len(diffs),
    }
    units = {k: "s" for k in vals}
    units["mean_HR"] = "bpm"
    units["pNN50" if normal_intervals else "pRR50"] = "1"
    return Measurements(
        vals,
        units,
        Provenance(
            "physioworld.instrumentation",
            "time_domain_hrv",
            notes={"normal_intervals": normal_intervals, "artifact_correction": None},
        ),
        (
            "Caller must validate peak detection and label normal beats for NN metrics.",
            "Short recordings do not support clinical HRV interpretation; no spectral RR resampling.",
        ),
    )


def blood_pressure(stream: SignalStream, beats: EventSeries, channel: int = 0) -> Measurements:
    if unit(stream.metadata.channels[channel].unit) != unit("mmHg"):
        raise ValueError("blood pressure requires explicitly calibrated mmHg")
    x = samples(stream, channel)
    systolic, diastolic, means = [], [], []
    for start, stop in zip(beats.times, beats.times[1:]):
        idx = [i for i, t in enumerate(stream.timestamps) if start <= t <= stop]
        if len(idx) < 2 or stream.timestamps[idx[0]] != start or stream.timestamps[idx[-1]] != stop:
            raise ValueError("BP cycle boundaries must align to samples and span >=2 samples")
        v = [x[i] for i in idx]
        systolic.append(max(v))
        diastolic.append(min(v))
        area = sum(
            (x[a] + x[b]) * 0.5 * (stream.timestamps[b] - stream.timestamps[a])
            for a, b in zip(idx, idx[1:])
        )
        means.append(area / (stop - start))
    if not means:
        raise ValueError("BP requires at least one complete beat")
    vals = {
        "systolic_mean": statistics.fmean(systolic),
        "diastolic_mean": statistics.fmean(diastolic),
        "MAP": statistics.fmean(means),
    }
    return Measurements(
        vals,
        {k: "mmHg" for k in vals},
        Provenance(
            "physioworld.instrumentation",
            "cycle_blood_pressure",
            notes={"n_cycles": len(means), "MAP_method": "time_weighted_trapezoid"},
        ),
        ("Requires calibrated pressure, complete externally identified beats; no cuff estimator.",),
    )


def respiration(
    stream: SignalStream, threshold: float = 0.0, refractory_s: float = 1.0, channel: int = 0
) -> Measurements:
    events = detect_events(stream, threshold, refractory_s, channel)
    intervals = [b - a for a, b in zip(events.times, events.times[1:])]
    if not intervals:
        raise ValueError("respiration requires at least two threshold crossings")
    return Measurements(
        {"rate": 60 / statistics.fmean(intervals), "period": statistics.fmean(intervals)},
        {"rate": "bpm", "period": "s"},
        Provenance(
            "physioworld.instrumentation",
            "respiration_crossings",
            notes={"threshold": threshold, "refractory_s": refractory_s},
        ),
        ("One rising threshold crossing per breath assumed; no apnea or tidal-volume diagnosis.",),
    )


def emg(stream: SignalStream, channel: int = 0) -> Measurements:
    unit(stream.metadata.channels[channel].unit).factor_to(unit("V"))
    x = samples(stream, channel)
    psd = spectrum(stream, "psd", channel, "hann", True)
    total = sum(_real_spectral_value(p) for p in psd.values)
    if total <= 0:
        raise ValueError("EMG spectral median is undefined for zero demeaned power")
    cumulative, median = 0.0, 0.0
    for f, p in zip(psd.frequencies, psd.values):
        cumulative += _real_spectral_value(p)
        if cumulative >= total / 2:
            median = f
            break
    return Measurements(
        {
            "RMS": math.sqrt(statistics.fmean(v * v for v in x)),
            "mean_absolute": statistics.fmean(abs(v) for v in x),
            "median_frequency": median,
        },
        {
            "RMS": stream.metadata.channels[channel].unit,
            "mean_absolute": stream.metadata.channels[channel].unit,
            "median_frequency": "Hz",
        },
        Provenance(
            "physioworld.instrumentation", "emg", notes={"spectral_window": "hann", "detrend": True}
        ),
        ("No automatic bandpass, rectification envelope, or fatigue diagnosis.",),
    )


def eeg(
    stream: SignalStream, channel: int = 0, bands: dict[str, tuple[float, float]] | None = None
) -> Measurements:
    unit(stream.metadata.channels[channel].unit).factor_to(unit("V"))
    bands = bands or {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    psd = spectrum(stream, "psd", channel, "hann", True)
    df = psd.sample_rate / psd.n_samples
    vals: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        if not 0 <= lo < hi <= psd.sample_rate / 2:
            raise ValueError("EEG bands must lie within acquired Nyquist range")
        vals[name] = sum(
            _real_spectral_value(p) * df
            for f, p in zip(psd.frequencies, psd.values)
            if lo <= f < hi
        )
    return Measurements(
        vals,
        {k: str(unit(stream.metadata.channels[channel].unit) ** 2) for k in vals},
        Provenance(
            "physioworld.instrumentation",
            "eeg_band_power",
            notes={
                "bands": bands,
                "bins": "left_closed_right_open",
                "detrend": True,
                "window": "hann",
            },
        ),
        ("Single-window periodogram, no artifact rejection or neurological classification.",),
    )


def pressure_volume_work(pressure: SignalStream, volume: SignalStream) -> Measurements:
    from .analysis import arithmetic

    # Checks timestamps, clocks and shape without converting units implicitly.
    arithmetic(pressure, volume, "multiply")
    if pressure.n_channels != 1:
        raise ValueError("PV loop requires scalar channels")
    p, v = samples(pressure), samples(volume)
    unit(pressure.metadata.channels[0].unit).factor_to(unit("Pa"))
    unit(volume.metadata.channels[0].unit).factor_to(unit("L"))
    work = sum(
        (p[i] + p[(i + 1) % len(p)]) * 0.5 * (v[(i + 1) % len(v)] - v[i]) for i in range(len(p))
    )
    return Measurements(
        {"signed_loop_work": work},
        {
            "signed_loop_work": str(
                unit(pressure.metadata.channels[0].unit) * unit(volume.metadata.channels[0].unit)
            )
        },
        Provenance("physioworld.instrumentation", "PV_loop", notes={"closure": "explicit_polygon"}),
        ("Caller supplies one calibrated complete loop; sign follows traversal direction.",),
    )


def gas_exchange(vo2: float, vco2: float, flow_unit: str = "L/min") -> Measurements:
    unit(flow_unit).factor_to(unit("L/min"))
    if not math.isfinite(vo2) or not math.isfinite(vco2) or vo2 <= 0 or vco2 < 0:
        raise ValueError("positive VO2 and nonnegative VCO2 are required")
    return Measurements(
        {"VO2": vo2, "VCO2": vco2, "RER": vco2 / vo2},
        {"VO2": flow_unit, "VCO2": flow_unit, "RER": "1"},
        Provenance("physioworld.instrumentation", "gas_exchange_ratio"),
        ("Inputs must already be calibrated and corrected for declared gas conditions.",),
    )

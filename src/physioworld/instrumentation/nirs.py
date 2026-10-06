"""Exploratory NIRS semantics and explicitly declared proxy operations.

The raw proxy is OD_lambda(t) = -log(I_lambda(t)), with log explicitly natural,
on a caller-supplied
dimensionless intensity scale. There is no implicit normalization, reference
division, baseline subtraction, motion detection, or hemoglobin reconstruction.
The helpers below are declared toy/reference algorithms, not a reconstruction
of a published preprocessing chain or clinically validated NIRS processing.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Sequence

from physioworld.core import EvidenceClass, Provenance
from physioworld.core.contracts import finite
from physioworld.physiology.stream import SignalStream

from .analysis import derived, samples, uniform_rate


class NIRSQuantity(str, Enum):
    INTENSITY = "intensity"
    OPTICAL_DENSITY = "optical_density"
    HBO = "HbO"
    HBR = "HbR"
    HBT = "HbT"


def _validate_stream(stream: SignalStream, quantity: NIRSQuantity) -> None:
    if not isinstance(stream, SignalStream):
        raise TypeError("NIRS semantic types require a SignalStream, not another semantic type")
    if stream.metadata.physical_quantity != quantity.value:
        raise ValueError(
            "NIRS quantity must match explicit stream semantics; relabeling is forbidden"
        )
    if not stream.n_channels:
        raise ValueError("NIRS requires at least one channel")
    for j, channel in enumerate(stream.metadata.channels):
        if channel.attributes.get("nirs_quantity", quantity.value) != quantity.value:
            raise ValueError("channel and stream NIRS semantics disagree")
        values = samples(stream, j)
        if quantity in {NIRSQuantity.INTENSITY, NIRSQuantity.OPTICAL_DENSITY}:
            if channel.unit != "1":
                raise ValueError("intensity scale and optical density must explicitly use unit '1'")
        elif channel.unit not in {"mol/L", "mmol/L", "umol/L", "µmol/L"}:
            raise ValueError("hemoglobin requires an explicit molar concentration unit")
        if quantity == NIRSQuantity.INTENSITY and any(value <= 0 for value in values):
            raise ValueError("NIRS intensity must be finite and strictly positive")


@dataclass(frozen=True)
class IntensitySignal:
    """Positive dimensionless intensity; its supplied numerical scale is retained."""

    stream: SignalStream
    quantity: NIRSQuantity = field(default=NIRSQuantity.INTENSITY, init=False)

    def __post_init__(self) -> None:
        _validate_stream(self.stream, self.quantity)

    @classmethod
    def from_samples(
        cls,
        values: Sequence[Any],
        sample_rate: float | None,
        *,
        timestamps: Sequence[float] | None = None,
        channel_names: Sequence[str] | None = None,
        provenance: Provenance | None = None,
    ) -> IntensitySignal:
        return cls(
            SignalStream.from_samples(
                values,
                sample_rate,
                "1",
                "NIRS",
                timestamps=timestamps,
                channel_names=channel_names,
                physical_quantity=NIRSQuantity.INTENSITY.value,
                provenance=provenance
                or Provenance(
                    "caller",
                    "declared_nirs_intensity",
                    notes={
                        "evidence_class": EvidenceClass.ASSUMPTION.value,
                        "nirs_quantity": NIRSQuantity.INTENSITY.value,
                        "normalization": None,
                    },
                ),
            )
        )


@dataclass(frozen=True)
class OpticalDensitySignal:
    """Dimensionless optical-density proxy, including negative OD when I > 1."""

    stream: SignalStream
    quantity: NIRSQuantity = field(default=NIRSQuantity.OPTICAL_DENSITY, init=False)

    def __post_init__(self) -> None:
        _validate_stream(self.stream, self.quantity)


@dataclass(frozen=True)
class HemoglobinSignal:
    """Already reconstructed, externally supplied HbO/HbR/HbT concentration.

    A source definition and provenance must accompany the concentration. This
    container does not implement or verify that external reconstruction.
    """

    stream: SignalStream
    quantity: NIRSQuantity
    source_definition: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity", NIRSQuantity(self.quantity))
        if self.quantity not in {NIRSQuantity.HBO, NIRSQuantity.HBR, NIRSQuantity.HBT}:
            raise ValueError("hemoglobin semantic type requires HbO, HbR, or HbT")
        _validate_stream(self.stream, self.quantity)
        if not isinstance(self.source_definition, str) or not self.source_definition.strip():
            raise ValueError("hemoglobin requires an external reconstruction source definition")
        if self.stream.metadata.provenance is None:
            raise ValueError("hemoglobin requires external reconstruction provenance")


def _seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def _od_result(
    stream: SignalStream,
    rows: Sequence[Sequence[float]],
    operation: str,
    *,
    seed: int | None = None,
    evidence_class: EvidenceClass = EvidenceClass.DEFINITION,
    **notes: Any,
) -> OpticalDensitySignal:
    result = derived(
        stream,
        rows,
        operation,
        units=["1"] * stream.n_channels,
        evidence_class=evidence_class.value,
        output_quantity=NIRSQuantity.OPTICAL_DENSITY.value,
        hemoglobin_reconstruction=False,
        **notes,
    )
    provenance = result.metadata.provenance
    assert provenance is not None
    return OpticalDensitySignal(
        replace(
            result,
            metadata=replace(
                result.metadata,
                physical_quantity=NIRSQuantity.OPTICAL_DENSITY.value,
                channels=tuple(
                    replace(
                        channel,
                        description="Optical-density proxy, not hemoglobin concentration",
                        attributes={
                            **channel.attributes,
                            "nirs_quantity": NIRSQuantity.OPTICAL_DENSITY.value,
                        },
                    )
                    for channel in result.metadata.channels
                ),
                provenance=replace(
                    provenance, source="physioworld.instrumentation.nirs", seed=seed
                ),
            ),
        )
    )


def optical_density(intensity: IntensitySignal) -> OpticalDensitySignal:
    """Apply exactly OD=-log(I), using natural log; no baseline is implicit."""
    if not isinstance(intensity, IntensitySignal):
        raise TypeError("optical_density requires an IntensitySignal")
    return _od_result(
        intensity.stream,
        [[-math.log(value) for value in row] for row in intensity.stream.values],
        "optical_density",
        formula="OD_lambda(t) = -log(I_lambda(t))",
        logarithm="natural",
        normalization=None,
        reference_intensity=None,
    )


def _require_od(signal: OpticalDensitySignal) -> SignalStream:
    if not isinstance(signal, OpticalDensitySignal):
        raise TypeError("proxy helper requires an OpticalDensitySignal; no HbT promotion")
    return signal.stream


def _indices(indices: Sequence[int], count: int) -> tuple[int, ...]:
    result = tuple(indices)
    if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < count for i in result):
        raise ValueError("motion indices must be integer sample indices within the signal")
    if len(set(result)) != len(result):
        raise ValueError("motion indices must be unique")
    return tuple(sorted(result))


def simulate_motion(
    signal: OpticalDensitySignal, indices: Sequence[int], amplitude: float, *, seed: int
) -> OpticalDensitySignal:
    """Toy additive OD motion: y[i,c] = x[i,c] + U(-amplitude, amplitude).

    Draw independently at caller-declared indices, in sorted index/channel order,
    using a local seeded RNG. Other samples are unchanged. This is a simulation,
    not a measured artifact model or a source-defined correction algorithm.
    """
    stream = _require_od(signal)
    selected = _indices(indices, stream.n_samples)
    _seed(seed)
    finite(amplitude, "motion amplitude")
    if amplitude < 0:
        raise ValueError("motion amplitude must be nonnegative OD")
    rng = random.Random(seed)
    rows = list(stream.values)
    for i in selected:
        rows[i] = tuple(value + rng.uniform(-amplitude, amplitude) for value in rows[i])
    return _od_result(
        stream,
        rows,
        "simulate_motion",
        seed=seed,
        evidence_class=EvidenceClass.SIMULATION,
        formula="y[i,c] = x[i,c] + Uniform(-amplitude, amplitude) at declared indices",
        indices=selected,
        amplitude=amplitude,
        rng="stdlib.random.Random",
    )


def motion_correct(
    signal: OpticalDensitySignal, artifact_indices: Sequence[int]
) -> OpticalDensitySignal:
    """Interpolate declared interior artifact runs; never detect motion implicitly.

    Between clean endpoints l,r, y[i,c] = x[l,c] + (x[r,c]-x[l,c])
    * (t[i]-t[l])/(t[r]-t[l]). Endpoints cannot be artifacts; no extrapolation
    or random imputation is allowed. This deterministic reference helper is
    not an assertion that linear interpolation is appropriate for real NIRS.
    """
    stream = _require_od(signal)
    selected = _indices(artifact_indices, stream.n_samples)
    if selected and (selected[0] == 0 or selected[-1] == stream.n_samples - 1):
        raise ValueError("motion correction needs clean bracketing endpoints")
    rows = list(stream.values)
    position = 0
    while position < len(selected):
        first = selected[position]
        last = first
        position += 1
        while position < len(selected) and selected[position] == last + 1:
            last = selected[position]
            position += 1
        left, right = first - 1, last + 1
        for i in range(first, last + 1):
            fraction = (stream.timestamps[i] - stream.timestamps[left]) / (
                stream.timestamps[right] - stream.timestamps[left]
            )
            rows[i] = tuple(
                a + fraction * (b - a) for a, b in zip(stream.values[left], stream.values[right])
            )
    return _od_result(
        stream,
        rows,
        "motion_correct",
        formula="y[i,c] = x[l,c] + (x[r,c]-x[l,c])*(t[i]-t[l])/(t[r]-t[l])",
        artifact_indices=selected,
        detection="caller_declared",
        endpoints="clean_brackets_required",
        stochastic=False,
    )


def filter_optical_density(signal: OpticalDensitySignal, window: int) -> OpticalDensitySignal:
    """Declared causal moving-average FIR on uniform samples.

    y[i,c] = sum(x[j,c], j=max(0,i-window+1)..i) / min(window,i+1).
    Startup uses partial windows; no zero padding, randomization, or implicit
    frequency cutoff is applied. This is not a published NIRS bandpass chain.
    """
    stream = _require_od(signal)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("filter window must be a positive integer sample count")
    uniform_rate(stream)
    columns = [stream.channel(j) for j in range(stream.n_channels)]
    rows = [
        [statistics.fmean(col[max(0, i - window + 1) : i + 1]) for col in columns]
        for i in range(stream.n_samples)
    ]
    return _od_result(
        stream,
        rows,
        "filter_optical_density",
        formula="y[i,c] = sum(x[max(0,i-window+1):i+1,c]) / min(window,i+1)",
        window=window,
        startup="partial",
        phase="causal",
        stochastic=False,
    )


@dataclass(frozen=True)
class RawProxyResult:
    """Sample-mean OD contrast on supplied data, with an incomplete source gate."""

    intensity: IntensitySignal
    optical_density: OpticalDensitySignal
    baseline_mean_od: float
    response_mean_od: float
    delta_od: float
    baseline_count: int
    response_count: int
    evidence_class: EvidenceClass
    provenance: Provenance
    quantity: NIRSQuantity = field(default=NIRSQuantity.OPTICAL_DENSITY, init=False)
    result_kind: str = field(default="raw_optical_density_proxy", init=False)
    preprocessing_gate: str = field(default="OPEN_INCOMPLETE", init=False)
    published_reproduction: bool = field(default=False, init=False)
    clinically_validated: bool = field(default=False, init=False)
    limitations: tuple[str, ...] = field(
        default=(
            "Exact source-defined preprocessing is unavailable; gate remains open/incomplete.",
            "No motion correction, filtering, ABP regression, or hemoglobin reconstruction applied.",
            "Exploratory optical-density proxy only; no published reproduction or clinical validation.",
        ),
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))
        if self.evidence_class not in {EvidenceClass.EMPIRICAL_ESTIMATE, EvidenceClass.SIMULATION}:
            raise ValueError("raw-proxy evidence must be EMPIRICAL_ESTIMATE or SIMULATION")
        if not isinstance(self.intensity, IntensitySignal) or not isinstance(
            self.optical_density, OpticalDensitySignal
        ):
            raise TypeError("raw-proxy result requires intensity and OD semantic types")
        for name in ("baseline_mean_od", "response_mean_od", "delta_od"):
            finite(getattr(self, name), name)

    @property
    def direction(self) -> str:
        return "negative" if self.delta_od < 0 else "positive" if self.delta_od > 0 else "zero"


def raw_proxy_experiment(
    intensity: IntensitySignal,
    *,
    event_time_s: float,
    baseline_window_s: tuple[float, float] = (-20.0, 0.0),
    response_window_s: tuple[float, float] = (20.0, 50.0),
    channel: int = 0,
    evidence_class: EvidenceClass = EvidenceClass.EMPIRICAL_ESTIMATE,
    seed: int = 0,
) -> RawProxyResult:
    """Compute mean(OD_response)-mean(OD_baseline), never a hemoglobin endpoint.

    Windows are event-relative seconds, left-closed/right-open; means weight
    each acquired sample equally. The caller labels simulated versus empirical
    input. The seed is recorded for experiment provenance; no random draws occur.
    """
    finite(event_time_s, "event time")
    _seed(seed)
    for window in (baseline_window_s, response_window_s):
        if len(window) != 2:
            raise ValueError("proxy windows require two endpoints")
        for endpoint in window:
            finite(endpoint, "window endpoint")
        if window[0] >= window[1]:
            raise ValueError("proxy windows must have increasing endpoints")
    if baseline_window_s[1] > 0 or response_window_s[0] < 0:
        raise ValueError("baseline must precede the event and response must follow it")
    od = optical_density(intensity)
    if (
        isinstance(channel, bool)
        or not isinstance(channel, int)
        or not 0 <= channel < od.stream.n_channels
    ):
        raise ValueError("proxy channel must be a valid nonnegative integer index")
    values = od.stream.channel(channel)
    selected = [
        tuple(
            value
            for t, value in zip(od.stream.timestamps, values)
            if low <= t - event_time_s < high
        )
        for low, high in (baseline_window_s, response_window_s)
    ]
    if any(not window_values for window_values in selected):
        raise ValueError("both proxy windows require acquired samples; no imputation")
    baseline, response = map(statistics.fmean, selected)
    return RawProxyResult(
        intensity,
        od,
        baseline,
        response,
        response - baseline,
        len(selected[0]),
        len(selected[1]),
        evidence_class,
        Provenance(
            "physioworld.instrumentation.nirs",
            "raw_proxy_experiment",
            seed=seed,
            parent_ids=(intensity.stream.metadata.record_id or "anonymous",),
            notes={
                "event_time_s": event_time_s,
                "baseline_window_s": tuple(baseline_window_s),
                "response_window_s": tuple(response_window_s),
                "window_boundaries": "left_closed_right_open",
                "channel": channel,
                "formula": "delta_od = sample_mean(OD_response) - sample_mean(OD_baseline)",
                "seed_use": "provenance_only_deterministic_algorithm",
                "evidence_class": EvidenceClass(evidence_class).value,
                "preprocessing_gate": "OPEN_INCOMPLETE",
                "output_quantity": NIRSQuantity.OPTICAL_DENSITY.value,
            },
        ),
    )


# Short semantic names are aliases, not conversions or promotions.
Intensity = IntensitySignal
OpticalDensity = OpticalDensitySignal
Hemoglobin = HemoglobinSignal

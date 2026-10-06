"""Scientifically labelled synthetic signals; no clinical waveform validation.

ECG is a PQRST morphology surrogate driven by engine heart/breath rates, EEG a
phenomenological spectral surrogate driven by an assumed perfusion/arousal
proxy, telemetry is sampled directly from the coupled physiological model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Iterator

from physioworld.core import Provenance

from .information import DataRequest, Observation
from .models import VARIABLE_UNITS
from .scenario import PhysiologicalScenario
from .stream import ChannelMetadata, SignalStream, StreamMetadata
from .time import ClockDomain, TimeMetadata


@dataclass(frozen=True)
class SyntheticPhysiology:
    seed: int = 0

    @staticmethod
    def _pqrst(phase: float, rr_s: float, atrial_amplitude: float = 1.0) -> float:
        # Gaussian morphology with fixed temporal widths, not ionic electrophysiology.
        result = 0.0
        for center_s, width_s, height in (
            (-0.18, 0.035, 0.12 * atrial_amplitude),
            (-0.025, 0.009, -0.15),
            (0.0, 0.012, 1.0),
            (0.025, 0.012, -0.25),
            (0.24, 0.065, 0.30),
        ):
            distance = ((phase * rr_s - center_s + rr_s / 2) % rr_s) - rr_s / 2
            result += height * math.exp(-0.5 * (distance / width_s) ** 2)
        return result

    def waveform(
        self,
        modality: str,
        *,
        duration_s: float,
        sample_rate: float,
        heart_rate_bpm: float = 60.0,
        amplitude: float = 1.0,
    ) -> SignalStream:
        if (
            not all(math.isfinite(v) for v in (duration_s, sample_rate, heart_rate_bpm, amplitude))
            or duration_s < 0
            or sample_rate <= 0
            or heart_rate_bpm <= 0
        ):
            raise ValueError("duration, sample rate, and heart rate must be positive")
        if modality.upper() == "EEG":
            return self.eeg(duration_s=duration_s, sample_rate=sample_rate)
        count = int(round(duration_s * sample_rate))
        frequency = heart_rate_bpm / 60.0
        rng = random.Random(self.seed)
        values = []
        for index in range(count):
            time = index / sample_rate
            phase = (time * frequency) % 1.0
            if modality.upper() == "ECG":
                value = amplitude * self._pqrst(phase, 1 / frequency)
                unit = "mV"
            elif modality.upper() == "PPG":
                value = amplitude * (0.5 + 0.5 * math.sin(2 * math.pi * phase))
                unit = "a.u."
            elif modality.upper() in {"RESP", "RESPIRATION"}:
                value = amplitude * math.sin(2 * math.pi * frequency / 4.0 * time)
                unit = "a.u."
            else:
                value = amplitude * math.sin(2 * math.pi * frequency * time)
                unit = "a.u."
            values.append(value + rng.gauss(0.0, amplitude * 0.001))
        return SignalStream.from_samples(
            values,
            sample_rate,
            unit,
            modality,
            clock=TimeMetadata(clock_domain=ClockDomain.SIMULATION),
            provenance=Provenance(
                "synthetic",
                "waveform",
                self.seed,
                notes={
                    "model": "PQRST_gaussian_surrogate"
                    if modality.upper() == "ECG"
                    else "periodic_demo",
                    "evidence_class": "SIMULATION",
                    "clinical_validation": False,
                },
            ),
        )

    def _samples(
        self,
        duration_s: float,
        sample_rate: float,
        scenario: PhysiologicalScenario | None,
        variables: tuple[str, ...],
    ) -> Iterator[tuple[float, Observation]]:
        if not math.isfinite(duration_s) or duration_s < 0 or duration_s > 21600:
            raise ValueError("duration outside 0..21600 s")
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("sample_rate must be finite positive")
        source = scenario or PhysiologicalScenario(duration_s=duration_s, seed=self.seed)
        if duration_s > source.duration_s:
            raise ValueError("waveform exceeds scenario duration")
        engine = replace(
            source, requests=DataRequest(variables), observation_noise={}
        ).create_engine()
        for i in range(int(round(duration_s * sample_rate))):
            time = i / sample_rate
            yield time, engine.advance_to(time)

    def ecg(
        self,
        *,
        duration_s: float,
        sample_rate: float = 250,
        scenario: PhysiologicalScenario | None = None,
    ) -> SignalStream:
        if sample_rate < 100:
            raise ValueError("PQRST surrogate requires at least 100 Hz")
        rng = random.Random(f"ecg:{self.seed}")
        phase, previous, values = 0.0, 0.0, []
        variables = ("heart_rate_bpm", "respiratory_rate_bpm", "oxygen_saturation")
        for time, o in self._samples(duration_s, sample_rate, scenario, variables):
            respiratory_rate = o.scalar("respiratory_rate_bpm")
            heart_rate = o.scalar("heart_rate_bpm")
            respiratory_phase = 2 * math.pi * respiratory_rate / 60 * time
            hr = heart_rate * (1 + 0.015 * math.sin(respiratory_phase))
            phase = (phase + (time - previous) * hr / 60) % 1
            previous = time
            values.append(
                self._pqrst(phase, 60 / hr)
                + 0.025 * math.sin(respiratory_phase)
                + rng.gauss(0, 0.005)
            )
        return SignalStream.from_samples(
            values,
            sample_rate,
            "mV",
            "ECG",
            subject_id=scenario.patient.subject_id if scenario else "synthetic-adult",
            clock=TimeMetadata(clock_domain=ClockDomain.SIMULATION),
            provenance=Provenance(
                "synthetic",
                "physiology_ecg",
                self.seed,
                notes={
                    "model": "PQRST_gaussian_surrogate",
                    "coupling": "engine HR + assumed respiratory sinus modulation",
                    "evidence_class": "SIMULATION",
                    "clinical_validation": False,
                    "limitations": "no conduction/ionic dynamics; rate changes do not identify arrhythmia type",
                },
            ),
        )

    def eeg(
        self,
        *,
        duration_s: float,
        sample_rate: float = 128,
        arousal: float = 0.6,
        scenario: PhysiologicalScenario | None = None,
    ) -> SignalStream:
        if sample_rate < 64 or not math.isfinite(arousal) or not 0 <= arousal <= 1:
            raise ValueError("EEG surrogate requires >=64 Hz and arousal in [0,1]")
        rng = random.Random(f"eeg:{self.seed}")
        phases = [rng.uniform(0, 2 * math.pi) for _ in range(3)]
        values, broadband = [], 0.0
        variables = ("mean_pressure_mmhg", "oxygen_saturation")
        for time, o in self._samples(duration_s, sample_rate, scenario, variables):
            pressure = o.scalar("mean_pressure_mmhg")
            oxygen_saturation = o.scalar("oxygen_saturation")
            perfusion = min(1.0, max(0.0, (pressure - 25) / 45))
            deficit = 1 - perfusion * oxygen_saturation
            active = arousal * (1 - deficit)
            broadband = math.exp(-1 / (0.03 * sample_rate)) * broadband + rng.gauss(0, 1.0)
            value = (
                18 * (1 - active) * (1 - deficit) * math.sin(2 * math.pi * 10 * time + phases[0])
                + (5 + 30 * deficit) * math.sin(2 * math.pi * 2 * time + phases[1])
                + 8 * active * math.sin(2 * math.pi * 20 * time + phases[2])
                + broadband
            )
            values.append(value)
        return SignalStream.from_samples(
            values,
            sample_rate,
            "uV",
            "EEG",
            subject_id=scenario.patient.subject_id if scenario else "synthetic-adult",
            clock=TimeMetadata(clock_domain=ClockDomain.SIMULATION),
            provenance=Provenance(
                "synthetic",
                "physiology_eeg",
                self.seed,
                notes={
                    "model": "phenomenological_alpha_delta_beta_surrogate",
                    "evidence_class": "SIMULATION",
                    "coupling": "assumed MAP/oxygenation perfusion proxy + declared arousal",
                    "clinical_validation": False,
                    "limitations": "not a cortical neural-mass/seizure model",
                },
            ),
        )

    def telemetry(
        self,
        *,
        duration_s: float,
        sample_rate: float = 1,
        scenario: PhysiologicalScenario | None = None,
        variables: tuple[str, ...] = (
            "heart_rate_bpm",
            "mean_pressure_mmhg",
            "systolic_pressure_mmhg",
            "diastolic_pressure_mmhg",
            "oxygen_saturation",
            "respiratory_rate_bpm",
            "arterial_pco2_mmhg",
            "cardiac_output_l_min",
            "lactate_mmol_l",
        ),
    ) -> SignalStream:
        times, rows, missing = [], [], []
        for time, observation in self._samples(duration_s, sample_rate, scenario, variables):
            times.append(time)
            row: list[float] = []
            row_missing = False
            for name in variables:
                value = observation[name]
                if value is None:
                    row.append(float("nan"))
                    row_missing = True
                else:
                    row.append(value)
            rows.append(tuple(row))
            missing.append(row_missing)
        metadata = StreamMetadata(
            subject_id=scenario.patient.subject_id if scenario else "synthetic-adult",
            modality="ICU_TELEMETRY",
            clock=TimeMetadata(clock_domain=ClockDomain.SIMULATION),
            channels=tuple(ChannelMetadata(n, VARIABLE_UNITS[n]) for n in variables),
            missing=tuple(missing),
            provenance=Provenance(
                "synthetic",
                "physiology_telemetry",
                self.seed,
                notes={
                    "model": "coupled_cardiorespiratory",
                    "evidence_class": "SIMULATION",
                    "clinical_validation": False,
                    "sampling": "direct engine observations; no interpolation",
                },
            ),
        )
        return SignalStream(tuple(times), tuple(rows), sample_rate, metadata)

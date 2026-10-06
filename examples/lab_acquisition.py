"""Run deterministic virtual lab acquisition, online math, views and replay.

Run from an installed checkout: python examples/lab_acquisition.py
Expected: 1000 samples, a 5 Hz spectral peak, exact replay, and safe outputs.
"""

import math
import tempfile
from pathlib import Path

from physioworld.devices import (
    AcquisitionConfiguration,
    AcquisitionRunner,
    ClockSynchronization,
    DeviceConfiguration,
    QualificationPlan,
    SimulatedDevice,
    qualify_device,
)
from physioworld.instrumentation import CalculationGraph, ChartView, Recording, VideoTrack


def main() -> None:
    device = SimulatedDevice(waveform=lambda t, channel: 2 * math.sin(2 * math.pi * 5 * t))
    graph = CalculationGraph()
    online_means = []

    def online(name, block):
        online_means.append(sum(block.channel(0)) / block.n_samples)

    with tempfile.TemporaryDirectory(prefix="physioworld-lab-") as directory:
        recording = Recording("lab-001", "virtual-subject", "session-001", group="simulator")
        recording.autosave_path = Path(directory) / "recording.json"
        recording.video.append(
            VideoTrack("metadata-only://virtual-camera", (0, 0.5, 1), 0, uncertainty_s=0.001)
        )
        runner = AcquisitionRunner(
            {"daq": device},
            {"daq": DeviceConfiguration(250, ("signal",))},
            {"daq": ClockSynchronization(0, method="shared_simulator_clock")},
        )
        result = runner.run(AcquisitionConfiguration(4, block_samples=125), recording, online)
        recording.mark(1, "virtual stimulus onset", source="simulation")
        graph.source("raw", recording.channels["daq"])
        graph.add("power", "psd", "raw", window="rectangular")
        power = graph.evaluate("power")
        peak = power.frequencies[max(range(len(power.values)), key=lambda i: power.values[i])]
        restored = Recording.load(recording.autosave_path)
        replayed = tuple(row for block in restored.replay("daq", 63) for row in block.values)
        assert replayed == recording.channels["daq"].values
        assert peak == 5 and len(replayed) == 1000
        view = ChartView(recording)
        view.zoom(0, 1)
        recording.export_csv(Path(directory) / "samples.csv")
        qualification = qualify_device(
            SimulatedDevice,
            QualificationPlan(DeviceConfiguration(250, ("signal",)), deterministic_replay=True),
        )
        assert qualification.passed
        print(
            {
                "samples": len(replayed),
                "peak_hz": peak,
                "online_blocks": len(online_means),
                "visible_points": len(view.data("daq")),
                "loss": result.dropped_samples,
                "outputs_after_stop": device.outputs,
                "qualification_scope": qualification.evidence_scope,
            }
        )


if __name__ == "__main__":
    main()

"""Independent analytic/reference and integration checks for owned instrument APIs.

Reference IDs and scientific status live with these static tests, not run logs.
Core tests use unittest/stdlib; optional NumPy/SciPy/FMPy tests are independent
comparisons, and are explicitly skipped when those packages are absent.
"""

import builtins
import importlib.util
import json
import math
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from physioworld.core import Provenance
from physioworld.coupling import (
    ArchiveLimits,
    Connection,
    CoupledSystem,
    FirstOrderModel,
    NativeCoSimulation,
    PythonModelExchange,
    ScheduledAction,
    ScheduledClock,
    UnsafeArchiveError,
    UnsupportedFMUError,
    export_fmu,
    import_fmu,
    model_description_xml,
    parse_model_description,
    safe_extract_fmu,
)
from physioworld.devices import (
    AcquisitionConfiguration,
    AcquisitionRunner,
    ActuatorDescriptor,
    Calibration,
    ClockSynchronization,
    DeviceConfiguration,
    DeviceState,
    JsonLineDevice,
    NetworkTransport,
    QualificationPlan,
    SensorDescriptor,
    SerialTransport,
    SimulatedDevice,
    StimulationBounds,
    Stimulus,
    qualify_calibration,
    qualify_device,
)
from physioworld.instrumentation import (
    CalculationGraph,
    ChartView,
    EventSeries,
    LinkedCursor,
    Recording,
    ScopeView,
    Session,
    SignalStream,
    VideoTrack,
    arithmetic,
    autocorrelation,
    blood_pressure,
    convert,
    derivative,
    detect_events,
    ecg_peaks,
    eeg,
    emg,
    event_average,
    filter_signal,
    hrv,
    integral,
    quality,
    resample,
    respiration,
    rr_intervals,
    smooth,
    spectrogram,
    spectrum,
    template_match,
    time_shift,
    unit,
)


def signal(values, fs=100, declared_unit="mV", timestamps=None, missing=()):
    return SignalStream.from_samples(
        values, fs, declared_unit, "reference", timestamps=timestamps, missing=missing
    )


def events(times, indices=()):
    return EventSeries(
        tuple(times), tuple(indices), "reference", Provenance("analytic", "reference")
    )


class CalculationReferenceTests(unittest.TestCase):
    def test_dag_types_forward_dependencies_cycles_and_provenance(self):
        graph = CalculationGraph()
        graph.source("raw", signal([1, 2, 3], 1))
        graph.add("area", "integral", "smooth")  # declared forward dependency
        graph.add("smooth", "smooth", "raw", window=1)
        result = graph.evaluate("area")
        self.assertEqual(result.channel(0), (0, 1.5, 4))
        self.assertEqual(result.metadata.provenance.parent_ids, ("smooth",))
        self.assertEqual(
            result.metadata.provenance.notes["algorithm"]["algorithm"], "cumulative_trapezoid"
        )
        self.assertEqual(len(result.metadata.provenance.notes["parent_hashes"][0]), 64)
        graph.add("bad", "integral", "area_events")
        graph.add("area_events", "events", "raw", threshold=2)
        with self.assertRaises(TypeError):
            graph.validate()
        cyclic = CalculationGraph()
        cyclic.add("a", "integral", "b")
        cyclic.add("b", "derivative", "a")
        with self.assertRaisesRegex(ValueError, "cycle"):
            cyclic.validate()

    def test_unit_rules_require_explicit_conversion_and_alignment(self):
        millivolts, volts = signal([1, 2]), signal([0.001, 0.002], declared_unit="V")
        with self.assertRaisesRegex(ValueError, "convert"):
            arithmetic(millivolts, volts, "add")
        summed = arithmetic(millivolts, convert(volts, "mV"), "add")
        self.assertEqual(summed.channel(0), (2, 4))
        product = arithmetic(millivolts, millivolts, "multiply")
        self.assertEqual(unit(product.metadata.channels[0].unit), unit("mV^2"))
        self.assertEqual(unit("mmHg*mL").factor_to(unit("J")), 0.000133322387415)
        with self.assertRaises(ValueError):
            arithmetic(millivolts, time_shift(millivolts, 1), "add")
        with self.assertRaises(ValueError):
            convert(millivolts, "mmHg")

    def test_analytic_integral_and_nonuniform_quadratic_derivative(self):
        # SIG-INT: integral of 3*t+2 is 1.5*t**2+2*t; tolerance 1e-12.
        times = (0, 0.1, 0.4, 0.8, 1.3, 2)
        line = signal([3 * t + 2 for t in times], None, timestamps=times)
        area = integral(line)
        for t, value in zip(times, area.channel(0)):
            self.assertAlmostEqual(value, 1.5 * t * t + 2 * t, places=12)
        quadratic = signal([t * t for t in times], None, timestamps=times)
        for t, value in zip(times[1:-1], derivative(quadratic).channel(0)[1:-1]):
            self.assertAlmostEqual(value, 2 * t, places=12)
        self.assertEqual(unit(area.metadata.channels[0].unit), unit("mV*s"))

    def test_missing_duplicate_irregular_samples_are_not_imputed(self):
        missing = signal([1, math.nan, 2], missing=(False, True, False))
        with self.assertRaises(ValueError):
            integral(missing)
        self.assertEqual(quality(missing).values["finite_fraction"], 2 / 3)
        flagged = replace(
            signal([1, 2, 3]), metadata=replace(missing.metadata, missing=(False, True, False))
        )
        with self.assertRaises(ValueError):
            resample(flagged, 200)
        duplicate = signal([1, 2, 3], None, timestamps=(0, 0, 0.1))
        with self.assertRaises(ValueError):
            derivative(duplicate)
        irregular = signal([1, 2, 3], 100, timestamps=(0, 0.01, 0.025))
        with self.assertRaisesRegex(ValueError, "uniform"):
            spectrum(irregular)
        interpolated = resample(irregular, 200)
        self.assertLessEqual(interpolated.timestamps[-1], irregular.timestamps[-1])
        self.assertIsNone(interpolated.metadata.provenance.notes["anti_alias_filter"])

    def test_analytic_sine_fft_psd_dc_nyquist_and_odd_length(self):
        # SIG-SPEC: 2*sin(2*pi*8*t) -> amplitude 2, mean power 2.
        fs, n = 128, 128
        stream = signal([2 * math.sin(2 * math.pi * 8 * i / fs) for i in range(n)], fs)
        amplitude = spectrum(stream, "fft", window="rectangular")
        self.assertAlmostEqual(abs(amplitude.values[8]), 2, places=12)
        power = spectrum(stream, "psd", window="rectangular")
        self.assertAlmostEqual(sum(power.values) * fs / n, 2, places=12)
        self.assertEqual(power.frequencies[-1], 64)
        nyquist = spectrum(
            signal([3 * (-1) ** i for i in range(n)], fs), "psd", window="rectangular"
        )
        self.assertAlmostEqual(nyquist.values[-1], 9, places=12)
        dc = spectrum(signal([3] * n, fs), "fft", window="rectangular")
        self.assertAlmostEqual(dc.values[0].real, 3, places=12)
        odd = spectrum(
            signal([math.sin(2 * math.pi * 3 * i / 15) for i in range(15)], 15),
            "psd",
            window="rectangular",
        )
        self.assertEqual(len(odd.values), 8)
        self.assertAlmostEqual(sum(odd.values), 0.5, places=12)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("scipy"),
        "independent spectral comparison requires numpy/scipy",
    )
    def test_spectral_and_filter_independent_numpy_scipy_references(self):
        import numpy as np
        from scipy.signal import butter, lfilter, periodogram

        fs, n = 200, 150
        x = [math.sin(0.13 * i) + 0.3 * math.cos(0.71 * i) + 0.002 * i for i in range(n)]
        stream = signal(x, fs)
        ours = spectrum(stream, "psd", window="hann", detrend=True)
        frequencies, reference = periodogram(
            x, fs=fs, window="hann", detrend="constant", scaling="density"
        )
        np.testing.assert_allclose(ours.frequencies, frequencies, atol=1e-12)
        np.testing.assert_allclose(ours.values, reference, atol=1e-12)
        b, a = butter(1, 12, fs=fs)
        np.testing.assert_allclose(
            filter_signal(stream, 12).channel(0), lfilter(b, a, x), atol=1e-12
        )
        expected = np.fft.rfft(x) / n
        expected[1:-1] *= 2
        np.testing.assert_allclose(
            spectrum(stream, "fft", window="rectangular").values, expected, atol=1e-12
        )

    def test_spectrogram_center_hop_and_no_padding(self):
        stream = signal(
            [math.sin(2 * math.pi * i / 8) for i in range(100)],
            64,
            timestamps=[10 + i / 64 for i in range(100)],
        )
        result = spectrogram(stream, 32, 16)
        self.assertEqual(len(result.spectra), 5)
        self.assertEqual(result.times[0], 10 + 31 / 128)
        self.assertEqual(result.times[1] - result.times[0], 0.25)
        self.assertEqual(result.spectra[0].frequencies[4], 8)

    def test_channel_missingness_and_declared_uncertainty_policy(self):
        source = SignalStream.from_samples(
            [[1, 10], [2, 20], [3, 30]],
            10,
            "mV",
            "reference",
            missing_channels=((False, False), (False, True), (False, False)),
            uncertainty=((0.1, 0.2), (0.1, 0.2), (0.1, 0.2)),
        )
        self.assertEqual(quality(source, 1).values["flagged_missing_fraction"], 1 / 3)
        with self.assertRaises(ValueError):
            smooth(source, 2)
        converted = convert(source, "V")
        self.assertEqual(converted.metadata.uncertainty[0], (0.0001, 0.0002))
        complete = replace(source, metadata=replace(source.metadata, missing_channels=()))
        computed = integral(complete)
        self.assertEqual(computed.metadata.uncertainty, ())
        self.assertEqual(
            computed.metadata.provenance.notes["uncertainty_policy"],
            "not_propagated_by_this_algorithm",
        )

    def test_events_averaging_template_autocorrelation_and_smoothing(self):
        stream = signal([0, 1, 0, 0, 1, 0, 0, 1, 0], 10)
        peaks = detect_events(stream, 0.5, mode="peaks")
        self.assertEqual(peaks.indices, (1, 4, 7))
        self.assertEqual(event_average(stream, peaks, 1, 1).channel(0), (0, 1, 0))
        match = template_match(stream, [0, 1, 0])
        self.assertAlmostEqual(match.channel(0)[0], 1)
        self.assertAlmostEqual(autocorrelation(stream, 0).channel(0)[0], 1)
        self.assertEqual(smooth(signal([2, 4, 6], 1), 2).channel(0), (2, 3, 5))
        with self.assertRaises(ValueError):
            event_average(stream, events((0,)), 1, 1)


class PhysiologyReferenceTests(unittest.TestCase):
    def test_ecg_rr_hrv_explicit_candidate_peaks(self):
        x = [0] * 401
        for index in (50, 150, 270, 370):
            x[index] = 2
        peaks = ecg_peaks(signal(x, 100), 1)
        self.assertEqual(peaks.indices, (50, 150, 270, 370))
        intervals = rr_intervals(peaks)
        self.assertEqual(intervals.sample_rate, None)
        metrics = hrv(intervals)
        self.assertAlmostEqual(metrics.values["RMSSD"], 0.2)
        self.assertNotIn("SDNN", metrics.values)
        self.assertIn("SDNN", hrv(intervals, normal_intervals=True).values)
        self.assertTrue(metrics.limitations)

    def test_bp_time_weighted_map_and_respiration_rate(self):
        times = [i / 100 for i in range(201)]
        bp = signal([100 + 20 * math.cos(2 * math.pi * t) for t in times], 100, "mmHg")
        metrics = blood_pressure(bp, events((0, 1, 2)))
        self.assertAlmostEqual(metrics.values["MAP"], 100)
        self.assertEqual(metrics.values["systolic_mean"], 120)
        self.assertEqual(metrics.values["diastolic_mean"], 80)
        resp = signal([math.sin(2 * math.pi * 0.25 * i / 50) for i in range(1201)], 50, "L")
        self.assertAlmostEqual(respiration(resp, threshold=0.1).values["rate"], 15)

    def test_emg_eeg_analytic_power_and_nyquist_limits(self):
        stream = signal([3 * math.sin(2 * math.pi * 10 * i / 128) for i in range(128)], 128, "uV")
        self.assertAlmostEqual(emg(stream).values["RMS"], 3 / math.sqrt(2), places=12)
        self.assertEqual(emg(stream).values["median_frequency"], 10)
        self.assertAlmostEqual(eeg(stream).values["alpha"], 4.5, places=12)
        with self.assertRaises(ValueError):
            eeg(signal([1, 2, 3], 20, "uV"))


class RecordingWorkflowTests(unittest.TestCase):
    def test_autosave_replay_metadata_marks_video_and_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Recording("record", "subject", "session", "group")
            recording.autosave_path = Path(directory) / "autosave.json"
            source = signal([1, 2, 3, 4], 2)
            source = replace(
                source,
                metadata=replace(
                    source.metadata,
                    provenance=Provenance(
                        "lab",
                        "source",
                        seed=4,
                        notes={"reference_indices": (1, 2), "$tuple": ["literal researcher key"]},
                    ),
                    uncertainty=((0.1,),) * 4,
                    missing_channels=((False,),) * 4,
                ),
            )
            recording.append("A", source)
            recording.video.append(VideoTrack("camera.mp4", (0, 0.5, 1), 0.1, uncertainty_s=0.02))
            recording.mark(0.5, "event", tag="baseline")
            loaded = Recording.load(recording.autosave_path)
            self.assertEqual(loaded.channels["A"], recording.channels["A"])
            self.assertEqual(loaded.video[0].recording_times, (0.1, 0.6, 1.1))
            rows = tuple(row for block in loaded.replay("A", 3) for row in block.values)
            self.assertEqual(rows, source.values)
            session = Session("session", "subject", "group")
            session.add(loaded)
            self.assertEqual(session.batch("A", lambda s: s.n_samples), {"record": 4})
            recording.export_csv(Path(directory) / "samples.csv")
            self.assertIn("mV", (Path(directory) / "samples.csv").read_text())
            imported = Recording.import_csv(
                Path(directory) / "samples.csv",
                subject_id="subject",
                session_id="session",
                sample_rates={"A": 2},
            )
            self.assertEqual(imported.channels["A"].values, source.values)
            self.assertEqual(imported.channels["A"].metadata.missing_channels, ((False,),) * 4)
            self.assertEqual(imported.channels["A"].metadata.provenance.operation, "import_csv")

    def test_chart_scope_xy_histogram_linked_cursors_and_statistics(self):
        recording = Recording("r", "s", "session")
        recording.append("x", signal([0, 1, 2, 3], 1))
        recording.append("y", signal([0, 2, 4, 6], 1))
        recording.mark(1, "peak")
        cursor = LinkedCursor(1.1)
        chart, scope = ChartView(recording, cursor), ScopeView(recording, cursor)
        self.assertEqual(chart.meter("x"), (1,))
        cursor.time_s = 2
        self.assertEqual(scope.meter("y"), (4,))
        chart.zoom(1, 2)
        self.assertEqual(chart.xy("x", "y"), ((1, 2), (2, 4)))
        self.assertEqual(chart.histogram("x", (0, 1, 2, 3)), (0, 1, 1))
        self.assertEqual(chart.extract("x").values["mean"], 1.5)
        self.assertEqual(chart.events()[0].label, "peak")
        self.assertEqual(scope.overlays("x", events((1, 2)), 1, 1)[0][0][0], -1)

    def test_replay_missing_values_and_overlap_rejection(self):
        recording = Recording("r", "s", "session")
        source = signal([1, math.nan, 3], 1, missing=(False, True, False))
        recording.append("x", source)
        with self.assertRaises(ValueError):
            recording.append("x", source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            recording.save(path)
            self.assertTrue(math.isnan(Recording.load(path).channels["x"].channel(0)[1]))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "MATLAB interchange requires scipy")
    def test_matlab_export_is_readable(self):
        from scipy.io import loadmat

        recording = Recording("r", "s", "session")
        recording.append("ECG lead I", signal([1, 2, 3], 100))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.mat"
            recording.export_mat(path)
            self.assertIn("channels", loadmat(path))

    def test_long_recording_keeps_flat_replayable_provenance(self):
        recording = Recording("long", "subject", "session")
        for i in range(1100):
            source = signal([i], 10, timestamps=(i / 10,))
            source = replace(
                source,
                metadata=replace(
                    source.metadata,
                    provenance=Provenance("simulator", "block", notes={"sequence_numbers": (i,)}),
                ),
            )
            recording.append("x", source)
        self.assertEqual(len(recording.channels["x"].metadata.lineage), 1100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.json"
            recording.save(path)
            loaded = Recording.load(path)
            self.assertEqual(loaded.channels["x"], recording.channels["x"])


class DeviceContractTests(unittest.TestCase):
    def test_simulator_real_samples_calibration_and_restart(self):
        device = SimulatedDevice(
            waveform=lambda t, channel: t, sensors=(SensorDescriptor("x", "count", 0, 10),)
        )
        calibration = Calibration.linear(((0, 1), (5, 11), (10, 21)), "count", "mmHg", "reference")
        device.calibrate("x", calibration)
        config = DeviceConfiguration(10, ("x",))
        with self.assertRaises(RuntimeError):
            device.start()
        device.configure(config)
        device.start()
        first = device.read(4)
        self.assertEqual(first.sequence_numbers, (0, 1, 2, 3))
        self.assertEqual(first.rows, ((1,), (1.2,), (1.4,), (1.6,)))
        self.assertEqual(first.sensors[0].unit, "mmHg")
        device.stop()
        device.reset()
        device.configure(config)
        device.start()
        self.assertEqual(first, device.read(4))
        self.assertTrue(qualify_calibration(calibration, ((2, 5), (7, 15)), 1e-12).passed)
        nonlinear = Calibration.polynomial((1, 0, 2), "count", "mV", (-2, 2))
        self.assertEqual(nonlinear.apply(-2), 9)
        self.assertEqual(nonlinear.output_bounds, (1, 9))
        with self.assertRaises(ValueError):
            nonlinear.apply(3)

    def test_synchronized_runner_heterogeneous_rates_and_online_callback(self):
        a = SimulatedDevice("a", waveform=lambda t, channel: 2 * t)
        b = SimulatedDevice("b", waveform=lambda t, channel: t + 1)
        callback_sizes = []
        runner = AcquisitionRunner(
            {"a": a, "b": b},
            {"a": DeviceConfiguration(100, ("signal",)), "b": DeviceConfiguration(25, ("signal",))},
            {"a": ClockSynchronization(0), "b": ClockSynchronization(2)},
        )
        result = runner.run(
            AcquisitionConfiguration(1, 13, session_start_s=2),
            on_block=lambda name, stream: callback_sizes.append(stream.n_samples),
        )
        self.assertEqual(result.recording.channels["a"].n_samples, 100)
        self.assertEqual(result.recording.channels["b"].n_samples, 25)
        self.assertEqual(result.recording.channels["a"].timestamps[0], 2)
        self.assertEqual(result.recording.channels["b"].timestamps[0], 2)
        self.assertEqual(a.outputs["analog"], 0)
        self.assertEqual(a.state, DeviceState.STOPPED)
        self.assertEqual(sum(callback_sizes), 125)

    def test_runner_failure_emergency_stop_all_devices(self):
        a, b = SimulatedDevice("a"), SimulatedDevice("b", fail_after=2)
        runner = AcquisitionRunner(
            {"a": a, "b": b},
            {k: DeviceConfiguration(10, ("signal",)) for k in ("a", "b")},
            {k: ClockSynchronization(0) for k in ("a", "b")},
        )
        with self.assertRaises(IOError):
            runner.run(AcquisitionConfiguration(1, 4))
        for device in (a, b):
            self.assertEqual(device.state, DeviceState.FAILED)
            self.assertEqual(device.outputs["analog"], 0)
            with self.assertRaises(RuntimeError):
                device.start()

    def test_generated_stimulation_drives_actual_virtual_acquisition_and_io(self):
        device = SimulatedDevice()
        device.waveform = lambda t, channel: device.output_at("analog", t)
        device.configure(DeviceConfiguration(1000, ("signal",)))
        device.start()
        device.digital_write("digital", True)
        self.assertTrue(device.digital_read("digital"))
        stimulus = Stimulus("analog", 2, "V", 0.1, 10, 0.005)
        device.stimulate(stimulus, StimulationBounds("V", 3, 1, 20, 0.1))
        block = device.read(10)
        self.assertEqual([r[0] for r in block.rows], [2] * 5 + [0] * 5)
        self.assertAlmostEqual(device.trigger(0.01).latency_s, 0.001)
        device.analog_write("analog", 1)
        with self.assertRaises(ValueError):
            device.analog_write("analog", 10)
        self.assertEqual(device.outputs, {"analog": 0, "digital": 0})
        self.assertEqual(device.state, DeviceState.FAILED)

    def test_stimulation_charge_duty_bounds_and_safe_latch(self):
        device = SimulatedDevice(actuators=(ActuatorDescriptor("stim", "mA", -2, 2),))
        device.configure(DeviceConfiguration(100, ("signal",)))
        device.start()
        bounds = StimulationBounds("mA", 1, 2, 20, 0.1, 0.001)
        safe = Stimulus("stim", 0.5, "mA", 1, 10, 0.001)
        device.stimulate(safe, bounds)
        self.assertEqual(device.outputs["stim"], 0)
        with self.assertRaises(ValueError):
            device.stimulate(replace(safe, amplitude=1.5), bounds)
        self.assertEqual(device.state, DeviceState.FAILED)
        with self.assertRaisesRegex(ValueError, "charge"):
            StimulationBounds("mA", 2, 2, 20, 0.5, 1e-6).validate(safe)

    def test_qualification_loss_jitter_saturation_trigger_and_restart(self):
        plan = QualificationPlan(DeviceConfiguration(100, ("signal",)), deterministic_replay=True)
        self.assertTrue(qualify_device(SimulatedDevice, plan).passed)
        bad = qualify_device(
            lambda: SimulatedDevice(
                drop_every=4, jitter_s=0.001, waveform=lambda t, channel: 100, trigger_latency_s=0.2
            ),
            plan,
        )
        failed = {c.id for c in bad.checks if c.passed is False}
        self.assertTrue({"DEV-LOSS", "DEV-TIME", "DEV-SAT", "DEV-TRIGGER"} <= failed)
        self.assertIn("no physical validation", bad.evidence_scope)
        self.assertIsNone(next(c.passed for c in bad.checks if c.id == "DEV-POWER"))

    def test_actual_network_rpc_device_acquisition(self):
        simulator = SimulatedDevice()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        errors = []

        def server():
            try:
                with listener.accept()[0] as connection, connection.makefile("rwb") as stream:
                    for line in stream:
                        request = json.loads(line)
                        op, arguments = request["operation"], request["arguments"]
                        if op == "safe_startup":
                            result = None
                        elif op == "discover":
                            result = asdict(simulator.discover())
                        elif op == "configure":
                            simulator.configure(
                                DeviceConfiguration(
                                    arguments["sample_rate"], tuple(arguments["channels"])
                                )
                            )
                            result = None
                        elif op == "read":
                            result = asdict(simulator.read(arguments["sample_count"]))
                        else:
                            result = getattr(simulator, op)(**arguments)
                        stream.write(json.dumps({"ok": True, "result": result}).encode() + b"\n")
                        stream.flush()
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=server, daemon=True)
        worker.start()
        transport = NetworkTransport("127.0.0.1", listener.getsockname()[1])
        try:
            device = JsonLineDevice(transport)
            device.configure(DeviceConfiguration(100, ("signal",)))
            device.start()
            self.assertEqual(len(device.read(10).rows), 10)
            device.stop()
        finally:
            transport.close()
            worker.join(timeout=3)
            listener.close()
        self.assertFalse(errors)
        self.assertFalse(worker.is_alive())

    @unittest.skipUnless(importlib.util.find_spec("serial"), "serial loopback requires pyserial")
    def test_actual_optional_serial_loopback(self):
        serial = SerialTransport("loop://")
        try:
            self.assertEqual(serial.exchange(b'{"operation":"test"}'), b'{"operation":"test"}')
        finally:
            serial.close()


class FMIReferenceTests(unittest.TestCase):
    def test_fmi_302_capability_declarations_and_owned_metadata_subset(self):
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
<fmiModelDescription fmiVersion="3.0.2" modelName="metadata" instantiationToken="token">
  <ModelExchange modelIdentifier="model_exchange" needsCompletedIntegratorStep="true"/>
  <CoSimulation modelIdentifier="co_simulation" needsExecutionTool="true"
                canHandleVariableCommunicationStepSize="true" hasEventMode="true"/>
  <ScheduledExecution modelIdentifier="scheduled_execution" canSerializeFMUState="true"/>
  <UnitDefinitions>
    <Unit name="1"><BaseUnit/></Unit>
    <Unit name="V"><BaseUnit kg="1" m="2" s="-3" A="-1"/></Unit>
  </UnitDefinitions>
  <ModelVariables>
    <Float64 name="u" valueReference="1" causality="input" start="2"/>
    <Float64 name="y" valueReference="2" causality="output" unit="V" start="0"/>
    <Clock name="sample" valueReference="3" causality="input" variability="discrete"
           intervalVariability="constant" intervalDecimal="0.25" shiftDecimal="0.125"/>
  </ModelVariables>
  <ModelStructure><Output valueReference="2" dependencies=""/></ModelStructure>
</fmiModelDescription>"""
        description = parse_model_description(payload)
        self.assertEqual(description.fmi_version, "3.0.2")
        self.assertEqual(
            description.interfaces, ("ModelExchange", "CoSimulation", "ScheduledExecution")
        )
        capabilities = {
            cap.interface: dict(cap.boolean_flags) for cap in description.interface_capabilities
        }
        self.assertTrue(capabilities["ModelExchange"]["needsCompletedIntegratorStep"])
        self.assertTrue(capabilities["CoSimulation"]["needsExecutionTool"])
        self.assertTrue(capabilities["ScheduledExecution"]["canSerializeFMUState"])
        self.assertTrue(description.needs_execution_tool)
        self.assertEqual(description.variables[0].unit, "1")
        self.assertEqual(description.clocks[0].phase_s, 0.125)

    def test_float64_and_clock_edges_are_rejected_or_defaulted_explicitly(self):
        base = """<fmiModelDescription fmiVersion="3.0.2" modelName="edge" instantiationToken="token">
<CoSimulation modelIdentifier="edge" needsExecutionTool="true"/>
<ModelVariables>{variables}</ModelVariables><ModelStructure>{structure}</ModelStructure>
</fmiModelDescription>"""
        scalar = '<Float64 name="y" valueReference="1" causality="output" start="0"/>'
        valid = base.format(variables=scalar, structure='<Output valueReference="1"/>').encode()
        self.assertEqual(parse_model_description(valid).variables[0].unit, "1")
        cases = (
            (
                '<Float64 name="y" valueReference="1" causality="output" start="0"><Dimension start="2"/></Float64>',
                UnsupportedFMUError,
            ),
            (
                '<Float64 name="y" valueReference="1" causality="output" displayUnit="mV" start="0"/>',
                UnsupportedFMUError,
            ),
            (
                '<Int32 name="count" valueReference="1" causality="output" start="0"/>',
                UnsupportedFMUError,
            ),
            (
                '<Float64 name="derived" valueReference="1" causality="calculatedParameter" start="0"/>',
                UnsupportedFMUError,
            ),
            (
                '<Clock name="tick" valueReference="1" causality="input" variability="discrete" '
                'intervalVariability="fixed" intervalDecimal="1"/>',
                UnsupportedFMUError,
            ),
            (
                '<Clock name="tick" valueReference="1" causality="input" variability="discrete" '
                'intervalVariability="constant" supportsFraction="true" intervalDecimal="1"/>',
                UnsupportedFMUError,
            ),
            (
                '<Clock name="tick" valueReference="1" causality="input" variability="discrete" '
                'intervalVariability="constant"/>',
                ValueError,
            ),
        )
        for variable, error in cases:
            with self.subTest(variable=variable):
                structure = '<Output valueReference="1"/>' if "Float64" in variable else ""
                with self.assertRaises(error):
                    parse_model_description(
                        base.format(variables=variable, structure=structure).encode()
                    )
        inconsistent_unit = base.replace(
            "<ModelVariables>{variables}</ModelVariables>",
            '<UnitDefinitions><Unit name="V"><BaseUnit kg="1"/></Unit></UnitDefinitions>'
            "<ModelVariables>{variables}</ModelVariables>",
        ).format(
            variables='<Float64 name="y" valueReference="1" causality="output" unit="V"/>',
            structure='<Output valueReference="1"/>',
        )
        with self.assertRaises(UnsupportedFMUError):
            parse_model_description(inconsistent_unit.encode())
        with self.assertRaises(UnsupportedFMUError):
            parse_model_description(valid.replace(b'fmiVersion="3.0.2"', b'fmiVersion="3.1"'))

    def test_executable_archive_roundtrip_and_analytic_response(self):
        # FMI-ODE: exact first-order response versus independent exp formula, 1e-12.
        model = FirstOrderModel(output_unit="V", tau=2, gain=3)
        with tempfile.TemporaryDirectory() as directory:
            path = export_fmu(model, Path(directory) / "response.fmu")
            fmu = import_fmu(path)
            self.assertEqual(fmu.description.interfaces, ("CoSimulation",))
            self.assertTrue(fmu.description.needs_execution_tool)
            fmu.initialize()
            fmu.set_float64(1, 2, "V")
            for i in range(20):
                fmu.do_step(i / 10, 0.1)
            self.assertAlmostEqual(fmu.get_float64(2), 6 * (1 - math.exp(-1)), places=12)
            with self.assertRaises(ValueError):
                fmu.set_float64("u", 2, "mV")
            with self.assertRaises(ValueError):
                fmu.do_step(0, 0.1)
            fmu.terminate()
            with self.assertRaises(RuntimeError):
                fmu.get_float64("y")

    def test_hybrid_events_impulses_and_scheduled_heterogeneous_clocks(self):
        model = FirstOrderModel(
            tau=2,
            input_start=2,
            actions=(ScheduledAction(0.25, {"u": 4}, 1, "step_input"),),
            clocks=(
                ScheduledClock("beat", 0.5, 0.5, impulse=0.1),
                ScheduledClock("regulation", 10, 10),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            fmu = import_fmu(export_fmu(model, Path(directory) / "hybrid.fmu"))
            fmu.initialize()
            result = fmu.do_step(0, 1)
            y25 = 2 * (1 - math.exp(-0.25 / 2)) + 1
            y50 = 4 + (y25 - 4) * math.exp(-0.25 / 2) + 0.1
            expected = 4 + (y50 - 4) * math.exp(-0.5 / 2) + 0.1
            self.assertAlmostEqual(fmu.get_float64("y"), expected, places=12)
            self.assertEqual(tuple(e.time for e in result.events), (0.25, 0.5, 1))
            self.assertEqual([c.period_s for c in fmu.description.clocks], [0.5, 10])
        held = FirstOrderModel(input_start=1, actions=(ScheduledAction(0.5, {"mode": 0}),))
        held.initialize()
        held.do_step(0, 2)
        self.assertAlmostEqual(held.get("y"), 1 - math.exp(-0.5))

    def test_external_solver_against_analytic_solution_and_coupling_semantics(self):
        model = FirstOrderModel(input_start=2)
        model.initialize()
        solver = PythonModelExchange(model)
        for _ in range(100):
            solver.advance(0.01)
        self.assertAlmostEqual(solver.continuous_states()[0], 2 * (1 - math.exp(-1)), places=9)
        a, b = FirstOrderModel(initial=2, output_unit="mV"), FirstOrderModel(output_unit="V")
        coupling = CoupledSystem({"a": a, "b": b}, (Connection("a", "y", "b", "u", 0.001),))
        coupling.initialize()
        coupling.step(1)
        # Input is from START communication point, not future upstream output.
        self.assertAlmostEqual(b.get("y"), 0.002 * (1 - math.exp(-1)))
        with self.assertRaises(ValueError):
            CoupledSystem({"a": a, "b": b}, (Connection("a", "y", "b", "u"),))
        x, y = FirstOrderModel(), FirstOrderModel()
        with self.assertRaisesRegex(ValueError, "cycle"):
            CoupledSystem(
                {"x": x, "y": y}, (Connection("x", "y", "y", "u"), Connection("y", "y", "x", "u"))
            )
        system = CoupledSystem(
            {"x": x, "y": y},
            (Connection("x", "y", "y", "u"), Connection("y", "y", "x", "u", lagged=True)),
        )
        system.initialize()
        system.step(0.1)

    @unittest.skipUnless(
        importlib.util.find_spec("fmpy"), "independent FMI 3 schema validation requires fmpy"
    )
    def test_model_description_independent_fmi_schema_validation(self):
        from fmpy import read_model_description

        model = FirstOrderModel(output_unit="mmHg", clocks=(ScheduledClock("sample", 0.1),))
        with tempfile.TemporaryDirectory() as directory:
            path = export_fmu(model, Path(directory) / "schema.fmu")
            description = read_model_description(path, validate=True)
            self.assertEqual(description.fmiVersion, "3.0.2")
            self.assertEqual(description.modelVariables[2].unit, "mmHg")
            with self.assertRaisesRegex(UnsupportedFMUError, "tool-required"):
                NativeCoSimulation(path)

    def test_native_runtime_absence_is_explicit(self):
        original_import = builtins.__import__

        def absent(name, *args, **kwargs):
            if name == "fmpy" or name.startswith("fmpy."):
                raise ImportError("absent in reference environment")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=absent),
            self.assertRaisesRegex(ImportError, "fmpy"),
        ):
            NativeCoSimulation("unused.fmu")

    def test_metadata_resource_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original = export_fmu(FirstOrderModel(), Path(directory) / "original.fmu")
            damaged = Path(directory) / "damaged.fmu"
            with zipfile.ZipFile(original) as source, zipfile.ZipFile(damaged, "w") as target:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "resources/physioworld.json":
                        spec = json.loads(data)
                        spec["tau"] = 7
                        data = json.dumps(spec).encode()
                    target.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "does not match"):
                import_fmu(damaged)

    def test_malicious_paths_symlinks_duplicates_and_bombs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, name in enumerate(
                ("../escape", "/absolute", "C:/escape", "a\\escape", "a/../escape", "a//b")
            ):
                with self.subTest(name=name):
                    archive = root / f"bad-{index}.fmu"
                    with zipfile.ZipFile(archive, "w") as package:
                        package.writestr(name, "bad")
                    with self.assertRaises(UnsafeArchiveError):
                        safe_extract_fmu(archive, root / f"out-{index}")
            for kind in ("symlink", "duplicate", "collision", "bomb"):
                archive = root / f"{kind}.fmu"
                with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
                    if kind == "symlink":
                        info = zipfile.ZipInfo("resources/link")
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                        package.writestr(info, "../../outside")
                    elif kind == "duplicate":
                        package.writestr("A", "one")
                        package.writestr("a", "two")
                    elif kind == "collision":
                        package.writestr("a", "file")
                        package.writestr("a/b", "nested")
                    else:
                        package.writestr("huge", b"0" * 10000)
                with self.subTest(kind=kind), self.assertRaises(UnsafeArchiveError):
                    safe_extract_fmu(
                        archive, root / f"out-{kind}", ArchiveLimits(max_member_bytes=1000)
                    )
            self.assertFalse((root.parent / "escape").exists())

    def test_archive_portability_empty_destination_and_invalid_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("resources/CON", "resources/trailing.", "resources/trailing "):
                archive = root / "unsafe.fmu"
                with zipfile.ZipFile(archive, "w") as package:
                    package.writestr(name, "bad")
                with self.assertRaises(UnsafeArchiveError):
                    safe_extract_fmu(archive, root / "out")
                archive.unlink()
            invalid = root / "invalid.fmu"
            invalid.write_bytes(b"not a zip")
            with self.assertRaises(UnsafeArchiveError):
                safe_extract_fmu(invalid, root / "invalid-out")
            archive = root / "valid.fmu"
            export_fmu(FirstOrderModel(), archive)
            destination = root / "destination"
            destination.mkdir()
            (destination / "preexisting").write_text("do not overwrite")
            with self.assertRaises(UnsafeArchiveError):
                safe_extract_fmu(archive, destination)

    def test_supported_python_cosimulation_example_runs_end_to_end(self):
        result = subprocess.run(
            [sys.executable, "examples/fmi_roundtrip.py"],
            cwd=Path(__file__).parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("'fmi_version': '3.0.2'", result.stdout)
        self.assertIn("'needs_execution_tool': True", result.stdout)

    def test_xml_entities_forbidden_and_archive_export_deterministic(self):
        payload = b'<!DOCTYPE fmiModelDescription [<!ENTITY x "bad">]><fmiModelDescription fmiVersion="3.0" modelName="x" instantiationToken="x"/>'
        with self.assertRaises(UnsafeArchiveError):
            parse_model_description(payload)
        self.assertEqual(
            parse_model_description(model_description_xml(FirstOrderModel())).variables[0].unit, "s"
        )
        with tempfile.TemporaryDirectory() as directory:
            a = export_fmu(FirstOrderModel(), Path(directory) / "a.fmu")
            b = export_fmu(FirstOrderModel(), Path(directory) / "b.fmu")
            self.assertEqual(a.read_bytes(), b.read_bytes())


if __name__ == "__main__":
    unittest.main()

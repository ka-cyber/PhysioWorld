"""Executable tool-required FMI 3.0.2 Python FMU roundtrip, not a native C-ABI FMU.

Expected: y(1) = 2*(1-exp(-1)) V, and four quarter-second clock events.
"""

import math
import tempfile
from pathlib import Path

from physioworld.coupling import FirstOrderModel, ScheduledClock, export_fmu, import_fmu


def main() -> None:
    model = FirstOrderModel(
        "LabResponse",
        output_unit="V",
        tau=1,
        clocks=(ScheduledClock("quarter_second", 0.25, 0.25),),
    )
    with tempfile.TemporaryDirectory(prefix="physioworld-roundtrip-") as directory:
        archive = export_fmu(model, Path(directory) / "response.fmu")
        fmu = import_fmu(archive)
        fmu.initialize()
        fmu.set_float64("u", 2, "V")
        events = []
        for i in range(10):
            events.extend(fmu.do_step(i / 10, 0.1).events)
        expected = 2 * (1 - math.exp(-1))
        observed = fmu.get_float64("y")
        assert math.isclose(observed, expected, abs_tol=1e-12)
        assert len(events) == 4
        print(
            {
                "fmi_version": fmu.description.fmi_version,
                "y_V": observed,
                "analytic_y_V": expected,
                "clock_times_s": [e.time for e in events],
                "needs_execution_tool": fmu.description.needs_execution_tool,
            }
        )
        fmu.terminate()


if __name__ == "__main__":
    main()

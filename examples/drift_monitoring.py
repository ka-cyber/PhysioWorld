"""Deterministic event-time drift monitoring example.

Run from the repository root with ``python examples/drift_monitoring.py``.
The output is a compact JSON summary of synthetic software-verification data;
it is not a physiological or clinical result.
"""

from __future__ import annotations

import json

from physioworld.core import CausalStatus, EvidenceClass, MetricContext
from physioworld.monitoring import (
    DriftMonitorConfig,
    EventObservation,
    TaskValuePoint,
    WindowedDriftMonitor,
    scan_performance_drift,
)


def observations(prefix: str, start: float, values: tuple[float, ...], *, labels: bool = False):
    return tuple(
        EventObservation(
            f"{prefix}-{index}",
            f"subject-{index}",
            start + index,
            {"heart_rate": value},
            prediction=value,
            label=(value > 0.5 if labels else None),
            input_available_s=start + index,
            label_available_s=(start + index + 1 if labels else None),
            prediction_time_s=start + index,
            clock_basis="example-clock",
        )
        for index, value in enumerate(values)
    )


def main() -> None:
    config = DriftMonitorConfig(
        training_cutoff_s=5.0,
        clock_basis="example-clock",
        window_size_s=5.0,
        step_s=5.0,
        minimum_reference_n=2,
        minimum_current_n=2,
        performance_threshold=0.15,
        evaluation_as_of_s=30.0,
    )
    reference = observations("reference", 0.0, (0.0, 0.2, 0.4, 0.6, 0.8))
    current = observations("current", 10.0, (1.0, 1.2, 1.4, 1.6, 1.8))
    monitor = WindowedDriftMonitor(reference, config)
    covariate = monitor.scan_covariate(current, "heart_rate")
    context = MetricContext(
        estimand_name="forecast_value",
        data_domain="synthetic-example",
        evaluation_regime="prospective",
        conditioning=("event_time",),
        policy_class="frozen",
        reference_state="reference-v1",
        time_horizon=1.0,
        task_family=("forecast",),
        resource_state="none",
        uncertainty_treatment="none_declared",
        metric_type="task_value",
        utility="negative_error",
        population="synthetic",
        unit="utility",
        causal_status=CausalStatus.PREDICTIVE,
        evidence_class=EvidenceClass.NUMERICAL_VERIFICATION,
        estimator="mean",
    )
    reference_value = tuple(
        TaskValuePoint(float(index), 1.0, context, f"reference-value-{index}") for index in range(5)
    )
    current_value = tuple(
        TaskValuePoint(10.0 + index, 0.5, context, f"current-value-{index}") for index in range(5)
    )
    performance = scan_performance_drift(
        reference_value, current_value, context, config, as_of_s=30.0
    )
    print(
        json.dumps(
            {
                "covariate": {
                    "windows": len(covariate),
                    "drift_detected": any(result.drift_detected for result in covariate),
                    "method": covariate[0].method.value,
                },
                "performance": {
                    "windows": len(performance),
                    "drift_detected": any(result.drift_detected for result in performance),
                    "estimand": context.estimand_name,
                },
                "evidence": "synthetic numerical verification; no causal interpretation",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

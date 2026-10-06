"""Known synthetic confounding: guarded control/treatment values and sensitivity.

Run: python examples/causal_identification.py
All observations below are explicitly synthetic; no historical or clinical result
is reproduced by this example.
"""

from physioworld.causal import (
    MANDATORY_ASSUMPTIONS,
    AssumptionStatus,
    CausalAssumption,
    EvaluationPolicy,
    IdentificationRecord,
    LoggedStep,
    LoggedTrajectory,
    SupportRecord,
    TargetTrial,
    TimedCovariate,
    evaluate_policy,
    hidden_confounding_bounds,
)
from physioworld.uncertainty import cluster_bootstrap


def main() -> None:
    rows = []
    for x in (0, 1):
        propensity = 0.8 if x else 0.2
        for action in (0, 1):
            count = round(100 * 0.5 * (propensity if action else 1 - propensity))
            for _ in range(count):
                step = LoggedStep(
                    str(x),
                    (TimedCovariate("X", float(x), 0.0),),
                    str(action),
                    (("0", 1 - propensity), ("1", propensity)),
                    float(3 * action + 2 * x),
                    0.0,
                    1.0,
                    ("0", "1"),
                )
                rows.append(LoggedTrajectory(f"synthetic-{len(rows)}", 0.0, 0.0, (step,)))
    record = IdentificationRecord(
        "expected synthetic outcome under constant treatment",
        "binary action",
        TargetTrial(
            "synthetic baseline",
            "all synthetic subjects",
            ("0", "1"),
            ("control", "treat"),
            "Y=3A+2X",
            1.0,
            "no censoring",
        ),
        tuple(
            CausalAssumption(
                name,
                AssumptionStatus.ASSUMED,
                "explicit synthetic DGP assumption; not human evidence",
            )
            for name in MANDATORY_ASSUMPTIONS
        ),
        (
            SupportRecord("synthetic", "0", 0.5, 50, "generated finite population"),
            SupportRecord("synthetic", "1", 0.5, 50, "generated finite population"),
        ),
        "ipw",
        "synthetic finite population",
        "Y=3A+2X",
        "subject bootstrap",
        "Gamma-bounded hidden-confounding sensitivity",
        ("synthetic assumptions only",),
    )
    for action, name in ((0, "control"), (1, "treat")):
        policy = EvaluationPolicy(name, lambda history, a=action: {"0": 1.0 - a, "1": float(a)})
        estimate = evaluate_policy(rows, policy, record)
        uncertainty = cluster_bootstrap(
            estimate.subject_contributions,
            estimate.subject_ids,
            lambda values: sum(values) / len(values),
            repetitions=500,
        )
        sensitivity = hidden_confounding_bounds(rows, policy, record, gamma=2.0)
        print(f"{name}: value={estimate.value:.3f}, bootstrap={uncertainty.interval}")
        print(
            f"  status={estimate.claim_status}; Gamma=2 range=({sensitivity.lower:.3f}, {sensitivity.upper:.3f})"
        )
    print("Analytic synthetic truth: control=1, treat=4, treatment effect=3.")


if __name__ == "__main__":
    main()

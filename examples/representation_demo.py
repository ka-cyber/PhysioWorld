"""Run: python examples/representation_demo.py (after installing PhysioWorld).

Analytic/synthetic representation examples; no empirical or clinical claims.
The core methods use only Python's standard library.
"""

from __future__ import annotations

import json

from physioworld.representations import (
    EmpiricalTaskLawRepresentation,
    InformationBottleneckRepresentation,
    MinimalSufficientRepresentation,
    MultiscaleHistoryRepresentation,
    PCARepresentation,
    PredictiveStateRepresentation,
    RateDistortionRepresentation,
    SharedMultimodalRepresentation,
    UncertaintyAwareRepresentation,
    permutation_control,
)


def main() -> None:
    data = ((-3, -3), (-1, -1), (1, 1), (3, 3))
    pca = PCARepresentation(1, whiten=True).fit(data)
    encoded = pca.transform(data[-1])
    joint = ((0.225, 0.025), (0.225, 0.025), (0.025, 0.225), (0.025, 0.225))
    ib = tuple(
        InformationBottleneckRepresentation(2, beta=b, seed=7).fit_joint(joint)
        for b in (0.1, 2.0, 8.0)
    )
    psr = PredictiveStateRepresentation(
        {"observe": ((1, 0), (0, 1))},
        {"observe": ((0.9, 0.1), (0.1, 0.9))},
        (0.5, 0.5),
    )
    before = psr.initial_state()
    after = psr.update(before, "observe", 0)
    laws = EmpiricalTaskLawRepresentation("future_outcome").fit(("a", "a", "b", "b"), (0, 2, 1, 1))
    quotient = MinimalSufficientRepresentation(
        {
            "a": {"future_outcome": laws.predict_proba("a")},
            "b": {"future_outcome": laws.predict_proba("b")},
            "a_equivalent": {"future_outcome": laws.predict_proba("a")},
        }
    )
    rate_distortion = RateDistortionRepresentation(4).fit((0.5, 0.5), ((0, 1), (1, 0)))
    shared = SharedMultimodalRepresentation(("signal", "pressure"), 1).fit(
        tuple({"signal": (x,), "pressure": (2 * x,)} for x in (-3, -1, 1, 3))
    )
    uncertainty = UncertaintyAwareRepresentation(pca).transform((3, 3), ((1, 0.5), (0.5, 1)))
    multiscale = MultiscaleHistoryRepresentation((2, 4), statistics=("mean", "trend"))
    temporal_control = permutation_control((1, 2, 3, 4), control="temporal_shuffle", seed=3)
    summary = {
        "evidence": "synthetic_model_numerical_demonstration",
        "pca": {
            "components": pca.components_,
            "variance": pca.explained_variance_,
            "rank": pca.rank_,
            "whitened": encoded.values,
            "reconstructed": pca.inverse_transform(encoded),
        },
        "information_bottleneck": [model.metadata_ for model in ib],
        "predictive_state": {
            "before": before.values,
            "after_observation_0": after.values,
            "sufficiency": psr.predictive_sufficiency(),
        },
        "equal_means_distinct_task_laws": {
            "mean_a": laws.expectation("a"),
            "mean_b": laws.expectation("b"),
            "law_a": laws.predict_proba("a"),
            "law_b": laws.predict_proba("b"),
        },
        "quotient": quotient.certificate(),
        "rate_distortion": rate_distortion.metadata_,
        "multimodal": shared.transform({"signal": (2,), "pressure": (4,)}).values,
        "uncertainty": uncertainty.metadata,
        "no_future_leakage": multiscale.transform_at((1, 2, 3, 999), (0, 1, 2, 100), 2).metadata,
        "temporal_control": {
            "seed": temporal_control.seed,
            "permutation": temporal_control.permutation,
            "valid": multiscale.transform((1, 2, 3, 4)).values,
            "shuffled": multiscale.transform(temporal_control.values).values,
        },
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

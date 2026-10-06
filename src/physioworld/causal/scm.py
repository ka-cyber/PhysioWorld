"""Known-ground-truth synthetic structural causal models."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SCMObservation:
    confounder: float
    treatment: int
    outcome: float


@dataclass(frozen=True)
class BinaryTreatmentSCM:
    treatment_effect: float = 1.0
    confounding_effect: float = 0.5
    noise_scale: float = 0.1

    def generate(self, count: int, *, seed: int = 0) -> tuple[SCMObservation, ...]:
        if count < 1:
            raise ValueError("count must be positive")
        rng = random.Random(seed)
        observations = []
        for _ in range(count):
            confounder = rng.gauss(0.0, 1.0)
            probability = 1.0 / (1.0 + pow(2.718281828, -confounder))
            treatment = int(rng.random() < probability)
            outcome = (
                self.treatment_effect * treatment
                + self.confounding_effect * confounder
                + rng.gauss(0.0, self.noise_scale)
            )
            observations.append(SCMObservation(confounder, treatment, outcome))
        return tuple(observations)

    def counterfactual_mean(self, action: int, count: int = 10000, *, seed: int = 0) -> float:
        rng = random.Random(seed)
        return (
            sum(
                self.treatment_effect * action
                + self.confounding_effect * rng.gauss(0.0, 1.0)
                + rng.gauss(0.0, self.noise_scale)
                for _ in range(count)
            )
            / count
        )

    def true_average_treatment_effect(self) -> float:
        return self.treatment_effect


def known_ground_truth_interventional_gap(model: BinaryTreatmentSCM) -> float:
    return model.counterfactual_mean(1) - model.counterfactual_mean(0)

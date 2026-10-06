"""Independent reference checks for the network-physiology experiment."""

import cmath
import math
import unittest

from physioworld.experiments.network_physiology import (
    analyze_network_physiology,
    mutual_information_matrix,
    pearson_matrix,
    phase_randomized_surrogate,
    topology_summary,
)


def reference_pearson(columns):
    centered = []
    for column in columns:
        mean = sum(column) / len(column)
        values = tuple(value - mean for value in column)
        norm = math.sqrt(sum(value * value for value in values))
        centered.append(tuple(value / norm for value in values))
    return tuple(
        tuple(sum(a * b for a, b in zip(centered[i], centered[j])) for j in range(len(columns)))
        for i in range(len(columns))
    )


def reference_mutual_information(columns, bins):
    labels = []
    counts = []
    for column in columns:
        low, high = min(column), max(column)
        symbols = tuple(min(bins - 1, int(bins * (value - low) / (high - low))) for value in column)
        labels.append(symbols)
        counts.append(tuple(symbols.count(symbol) for symbol in range(bins)))
    result = []
    n_samples = len(columns[0])
    for i in range(len(columns)):
        row = []
        for j in range(len(columns)):
            joint = {
                (a, b): sum(x == a and y == b for x, y in zip(labels[i], labels[j]))
                for a in range(bins)
                for b in range(bins)
            }
            row.append(
                sum(
                    (frequency / n_samples)
                    * math.log(frequency * n_samples / (counts[i][a] * counts[j][b]))
                    for (a, b), frequency in joint.items()
                    if frequency
                )
            )
        result.append(tuple(row))
    return tuple(result)


def direct_dft(values):
    return tuple(
        sum(
            value * cmath.exp(-2j * math.pi * frequency * index / len(values))
            for index, value in enumerate(values)
        )
        for frequency in range(len(values))
    )


class NetworkPhysiologyTests(unittest.TestCase):
    @staticmethod
    def synthetic_data():
        first = tuple(float(index) for index in range(16))
        second = tuple(float((index * index + 3 * index) % 11) for index in range(16))
        third = tuple(float((index % 4) - (index // 4)) for index in range(16))
        return {"linear": first, "nonlinear": second, "structured": third}

    def test_pearson_and_histogram_mi_match_independent_references(self):
        samples = self.synthetic_data()
        columns = tuple(samples.values())

        expected_pearson = reference_pearson(columns)
        actual_pearson = pearson_matrix(samples)
        expected_mi = reference_mutual_information(columns, bins=4)
        actual_mi = mutual_information_matrix(samples, bins=4)

        for actual_row, expected_row in zip(actual_pearson, expected_pearson):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=14)
        for actual_row, expected_row in zip(actual_mi, expected_mi):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=14)
        self.assertAlmostEqual(actual_pearson[0][1], expected_pearson[0][1], places=14)
        self.assertGreater(actual_mi[0][0], 0.0)
        self.assertGreaterEqual(actual_mi[0][1], 0.0)

    def test_phase_surrogate_is_seeded_and_preserves_each_fourier_magnitude(self):
        samples = {
            "a": tuple(1.0 + math.sin(2 * math.pi * index / 8) + index / 10 for index in range(8)),
            "b": tuple(math.cos(2 * math.pi * index / 8) + (index % 3) for index in range(8)),
        }
        first = phase_randomized_surrogate(samples, seed=41)
        repeated = phase_randomized_surrogate(samples, seed=41)
        changed = phase_randomized_surrogate(samples, seed=42)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        for name, values in samples.items():
            expected_spectrum = direct_dft(values)
            surrogate_spectrum = direct_dft(first[name])
            for expected, actual in zip(expected_spectrum, surrogate_spectrum):
                self.assertAlmostEqual(abs(actual), abs(expected), places=11)
            self.assertAlmostEqual(sum(first[name]) / len(first[name]), sum(values) / len(values))

    def test_topology_summary_matches_known_triangle_with_a_tail(self):
        adjacency = (
            (0, 1, 1, 0),
            (1, 0, 1, 0),
            (1, 1, 0, 1),
            (0, 0, 1, 0),
        )
        summary = topology_summary(adjacency)
        self.assertEqual(summary.n_nodes, 4)
        self.assertEqual(summary.n_edges, 4)
        self.assertAlmostEqual(summary.density, 2 / 3)
        self.assertEqual(summary.degrees, (2, 2, 3, 1))
        self.assertEqual(summary.connected_components, ((0, 1, 2, 3),))
        self.assertEqual(summary.clustering_coefficients, (1.0, 1.0, 1 / 3, 0.0))
        self.assertAlmostEqual(summary.mean_clustering, 7 / 12)
        self.assertAlmostEqual(summary.transitivity, 3 / 5)
        self.assertAlmostEqual(summary.global_efficiency, 5 / 6)

    def test_full_result_is_deterministic_and_explicitly_noncausal(self):
        result = analyze_network_physiology(
            self.synthetic_data(),
            bins=4,
            n_surrogates=11,
            seed=9,
            alpha=0.05,
            pearson_threshold=0.2,
            mi_threshold=0.05,
        )
        repeated = analyze_network_physiology(
            self.synthetic_data(),
            bins=4,
            n_surrogates=11,
            seed=9,
            alpha=0.05,
            pearson_threshold=0.2,
            mi_threshold=0.05,
        )
        self.assertEqual(result, repeated)
        self.assertEqual(result.status, "ASSOCIATION_NOT_CAUSALITY")
        self.assertFalse(result.causal_claim)
        self.assertFalse(result.external_validation)
        self.assertEqual(result.input_scope, "SUPPLIED_SAMPLES_ONLY")
        self.assertEqual(result.surrogate_method, "INDEPENDENT_CHANNEL_PHASE_RANDOMIZATION")
        self.assertEqual(result.pearson.metric, "pearson")
        self.assertEqual(result.mutual_information.units, "nats")
        self.assertEqual(result.sample_count, 16)
        self.assertEqual(result.pearson.topology.n_nodes, 3)
        self.assertEqual(result.mutual_information.topology.n_nodes, 3)

    def test_invalid_constant_channel_is_rejected_for_pearson_network(self):
        samples = {"varying": (0.0, 1.0, 2.0), "constant": (1.0, 1.0, 1.0)}
        with self.assertRaises(ValueError):
            pearson_matrix(samples)
        with self.assertRaises(ValueError):
            analyze_network_physiology(samples, n_surrogates=2)


if __name__ == "__main__":
    unittest.main()

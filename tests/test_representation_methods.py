"""Scientific method verification using analytic models and negative controls."""

import math
import random
import unittest

from physioworld.representations import (
    EmpiricalTaskLawRepresentation,
    FiniteHistoryRepresentation,
    HandcraftedPhysiologyRepresentation,
    InformationBottleneckRepresentation,
    InstantaneousStateRepresentation,
    LearnedLatentRepresentation,
    MinimalSufficientRepresentation,
    MultiscaleHistoryRepresentation,
    PCARepresentation,
    PredictiveHistoryRepresentation,
    PredictiveStateRepresentation,
    QuotientRepresentation,
    RandomProjectionRepresentation,
    RateDistortionRepresentation,
    RawSignalRepresentation,
    SharedMultimodalRepresentation,
    StateAbstractionRepresentation,
    SupervisedBottleneckRepresentation,
    TaskSpecificRepresentation,
    UncertaintyAwareRepresentation,
    VariationalInformationBottleneck,
    minimal_sufficient_representation,
    permutation_control,
    randomized_representation_control,
)
from physioworld.representations._numeric import symmetric_eigen


class NumericAssertions(unittest.TestCase):
    def assertVectorClose(self, actual, expected, places=10):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=places)


class PCAMethodTests(NumericAssertions):
    @staticmethod
    def rotated_data():
        theta, phi = 0.63, 0.31
        c, s, cp, sp = math.cos(theta), math.sin(theta), math.cos(phi), math.sin(phi)
        basis = ((c * cp, s * cp, sp), (-s, c, 0.0), (-c * sp, -s * sp, cp))
        mean = (8.0, -2.0, 4.0)
        rows = tuple(
            tuple(mean[j] + sign * amplitude * basis[i][j] for j in range(3))
            for i, amplitude in enumerate((3.0, 2.0, 1.0))
            for sign in (-1, 1)
        )
        return rows, basis, mean

    def test_covariance_eigenvectors_on_correlated_rotated_data(self):
        rows, basis, mean = self.rotated_data()
        model = PCARepresentation(3).fit(rows)
        self.assertVectorClose(model.mean, mean)
        self.assertVectorClose(model.explained_variance_, (18 / 5, 8 / 5, 2 / 5))
        self.assertVectorClose(model.explained_variance_ratio_, (9 / 14, 4 / 14, 1 / 14))
        self.assertEqual(model.rank_, 3)
        for learned, expected in zip(model.components_, basis):
            self.assertAlmostEqual(abs(sum(a * b for a, b in zip(learned, expected))), 1.0)
        positive = model.transform(rows[1]).values
        negative = model.transform(rows[0]).values
        self.assertVectorClose(positive, tuple(-v for v in negative))
        self.assertAlmostEqual(abs(positive[0]), 3.0)

    def test_inverse_round_trip_and_orthogonal_projection(self):
        rows, basis, mean = self.rotated_data()
        full, reduced = PCARepresentation(3).fit(rows), PCARepresentation(1).fit(rows)
        for row in rows + ((10.0, -7.0, 9.0),):
            self.assertVectorClose(full.inverse_transform(full.transform(row)), row)
        residual = tuple(
            a - b for a, b in zip(rows[3], reduced.inverse_transform(reduced.transform(rows[3])))
        )
        self.assertAlmostEqual(sum(a * b for a, b in zip(residual, basis[0])), 0.0)
        self.assertAlmostEqual(sum(v * v for v in residual), 4.0)

    def test_whitening_has_identity_sample_covariance_and_inverse(self):
        rows, _, _ = self.rotated_data()
        model = PCARepresentation(3, whiten=True).fit(rows)
        z = tuple(model.transform(row).values for row in rows)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    sum(row[i] * row[j] for row in z) / (len(z) - 1), float(i == j)
                )
        self.assertVectorClose(model.inverse_transform(model.transform(rows[2])), rows[2])

    def test_rank_selection_and_singular_whitening(self):
        rows, _, _ = self.rotated_data()
        self.assertEqual(PCARepresentation(variance_threshold=0.9).fit(rows).n_components_, 2)
        with self.assertRaises(ValueError):
            PCARepresentation(1, variance_threshold=0.9).fit(rows)
        singular = ((-2.0, -2.0), (-1.0, -1.0), (1.0, 1.0), (2.0, 2.0))
        model = PCARepresentation(2).fit(singular)
        self.assertEqual(model.rank_, 1)
        self.assertAlmostEqual(model.explained_variance_[1], 0.0)
        with self.assertRaises(ValueError):
            PCARepresentation(2, whiten=True).fit(singular)
        white = PCARepresentation(1, whiten=True).fit(singular)
        self.assertVectorClose(white.inverse_transform(white.transform(singular[0])), singular[0])

    def test_constant_data_and_explicit_second_moment_option(self):
        model = PCARepresentation(2).fit(((3, 4), (3, 4)))
        self.assertEqual(model.rank_, 0)
        self.assertVectorClose(model.transform((3, 4)).values, (0, 0))
        self.assertVectorClose(model.inverse_transform((0, 0)), (3, 4))
        with self.assertRaises(ValueError):
            PCARepresentation(variance_threshold=0.8).fit(((3, 4), (3, 4)))
        uncentered = PCARepresentation(1, center=False).fit(((3, 4), (3, 4)))
        self.assertAlmostEqual(uncentered.explained_variance_[0], 25)
        self.assertVectorClose(uncentered.inverse_transform(uncentered.transform((3, 4))), (3, 4))

    def test_stability_on_large_offsets_and_wide_scales(self):
        baseline = 1e12
        model = PCARepresentation(1).fit(
            tuple((baseline + i, baseline + 2 * i) for i in (-2, -1, 1, 2))
        )
        self.assertAlmostEqual(model.explained_variance_[0], 50 / 3)
        for scale in (1e140, 1e-140):
            m = PCARepresentation(1, whiten=True).fit(((-scale, -scale), (scale, scale)))
            self.assertAlmostEqual(m.explained_variance_[0] / (scale * scale), 4.0)
            self.assertAlmostEqual(abs(m.transform((scale, scale)).values[0]), 1 / math.sqrt(2))
        for scale in (1e200, 1e-200):
            with self.assertRaises(ValueError):
                PCARepresentation(1).fit(((-scale,), (scale,)))

    def test_failures_do_not_silently_truncate_drop_or_impute(self):
        for rows in (
            (),
            ((1, 2),),
            ((), ()),
            ((1, 2), (1,)),
            ((1, math.nan), (2, 3)),
            ((1, math.inf), (2, 3)),
        ):
            with self.assertRaises(ValueError):
                PCARepresentation(1).fit(rows)
        for components in (0, -1, True, 1.2):
            with self.assertRaises(ValueError):
                PCARepresentation(components)
        with self.assertRaises(RuntimeError):
            PCARepresentation(1).transform((1, 2))
        with self.assertRaises(ValueError):
            PCARepresentation(3).fit(((1, 2), (2, 3)))
        model = PCARepresentation(1).fit(((1, 2), (2, 3)))
        for row in ((1,), (1, 2, 3), (math.nan, 1)):
            with self.assertRaises(ValueError):
                model.transform(row)
        with self.assertRaises(ValueError):
            model.inverse_transform((1, 2))

    def test_jacobi_eigen_residual_and_explicit_nonconvergence(self):
        a = ((4.0, 1.0, 2.0), (1.0, 3.0, 0.5), (2.0, 0.5, 2.0))
        eig, basis = symmetric_eigen(a)
        for value, row in zip(eig, basis):
            self.assertVectorClose(
                tuple(sum(x * y for x, y in zip(r, row)) for r in a), tuple(value * v for v in row)
            )
        with self.assertRaises(RuntimeError):
            symmetric_eigen(a, max_sweeps=1)
        with self.assertRaises(ValueError):
            symmetric_eigen(((1, 2), (0, 1)))

    def test_optional_numpy_differential_reference(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy differential reference unavailable")
        rng = random.Random(41)
        rows = tuple(tuple(rng.gauss(0, 1) for _ in range(5)) for _ in range(60))
        rows = tuple((r[0], 2 * r[0] + r[1], r[2] - r[0], r[3], r[4]) for r in rows)
        model = PCARepresentation(5).fit(rows)
        expected, _ = np.linalg.eigh(np.cov(np.array(rows), rowvar=False))
        self.assertVectorClose(model.eigenvalues_, tuple(reversed(expected)))


class InformationMethodTests(NumericAssertions):
    def test_blahut_arimoto_information_rate_relevance_tradeoff(self):
        joint = ((0.225, 0.025), (0.225, 0.025), (0.025, 0.225), (0.025, 0.225))
        low = InformationBottleneckRepresentation(2, beta=0.1, seed=7).fit_joint(joint)
        high = InformationBottleneckRepresentation(2, beta=8, seed=7).fit_joint(joint)
        self.assertLess(low.rate_, 1e-8)
        self.assertLess(low.relevance_, 1e-8)
        self.assertGreater(high.rate_, 0.69)
        expected_relevance = math.log(2) + 0.9 * math.log(0.9) + 0.1 * math.log(0.1)
        self.assertAlmostEqual(high.relevance_, expected_relevance, places=5)
        self.assertVectorClose(high.transform(0).values, high.transform(1).values)
        self.assertVectorClose(high.transform(2).values, high.transform(3).values)
        self.assertGreater(high.predict_proba(3)[1], 0.89)
        self.assertAlmostEqual(high.objective_, high.rate_ - high.beta * high.relevance_)
        for model in (low, high):
            self.assertTrue(model.converged_)
            self.assertTrue(
                all(
                    b <= a + 1e-10
                    for a, b in zip(model.objective_history_, model.objective_history_[1:])
                )
            )
            self.assertIn("blahut_arimoto", model.metadata_["optimization"])
            self.assertEqual(model.metadata_["stochasticity"], "categorical_encoder")

    def test_empirical_fit_is_seeded_and_semantic_labels_matter(self):
        x = ((0, 9),) * 200 + ((1, 9),) * 200
        y = (0,) * 200 + (1,) * 200
        valid = InformationBottleneckRepresentation(2, "binary", beta=5, seed=4).fit(x, y)
        repeat = InformationBottleneckRepresentation(2, "binary", beta=5, seed=4).fit(x, y)
        self.assertEqual(valid.encoder_, repeat.encoder_)
        self.assertEqual(valid.objective_history_, repeat.objective_history_)
        self.assertEqual(valid.sample((1, 9), seed=31), repeat.sample((1, 9), seed=31))
        control = permutation_control(y, control="label_permutation", seed=23)
        permuted = InformationBottleneckRepresentation(2, beta=5, seed=4).fit(x, control.values)
        self.assertGreater(valid.relevance_, 0.69)
        self.assertLess(permuted.relevance_, 0.02)
        self.assertLess(permuted.rate_, 0.02)
        self.assertEqual(permuted.metadata_["estimator"], "empirical_exact_discrete_joint")

    def test_independent_joint_zero_beta_and_single_codeword(self):
        for beta in (0, 20):
            model = InformationBottleneckRepresentation(2, beta=beta).fit_joint(
                ((0.25, 0.25), (0.25, 0.25))
            )
            self.assertLess(model.relevance_, 1e-10)
            self.assertLess(model.rate_, 1e-10)
        single = InformationBottleneckRepresentation(1).fit(("a", "b"), (0, 1))
        self.assertVectorClose(single.transform("a").values, (1,))
        self.assertVectorClose(single.predict_proba("b"), (0.5, 0.5))
        supervised = SupervisedBottleneckRepresentation(2, "binary", beta=3).fit((0, 1), (0, 1))
        self.assertGreater(supervised.relevance_, 0.69)

    def test_nonconvergence_reported_and_errors_explicit(self):
        model = InformationBottleneckRepresentation(2, max_iterations=1).fit((0, 1), (0, 1))
        self.assertFalse(model.converged_)
        self.assertEqual(model.metadata_["iterations"], 1)
        with self.assertRaises(RuntimeError):
            InformationBottleneckRepresentation(2).transform(0)
        for joint in (
            (),
            ((0.0, 0.0), (0.5, 0.5)),
            ((0.1, 0.2),),
            ((-0.2, 1.2),),
            ((math.nan, 1),),
        ):
            with self.assertRaises(ValueError):
                InformationBottleneckRepresentation(2).fit_joint(joint)
        with self.assertRaises(ValueError):
            model.transform("unknown")
        with self.assertRaises(ValueError):
            model.fit((1,), ())
        for beta in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                InformationBottleneckRepresentation(2, beta=beta)

    def test_rate_distortion_matches_binary_analytic_solution(self):
        beta = 4.0
        high = RateDistortionRepresentation(beta).fit((0.5, 0.5), ((0, 1), (1, 0)))
        low = RateDistortionRepresentation(0).fit((0.5, 0.5), ((0, 1), (1, 0)))
        error = 1 / (1 + math.exp(beta))
        entropy = -error * math.log(error) - (1 - error) * math.log(1 - error)
        self.assertAlmostEqual(high.distortion_, error)
        self.assertAlmostEqual(high.rate_, math.log(2) - entropy)
        self.assertAlmostEqual(low.rate_, 0)
        self.assertAlmostEqual(low.distortion_, 0.5)
        self.assertLess(high.distortion_, low.distortion_)
        self.assertTrue(high.converged_)
        self.assertEqual(high.sample(0, seed=7), high.sample(0, seed=7))
        with self.assertRaises(ValueError):
            RateDistortionRepresentation().fit((0.5, 0.5), ((0, -1), (1, 0)))

    def test_asymmetric_rate_distortion_has_monotone_objective(self):
        model = RateDistortionRepresentation(2).fit((0.8, 0.2), ((0, 1, 2), (1, 0, 2)))
        self.assertTrue(model.converged_)
        self.assertTrue(
            all(
                b <= a + 1e-10
                for a, b in zip(model.objective_history_, model.objective_history_[1:])
            )
        )
        self.assertAlmostEqual(sum(model.transform(0).values), 1.0)
        self.assertLess(model.latent_probability_[2], 1e-9)

    def test_stochastic_vib_optimizes_declared_objective_and_sampling(self):
        x, y = ((-3.0,), (-2.0,), (2.0,), (3.0,)), (0, 0, 1, 1)
        kwargs = dict(latent_dimension=1, beta=0.01, epochs=300, seed=3)
        model, repeat = (
            VariationalInformationBottleneck(1, **kwargs),
            VariationalInformationBottleneck(1, **kwargs),
        )
        model.fit(x, y)
        repeat.fit(x, y)
        self.assertEqual(model.training_loss, repeat.training_loss)
        self.assertLess(model.training_loss[-1], model.training_loss[0] * 0.4)
        self.assertGreater(model.predict_proba((3.0,))[1], 0.95)
        self.assertEqual(model.sample((1,), seed=5).values, model.sample((1,), seed=5).values)
        self.assertNotEqual(model.sample((1,), seed=5).values, model.transform((1,)).values)
        samples = tuple(model.sample((1,), seed=i).values[0] for i in range(1000))
        mean = sum(samples) / len(samples)
        self.assertAlmostEqual(mean, model.transform((1,)).values[0], delta=0.08)
        self.assertAlmostEqual(
            sum((v - mean) ** 2 for v in samples) / len(samples), 1.0, delta=0.12
        )
        analytic_kl = sum(0.5 * sum(m * m for m in model.transform(row).values) for row in x) / len(
            x
        )
        self.assertAlmostEqual(model.rate_upper_bound_, analytic_kl)
        self.assertEqual(model.metadata_["estimator"], "reparameterized_monte_carlo")
        self.assertFalse(model.metadata_["mutual_information_estimate"])
        # Refit must reset classifier bias as well as weights and seed.
        model.fit(x, y)
        self.assertEqual(model.training_loss, repeat.training_loss)

    def test_vib_noise_is_in_objective_not_only_added_at_inference(self):
        model = VariationalInformationBottleneck(1, 1, epochs=1).fit(((0,), (1,)), (0, 1))
        model._weights = [[0.0]]
        model._bias = [0.0]
        model._classifier = [2.0]
        model._classifier_bias = 0.0
        expected = (math.log1p(math.exp(2)) + math.log1p(math.exp(-2))) / 2
        stochastic_loss = model._objective(((0,),), (0,), (((-1.0,), (1.0,)),))
        self.assertAlmostEqual(stochastic_loss, expected)
        self.assertGreater(stochastic_loss, math.log(2) + 0.3)
        for x, y in ((((math.nan,),), (0,)), (((1,),), (2,))):
            with self.assertRaises(ValueError):
                model.fit(x, y)


class PredictiveStateMethodTests(NumericAssertions):
    def model(self):
        return PredictiveStateRepresentation(
            {"stay": ((1, 0), (0, 1)), "mix": ((0.75, 0.25), (0.2, 0.8))},
            {"stay": ((0.9, 0.1), (0.1, 0.9)), "mix": ((0.8, 0.2), (0.2, 0.8))},
            (0.5, 0.5),
            observations=("low", "high"),
        )

    def test_observable_tests_and_recursive_bayes_update(self):
        model = self.model()
        state = model.initial_state()
        self.assertVectorClose(model.predict_observation(state, "stay"), (0.5, 0.5))
        state = model.update(state, "stay", "low")
        self.assertVectorClose(model.belief_from_state(state), (0.9, 0.1))
        self.assertVectorClose(model.predict_observation(state, "stay"), (0.82, 0.18))
        state = model.update(state, "mix", "high")
        prior = (0.9 * 0.75 + 0.1 * 0.2, 0.9 * 0.25 + 0.1 * 0.8)
        likelihood = prior[0] * 0.2 + prior[1] * 0.8
        expected = (prior[0] * 0.2 / likelihood, prior[1] * 0.8 / likelihood)
        self.assertVectorClose(model.belief_from_state(state), expected)
        self.assertVectorClose(
            state.values, model.transform((("stay", "low"), ("mix", "high"))).values
        )

    def test_future_sequence_prediction_matches_enumeration(self):
        model = self.model()
        state = model.from_belief((0.7, 0.3))
        test = (("mix", "low"), ("stay", "high"))
        expected = sum(
            (0.7, 0.3)[i]
            * model.transitions["mix"][i][j]
            * model.emissions["mix"][j][0]
            * model.transitions["stay"][j][k]
            * model.emissions["stay"][k][1]
            for i in range(2)
            for j in range(2)
            for k in range(2)
        )
        self.assertAlmostEqual(model.predict_test(state, test), expected)
        self.assertAlmostEqual(
            sum(
                model.predict_test(state, (("mix", a), ("stay", b)))
                for a in model.observations
                for b in model.observations
            ),
            1.0,
        )

    def test_sufficiency_rank_and_belief_round_trips(self):
        model = self.model()
        certificate = model.predictive_sufficiency()
        self.assertTrue(certificate["sufficient_under_model"])
        self.assertEqual(certificate["observability_rank"], 2)
        self.assertLess(certificate["left_inverse_residual"], 1e-12)
        self.assertFalse(certificate["causal_identification"])
        for belief in ((1, 0), (0, 1), (0.31, 0.69)):
            self.assertVectorClose(model.belief_from_state(model.from_belief(belief)), belief)
        with self.assertRaises(ValueError):
            model.belief_from_state((0.5, 0.5, 0.9, 0.9))
        with self.assertRaises(ValueError):
            PredictiveStateRepresentation(
                {"a": ((1, 0), (0, 1))}, {"a": ((0.5, 0.5), (0.5, 0.5))}, (0.5, 0.5)
            )

    def test_multi_step_tests_identify_three_state_model(self):
        transitions = {"a": ((0.8, 0.2, 0), (0.1, 0.6, 0.3), (0.2, 0, 0.8))}
        emissions = {"a": ((0.9, 0.1), (0.5, 0.5), (0.1, 0.9))}
        with self.assertRaises(ValueError):
            PredictiveStateRepresentation(transitions, emissions, (0.2, 0.3, 0.5))
        model = PredictiveStateRepresentation(
            transitions, emissions, (0.2, 0.3, 0.5), tests=((("a", 0),), (("a", 0), ("a", 0)))
        )
        self.assertEqual(model.observability_rank_, 3)
        self.assertVectorClose(
            model.belief_from_state(model.initial_state()), (0.2, 0.3, 0.5), places=8
        )

    def test_long_history_beats_identical_one_sample_baseline_under_model(self):
        model = self.model()
        h1 = (("stay", "low"),) * 8 + (("stay", "high"),)
        h2 = (("stay", "high"),) * 9
        baseline = PredictiveHistoryRepresentation(1)
        self.assertEqual(
            baseline.transform((0,) * 8 + (1,)).values, baseline.transform((1,) * 9).values
        )
        prediction1 = model.predict_observation(model.transform(h1), "stay")[1]
        prediction2 = model.predict_observation(model.transform(h2), "stay")[1]
        self.assertLess(prediction1, 0.11)
        self.assertGreater(prediction2, 0.89)

    def test_empty_history_impossible_event_unknown_symbols_and_future_cutoff(self):
        model = self.model()
        self.assertEqual(model.transform(()).values, model.initial_state().values)
        with self.assertRaises(ValueError):
            model.update(model.initial_state(), "invalid", "low")
        impossible = PredictiveStateRepresentation({"a": ((1,),)}, {"a": ((1, 0),)}, (1,))
        with self.assertRaises(ValueError):
            impossible.update(impossible.initial_state(), "a", 1)
        history = (("stay", "low"), ("stay", "high"))
        at = model.transform_at(history, (1, 100), 2)
        self.assertEqual(at.values, model.transform(history[:1]).values)
        self.assertEqual(at.metadata["excluded_future_samples"], 1)


class OtherRepresentationMethodTests(NumericAssertions):
    def test_history_and_multiscale_are_nonanticipative(self):
        history, times = (1, 3, 5, 1000), (0, 1, 2, 10)
        model = FiniteHistoryRepresentation(4, pad_value=-1)
        self.assertEqual(model.transform_at(history, times, 2).values, (-1, 1, 3, 5))
        self.assertEqual(model.transform(()).metadata["padded_samples"], 4)
        multi = MultiscaleHistoryRepresentation((2, 3), statistics=("mean", "std", "trend", "last"))
        result = multi.transform_at(history, times, 2)
        self.assertVectorClose(result.values, (4, 1, 2, 5, 3, math.sqrt(8 / 3), 2, 5))
        self.assertEqual(
            InstantaneousStateRepresentation().transform_at(history, times, 2).values, (5,)
        )
        self.assertEqual(
            RawSignalRepresentation().transform_at(history, times, 2).values, (1, 3, 5)
        )
        for times in ((0, 1, 1, 2), (0, 2, 1, 3), (0, math.nan, 2, 3)):
            with self.assertRaises(ValueError):
                model.transform_at(history, times, 2)
        with self.assertRaises(ValueError):
            multi.transform(())
        with self.assertRaises(ValueError):
            multi.transform((1, math.nan))

    def test_generic_summaries_and_task_selection_are_explicit(self):
        summary = HandcraftedPhysiologyRepresentation().transform((-2, 2))
        self.assertVectorClose(summary.values, (0, 2, 2, -2, 2, 1))
        selected = TaskSpecificRepresentation("task", (2, 0))
        self.assertEqual(selected.transform((1, 2, 3)).values, (3, 1))
        self.assertEqual(selected.spec.training_dependence, "none")
        with self.assertRaises(ValueError):
            selected.transform((1,))
        with self.assertRaises(ValueError):
            TaskSpecificRepresentation("task", (-1,))
        with self.assertRaises(ValueError):
            HandcraftedPhysiologyRepresentation().transform(())
        with self.assertRaises(ValueError):
            RawSignalRepresentation().transform((math.inf,))

    def test_random_projection_is_seeded_linear_and_preserves_expected_norm(self):
        a, b = (
            RandomProjectionRepresentation(500, 3, seed=10),
            RandomProjectionRepresentation(500, 3, seed=10),
        )
        self.assertEqual(a.matrix, b.matrix)
        x, y = (1, 2, -3), (2, 0, 1)
        self.assertVectorClose(
            a.transform(tuple(p + q for p, q in zip(x, y))).values,
            tuple(p + q for p, q in zip(a.transform(x).values, a.transform(y).values)),
        )
        self.assertAlmostEqual(sum(v * v for v in a.transform(x).values), 14, delta=2)
        with self.assertRaises(ValueError):
            a.transform((1, 2))

    def test_learned_latent_is_fitted_and_not_alias(self):
        self.assertIsNot(LearnedLatentRepresentation, VariationalInformationBottleneck)
        model = LearnedLatentRepresentation(2, 1)
        rows = ((-2, -4), (-1, -2), (1, 2), (2, 4))
        model.fit(rows)
        self.assertLess(model.reconstruction_loss_, 1e-20)
        for row in rows:
            self.assertVectorClose(model.inverse_transform(model.transform(row)), row)
        self.assertEqual(model.transform(rows[0]).spec.name, "learned_linear_latent")

    def test_multimodal_joint_pca_uses_channel_data_and_round_trips(self):
        rows = tuple(
            {"ecg": (x, y), "pressure": (z,)}
            for x, y, z in ((-3, 1, 5), (-1, -2, 2), (2, 3, -1), (4, -1, -4), (0, 2, 1))
        )
        model = SharedMultimodalRepresentation(("ecg", "pressure"), 3).fit(rows)
        for row in rows:
            reconstructed = model.inverse_transform(model.transform(row))
            for modality in row:
                self.assertVectorClose(reconstructed[modality], row[modality])
        self.assertNotEqual(model.transform(rows[0]).values, model.transform(rows[1]).values)
        self.assertEqual(len(set(round(v, 8) for v in model.transform(rows[0]).values)), 3)
        with self.assertRaises(ValueError):
            model.transform({"wrong_channel": (1, 2), "pressure": (3,)})
        with self.assertRaises(ValueError):
            model.transform({"ecg": (1,), "pressure": (3,)})
        with self.assertRaises(RuntimeError):
            SharedMultimodalRepresentation(("ecg",), 1).transform({"ecg": (1,)})

    def test_uncertainty_full_covariance_and_seeded_gaussian_samples(self):
        model = PCARepresentation(2).fit(((-2, -2), (2, 2), (-1, 1), (1, -1)))
        wrapper = UncertaintyAwareRepresentation(model)
        result = wrapper.transform((1, 2), ((1, 0.5), (0.5, 4)))
        self.assertVectorClose(tuple(result.metadata["covariance"][i][i] for i in range(2)), (3, 2))
        self.assertAlmostEqual(abs(result.metadata["covariance"][0][1]), 1.5)
        self.assertVectorClose(result.values[2:], (math.sqrt(3), math.sqrt(2)))
        self.assertEqual(
            wrapper.sample((1, 2), ((1, 0.5), (0.5, 4)), seed=3),
            wrapper.sample((1, 2), ((1, 0.5), (0.5, 4)), seed=3),
        )
        with self.assertRaises(ValueError):
            wrapper.transform((1, 2))
        for cov in (((1, 2), (0, 1)), ((1, 2), (2, 1)), ((1,),)):
            with self.assertRaises(ValueError):
                wrapper.transform((1, 2), cov)
        zero = wrapper.transform((1, 2), ((0, 0), (0, 0)))
        self.assertVectorClose(zero.values[2:], (0, 0))

    def test_uncertainty_respects_pca_whitening(self):
        model = PCARepresentation(1, whiten=True).fit(((-1,), (1,)))
        result = UncertaintyAwareRepresentation(model, append_std=False).transform((0,), ((2,),))
        self.assertVectorClose(result.values, (0,))
        self.assertAlmostEqual(result.metadata["covariance"][0][0], 1.0)

    def test_minimal_sufficient_quotient_preserves_full_laws_not_only_means(self):
        laws = {
            "h1": {"outcome": (0.5, 0, 0.5)},
            "h2": {"outcome": (0.5, 0, 0.5)},
            "h3": {"outcome": (0, 1, 0)},
        }
        model = minimal_sufficient_representation(laws)
        self.assertIsInstance(model, MinimalSufficientRepresentation)
        self.assertEqual(model.dimension, 2)
        self.assertEqual(model.transform("h1").values, model.transform("h2").values)
        self.assertNotEqual(model.transform("h1").values, model.transform("h3").values)
        self.assertEqual(model.inverse_class(model.transform("h1")), ("h1", "h2"))
        self.assertEqual(model.task_law("h3", "outcome"), (0, 1, 0))
        self.assertEqual(model.certificate()["minimal_classes"], 2)
        with self.assertRaises(ValueError):
            model.transform("unseen")
        with self.assertRaises(ValueError):
            QuotientRepresentation({})

    def test_controlled_markov_abstraction_refines_reward_equivalence(self):
        transitions = {"a": ((0.5, 0.5, 0), (0.25, 0.75, 0), (0, 0, 1))}
        rewards = {"a": (0, 0, 1)}
        model = StateAbstractionRepresentation(transitions, rewards)
        self.assertEqual(model.dimension, 2)
        self.assertEqual(model.transform(0).values, model.transform(1).values)
        self.assertEqual(model.transitions_["a"], ((1, 0), (0, 1)))
        self.assertEqual(model.rewards_["a"], (0, 1))
        transitions["b"] = ((0, 0, 1), (1, 0, 0), (0, 0, 1))
        rewards["b"] = (0, 0, 1)
        refined = StateAbstractionRepresentation(transitions, rewards)
        self.assertEqual(refined.dimension, 3)
        self.assertNotEqual(refined.transform(0).values, refined.transform(1).values)

    def test_negative_control_provenance_and_temporal_semantic_effects(self):
        values = (1, 2, 4, 8, 16, 32)
        for name in ("temporal_shuffle", "semantic_permutation", "label_permutation"):
            control = permutation_control(values, control=name, seed=3)
            self.assertEqual(
                control.values, permutation_control(values, control=name, seed=3).values
            )
            self.assertEqual(sorted(control.values), sorted(values))
            self.assertEqual(control.values, tuple(values[i] for i in control.permutation))
        trend = MultiscaleHistoryRepresentation((6,), statistics=("trend",))
        shuffled = permutation_control(values, control="temporal_shuffle", seed=3)
        self.assertNotAlmostEqual(
            trend.transform(values).values[0], trend.transform(shuffled.values).values[0]
        )
        random_control = randomized_representation_control(((1, 2), (3, 4)), seed=4)
        self.assertEqual(
            random_control.values,
            randomized_representation_control(((1, 2), (3, 4)), seed=4).values,
        )
        self.assertEqual(tuple(map(len, random_control.values)), (2, 2))

    def test_removed_predictive_proxy_is_not_exported(self):
        import physioworld.representations as representations

        self.assertFalse(hasattr(representations, "PredictiveStateProxy"))
        self.assertEqual(PredictiveHistoryRepresentation(2).spec.name, "predictive_history")


class ConditionalTaskLawTests(NumericAssertions):
    def test_semantic_permutation_breaks_task_fidelity_despite_valid_probability_vectors(self):
        x = ("low",) * 100 + ("high",) * 100
        y = (0,) * 100 + (1,) * 100
        model = EmpiricalTaskLawRepresentation("binary").fit(x, y)
        control = permutation_control(x, control="semantic_permutation", seed=9)
        valid = model.evaluate(x, y)
        corrupted = model.evaluate(control.values, y)
        self.assertEqual(valid["brier"], 0)
        self.assertGreater(corrupted["brier"], 0.8)
        self.assertTrue(all(sum(model.predict_proba(key)) == 1 for key in control.values))

    def test_distribution_mean_and_quantile_are_distinct_estimands(self):
        model = EmpiricalTaskLawRepresentation("outcome").fit(("a", "a", "b", "b"), (0, 2, 1, 1))
        self.assertVectorClose(model.transform("a").values, (0.5, 0, 0.5))
        self.assertVectorClose(model.transform("b").values, (0, 1, 0))
        self.assertEqual(model.expectation("a"), model.expectation("b"))
        self.assertEqual(model.quantile("a", 0.75), 2)
        self.assertEqual(model.quantile("b", 0), 1)
        self.assertEqual(model.expectation("a", lambda v: float(v >= 2)), 0.5)
        self.assertEqual(
            model.transform("a").metadata["estimand"], "conditional_probability_mass_function"
        )
        with self.assertRaises(ValueError):
            model.transform("unknown")

    def test_knn_conditional_distribution_and_out_of_sample_calibration_scores(self):
        model = EmpiricalTaskLawRepresentation("outcome", neighbors=2).fit(
            ((0,), (1,), (10,), (11,)), (0, 2, 1, 1)
        )
        self.assertVectorClose(model.predict_proba((0.2,)), (0.5, 0, 0.5))
        self.assertVectorClose(model.predict_proba((10.2,)), (0, 1, 0))
        scored = model.evaluate(
            ((0.2,), (0.2,), (10.2,)), (0, 2, 1), conditioning_groups=("a", "a", "b")
        )
        self.assertAlmostEqual(scored["brier"], 1 / 3)
        self.assertAlmostEqual(scored["mean_classwise_calibration_error"], 0)
        self.assertEqual(scored["conditional_coverage"], {"a": 1.0, "b": 1.0})
        self.assertFalse(scored["calibration_guarantee"])

    def test_cluster_bootstrap_is_seeded_and_resamples_entire_subjects(self):
        model = EmpiricalTaskLawRepresentation("mean").fit(("a",) * 4, (0, 2, 4, 6))
        result = model.bootstrap_expectation("a", ("s1", "s1", "s2", "s2"), draws=100, seed=3)
        self.assertEqual(
            result, model.bootstrap_expectation("a", ("s1", "s1", "s2", "s2"), draws=100, seed=3)
        )
        self.assertEqual(set(result["replicates"]), {1.0, 3.0, 5.0})
        self.assertEqual(result["unsupported_replicates"], 0)
        self.assertEqual(result["percentile_interval_95"], (1, 5))
        sparse = EmpiricalTaskLawRepresentation("mean").fit(("a", "b"), (1, 2))
        report = sparse.bootstrap_expectation("a", ("s1", "s2"), draws=100, seed=3)
        self.assertGreater(report["unsupported_replicates"], 0)
        self.assertIsNone(report["percentile_interval_95"])

    def test_future_training_injection_and_empty_missing_input_rejected(self):
        model = EmpiricalTaskLawRepresentation("outcome")
        with self.assertRaises(ValueError):
            model.fit(("a", "b"), (0, 1), acquisition_times=(1, 100), training_cutoff=2)
        model.fit(("a", "b"), (0, 1), acquisition_times=(1, 2), training_cutoff=2)
        for samples, outcomes in (((), ()), (("a",), (math.nan,)), (("a",), ())):
            with self.assertRaises(ValueError):
                model.fit(samples, outcomes)
        with self.assertRaises(ValueError):
            model.quantile("a", 1.1)


if __name__ == "__main__":
    unittest.main()

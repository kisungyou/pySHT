"""Independent checks for joint multivariate mean/covariance tests."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import stats

from pysht.mean_covariance import hn_2samp, llzs_1samp, lrt_1samp


class TestOneSampleMeanCovariance(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [2.1, 1.2, 0.4],
                [1.4, 0.7, -0.3],
                [2.8, 1.9, 0.8],
                [0.9, 1.4, -0.2],
                [1.7, 0.3, 0.6],
                [2.5, 1.1, 0.2],
                [1.2, -0.4, 0.9],
                [1.8, 0.5, -0.6],
            ]
        )

    def test_lrt_matches_literal_whitened_formula(self) -> None:
        null_mean = np.array([0.3, -0.2, 0.4])
        null_covariance = np.array(
            [[1.7, 0.2, -0.1], [0.2, 1.2, 0.15], [-0.1, 0.15, 0.9]]
        )
        result = lrt_1samp(
            self.x,
            popmean=null_mean,
            popcov=null_covariance,
        )
        n, p = self.x.shape
        difference = np.mean(self.x, axis=0) - null_mean
        centered = self.x - np.mean(self.x, axis=0)
        mle_covariance = centered.T @ centered / n
        solved_covariance = np.linalg.solve(null_covariance, mle_covariance)
        sign_null, logdet_null = np.linalg.slogdet(null_covariance)
        sign_sample, logdet_sample = np.linalg.slogdet(mle_covariance)
        self.assertEqual(sign_null, 1.0)
        self.assertEqual(sign_sample, 1.0)
        divergence = (
            np.trace(solved_covariance)
            - (logdet_sample - logdet_null)
            - p
            + difference @ np.linalg.solve(null_covariance, difference)
        )
        expected = n * divergence
        df = p * (p + 3) / 2

        np.testing.assert_allclose(result.statistic, expected, rtol=2e-13)
        np.testing.assert_allclose(result.pvalue, stats.chi2.sf(expected, df))
        self.assertEqual(result.df, df)

    def test_llzs_matches_published_raw_second_moment_formula(self) -> None:
        result = llzs_1samp(self.x)
        n, p = self.x.shape
        mean = np.mean(self.x, axis=0)
        second_moment = self.x.T @ self.x / n
        raw = float(mean @ mean + np.sum((second_moment - np.eye(p)) ** 2))
        excess_kurtosis = float(np.mean(self.x**4) - 3.0)
        aspect_ratio = p / n
        center = aspect_ratio * (p + excess_kurtosis + 2.0)
        variance = 4 * aspect_ratio**2 * (aspect_ratio * (2 + excess_kurtosis) + 1)
        expected = (raw - center) / math.sqrt(variance)

        np.testing.assert_allclose(result.statistic, expected)
        np.testing.assert_allclose(result.pvalue, stats.norm.sf(expected))
        self.assertEqual(
            result.diagnostics,
            (
                ("aspect ratio p/n", aspect_ratio),
                ("estimated marginal excess kurtosis", excess_kurtosis),
            ),
        )
        self.assertEqual(result.estimates, ())
        # Regression guard: SHT 0.1.9 incorrectly centers this matrix at the
        # sample mean and therefore produces a different statistic.
        centered = self.x - mean
        legacy_second_moment = centered.T @ centered / n
        legacy_raw = mean @ mean + np.sum((legacy_second_moment - np.eye(p)) ** 2)
        legacy = (legacy_raw - center) / math.sqrt(variance)
        self.assertGreater(abs(result.statistic - legacy), 0.1)

    def test_llzs_high_dimensional_gram_route_matches_matrix_formula(self) -> None:
        sample = np.random.default_rng(718).normal(size=(7, 25))
        result = llzs_1samp(sample)
        n, p = sample.shape
        mean = np.mean(sample, axis=0)
        second_moment = sample.T @ sample / n
        raw = float(mean @ mean + np.sum((second_moment - np.eye(p)) ** 2))
        excess_kurtosis = float(np.mean(sample**4) - 3.0)
        aspect_ratio = p / n
        center = aspect_ratio * (p + excess_kurtosis + 2.0)
        variance = 4 * aspect_ratio**2 * (aspect_ratio * (2 + excess_kurtosis) + 1)
        expected = (raw - center) / math.sqrt(variance)

        np.testing.assert_allclose(result.statistic, expected, rtol=3e-14)
        np.testing.assert_allclose(result.pvalue, stats.norm.sf(expected), rtol=3e-14)

    def test_one_sample_tests_are_invariant_to_null_whitening(self) -> None:
        rng = np.random.default_rng(881)
        standardized = rng.normal(size=(40, 5))
        factor = np.array(
            [
                [1.4, 0.0, 0.0, 0.0, 0.0],
                [0.2, 1.1, 0.0, 0.0, 0.0],
                [-0.1, 0.1, 0.9, 0.0, 0.0],
                [0.3, -0.2, 0.1, 1.3, 0.0],
                [0.0, 0.2, -0.1, 0.15, 0.8],
            ]
        )
        mean = np.array([2.0, -1.0, 0.5, 1.5, -0.7])
        transformed = standardized @ factor.T + mean
        covariance = factor @ factor.T

        for method in (llzs_1samp, lrt_1samp):
            with self.subTest(method=method.__name__):
                baseline = method(standardized)
                actual = method(transformed, popmean=mean, popcov=covariance)
                np.testing.assert_allclose(
                    actual.statistic, baseline.statistic, rtol=2e-13
                )
                np.testing.assert_allclose(actual.pvalue, baseline.pvalue, rtol=2e-13)

    def test_null_whitening_preserves_ulps_around_a_huge_location(self) -> None:
        rng = np.random.default_rng(777)
        shift = 1.0e100
        unit = 2.0 * np.spacing(shift)
        for _ in range(11):
            standardized = rng.normal(size=(40, 3)) + rng.choice([0.0, 0.1, 0.2])
        values = shift + unit * standardized
        stable = (values - shift) / unit
        null_mean = np.full(3, shift)
        null_covariance = np.eye(3) * unit**2

        for method in (llzs_1samp, lrt_1samp):
            with self.subTest(method=method.__name__):
                actual = method(
                    values,
                    popmean=null_mean,
                    popcov=null_covariance,
                )
                expected = method(stable)
                np.testing.assert_allclose(
                    actual.statistic, expected.statistic, rtol=3e-13, atol=1e-14
                )
                np.testing.assert_allclose(
                    actual.pvalue, expected.pvalue, rtol=3e-13, atol=1e-15
                )

    def test_llzs_null_calibration_smoke(self) -> None:
        rng = np.random.default_rng(5123)
        pvalues = np.empty(1200)
        for index in range(pvalues.size):
            sample = rng.normal(size=(50, 100))
            pvalues[index] = llzs_1samp(sample).pvalue
        rejection_rate = float(np.mean(pvalues < 0.05))
        self.assertGreater(rejection_rate, 0.025)
        self.assertLess(rejection_rate, 0.08)

    def test_invalid_nulls_and_classical_rank_fail(self) -> None:
        with self.assertRaises(ValueError):
            llzs_1samp(self.x, popmean=[0.0, 0.0])
        with self.assertRaises(ValueError):
            llzs_1samp(self.x, popcov=np.eye(2))
        with self.assertRaises(ValueError):
            llzs_1samp(self.x, popcov=np.diag([1.0, 1.0, 0.0]))
        with self.assertRaises(ValueError):
            llzs_1samp(
                self.x, popcov=np.array([[1.0, 0.2, 0.0], [0.0, 1, 0], [0, 0, 1]])
            )
        with self.assertRaises(ValueError):
            lrt_1samp(np.eye(4))
        with self.assertRaises(ValueError):
            lrt_1samp(np.ones((8, 3)))
        tiny_sample = np.random.default_rng(0).normal(size=(10, 2)) * 1.0e-150
        nonsymmetric_tiny_null = np.array([[1.0e-300, 1.0e-300], [0.0, 1.0e-300]])
        with self.assertRaises(ValueError):
            lrt_1samp(tiny_sample, popcov=nonsymmetric_tiny_null)

    def test_extreme_finite_alternatives_reach_the_limiting_upper_tail(self) -> None:
        base = np.random.default_rng(0).normal(size=(10, 3))
        llzs = llzs_1samp(base * 1.0e100)
        likelihood_ratio = lrt_1samp(base * 1.0e200)

        self.assertEqual(llzs.statistic, math.inf)
        self.assertEqual(llzs.pvalue, 0.0)
        self.assertEqual(likelihood_ratio.statistic, math.inf)
        self.assertEqual(likelihood_ratio.pvalue, 0.0)


class TestHyodoNishiyama(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(901)
        self.x = rng.normal(size=(24, 6))
        self.y = rng.normal(size=(31, 6))

    @staticmethod
    def _trace_square_estimate(values: np.ndarray) -> tuple[np.ndarray, float]:
        n = values.shape[0]
        centered = values - np.mean(values, axis=0)
        covariance = centered.T @ centered / (n - 1)
        row_norms = np.sum(centered**2, axis=1)
        k = np.sum(row_norms**2) / (n - 1)
        estimate = (
            (n - 1)
            / (n * (n - 2) * (n - 3))
            * (
                (n - 1) * (n - 2) * np.trace(covariance @ covariance)
                + np.trace(covariance) ** 2
                - n * k
            )
        )
        return covariance, float(estimate)

    def test_hn_matches_literal_paper_formula(self) -> None:
        result = hn_2samp(self.x, self.y)
        n1 = self.x.shape[0]
        n2 = self.y.shape[0]
        covariance1, trace_square1 = self._trace_square_estimate(self.x)
        covariance2, trace_square2 = self._trace_square_estimate(self.y)
        trace_cross = float(np.trace(covariance1 @ covariance2))
        mean_difference = np.mean(self.x, axis=0) - np.mean(self.y, axis=0)
        mean_distance = float(
            mean_difference @ mean_difference
            - np.trace(covariance1) / n1
            - np.trace(covariance2) / n2
        )
        covariance_distance = trace_square1 + trace_square2 - 2 * trace_cross
        variance1 = (
            2 * trace_square1 / n1**2
            + 2 * trace_square2 / n2**2
            + 4 * trace_cross / (n1 * n2)
        )
        variance2 = (
            4 * trace_square1**2 / n1**2
            + 4 * trace_square2**2 / n2**2
            + 8 * trace_cross**2 / (n1 * n2)
        )
        expected = mean_distance / math.sqrt(
            variance1
        ) + covariance_distance / math.sqrt(variance2)

        np.testing.assert_allclose(result.statistic, expected, rtol=3e-13)
        np.testing.assert_allclose(
            result.pvalue, stats.norm.sf(expected / math.sqrt(2))
        )
        np.testing.assert_allclose(
            dict(result.estimates)["estimated squared mean distance"],
            mean_distance,
            rtol=5e-14,
            atol=5e-16,
        )
        np.testing.assert_allclose(
            dict(result.estimates)["estimated squared covariance distance"],
            covariance_distance,
            rtol=5e-14,
            atol=5e-16,
        )

    def test_hn_high_dimensional_gram_route_matches_matrix_formula(self) -> None:
        rng = np.random.default_rng(3901)
        first = rng.normal(size=(8, 40))
        second = rng.normal(size=(11, 40))
        result = hn_2samp(first, second)
        n1 = first.shape[0]
        n2 = second.shape[0]
        covariance1, trace_square1 = self._trace_square_estimate(first)
        covariance2, trace_square2 = self._trace_square_estimate(second)
        trace_cross = float(np.trace(covariance1 @ covariance2))
        difference = np.mean(first, axis=0) - np.mean(second, axis=0)
        mean_distance = float(
            difference @ difference
            - np.trace(covariance1) / n1
            - np.trace(covariance2) / n2
        )
        covariance_distance = trace_square1 + trace_square2 - 2 * trace_cross
        variance1 = (
            2 * trace_square1 / n1**2
            + 2 * trace_square2 / n2**2
            + 4 * trace_cross / (n1 * n2)
        )
        variance2 = (
            4 * trace_square1**2 / n1**2
            + 4 * trace_square2**2 / n2**2
            + 8 * trace_cross**2 / (n1 * n2)
        )
        expected = mean_distance / math.sqrt(
            variance1
        ) + covariance_distance / math.sqrt(variance2)

        np.testing.assert_allclose(result.statistic, expected, rtol=4e-13)
        np.testing.assert_allclose(
            result.pvalue,
            stats.norm.sf(expected / math.sqrt(2)),
            rtol=4e-13,
        )

    def test_hn_group_exchange_and_orthogonal_invariance(self) -> None:
        baseline = hn_2samp(self.x, self.y)
        swapped = hn_2samp(self.y, self.x)
        q, _ = np.linalg.qr(np.random.default_rng(73).normal(size=(6, 6)))
        transformed = hn_2samp(1e100 * self.x @ q + 4e99, 1e100 * self.y @ q + 4e99)

        np.testing.assert_allclose(swapped.statistic, baseline.statistic, rtol=2e-13)
        np.testing.assert_allclose(swapped.pvalue, baseline.pvalue, rtol=2e-13)
        np.testing.assert_allclose(
            transformed.statistic, baseline.statistic, rtol=2e-12
        )
        np.testing.assert_allclose(transformed.pvalue, baseline.pvalue, rtol=2e-12)

    def test_hn_preserves_ulps_around_a_huge_common_shift(self) -> None:
        rng = np.random.default_rng(515)
        shift = 1.0e100
        unit = 2.0 * np.spacing(shift)
        for _ in range(27):
            first_base = rng.normal(size=(24, 6))
            second_base = rng.normal(size=(31, 6))
            mode = rng.integers(3)
            if mode == 0:
                second_base += rng.uniform(0.0, 0.8)
            elif mode == 1:
                second_base *= rng.uniform(1.0, 1.5)
            else:
                second_base = second_base @ np.diag(
                    [rng.uniform(1.0, 1.8), 1.0, 1.0, 1.0, 1.0, 1.0]
                )
        first = shift + unit * first_base
        second = shift + unit * second_base
        stable_first = (first - shift) / unit
        stable_second = (second - shift) / unit

        actual = hn_2samp(first, second)
        expected = hn_2samp(stable_first, stable_second)
        np.testing.assert_allclose(actual.statistic, expected.statistic, rtol=5e-13)
        np.testing.assert_allclose(actual.pvalue, expected.pvalue, rtol=5e-13)

    def test_hn_distance_estimates_use_original_measurement_units(self) -> None:
        baseline = dict(hn_2samp(self.x, self.y).estimates)
        transformed = dict(hn_2samp(3.0 * self.x + 8.0, 3.0 * self.y + 8.0).estimates)

        np.testing.assert_allclose(
            transformed["estimated squared mean distance"],
            3.0**2 * baseline["estimated squared mean distance"],
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            transformed["estimated squared covariance distance"],
            3.0**4 * baseline["estimated squared covariance distance"],
            rtol=2e-14,
            atol=2e-14,
        )

    def test_hn_detects_mean_or_covariance_departures(self) -> None:
        mean_alternative = self.y + 4.0
        covariance_alternative = 4.0 * self.y
        self.assertLess(hn_2samp(self.x, mean_alternative).pvalue, 1e-6)
        self.assertLess(hn_2samp(self.x, covariance_alternative).pvalue, 1e-6)

    def test_hn_rejects_undefined_designs(self) -> None:
        with self.assertRaises(ValueError):
            hn_2samp(np.ones((3, 2)), np.ones((5, 2)))
        with self.assertRaises(ValueError):
            hn_2samp(np.ones((5, 2)), np.ones((5, 3)))
        with self.assertRaises(ValueError):
            hn_2samp(np.ones((5, 2)), np.ones((5, 2)))


if __name__ == "__main__":
    unittest.main()

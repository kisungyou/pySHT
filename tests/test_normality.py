"""Independent checks for univariate normality procedures."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import stats

from pysht._results import ResamplingTestResult
from pysht.normality import (
    adjusted_jarque_bera,
    jarque_bera,
    robust_jarque_bera,
    shapiro_francia,
    shapiro_wilk,
)


class ShapiroTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-1.2, -0.4, 0.1, 0.7, 1.0, 1.8, 3.1, 4.2, 5.5, 7.0])

    def test_shapiro_wilk_matches_scipy_after_standardization(self) -> None:
        expected = stats.shapiro(self.x)
        actual = shapiro_wilk(self.x)
        np.testing.assert_allclose(
            actual.statistic, float(expected.statistic), rtol=2e-14, atol=2e-15
        )
        np.testing.assert_allclose(
            actual.pvalue, float(expected.pvalue), rtol=2e-14, atol=2e-15
        )
        self.assertEqual(actual.statistic_name, "W")

    def test_shapiro_francia_matches_literal_formula(self) -> None:
        n = self.x.size
        probabilities = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        scores = stats.norm.ppf(probabilities)
        expected_w = float(np.corrcoef(np.sort(self.x), scores)[0, 1] ** 2)
        log_n = math.log(n)
        log_log_n = math.log(log_n)
        mean = -1.2725 + 1.0521 * (log_log_n - log_n)
        scale = 1.0308 - 0.26758 * (log_log_n + 2.0 / log_n)
        expected_pvalue = float(stats.norm.sf((math.log1p(-expected_w) - mean) / scale))

        actual = shapiro_francia(self.x)
        np.testing.assert_allclose(actual.statistic, expected_w, rtol=2e-14, atol=2e-15)
        np.testing.assert_allclose(
            actual.pvalue, expected_pvalue, rtol=2e-14, atol=2e-15
        )

    def test_perfect_normal_scores_are_a_supported_boundary(self) -> None:
        n = 12
        scores = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
        result = shapiro_francia(scores)
        self.assertAlmostEqual(result.statistic, 1.0, places=14)
        self.assertEqual(result.pvalue, 1.0)

    def test_affine_transformations_preserve_shapiro_results(self) -> None:
        for function in (shapiro_wilk, shapiro_francia):
            baseline = function(self.x)
            transformed = function(-3.5e100 * self.x + 2.0e100)
            with self.subTest(function=function.__name__):
                self.assertTrue(
                    math.isclose(
                        transformed.statistic,
                        baseline.statistic,
                        rel_tol=2e-13,
                        abs_tol=2e-15,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        transformed.pvalue,
                        baseline.pvalue,
                        rel_tol=2e-13,
                        abs_tol=2e-15,
                    )
                )

    def test_representable_ulps_survive_a_large_common_offset(self) -> None:
        for shift in (1.0e14, 1.0e100, 1.0e308):
            unit = np.spacing(shift)
            shifted = shift + unit * np.array(
                [-31.0, -18.0, -9.0, -3.0, 1.0, 4.0, 11.0, 17.0, 28.0, 43.0]
            )
            stable = (shifted - np.min(shifted)) / unit

            for function in (shapiro_wilk, shapiro_francia):
                with self.subTest(shift=shift, function=function.__name__):
                    actual = function(shifted)
                    expected = function(stable)
                    np.testing.assert_allclose(actual.statistic, expected.statistic)
                    np.testing.assert_allclose(actual.pvalue, expected.pvalue)

    def test_opposite_float64_endpoints_use_the_safe_fallback(self) -> None:
        maximum = float(np.finfo(np.float64).max)
        coefficients = np.array([-1.0, -0.5, -0.125, 0.2, 0.55, 1.0])
        extreme = maximum * coefficients
        stable = extreme / maximum

        for function in (shapiro_wilk, shapiro_francia):
            with self.subTest(function=function.__name__):
                actual = function(extreme[::-1])
                expected = function(stable)
                np.testing.assert_allclose(
                    actual.statistic, expected.statistic, rtol=2e-14
                )
                np.testing.assert_allclose(actual.pvalue, expected.pvalue, rtol=2e-14)

    def test_shapiro_domains_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 3"):
            shapiro_wilk([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "at least 5"):
            shapiro_francia([1.0, 2.0, 3.0, 4.0])
        with self.assertRaisesRegex(ValueError, "at most 5000"):
            shapiro_wilk(np.arange(5_001.0))
        for function in (shapiro_wilk, shapiro_francia):
            with (
                self.subTest(function=function.__name__),
                self.assertRaisesRegex(ValueError, "constant sample"),
            ):
                function(np.ones(8))


class MomentNormalityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-1.2, -0.4, 0.1, 0.7, 1.0, 1.8, 3.1, 4.2, 5.5, 7.0])

    @staticmethod
    def _standardized(x: np.ndarray) -> np.ndarray:
        centered = x - np.mean(x)
        return np.asarray(
            centered / math.sqrt(float(np.mean(centered**2))),
            dtype=np.float64,
        )

    def test_jarque_bera_matches_literal_moment_formula(self) -> None:
        z = self._standardized(self.x)
        skewness = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4))
        expected = self.x.size * (skewness**2 / 6.0 + (kurtosis - 3.0) ** 2 / 24.0)
        result = jarque_bera(self.x, calibration="asymptotic")
        self.assertAlmostEqual(result.statistic, expected, places=14)
        self.assertAlmostEqual(result.pvalue, float(stats.chi2.sf(expected, 2)))
        self.assertEqual(result.df, 2.0)

    def test_adjusted_jarque_bera_matches_finite_sample_formula(self) -> None:
        z = self._standardized(self.x)
        n = float(self.x.size)
        skewness = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4))
        variance_skewness = 6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0))
        expected_kurtosis = 3.0 * (n - 1.0) / (n + 1.0)
        variance_kurtosis = (
            24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) ** 2 * (n + 3.0) * (n + 5.0))
        )
        expected = (
            skewness**2 / variance_skewness
            + (kurtosis - expected_kurtosis) ** 2 / variance_kurtosis
        )
        result = adjusted_jarque_bera(self.x, calibration="asymptotic")
        self.assertAlmostEqual(result.statistic, expected, places=14)
        self.assertAlmostEqual(result.pvalue, float(stats.chi2.sf(expected, 2)))

    def test_robust_jarque_bera_uses_published_default_constants(self) -> None:
        z = self._standardized(self.x)
        third = float(np.mean(z**3))
        fourth = float(np.mean(z**4))
        robust_scale = math.sqrt(math.pi / 2.0) * float(
            np.mean(np.abs(z - np.median(z)))
        )
        expected = self.x.size * (
            (third / robust_scale**3) ** 2 / 6.0
            + (fourth / robust_scale**4 - 3.0) ** 2 / 64.0
        )
        result = robust_jarque_bera(self.x, calibration="asymptotic")
        legacy_constant = robust_jarque_bera(
            self.x, c2=24.0, calibration="monte-carlo", n_resamples=31, rng=9
        )
        self.assertAlmostEqual(result.statistic, expected, places=14)
        self.assertNotAlmostEqual(result.statistic, legacy_constant.statistic)

    def test_moment_statistics_preserve_ulps_at_a_large_offset(self) -> None:
        for shift in (1.0e14, 1.0e100, 1.0e308):
            unit = np.spacing(shift)
            shifted = shift + unit * np.array(
                [-29.0, -17.0, -8.0, -4.0, 0.0, 3.0, 9.0, 16.0, 27.0, 41.0]
            )
            stable = (shifted - np.min(shifted)) / unit

            for function in (jarque_bera, adjusted_jarque_bera, robust_jarque_bera):
                with self.subTest(shift=shift, function=function.__name__):
                    actual = function(shifted, calibration="asymptotic")
                    expected = function(stable, calibration="asymptotic")
                    np.testing.assert_allclose(actual.statistic, expected.statistic)
                    np.testing.assert_allclose(actual.pvalue, expected.pvalue)

    def test_moment_statistics_cover_endpoint_and_subnormal_scales(self) -> None:
        maximum = float(np.finfo(np.float64).max)
        smallest = float(np.nextafter(0.0, 1.0))
        endpoint_coefficients = np.array([-1.0, -0.5, -0.125, 0.2, 0.55, 1.0])
        subnormal_coefficients = np.array([0.0, 1.0, 2.0, 4.0, 7.0, 11.0])
        cases = (
            (maximum * endpoint_coefficients, endpoint_coefficients),
            (smallest * subnormal_coefficients, subnormal_coefficients),
        )
        for values, stable in cases:
            for function in (jarque_bera, adjusted_jarque_bera, robust_jarque_bera):
                with self.subTest(
                    scale=float(np.max(np.abs(values))), function=function.__name__
                ):
                    actual = function(values[::-1], calibration="asymptotic")
                    expected = function(stable, calibration="asymptotic")
                    np.testing.assert_allclose(
                        actual.statistic, expected.statistic, rtol=2e-14
                    )
                    np.testing.assert_allclose(
                        actual.pvalue, expected.pvalue, rtol=2e-14
                    )

    def test_monte_carlo_calibration_is_seeded_and_nonzero(self) -> None:
        resamples = 257
        seed = 410
        result = jarque_bera(
            self.x,
            n_resamples=resamples,
            rng=seed,
        )
        repeated = jarque_bera(
            self.x,
            calibration="monte-carlo",
            n_resamples=resamples,
            rng=seed,
        )
        self.assertIsInstance(result, ResamplingTestResult)
        assert isinstance(result, ResamplingTestResult)
        self.assertEqual(result, repeated)

        generator = np.random.default_rng(seed)
        simulated = generator.standard_normal((resamples, self.x.size))
        centered = simulated - np.mean(simulated, axis=1, keepdims=True)
        standardized = centered / np.sqrt(
            np.mean(centered * centered, axis=1, keepdims=True)
        )
        skewness = np.mean(standardized**3, axis=1)
        kurtosis = np.mean(standardized**4, axis=1)
        statistics = self.x.size * (skewness**2 / 6.0 + (kurtosis - 3.0) ** 2 / 24.0)
        exceedances = int(np.count_nonzero(statistics >= result.statistic))
        expected_pvalue = (exceedances + 1.0) / (resamples + 1.0)
        self.assertEqual(result.exceedances, exceedances)
        self.assertEqual(result.pvalue, expected_pvalue)
        self.assertGreater(result.pvalue, 0.0)
        self.assertIsNotNone(result.tail_probability_interval)

    def test_monte_carlo_does_not_touch_numpy_global_rng(self) -> None:
        np.random.seed(90210)
        expected = np.random.random(4)
        np.random.seed(90210)
        adjusted_jarque_bera(
            self.x,
            calibration="monte-carlo",
            n_resamples=31,
            rng=8,
        )
        actual = np.random.random(4)
        np.testing.assert_array_equal(actual, expected)

    def test_adjusted_and_robust_monte_carlo_counts_match_literal_null(self) -> None:
        resamples = 131
        seed = 912
        generator = np.random.default_rng(seed)
        simulated = generator.standard_normal((resamples, self.x.size))
        centered = simulated - np.mean(simulated, axis=1, keepdims=True)
        z = centered / np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
        skewness = np.mean(z**3, axis=1)
        kurtosis = np.mean(z**4, axis=1)
        n = float(self.x.size)

        expected_kurtosis = 3.0 * (n - 1.0) / (n + 1.0)
        variance_skewness = 6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0))
        variance_kurtosis = (
            24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) ** 2 * (n + 3.0) * (n + 5.0))
        )
        simulated_ajb = (
            skewness**2 / variance_skewness
            + (kurtosis - expected_kurtosis) ** 2 / variance_kurtosis
        )
        medians = np.median(z, axis=1, keepdims=True)
        robust_scale = math.sqrt(math.pi / 2.0) * np.mean(np.abs(z - medians), axis=1)
        simulated_rjb = n * (
            (skewness / robust_scale**3) ** 2 / 6.0
            + (kurtosis / robust_scale**4 - 3.0) ** 2 / 64.0
        )

        adjusted = adjusted_jarque_bera(self.x, n_resamples=resamples, rng=seed)
        robust = robust_jarque_bera(self.x, n_resamples=resamples, rng=seed)
        assert isinstance(adjusted, ResamplingTestResult)
        assert isinstance(robust, ResamplingTestResult)
        adjusted_count = int(np.count_nonzero(simulated_ajb >= adjusted.statistic))
        robust_count = int(np.count_nonzero(simulated_rjb >= robust.statistic))
        self.assertEqual(adjusted.exceedances, adjusted_count)
        self.assertEqual(robust.exceedances, robust_count)
        self.assertEqual(
            adjusted.pvalue,
            (adjusted_count + 1.0) / (resamples + 1.0),
        )
        self.assertEqual(
            robust.pvalue,
            (robust_count + 1.0) / (resamples + 1.0),
        )

    def test_location_scale_invariance_for_all_moment_tests(self) -> None:
        for function in (jarque_bera, adjusted_jarque_bera, robust_jarque_bera):
            baseline = function(self.x, calibration="asymptotic")
            transformed = function(
                -2.5e100 * self.x + 4.0e100,
                calibration="asymptotic",
            )
            with self.subTest(function=function.__name__):
                self.assertTrue(
                    math.isclose(
                        transformed.statistic,
                        baseline.statistic,
                        rel_tol=1e-13,
                        abs_tol=1e-14,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        transformed.pvalue,
                        baseline.pvalue,
                        rel_tol=1e-13,
                        abs_tol=1e-14,
                    )
                )

    def test_strict_invalid_input_and_control_validation(self) -> None:
        functions = (jarque_bera, adjusted_jarque_bera, robust_jarque_bera)
        for function in functions:
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(np.ones(8))
                with self.assertRaises(ValueError):
                    function([1.0, 2.0, np.nan, 4.0])
                with self.assertRaises(TypeError):
                    function([True, False, True, False])
                with self.assertRaises(ValueError):
                    function(np.ones((4, 2)))
                with self.assertRaisesRegex(ValueError, "calibration"):
                    function(self.x, calibration="m")
                with self.assertRaises(TypeError):
                    function(self.x, n_resamples=True)
                with self.assertRaises(TypeError):
                    function(self.x, rng=True)
        with self.assertRaisesRegex(ValueError, "at least 4"):
            adjusted_jarque_bera([1.0, 2.0, 4.0])
        with self.assertRaisesRegex(ValueError, "c1"):
            robust_jarque_bera(self.x, c1=0.0)
        with self.assertRaisesRegex(ValueError, "c2"):
            robust_jarque_bera(self.x, c2=math.inf)
        with self.assertRaisesRegex(ValueError, "published"):
            robust_jarque_bera(
                self.x,
                c2=24.0,
                calibration="asymptotic",
            )


if __name__ == "__main__":
    unittest.main()

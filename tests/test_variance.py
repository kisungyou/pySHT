"""Tests for scalar variance procedures."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import stats

from pysht.variance import (
    bartlett,
    brown_forsythe,
    chisquare_1samp,
    f_2samp,
    levene,
)


class OneSampleChiSquareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-1.2, 0.4, 1.8, 2.1, 3.7, 4.2, 5.6])

    def test_matches_literal_chi_square_formula(self) -> None:
        null_variance = 2.25
        result = chisquare_1samp(self.x, null_variance)
        df = self.x.size - 1
        sample_variance = float(np.var(self.x, ddof=1))
        expected_statistic = df * sample_variance / null_variance
        expected_pvalue = min(
            1.0,
            2.0
            * min(
                float(stats.chi2.cdf(expected_statistic, df)),
                float(stats.chi2.sf(expected_statistic, df)),
            ),
        )

        self.assertAlmostEqual(result.statistic, expected_statistic, places=14)
        self.assertAlmostEqual(result.pvalue, expected_pvalue, places=14)
        self.assertEqual(result.df, float(df))
        self.assertEqual(result.estimates[0][0], "sample variance")
        self.assertAlmostEqual(result.estimates[0][1], sample_variance, places=14)

    def test_confidence_interval_inverts_the_pivot(self) -> None:
        level = 0.9
        result = chisquare_1samp(self.x, confidence_level=level)
        assert result.confidence_interval is not None
        df = self.x.size - 1
        numerator = df * float(np.var(self.x, ddof=1))
        alpha = 1.0 - level
        expected = (
            numerator / float(stats.chi2.ppf(1.0 - alpha / 2.0, df)),
            numerator / float(stats.chi2.ppf(alpha / 2.0, df)),
        )
        np.testing.assert_allclose(result.confidence_interval, expected, rtol=2e-14)

    def test_one_sided_tails_and_intervals(self) -> None:
        null_variance = 3.0
        df = self.x.size - 1
        statistic = df * float(np.var(self.x, ddof=1)) / null_variance
        less = chisquare_1samp(
            self.x,
            null_variance,
            alternative="less",
            confidence_level=0.95,
        )
        greater = chisquare_1samp(
            self.x,
            null_variance,
            alternative="greater",
            confidence_level=0.95,
        )

        self.assertAlmostEqual(less.pvalue, float(stats.chi2.cdf(statistic, df)))
        self.assertAlmostEqual(greater.pvalue, float(stats.chi2.sf(statistic, df)))
        assert less.confidence_interval is not None
        assert greater.confidence_interval is not None
        self.assertEqual(less.confidence_interval[0], 0.0)
        self.assertEqual(greater.confidence_interval[1], math.inf)

    def test_joint_rescaling_preserves_inference(self) -> None:
        baseline = chisquare_1samp(self.x, variance=2.0)
        assert baseline.confidence_interval is not None
        for scale in (1e-100, 1e100):
            with self.subTest(scale=scale):
                scaled = chisquare_1samp(
                    scale * self.x,
                    variance=2.0 * scale * scale,
                )
                assert scaled.confidence_interval is not None
                self.assertTrue(
                    math.isclose(
                        scaled.statistic,
                        baseline.statistic,
                        rel_tol=1e-12,
                    )
                )
                self.assertTrue(
                    math.isclose(scaled.pvalue, baseline.pvalue, rel_tol=1e-12)
                )
                np.testing.assert_allclose(
                    np.asarray(scaled.confidence_interval) / (scale * scale),
                    baseline.confidence_interval,
                    rtol=5e-14,
                )

    def test_constant_sample_is_an_explicit_boundary(self) -> None:
        result = chisquare_1samp(np.ones(5), variance=1.0)
        self.assertEqual(result.statistic, 0.0)
        self.assertEqual(result.pvalue, 0.0)
        self.assertEqual(result.confidence_interval, (0.0, 0.0))

    def test_htest_style_display_includes_interval_and_estimate(self) -> None:
        rendered = str(chisquare_1samp(self.x))
        self.assertIn("One-sample chi-square test for variance", rendered)
        self.assertIn("X-squared =", rendered)
        self.assertIn("df =", rendered)
        self.assertIn("95 percent confidence interval:", rendered)
        self.assertIn("sample estimates:", rendered)


class TwoSampleFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-2.1, -0.5, 0.2, 1.7, 3.4, 4.8])
        self.y = np.array([-1.4, -0.7, 0.1, 0.8, 1.2, 2.0, 2.6, 3.1])

    def test_matches_f_distribution_formula(self) -> None:
        result = f_2samp(self.x, self.y)
        df1 = self.x.size - 1
        df2 = self.y.size - 1
        statistic = float(np.var(self.x, ddof=1) / np.var(self.y, ddof=1))
        expected_pvalue = min(
            1.0,
            2.0
            * min(
                float(stats.f.cdf(statistic, df1, df2)),
                float(stats.f.sf(statistic, df1, df2)),
            ),
        )

        self.assertAlmostEqual(result.statistic, statistic, places=14)
        self.assertAlmostEqual(result.pvalue, expected_pvalue, places=14)
        self.assertEqual(result.df, (float(df1), float(df2)))

    def test_confidence_interval_matches_f_quantiles(self) -> None:
        level = 0.9
        result = f_2samp(self.x, self.y, confidence_level=level)
        assert result.confidence_interval is not None
        ratio = float(np.var(self.x, ddof=1) / np.var(self.y, ddof=1))
        df1 = self.x.size - 1
        df2 = self.y.size - 1
        alpha = 1.0 - level
        expected = (
            ratio / float(stats.f.ppf(1.0 - alpha / 2.0, df1, df2)),
            ratio / float(stats.f.ppf(alpha / 2.0, df1, df2)),
        )
        np.testing.assert_allclose(result.confidence_interval, expected, rtol=2e-14)

    def test_directional_swap_identity(self) -> None:
        xy_greater = f_2samp(self.x, self.y, alternative="greater")
        yx_less = f_2samp(self.y, self.x, alternative="less")
        xy_two_sided = f_2samp(self.x, self.y)
        yx_two_sided = f_2samp(self.y, self.x)

        self.assertAlmostEqual(
            xy_greater.statistic * yx_less.statistic,
            1.0,
            places=14,
        )
        self.assertAlmostEqual(xy_greater.pvalue, yx_less.pvalue, places=14)
        self.assertAlmostEqual(xy_two_sided.pvalue, yx_two_sided.pvalue, places=14)
        assert xy_two_sided.confidence_interval is not None
        assert yx_two_sided.confidence_interval is not None
        self.assertAlmostEqual(
            xy_two_sided.confidence_interval[0],
            1.0 / yx_two_sided.confidence_interval[1],
            places=14,
        )
        self.assertAlmostEqual(
            xy_two_sided.confidence_interval[1],
            1.0 / yx_two_sided.confidence_interval[0],
            places=13,
        )

    def test_common_scaling_and_translation_preserve_result(self) -> None:
        baseline = f_2samp(self.x, self.y)
        for scale in (1e-100, 1e100):
            with self.subTest(scale=scale):
                scaled = f_2samp(scale * self.x, scale * self.y)
                self.assertTrue(
                    math.isclose(
                        scaled.statistic,
                        baseline.statistic,
                        rel_tol=1e-12,
                    )
                )
                self.assertTrue(
                    math.isclose(scaled.pvalue, baseline.pvalue, rel_tol=1e-12)
                )

        shifted = f_2samp(self.x + 100.0, self.y + 100.0)
        self.assertAlmostEqual(shifted.statistic, baseline.statistic, places=13)
        self.assertAlmostEqual(shifted.pvalue, baseline.pvalue, places=14)

    def test_constant_sample_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive sample variance"):
            f_2samp(np.ones(5), self.y)
        with self.assertRaisesRegex(ValueError, "positive sample variance"):
            f_2samp(self.x, np.ones(5))


class MultiSampleVarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = np.array([-1.3, -0.2, 0.1, 1.0, 1.8, 2.7])
        self.b = np.array([-2.0, -0.9, 0.5, 0.9, 1.7, 3.2, 4.0])
        self.c = np.array([-1.7, -0.4, 0.0, 0.8, 1.4, 2.2, 3.5, 4.6])

    def test_bartlett_matches_scipy(self) -> None:
        expected = stats.bartlett(self.a, self.b, self.c)
        actual = bartlett(self.a, self.b, self.c)
        self.assertAlmostEqual(actual.statistic, float(expected.statistic), places=13)
        self.assertAlmostEqual(actual.pvalue, float(expected.pvalue), places=13)

    def test_levene_matches_scipy_mean_center(self) -> None:
        expected = stats.levene(self.a, self.b, self.c, center="mean")
        actual = levene(self.a, self.b, self.c)
        self.assertAlmostEqual(actual.statistic, float(expected.statistic), places=14)
        self.assertAlmostEqual(actual.pvalue, float(expected.pvalue), places=14)

    def test_brown_forsythe_matches_scipy_median_center(self) -> None:
        expected = stats.levene(self.a, self.b, self.c, center="median")
        actual = brown_forsythe(self.a, self.b, self.c)
        self.assertAlmostEqual(actual.statistic, float(expected.statistic), places=14)
        self.assertAlmostEqual(actual.pvalue, float(expected.pvalue), places=14)

    def test_group_and_row_order_invariance(self) -> None:
        functions = (bartlett, levene, brown_forsythe)
        for function in functions:
            with self.subTest(function=function.__name__):
                baseline = function(self.a, self.b, self.c)
                reordered = function(self.c[::-1], self.a[::-1], self.b[::-1])
                self.assertAlmostEqual(
                    reordered.statistic, baseline.statistic, places=13
                )
                self.assertAlmostEqual(reordered.pvalue, baseline.pvalue, places=14)

    def test_common_scale_and_translation_invariance(self) -> None:
        functions = (bartlett, levene, brown_forsythe)
        for function in functions:
            baseline = function(self.a, self.b, self.c)
            for scale in (1e-100, 1e100):
                with self.subTest(function=function.__name__, scale=scale):
                    scaled = function(scale * self.a, scale * self.b, scale * self.c)
                    self.assertTrue(
                        math.isclose(
                            scaled.statistic,
                            baseline.statistic,
                            rel_tol=1e-11,
                        )
                    )
                    self.assertTrue(
                        math.isclose(
                            scaled.pvalue,
                            baseline.pvalue,
                            rel_tol=1e-11,
                        )
                    )

            shifted = function(self.a + 50.0, self.b + 50.0, self.c + 50.0)
            self.assertAlmostEqual(shifted.statistic, baseline.statistic, places=12)
            self.assertAlmostEqual(shifted.pvalue, baseline.pvalue, places=13)

    def test_degenerate_cases_are_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive sample variance"):
            bartlett(self.a, np.ones(5))
        for function in (levene, brown_forsythe):
            with (
                self.subTest(function=function.__name__),
                self.assertRaisesRegex(ValueError, "undefined"),
            ):
                function(np.ones(5), np.full(6, 2.0))

        infinite = levene(np.array([-1.0, 1.0]), np.array([-3.0, 3.0]))
        self.assertEqual(infinite.statistic, math.inf)
        self.assertEqual(infinite.pvalue, 0.0)


class VarianceValidationTests(unittest.TestCase):
    def test_extreme_float64_scales_remain_computable(self) -> None:
        baseline_x = np.array([-1.0, -0.5, 0.25, 1.0])
        baseline_y = np.array([-0.8, -0.1, 0.2, 0.7, 0.9])
        baseline = f_2samp(baseline_x, baseline_y)
        huge = f_2samp(1e308 * baseline_x, 1e308 * baseline_y)

        np.testing.assert_allclose(huge.statistic, baseline.statistic, rtol=2e-14)
        np.testing.assert_allclose(huge.pvalue, baseline.pvalue, rtol=2e-14)

        smallest = float(np.nextafter(0.0, 1.0))
        tiny = f_2samp(
            smallest * np.array([0.0, 1.0, 2.0, 3.0]),
            smallest * np.array([0.0, 1.0, 3.0, 4.0]),
        )
        tiny_baseline = f_2samp(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 1.0, 3.0, 4.0]),
        )
        np.testing.assert_allclose(tiny.statistic, tiny_baseline.statistic, rtol=2e-14)
        np.testing.assert_allclose(tiny.pvalue, tiny_baseline.pvalue, rtol=2e-14)

    def test_rejects_invalid_scalar_controls(self) -> None:
        sample = np.array([1.0, 2.0, 4.0])
        for variance in (0.0, -1.0):
            with self.subTest(variance=variance), self.assertRaises(ValueError):
                chisquare_1samp(sample, variance)
        for variance in (True, 1.0 + 2.0j, "abc"):
            with self.subTest(variance=variance), self.assertRaises(TypeError):
                chisquare_1samp(sample, variance)

        with self.assertRaises(ValueError):
            chisquare_1samp(sample, confidence_level=1.0)
        with self.assertRaises(TypeError):
            chisquare_1samp(sample, confidence_level=True)
        with self.assertRaises(ValueError):
            chisquare_1samp(sample, alternative="two.sided")

    def test_rejects_invalid_samples_and_groups(self) -> None:
        valid = np.array([1.0, 2.0, 3.0])
        invalid_samples = (
            np.array([1.0]),
            np.array([[1.0, 2.0]]),
            np.array([1.0, np.nan]),
            np.array([True, False]),
            np.array([1.0 + 1.0j, 2.0]),
        )
        for invalid in invalid_samples:
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                chisquare_1samp(invalid)
            with (
                self.subTest(invalid=invalid),
                self.assertRaises((TypeError, ValueError)),
            ):
                f_2samp(invalid, valid)

        for function in (bartlett, levene, brown_forsythe):
            with (
                self.subTest(function=function.__name__),
                self.assertRaises(ValueError),
            ):
                function(valid)
            with (
                self.subTest(function=function.__name__),
                self.assertRaises((TypeError, ValueError)),
            ):
                function(valid, np.array([1.0, np.inf]))


if __name__ == "__main__":
    unittest.main()

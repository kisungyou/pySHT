"""Tests for scalar variance procedures."""

from __future__ import annotations

import math
import unittest
from decimal import Decimal, localcontext

import numpy as np
from scipy import stats

from pysht.variance import (
    _f_log_tails,
    _log_beta_normalizer,
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
        result = chisquare_1samp(self.x, variance=null_variance)
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

        np.testing.assert_allclose(result.statistic, expected_statistic, rtol=2e-14)
        np.testing.assert_allclose(result.pvalue, expected_pvalue, rtol=2e-14)
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
            variance=null_variance,
            alternative="less",
            confidence_level=0.95,
        )
        greater = chisquare_1samp(
            self.x,
            variance=null_variance,
            alternative="greater",
            confidence_level=0.95,
        )

        self.assertAlmostEqual(less.pvalue, float(stats.chi2.cdf(statistic, df)))
        self.assertAlmostEqual(greater.pvalue, float(stats.chi2.sf(statistic, df)))
        assert less.confidence_interval is not None
        assert greater.confidence_interval is not None
        self.assertEqual(less.confidence_interval[0], 0.0)
        self.assertEqual(greater.confidence_interval[1], math.inf)

    def test_extreme_confidence_interval_retains_upper_tail_quantile(self) -> None:
        level = np.nextafter(1.0, 0.0)
        result = chisquare_1samp(self.x, confidence_level=level)
        assert result.confidence_interval is not None
        alpha = 1.0 - level
        df = self.x.size - 1
        numerator = df * float(np.var(self.x, ddof=1))
        expected_lower = numerator / float(stats.chi2.isf(alpha / 2.0, df))

        np.testing.assert_allclose(
            result.confidence_interval[0], expected_lower, rtol=2e-15
        )
        assert result.confidence_interval[0] > 0.0

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
    def test_large_common_offsets_preserve_representable_spreads(self) -> None:
        for shift in (1.0e14, 1.0e100, 1.0e308):
            unit = float(np.spacing(shift))
            first = shift + unit * np.array([-31.0, -9.0, -2.0, 8.0, 17.0, 29.0])
            second = shift + unit * np.array([-24.0, -13.0, 1.0, 6.0, 19.0, 35.0])
            third = shift + unit * np.array([-37.0, -5.0, 4.0, 12.0, 21.0, 26.0])
            stable = tuple((sample - shift) / unit for sample in (first, second, third))

            if unit <= math.sqrt(float(np.finfo(np.float64).max)):
                chi_shifted = chisquare_1samp(first, variance=unit * unit)
                chi_stable = chisquare_1samp(stable[0])
                np.testing.assert_allclose(chi_shifted.statistic, chi_stable.statistic)
                np.testing.assert_allclose(chi_shifted.pvalue, chi_stable.pvalue)

            for function in (f_2samp, bartlett, levene, brown_forsythe):
                shifted_arguments = (
                    (first, second) if function is f_2samp else (first, second, third)
                )
                stable_arguments = stable[: len(shifted_arguments)]
                with self.subTest(shift=shift, function=function.__name__):
                    shifted_result = function(*shifted_arguments)
                    stable_result = function(*stable_arguments)
                    np.testing.assert_allclose(
                        shifted_result.statistic,
                        stable_result.statistic,
                        rtol=2e-13,
                        atol=5e-13,
                    )
                    np.testing.assert_allclose(
                        shifted_result.pvalue,
                        stable_result.pvalue,
                        rtol=2e-13,
                        atol=5e-13,
                    )

    def test_variance_ratios_retain_groups_across_the_float64_range(self) -> None:
        tiny = 1.0e-300 * np.array([-2.0, -0.5, 0.25, 1.0])
        huge = 1.0e300 * np.array([-1.0, -0.25, 0.5, 2.0])

        result = f_2samp(tiny, huge)
        swapped = f_2samp(huge, tiny)

        self.assertEqual(result.statistic, 0.0)
        self.assertEqual(result.pvalue, 0.0)
        self.assertEqual(swapped.statistic, math.inf)
        self.assertEqual(swapped.pvalue, 0.0)

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
                chisquare_1samp(sample, variance=variance)
        for variance in (True, 1.0 + 2.0j, "abc"):
            with self.subTest(variance=variance), self.assertRaises(TypeError):
                chisquare_1samp(sample, variance=variance)

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


class VarianceTailRangeTests(unittest.TestCase):
    def test_partial_subnormal_scaling_preserves_translated_group_spreads(self) -> None:
        offset = 1e300
        gap = float(np.spacing(offset))
        shifted = np.array([offset, offset + gap])
        centered = np.array([0.0, gap])
        for tiny_gap in (1e-20, 1e-21, 1e-22):
            tiny = np.array([0.0, tiny_gap])
            expected = (4.0 / math.pi) * math.atan(tiny_gap / gap)
            with self.subTest(tiny_gap=tiny_gap):
                actual = f_2samp(shifted, tiny)
                recentered = f_2samp(centered, tiny)
                swapped = f_2samp(tiny, shifted)
                for result in (actual, recentered, swapped):
                    np.testing.assert_allclose(
                        result.pvalue, expected, rtol=5e-13, atol=0.0
                    )
                np.testing.assert_allclose(
                    bartlett(shifted, tiny).statistic,
                    bartlett(centered, tiny).statistic,
                    rtol=5e-13,
                )

    def test_chi_square_one_df_uses_squared_normal_identity(self) -> None:
        # For x=(0,d), X²=d²/2 and its lower tail is erf(abs(d)/2).
        # Include the pivot's normal/subnormal/zero transitions and endpoints.
        for difference in (1.0, 1e-7, 1e-8, 1e-150, 1e-160, 1e-170, 1e-200):
            lower = math.erf(difference / 2.0)
            for alternative, expected in (
                ("less", lower),
                ("greater", 1.0 - lower),
                ("two-sided", min(1.0, 2.0 * min(lower, 1.0 - lower))),
            ):
                with self.subTest(difference=difference, alternative=alternative):
                    result = chisquare_1samp([0.0, difference], alternative=alternative)
                    np.testing.assert_allclose(
                        result.pvalue, expected, rtol=2e-13, atol=0.0
                    )
                    self.assertGreater(result.pvalue, 0.0)

    def test_f_one_df_uses_squared_cauchy_identity_and_reciprocal(self) -> None:
        # For two observations per sample, sqrt(F)=abs(dx/dy).  Evaluate
        # the reciprocal Cauchy tail without forming or squaring that ratio.
        for magnitude in (1.0, 1e7, 1e8, 1e150, 1e160, 1e170, 1e200, 1e308):
            upper = (2.0 / math.pi) * math.atan(1.0 / magnitude)
            for alternative, reverse_alternative, expected in (
                ("less", "greater", 1.0 - upper),
                ("greater", "less", upper),
                ("two-sided", "two-sided", min(1.0, 2.0 * upper)),
            ):
                with self.subTest(magnitude=magnitude, alternative=alternative):
                    result = f_2samp(
                        [0.0, magnitude], [0.0, 1.0], alternative=alternative
                    )
                    reverse = f_2samp(
                        [0.0, 1.0],
                        [0.0, magnitude],
                        alternative=reverse_alternative,
                    )
                    np.testing.assert_allclose(
                        result.pvalue, expected, rtol=2e-13, atol=0.0
                    )
                    np.testing.assert_allclose(
                        reverse.pvalue, expected, rtol=2e-13, atol=0.0
                    )
                    self.assertGreater(result.pvalue, 0.0)

    def test_both_groups_can_lie_outside_the_variance_exponent_range(self) -> None:
        actual = f_2samp([0.0, 1e-200], [0.0, 1e200])
        self.assertEqual(actual.pvalue, 0.0)
        # This less extreme ratio still has a representable Cauchy tail,
        # although both sample variances and their ratio lose range.
        actual = f_2samp([0.0, 1e-170], [0.0, 1e150])
        expected = (4.0 / math.pi) * 1e-320
        self.assertLessEqual(abs(actual.pvalue - expected), np.nextafter(0.0, 1.0))
        self.assertGreater(actual.pvalue, 0.0)

    def test_doubling_happens_before_a_tail_rounds_to_zero(self) -> None:
        magnitude = 1.25e-162
        tiny = np.array([-magnitude, 0.0, magnitude])
        expected = float(2 * Decimal.from_float(magnitude) ** 2)
        self.assertEqual(expected, np.nextafter(0.0, 1.0))
        # At two degrees of freedom, the lower tails are 1-exp(-d²)
        # and d²/(1+d²), respectively.  Their doubled values both round
        # to the smallest positive float, although each single tail is zero.
        self.assertEqual(chisquare_1samp(tiny).pvalue, expected)
        self.assertEqual(f_2samp(tiny, [-1.0, 0.0, 1.0]).pvalue, expected)
        self.assertEqual(chisquare_1samp(tiny, alternative="less").pvalue, 0.0)
        self.assertEqual(
            f_2samp(tiny, [-1.0, 0.0, 1.0], alternative="less").pvalue, 0.0
        )

    def test_f_subnormal_tail_with_a_central_beta_argument(self) -> None:
        # Beta(a,1) has the exact CDF x**a.  Choose a=1075 and x near
        # 1/2 so its single tail rounds to zero, but its doubled tail does
        # not.  An expansion valid only for tiny beta arguments misses this.
        shape = 1075
        beta_argument = 0.5 * 0.75 ** (1.0 / shape)
        magnitude = math.sqrt(beta_argument / (1.0 - beta_argument))
        first = np.zeros(2 * shape + 1)
        first[:2] = [-magnitude, magnitude]
        second = np.array([-1.0, 0.0, 1.0])
        with localcontext() as context:
            context.prec = 100
            square = Decimal.from_float(magnitude) ** 2
            expected = float(2 * (square / (1 + square)) ** shape)
        self.assertEqual(expected, np.nextafter(0.0, 1.0))
        self.assertEqual(f_2samp(first, second).pvalue, expected)
        self.assertEqual(f_2samp(second, first).pvalue, expected)
        self.assertEqual(f_2samp(first, second, alternative="less").pvalue, 0.0)
        self.assertEqual(f_2samp(second, first, alternative="greater").pvalue, 0.0)

    def test_large_df_f_subnormal_tail_near_beta_endpoint(self) -> None:
        # F(2,2a) has survival (a/(a+F))**a.  A near-one beta argument
        # requires the continued fraction at practically reachable sizes;
        # a direct positive series can exceed its iteration budget.
        shape = 2_000_000
        log_pivot = math.log(shape * math.expm1(740.0 / shape))
        with localcontext() as context:
            context.prec = 100
            pivot = Decimal.from_float(log_pivot).exp()
            expected = float((Decimal(shape) / (shape + pivot)) ** shape)
        _, log_upper = _f_log_tails(log_pivot, 2.0, float(2 * shape))
        actual = math.exp(log_upper)
        self.assertGreater(actual, 0.0)
        self.assertLessEqual(abs(actual - expected), 2 * np.nextafter(0.0, 1.0))

        # This half-integer beta case formerly hit the series cap. The
        # reference integrates t=exp(-u/a) with 32/64/128-point Laguerre
        # rules, using a separately evaluated 100-digit Stirling normalizer.
        # All three log integrals give -703.7259717937175. SciPy 1.15's
        # direct F tail loses about 4e-11 relatively here, so it is not a
        # suitable high-precision oracle for this extreme fixture.
        _, log_upper = _f_log_tails(math.log(1400.0), 1.0, 4_000_000.0)
        np.testing.assert_allclose(
            math.exp(log_upper), 2.3751644150037874e-306, rtol=2e-12, atol=0.0
        )
        _, log_upper = _f_log_tails(math.log(1440.0), 1.0, 4_000_000.0)
        self.assertGreater(math.exp(log_upper), math.erfc(math.sqrt(1440.0 / 2.0)))

    def test_unbalanced_f_beta_normalization_uses_exact_integer_identity(self) -> None:
        # For integer b, I_x(a,b) = x**a * sum_{j=0}^{b-1}
        # (a)_j (1-x)**j / j!.  The short finite sum independently
        # resolves the normalization at large a without log-gamma subtraction.
        for shape1, shape2, exponent in ((2_000_000, 3, 730), (10_000_000, 25, 850)):
            log_value = math.log1p(-exponent / shape1)
            log_pivot = (
                log_value
                - math.log(-math.expm1(log_value))
                + math.log(shape2)
                - math.log(shape1)
            )
            with localcontext() as context:
                context.prec = 100
                odds = Decimal.from_float(log_pivot).exp() * shape1 / shape2
                value = odds / (1 + odds)
                term = total = Decimal(1)
                for index in range(1, shape2):
                    term *= (shape1 + index - 1) * (1 - value) / index
                    total += term
                expected = float(2 * value**shape1 * total)
            with self.subTest(shape1=shape1, shape2=shape2):
                self.assertGreater(expected, 0.0)
                for pivot, first, second in (
                    (log_pivot, shape1, shape2),
                    (-log_pivot, shape2, shape1),
                ):
                    tails = _f_log_tails(pivot, float(2 * first), float(2 * second))
                    actual = math.exp(math.log(2.0) + min(tails))
                    self.assertLessEqual(
                        abs(actual - expected),
                        max(4 * np.nextafter(0.0, 1.0), abs(expected) * 2e-11),
                    )

    def test_half_integer_beta_normalization_matches_exact_combinatorics(self) -> None:
        # B(n,1/2) = 4**n / (n * binom(2n,n)).  This exact finite
        # identity independently checks the gamma-ratio expansion at the
        # smallest shape for which the asymptotic branch is permitted.
        shape = 10_000
        with localcontext() as context:
            context.prec = 100
            base = Decimal(4) ** shape / (shape * Decimal(math.comb(2 * shape, shape)))
            for offset in (0, 1, 7, 31):
                expected = base
                for index in range(offset):
                    half_shape = Decimal(index) + Decimal("0.5")
                    expected *= half_shape / (shape + half_shape)
                expected_log = float(expected.ln())
                with self.subTest(half_shape=offset + 0.5):
                    for first, second in (
                        (float(shape), offset + 0.5),
                        (offset + 0.5, float(shape)),
                    ):
                        self.assertLessEqual(
                            abs(_log_beta_normalizer(first, second) - expected_log),
                            2e-13,
                        )

    def test_chi_square_subnormal_upper_tails_use_exact_even_df_identities(
        self,
    ) -> None:
        for df, gamma_argument in ((2, 740.0), (4, 745.0), (20, 780.0), (50, 840.0)):
            magnitude = math.sqrt(gamma_argument)
            sample = np.zeros(df + 1)
            sample[:2] = [-magnitude, magnitude]
            with localcontext() as context:
                context.prec = 100
                value = Decimal.from_float(magnitude) ** 2
                upper = (-value).exp() * sum(
                    value**index / Decimal(math.factorial(index))
                    for index in range(df // 2)
                )
                for alternative, expected in (
                    ("greater", float(upper)),
                    ("two-sided", float(2 * upper)),
                ):
                    with self.subTest(df=df, alternative=alternative):
                        actual = chisquare_1samp(sample, alternative=alternative).pvalue
                        self.assertGreater(actual, 0.0)
                        self.assertLessEqual(
                            abs(actual - expected),
                            max(4 * np.nextafter(0.0, 1.0), abs(expected) * 3e-12),
                        )

    def test_chi_square_one_df_subnormal_upper_tail_matches_erfc(self) -> None:
        for statistic in (1440.0, 1460.0):
            difference = math.sqrt(2.0 * statistic)
            expected = math.erfc(difference / 2.0)
            actual = chisquare_1samp([0.0, difference], alternative="greater").pvalue
            self.assertGreater(actual, 0.0)
            self.assertLessEqual(
                abs(actual - expected),
                max(4 * np.nextafter(0.0, 1.0), abs(expected) * 3e-12),
            )

    def test_chi_square_subnormal_lower_tail_with_a_nonzero_pivot(self) -> None:
        shape = 100
        magnitude = math.sqrt(0.025)
        sample = np.zeros(2 * shape + 1)
        sample[:2] = [-magnitude, magnitude]
        # Use the finite Poisson sum with enough precision to subtract it
        # from one.  This is independent of the positive infinite series
        # used to preserve the implementation's lower tail.
        with localcontext() as context:
            context.prec = 500
            value = Decimal.from_float(magnitude) ** 2
            lower = 1 - (-value).exp() * sum(
                value**index / Decimal(math.factorial(index)) for index in range(shape)
            )
            for alternative, expected in (
                ("less", float(lower)),
                ("two-sided", float(2 * lower)),
            ):
                with self.subTest(alternative=alternative):
                    result = chisquare_1samp(sample, alternative=alternative)
                    self.assertGreater(result.statistic, 0.0)
                    self.assertGreater(result.pvalue, 0.0)
                    self.assertLessEqual(
                        abs(result.pvalue - expected), 4 * np.nextafter(0.0, 1.0)
                    )

    def test_ordinary_ranges_match_scipy_for_unequal_degrees_of_freedom(self) -> None:
        generator = np.random.default_rng(20260910)
        for first_size, second_size in ((2, 3), (3, 2), (8, 17), (41, 23)):
            first = generator.normal(size=first_size)
            second = generator.normal(size=second_size)
            for scale in (1e-3, 0.7, 1.0, 3.0, 1e3):
                scaled_first = scale * first
                ratio = np.var(scaled_first, ddof=1) / np.var(second, ddof=1)
                chi_square = (first_size - 1) * np.var(scaled_first, ddof=1)
                for function, arguments, lower, upper in (
                    (
                        f_2samp,
                        (scaled_first, second),
                        stats.f.cdf(ratio, first_size - 1, second_size - 1),
                        stats.f.sf(ratio, first_size - 1, second_size - 1),
                    ),
                    (
                        chisquare_1samp,
                        (scaled_first,),
                        stats.chi2.cdf(chi_square, first_size - 1),
                        stats.chi2.sf(chi_square, first_size - 1),
                    ),
                ):
                    for alternative, expected in (
                        ("less", lower),
                        ("greater", upper),
                        ("two-sided", min(1.0, 2.0 * min(lower, upper))),
                    ):
                        with self.subTest(
                            function=function.__name__,
                            first_size=first_size,
                            second_size=second_size,
                            scale=scale,
                            alternative=alternative,
                        ):
                            result = function(*arguments, alternative=alternative)
                            np.testing.assert_allclose(
                                result.pvalue, expected, rtol=5e-13, atol=0.0
                            )


if __name__ == "__main__":
    unittest.main()

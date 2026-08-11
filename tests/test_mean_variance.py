"""Independent checks for joint tests of univariate mean and variance."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import special, stats

from pysht.mean_variance import (
    _exact_lrt_pvalue,
    as_1samp,
    lrt_2samp,
    muirhead_2samp,
    pl_2samp,
    pn_2samp,
    zxc_2samp,
)


class TestMeanVarianceOneSample(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-1.2, -0.4, 0.1, 0.5, 0.8, 1.4, 2.1])

    def test_as_matches_literal_likelihood_ratio(self) -> None:
        result = as_1samp(self.x, popmean=0.2, variance=1.7)
        n = self.x.size
        mean = float(np.mean(self.x))
        mle_variance = float(np.sum((self.x - mean) ** 2) / n)
        expected = n * (
            math.log(1.7 / mle_variance)
            + mle_variance / 1.7
            + (mean - 0.2) ** 2 / 1.7
            - 1.0
        )

        np.testing.assert_allclose(result.statistic, expected, rtol=2e-14)
        np.testing.assert_allclose(result.pvalue, stats.chi2.sf(expected, 2))
        self.assertEqual(result.df, 2.0)
        self.assertIn("Arnold-Shavelle", str(result))

    def test_as_is_joint_location_scale_invariant(self) -> None:
        baseline = as_1samp(self.x, popmean=0.2, variance=1.7)
        scale = 1e100
        shift = -3e100
        transformed = as_1samp(
            shift + scale * self.x,
            popmean=shift + scale * 0.2,
            variance=scale**2 * 1.7,
        )

        np.testing.assert_allclose(transformed.statistic, baseline.statistic)
        np.testing.assert_allclose(transformed.pvalue, baseline.pvalue)

    def test_as_rejects_invalid_null_or_boundary_sample(self) -> None:
        with self.assertRaises(ValueError):
            as_1samp([1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            as_1samp(self.x, variance=0.0)
        with self.assertRaises(TypeError):
            as_1samp(self.x, popmean=True)
        with self.assertRaises(TypeError):
            as_1samp(self.x, popvariance=1.0)  # type: ignore[call-arg]

    def test_as_handles_an_overwhelming_float64_scale_departure(self) -> None:
        tiny = np.array([1.0e-300, 2.0e-300, 3.0e-300, 4.0e-300])
        result = as_1samp(tiny, variance=1.0e300)

        n = tiny.size
        scaled = tiny / 1.0e-300
        log_mle_variance = math.log(
            float(np.sum((scaled - np.mean(scaled)) ** 2) / n)
        ) + 2.0 * math.log(1.0e-300)
        log_variance_ratio = log_mle_variance - math.log(1.0e300)
        expected = n * (-1.0 - log_variance_ratio)

        self.assertTrue(math.isfinite(result.statistic))
        np.testing.assert_allclose(result.statistic, expected, rtol=2e-14)
        self.assertEqual(result.pvalue, 0.0)

    def test_as_preserves_ulps_around_a_huge_common_null_location(self) -> None:
        shift = 1.0e100
        unit = 2.0 * np.spacing(shift)
        standardized = np.array(
            [
                0.6244686893779683,
                0.2631942401235858,
                -0.3627114007616576,
                0.5191271973658377,
                -0.6901218808198686,
                0.9451388982345804,
                -0.10904029794559497,
                0.23391758701545876,
            ]
        )
        values = shift + unit * standardized
        stable_coordinates = (values - shift) / unit

        actual = as_1samp(values, popmean=shift, variance=unit**2)
        expected = as_1samp(stable_coordinates)

        np.testing.assert_allclose(actual.statistic, expected.statistic, rtol=2e-14)
        np.testing.assert_allclose(actual.pvalue, expected.pvalue, rtol=2e-14)


class TestMeanVarianceTwoSample(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([-1.2, -0.4, 0.1, 0.5, 0.8, 1.4, 2.1])
        self.y = np.array([-0.9, -0.2, 0.3, 0.7, 1.0, 1.8, 2.4, 2.7])

    def _literal_log_lambda(self) -> float:
        n = self.x.size
        m = self.y.size
        total = n + m
        mean_x = float(np.mean(self.x))
        mean_y = float(np.mean(self.y))
        ss_x = float(np.sum((self.x - mean_x) ** 2))
        ss_y = float(np.sum((self.y - mean_y) ** 2))
        pooled_mean = (n * mean_x + m * mean_y) / total
        pooled_ss = float(
            np.sum((self.x - pooled_mean) ** 2) + np.sum((self.y - pooled_mean) ** 2)
        )
        return (
            0.5 * n * math.log(ss_x / n)
            + 0.5 * m * math.log(ss_y / m)
            - 0.5 * total * math.log(pooled_ss / total)
        )

    def test_lrt_matches_literal_formula(self) -> None:
        log_lambda = self._literal_log_lambda()
        result = lrt_2samp(self.x, self.y)

        np.testing.assert_allclose(result.statistic, -2.0 * log_lambda)
        np.testing.assert_allclose(result.pvalue, stats.chi2.sf(-2 * log_lambda, 2))

    def test_pn_moments_and_lower_tail_match_published_formula(self) -> None:
        result = pn_2samp(self.x, self.y)
        n = self.x.size
        m = self.y.size
        total = n + m
        log_mean = (
            0.5 * total * math.log(total)
            - 0.5 * n * math.log(n)
            - 0.5 * m * math.log(m)
            + special.gammaln(n - 0.5)
            + special.gammaln(m - 0.5)
            - special.gammaln(total - 0.5)
            + special.gammaln(0.5 * (total - 1))
            - special.gammaln(0.5 * (n - 1))
            - special.gammaln(0.5 * (m - 1))
        )
        log_second = (
            total * math.log(total)
            - n * math.log(n)
            - m * math.log(m)
            + special.gammaln(0.5 * (3 * n - 1))
            + special.gammaln(0.5 * (3 * m - 1))
            - special.gammaln(0.5 * (3 * total - 1))
            + special.gammaln(0.5 * (total - 1))
            - special.gammaln(0.5 * (n - 1))
            - special.gammaln(0.5 * (m - 1))
        )
        mean = math.exp(log_mean)
        variance = math.exp(log_second) - mean**2
        concentration = mean * (1 - mean) / variance - 1
        shape1 = mean * concentration
        shape2 = (1 - mean) * concentration
        likelihood_ratio = math.exp(self._literal_log_lambda())

        np.testing.assert_allclose(result.statistic, likelihood_ratio)
        np.testing.assert_allclose(
            result.pvalue,
            stats.beta.cdf(likelihood_ratio, shape1, shape2),
            rtol=2e-13,
        )
        # This fixture catches the reversed upper tail in SHT 0.1.9.
        np.testing.assert_allclose(result.pvalue, 0.6917901786656631, rtol=2e-13)
        self.assertEqual(
            tuple(name for name, _ in result.diagnostics),
            ("beta shape 1", "beta shape 2"),
        )
        np.testing.assert_allclose(
            tuple(float(value) for _, value in result.diagnostics),
            (shape1, shape2),
            rtol=3e-15,
        )
        self.assertEqual(result.estimates, ())

    def test_pl_matches_independent_component_tests(self) -> None:
        result = pl_2samp(self.x, self.y)
        t_pvalue = float(stats.ttest_ind(self.x, self.y, equal_var=True).pvalue)
        ratio = np.var(self.x, ddof=1) / np.var(self.y, ddof=1)
        f_pvalue = float(
            min(
                1.0,
                2.0
                * min(
                    stats.f.cdf(ratio, self.x.size - 1, self.y.size - 1),
                    stats.f.sf(ratio, self.x.size - 1, self.y.size - 1),
                ),
            ),
        )
        expected = -2.0 * math.log(t_pvalue * f_pvalue)

        np.testing.assert_allclose(result.statistic, expected)
        np.testing.assert_allclose(result.pvalue, stats.chi2.sf(expected, 4))
        self.assertEqual(
            tuple(name for name, _ in result.diagnostics),
            ("pooled t p-value", "F-test p-value"),
        )
        np.testing.assert_allclose(
            tuple(float(value) for _, value in result.diagnostics),
            (t_pvalue, f_pvalue),
            rtol=3e-15,
        )
        self.assertEqual(result.estimates, ())

    def test_muirhead_uses_corrected_upper_tail(self) -> None:
        result = muirhead_2samp(self.x, self.y)
        n = self.x.size
        m = self.y.size
        total = n + m
        ratio_sum = total / n + total / m - 1
        rho = 1 - 22 / (24 * total) * ratio_sum
        gamma = (
            0.5 * ((total / n) ** 2 + (total / m) ** 2 - 1) - 121 / 96 * ratio_sum**2
        )
        expected_statistic = -2 * rho * self._literal_log_lambda()
        coefficient = gamma / (rho**2 * total**2)
        expected_pvalue = stats.chi2.sf(expected_statistic, 2) + coefficient * (
            stats.chi2.sf(expected_statistic, 6) - stats.chi2.sf(expected_statistic, 2)
        )

        np.testing.assert_allclose(result.statistic, expected_statistic)
        np.testing.assert_allclose(result.pvalue, expected_pvalue)
        self.assertGreater(result.pvalue, 0.5)
        self.assertEqual(
            result.diagnostics,
            (("rho", rho), ("second-order coefficient", coefficient)),
        )
        self.assertEqual(result.estimates, ())

    def test_zxc_matches_exact_dirichlet_fixture(self) -> None:
        result = zxc_2samp(self.x, self.y)

        np.testing.assert_allclose(
            result.statistic, math.exp(self._literal_log_lambda())
        )
        # Independent high-accuracy integration of the published Dirichlet
        # probability; the legacy 500-point nested rule gives 0.6921061.
        np.testing.assert_allclose(result.pvalue, 0.6918190488406402, rtol=3e-12)

    def test_exact_test_is_one_at_the_fitted_null_boundary(self) -> None:
        result = zxc_2samp(self.x, self.x.copy())
        np.testing.assert_allclose(result.statistic, 1.0)
        np.testing.assert_allclose(result.pvalue, 1.0)

    def test_exact_test_splits_unbalanced_integral_at_its_cusp(self) -> None:
        first = np.array([-1.0, 1.0])
        second = np.linspace(-1.0, 1.0, 20) + 3.45834340628787

        result = zxc_2samp(first, second)

        np.testing.assert_allclose(result.pvalue, 0.00098113588247435, rtol=2e-11)

    def test_exact_test_resolves_roots_in_the_far_log_tail(self) -> None:
        difference = 2.0 * math.sqrt(math.exp(25.0) - 1.0)
        first = np.array([-1.0, 1.0])
        second = np.array([difference - 1.0, difference + 1.0])

        result = zxc_2samp(first, second)

        # For n=m=2 the Dirichlet(1/2,1/2,1/2) null admits a separate
        # log-coordinate quadrature oracle for P(4XY <= Lambda).
        np.testing.assert_allclose(result.pvalue, 1.2412994009417087e-10, rtol=2e-11)

    def test_exact_logit_quadrature_keeps_a_representable_far_tail(self) -> None:
        # A 500-decimal mpmath integration of the Dirichlet probability gives
        # this nonzero result even though exp(-800) underflows in float64.
        expected = 2.4572424773752458e-172

        actual = _exact_lrt_pvalue(-800.0, 2, 2)

        self.assertGreater(actual, 0.0)
        np.testing.assert_allclose(actual, expected, rtol=2e-12)

    def test_exact_logit_quadrature_scales_an_unbalanced_tiny_integral(
        self,
    ) -> None:
        # Independent high-precision conditional-beta quadrature.  Integrating
        # this probability with a fixed absolute tolerance loses about half
        # of its mass because the whole integrand is around 1e-21.
        expected = 2.2176768591905327e-21

        actual = _exact_lrt_pvalue(-50.0, 20, 200)

        np.testing.assert_allclose(actual, expected, rtol=2e-12)

    def test_exact_test_preserves_group_variances_across_extreme_scales(
        self,
    ) -> None:
        first = np.array([-1.0e-87, 1.0e-87])
        second = np.array([-1.0e87, 1.0e87])

        exact = zxc_2samp(first, second)
        swapped = zxc_2samp(second, first)
        asymptotic = lrt_2samp(first, second)
        combined = pl_2samp(first, second)

        # Lambda underflows, but its log and the exact Dirichlet probability
        # remain representable when each group's sum of squares keeps its own
        # numerical scale.
        self.assertEqual(exact.statistic, 0.0)
        np.testing.assert_allclose(exact.pvalue, 2.5658075058301904e-172, rtol=2e-12)
        np.testing.assert_equal(swapped.statistic, exact.statistic)
        np.testing.assert_equal(swapped.pvalue, exact.pvalue)
        np.testing.assert_allclose(asymptotic.statistic, 1599.826636001616, rtol=2e-14)
        self.assertEqual(asymptotic.pvalue, 0.0)
        component_probability = 4.0e-174 / math.pi
        np.testing.assert_allclose(
            combined.statistic, -2.0 * math.log(component_probability), rtol=2e-14
        )
        np.testing.assert_allclose(
            combined.pvalue,
            component_probability * (1.0 - math.log(component_probability)),
            rtol=2e-13,
        )

    def test_two_sample_summaries_are_row_order_invariant_at_mixed_scales(
        self,
    ) -> None:
        first = np.array([-4.0, -1.0, 8.0]) * 1.0e-40
        second = np.array([-7.0, 1.0, 11.0]) * 1.0e40
        first_permutation = first[[2, 0, 1]]
        second_permutation = second[[1, 2, 0]]

        for method in (pn_2samp, pl_2samp, muirhead_2samp, lrt_2samp, zxc_2samp):
            with self.subTest(method=method.__name__):
                baseline = method(first, second)
                permuted = method(first_permutation, second_permutation)
                np.testing.assert_equal(permuted.statistic, baseline.statistic)
                np.testing.assert_equal(permuted.pvalue, baseline.pvalue)

    def test_every_method_is_exchange_location_and_scale_invariant(self) -> None:
        methods = (pn_2samp, pl_2samp, muirhead_2samp, lrt_2samp, zxc_2samp)
        for method in methods:
            with self.subTest(method=method.__name__):
                baseline = method(self.x, self.y)
                swapped = method(self.y, self.x)
                transformed = method(4.5 + 1e100 * self.x, 4.5 + 1e100 * self.y)
                np.testing.assert_allclose(swapped.statistic, baseline.statistic)
                np.testing.assert_allclose(swapped.pvalue, baseline.pvalue, rtol=2e-10)
                np.testing.assert_allclose(
                    transformed.statistic, baseline.statistic, rtol=2e-13
                )
                np.testing.assert_allclose(
                    transformed.pvalue, baseline.pvalue, rtol=2e-10
                )

    def test_strong_alternative_reduces_every_pvalue(self) -> None:
        alternative = 8.0 + 3.0 * self.y
        for method in (pn_2samp, pl_2samp, muirhead_2samp, lrt_2samp, zxc_2samp):
            with self.subTest(method=method.__name__):
                self.assertLess(method(self.x, alternative).pvalue, 0.01)

    def test_heterogeneous_float64_scales_reach_the_limiting_tail(self) -> None:
        tiny = np.array([1.0e-300, 2.0e-300, 3.0e-300, 4.0e-300])
        huge = np.array([0.9e300, 1.0e300, 1.1e300, 1.2e300])

        for method in (pn_2samp, pl_2samp, muirhead_2samp, lrt_2samp, zxc_2samp):
            with self.subTest(method=method.__name__):
                forward = method(tiny, huge)
                reverse = method(huge, tiny)
                self.assertEqual(forward.pvalue, 0.0)
                self.assertEqual(reverse.pvalue, 0.0)
                np.testing.assert_equal(reverse.statistic, forward.statistic)

    def test_shared_summary_preserves_ulps_around_a_huge_common_shift(self) -> None:
        shift = 1.0e300
        unit = 4.0 * np.spacing(shift)
        first_base = np.array(
            [
                -0.48156285818994926,
                -0.5834075004641189,
                -0.8621605020712843,
                -1.4881746132515903,
                0.2163068331092121,
                0.9843763506958761,
                -0.5430841410126281,
                -0.5586150390437845,
            ]
        )
        second_base = np.array(
            [
                0.4011683363894241,
                -0.8671506737324277,
                2.774640441647995,
                1.5706998538247066,
                0.07479646464255792,
                1.3863232900018068,
                0.4300055455995483,
                0.7248713366129158,
                1.4731529810528485,
            ]
        )
        first = shift + unit * first_base
        second = shift + unit * second_base
        anchor = first[0]
        stable_first = (first - anchor) / unit
        stable_second = (second - anchor) / unit

        for method in (pn_2samp, pl_2samp, muirhead_2samp, lrt_2samp, zxc_2samp):
            with self.subTest(method=method.__name__):
                actual = method(first, second)
                expected = method(stable_first, stable_second)
                np.testing.assert_allclose(
                    actual.statistic, expected.statistic, rtol=3e-13, atol=1e-15
                )
                np.testing.assert_allclose(
                    actual.pvalue, expected.pvalue, rtol=3e-11, atol=1e-15
                )

    def test_invalid_or_degenerate_samples_fail(self) -> None:
        for method in (pn_2samp, pl_2samp, muirhead_2samp, lrt_2samp, zxc_2samp):
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method([1.0, 1.0, 1.0], self.y)
                with self.assertRaises(ValueError):
                    method(self.x, [1.0, np.nan])


if __name__ == "__main__":
    unittest.main()

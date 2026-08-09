"""Independent checks for tests of population means."""

from __future__ import annotations

import unittest
import warnings

import numpy as np
from scipy import stats

from pysht.mean import (
    anova_oneway,
    hotelling_1samp,
    hotelling_2samp,
    ttest_1samp,
    ttest_2samp,
)


class TestUnivariateMeanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([1.2, 2.4, -0.7, 3.1, 0.8, 1.9, -1.1])
        self.y = np.array([0.4, -0.2, 1.7, 0.3, 2.2, -0.8, 1.1, 0.6, 1.4])

    def test_one_sample_matches_scipy_for_every_alternative(self) -> None:
        for alternative in ("two-sided", "less", "greater"):
            with self.subTest(alternative=alternative):
                actual = ttest_1samp(
                    self.x,
                    popmean=0.35,
                    alternative=alternative,
                    confidence_level=0.9,
                )
                expected = stats.ttest_1samp(
                    self.x,
                    popmean=0.35,
                    alternative=alternative,
                )
                expected_interval = expected.confidence_interval(confidence_level=0.9)

                np.testing.assert_allclose(actual.statistic, expected.statistic)
                np.testing.assert_allclose(actual.pvalue, expected.pvalue)
                np.testing.assert_allclose(
                    actual.confidence_interval,
                    (expected_interval.low, expected_interval.high),
                )
                self.assertEqual(actual.df, self.x.size - 1)

    def test_independent_tests_match_scipy(self) -> None:
        for equal_var in (False, True):
            for alternative in ("two-sided", "less", "greater"):
                with self.subTest(equal_var=equal_var, alternative=alternative):
                    actual = ttest_2samp(
                        self.x,
                        self.y,
                        equal_var=equal_var,
                        alternative=alternative,
                    )
                    expected = stats.ttest_ind(
                        self.x,
                        self.y,
                        equal_var=equal_var,
                        alternative=alternative,
                    )
                    np.testing.assert_allclose(actual.statistic, expected.statistic)
                    np.testing.assert_allclose(actual.pvalue, expected.pvalue)
                    np.testing.assert_allclose(actual.df, expected.df)

    def test_paired_test_matches_scipy(self) -> None:
        paired_y = np.array([0.8, 2.0, -0.4, 2.7, 0.1, 1.6, -0.3])
        actual = ttest_2samp(self.x, paired_y, paired=True, alternative="greater")
        expected = stats.ttest_rel(self.x, paired_y, alternative="greater")

        np.testing.assert_allclose(actual.statistic, expected.statistic)
        np.testing.assert_allclose(actual.pvalue, expected.pvalue)
        np.testing.assert_allclose(actual.df, expected.df)

    def test_independent_test_allows_one_constant_sample(self) -> None:
        constant = np.ones(5)
        variable = np.array([0.1, 0.7, 1.4, 2.1, 3.0, 4.2])
        actual = ttest_2samp(constant, variable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = stats.ttest_ind(constant, variable, equal_var=False)

        np.testing.assert_allclose(actual.statistic, expected.statistic)
        np.testing.assert_allclose(actual.pvalue, expected.pvalue)
        np.testing.assert_allclose(actual.df, expected.df)

    def test_t_statistics_are_stable_at_large_scale(self) -> None:
        x = np.array([1.0, 2.0, 4.0, 5.0])
        baseline = ttest_1samp(x, popmean=2.5)
        scaled = ttest_1samp(1e200 * x, popmean=2.5e200)

        np.testing.assert_allclose(scaled.statistic, baseline.statistic)
        np.testing.assert_allclose(scaled.pvalue, baseline.pvalue)
        self.assertTrue(np.isfinite(scaled.estimates[0][1]))

    def test_anova_matches_scipy_and_is_scale_invariant(self) -> None:
        groups = (
            np.array([1.0, 1.4, 0.7, 1.8]),
            np.array([2.1, 2.4, 1.8, 2.7, 2.0]),
            np.array([-0.2, 0.3, 0.1, -0.5]),
        )
        actual = anova_oneway(*groups)
        expected = stats.f_oneway(*groups)
        scaled = anova_oneway(*(1e200 * group for group in groups))

        np.testing.assert_allclose(actual.statistic, expected.statistic)
        np.testing.assert_allclose(actual.pvalue, expected.pvalue)
        np.testing.assert_allclose(scaled.statistic, actual.statistic)
        np.testing.assert_allclose(scaled.pvalue, actual.pvalue)
        self.assertEqual(actual.df, (2.0, 10.0))

    def test_invalid_or_degenerate_inputs_fail_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            ttest_1samp([1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            ttest_1samp([1.0, np.nan])
        with self.assertRaises(ValueError):
            ttest_2samp(self.x, self.y, paired=True)
        with self.assertRaises(ValueError):
            ttest_2samp(self.x, self.x + 1, paired=True, equal_var=True)
        with self.assertRaises(ValueError):
            anova_oneway([1.0, 1.0], [1.0, 1.0])
        with self.assertRaises(TypeError):
            ttest_1samp(self.x, alternative=object())  # type: ignore[arg-type]


class TestHotellingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [2.1, 1.2],
                [1.4, 0.7],
                [2.8, 1.9],
                [0.9, 1.4],
                [1.7, 0.3],
                [2.5, 1.1],
            ]
        )
        self.y = np.array(
            [
                [1.1, 0.2],
                [0.7, 1.3],
                [1.8, 0.8],
                [0.4, -0.1],
                [1.5, 0.5],
                [0.9, 0.9],
                [1.2, -0.4],
            ]
        )

    def test_one_sample_matches_literal_formula(self) -> None:
        null = np.array([1.0, 0.5])
        actual = hotelling_1samp(self.x, popmean=null)
        n, p = self.x.shape
        difference = np.mean(self.x, axis=0) - null
        covariance = np.cov(self.x, rowvar=False, ddof=1)
        expected_t2 = n * float(difference @ np.linalg.solve(covariance, difference))
        expected_f = (n - p) * expected_t2 / (p * (n - 1))
        expected_p = stats.f.sf(expected_f, p, n - p)

        np.testing.assert_allclose(actual.statistic, expected_t2, rtol=1e-13)
        np.testing.assert_allclose(actual.pvalue, expected_p, rtol=1e-13)
        self.assertEqual(actual.df, (2.0, 4.0))

    def test_one_sample_is_invariant_to_nonsingular_linear_transform(self) -> None:
        null = np.array([1.0, 0.5])
        transform = np.array([[2.0, -0.4], [0.3, 1.5]])
        baseline = hotelling_1samp(self.x, popmean=null)
        transformed = hotelling_1samp(
            self.x @ transform,
            popmean=null @ transform,
        )

        np.testing.assert_allclose(transformed.statistic, baseline.statistic)
        np.testing.assert_allclose(transformed.pvalue, baseline.pvalue)

    def test_two_sample_matches_literal_formula_and_group_swap(self) -> None:
        actual = hotelling_2samp(self.x, self.y)
        swapped = hotelling_2samp(self.y, self.x)
        nx, p = self.x.shape
        ny = self.y.shape[0]
        difference = np.mean(self.x, axis=0) - np.mean(self.y, axis=0)
        pooled = (
            (nx - 1) * np.cov(self.x, rowvar=False, ddof=1)
            + (ny - 1) * np.cov(self.y, rowvar=False, ddof=1)
        ) / (nx + ny - 2)
        expected_t2 = (
            nx
            * ny
            / (nx + ny)
            * float(difference @ np.linalg.solve(pooled, difference))
        )
        expected_f = (nx + ny - p - 1) * expected_t2 / (p * (nx + ny - 2))
        expected_p = stats.f.sf(expected_f, p, nx + ny - p - 1)

        np.testing.assert_allclose(actual.statistic, expected_t2, rtol=1e-13)
        np.testing.assert_allclose(actual.pvalue, expected_p, rtol=1e-13)
        np.testing.assert_allclose(swapped.statistic, actual.statistic)
        np.testing.assert_allclose(swapped.pvalue, actual.pvalue)

    def test_paired_hotelling_equals_one_sample_difference_test(self) -> None:
        paired_y = self.x - np.array(
            [
                [0.2, -0.1],
                [0.1, 0.2],
                [0.4, 0.0],
                [-0.1, 0.3],
                [0.3, -0.2],
                [0.0, 0.1],
            ]
        )
        paired = hotelling_2samp(self.x, paired_y, paired=True)
        one_sample = hotelling_1samp(self.x - paired_y)

        np.testing.assert_allclose(paired.statistic, one_sample.statistic)
        np.testing.assert_allclose(paired.pvalue, one_sample.pvalue)

    def test_hotelling_rejects_undefined_designs(self) -> None:
        with self.assertRaises(ValueError):
            hotelling_1samp(np.eye(3))
        with self.assertRaises(ValueError):
            hotelling_1samp(np.ones((5, 2)))
        with self.assertRaises(ValueError):
            hotelling_2samp(np.ones((5, 2)), np.ones((5, 3)))
        with self.assertRaises(ValueError):
            hotelling_2samp(self.x, self.y, paired=True)


if __name__ == "__main__":
    unittest.main()

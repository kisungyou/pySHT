"""Independent formula, invariance, and boundary checks for covariance tests."""

from __future__ import annotations

import inspect
import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

from pysht import covariance
from pysht._results import BayesFactorTestResult, HypothesisTestResult
from pysht.covariance import (
    clx_2samp,
    lc_2samp,
    lyl_2samp,
    schott_2001_ksamp,
    schott_2007_ksamp,
    wl_1samp,
    wl_2samp,
)

fisher_1samp = covariance._fisher_1samp


def _center(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return values - np.mean(values, axis=0)


def _literal_lc_a(values: NDArray[np.float64]) -> float:
    n = int(values.shape[0])
    first = 0.0
    second = 0.0
    third = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gij = float(values[i] @ values[j])
            first += gij * gij
            for k in range(n):
                if k in (i, j):
                    continue
                second += gij * float(values[j] @ values[k])
                for ell in range(n):
                    if ell in (i, j, k):
                        continue
                    third += gij * float(values[k] @ values[ell])
    return (
        first / (n * (n - 1))
        - 2.0 * second / (n * (n - 1) * (n - 2))
        + third / (n * (n - 1) * (n - 2) * (n - 3))
    )


def _literal_lc_c(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    n1 = int(first.shape[0])
    n2 = int(second.shape[0])
    term1 = 0.0
    term2 = 0.0
    term3 = 0.0
    term4 = 0.0
    for i in range(n1):
        for j in range(n2):
            hij = float(first[i] @ second[j])
            term1 += hij * hij
            for k in range(n1):
                if k != i:
                    term2 += hij * float(first[k] @ second[j])
                    for ell in range(n2):
                        if ell != j:
                            term4 += hij * float(first[k] @ second[ell])
            for ell in range(n2):
                if ell != j:
                    term3 += hij * float(first[i] @ second[ell])
    return (
        term1 / (n1 * n2)
        - term2 / (n1 * n2 * (n1 - 1))
        - term3 / (n1 * n2 * (n2 - 1))
        + term4 / (n1 * n2 * (n1 - 1) * (n2 - 1))
    )


def _literal_lyl(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    a0: float,
    b0: float,
    gamma: float,
) -> NDArray[np.float64]:
    first = x
    second = y
    pooled = np.concatenate((first, second), axis=0)
    n1 = int(first.shape[0])
    p = int(first.shape[1])
    n2 = int(second.shape[0])
    n = n1 + n2
    constant = (
        0.5 * math.log(gamma / (1.0 + gamma))
        + special.gammaln(n1 / 2.0 + a0)
        + special.gammaln(n2 / 2.0 + a0)
        - special.gammaln(n / 2.0 + a0)
        + a0 * math.log(b0)
        - special.gammaln(a0)
    )

    def tau(response: NDArray[np.float64], predictor: NDArray[np.float64]) -> float:
        coefficient = float(response @ predictor) / float(predictor @ predictor)
        residual = response - coefficient * predictor
        return float(residual @ residual) / response.size

    output = np.full((p, p), -math.inf)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            tau1 = tau(first[:, i], first[:, j])
            tau2 = tau(second[:, i], second[:, j])
            tau0 = tau(pooled[:, i], pooled[:, j])
            output[i, j] = (
                constant
                - (n1 / 2.0 + a0) * math.log(b0 + n1 * tau1 / 2.0)
                - (n2 / 2.0 + a0) * math.log(b0 + n2 * tau2 / 2.0)
                + (n / 2.0 + a0) * math.log(b0 + n * tau0 / 2.0)
            )
    return output


class CovarianceFixtures(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [1.2, -0.3, 0.7],
                [-0.4, 1.1, 0.2],
                [0.8, 0.6, -1.0],
                [1.7, -0.8, 0.4],
                [-1.1, 0.3, 1.5],
                [0.2, -1.4, -0.6],
                [1.0, 0.9, 0.1],
                [-0.7, -0.2, 0.8],
            ]
        )
        self.y = np.array(
            [
                [0.5, 0.1, -0.3],
                [-0.8, 1.4, 0.6],
                [1.1, -0.5, 0.9],
                [0.2, 0.8, -1.2],
                [-1.3, -0.7, 0.4],
                [0.9, 1.0, 0.2],
                [-0.1, -1.2, 1.3],
                [1.4, 0.4, -0.8],
                [-0.6, 0.2, 0.5],
            ]
        )
        self.z = np.array(
            [
                [0.2, -0.9, 1.0],
                [1.3, 0.4, -0.1],
                [-0.5, 1.2, 0.8],
                [0.7, -0.2, -1.1],
                [-1.0, -0.4, 0.2],
                [0.4, 0.9, 0.5],
                [1.1, -1.3, -0.6],
                [-0.3, 0.5, 1.4],
                [0.8, 0.1, -0.7],
                [-0.9, 1.0, 0.3],
            ]
        )


class TestPublicContract(CovarianceFixtures):
    def test_exports_and_signatures_are_exact(self) -> None:
        self.assertEqual(
            set(covariance.__all__),
            {
                "wl_1samp",
                "lc_2samp",
                "clx_2samp",
                "wl_2samp",
                "lyl_2samp",
                "schott_2001_ksamp",
                "schott_2007_ksamp",
            },
        )
        self.assertEqual(
            str(inspect.signature(covariance._fisher_1samp)),
            "(x: 'ArrayLike', *, popcov: 'ArrayLike | None' = None, "
            "variant: 'int' = 1) -> 'HypothesisTestResult'",
        )
        self.assertEqual(
            inspect.signature(wl_1samp).parameters["n_projections"].default, 25
        )
        self.assertEqual(
            inspect.signature(wl_2samp).parameters["n_projections"].default, 50
        )
        self.assertEqual(inspect.signature(lyl_2samp).parameters["a0"].default, 0.01)
        self.assertEqual(inspect.signature(lyl_2samp).parameters["b0"].default, 0.01)
        self.assertEqual(inspect.signature(lyl_2samp).parameters["alpha"].default, 2.01)
        self.assertIsNone(inspect.signature(lyl_2samp).parameters["gamma"].default)

    def test_results_are_immutable_and_have_htest_rendering(self) -> None:
        result = clx_2samp(self.x, self.y)
        self.assertIsInstance(result, HypothesisTestResult)
        with self.assertRaises(FrozenInstanceError):
            result.statistic = 0.0  # type: ignore[misc]
        rendered = str(result)
        self.assertIn("p-value", rendered)
        self.assertIn("alternative hypothesis", rendered)
        self.assertIn("CLX", rendered)


class TestFisher(CovarianceFixtures):
    def _literal(self, popcov: np.ndarray, variant: int) -> float:
        centered = _center(self.x)
        root = np.linalg.cholesky(popcov)
        whitened = np.linalg.solve(root, centered.T).T
        n = int(self.x.shape[0] - 1)
        p = int(self.x.shape[1])
        sample_covariance = whitened.T @ whitened / n
        powers = [sample_covariance]
        for _ in range(3):
            powers.append(powers[-1] @ sample_covariance)
        tr1, tr2, tr3, tr4 = (float(np.trace(power)) for power in powers)
        nf = float(n)
        a1 = tr1 / p
        a2 = nf**2 / ((nf - 1) * (nf + 2) * p) * (tr2 - tr1**2 / nf)
        tau = nf**4 / ((nf - 1) * (nf - 2) * (nf + 2) * (nf + 4))
        a3 = tau / p * (tr3 - 3 * tr2 * tr1 / nf + 2 * tr1**3 / nf**2)
        gamma = (
            nf**5
            * (nf**2 + nf + 2)
            / (
                (nf + 1)
                * (nf + 2)
                * (nf + 4)
                * (nf + 6)
                * (nf - 1)
                * (nf - 2)
                * (nf - 3)
            )
        )
        a4 = (
            gamma
            / p
            * (
                tr4
                - 4 * tr3 * tr1 / nf
                - (2 * nf**2 + 3 * nf - 6) / (nf * (nf**2 + nf + 2)) * tr2**2
                + 2 * (5 * nf + 6) * tr2 * tr1**2 / (nf * (nf**2 + nf + 2))
                - (5 * nf + 6) * tr1**4 / (nf**4 + nf**3 + 2 * nf**2)
            )
        )
        ratio = p / nf
        if variant == 1:
            return nf / (ratio * math.sqrt(8)) * (a4 - 4 * a3 + 6 * a2 - 4 * a1 + 1)
        return nf / math.sqrt(8 * (ratio**2 + 12 * ratio + 8)) * (a4 - 2 * a2 + 1)

    def test_both_variants_match_whitened_literal_formulas(self) -> None:
        popcov = np.array([[1.5, 0.2, -0.1], [0.2, 0.8, 0.15], [-0.1, 0.15, 1.2]])
        for variant in (1, 2):
            with self.subTest(variant=variant):
                actual = fisher_1samp(self.x, popcov=popcov, variant=variant)
                expected = self._literal(popcov, variant)
                np.testing.assert_allclose(actual.statistic, expected, rtol=2e-13)
                np.testing.assert_allclose(actual.pvalue, stats.norm.sf(expected))

    def test_null_affine_units_and_feature_coordinates_do_not_matter(self) -> None:
        popcov = np.array([[1.5, 0.2, -0.1], [0.2, 0.8, 0.15], [-0.1, 0.15, 1.2]])
        transform = np.array([[1.7, -0.2, 0.3], [0.4, 0.9, -0.1], [0.2, 0.5, 1.4]])
        baseline = fisher_1samp(self.x, popcov=popcov, variant=2)
        transformed = fisher_1samp(
            self.x @ transform + np.array([1e3, -2e3, 3e3]),
            popcov=transform.T @ popcov @ transform,
            variant=2,
        )
        scaled = fisher_1samp(
            self.x * 1e100 + np.array([2e100, -3e100, 0.5e100]),
            popcov=popcov * 1e200,
            variant=2,
        )
        np.testing.assert_allclose(
            transformed.statistic, baseline.statistic, rtol=2e-11
        )
        np.testing.assert_allclose(scaled.statistic, baseline.statistic, rtol=2e-13)

    def test_undefined_inputs_fail_before_calculation(self) -> None:
        with self.assertRaises(ValueError):
            fisher_1samp(self.x[:4])
        with self.assertRaises(ValueError):
            fisher_1samp(self.x, popcov=np.ones((3, 3)))
        with self.assertRaises(ValueError):
            fisher_1samp(self.x, variant=3)
        with self.assertRaises(TypeError):
            fisher_1samp(self.x, variant=True)

    def test_extreme_finite_alternative_reaches_upper_tail(self) -> None:
        for variant in (1, 2):
            with self.subTest(variant=variant):
                result = fisher_1samp(self.x * 1.0e100, variant=variant)
                self.assertEqual(result.statistic, math.inf)
                self.assertEqual(result.pvalue, 0.0)

        # The finite-sample unbiased fourth-moment estimator is not constrained
        # to be positive.  Preserve its sign rather than assuming that every
        # overflow must favor the alternative.
        small = np.random.default_rng(12).normal(size=(5, 2)) * 1.0e100
        signed = fisher_1samp(small, variant=2)
        self.assertEqual(signed.statistic, -math.inf)
        self.assertEqual(signed.pvalue, 1.0)


class TestWuLi(CovarianceFixtures):
    def test_one_sample_is_seeded_two_sided_and_scale_stable(self) -> None:
        baseline = wl_1samp(self.x, n_projections=17, rng=901)
        replay = wl_1samp(self.x, n_projections=17, rng=901)
        shifted = wl_1samp(
            self.x * 1e100 + np.array([2e100, -3e100, 0.5e100]),
            popcov=np.eye(3) * 1e200,
            n_projections=17,
            rng=901,
        )
        expected_p = 1.0 - special.erf(baseline.statistic / math.sqrt(2)) ** 17
        np.testing.assert_allclose(replay.statistic, baseline.statistic)
        np.testing.assert_allclose(shifted.statistic, baseline.statistic, rtol=2e-14)
        np.testing.assert_allclose(baseline.pvalue, expected_p)

        low_variance = wl_1samp(self.x * 1e-4, n_projections=17, rng=901)
        high_variance = wl_1samp(self.x * 1e4, n_projections=17, rng=901)
        self.assertLess(low_variance.pvalue, 0.01)
        self.assertLess(high_variance.pvalue, 1e-10)

    def test_two_sided_max_tail_does_not_round_to_zero_prematurely(self) -> None:
        result = wl_1samp(self.x * 3.0, n_projections=17, rng=901)
        marginal_tail = 2.0 * float(special.ndtr(-result.statistic))
        expected = -math.expm1(17 * math.log1p(-marginal_tail))

        self.assertGreater(result.pvalue, 0.0)
        np.testing.assert_allclose(result.pvalue, expected, rtol=2e-15)

    def test_two_sample_group_exchange_and_common_units(self) -> None:
        actual = wl_2samp(self.x, self.y, n_projections=19, rng=33)
        swapped = wl_2samp(self.y, self.x, n_projections=19, rng=33)
        scaled = wl_2samp(
            1e100 * self.x + 2e100,
            1e100 * self.y - 3e100,
            n_projections=19,
            rng=33,
        )
        np.testing.assert_allclose(swapped.statistic, actual.statistic)
        np.testing.assert_allclose(swapped.pvalue, actual.pvalue)
        np.testing.assert_allclose(scaled.statistic, actual.statistic, rtol=2e-14)

    def test_two_sample_extreme_variance_ratio_stays_in_log_domain(self) -> None:
        first = self.x[:, :1]
        second_base = self.y[:, :1]
        second = second_base * 1.0e-200
        result = wl_2samp(first, second, n_projections=7, rng=11)
        swapped = wl_2samp(second, first, n_projections=7, rng=11)

        variance1 = float(np.var(first[:, 0], ddof=1))
        variance2_base = float(np.var(second_base[:, 0], ddof=1))
        log_ratio = (
            math.log(variance1) - math.log(variance2_base) - 2.0 * math.log(1.0e-200)
        )
        expected = abs(log_ratio) / math.sqrt(
            2.0 / (first.shape[0] - 1) + 2.0 / (second.shape[0] - 1)
        )
        self.assertTrue(math.isfinite(result.statistic))
        np.testing.assert_allclose(result.statistic, expected, rtol=2e-15)
        np.testing.assert_allclose(swapped.statistic, result.statistic, rtol=2e-15)
        self.assertEqual(result.pvalue, 0.0)

    def test_randomized_calls_do_not_touch_legacy_global_rng(self) -> None:
        np.random.seed(109)
        expected = np.random.random(4)
        np.random.seed(109)
        wl_1samp(self.x, rng=5)
        actual = np.random.random(4)
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_controls_and_degenerate_projections_fail(self) -> None:
        with self.assertRaises(ValueError):
            wl_1samp(self.x, n_projections=0)
        with self.assertRaises(TypeError):
            wl_2samp(self.x, self.y, rng=object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            wl_2samp(np.ones((4, 3)), np.ones((5, 3)), rng=1)


class TestLiChen(CovarianceFixtures):
    def test_unbiased_statistic_matches_literal_equations_and_correct_sd(self) -> None:
        first = self.x[:5]
        second = self.y[:6]
        a1 = _literal_lc_a(first)
        a2 = _literal_lc_a(second)
        cross = _literal_lc_c(first, second)
        raw = a1 + a2 - 2 * cross
        sd_hat = 2 * a1 / second.shape[0] + 2 * a2 / first.shape[0]
        expected = raw / sd_hat
        actual = lc_2samp(first, second)
        np.testing.assert_allclose(actual.statistic, expected, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(actual.pvalue, stats.norm.sf(expected))
        self.assertGreater(abs(expected - raw / math.sqrt(sd_hat)), 0.05)

    def test_group_exchange_location_and_common_scale_invariance(self) -> None:
        actual = lc_2samp(self.x, self.y)
        swapped = lc_2samp(self.y, self.x)
        transformed = lc_2samp(
            1e70 * self.x + np.array([2e70, -4e70, 1e70]),
            1e70 * self.y + np.array([-3e70, 2e70, 5e70]),
        )
        np.testing.assert_allclose(swapped.statistic, actual.statistic, rtol=2e-13)
        np.testing.assert_allclose(transformed.statistic, actual.statistic, rtol=3e-13)

    def test_boundaries_fail_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            lc_2samp(self.x[:3], self.y)
        with self.assertRaises(ValueError):
            lc_2samp(np.ones((5, 3)), np.ones((6, 3)))

    def test_uncalibrated_leading_term_shortcut_is_not_exposed(self) -> None:
        self.assertNotIn("unbiased", inspect.signature(lc_2samp).parameters)
        with self.assertRaises(TypeError):
            lc_2samp(self.x, self.y, unbiased=False)  # type: ignore[call-arg]


class TestCaiLiuXia(CovarianceFixtures):
    def test_matches_literal_entrywise_formula_and_extreme_value_tail(self) -> None:
        first = _center(self.x)
        second = _center(self.y)
        n1 = first.shape[0]
        n2 = second.shape[0]
        covariance1 = first.T @ first / n1
        covariance2 = second.T @ second / n2
        theta1 = np.empty((3, 3))
        theta2 = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                theta1[i, j] = np.mean(
                    (first[:, i] * first[:, j] - covariance1[i, j]) ** 2
                )
                theta2[i, j] = np.mean(
                    (second[:, i] * second[:, j] - covariance2[i, j]) ** 2
                )
        expected = float(
            np.max((covariance1 - covariance2) ** 2 / (theta1 / n1 + theta2 / n2))
        )
        centered = expected - 4 * math.log(3) + math.log(math.log(3))
        expected_p = 1 - math.exp(-math.exp(-centered / 2) / math.sqrt(8 * math.pi))
        actual = clx_2samp(self.x, self.y)
        np.testing.assert_allclose(actual.statistic, expected, rtol=2e-13)
        np.testing.assert_allclose(actual.pvalue, expected_p)

    def test_group_exchange_translation_and_scale_invariance(self) -> None:
        actual = clx_2samp(self.x, self.y)
        swapped = clx_2samp(self.y, self.x)
        transformed = clx_2samp(1e90 * self.x + 2e90, 1e90 * self.y - 3e90)
        np.testing.assert_allclose(swapped.statistic, actual.statistic)
        np.testing.assert_allclose(transformed.statistic, actual.statistic, rtol=2e-14)

    def test_invalid_dimension_and_zero_entrywise_variance_fail(self) -> None:
        with self.assertRaises(ValueError):
            clx_2samp(self.x[:, :1], self.y[:, :1])
        with self.assertRaises(ValueError):
            clx_2samp(np.ones((5, 3)), np.ones((6, 3)))


class TestLeeYouLin(CovarianceFixtures):
    def test_components_match_equations_12_to_15(self) -> None:
        a0 = 1.3
        b0 = 0.7
        gamma = 0.4
        expected = _literal_lyl(self.x, self.y, a0=a0, b0=b0, gamma=gamma)
        actual = lyl_2samp(self.x, self.y, a0=a0, b0=b0, gamma=gamma)
        self.assertIsInstance(actual, BayesFactorTestResult)
        components = np.asarray(actual.component_log_bayes_factors)
        np.testing.assert_allclose(components, expected, rtol=2e-13)
        np.testing.assert_allclose(actual.statistic, np.max(expected))
        self.assertFalse(hasattr(actual, "pvalue"))
        self.assertEqual(actual.statistic_name, "maximum log BF")

    def test_published_defaults_match_equations_12_to_15(self) -> None:
        gamma = max(self.x.shape[0] + self.y.shape[0], self.x.shape[1]) ** -2.01
        expected = _literal_lyl(self.x, self.y, a0=0.01, b0=0.01, gamma=gamma)
        actual = lyl_2samp(self.x, self.y)
        np.testing.assert_allclose(
            np.asarray(actual.component_log_bayes_factors), expected, rtol=2e-13
        )
        diagnostics = dict(actual.diagnostics)
        np.testing.assert_allclose(diagnostics["gamma"], gamma)
        self.assertEqual(diagnostics["gamma source"], "derived from alpha")
        self.assertEqual(diagnostics["mean assumption"], "known zero")

    def test_group_exchange_and_feature_permutation(self) -> None:
        actual = lyl_2samp(self.x, self.y)
        swapped = lyl_2samp(self.y, self.x)
        order = np.array([2, 0, 1])
        permuted = lyl_2samp(self.x[:, order], self.y[:, order])
        np.testing.assert_allclose(swapped.statistic, actual.statistic, rtol=2e-13)
        np.testing.assert_allclose(permuted.statistic, actual.statistic, rtol=2e-13)

    def test_b0_has_units_and_scales_with_squared_data_units(self) -> None:
        actual = lyl_2samp(self.x, self.y, b0=0.01)
        for scale in (1e-150, 1e150):
            with self.subTest(scale=scale):
                scaled = lyl_2samp(
                    scale * self.x,
                    scale * self.y,
                    b0=0.01 * scale * scale,
                )
                np.testing.assert_allclose(
                    scaled.statistic, actual.statistic, rtol=5e-13
                )

        for scale in (1e-300, 1e300):
            with self.subTest(finite_scale=scale):
                result = lyl_2samp(scale * self.x, scale * self.y)
                self.assertTrue(math.isfinite(result.statistic))

    def test_heterogeneous_feature_scales_remain_evaluable(self) -> None:
        first = self.x.copy()
        second = self.y.copy()
        feature_scale = np.array([1e-150, 1.0, 1e150])
        first *= feature_scale
        second *= feature_scale
        actual = lyl_2samp(first, second)
        order = np.array([2, 0, 1])
        permuted = lyl_2samp(first[:, order], second[:, order])
        self.assertTrue(math.isfinite(actual.statistic))
        np.testing.assert_allclose(permuted.statistic, actual.statistic, rtol=2e-13)

    def test_invalid_hyperparameters_and_predictors_fail(self) -> None:
        for name, kwargs in (
            ("a0", {"a0": 0}),
            ("b0", {"b0": 0}),
            ("alpha", {"alpha": 0}),
            ("gamma", {"gamma": 0}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                lyl_2samp(self.x, self.y, **kwargs)
        with self.assertRaises(ValueError):
            lyl_2samp(self.x[:, :1], self.y[:, :1])
        with self.assertRaises(ValueError):
            lyl_2samp(np.zeros((5, 3)), np.zeros((6, 3)))


class TestSchott(CovarianceFixtures):
    def test_2001_matches_literal_wald_trace_formula(self) -> None:
        groups = (self.x, self.y, self.z)
        degrees = np.array([group.shape[0] - 1 for group in groups])
        covariances = [np.cov(group, rowvar=False, ddof=1) for group in groups]
        total = int(np.sum(degrees))
        pooled = sum(
            degree / total * sample_covariance
            for degree, sample_covariance in zip(degrees, covariances, strict=True)
        )
        inverse = np.linalg.inv(pooled)
        term1 = sum(
            degree
            / total
            * np.trace(sample_covariance @ inverse @ sample_covariance @ inverse)
            for degree, sample_covariance in zip(degrees, covariances, strict=True)
        )
        term2 = 0.0
        for i, first in enumerate(covariances):
            for j, second in enumerate(covariances):
                term2 += (
                    degrees[i]
                    / total
                    * degrees[j]
                    / total
                    * np.trace(first @ inverse @ second @ inverse)
                )
        expected = total * (term1 - term2) / 2
        actual = schott_2001_ksamp(*groups)
        np.testing.assert_allclose(actual.statistic, expected, rtol=2e-13)
        self.assertEqual(actual.df, 12.0)
        np.testing.assert_allclose(actual.pvalue, stats.chi2.sf(expected, 12))

    def test_2007_matches_literal_formula(self) -> None:
        groups = (self.x, self.y, self.z)
        degrees = np.array([group.shape[0] - 1 for group in groups], dtype=float)
        covariances = [np.cov(group, rowvar=False, ddof=1) for group in groups]
        total = float(np.sum(degrees))
        pooled = sum(
            degree / total * sample_covariance
            for degree, sample_covariance in zip(degrees, covariances, strict=True)
        )
        tnm = 0.0
        inner1 = 0.0
        for i in range(len(groups) - 1):
            ni = degrees[i]
            ei = (ni + 2) * (ni - 1)
            for j in range(i + 1, len(groups)):
                nj = degrees[j]
                ej = (nj + 2) * (nj - 1)
                tnm += (
                    (1 - (ni - 2) / ei) * np.trace(covariances[i] @ covariances[i])
                    + (1 - (nj - 2) / ej) * np.trace(covariances[j] @ covariances[j])
                    - 2 * np.trace(covariances[i] @ covariances[j])
                    - ni / ei * np.trace(covariances[i]) ** 2
                    - nj / ej * np.trace(covariances[j]) ** 2
                )
                inner1 += ((ni + nj) / (ni * nj)) ** 2
        a_estimate = (
            total**2
            / ((total + 2) * (total - 1))
            * (np.trace(pooled @ pooled) - np.trace(pooled) ** 2 / total)
        )
        inner2 = 2 * float(np.sum(1 / degrees**2))
        theta = 2 * math.sqrt(inner1 + inner2) * a_estimate
        expected = tnm / theta
        actual = schott_2007_ksamp(*groups)
        np.testing.assert_allclose(actual.statistic, expected, rtol=2e-13)
        np.testing.assert_allclose(actual.pvalue, stats.norm.sf(expected))

    def test_group_order_location_and_common_scale_invariance(self) -> None:
        for procedure in (schott_2001_ksamp, schott_2007_ksamp):
            with self.subTest(procedure=procedure.__name__):
                baseline = procedure(self.x, self.y, self.z)
                reordered = procedure(self.z, self.x, self.y)
                transformed = procedure(
                    1e80 * self.x + 2e80,
                    1e80 * self.y - 3e80,
                    1e80 * self.z + 5e80,
                )
                np.testing.assert_allclose(reordered.statistic, baseline.statistic)
                np.testing.assert_allclose(
                    transformed.statistic, baseline.statistic, rtol=2e-13
                )

    def test_2001_rank_check_is_relative_to_covariance_scale(self) -> None:
        baseline = schott_2001_ksamp(self.x, self.y, self.z)
        translated = schott_2001_ksamp(
            self.x + 1e8,
            self.y + 1e8,
            self.z + 1e8,
        )
        np.testing.assert_allclose(translated.statistic, baseline.statistic, rtol=5e-8)

    def test_singular_and_small_sample_boundaries_fail(self) -> None:
        singular = np.column_stack((np.arange(5.0), np.arange(5.0)))
        with self.assertRaises(ValueError):
            schott_2001_ksamp(singular, singular + 1)
        with self.assertRaises(ValueError):
            schott_2007_ksamp(self.x[:2], self.y)


class TestLargeLocationNumerics(CovarianceFixtures):
    def test_centered_frequentist_methods_preserve_representable_variation(
        self,
    ) -> None:
        first = 10.0 * self.x
        second = 10.0 * self.y
        third = 10.0 * self.z
        shift1 = np.array([1e14, -2e14, 3e14])
        shift2 = np.array([-3e14, 1e14, 2e14])
        shift3 = np.array([2e14, 3e14, -1e14])

        cases = (
            (
                fisher_1samp(first, popcov=100.0 * np.eye(3), variant=2),
                fisher_1samp(first + shift1, popcov=100.0 * np.eye(3), variant=2),
            ),
            (
                wl_1samp(first, popcov=100.0 * np.eye(3), rng=71),
                wl_1samp(first + shift1, popcov=100.0 * np.eye(3), rng=71),
            ),
            (lc_2samp(first, second), lc_2samp(first + shift1, second + shift2)),
            (clx_2samp(first, second), clx_2samp(first + shift1, second + shift2)),
            (
                wl_2samp(first, second, rng=71),
                wl_2samp(first + shift1, second + shift2, rng=71),
            ),
            (
                schott_2001_ksamp(first, second, third),
                schott_2001_ksamp(first + shift1, second + shift2, third + shift3),
            ),
            (
                schott_2007_ksamp(first, second, third),
                schott_2007_ksamp(first + shift1, second + shift2, third + shift3),
            ),
        )
        for baseline, translated in cases:
            with self.subTest(method=baseline.method):
                np.testing.assert_allclose(
                    translated.statistic, baseline.statistic, rtol=2e-13, atol=2e-13
                )


if __name__ == "__main__":
    unittest.main()

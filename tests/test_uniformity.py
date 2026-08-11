"""Independent checks for rectangular-uniformity procedures."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import integrate, stats
from scipy.spatial import distance

from pysht._results import ResamplingTestResult
from pysht.uniformity import ym_interpoint, ym_quantile


class YangModarresInterpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [0.08, 0.21, 0.74, 0.33],
                [0.17, 0.63, 0.41, 0.92],
                [0.39, 0.12, 0.88, 0.55],
                [0.52, 0.79, 0.19, 0.67],
                [0.68, 0.34, 0.57, 0.14],
                [0.81, 0.91, 0.28, 0.48],
                [0.93, 0.46, 0.65, 0.83],
            ]
        )

    def _literal_components(self) -> tuple[float, float]:
        n, dimension = self.x.shape
        squared_distances = distance.pdist(self.x, metric="sqeuclidean")
        expected = dimension / 6.0
        mean_distance = float(np.mean(squared_distances))
        second = float(np.mean((squared_distances - expected) ** 2))
        variance_mean = dimension * (2 * n + 3) / (90 * n * (n - 1))
        variance_second = (
            49 * dimension**2 / 16_200
            + 101 * dimension / 37_800
            + 2 * (n - 2) * (dimension**2 / 16_200 + 29 * dimension / 37_800)
        ) * (2 / (n * (n - 1)))
        q1 = (mean_distance - expected) ** 2 / variance_mean
        q2 = (second - 7 * dimension / 180) ** 2 / variance_second
        return q1, q2

    def test_statistics_match_paper_and_q3_uses_corrected_covariance(self) -> None:
        q1, q2 = self._literal_components()
        results = {
            "q1": ym_interpoint(self.x, statistic="q1", calibration="asymptotic"),
            "q2": ym_interpoint(self.x, statistic="q2", calibration="asymptotic"),
            "q3": ym_interpoint(self.x, statistic="q3", calibration="asymptotic"),
        }
        self.assertAlmostEqual(results["q1"].statistic, q1, places=14)
        self.assertAlmostEqual(results["q2"].statistic, q2, places=14)
        self.assertAlmostEqual(results["q3"].statistic, q1 + q2, places=14)
        self.assertAlmostEqual(results["q1"].pvalue, float(stats.chi2.sf(q1, 1)))
        self.assertAlmostEqual(results["q2"].pvalue, float(stats.chi2.sf(q2, 1)))

        n, dimension = self.x.shape
        variance_mean = dimension * (2 * n + 3) / (90 * n * (n - 1))
        variance_second = (
            49 * dimension**2 / 16_200
            + 101 * dimension / 37_800
            + 2 * (n - 2) * (dimension**2 / 16_200 + 29 * dimension / 37_800)
        ) * (2 / (n * (n - 1)))
        covariance = dimension * (4 * n + 3) / 945 * (2 / (n * (n - 1)))
        correlation = covariance / math.sqrt(variance_mean * variance_second)
        larger_weight = 1.0 + correlation
        smaller_weight = 1.0 - correlation
        threshold = math.sqrt((q1 + q2) / larger_weight)

        # Independent convolution oracle for
        # (1+rho) U^2 + (1-rho) V^2, U,V independently standard normal.
        integral = integrate.quad(
            lambda value: (
                2.0
                * stats.norm.pdf(value)
                * stats.chi2.sf(
                    (q1 + q2 - larger_weight * value * value) / smaller_weight,
                    1,
                )
            ),
            0.0,
            threshold,
            epsabs=1e-14,
            epsrel=1e-13,
        )[0]
        expected_q3_pvalue = 2.0 * stats.norm.sf(threshold) + integral
        np.testing.assert_allclose(results["q3"].pvalue, expected_q3_pvalue, rtol=2e-13)
        self.assertNotAlmostEqual(
            results["q3"].pvalue, float(stats.chi2.sf(q1 + q2, 2))
        )
        self.assertIsNone(results["q3"].df)
        self.assertEqual(
            results["q3"].diagnostics,
            (("signed-component correlation", correlation),),
        )

    def test_special_q2_variances_for_dimensions_two_and_three(self) -> None:
        for dimension in (2, 3):
            values = self.x[:, :dimension]
            n = values.shape[0]
            distances = distance.pdist(values, metric="sqeuclidean")
            second = float(np.mean((distances - dimension / 6.0) ** 2))
            if dimension == 2:
                variance = ((989 + 202 * (n - 2)) / 56_700) * (2 / (n * (n - 1)))
            else:
                variance = ((37 + 6 * (n - 2)) / 1_050) * (2 / (n * (n - 1)))
            expected = (second - 7 * dimension / 180.0) ** 2 / variance
            actual = ym_interpoint(values, statistic="q2", calibration="asymptotic")
            with self.subTest(dimension=dimension):
                self.assertAlmostEqual(actual.statistic, expected, places=14)

    def test_coordinate_order_and_rectangular_affine_maps_are_invariant(self) -> None:
        baseline = ym_interpoint(self.x, statistic="q3", calibration="asymptotic")
        reordered = ym_interpoint(
            self.x[:, [3, 0, 2, 1]], statistic="q3", calibration="asymptotic"
        )
        np.testing.assert_allclose(reordered.statistic, baseline.statistic, rtol=1e-14)
        np.testing.assert_allclose(reordered.pvalue, baseline.pvalue, rtol=1e-14)

        lower = np.array([-4.0, 2.0, 10.0, -0.25])
        upper = np.array([7.0, 2.5, 31.0, 3.75])
        mapped = lower + self.x * (upper - lower)
        transformed = ym_interpoint(
            mapped,
            statistic="q3",
            lower=lower,
            upper=upper,
            calibration="asymptotic",
        )
        np.testing.assert_allclose(
            transformed.statistic, baseline.statistic, rtol=2e-14
        )
        np.testing.assert_allclose(transformed.pvalue, baseline.pvalue, rtol=2e-14)

    def test_overflowing_support_width_is_standardized_safely(self) -> None:
        values = self.x[:, :2]
        scale = 1.0e308
        mapped = scale * (2.0 * values - 1.0)
        baseline = ym_interpoint(values, statistic="q3", calibration="asymptotic")
        extreme = ym_interpoint(
            mapped,
            statistic="q3",
            lower=np.array([-scale, -scale]),
            upper=np.array([scale, scale]),
            calibration="asymptotic",
        )
        self.assertTrue(
            math.isclose(extreme.statistic, baseline.statistic, rel_tol=2e-13)
        )
        self.assertTrue(math.isclose(extreme.pvalue, baseline.pvalue, rel_tol=2e-13))

    def test_interpoint_allows_declared_boundary_observations(self) -> None:
        values = np.array([[0.0, 0.0], [1.0, 1.0], [0.25, 0.8]])
        result = ym_interpoint(values, statistic="Q1", calibration="asymptotic")
        self.assertTrue(math.isfinite(result.statistic))
        self.assertTrue(0.0 <= result.pvalue <= 1.0)

    def test_default_monte_carlo_calibration_is_seeded_and_corrected(self) -> None:
        resamples = 197
        seed = 616
        result = ym_interpoint(
            self.x,
            statistic="q3",
            n_resamples=resamples,
            rng=seed,
        )
        repeated = ym_interpoint(
            self.x,
            statistic="q3",
            n_resamples=resamples,
            rng=seed,
        )
        self.assertIsInstance(result, ResamplingTestResult)
        assert isinstance(result, ResamplingTestResult)
        self.assertEqual(result, repeated)

        generator = np.random.default_rng(seed)
        simulated = generator.random((resamples, *self.x.shape))
        first, second = np.triu_indices(self.x.shape[0], k=1)
        squared_distances = np.sum(
            (simulated[:, first, :] - simulated[:, second, :]) ** 2,
            axis=2,
        )
        n, dimension = self.x.shape
        expected_distance = dimension / 6.0
        means = np.mean(squared_distances, axis=1)
        second_moments = np.mean((squared_distances - expected_distance) ** 2, axis=1)
        variance_mean = dimension * (2 * n + 3) / (90 * n * (n - 1))
        variance_second = (
            49 * dimension**2 / 16_200
            + 101 * dimension / 37_800
            + 2 * (n - 2) * (dimension**2 / 16_200 + 29 * dimension / 37_800)
        ) * (2 / (n * (n - 1)))
        q1 = (means - expected_distance) ** 2 / variance_mean
        q2 = (second_moments - 7 * dimension / 180.0) ** 2 / variance_second
        exceedances = int(np.count_nonzero(q1 + q2 >= result.statistic))
        self.assertEqual(result.exceedances, exceedances)
        self.assertEqual(result.pvalue, (exceedances + 1.0) / (resamples + 1.0))
        self.assertIsNotNone(result.tail_probability_interval)

    def test_monte_carlo_does_not_touch_numpy_global_rng(self) -> None:
        np.random.seed(103)
        expected = np.random.random(3)
        np.random.seed(103)
        ym_interpoint(
            self.x,
            statistic="q1",
            n_resamples=29,
            rng=7,
        )
        actual = np.random.random(3)
        np.testing.assert_array_equal(actual, expected)


class YangModarresQuantileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [0.08, 0.21, 0.74],
                [0.17, 0.63, 0.41],
                [0.39, 0.12, 0.88],
                [0.52, 0.79, 0.19],
                [0.68, 0.34, 0.57],
                [0.81, 0.91, 0.28],
                [0.93, 0.46, 0.65],
            ]
        )

    def test_matches_normal_quantile_formula(self) -> None:
        quantiles = stats.norm.ppf(self.x)
        expected = self.x.shape[0] * float(
            np.dot(np.mean(quantiles, axis=0), np.mean(quantiles, axis=0))
        )
        result = ym_quantile(self.x)
        self.assertAlmostEqual(result.statistic, expected, places=14)
        self.assertAlmostEqual(
            result.pvalue, float(stats.chi2.sf(expected, self.x.shape[1]))
        )
        self.assertEqual(result.df, float(self.x.shape[1]))

    def test_affine_domain_and_column_order_are_invariant(self) -> None:
        baseline = ym_quantile(self.x)
        lower = np.array([-4.0, 2.0, 10.0])
        upper = np.array([7.0, 2.5, 31.0])
        mapped = lower + self.x * (upper - lower)
        transformed = ym_quantile(mapped, lower=lower, upper=upper)
        reordered = ym_quantile(self.x[:, [2, 0, 1]])
        np.testing.assert_allclose(
            transformed.statistic, baseline.statistic, rtol=1e-14
        )
        np.testing.assert_allclose(transformed.pvalue, baseline.pvalue, rtol=1e-14)
        np.testing.assert_allclose(reordered.statistic, baseline.statistic, rtol=1e-14)
        np.testing.assert_allclose(reordered.pvalue, baseline.pvalue, rtol=1e-14)

    def test_quantile_rejects_every_boundary(self) -> None:
        for boundary in (
            np.array([[0.0, 0.3], [0.4, 0.6]]),
            np.array([[0.2, 1.0], [0.4, 0.6]]),
        ):
            with (
                self.subTest(boundary=boundary),
                self.assertRaisesRegex(ValueError, "strictly inside"),
            ):
                ym_quantile(boundary)


class RectangularUniformityValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array([[0.1, 0.2], [0.4, 0.7], [0.8, 0.3]])

    def test_malformed_data_are_rejected(self) -> None:
        functions = (ym_interpoint, ym_quantile)
        for function in functions:
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function([0.1, 0.2, 0.3])
                with self.assertRaises(ValueError):
                    function([[0.1], [0.2], [0.3]])
                with self.assertRaises(ValueError):
                    function([[0.1, 0.2]])
                with self.assertRaises(TypeError):
                    function(np.array([[True, False], [False, True]]))
                with self.assertRaises(ValueError):
                    function([[0.1, np.nan], [0.3, 0.7]])

    def test_bounds_are_strictly_validated(self) -> None:
        functions = (ym_interpoint, ym_quantile)
        invalid_controls = (
            {"lower": [0.0]},
            {"upper": [1.0]},
            {"lower": [0.0, 0.5], "upper": [1.0, 0.5]},
            {"lower": [0.0, -math.inf]},
            {"upper": [True, False]},
        )
        for function in functions:
            for controls in invalid_controls:
                with (
                    self.subTest(function=function.__name__, controls=controls),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    function(self.x, **controls)
            with self.assertRaisesRegex(ValueError, "declared bounds"):
                function(self.x + 1.0)

    def test_interpoint_statistic_control_rejects_legacy_initials(self) -> None:
        with self.assertRaisesRegex(ValueError, "statistic"):
            ym_interpoint(self.x, statistic="1")
        with self.assertRaises(TypeError):
            ym_interpoint(self.x, statistic=object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

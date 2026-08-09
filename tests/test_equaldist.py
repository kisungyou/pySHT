"""Tests for distribution-equality procedures."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from pysht.equaldist import biswas_ghosh_2samp


class TestBiswasGhoshTwoSample(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [-1.0, 0.5],
                [0.2, -0.3],
                [1.1, 0.8],
                [-0.4, 1.5],
                [0.7, -1.2],
            ]
        )
        self.y = np.array(
            [
                [0.3, 0.4],
                [1.8, -0.2],
                [-0.7, 0.9],
                [0.6, 1.6],
                [-1.1, -0.8],
                [1.2, 0.1],
            ]
        )

    def test_identical_constant_samples_have_zero_statistic(self) -> None:
        x = np.zeros((5, 3))
        y = np.zeros((7, 3))
        result = biswas_ghosh_2samp(x, y, n_resamples=39, rng=7)

        self.assertEqual(result.statistic, 0.0)
        self.assertEqual(result.pvalue, 1.0)
        self.assertEqual(result.exceedances, 39)
        self.assertEqual(result.alternative, "the two distributions are not equal")

    def test_clear_distributional_alternative_has_small_pvalue(self) -> None:
        x = np.zeros((8, 2))
        y = np.full((8, 2), 10.0)
        result = biswas_ghosh_2samp(x, y, n_resamples=499, rng=12)

        self.assertGreater(result.statistic, 0.0)
        self.assertLess(result.pvalue, 0.02)

    def test_integer_seed_is_deterministic(self) -> None:
        first = biswas_ghosh_2samp(self.x, self.y, n_resamples=79, rng=2026)
        second = biswas_ghosh_2samp(self.x, self.y, n_resamples=79, rng=2026)
        self.assertEqual(first, second)

    def test_row_order_does_not_change_statistic_or_calibration(self) -> None:
        baseline = biswas_ghosh_2samp(self.x, self.y, n_resamples=99, rng=101)
        reordered = biswas_ghosh_2samp(
            self.x[[3, 0, 4, 1, 2]],
            self.y[[5, 2, 0, 4, 1, 3]],
            n_resamples=99,
            rng=101,
        )

        self.assertEqual(reordered.statistic, baseline.statistic)
        self.assertEqual(reordered.exceedances, baseline.exceedances)
        self.assertEqual(reordered.pvalue, baseline.pvalue)

    def test_swapping_groups_does_not_change_result(self) -> None:
        xy = biswas_ghosh_2samp(self.x, self.y, n_resamples=99, rng=42)
        yx = biswas_ghosh_2samp(self.y, self.x, n_resamples=99, rng=42)

        self.assertEqual(yx.statistic, xy.statistic)
        self.assertEqual(yx.exceedances, xy.exceedances)
        self.assertEqual(yx.pvalue, xy.pvalue)

    def test_plus_one_pvalue_is_nonzero_and_in_range(self) -> None:
        x = np.zeros((8, 1))
        y = np.full((8, 1), 100.0)
        result = biswas_ghosh_2samp(x, y, n_resamples=19, rng=8)

        self.assertGreater(result.pvalue, 0.0)
        self.assertGreaterEqual(result.pvalue, 1.0 / 20.0)
        self.assertLessEqual(result.pvalue, 1.0)

    def test_common_rescaling_preserves_normalized_inference(self) -> None:
        x = np.zeros((8, 1))
        y = np.ones((8, 1))
        baseline = biswas_ghosh_2samp(x, y, n_resamples=199, rng=5)

        for scale in (1e-200, 1e200):
            with self.subTest(scale=scale):
                scaled = biswas_ghosh_2samp(
                    scale * x,
                    scale * y,
                    n_resamples=199,
                    rng=5,
                )
                self.assertEqual(
                    scaled.normalized_statistic, baseline.normalized_statistic
                )
                self.assertEqual(scaled.exceedances, baseline.exceedances)
                self.assertEqual(scaled.pvalue, baseline.pvalue)

        moderate = biswas_ghosh_2samp(3.0 * x, 3.0 * y, n_resamples=199, rng=5)
        np.testing.assert_allclose(moderate.statistic, 9.0 * baseline.statistic)

    def test_rotation_preserves_theoretical_permutation_ties(self) -> None:
        x = np.array([[0.0, 0.0], [1.0, 0.0]])
        y = np.array([[1.0, 1.0], [0.0, 1.0]])
        angle = 0.37
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

        baseline = biswas_ghosh_2samp(x, y, n_resamples=999, rng=11)
        rotated = biswas_ghosh_2samp(
            x @ rotation,
            y @ rotation,
            n_resamples=999,
            rng=11,
        )

        self.assertEqual(baseline.pvalue, 1.0)
        self.assertEqual(rotated.pvalue, 1.0)
        self.assertEqual(baseline.exceedances, 6)
        self.assertEqual(rotated.exceedances, 6)
        self.assertEqual(baseline.calibration, "exact permutation")

    def test_formula_and_exact_permutation_oracles(self) -> None:
        result = biswas_ghosh_2samp(
            np.array([0.0, 2.0]),
            np.array([1.0, 5.0, 8.0]),
            calibration="exact",
            n_resamples=10,
        )

        np.testing.assert_allclose(result.statistic, 40.0 / 9.0, rtol=1e-14)
        np.testing.assert_allclose(result.normalized_statistic, 5.0 / 72.0, rtol=1e-14)
        self.assertEqual(result.distance_scale, 8.0)
        self.assertEqual(result.pvalue, 0.8)
        self.assertEqual(result.exceedances, 8)
        self.assertEqual(result.n_resamples, 10)
        self.assertIsNone(result.monte_carlo_standard_error)

    def test_extreme_coordinate_scales_do_not_change_calibration(self) -> None:
        baseline = biswas_ghosh_2samp(
            np.array([-1.0, -0.5]),
            np.array([0.5, 1.0]),
            n_resamples=6,
        )
        huge = biswas_ghosh_2samp(
            np.array([-1e308, -5e307]),
            np.array([5e307, 1e308]),
            n_resamples=6,
        )
        subnormal = float(np.nextafter(0.0, 1.0))
        tiny = biswas_ghosh_2samp(
            np.array([0.0, subnormal]),
            np.array([2.0 * subnormal, 3.0 * subnormal]),
            n_resamples=6,
        )
        tiny_baseline = biswas_ghosh_2samp(
            np.array([0.0, 1.0]),
            np.array([2.0, 3.0]),
            n_resamples=6,
        )

        np.testing.assert_allclose(
            huge.normalized_statistic, baseline.normalized_statistic
        )
        self.assertEqual(huge.pvalue, baseline.pvalue)
        np.testing.assert_allclose(
            tiny.normalized_statistic, tiny_baseline.normalized_statistic
        )
        self.assertEqual(tiny.pvalue, tiny_baseline.pvalue)

    def test_result_is_immutable_and_has_htest_style_display(self) -> None:
        result = biswas_ghosh_2samp(self.x, self.y, n_resamples=19, rng=2)
        with self.assertRaises(FrozenInstanceError):
            result.pvalue = 0.0  # type: ignore[misc]

        rendered = str(result)
        self.assertIn("Biswas-Ghosh two-sample test", rendered)
        self.assertIn("T_mn =", rendered)
        self.assertIn("p-value =", rendered)
        self.assertIn("alternative hypothesis:", rendered)
        self.assertIn("calibration:", rendered)
        self.assertIn("permutation", rendered)
        self.assertIn("numerical normalization:", rendered)

    def test_accepts_univariate_inputs_and_generator(self) -> None:
        generator = np.random.default_rng(19)
        result = biswas_ghosh_2samp(
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 2.0, 4.0, 8.0]),
            n_resamples=np.int64(11),
            rng=generator,
        )
        self.assertEqual(result.n_resamples, 11)
        self.assertTrue(0.0 < result.pvalue <= 1.0)

    def test_rejects_invalid_samples(self) -> None:
        invalid_calls = (
            lambda: biswas_ghosh_2samp([1.0], [1.0, 2.0], n_resamples=3),
            lambda: biswas_ghosh_2samp(np.ones((3, 2)), np.ones((3, 3)), n_resamples=3),
            lambda: biswas_ghosh_2samp([0.0, np.nan], [1.0, 2.0], n_resamples=3),
            lambda: biswas_ghosh_2samp(
                np.ones((2, 2, 2)), np.ones((2, 2)), n_resamples=3
            ),
            lambda: biswas_ghosh_2samp([True, False], [False, True], n_resamples=3),
            lambda: biswas_ghosh_2samp([1.0 + 1.0j, 2.0], [1.0, 2.0], n_resamples=3),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                call()

    def test_rejects_invalid_resampling_controls(self) -> None:
        for invalid in (0, -1):
            with self.subTest(n_resamples=invalid), self.assertRaises(ValueError):
                biswas_ghosh_2samp(self.x, self.y, n_resamples=invalid)
        for invalid in (True, 2.5):
            with self.subTest(n_resamples=invalid), self.assertRaises(TypeError):
                biswas_ghosh_2samp(
                    self.x,
                    self.y,
                    n_resamples=invalid,  # type: ignore[arg-type]
                )

        with self.assertRaises(TypeError):
            biswas_ghosh_2samp(
                self.x,
                self.y,
                n_resamples=3,
                rng=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(NotImplementedError):
            biswas_ghosh_2samp(self.x, self.y, n_resamples=3, n_jobs=2)

    def test_asymptotic_and_unknown_calibrations_are_not_silently_used(self) -> None:
        with self.assertRaises(NotImplementedError):
            biswas_ghosh_2samp(
                self.x,
                self.y,
                calibration="asymptotic",
                n_resamples=3,
            )
        with self.assertRaises(ValueError):
            biswas_ghosh_2samp(self.x, self.y, calibration="bootstrap", n_resamples=3)

    def test_monte_carlo_correction_and_uncertainty_are_explicit(self) -> None:
        result = biswas_ghosh_2samp(
            self.x,
            self.y,
            calibration="monte-carlo",
            n_resamples=29,
            rng=91,
        )
        expected_pvalue = (result.exceedances + 1.0) / 30.0
        expected_se = np.sqrt(29.0 * expected_pvalue * (1.0 - expected_pvalue)) / 30.0

        self.assertEqual(result.calibration, "Monte Carlo permutation")
        self.assertEqual(result.pvalue, expected_pvalue)
        np.testing.assert_allclose(result.monte_carlo_standard_error, expected_se)

    def test_exact_calibration_respects_computational_budget(self) -> None:
        with self.assertRaises(ValueError):
            biswas_ghosh_2samp(
                np.array([0.0, 2.0]),
                np.array([1.0, 5.0, 8.0]),
                calibration="exact",
                n_resamples=9,
            )


if __name__ == "__main__":
    unittest.main()

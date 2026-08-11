"""Independent checks for simplex-uniformity likelihood-ratio tests."""

from __future__ import annotations

import math
import unittest

import numpy as np
from scipy import optimize, special, stats

from pysht.simplex import uniformity


def _log_likelihood(alpha: np.ndarray, x: np.ndarray) -> float:
    return float(
        x.shape[0] * (special.gammaln(np.sum(alpha)) - np.sum(special.gammaln(alpha)))
        + np.sum((alpha - 1.0) * np.sum(np.log(x), axis=0))
    )


def _independent_general_lrt(x: np.ndarray) -> float:
    """Reference fit by solving scores in log-parameter coordinates."""

    mean_log = np.mean(np.log(x), axis=0)

    def score(log_alpha: np.ndarray) -> np.ndarray:
        alpha = np.exp(log_alpha)
        alpha_sum = float(np.sum(alpha))
        return np.asarray(
            special.digamma(alpha_sum) - special.digamma(alpha) + mean_log,
            dtype=np.float64,
        )

    solution = optimize.root(
        score,
        np.zeros(x.shape[1]),
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 2_000},
    )
    residual = float(np.max(np.abs(score(solution.x))))
    if not solution.success or residual > 1e-10:
        solution = optimize.root(
            score,
            np.zeros(x.shape[1]),
            method="lm",
            options={
                "ftol": 1e-13,
                "xtol": 1e-13,
                "gtol": 1e-13,
                "maxiter": 2_000,
            },
        )
        residual = float(np.max(np.abs(score(solution.x))))
        if not solution.success or residual > 1e-10:
            raise AssertionError(solution.message)
    fitted = np.exp(solution.x)
    null = np.ones(x.shape[1])
    return 2.0 * (_log_likelihood(fitted, x) - _log_likelihood(null, x))


class SimplexUniformityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [0.12, 0.33, 0.55],
                [0.41, 0.27, 0.32],
                [0.08, 0.71, 0.21],
                [0.62, 0.19, 0.19],
                [0.23, 0.49, 0.28],
                [0.36, 0.11, 0.53],
                [0.17, 0.24, 0.59],
                [0.48, 0.38, 0.14],
            ]
        )

    def test_symmetric_lrt_matches_independent_scalar_optimization(self) -> None:
        dimension = self.x.shape[1]

        def objective(log_alpha: float) -> float:
            alpha = math.exp(log_alpha)
            return -_log_likelihood(np.full(dimension, alpha), self.x)

        solution = optimize.minimize_scalar(
            objective,
            bounds=(-20.0, 20.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        self.assertTrue(solution.success)
        null = np.ones(dimension)
        expected = 2.0 * (-solution.fun - _log_likelihood(null, self.x))
        result = uniformity(self.x, model="symmetric")
        self.assertTrue(math.isclose(result.statistic, expected, rel_tol=2e-8))
        self.assertAlmostEqual(
            result.pvalue, float(stats.chi2.sf(expected, 1)), places=8
        )
        self.assertEqual(result.df, 1.0)

    def test_general_lrt_matches_independent_log_parameter_fit(self) -> None:
        expected = _independent_general_lrt(self.x)
        result = uniformity(self.x, model="general")
        self.assertTrue(math.isclose(result.statistic, expected, rel_tol=2e-9))
        self.assertAlmostEqual(
            result.pvalue,
            float(stats.chi2.sf(expected, self.x.shape[1])),
            places=9,
        )
        self.assertEqual(result.df, float(self.x.shape[1]))

    def test_asymmetric_general_fit_uses_verified_log_score_fallback(self) -> None:
        values = np.array(
            [
                [
                    2.5373980896327728e-04,
                    2.0111642082640890e-01,
                    5.8447248940995655e-01,
                    8.0101443971709041e-02,
                    8.6085766099162252e-11,
                    1.0674671386322668e-04,
                    7.8684260307661835e-12,
                    1.3270121375168817e-01,
                    7.0762592515930554e-07,
                    1.2472377975315438e-03,
                ],
                [
                    1.3844172997486782e-06,
                    1.8080013638634077e-01,
                    6.2018724435422468e-01,
                    1.5456228256010118e-01,
                    4.8600654431665042e-09,
                    1.1951312343692682e-09,
                    9.3595991760817587e-40,
                    4.0934159569735076e-02,
                    4.7646515210469882e-04,
                    3.0383215049972052e-03,
                ],
            ]
        )

        expected = _independent_general_lrt(values)
        actual = uniformity(values, model="general")

        np.testing.assert_allclose(actual.statistic, expected, rtol=2e-11)
        np.testing.assert_allclose(
            actual.pvalue, stats.chi2.sf(expected, values.shape[1]), rtol=2e-11
        )

    def test_general_model_contains_symmetric_model(self) -> None:
        symmetric = uniformity(self.x, model="symmetric")
        general = uniformity(self.x, model="general")
        self.assertGreaterEqual(general.statistic + 1e-11, symmetric.statistic)

    def test_component_permutations_preserve_both_tests(self) -> None:
        permuted = self.x[:, [2, 0, 1]]
        for model in ("symmetric", "general"):
            baseline = uniformity(self.x, model=model)
            reordered = uniformity(permuted, model=model)
            with self.subTest(model=model):
                self.assertTrue(
                    math.isclose(
                        reordered.statistic,
                        baseline.statistic,
                        rel_tol=2e-11,
                        abs_tol=2e-12,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        reordered.pvalue,
                        baseline.pvalue,
                        rel_tol=2e-11,
                        abs_tol=2e-12,
                    )
                )

    def test_rows_with_roundoff_sized_sum_error_are_normalized(self) -> None:
        perturbed = self.x.copy()
        perturbed[0, 0] += 2.0 * np.finfo(np.float64).eps
        baseline = uniformity(self.x)
        corrected = uniformity(perturbed)
        self.assertTrue(
            math.isclose(corrected.statistic, baseline.statistic, rel_tol=1e-13)
        )

    def test_concentrated_alternative_is_detected(self) -> None:
        rng = np.random.default_rng(6)
        concentrated = rng.dirichlet(np.full(4, 20.0), size=250)
        result = uniformity(concentrated, model="symmetric")
        self.assertLess(result.pvalue, 1e-12)

    def test_display_reports_model_and_calibration(self) -> None:
        rendered = str(uniformity(self.x, model="general"))
        self.assertIn("general Dirichlet model", rendered)
        self.assertIn("LR =", rendered)
        self.assertIn("df = 3", rendered)
        self.assertIn("Wilks", rendered)


class SimplexValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.array(
            [
                [0.2, 0.3, 0.5],
                [0.4, 0.2, 0.4],
                [0.1, 0.7, 0.2],
            ]
        )

    def test_simplex_interior_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            uniformity([[0.0, 0.4, 0.6], [0.2, 0.3, 0.5]])
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            uniformity([[-0.1, 0.5, 0.6], [0.2, 0.3, 0.5]])
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            uniformity([[0.2, 0.3, 0.4], [0.2, 0.3, 0.5]])
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            uniformity(
                [
                    [1.0, 1.0e-30, 2.0e-30],
                    [1.0, 3.0e-30, 1.0e-30],
                ]
            )

    def test_identical_rows_have_model_specific_mle_boundaries(self) -> None:
        repeated = np.tile([0.2, 0.3, 0.5], (6, 1))
        symmetric = uniformity(repeated)
        self.assertTrue(math.isfinite(symmetric.statistic))
        with self.assertRaisesRegex(ValueError, "unbounded"):
            uniformity(repeated, model="general")

        barycenter = np.full((6, 3), 1.0 / 3.0)
        with self.assertRaisesRegex(ValueError, "unbounded"):
            uniformity(barycenter, model="symmetric")

    def test_malformed_arrays_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            uniformity([0.2, 0.3, 0.5])
        with self.assertRaises(ValueError):
            uniformity([[1.0], [1.0]])
        with self.assertRaises(ValueError):
            uniformity([[0.2, 0.8]])
        with self.assertRaises(TypeError):
            uniformity(np.array([[True, False], [False, True]]))
        with self.assertRaises(TypeError):
            uniformity(np.array([[0.2 + 0.0j, 0.8], [0.4, 0.6]]))
        with self.assertRaises(ValueError):
            uniformity([[0.2, np.nan, 0.8], [0.3, 0.3, 0.4]])

    def test_optimizer_controls_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "model"):
            uniformity(self.x, model="lrt")
        with self.assertRaises(TypeError):
            uniformity(self.x, model=object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "tolerance"):
            uniformity(self.x, tolerance=0.0)
        with self.assertRaisesRegex(ValueError, "tolerance"):
            uniformity(self.x, tolerance=math.inf)
        with self.assertRaises(TypeError):
            uniformity(self.x, max_iter=True)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            uniformity(self.x, model="general", max_iter=1)


if __name__ == "__main__":
    unittest.main()

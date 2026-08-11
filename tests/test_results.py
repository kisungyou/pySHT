from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from pysht._resampling import monte_carlo_calibration
from pysht._results import (
    BayesFactorTestResult,
    DistanceTestResult,
    HypothesisTestResult,
    ResamplingTestResult,
    StatisticalTestResult,
)


class HypothesisTestResultTests(unittest.TestCase):
    def test_formats_like_an_r_htest_object(self) -> None:
        result = HypothesisTestResult(
            statistic=2.34567,
            pvalue=0.03125,
            method="One Sample Location Test",
            alternative="true location is not equal to 0",
            data_name="x",
            statistic_name="T",
            calibration="Student t approximation",
            df=12,
        )

        expected = (
            "One Sample Location Test\n"
            "\n"
            "data: x\n"
            "T = 2.34567, df = 12, p-value = 0.03125\n"
            "alternative hypothesis: true location is not equal to 0\n"
            "calibration: Student t approximation"
        )
        self.assertEqual(str(result), expected)
        self.assertEqual(repr(result), expected)

    def test_omits_optional_metadata(self) -> None:
        result = HypothesisTestResult(
            statistic=-0.5,
            pvalue=0.8,
            method="A Test",
            alternative="less",
        )

        self.assertEqual(
            str(result),
            "A Test\n\nstatistic = -0.5, p-value = 0.8\nalternative hypothesis: less",
        )

    def test_supports_multiple_degrees_of_freedom(self) -> None:
        result = HypothesisTestResult(
            statistic=4.2,
            pvalue=0.02,
            method="F Test",
            alternative="greater",
            statistic_name="F",
            df=(2, 18),
        )

        self.assertIn("F = 4.2, df = (2, 18), p-value = 0.02", str(result))

    def test_formats_confidence_interval_and_estimates(self) -> None:
        result = HypothesisTestResult(
            statistic=2.5,
            pvalue=0.025,
            method="One Sample t-test",
            alternative="true mean is not equal to 0",
            statistic_name="t",
            df=9,
            confidence_interval=(-0.125, float("inf")),
            confidence_level=0.95,
            estimates=(("mean of x", 1.25),),
        )

        rendered = str(result)
        self.assertIn("95 percent confidence interval:\n -0.125 Inf", rendered)
        self.assertIn("sample estimates:\nmean of x = 1.25", rendered)
        self.assertEqual(result.p_value, result.pvalue)

    def test_is_frozen_and_slotted(self) -> None:
        result = HypothesisTestResult(
            statistic=1,
            pvalue=0.5,
            method="A Test",
            alternative="two-sided",
        )

        with self.assertRaises(FrozenInstanceError):
            result.pvalue = 0.1  # type: ignore[misc]
        self.assertFalse(hasattr(result, "__dict__"))

    def test_rejects_invalid_values(self) -> None:
        common = {"statistic": 1, "method": "A Test", "alternative": "greater"}
        for pvalue in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(pvalue=pvalue), self.assertRaises(ValueError):
                HypothesisTestResult(pvalue=pvalue, **common)

        with self.assertRaises(ValueError):
            HypothesisTestResult(pvalue=0.5, df=0, **common)
        with self.assertRaises(ValueError):
            HypothesisTestResult(pvalue=0.5, data_name="  ", **common)
        with self.assertRaises(TypeError):
            HypothesisTestResult(
                pvalue=0.5,
                method=None,  # type: ignore[arg-type]
                alternative="greater",
                statistic=1,
            )
        with self.assertRaises(TypeError):
            HypothesisTestResult(
                pvalue="0.5",  # type: ignore[arg-type]
                **common,
            )
        with self.assertRaises(TypeError):
            HypothesisTestResult(
                pvalue=0.5,
                statistic=np.bool_(True),  # type: ignore[arg-type]
                method="A Test",
                alternative="greater",
            )
        with self.assertRaises(ValueError):
            HypothesisTestResult(
                pvalue=0.5,
                confidence_interval=(2, 1),
                confidence_level=0.95,
                **common,
            )
        with self.assertRaises(ValueError):
            HypothesisTestResult(
                pvalue=0.5,
                confidence_interval=(0, 1),
                **common,
            )
        with self.assertRaises(ValueError):
            HypothesisTestResult(
                pvalue=0.5,
                confidence_interval=(0, 1),
                confidence_level=1.0,
                **common,
            )
        with self.assertRaises(ValueError):
            HypothesisTestResult(
                pvalue=0.5,
                estimates=(("estimate", 1), ("estimate", 2)),
                **common,
            )

    def test_diagnostics_are_immutable_validated_and_rendered(self) -> None:
        result = HypothesisTestResult(
            statistic=1.5,
            pvalue=0.25,
            method="Projected test",
            alternative="not equal",
            diagnostics=(("n_projections", np.int64(25)), ("estimator", "CLIME")),
        )

        self.assertEqual(
            result.diagnostics,
            (("n_projections", 25), ("estimator", "CLIME")),
        )
        self.assertIn(
            "diagnostics:\nn_projections = 25\nestimator = CLIME",
            str(result),
        )
        with self.assertRaises(ValueError):
            HypothesisTestResult(
                statistic=1,
                pvalue=0.5,
                method="A Test",
                alternative="two-sided",
                diagnostics=(("count", 1), ("count", 2)),
            )

    def test_common_metadata_contract_rejects_malformed_values(self) -> None:
        common = {
            "statistic": 1,
            "pvalue": 0.5,
            "method": "A Test",
            "alternative": "two-sided",
        }
        invalid_overrides = (
            ("non-numeric statistic", {"statistic": object()}, TypeError),
            ("NaN statistic", {"statistic": float("nan")}, ValueError),
            ("non-string data name", {"data_name": 1}, TypeError),
            ("diagnostics list", {"diagnostics": [("count", 1)]}, TypeError),
            ("malformed diagnostic", {"diagnostics": (("count",),)}, TypeError),
            (
                "non-finite diagnostic",
                {"diagnostics": (("score", float("inf")),)},
                ValueError,
            ),
            ("empty diagnostic string", {"diagnostics": (("label", " "),)}, ValueError),
            (
                "unsupported diagnostic type",
                {"diagnostics": (("array", np.array([1])),)},
                TypeError,
            ),
        )
        for label, override, exception in invalid_overrides:
            with self.subTest(label=label), self.assertRaises(exception):
                HypothesisTestResult(**(common | override))

    def test_common_metadata_formats_numeric_diagnostic_boundaries(self) -> None:
        result = HypothesisTestResult(
            statistic=-float("inf"),
            pvalue=0,
            method="Boundary test",
            alternative="two-sided",
            diagnostics=(("enabled", np.bool_(True)), ("score", 0.0)),
        )

        self.assertIn("statistic = -Inf, p-value = 0", str(result))
        self.assertIn("enabled = True\nscore = 0", str(result))

    def test_frequentist_metadata_contract_rejects_malformed_values(self) -> None:
        common = {
            "statistic": 1,
            "pvalue": 0.5,
            "method": "A Test",
            "alternative": "two-sided",
        }
        invalid_overrides = (
            ("empty df tuple", {"df": ()}, ValueError),
            ("confidence level only", {"confidence_level": 0.95}, ValueError),
            (
                "confidence interval list",
                {"confidence_interval": [0, 1], "confidence_level": 0.95},
                TypeError,
            ),
            (
                "NaN confidence bound",
                {
                    "confidence_interval": (float("nan"), 1),
                    "confidence_level": 0.95,
                },
                ValueError,
            ),
            ("estimates list", {"estimates": [("mean", 1)]}, TypeError),
            ("malformed estimate", {"estimates": (("mean",),)}, TypeError),
            (
                "non-finite estimate",
                {"estimates": (("mean", float("inf")),)},
                ValueError,
            ),
        )
        for label, override, exception in invalid_overrides:
            with self.subTest(label=label), self.assertRaises(exception):
                HypothesisTestResult(**(common | override))


class ResamplingTestResultTests(unittest.TestCase):
    def test_formats_resampling_diagnostics(self) -> None:
        _, standard_error, interval = monte_carlo_calibration(14, 9_999)
        result = ResamplingTestResult(
            statistic=1.75,
            pvalue=0.0015,
            method="Permutation Equality Test",
            alternative="distributions are different",
            data_name="x and y",
            statistic_name="BG",
            calibration="permutation",
            n_resamples=9_999,
            exceedances=14,
            monte_carlo_standard_error=standard_error,
            tail_probability_interval=interval,
        )

        expected = (
            "Permutation Equality Test\n"
            "\n"
            "data: x and y\n"
            "BG = 1.75, p-value = 0.0015\n"
            "alternative hypothesis: distributions are different\n"
            "calibration: permutation\n"
            "resampling: 9,999 resamples, 14 exceedances\n"
            "Monte Carlo standard error: 0.000373904\n"
            "95 percent tail-probability interval: 0.000765674 0.00234808"
        )
        self.assertEqual(str(result), expected)
        self.assertEqual(repr(result), expected)

    def test_validates_resampling_counts(self) -> None:
        common = {
            "statistic": 1,
            "pvalue": 0.5,
            "method": "A Test",
            "alternative": "two-sided",
        }
        with self.assertRaises(ValueError):
            ResamplingTestResult(n_resamples=0, exceedances=0, **common)
        with self.assertRaises(ValueError):
            ResamplingTestResult(n_resamples=10, exceedances=11, **common)
        with self.assertRaises(TypeError):
            ResamplingTestResult(n_resamples=10.5, exceedances=2, **common)
        with self.assertRaises(ValueError):
            ResamplingTestResult(
                n_resamples=10,
                exceedances=2,
                monte_carlo_standard_error=-0.1,
                **common,
            )

    def test_monte_carlo_uncertainty_must_match_counts(self) -> None:
        common = {
            "statistic": 1,
            "pvalue": 3 / 11,
            "method": "A Test",
            "alternative": "two-sided",
            "n_resamples": 10,
            "exceedances": 2,
        }
        _, standard_error, interval = monte_carlo_calibration(2, 10)
        with self.assertRaises(ValueError, msg="missing uncertainty metadata"):
            ResamplingTestResult(**common)
        with self.assertRaises(ValueError, msg="inconsistent standard error"):
            ResamplingTestResult(
                monte_carlo_standard_error=standard_error + 0.01,
                tail_probability_interval=interval,
                **common,
            )
        with self.assertRaises(ValueError, msg="inconsistent interval"):
            ResamplingTestResult(
                monte_carlo_standard_error=standard_error,
                tail_probability_interval=(interval[0], interval[1] - 0.01),
                **common,
            )

    def test_count_derived_values_require_exact_consistency(self) -> None:
        # Regression guard: the former absolute tolerance of 1e-15 accepted a
        # zero p-value even though one of 10**16 exact outcomes exceeded the
        # observed statistic.
        with self.assertRaisesRegex(ValueError, "pvalue is inconsistent"):
            ResamplingTestResult(
                statistic=1,
                pvalue=0.0,
                method="Exact test",
                alternative="two-sided",
                n_resamples=10**16,
                exceedances=1,
                exact=True,
            )

        pvalue, standard_error, interval = monte_carlo_calibration(2, 10)
        with self.assertRaisesRegex(ValueError, "standard_error"):
            ResamplingTestResult(
                statistic=1,
                pvalue=pvalue,
                method="Monte Carlo test",
                alternative="two-sided",
                n_resamples=10,
                exceedances=2,
                monte_carlo_standard_error=math.nextafter(standard_error, math.inf),
                tail_probability_interval=interval,
            )

    def test_tail_interval_is_deliberately_fixed_at_95_percent(self) -> None:
        pvalue, standard_error, interval = monte_carlo_calibration(
            2, 10, confidence_level=0.90
        )
        with self.assertRaisesRegex(ValueError, "interval is inconsistent"):
            ResamplingTestResult(
                statistic=1,
                pvalue=pvalue,
                method="Monte Carlo test",
                alternative="two-sided",
                n_resamples=10,
                exceedances=2,
                monte_carlo_standard_error=standard_error,
                tail_probability_interval=interval,
            )

    def test_singular_exceedance_is_grammatical(self) -> None:
        result = ResamplingTestResult(
            statistic=1,
            pvalue=0.02,
            method="A Test",
            alternative="two-sided",
            n_resamples=99,
            exceedances=1,
            monte_carlo_standard_error=monte_carlo_calibration(1, 99)[1],
            tail_probability_interval=monte_carlo_calibration(1, 99)[2],
        )

        self.assertIn("99 resamples, 1 exceedance", str(result))

    def test_rejects_malformed_boolean_and_interval_metadata(self) -> None:
        common = {
            "statistic": 1,
            "pvalue": 0.5,
            "method": "A Test",
            "alternative": "two-sided",
            "n_resamples": 10,
            "exceedances": 5,
        }
        invalid_overrides = (
            ("boolean resample count", {"n_resamples": True}, TypeError),
            ("boolean exceedance count", {"exceedances": np.bool_(True)}, TypeError),
            ("non-integer exceedance count", {"exceedances": 2.5}, TypeError),
            ("non-boolean exact flag", {"exact": 1}, TypeError),
            (
                "interval list",
                {
                    "monte_carlo_standard_error": 0.1,
                    "tail_probability_interval": [0.1, 0.9],
                },
                TypeError,
            ),
            (
                "interval outside unit range",
                {
                    "monte_carlo_standard_error": 0.1,
                    "tail_probability_interval": (-0.1, 0.9),
                },
                ValueError,
            ),
            (
                "uncertainty on exact result",
                {
                    "exact": True,
                    "monte_carlo_standard_error": 0.1,
                    "tail_probability_interval": (0.1, 0.9),
                },
                ValueError,
            ),
        )
        for label, override, exception in invalid_overrides:
            with self.subTest(label=label), self.assertRaises(exception):
                ResamplingTestResult(**(common | override))


class DistanceTestResultTests(unittest.TestCase):
    def test_displays_raw_and_normalized_statistics(self) -> None:
        result = DistanceTestResult(
            statistic=40 / 9,
            normalized_statistic=5 / 72,
            distance_scale=8,
            exact=True,
            pvalue=0.8,
            method="Distance test",
            alternative="distributions differ",
            statistic_name="T_mn",
            calibration="exact permutation",
            n_resamples=10,
            exceedances=8,
        )

        rendered = str(result)
        self.assertIn("T_mn = 4.44444", rendered)
        self.assertIn("T_mn / d_max^2 = 0.0694444, d_max = 8", rendered)

    def test_rejects_invalid_normalization_metadata(self) -> None:
        common = {
            "statistic": 1,
            "pvalue": 0.2,
            "method": "Distance test",
            "alternative": "distributions differ",
            "n_resamples": 10,
            "exceedances": 2,
            "exact": True,
        }
        with self.assertRaises(ValueError):
            DistanceTestResult(
                normalized_statistic=-1,
                distance_scale=1,
                **common,
            )
        with self.assertRaises(ValueError):
            DistanceTestResult(
                normalized_statistic=1,
                distance_scale=float("nan"),
                **common,
            )
        with self.assertRaises(ValueError):
            DistanceTestResult(
                normalized_statistic=float("inf"),
                distance_scale=1,
                **common,
            )
        with self.assertRaises(ValueError):
            DistanceTestResult(
                normalized_statistic=1,
                distance_scale=-1,
                **common,
            )


class BayesFactorTestResultTests(unittest.TestCase):
    def test_has_no_pvalue_and_displays_log_evidence(self) -> None:
        result = BayesFactorTestResult(
            statistic=2.5,
            component_log_bayes_factors=(-1.0, 2.5, 0.25),
            method="Lee-You-Lin two-sample mean test",
            alternative="the mean vectors differ",
            statistic_name="maximum log BF",
            calibration="Bayesian model comparison",
        )

        self.assertIsInstance(result, StatisticalTestResult)
        self.assertEqual(result.max_log_bayes_factor, 2.5)
        self.assertFalse(hasattr(result, "pvalue"))
        self.assertIn("maximum log BF = 2.5", str(result))
        self.assertIn("frequentist p-value: not defined", str(result))

    def test_validates_component_shape_and_maximum(self) -> None:
        common = {
            "method": "Bayesian test",
            "alternative": "models differ",
        }
        with self.assertRaises(ValueError):
            BayesFactorTestResult(
                statistic=1,
                component_log_bayes_factors=(0.0, 2.0),
                **common,
            )
        with self.assertRaises(ValueError):
            BayesFactorTestResult(
                statistic=2,
                component_log_bayes_factors=((1.0, 2.0), (0.0,)),
                **common,
            )

    def test_large_maximum_must_match_exactly(self) -> None:
        # A relative tolerance made the old invariant weaker as the evidence
        # magnitude grew; at 1e20 it accepted errors of order ten million.
        with self.assertRaisesRegex(ValueError, "maximum component"):
            BayesFactorTestResult(
                statistic=1.0e20 + 1.0e6,
                component_log_bayes_factors=(1.0e20, 2.0),
                method="Bayesian test",
                alternative="models differ",
            )

    def test_rejects_malformed_or_nan_components(self) -> None:
        common = {
            "statistic": 1,
            "method": "Bayesian test",
            "alternative": "models differ",
        }
        invalid_components = (
            (),
            (float("nan"),),
            ((1.0,), 2.0),
            ((),),
            ((float("nan"),),),
        )
        for components in invalid_components:
            with (
                self.subTest(components=components),
                self.assertRaises((TypeError, ValueError)),
            ):
                BayesFactorTestResult(
                    component_log_bayes_factors=components,  # type: ignore[arg-type]
                    **common,
                )


if __name__ == "__main__":
    unittest.main()

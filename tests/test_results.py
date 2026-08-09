from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from pysht._results import (
    DistanceTestResult,
    HypothesisTestResult,
    ResamplingTestResult,
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


class ResamplingTestResultTests(unittest.TestCase):
    def test_formats_resampling_diagnostics(self) -> None:
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
            monte_carlo_standard_error=0.0012,
        )

        expected = (
            "Permutation Equality Test\n"
            "\n"
            "data: x and y\n"
            "BG = 1.75, p-value = 0.0015\n"
            "alternative hypothesis: distributions are different\n"
            "calibration: permutation\n"
            "resampling: 9,999 resamples, 14 exceedances\n"
            "Monte Carlo standard error: 0.0012"
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

    def test_singular_exceedance_is_grammatical(self) -> None:
        result = ResamplingTestResult(
            statistic=1,
            pvalue=0.02,
            method="A Test",
            alternative="two-sided",
            n_resamples=99,
            exceedances=1,
        )

        self.assertIn("99 resamples, 1 exceedance", str(result))


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


if __name__ == "__main__":
    unittest.main()

"""Independent formulas for the October 2026 mean-test numerical defects."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.typing import NDArray

from pysht import mean


@pytest.mark.parametrize("null", [0.0, 1e15, 1e20])
def test_distant_null_preserves_t_and_hotelling_residuals(null: float) -> None:
    values = np.array([-1.0, 0.0, 1.0])
    statistic = -math.sqrt(3.0) * null
    # The df=2 t survival function has a closed form. Rationalization
    # avoids subtracting two nearly equal numbers for a distant null.
    root = math.sqrt(statistic**2 + 2.0)
    expected_p = 2.0 / (root * (root + abs(statistic)))
    result = mean.ttest_1samp(values, popmean=null)
    assert result.statistic == pytest.approx(statistic, rel=1e-14, abs=0.0)
    assert result.pvalue == pytest.approx(expected_p, rel=1e-13, abs=0.0)
    # For df=2 the central probability is t/sqrt(t**2+2).
    margin = math.sqrt(2.0 / 3.0) * 0.95 / math.sqrt(1.0 - 0.95**2)
    assert result.confidence_interval == pytest.approx((-margin, margin), rel=1e-10)
    assert result.estimates == (("mean of x", 0.0),)
    multivariate = mean.hotelling_1samp(values[:, None], popmean=[null])
    assert multivariate.statistic == pytest.approx(statistic**2, rel=1e-14, abs=0.0)
    assert multivariate.pvalue == pytest.approx(expected_p, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("name", ["dempster_1samp", "bs_1samp", "sd_1samp"])
def test_high_dimensional_distant_null_matches_dense_covariance(name: str) -> None:
    values = np.random.default_rng(784).normal(size=(40, 80))
    n, p = values.shape
    nu = n - 1
    covariance = np.cov(values, rowvar=False)
    trace = float(np.trace(covariance))
    trace_square = float(np.sum(covariance * covariance))
    unbiased_trace_square = (
        nu**2 / ((nu - 1) * (nu + 2)) * (trace_square - trace**2 / nu)
    )
    difference = values.mean(axis=0) - 1e20
    squared_norm = float(difference @ difference)
    if name == "dempster_1samp":
        expected = n * squared_norm / trace
    elif name == "bs_1samp":
        expected = (n * squared_norm - trace) / math.sqrt(
            2 * (n / nu) * unbiased_trace_square
        )
    else:
        variances = np.diag(covariance)
        correlation = covariance / np.sqrt(np.outer(variances, variances))
        correlation_trace_square = float(np.sum(correlation * correlation))
        expected = (
            n * float(np.sum(difference**2 / variances)) - nu * p / (nu - 2)
        ) / math.sqrt(
            2
            * (correlation_trace_square - p**2 / nu)
            * (1 + correlation_trace_square / p**1.5)
        )
    actual = getattr(mean, name)(values, popmean=np.full(p, 1e20))
    assert actual.statistic == pytest.approx(expected, rel=3e-13)
    assert actual.pvalue == 0.0


@pytest.mark.parametrize(
    "confidence", [np.nextafter(0.0, 1.0), 1e-200, 1e-20, 1e-6, 1e-4, 0.1, 0.95]
)
@pytest.mark.parametrize("design", ["one-sample", "paired", "welch"])
def test_central_t_interval_matches_closed_df2_formula(
    confidence: float, design: str
) -> None:
    values = np.array([-1.0, 0.0, 1.0])
    if design == "one-sample":
        result = mean.ttest_1samp(values, confidence_level=confidence)
    else:
        result = mean.ttest_2samp(
            values,
            np.zeros(3),
            paired=design == "paired",
            confidence_level=confidence,
        )
    with localcontext() as context:
        context.prec = 90
        c = Decimal.from_float(float(confidence))
        margin = float((Decimal(2) / 3).sqrt() * c / (1 - c**2).sqrt())
    assert result.confidence_interval == pytest.approx(
        (-margin, margin), rel=1e-10, abs=0.0
    )


@pytest.mark.parametrize("confidence", [1e-200, 1e-20, 1e-6, 1e-4, 0.1])
def test_central_t_interval_matches_closed_cauchy_formula(confidence: float) -> None:
    # With two observations [-1,1], the standard error is one and df=1.
    result = mean.ttest_1samp([-1.0, 1.0], confidence_level=confidence)
    margin = math.tan(math.pi * confidence / 2.0)
    assert result.confidence_interval == pytest.approx(
        (-margin, margin), rel=1e-13, abs=0.0
    )


@pytest.mark.parametrize("alternative", ["less", "greater"])
@pytest.mark.parametrize("design", ["one-sample", "paired", "welch"])
@pytest.mark.parametrize("confidence", [1e-20, 1e-200, 1e-300])
def test_one_sided_t_interval_retains_tiny_confidence(
    alternative: str, design: str, confidence: float
) -> None:
    values = np.array([-1.0, 0.0, 1.0])
    if design == "one-sample":
        result = mean.ttest_1samp(
            values, alternative=alternative, confidence_level=confidence
        )
    else:
        result = mean.ttest_2samp(
            values,
            np.zeros(3),
            alternative=alternative,
            paired=design == "paired",
            confidence_level=confidence,
        )
    # Inverting the df=2 CDF without computing 1 - a tiny confidence.
    quantile = (2 * confidence - 1) / math.sqrt(2 * confidence * (1 - confidence))
    bound = quantile / math.sqrt(3.0)
    expected = (-math.inf, bound) if alternative == "less" else (-bound, math.inf)
    assert result.confidence_interval == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("confidence", [1e-20, 1e-200, 1e-300])
def test_tiny_one_sided_cauchy_interval_matches_cotangent(confidence: float) -> None:
    result = mean.ttest_1samp(
        [-1.0, 1.0], alternative="less", confidence_level=confidence
    )
    bound = -1.0 / math.tan(math.pi * confidence)
    assert result.confidence_interval == pytest.approx((-math.inf, bound), rel=2e-13)


def _decimal_column(values: NDArray[np.float64]) -> list[Decimal]:
    return [Decimal.from_float(float(value)) for value in values]


def _average(values: list[Decimal]) -> Decimal:
    return sum(values, Decimal(0)) / len(values)


def _rss(values: list[Decimal]) -> Decimal:
    average = _average(values)
    return sum(((value - average) ** 2 for value in values), Decimal(0))


@pytest.mark.parametrize("units", [[1.0, 1e-160], [1.0, 1e-200], [1e200, 1e-200]])
def test_mean_bayes_factor_matches_decimal_featurewise_rss(units: list[float]) -> None:
    rng = np.random.default_rng(41)
    first = rng.normal(size=(10, 2)) * units
    second = rng.normal(size=(12, 2)) * units
    expected = []
    with localcontext() as context:
        context.prec = 90
        log_gamma = -Decimal.from_float(2.01) * Decimal(22).ln()
        gamma = log_gamma.exp()
        for column in range(2):
            x = _decimal_column(first[:, column])
            y = _decimal_column(second[:, column])
            ratio = _rss(x + y) / (_rss(x) + _rss(y))
            expected.append(float((log_gamma - (1 + gamma).ln()) / 2 + 11 * ratio.ln()))
    result = mean.maximum_pairwise_bayes_factor_2samp(first, second)
    assert result.component_log_bayes_factors == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("units", [[1.0, 1e-200], [1e200, 1e-200]])
def test_zx_hotelling_matches_decimal_scheffe_transform(units: list[float]) -> None:
    rng = np.random.default_rng(41)
    first = rng.normal(size=(10, 2)) * units
    second = rng.normal(size=(12, 2)) * units
    # Form the literal Scheffe sample, then invert its 2x2 covariance with
    # Decimal arithmetic. This checks the actual scaled float inputs and
    # never calls pySHT's centering, scaling, covariance, or inverse helpers.
    with localcontext() as context:
        context.prec = 90
        ratio = (Decimal(10) / 12).sqrt()
        columns = []
        for column in range(2):
            x = _decimal_column(first[:, column])
            y = _decimal_column(second[:, column])
            full_mean, partial_mean = _average(y), _average(y[:10])
            columns.append(
                [x[i] - full_mean + ratio * (y[i] - partial_mean) for i in range(10)]
            )
        means = [_average(column) for column in columns]
        centered = [
            [value - average for value in column]
            for column, average in zip(columns, means, strict=True)
        ]
        s00 = sum((value * value for value in centered[0]), Decimal(0)) / 9
        s11 = sum((value * value for value in centered[1]), Decimal(0)) / 9
        s01 = sum((x * y for x, y in zip(*centered, strict=True)), Decimal(0)) / 9
        determinant = s00 * s11 - s01 * s01
        assert determinant > 0
        statistic = (
            10
            * (
                s11 * means[0] ** 2
                - 2 * s01 * means[0] * means[1]
                + s00 * means[1] ** 2
            )
            / determinant
        )
        f_statistic = Decimal(8) * statistic / 18
        # F(2,8) upper tail, independently of SciPy's F survival function.
        pvalue = (Decimal(8) / (8 + 2 * f_statistic)) ** 4
    result = mean.zx_ksamp(first, second, base_test="hotelling")
    assert result.statistic == pytest.approx(float(statistic), rel=2e-13)
    assert result.pvalue == pytest.approx(float(pvalue), rel=2e-13)

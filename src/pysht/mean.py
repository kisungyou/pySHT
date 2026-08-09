"""Tests for univariate and multivariate population means.

The routines in this module implement their statistics directly and use
SciPy only for well-tested distribution functions.  Calculations are carried
out after a positive common rescaling where possible, which preserves every
reported test statistic while avoiding avoidable overflow and underflow.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._results import HypothesisTestResult
from ._validation import (
    Alternative,
    validate_1d_sample,
    validate_2d_sample,
    validate_alternative,
    validate_bool,
    validate_confidence_level,
    validate_groups,
    validate_real_scalar,
)

__all__ = [
    "anova_oneway",
    "hotelling_1samp",
    "hotelling_2samp",
    "ttest_1samp",
    "ttest_2samp",
]

_T_CALIBRATION: Final = "Student t distribution"
_F_CALIBRATION: Final = "F distribution"


def _display_number(value: float) -> str:
    return format(value, ".6g")


def _scale_for(*arrays: NDArray[np.float64], scalars: tuple[float, ...] = ()) -> float:
    maxima = [float(np.max(np.abs(values))) for values in arrays if values.size]
    maxima.extend(abs(value) for value in scalars)
    scale = max(maxima, default=0.0)
    return scale if scale > 0.0 else 1.0


def _scaled_mean_sd(values: NDArray[np.float64], scale: float) -> tuple[float, float]:
    scaled = values / scale
    mean = float(np.mean(scaled, dtype=np.float64))
    centered = scaled - mean
    squared_norm = float(np.dot(centered, centered))
    standard_deviation = math.sqrt(squared_norm / (values.size - 1))
    return mean, standard_deviation


def _rescale(value: float, scale: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.float64(value) * np.float64(scale))


def _t_pvalue(statistic: float, df: float, alternative: Alternative) -> float:
    if alternative == "two-sided":
        return float(2.0 * stats.t.sf(abs(statistic), df))
    if alternative == "less":
        return float(stats.t.cdf(statistic, df))
    return float(stats.t.sf(statistic, df))


def _t_confidence_interval(
    estimate_scaled: float,
    standard_error_scaled: float,
    df: float,
    alternative: Alternative,
    confidence_level: float,
    scale: float,
) -> tuple[float, float]:
    if alternative == "two-sided":
        quantile = float(stats.t.ppf((1.0 + confidence_level) / 2.0, df))
        lower_scaled = estimate_scaled - quantile * standard_error_scaled
        upper_scaled = estimate_scaled + quantile * standard_error_scaled
    else:
        quantile = float(stats.t.ppf(confidence_level, df))
        if alternative == "less":
            lower_scaled = -math.inf
            upper_scaled = estimate_scaled + quantile * standard_error_scaled
        else:
            lower_scaled = estimate_scaled - quantile * standard_error_scaled
            upper_scaled = math.inf
    return _rescale(lower_scaled, scale), _rescale(upper_scaled, scale)


def _mean_alternative(subject: str, alternative: Alternative, null: float) -> str:
    relation = {
        "two-sided": "is not equal to",
        "less": "is less than",
        "greater": "is greater than",
    }[alternative]
    return f"{subject} {relation} {_display_number(null)}"


def ttest_1samp(
    x: ArrayLike,
    *,
    popmean: float = 0.0,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
) -> HypothesisTestResult:
    """Perform a one-sample Student t test.

    The observations are assumed independent and normally distributed with an
    unknown, positive variance.  ``alternative`` describes the population
    mean relative to ``popmean``.
    """
    values = validate_1d_sample(x, name="x")
    null = validate_real_scalar(popmean, name="popmean")
    selected_alternative = validate_alternative(alternative)
    level = validate_confidence_level(confidence_level)

    scale = _scale_for(values, scalars=(null,))
    mean_scaled, sd_scaled = _scaled_mean_sd(values, scale)
    if sd_scaled == 0.0:
        raise ValueError("the sample variance must be positive")
    null_scaled = null / scale
    n = values.size
    df = float(n - 1)
    standard_error_scaled = sd_scaled / math.sqrt(n)
    statistic = (mean_scaled - null_scaled) / standard_error_scaled
    pvalue = _t_pvalue(statistic, df, selected_alternative)
    interval = _t_confidence_interval(
        mean_scaled,
        standard_error_scaled,
        df,
        selected_alternative,
        level,
        scale,
    )

    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="One Sample t-test",
        alternative=_mean_alternative("true mean", selected_alternative, null),
        data_name="x",
        statistic_name="t",
        calibration=_T_CALIBRATION,
        df=df,
        confidence_interval=interval,
        confidence_level=level,
        estimates=(("mean of x", _rescale(mean_scaled, scale)),),
    )


def ttest_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alternative: str = "two-sided",
    paired: bool = False,
    equal_var: bool = False,
    confidence_level: float = 0.95,
) -> HypothesisTestResult:
    """Perform an independent or paired two-sample t test.

    Welch's unequal-variance test is the default for independent samples. Set
    ``equal_var=True`` for the pooled-variance Student test, or ``paired=True``
    for a test of paired differences.
    """
    first = validate_1d_sample(x, name="x")
    second = validate_1d_sample(y, name="y")
    selected_alternative = validate_alternative(alternative)
    is_paired = validate_bool(paired, name="paired")
    uses_equal_variance = validate_bool(equal_var, name="equal_var")
    level = validate_confidence_level(confidence_level)
    if is_paired and uses_equal_variance:
        raise ValueError("equal_var is not applicable when paired=True")
    if is_paired and first.size != second.size:
        raise ValueError("paired samples must contain the same number of observations")

    scale = _scale_for(first, second)
    estimates: tuple[tuple[str, float], ...]
    if is_paired:
        differences_scaled = first / scale - second / scale
        difference_mean_scaled, difference_sd_scaled = _scaled_mean_sd(
            differences_scaled, 1.0
        )
        if difference_sd_scaled == 0.0:
            raise ValueError("the variance of paired differences must be positive")
        df = float(first.size - 1)
        standard_error_scaled = difference_sd_scaled / math.sqrt(first.size)
        statistic = difference_mean_scaled / standard_error_scaled
        method = "Paired t-test"
        estimates = (
            ("mean paired difference", _rescale(difference_mean_scaled, scale)),
        )
        difference_scaled = difference_mean_scaled
    else:
        first_mean_scaled, first_sd_scaled = _scaled_mean_sd(first, scale)
        second_mean_scaled, second_sd_scaled = _scaled_mean_sd(second, scale)
        difference_scaled = first_mean_scaled - second_mean_scaled
        estimates = (
            ("mean of x", _rescale(first_mean_scaled, scale)),
            ("mean of y", _rescale(second_mean_scaled, scale)),
        )

    if not is_paired and uses_equal_variance:
        df = float(first.size + second.size - 2)
        pooled_variance_scaled = (
            (first.size - 1) * first_sd_scaled**2
            + (second.size - 1) * second_sd_scaled**2
        ) / df
        standard_error_scaled = math.sqrt(
            pooled_variance_scaled * (1.0 / first.size + 1.0 / second.size)
        )
        if standard_error_scaled == 0.0:
            raise ValueError("the pooled variance must be positive")
        statistic = difference_scaled / standard_error_scaled
        method = "Two Sample t-test"
    elif not is_paired:
        first_component = first_sd_scaled**2 / first.size
        second_component = second_sd_scaled**2 / second.size
        standard_error_scaled = math.sqrt(first_component + second_component)
        if standard_error_scaled == 0.0:
            raise ValueError("at least one sample variance must be positive")
        df = (first_component + second_component) ** 2 / (
            first_component**2 / (first.size - 1)
            + second_component**2 / (second.size - 1)
        )
        statistic = difference_scaled / standard_error_scaled
        method = "Welch Two Sample t-test"

    pvalue = _t_pvalue(statistic, df, selected_alternative)
    interval = _t_confidence_interval(
        difference_scaled,
        standard_error_scaled,
        df,
        selected_alternative,
        level,
        scale,
    )
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method=method,
        alternative=_mean_alternative(
            "true difference in means", selected_alternative, 0.0
        ),
        data_name="x and y",
        statistic_name="t",
        calibration=_T_CALIBRATION,
        df=df,
        confidence_interval=interval,
        confidence_level=level,
        estimates=estimates,
    )


def anova_oneway(*samples: ArrayLike) -> HypothesisTestResult:
    """Perform the classical fixed-effects one-way analysis of variance.

    The test assumes independent normal observations and a common positive
    population variance across groups.
    """
    groups = validate_groups(samples)
    scale = _scale_for(*groups)
    scaled_groups = tuple(group / scale for group in groups)
    sizes = np.asarray([group.size for group in scaled_groups], dtype=np.float64)
    means = np.asarray(
        [np.mean(group, dtype=np.float64) for group in scaled_groups],
        dtype=np.float64,
    )
    total_size = int(np.sum(sizes))
    group_count = len(groups)
    grand_mean = float(np.dot(sizes, means) / total_size)
    between = float(np.dot(sizes, (means - grand_mean) ** 2))
    within = float(
        sum(
            np.dot(group - mean, group - mean)
            for group, mean in zip(scaled_groups, means)
        )
    )
    if within == 0.0:
        raise ValueError("the pooled within-group variance must be positive")

    numerator_df = float(group_count - 1)
    denominator_df = float(total_size - group_count)
    statistic = (between / numerator_df) / (within / denominator_df)
    pvalue = float(stats.f.sf(statistic, numerator_df, denominator_df))
    estimates = tuple(
        (f"mean of sample {index}", _rescale(float(mean), scale))
        for index, mean in enumerate(means, start=1)
    )
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="One-way analysis of variance",
        alternative="at least one population mean differs",
        data_name="samples",
        statistic_name="F",
        calibration=_F_CALIBRATION,
        df=(numerator_df, denominator_df),
        estimates=estimates,
    )


def _scaled_covariance(
    values: NDArray[np.float64], scales: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    scaled = values / scales
    mean = np.mean(scaled, axis=0, dtype=np.float64)
    centered = scaled - mean
    covariance = centered.T @ centered / (values.shape[0] - 1)
    return mean, covariance


def _positive_definite_quadratic(
    covariance: NDArray[np.float64], difference: NDArray[np.float64]
) -> float:
    try:
        factor = np.linalg.cholesky(covariance)
        standardized = np.linalg.solve(factor, difference)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the covariance estimate must be positive definite") from exc
    result = float(np.dot(standardized, standardized))
    if math.isnan(result) or result < 0.0:
        raise ValueError("the Hotelling quadratic form could not be evaluated")
    return result


def hotelling_1samp(
    x: ArrayLike, *, popmean: ArrayLike | None = None
) -> HypothesisTestResult:
    """Perform the exact one-sample Hotelling :math:`T^2` test.

    Exact F calibration requires multivariate normal observations, ``n > p``,
    and a positive-definite sample covariance matrix.
    """
    values = validate_2d_sample(x, name="x")
    n, p = values.shape
    if n <= p:
        raise ValueError("hotelling_1samp requires more observations than features")
    if popmean is None:
        null = np.zeros(p, dtype=np.float64)
    else:
        null = validate_1d_sample(popmean, name="popmean", minimum_size=1)
        if null.size != p:
            raise ValueError("popmean must contain one value per feature")

    scales = np.maximum(np.max(np.abs(values), axis=0), np.abs(null))
    scales[scales == 0.0] = 1.0
    mean_scaled, covariance_scaled = _scaled_covariance(values, scales)
    difference_scaled = mean_scaled - null / scales
    quadratic = _positive_definite_quadratic(covariance_scaled, difference_scaled)
    statistic = n * quadratic
    numerator_df = float(p)
    denominator_df = float(n - p)
    f_statistic = denominator_df * statistic / (p * (n - 1))
    pvalue = float(stats.f.sf(f_statistic, numerator_df, denominator_df))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="One-sample Hotelling's T-squared test",
        alternative="true mean vector differs from popmean",
        data_name="x",
        statistic_name="T2",
        calibration=_F_CALIBRATION,
        df=(numerator_df, denominator_df),
    )


def hotelling_2samp(
    x: ArrayLike, y: ArrayLike, *, paired: bool = False
) -> HypothesisTestResult:
    """Perform the exact equal-covariance two-sample Hotelling test.

    Independent samples are assumed to share a positive-definite covariance
    matrix. With ``paired=True``, the exact one-sample test is applied to the
    row-wise differences.
    """
    first = validate_2d_sample(x, name="x")
    second = validate_2d_sample(y, name="y")
    is_paired = validate_bool(paired, name="paired")
    if first.shape[1] != second.shape[1]:
        raise ValueError("x and y must have the same number of features")
    if is_paired:
        if first.shape[0] != second.shape[0]:
            raise ValueError("paired samples must have the same number of rows")
        scales = np.maximum(
            np.max(np.abs(first), axis=0), np.max(np.abs(second), axis=0)
        )
        scales[scales == 0.0] = 1.0
        one_sample = hotelling_1samp(first / scales - second / scales)
        return HypothesisTestResult(
            statistic=one_sample.statistic,
            pvalue=one_sample.pvalue,
            method="Paired Hotelling's T-squared test",
            alternative="true mean difference vector is not zero",
            data_name="x and y",
            statistic_name="T2",
            calibration=one_sample.calibration,
            df=one_sample.df,
        )

    first_size, p = first.shape
    second_size = second.shape[0]
    total_size = first_size + second_size
    if total_size <= p + 1:
        raise ValueError("hotelling_2samp requires n_x + n_y to exceed p + 1")

    scales = np.maximum(np.max(np.abs(first), axis=0), np.max(np.abs(second), axis=0))
    scales[scales == 0.0] = 1.0
    first_mean, first_covariance = _scaled_covariance(first, scales)
    second_mean, second_covariance = _scaled_covariance(second, scales)
    pooled_covariance = (
        (first_size - 1) * first_covariance + (second_size - 1) * second_covariance
    ) / (total_size - 2)
    difference = first_mean - second_mean
    quadratic = _positive_definite_quadratic(pooled_covariance, difference)
    statistic = first_size * second_size / total_size * quadratic
    numerator_df = float(p)
    denominator_df = float(total_size - p - 1)
    f_statistic = denominator_df * statistic / (p * (total_size - 2))
    pvalue = float(stats.f.sf(f_statistic, numerator_df, denominator_df))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Two-sample Hotelling's T-squared test",
        alternative="true mean vectors differ",
        data_name="x and y",
        statistic_name="T2",
        calibration=_F_CALIBRATION,
        df=(numerator_df, denominator_df),
    )

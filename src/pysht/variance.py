"""Tests for one, two, and multiple population variances.

The chi-square, F, and Bartlett procedures are normal-theory tests.  Levene's
test and its Brown--Forsythe median-centered variant are less sensitive to
departures from normality, but their reported F distributions are still
finite-sample approximations.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats

from ._results import HypothesisTestResult
from ._validation import (
    Alternative,
    validate_1d_sample,
    validate_alternative,
    validate_confidence_level,
    validate_groups,
    validate_real_scalar,
)

__all__ = [
    "bartlett",
    "brown_forsythe",
    "chisquare_1samp",
    "f_2samp",
    "levene",
]


_LOG_FLOAT_MAX = math.log(float(np.finfo(np.float64).max))
_LOG_FLOAT_MIN = math.log(float(np.nextafter(0.0, 1.0)))


def _sample_variance_log(values: NDArray[np.float64]) -> float:
    """Return the natural logarithm of the unbiased sample variance.

    Scaling before centering prevents avoidable overflow in both the mean and
    squared deviations.  ``-inf`` represents an exactly zero sample variance.
    """
    magnitude = float(np.max(np.abs(values)))
    if magnitude == 0.0:
        return -math.inf

    scaled = values / magnitude
    center = float(np.mean(scaled, dtype=np.float64))
    deviations = scaled - center
    sum_squares = float(np.dot(deviations, deviations))
    if sum_squares == 0.0:
        return -math.inf

    normalized_variance = sum_squares / float(values.size - 1)
    return math.log(normalized_variance) + 2.0 * math.log(magnitude)


def _relative_variance_logs(
    groups: tuple[NDArray[np.float64], ...],
) -> tuple[float, tuple[float, ...]]:
    """Return variance logs after one common normalization of all groups."""
    common_scale = max(float(np.max(np.abs(group))) for group in groups)
    if common_scale == 0.0:
        return (0.0, tuple(-math.inf for _ in groups))
    return (
        common_scale,
        tuple(_sample_variance_log(group / common_scale) for group in groups),
    )


def _exp_extended(log_value: float) -> float:
    """Exponentiate to binary64, returning the appropriate finite boundary."""
    if log_value == -math.inf or log_value < _LOG_FLOAT_MIN:
        return 0.0
    if log_value > _LOG_FLOAT_MAX:
        return math.inf
    return math.exp(log_value)


def _divide_log_quantity(log_numerator: float, denominator: float) -> float:
    """Compute ``exp(log_numerator) / denominator`` without an intermediate."""
    if log_numerator == -math.inf:
        return 0.0
    if denominator == 0.0:
        return math.inf
    if math.isinf(denominator):
        return 0.0
    return _exp_extended(log_numerator - math.log(denominator))


def _tail_probability(
    statistic: float,
    alternative: Alternative,
    *,
    cdf: Callable[[float], float],
    sf: Callable[[float], float],
) -> float:
    """Evaluate a one- or two-sided probability with stable opposite tails."""
    lower = float(cdf(statistic))
    upper = float(sf(statistic))
    if alternative == "less":
        return lower
    if alternative == "greater":
        return upper
    return min(1.0, 2.0 * min(lower, upper))


def _confidence_interval(
    log_estimate: float,
    alternative: Alternative,
    confidence_level: float,
    *,
    ppf: Callable[[float], float],
) -> tuple[float, float]:
    """Invert a positive pivot to form a one- or two-sided interval."""
    alpha = 1.0 - confidence_level
    if alternative == "less":
        return (0.0, _divide_log_quantity(log_estimate, float(ppf(alpha))))
    if alternative == "greater":
        lower = _divide_log_quantity(log_estimate, float(ppf(1.0 - alpha)))
        return (lower, math.inf)

    lower = _divide_log_quantity(log_estimate, float(ppf(1.0 - alpha / 2.0)))
    upper = _divide_log_quantity(log_estimate, float(ppf(alpha / 2.0)))
    return (lower, upper)


def _available_estimates(
    values: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    """Keep estimates representable by the scalar result schema."""
    return tuple((name, value) for name, value in values if math.isfinite(value))


def _variance_alternative(alternative: Alternative, reference: float) -> str:
    formatted = format(reference, ".6g")
    relation = {
        "two-sided": "is not equal to",
        "less": "is less than",
        "greater": "is greater than",
    }[alternative]
    return f"the true variance {relation} {formatted}"


def _ratio_alternative(alternative: Alternative) -> str:
    relation = {
        "two-sided": "is not equal to",
        "less": "is less than",
        "greater": "is greater than",
    }[alternative]
    return f"the ratio of true variances {relation} 1"


def chisquare_1samp(
    x: ArrayLike,
    variance: object = 1.0,
    *,
    alternative: str = "two-sided",
    confidence_level: object = 0.95,
) -> HypothesisTestResult:
    """Test one normal population variance against a positive value.

    Parameters
    ----------
    x
        One-dimensional sample containing at least two finite real values.
    variance
        Positive variance under the null hypothesis.
    alternative
        ``"two-sided"``, ``"less"``, or ``"greater"``.
    confidence_level
        Confidence coefficient for the interval obtained by inverting the
        chi-square pivot.

    Notes
    -----
    Exact chi-square calibration requires independent normal observations.  A
    constant sample is handled as the boundary statistic zero rather than as
    an arithmetic error.

    References
    ----------
    Snedecor, G. W. and Cochran, W. G. (1996). *Statistical Methods*, 8th ed.
    """
    values = validate_1d_sample(x, name="x")
    null_variance = validate_real_scalar(variance, name="variance")
    if null_variance <= 0.0:
        raise ValueError("variance must be greater than 0")
    selected_alternative = validate_alternative(alternative)
    level = validate_confidence_level(confidence_level)

    degrees_of_freedom = float(values.size - 1)
    log_sample_variance = _sample_variance_log(values)
    if log_sample_variance == -math.inf:
        log_statistic = -math.inf
    else:
        log_statistic = (
            math.log(degrees_of_freedom) + log_sample_variance - math.log(null_variance)
        )
    statistic = _exp_extended(log_statistic)
    pvalue = _tail_probability(
        statistic,
        selected_alternative,
        cdf=lambda value: float(stats.chi2.cdf(value, degrees_of_freedom)),
        sf=lambda value: float(stats.chi2.sf(value, degrees_of_freedom)),
    )

    log_interval_numerator = (
        -math.inf
        if log_sample_variance == -math.inf
        else math.log(degrees_of_freedom) + log_sample_variance
    )
    interval = _confidence_interval(
        log_interval_numerator,
        selected_alternative,
        level,
        ppf=lambda probability: float(stats.chi2.ppf(probability, degrees_of_freedom)),
    )
    sample_variance = _exp_extended(log_sample_variance)

    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="One-sample chi-square test for variance",
        alternative=_variance_alternative(selected_alternative, null_variance),
        data_name="x",
        statistic_name="X-squared",
        calibration="exact chi-square under normality",
        df=degrees_of_freedom,
        confidence_interval=interval,
        confidence_level=level,
        estimates=_available_estimates((("sample variance", sample_variance),)),
    )


def f_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    alternative: str = "two-sided",
    confidence_level: object = 0.95,
) -> HypothesisTestResult:
    """Test equality of two normal population variances with an F ratio.

    The statistic is the unbiased sample variance of ``x`` divided by that of
    ``y``.  Both sample variances must be strictly positive; constant samples
    are rejected explicitly because the ratio and its confidence interval are
    then degenerate.

    References
    ----------
    Snedecor, G. W. and Cochran, W. G. (1996). *Statistical Methods*, 8th ed.
    """
    first = validate_1d_sample(x, name="x")
    second = validate_1d_sample(y, name="y")
    selected_alternative = validate_alternative(alternative)
    level = validate_confidence_level(confidence_level)

    common_scale, relative_logs = _relative_variance_logs((first, second))
    relative_first_variance, relative_second_variance = relative_logs
    if relative_first_variance == -math.inf or relative_second_variance == -math.inf:
        raise ValueError("f_2samp requires positive sample variance in x and y")

    first_df = float(first.size - 1)
    second_df = float(second.size - 1)
    log_common_square = 2.0 * math.log(common_scale)
    log_first_variance = relative_first_variance + log_common_square
    log_second_variance = relative_second_variance + log_common_square
    log_ratio = relative_first_variance - relative_second_variance
    statistic = _exp_extended(log_ratio)
    pvalue = _tail_probability(
        statistic,
        selected_alternative,
        cdf=lambda value: float(stats.f.cdf(value, first_df, second_df)),
        sf=lambda value: float(stats.f.sf(value, first_df, second_df)),
    )
    interval = _confidence_interval(
        log_ratio,
        selected_alternative,
        level,
        ppf=lambda probability: float(stats.f.ppf(probability, first_df, second_df)),
    )

    first_variance = _exp_extended(log_first_variance)
    second_variance = _exp_extended(log_second_variance)
    ratio = _exp_extended(log_ratio)
    estimates = _available_estimates(
        (
            ("variance of x", first_variance),
            ("variance of y", second_variance),
            ("ratio of variances", ratio),
        )
    )

    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Two-sample F test for variances",
        alternative=_ratio_alternative(selected_alternative),
        data_name="x and y",
        statistic_name="F",
        calibration="exact F under independent normal sampling",
        df=(first_df, second_df),
        confidence_interval=interval,
        confidence_level=level,
        estimates=estimates,
    )


def bartlett(*samples: ArrayLike) -> HypothesisTestResult:
    """Test homogeneity of normal population variances across groups.

    Bartlett's statistic is especially sensitive to non-normality.  Every
    sample must have positive sample variance; otherwise its logarithmic
    statistic is undefined and a :class:`ValueError` is raised.

    References
    ----------
    Bartlett, M. S. (1937). Properties of sufficiency and statistical tests.
    *Proceedings of the Royal Society A*, 160, 268--282.
    """
    groups = validate_groups(samples)
    _, log_variances = _relative_variance_logs(groups)
    if any(value == -math.inf for value in log_variances):
        raise ValueError("bartlett requires positive sample variance in every sample")

    dfs = np.asarray([group.size - 1 for group in groups], dtype=np.float64)
    total_df = float(np.sum(dfs, dtype=np.float64))
    log_weighted_variances = np.log(dfs) + np.asarray(log_variances)
    log_pooled_variance = float(special.logsumexp(log_weighted_variances)) - math.log(
        total_df
    )

    numerator = total_df * log_pooled_variance - float(
        np.dot(dfs, np.asarray(log_variances))
    )
    numerator = max(0.0, numerator)
    correction = 1.0 + (float(np.sum(1.0 / dfs, dtype=np.float64)) - 1.0 / total_df) / (
        3.0 * float(len(groups) - 1)
    )
    statistic = numerator / correction
    degrees_of_freedom = float(len(groups) - 1)
    pvalue = float(stats.chi2.sf(statistic, degrees_of_freedom))

    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Bartlett's test for homogeneity of variances",
        alternative="at least one population variance differs",
        data_name=f"{len(groups)} samples",
        statistic_name="K-squared",
        calibration="chi-square approximation under normality",
        df=degrees_of_freedom,
    )


def _absolute_deviation_anova(
    samples: tuple[ArrayLike, ...],
    *,
    center: Callable[[NDArray[np.float64]], float],
    method: str,
    calibration: str,
) -> HypothesisTestResult:
    """Apply one-way ANOVA to absolute within-group deviations."""
    groups = validate_groups(samples)
    common_scale = max(float(np.max(np.abs(group))) for group in groups)
    if common_scale == 0.0:
        raise ValueError("the test is undefined when all samples are constant")

    deviations: list[NDArray[np.float64]] = []
    for group in groups:
        normalized = group / common_scale
        group_center = center(normalized)
        deviations.append(np.abs(normalized - group_center))

    sizes = np.asarray([group.size for group in groups], dtype=np.float64)
    group_means = np.asarray(
        [np.mean(group, dtype=np.float64) for group in deviations],
        dtype=np.float64,
    )
    total_size = float(np.sum(sizes, dtype=np.float64))
    grand_mean = float(np.dot(sizes, group_means) / total_size)
    between = float(np.dot(sizes, np.square(group_means - grand_mean)))
    within = float(
        math.fsum(
            float(np.dot(group - group_mean, group - group_mean))
            for group, group_mean in zip(deviations, group_means, strict=True)
        )
    )

    numerator_df = float(len(groups) - 1)
    denominator_df = total_size - float(len(groups))
    if within == 0.0:
        if between == 0.0:
            raise ValueError(
                "the test is undefined when all absolute deviations are identical"
            )
        statistic = math.inf
        pvalue = 0.0
    else:
        statistic = (between / numerator_df) / (within / denominator_df)
        pvalue = float(stats.f.sf(statistic, numerator_df, denominator_df))

    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method=method,
        alternative="at least one population variance differs",
        data_name=f"{len(groups)} samples",
        statistic_name="F",
        calibration=calibration,
        df=(numerator_df, denominator_df),
    )


def levene(*samples: ArrayLike) -> HypothesisTestResult:
    """Run Levene's mean-centered test for homogeneity of variances.

    References
    ----------
    Levene, H. (1960). Robust tests for equality of variances. In
    *Contributions to Probability and Statistics*, 278--292.
    """
    return _absolute_deviation_anova(
        samples,
        center=lambda values: float(np.mean(values, dtype=np.float64)),
        method="Levene's test for homogeneity of variances",
        calibration="F approximation on absolute mean deviations",
    )


def brown_forsythe(*samples: ArrayLike) -> HypothesisTestResult:
    """Run the Brown--Forsythe median-centered test for equal variances.

    References
    ----------
    Brown, M. B. and Forsythe, A. B. (1974). Robust tests for the equality of
    variances. *Journal of the American Statistical Association*, 69,
    364--367.
    """
    return _absolute_deviation_anova(
        samples,
        center=lambda values: float(np.median(values)),
        method="Brown-Forsythe test for homogeneity of variances",
        calibration="F approximation on absolute median deviations",
    )

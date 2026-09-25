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
_LOG_EPSILON = math.log(float(np.finfo(np.float64).eps))
_TAIL_REEVALUATION_THRESHOLD = float(np.finfo(np.float64).tiny)
_TAIL_SERIES_LIMIT = 100_000


def _sample_variance_log(values: NDArray[np.float64]) -> float:
    """Return the natural logarithm of the unbiased sample variance.

    Subtracting an exactly representable sample anchor before scaling preserves
    within-sample differences near a large common offset.  Opposite-sign
    endpoints can make that subtraction overflow, in which case scaling first
    is the safe fallback.  ``-inf`` represents an exactly zero sample variance.
    """
    anchor = float(np.min(values))
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = values - anchor
    if np.all(np.isfinite(shifted)):
        magnitude = float(np.max(np.abs(shifted)))
        if magnitude == 0.0:
            return -math.inf
        scaled = shifted / magnitude
    else:
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
    """Return variance logs without erasing a much smaller group's spread."""
    common_scale = max(float(np.max(np.abs(group))) for group in groups)
    if common_scale == 0.0:
        return (0.0, tuple(-math.inf for _ in groups))

    relative_logs: list[float] = []
    log_common_square = 2.0 * math.log(common_scale)
    for group in groups:
        anchor = float(np.min(group))
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = group - anchor
        anchored = shifted if np.all(np.isfinite(shifted)) else group
        normalized = anchored / common_scale
        if np.any((anchored != 0.0) & (np.abs(normalized) < np.finfo(np.float64).tiny)):
            # Even partial subnormal rounding can alter the variance before
            # its logarithm is taken.  Preserve an independently scaled log
            # as soon as any nonzero anchored entry loses normal precision.
            relative_log = _sample_variance_log(group) - log_common_square
        else:
            relative_log = _sample_variance_log(normalized)
        relative_logs.append(relative_log)
    return common_scale, tuple(relative_logs)


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
    log_lower: float, log_upper: float, alternative: Alternative
) -> float:
    """Convert log tails only after choosing and, if necessary, doubling one."""
    if alternative == "less":
        return math.exp(log_lower)
    if alternative == "greater":
        return math.exp(log_upper)
    return math.exp(min(0.0, math.log(2.0) + min(log_lower, log_upper)))


def _chi_square_log_tails(log_statistic: float, df: float) -> tuple[float, float]:
    """Retain lower-tail information even when the pivot is below binary64."""
    if log_statistic < _LOG_EPSILON:
        # P(a, z) = z**a / Gamma(a+1) * (1 + O(z)), with z = X²/2.
        # Integrating exp(-t) between exp(-z) and 1 bounds the relative
        # error of this leading term by exp(z)-1 < epsilon/2 here.
        log_lower = 0.5 * df * (log_statistic - math.log(2.0)) - float(
            special.gammaln(0.5 * df + 1.0)
        )
        return log_lower, math.log1p(-math.exp(log_lower))
    statistic = _exp_extended(log_statistic)
    if math.isinf(statistic):
        return 0.0, -math.inf
    log_lower = float(stats.chi2.logcdf(statistic, df))
    log_upper = float(stats.chi2.logsf(statistic, df))
    # SciPy can underflow an intermediate even for representable subnormal
    # probabilities.  Reevaluate before that range with a scaled recurrence.
    if log_lower < math.log(_TAIL_REEVALUATION_THRESHOLD):
        log_lower = _gamma_log_lower_series(0.5 * statistic, 0.5 * df)
        return log_lower, math.log1p(-math.exp(log_lower))
    if log_upper < math.log(_TAIL_REEVALUATION_THRESHOLD):
        log_upper = _chi_square_log_upper_recurrence(0.5 * statistic, 0.5 * df)
        return math.log1p(-math.exp(log_upper)), log_upper
    return log_lower, log_upper


def _gamma_log_lower_series(value: float, shape: float) -> float:
    """Sum the positive lower-gamma series, bounding its remaining terms."""
    term = total = 1.0
    epsilon = float(np.finfo(np.float64).eps)
    for index in range(1, _TAIL_SERIES_LIMIT + 1):
        term *= value / (shape + index)
        total += term
        ratio = value / (shape + index + 1.0)
        if ratio < 1.0 and term * ratio / (1.0 - ratio) <= epsilon * total:
            return (
                shape * math.log(value)
                - value
                - float(special.gammaln(shape + 1.0))
                + math.log(total)
            )
    raise ArithmeticError("lower chi-square tail series did not converge")


def _chi_square_log_upper_recurrence(value: float, shape: float) -> float:
    """Use the finite positive recurrence for integer/half-integer shapes.

    Repeatedly apply Q(a,z) = Q(a-1,z) + z**(a-1) exp(-z)/Gamma(a).
    Factoring out the largest term avoids underflow.  The base cases are
    Q(1,z)=exp(-z) and Q(1/2,z)=exp(-z) erfcx(sqrt(z)).
    """
    if shape == 0.5:
        return -value + math.log(float(special.erfcx(math.sqrt(value))))
    log_factor = (
        -value + (shape - 1.0) * math.log(value) - float(special.gammaln(shape))
    )
    term = total = 1.0
    remaining = shape - 1.0
    epsilon = float(np.finfo(np.float64).eps)
    for _ in range(_TAIL_SERIES_LIMIT):
        if remaining == 0.0:
            return log_factor + math.log(total)
        ratio = remaining / value
        # Remaining ratios decrease.  The half-integer base factor
        # sqrt(pi*z)*erfcx(sqrt(z)) is <= 1, so the geometric remainder
        # is an upper bound in the half-integer case as well.
        if ratio < 1.0 and term * ratio / (1.0 - ratio) <= epsilon * total:
            return log_factor + math.log(total)
        term *= ratio
        if remaining == 0.5:
            term *= (
                math.sqrt(math.pi)
                * math.sqrt(value)
                * float(special.erfcx(math.sqrt(value)))
            )
            return log_factor + math.log(total + term)
        total += term
        remaining -= 1.0
    raise ArithmeticError("upper chi-square tail recurrence did not converge")


def _softplus(value: float) -> float:
    """Evaluate log(1 + exp(value)) without overflowing."""
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _log_beta_normalizer(shape1: float, shape2: float) -> float:
    """Avoid log-gamma cancellation for short integer/half-integer shapes."""
    small, large = sorted((shape1, shape2))
    if large >= 10_000.0 and small <= 32.0:
        log_large = math.log(large)
        if small.is_integer():
            # B(a,m) = Gamma(m) / product(a+j, j=0,...,m-1).
            return math.lgamma(small) - math.fsum(
                log_large + math.log1p(index / large) for index in range(int(small))
            )
        if (2.0 * small).is_integer():
            # Stirling's gamma-ratio expansion gives log B(a,1/2).
            # The next term is 1/(640*a**5), below 1.6e-23 at a=10,000.
            inverse = 1.0 / large
            base = (
                0.5 * (math.log(math.pi) - log_large)
                + inverse / 8.0
                - inverse**3 / 192.0
            )
            # B(a,b+1) = b/(a+b) * B(a,b), starting from b=1/2.
            return base + math.fsum(
                math.log(index + 0.5) - log_large - math.log1p((index + 0.5) / large)
                for index in range(int(small - 0.5))
            )
    return float(special.betaln(shape1, shape2))


def _beta_log_tails(
    log_value: float, shape1: float, shape2: float
) -> tuple[float, float]:
    """Evaluate beta tails using a log argument that is at most log(1/2)."""
    if log_value < _LOG_EPSILON - math.log(max(1.0, shape2)) - math.log(2.0):
        # I_x(a,b) = x**a/(a B(a,b)) times the weighted mean of
        # (1-t)**(b-1), 0 <= t <= x.  The threshold ensures that the log
        # of this factor has magnitude less than epsilon.  This bound also
        # covers b=1/2 and is valid when exp(log_value) underflows entirely.
        log_lower = (
            shape1 * log_value - math.log(shape1) - _log_beta_normalizer(shape1, shape2)
        )
        return log_lower, math.log1p(-math.exp(log_lower))
    value = math.exp(log_value)
    lower = float(special.betainc(shape1, shape2, value))
    upper = float(special.betaincc(shape1, shape2, value))
    if lower < _TAIL_REEVALUATION_THRESHOLD:
        log_lower = _beta_log_lower_series(log_value, shape1, shape2)
        return log_lower, math.log1p(-math.exp(log_lower))
    if upper < _TAIL_REEVALUATION_THRESHOLD:
        log_upper = _beta_log_lower_series(math.log1p(-value), shape2, shape1)
        return math.log1p(-math.exp(log_upper)), log_upper
    # Retain whichever direct tail is smaller.  Some beta implementations
    # lose accuracy evaluating a probability close to one even when the
    # opposite small tail is accurate; log1p preserves their complement.
    if lower <= upper:
        return math.log(lower) if lower > 0.0 else -math.inf, math.log1p(-lower)
    return math.log1p(-upper), math.log(upper) if upper > 0.0 else -math.inf


def _beta_log_lower_series(log_value: float, shape1: float, shape2: float) -> float:
    """Evaluate DLMF 8.17.8 with a positive, geometrically bounded series."""
    value = math.exp(log_value)
    if value > 0.9:
        # In an unbalanced F distribution the small probability can have a
        # beta argument close to one.  Its positive series then approaches
        # a unit term ratio; the lower-tail continued fraction is faster.
        return _beta_log_lower_continued_fraction(log_value, shape1, shape2)
    term = total = 1.0
    epsilon = float(np.finfo(np.float64).eps)
    for index in range(_TAIL_SERIES_LIMIT):
        term *= value * (shape1 + shape2 + index) / (shape1 + 1.0 + index)
        total += term
        # Ratios approach value monotonically.  The maximum of the next
        # ratio and value therefore bounds every subsequent ratio.
        ratio = max(
            value,
            value * (shape1 + shape2 + index + 1.0) / (shape1 + index + 2.0),
        )
        if ratio < 1.0 and term * ratio / (1.0 - ratio) <= epsilon * total:
            return (
                shape1 * log_value
                + shape2 * math.log1p(-value)
                - math.log(shape1)
                - _log_beta_normalizer(shape1, shape2)
                + math.log(total)
            )
    raise ArithmeticError("F tail series did not converge")


def _beta_log_lower_continued_fraction(
    log_value: float, shape1: float, shape2: float
) -> float:
    """Evaluate the small beta tail by DLMF 8.17.22--23 and modified Lentz."""
    value = math.exp(log_value)
    complement = -math.expm1(log_value)
    # This equals 1 - (a+b)*x/(a+1), retaining 1-x near the endpoint.
    denominator = complement + value * (1.0 - shape2) / (shape1 + 1.0)
    reciprocal = 1.0 / denominator
    numerator = 1.0
    fraction = reciprocal
    floor = float(np.finfo(np.float64).tiny / np.finfo(np.float64).eps)
    epsilon = float(np.finfo(np.float64).eps)
    stable_iterations = 0
    for index in range(1, _TAIL_SERIES_LIMIT + 1):
        twice_index = 2.0 * index
        coefficients = (
            index
            * (shape2 - index)
            * value
            / ((shape1 + twice_index - 1.0) * (shape1 + twice_index)),
            -(shape1 + index)
            * (shape1 + shape2 + index)
            * value
            / ((shape1 + twice_index) * (shape1 + twice_index + 1.0)),
        )
        previous = fraction
        for coefficient in coefficients:
            denominator = 1.0 + coefficient * reciprocal
            numerator = 1.0 + coefficient / numerator
            if abs(denominator) < floor:
                denominator = math.copysign(floor, denominator)
            if abs(numerator) < floor:
                numerator = math.copysign(floor, numerator)
            reciprocal = 1.0 / denominator
            fraction *= reciprocal * numerator
        if abs(fraction - previous) <= 4.0 * epsilon * abs(fraction):
            stable_iterations += 1
        else:
            stable_iterations = 0
        if stable_iterations >= 3:
            return (
                shape1 * log_value
                + shape2 * math.log(complement)
                - math.log(shape1)
                - _log_beta_normalizer(shape1, shape2)
                + math.log(fraction)
            )
    raise ArithmeticError("F tail continued fraction did not converge")


def _f_log_tails(
    log_statistic: float, first_df: float, second_df: float
) -> tuple[float, float]:
    """Use complementary beta arguments calculated directly from log(F)."""
    logit = log_statistic + math.log(first_df) - math.log(second_df)
    if logit <= 0.0:
        return _beta_log_tails(-_softplus(-logit), 0.5 * first_df, 0.5 * second_df)
    upper, lower = _beta_log_tails(-_softplus(logit), 0.5 * second_df, 0.5 * first_df)
    return lower, upper


def _confidence_interval(
    log_estimate: float,
    alternative: Alternative,
    confidence_level: float,
    *,
    ppf: Callable[[float], float],
    isf: Callable[[float], float],
) -> tuple[float, float]:
    """Invert a positive pivot to form a one- or two-sided interval."""
    alpha = 1.0 - confidence_level
    if alternative == "less":
        return (0.0, _divide_log_quantity(log_estimate, float(ppf(alpha))))
    if alternative == "greater":
        lower = _divide_log_quantity(log_estimate, float(isf(alpha)))
        return (lower, math.inf)

    lower = _divide_log_quantity(log_estimate, float(isf(alpha / 2.0)))
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
    *,
    variance: float = 1.0,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
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
    an arithmetic error.  Tail probabilities retain the log pivot even when
    its displayed value rounds to zero or infinity.

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
    log_lower, log_upper = _chi_square_log_tails(log_statistic, degrees_of_freedom)
    pvalue = _tail_probability(log_lower, log_upper, selected_alternative)

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
        isf=lambda probability: float(stats.chi2.isf(probability, degrees_of_freedom)),
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
    confidence_level: float = 0.95,
) -> HypothesisTestResult:
    """Test equality of two normal population variances with an F ratio.

    The statistic is the unbiased sample variance of ``x`` divided by that of
    ``y``.  Both sample variances must be strictly positive; constant samples
    are rejected explicitly because the ratio and its confidence interval are
    then degenerate.  P-values use the log ratio, so a statistic displayed as
    zero or infinity can still have a positive, representable tail probability.

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
    log_lower, log_upper = _f_log_tails(log_ratio, first_df, second_df)
    pvalue = _tail_probability(log_lower, log_upper, selected_alternative)
    interval = _confidence_interval(
        log_ratio,
        selected_alternative,
        level,
        ppf=lambda probability: float(stats.f.ppf(probability, first_df, second_df)),
        isf=lambda probability: float(stats.f.isf(probability, first_df, second_df)),
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
    normalized_deviations: list[NDArray[np.float64]] = []
    log_scales: list[float] = []
    for group in groups:
        anchor = float(np.min(group))
        with np.errstate(over="ignore", invalid="ignore"):
            shifted = group - anchor
        if np.all(np.isfinite(shifted)):
            group_scale = float(np.max(np.abs(shifted)))
            normalized = (
                np.zeros_like(group) if group_scale == 0.0 else shifted / group_scale
            )
        else:
            group_scale = float(np.max(np.abs(group)))
            normalized = group / group_scale
        group_center = center(normalized)
        group_deviations = np.abs(normalized - group_center)
        deviation_scale = float(np.max(group_deviations))
        if deviation_scale == 0.0:
            normalized_deviations.append(np.zeros_like(group_deviations))
            log_scales.append(-math.inf)
        else:
            normalized_deviations.append(group_deviations / deviation_scale)
            log_scales.append(math.log(group_scale) + math.log(deviation_scale))

    finite_log_scales = [value for value in log_scales if math.isfinite(value)]
    if not finite_log_scales:
        raise ValueError("the test is undefined when all samples are constant")
    common_log_scale = max(finite_log_scales)
    deviations = [
        values * math.exp(log_scale - common_log_scale)
        if math.isfinite(log_scale)
        else np.zeros_like(values)
        for values, log_scale in zip(normalized_deviations, log_scales, strict=True)
    ]

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

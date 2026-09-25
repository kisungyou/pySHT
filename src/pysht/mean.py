"""Tests for univariate and multivariate population means.

The routines in this module implement their statistics directly and use
SciPy only for well-tested distribution functions.  Calculations are carried
out after a positive common rescaling where possible, which preserves every
reported test statistic while avoiding avoidable overflow and underflow.
"""

from __future__ import annotations

import math
import operator
from typing import Final, Literal, SupportsIndex, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

from ._resampling import monte_carlo_calibration, upper_tail_threshold
from ._results import BayesFactorTestResult, HypothesisTestResult, ResamplingTestResult
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
    "bs_1samp",
    "bs_2samp",
    "cph_ksamp",
    "cq_2samp",
    "dempster_1samp",
    "dempster_2samp",
    "hotelling_1samp",
    "hotelling_2samp",
    "johansen_2samp",
    "ky_2samp",
    "li_1samp",
    "li_2samp",
    "li_ksamp",
    "ljw_2samp",
    "maximum_pairwise_bayes_factor_2samp",
    "nvm_2samp",
    "schott_ksamp",
    "sd_1samp",
    "sd_2samp",
    "thulin_2samp",
    "ttest_1samp",
    "ttest_2samp",
    "yao_2samp",
    "zx_ksamp",
]

_T_CALIBRATION: Final = "Student t distribution"
_F_CALIBRATION: Final = "F distribution"
_NORMAL_CALIBRATION: Final = "asymptotic standard normal distribution"


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


def _scaled_centered_values(
    values: NDArray[np.float64], center: NDArray[np.float64] | float
) -> tuple[NDArray[np.float64], float]:
    """Center before scaling, with an overflow-safe opposite-sign fallback."""
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = np.asarray(values - center, dtype=np.float64)
    base_scale = 1.0
    if not np.all(np.isfinite(shifted)):
        center_array = np.asarray(center, dtype=np.float64)
        base_scale = _scale_for(values, center_array)
        # All validated inputs are finite.  Division first therefore keeps an
        # opposite-sign range such as [-1e308, 1e308] representable.
        shifted = np.asarray(values / base_scale - center_array / base_scale)
        if not np.all(np.isfinite(shifted)):
            raise ValueError("the centered observations could not be represented")

    local_scale = _scale_for(shifted)
    maximum = np.finfo(np.float64).max
    if local_scale <= maximum / base_scale:
        return shifted / local_scale, base_scale * local_scale
    # ``base_scale * local_scale`` would overflow, but ``shifted`` already
    # represents the centered data in units of ``base_scale``.
    return shifted, base_scale


def _rescale(value: float, scale: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.float64(value) * np.float64(scale))


def _rescale_square(value: float, scale: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.float64(value) * np.float64(scale) * np.float64(scale))


def _stable_mean(values: NDArray[np.float64]) -> float:
    scale = _scale_for(values)
    return _rescale(float(np.mean(values / scale, dtype=np.float64)), scale)


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
        quantile = float(stats.t.isf((1.0 - confidence_level) / 2.0, df))
        lower_scaled = estimate_scaled - quantile * standard_error_scaled
        upper_scaled = estimate_scaled + quantile * standard_error_scaled
    else:
        quantile = float(stats.t.isf(1.0 - confidence_level, df))
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

    # Form the null-centered observations in the original coordinate system
    # before selecting a numerical scale.  Scaling the raw location first can
    # erase all meaningful variation when ``popmean`` is very large.
    scaled, scale = _scaled_centered_values(values, null)
    mean_scaled, sd_scaled = _scaled_mean_sd(scaled, 1.0)
    if sd_scaled == 0.0:
        raise ValueError("the sample variance must be positive")
    n = values.size
    df = float(n - 1)
    standard_error_scaled = sd_scaled / math.sqrt(n)
    statistic = mean_scaled / standard_error_scaled
    pvalue = _t_pvalue(statistic, df, selected_alternative)
    difference_interval = _t_confidence_interval(
        mean_scaled,
        standard_error_scaled,
        df,
        selected_alternative,
        level,
        scale,
    )
    interval = (
        difference_interval[0]
        if math.isinf(difference_interval[0])
        else null + difference_interval[0],
        difference_interval[1]
        if math.isinf(difference_interval[1])
        else null + difference_interval[1],
    )
    estimate = _stable_mean(values)

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
        estimates=(("mean of x", estimate),),
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

    estimates: tuple[tuple[str, float], ...]
    if is_paired:
        differences_scaled, scale = _scaled_centered_values(first, second)
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
        (first_scaled, second_scaled), scale = _globally_scaled_anchored_groups(
            (first, second)
        )
        first_mean_scaled, first_sd_scaled = _scaled_mean_sd(first_scaled, 1.0)
        second_mean_scaled, second_sd_scaled = _scaled_mean_sd(second_scaled, 1.0)
        difference_scaled = first_mean_scaled - second_mean_scaled
        estimates = (
            ("mean of x", _stable_mean(first)),
            ("mean of y", _stable_mean(second)),
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
    scaled_groups, _ = _globally_scaled_anchored_groups(groups)
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
        (f"mean of sample {index}", _stable_mean(group))
        for index, group in enumerate(groups, start=1)
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


def _residual_svd(
    centered: NDArray[np.float64], *, name: str
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Factor equilibrated residuals without squaring their condition number.

    Each group is centered independently, then columns of the stacked
    residuals are equilibrated. Positive weights on the groups do not change
    this rank, so the check applies to both pooled and mean covariances.
    The usual SVD backward-error threshold depends on dimension and machine
    precision, rather than on the physical units of individual features.
    Inputs have already been scaled to keep centering representable.
    """
    scales = np.max(np.abs(centered), axis=0)
    if np.any(scales == 0.0):
        raise ValueError(f"{name} must be positive definite (dependent features)")
    try:
        left, singular_values, right = np.linalg.svd(
            centered / scales, full_matrices=False
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} rank could not be determined") from exc
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular_values[0]
    if singular_values.size < centered.shape[1] or singular_values[-1] <= tolerance:
        raise ValueError(f"{name} must be positive definite (dependent features)")
    return left, singular_values, right, scales


def _require_full_covariance_rank(*groups: NDArray[np.float64], name: str) -> None:
    centered = np.vstack(
        tuple(group - np.mean(group, axis=0, dtype=np.float64) for group in groups)
    )
    _residual_svd(centered, name=name)


def _hotelling_quadratic(
    difference: NDArray[np.float64],
    *groups: NDArray[np.float64],
    name: str,
) -> float:
    """Evaluate a pooled-covariance quadratic directly from residuals."""
    centered = np.vstack(
        tuple(group - np.mean(group, axis=0, dtype=np.float64) for group in groups)
    )
    _, singular_values, right, scales = _residual_svd(centered, name=name)
    standardized = (right @ (difference / scales)) / singular_values
    within_df = sum(group.shape[0] - 1 for group in groups)
    return float(within_df) * float(standardized @ standardized)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return result


def _random_generator(
    rng: np.random.Generator | int | np.integer | None,
) -> np.random.Generator:
    if isinstance(rng, (bool, np.bool_)):
        raise TypeError("rng must be None, an integer seed, or a Generator")
    if rng is not None and not isinstance(rng, (np.random.Generator, int, np.integer)):
        raise TypeError("rng must be None, an integer seed, or a Generator")
    return np.random.default_rng(rng)


def _choice(value: object, *, name: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower().replace("_", "-")
    if normalized not in choices:
        options = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {options}")
    return normalized


def _multivariate_pair(
    x: ArrayLike, y: ArrayLike, *, minimum_rows: int = 2
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    first = validate_2d_sample(x, name="x", minimum_rows=minimum_rows)
    second = validate_2d_sample(y, name="y", minimum_rows=minimum_rows)
    if first.shape[1] != second.shape[1]:
        raise ValueError("x and y must have the same number of features")
    return first, second


def _lexicographically_sorted_rows(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return a canonical row ordering for seeded randomized procedures."""
    # Signed zero is numerically irrelevant but has a different byte
    # representation. Normalize it before constructing canonical keys.
    normalized = np.asarray(values, dtype=np.float64, order="C")
    if np.any(normalized == 0.0):
        normalized = normalized.copy()
        normalized[normalized == 0.0] = 0.0
    keys = tuple(
        normalized[:, column] for column in range(normalized.shape[1] - 1, -1, -1)
    )
    order = np.lexsort(keys)
    return np.ascontiguousarray(normalized[order])


def _canonical_randomization_pair(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Canonicalize pooled order without changing a two-sample statistic."""
    first_sorted = _lexicographically_sorted_rows(first)
    second_sorted = _lexicographically_sorted_rows(second)
    if first_sorted.shape[0] < second_sorted.shape[0]:
        return first_sorted, second_sorted
    if first_sorted.shape[0] > second_sorted.shape[0]:
        return second_sorted, first_sorted
    if _compare_row_major(second_sorted, first_sorted) < 0:
        return second_sorted, first_sorted
    return first_sorted, second_sorted


def _compare_row_major(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> int:
    """Compare equal-shaped finite arrays lexicographically without byte order."""
    first_flat = first.ravel(order="C")
    second_flat = second.ravel(order="C")
    unequal = np.flatnonzero(first_flat != second_flat)
    if unequal.size == 0:
        return 0
    index = int(unequal[0])
    return -1 if first_flat[index] < second_flat[index] else 1


def _multivariate_groups(
    samples: tuple[ArrayLike, ...], *, minimum_rows: int = 2
) -> tuple[NDArray[np.float64], ...]:
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    groups = tuple(
        validate_2d_sample(
            sample,
            name=f"samples[{index}]",
            minimum_rows=minimum_rows,
        )
        for index, sample in enumerate(samples)
    )
    feature_count = groups[0].shape[1]
    if any(group.shape[1] != feature_count for group in groups[1:]):
        raise ValueError("all samples must have the same number of features")
    return groups


def _feature_anchored_groups(
    groups: tuple[NDArray[np.float64], ...],
) -> tuple[tuple[NDArray[np.float64], ...], float]:
    """Remove one shared feature-wise origin in the input coordinate system.

    Every multivariate mean statistic below is invariant to adding the same
    vector to every observation.  Removing an observed vector before choosing
    a numerical scale preserves that invariance when the common location is
    many orders of magnitude larger than the within-group variation.  The
    feature-wise minimum over all groups makes the choice deterministic under
    row and group permutations.
    """
    # A feature-wise minimum is deterministic under row and group
    # permutations.  Using the first observation would make the last few bits
    # of a centered computation depend on input order.
    origin = np.minimum.reduce(tuple(np.min(group, axis=0) for group in groups))
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = tuple(np.asarray(group - origin) for group in groups)
    if all(np.all(np.isfinite(group)) for group in shifted):
        return shifted, 1.0

    base_scale = _scale_for(*groups)
    normalized_origin = origin / base_scale
    shifted = tuple(
        np.asarray(group / base_scale - normalized_origin) for group in groups
    )
    if not all(np.all(np.isfinite(group)) for group in shifted):
        raise ValueError("the shared-centered observations could not be represented")
    return shifted, base_scale


def _globally_scaled_anchored_groups(
    groups: tuple[NDArray[np.float64], ...], *, extra_scale: float = 0.0
) -> tuple[tuple[NDArray[np.float64], ...], float]:
    shifted, base_scale = _feature_anchored_groups(groups)
    normalized_extra = extra_scale / base_scale
    local_scale = _scale_for(*shifted, scalars=(normalized_extra,))
    maximum = np.finfo(np.float64).max
    if local_scale <= maximum / base_scale:
        return tuple(group / local_scale for group in shifted), base_scale * local_scale
    return shifted, base_scale


def _feature_scaled_anchored_groups(
    groups: tuple[NDArray[np.float64], ...],
) -> tuple[tuple[NDArray[np.float64], ...], NDArray[np.float64]]:
    origin = np.minimum.reduce(tuple(np.min(group, axis=0) for group in groups))
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = tuple(np.asarray(group - origin) for group in groups)
    overflow_columns = np.logical_or.reduce(
        tuple(np.any(~np.isfinite(group), axis=0) for group in shifted)
    )
    base_scale = np.ones(origin.size, dtype=np.float64)
    if np.any(overflow_columns):
        # Only the overflowing columns need division before subtraction.
        # Applying a global fallback can erase another column's small but
        # representable variation around its own large location.
        maxima = np.maximum.reduce(
            tuple(np.max(np.abs(group), axis=0) for group in groups)
        )
        base_scale[overflow_columns] = maxima[overflow_columns]
        for raw, normalized in zip(groups, shifted, strict=True):
            normalized[:, overflow_columns] = (
                raw[:, overflow_columns] / base_scale[overflow_columns]
                - origin[overflow_columns] / base_scale[overflow_columns]
            )
    feature_scale = np.maximum.reduce(
        tuple(np.max(np.abs(group), axis=0) for group in shifted)
    )
    feature_scale[feature_scale == 0.0] = 1.0
    scaled = tuple(group / feature_scale for group in shifted)
    with np.errstate(over="ignore", invalid="ignore"):
        raw_feature_scale = feature_scale * base_scale
    return scaled, raw_feature_scale


def _null_mean(popmean: ArrayLike | None, feature_count: int) -> NDArray[np.float64]:
    if popmean is None:
        return np.zeros(feature_count, dtype=np.float64)
    result = validate_1d_sample(popmean, name="popmean", minimum_size=1)
    if result.size != feature_count:
        raise ValueError("popmean must contain one value per feature")
    return result


def _sample_covariance(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = values - np.mean(values, axis=0, dtype=np.float64)
    return np.asarray(
        centered.T @ centered / (values.shape[0] - 1),
        dtype=np.float64,
    )


def _cross_product_trace_square(values: NDArray[np.float64]) -> float:
    """Return ``tr((values.T @ values) ** 2)`` using the smaller Gram."""
    row_count, feature_count = values.shape
    gram = values.T @ values if feature_count < row_count else values @ values.T
    return float(np.sum(gram * gram, dtype=np.float64))


def _covariance_trace_moments(
    values: NDArray[np.float64],
) -> tuple[float, float]:
    """Return ``tr(S)`` and ``tr(S @ S)`` using the smaller Gram matrix."""
    centered = values - np.mean(values, axis=0, dtype=np.float64)
    within_df = float(values.shape[0] - 1)
    trace = float(np.sum(centered * centered, dtype=np.float64)) / within_df
    trace_squared = _cross_product_trace_square(centered) / within_df**2
    return trace, trace_squared


def _trace_sample_covariance_product(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float:
    """Return ``tr(S_first @ S_second)`` using the cheaper exact product."""
    first_centered = first - np.mean(first, axis=0, dtype=np.float64)
    second_centered = second - np.mean(second, axis=0, dtype=np.float64)
    feature_count = first.shape[1]
    # A feature-space calculation costs O((n1+n2)p^2), versus
    # O(n1*n2*p) for the rectangular row Gram. Select by that operation count
    # so both tall-low-dimensional and short-high-dimensional inputs remain
    # practical.
    if feature_count * (first.shape[0] + second.shape[0]) <= (
        first.shape[0] * second.shape[0]
    ):
        first_product = first_centered.T @ first_centered
        second_product = second_centered.T @ second_centered
        numerator = float(np.sum(first_product * second_product.T, dtype=np.float64))
    else:
        cross = first_centered @ second_centered.T
        numerator = float(np.sum(cross * cross, dtype=np.float64))
    return float(numerator / ((first.shape[0] - 1) * (second.shape[0] - 1)))


def _pooled_diagonal_and_correlation_trace_square(
    groups: tuple[NDArray[np.float64], ...],
) -> tuple[NDArray[np.float64], float]:
    """Return a pooled covariance diagonal and ``tr(R @ R)`` via row Grams."""
    centered = tuple(
        group - np.mean(group, axis=0, dtype=np.float64) for group in groups
    )
    within_df = float(sum(group.shape[0] - 1 for group in groups))
    diagonal_numerator = np.sum(
        np.stack(
            [np.sum(group * group, axis=0, dtype=np.float64) for group in centered]
        ),
        axis=0,
        dtype=np.float64,
    )
    diagonal = np.asarray(diagonal_numerator / within_df, dtype=np.float64)
    if np.any(diagonal <= 0.0):
        raise ValueError("every feature must have positive pooled variance")

    standardized = np.vstack(tuple(group / np.sqrt(diagonal) for group in centered))
    trace_r2 = _cross_product_trace_square(standardized) / within_df**2
    return diagonal, trace_r2


def _trace_square(matrix: NDArray[np.float64]) -> float:
    return float(np.sum(matrix * matrix.T, dtype=np.float64))


def _require_positive(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _require_nonnegative(value: float, *, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _solve_positive_definite(
    matrix: NDArray[np.float64], rhs: NDArray[np.float64], *, name: str
) -> NDArray[np.float64]:
    try:
        factor = np.linalg.cholesky(matrix)
        return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc


def _mean_result(
    *,
    statistic: float,
    pvalue: float,
    method: str,
    statistic_name: str = "Z",
    calibration: str = _NORMAL_CALIBRATION,
    data_name: str = "x",
    df: float | tuple[float, ...] | None = None,
    alternative: str = "true mean vectors differ",
    diagnostics: tuple[tuple[str, bool | int | float | str], ...] = (),
) -> HypothesisTestResult:
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method=method,
        alternative=alternative,
        data_name=data_name,
        statistic_name=statistic_name,
        calibration=calibration,
        diagnostics=diagnostics,
        df=df,
    )


def _pairwise_inner_products(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``x_i.T @ x_j`` for every unordered pair ``i < j``."""
    rows = values.shape[0]
    indices = np.triu_indices(rows, k=1)
    gram = values @ values.T
    return np.asarray(gram[indices], dtype=np.float64)


def _li_studentized_result(
    pairwise_products: NDArray[np.float64],
    *,
    method: str,
    data_name: str,
    alternative: str,
) -> HypothesisTestResult:
    """Studentize Li's asymptotically independent pairwise products."""
    pair_count = pairwise_products.size
    if pair_count < 2:
        raise ValueError("Li's test requires at least three observations")
    pair_mean = float(np.mean(pairwise_products, dtype=np.float64))
    centered = pairwise_products - pair_mean
    pair_variance = float(np.dot(centered, centered)) / (pair_count - 1)
    _require_positive(pair_variance, name="Li pairwise-product variance")
    statistic = pair_mean / math.sqrt(pair_variance / pair_count)
    degrees = float(pair_count - 1)
    return _mean_result(
        statistic=statistic,
        pvalue=float(stats.t.sf(statistic, degrees)),
        method=method,
        statistic_name="t",
        calibration="asymptotic Student t distribution as dimension diverges",
        data_name=data_name,
        df=degrees,
        alternative=alternative,
        diagnostics=(("pairwise products", pair_count),),
    )


def li_1samp(x: ArrayLike, *, popmean: ArrayLike | None = None) -> HypothesisTestResult:
    """Perform Li's fixed-small-sample high-dimensional one-sample test.

    The procedure treats the inner products from all unordered observation
    pairs as an asymptotically independent univariate sample as dimension
    diverges.  Its Student calibration requires at least three observations
    and the factor-model and trace conditions in Li (2023).
    """
    values = validate_2d_sample(x, name="x", minimum_rows=3)
    null = _null_mean(popmean, values.shape[1])
    scaled, _ = _scaled_centered_values(values, null)
    return _li_studentized_result(
        _pairwise_inner_products(scaled),
        method="Li fixed-small-sample one-sample mean test (2023)",
        data_name="x",
        alternative="true mean vector differs from popmean",
    )


def _li_two_sample_vectors(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Construct the Scheffe difference vectors in Li (2023), Equation (6)."""
    if first.shape[0] > second.shape[0]:
        first, second = second, first
    smaller_size = first.shape[0]
    larger_size = second.shape[0]
    ratio = math.sqrt(smaller_size / larger_size)
    partial = second[:smaller_size]
    return np.asarray(
        first
        - ratio * partial
        + ratio * np.mean(partial, axis=0, dtype=np.float64)
        - np.mean(second, axis=0, dtype=np.float64),
        dtype=np.float64,
    )


def li_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform Li's fixed-small-sample high-dimensional two-sample test.

    The defining Scheffe construction pairs the smaller sample with the first
    rows of the larger sample.  Row order is consequently part of the realized
    statistic and must be unrelated to the measurements.
    """
    first, second = _multivariate_pair(x, y, minimum_rows=3)
    (first_scaled, second_scaled), _ = _globally_scaled_anchored_groups((first, second))
    differences = _li_two_sample_vectors(first_scaled, second_scaled)
    return _li_studentized_result(
        _pairwise_inner_products(differences),
        method="Li fixed-small-sample two-sample mean test (2023)",
        data_name="x and y",
        alternative="true mean vectors differ",
    )


def li_ksamp(*samples: ArrayLike) -> HypothesisTestResult:
    """Perform Li's fixed-small-sample high-dimensional ANOVA test.

    The first smallest group in caller order is the paper's reference
    population. Each other group is paired with that reference through the
    paper's Scheffe construction, so row order is part of the realized
    statistic. Li's theorem permits non-strictly ordered sample sizes. When
    several groups tie for the minimum, changing which tied group is supplied
    first can therefore change the realized statistic, while the theorem's
    null calibration remains valid for every fixed reference choice.
    """
    groups = _multivariate_groups(samples, minimum_rows=3)
    sizes = tuple(group.shape[0] for group in groups)
    minimum_size = min(sizes)
    reference_index = sizes.index(minimum_size)
    ordered = (groups[reference_index],) + tuple(
        group for index, group in enumerate(groups) if index != reference_index
    )
    scaled, _ = _globally_scaled_anchored_groups(ordered)
    reference = scaled[0]
    product_rows = tuple(
        _pairwise_inner_products(_li_two_sample_vectors(reference, group))
        for group in scaled[1:]
    )
    pairwise_sum = np.fromiter(
        (
            math.fsum(float(value) for value in coordinate)
            for coordinate in zip(*product_rows, strict=True)
        ),
        dtype=np.float64,
        count=product_rows[0].size,
    )
    return _li_studentized_result(
        pairwise_sum,
        method="Li fixed-small-sample high-dimensional ANOVA mean test (2023)",
        data_name="samples",
        alternative="at least one true mean vector differs",
    )


def _trace_covariance_square_u(values: NDArray[np.float64]) -> float:
    """Li--Chen (2012), Equation (2.1), unbiased covariance-square trace.

    This translation-invariant U-statistic is not the different finite-sample
    leave-two-out expression displayed in Chen--Qin (2010).
    """
    n = values.shape[0]
    gram = values @ values.T
    np.fill_diagonal(gram, 0.0)
    first = float(np.sum(gram * gram, dtype=np.float64))
    row_sums = np.sum(gram, axis=1, dtype=np.float64)
    row_squares = np.sum(gram * gram, axis=1, dtype=np.float64)
    second = float(np.sum(row_sums * row_sums - row_squares, dtype=np.float64))
    total = float(np.sum(row_sums, dtype=np.float64))
    complement = total - 2.0 * row_sums[:, None] - 2.0 * row_sums[None, :] + 2.0 * gram
    third = float(np.sum(gram * complement, dtype=np.float64))
    nf = float(n)
    return (
        first / (nf * (nf - 1.0))
        - 2.0 * second / (nf * (nf - 1.0) * (nf - 2.0))
        + third / (nf * (nf - 1.0) * (nf - 2.0) * (nf - 3.0))
    )


def _trace_cross_covariance_u(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float:
    """Literal ordered-sum estimator of ``tr(Sigma1 @ Sigma2)``."""
    n1 = first.shape[0]
    n2 = second.shape[0]
    cross = first @ second.T
    squared = cross * cross
    rows = np.sum(cross, axis=1, dtype=np.float64)
    columns = np.sum(cross, axis=0, dtype=np.float64)
    term1 = float(np.sum(squared, dtype=np.float64)) / (n1 * n2)
    term2 = float(
        np.sum(columns * columns - np.sum(squared, axis=0), dtype=np.float64)
    ) / (n1 * n2 * (n1 - 1))
    term3 = float(np.sum(rows * rows - np.sum(squared, axis=1), dtype=np.float64)) / (
        n1 * n2 * (n2 - 1)
    )
    total = float(np.sum(cross, dtype=np.float64))
    complement = total - rows[:, None] - columns[None, :] + cross
    term4 = float(np.sum(cross * complement, dtype=np.float64)) / (
        n1 * n2 * (n1 - 1) * (n2 - 1)
    )
    return float(term1 - term2 - term3 + term4)


def _cq_unscaled_statistic(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float:
    """Evaluate Chen--Qin Equation (2.1) without diagonal Gram terms."""
    n1 = first.shape[0]
    n2 = second.shape[0]
    sum1 = np.sum(first, axis=0, dtype=np.float64)
    sum2 = np.sum(second, axis=0, dtype=np.float64)
    within1 = (
        float(np.dot(sum1, sum1)) - float(np.sum(first * first, dtype=np.float64))
    ) / (n1 * (n1 - 1))
    within2 = (
        float(np.dot(sum2, sum2)) - float(np.sum(second * second, dtype=np.float64))
    ) / (n2 * (n2 - 1))
    cross = 2.0 * float(np.dot(sum1, sum2)) / (n1 * n2)
    return float(within1 + within2 - cross)


def cq_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the Chen--Qin dense high-dimensional two-sample mean test.

    The two covariance matrices may differ. The Chen--Qin statistic is
    studentized using the translation-invariant unbiased trace estimators
    from Li--Chen (2012), Equations (2.1)--(2.2), rather than Chen--Qin's
    different finite-sample leave-two-out estimate. At least four observations
    are required in each group. The asymptotic normal calibration requires
    the moment, factor, balance, and trace conditions in the validation ledger.
    """
    first, second = _multivariate_pair(x, y, minimum_rows=4)
    (first_scaled, second_scaled), _ = _globally_scaled_anchored_groups((first, second))
    statistic_unscaled = _cq_unscaled_statistic(first_scaled, second_scaled)
    first_centered = first_scaled - np.mean(first_scaled, axis=0, dtype=np.float64)
    second_centered = second_scaled - np.mean(second_scaled, axis=0, dtype=np.float64)
    trace1 = _trace_covariance_square_u(first_centered)
    trace2 = _trace_covariance_square_u(second_centered)
    cross_trace = _trace_cross_covariance_u(first_centered, second_centered)
    n1 = first.shape[0]
    n2 = second.shape[0]
    variance = (
        2.0 * trace1 / (n1 * (n1 - 1))
        + 2.0 * trace2 / (n2 * (n2 - 1))
        + 4.0 * cross_trace / (n1 * n2)
    )
    _require_positive(variance, name="Chen-Qin null variance estimate")
    statistic = statistic_unscaled / math.sqrt(variance)
    return _mean_result(
        statistic=statistic,
        pvalue=float(stats.norm.sf(statistic)),
        method="Chen-Qin two-sample high-dimensional mean test (2010)",
        statistic_name="CQ",
        data_name="x and y",
        diagnostics=(
            ("variance estimator", "Li-Chen (2012) unbiased trace U-statistics"),
        ),
    )


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

    (scaled_values, scaled_null), _ = _feature_scaled_anchored_groups(
        (values, null[None, :])
    )
    scaled = scaled_values - scaled_null
    difference_scaled = np.mean(scaled, axis=0, dtype=np.float64)
    quadratic = _hotelling_quadratic(
        difference_scaled, scaled, name="the covariance estimate"
    )
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
        (first_scaled, second_scaled), _ = _feature_scaled_anchored_groups(
            (first, second)
        )
        one_sample = hotelling_1samp(first_scaled - second_scaled)
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

    (first_scaled, second_scaled), _ = _feature_scaled_anchored_groups((first, second))
    first_mean = np.mean(first_scaled, axis=0, dtype=np.float64)
    second_mean = np.mean(second_scaled, axis=0, dtype=np.float64)
    difference = first_mean - second_mean
    quadratic = _hotelling_quadratic(
        difference, first_scaled, second_scaled, name="the pooled covariance estimate"
    )
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


def _dempster_degrees_of_freedom(
    trace: float, trace_squared: float, within_df: int
) -> tuple[float, float]:
    if within_df <= 1:
        raise ValueError(
            "Dempster's calibration requires at least two within-group degrees of freedom"
        )
    trace = _require_positive(trace, name="covariance trace")
    correction = trace_squared - trace * trace / within_df
    estimate_trace_sigma_squared = (
        within_df * within_df / ((within_df - 1) * (within_df + 2)) * correction
    )
    _require_positive(
        estimate_trace_sigma_squared,
        name="estimated squared-covariance trace",
    )
    effective_rank = trace * trace / estimate_trace_sigma_squared
    numerator_df = float(math.floor(effective_rank))
    denominator_df = float(math.floor(within_df * effective_rank))
    if numerator_df < 1.0 or denominator_df < 1.0:
        raise ValueError("Dempster's estimated degrees of freedom are undefined")
    return numerator_df, denominator_df


def dempster_1samp(
    x: ArrayLike, *, popmean: ArrayLike | None = None
) -> HypothesisTestResult:
    """Perform Dempster's one-sample non-exact mean-vector test.

    The test uses a Euclidean mean-square ratio and an approximate F law with
    trace-estimated effective degrees of freedom.  It assumes multivariate
    normal observations but does not require an invertible covariance matrix.
    """
    values = validate_2d_sample(x, name="x", minimum_rows=3)
    sample_size, feature_count = values.shape
    null = _null_mean(popmean, feature_count)
    scaled, _ = _scaled_centered_values(values, null)
    difference = np.mean(scaled, axis=0, dtype=np.float64)
    trace, trace_squared = _covariance_trace_moments(scaled)
    trace = _require_positive(trace, name="covariance trace")
    statistic = sample_size * float(np.dot(difference, difference)) / trace
    df = _dempster_degrees_of_freedom(trace, trace_squared, sample_size - 1)
    pvalue = float(stats.f.sf(statistic, *df))
    return _mean_result(
        statistic=statistic,
        pvalue=pvalue,
        method="Dempster one-sample non-exact mean test (1958, 1960)",
        statistic_name="F",
        calibration="Dempster approximate F distribution",
        df=df,
    )


def dempster_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform Dempster's equal-covariance two-sample non-exact test.

    Unlike the legacy SHT routine, the positive mean-square ratio is calibrated
    with Dempster's approximate F distribution, not a standard-normal tail.
    """
    first, second = _multivariate_pair(x, y)
    first_size = first.shape[0]
    second_size = second.shape[0]
    within_df = first_size + second_size - 2
    if within_df <= 1:
        raise ValueError(
            "Dempster's calibration requires at least two within-group degrees of freedom"
        )
    (first_scaled, second_scaled), _ = _globally_scaled_anchored_groups((first, second))
    first_trace, first_trace_squared = _covariance_trace_moments(first_scaled)
    second_trace, second_trace_squared = _covariance_trace_moments(second_scaled)
    # For the pooled squared trace, the cross term is evaluated from a
    # rectangular row Gram rather than a dense feature covariance.
    cross_trace = _trace_sample_covariance_product(first_scaled, second_scaled)
    first_weight = (first_size - 1) / within_df
    second_weight = (second_size - 1) / within_df
    trace = _require_positive(
        first_weight * first_trace + second_weight * second_trace,
        name="pooled covariance trace",
    )
    trace_squared = (
        first_weight**2 * first_trace_squared
        + second_weight**2 * second_trace_squared
        + 2.0 * first_weight * second_weight * cross_trace
    )
    difference = np.mean(first_scaled, axis=0) - np.mean(second_scaled, axis=0)
    multiplier = first_size * second_size / (first_size + second_size)
    statistic = multiplier * float(np.dot(difference, difference)) / trace
    df = _dempster_degrees_of_freedom(trace, trace_squared, within_df)
    pvalue = float(stats.f.sf(statistic, *df))
    return _mean_result(
        statistic=statistic,
        pvalue=pvalue,
        method="Dempster two-sample non-exact mean test (1958, 1960)",
        statistic_name="F",
        calibration="Dempster approximate F distribution",
        data_name="x and y",
        df=df,
    )


def _bs_standardized_statistic(
    difference: NDArray[np.float64],
    *,
    trace: float,
    trace_squared: float,
    mean_multiplier: float,
    within_df: int,
) -> float:
    if within_df <= 1:
        raise ValueError(
            "Bai-Saranadasa calibration requires at least two within-group degrees of freedom"
        )
    correction = trace_squared - trace * trace / within_df
    variance = (
        2.0
        * within_df
        * (within_df + 1)
        / ((within_df - 1) * (within_df + 2))
        * correction
    )
    _require_positive(variance, name="Bai-Saranadasa variance estimate")
    numerator = mean_multiplier * float(np.dot(difference, difference)) - trace
    return numerator / math.sqrt(variance)


def bs_1samp(x: ArrayLike, *, popmean: ArrayLike | None = None) -> HypothesisTestResult:
    """Perform the one-sample Bai-Saranadasa trace test.

    The denominator uses the unbiased estimator of ``tr(Sigma @ Sigma)``;
    this corrects the missing ``n + 2`` factor in legacy SHT's one-sample code.
    """
    values = validate_2d_sample(x, name="x", minimum_rows=3)
    sample_size, feature_count = values.shape
    null = _null_mean(popmean, feature_count)
    scaled, _ = _scaled_centered_values(values, null)
    difference = np.mean(scaled, axis=0)
    trace, trace_squared = _covariance_trace_moments(scaled)
    statistic = _bs_standardized_statistic(
        difference,
        trace=trace,
        trace_squared=trace_squared,
        mean_multiplier=float(sample_size),
        within_df=sample_size - 1,
    )
    return _mean_result(
        statistic=statistic,
        pvalue=float(stats.norm.sf(statistic)),
        method="Bai-Saranadasa one-sample high-dimensional mean test (1996)",
    )


def bs_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the equal-covariance Bai-Saranadasa two-sample test."""
    first, second = _multivariate_pair(x, y)
    first_size = first.shape[0]
    second_size = second.shape[0]
    within_df = first_size + second_size - 2
    (first_scaled, second_scaled), _ = _globally_scaled_anchored_groups((first, second))
    first_trace, first_trace_squared = _covariance_trace_moments(first_scaled)
    second_trace, second_trace_squared = _covariance_trace_moments(second_scaled)
    cross_trace = _trace_sample_covariance_product(first_scaled, second_scaled)
    first_weight = (first_size - 1) / within_df
    second_weight = (second_size - 1) / within_df
    trace = first_weight * first_trace + second_weight * second_trace
    trace_squared = (
        first_weight**2 * first_trace_squared
        + second_weight**2 * second_trace_squared
        + 2.0 * first_weight * second_weight * cross_trace
    )
    difference = np.mean(first_scaled, axis=0) - np.mean(second_scaled, axis=0)
    statistic = _bs_standardized_statistic(
        difference,
        trace=trace,
        trace_squared=trace_squared,
        mean_multiplier=first_size * second_size / (first_size + second_size),
        within_df=within_df,
    )
    return _mean_result(
        statistic=statistic,
        pvalue=float(stats.norm.sf(statistic)),
        method="Bai-Saranadasa two-sample high-dimensional mean test (1996)",
        data_name="x and y",
    )


def _sd_standardized_statistic(
    difference: NDArray[np.float64],
    *,
    diagonal: NDArray[np.float64],
    trace_r2: float,
    mean_multiplier: float,
    within_df: int,
) -> float:
    if within_df <= 2:
        raise ValueError(
            "Srivastava-Du calibration requires more than two within-group degrees of freedom"
        )
    if np.any(diagonal <= 0.0):
        raise ValueError("every feature must have positive pooled variance")
    feature_count = diagonal.size
    finite_sample = 1.0 + trace_r2 / feature_count**1.5
    variance = 2.0 * (trace_r2 - feature_count**2 / within_df) * finite_sample
    _require_positive(variance, name="Srivastava-Du variance estimate")
    quadratic = float(np.sum(difference * difference / diagonal))
    center = within_df * feature_count / (within_df - 2)
    return float((mean_multiplier * quadratic - center) / math.sqrt(variance))


def sd_1samp(x: ArrayLike, *, popmean: ArrayLike | None = None) -> HypothesisTestResult:
    """Perform the Srivastava-Du one-sample diagonal mean test."""
    values = validate_2d_sample(x, name="x", minimum_rows=4)
    sample_size, feature_count = values.shape
    null = _null_mean(popmean, feature_count)
    # The Srivastava--Du statistic is invariant to a separate positive change
    # of units in every feature.  Apply that invariance before forming sample
    # variances so a valid small-scale feature is not rounded to zero merely
    # because another column is close to the top of the float64 range.
    (scaled_values, scaled_null), _ = _feature_scaled_anchored_groups(
        (values, null[None, :])
    )
    scaled = scaled_values - scaled_null
    diagonal, trace_r2 = _pooled_diagonal_and_correlation_trace_square((scaled,))
    statistic = _sd_standardized_statistic(
        np.mean(scaled, axis=0),
        diagonal=diagonal,
        trace_r2=trace_r2,
        mean_multiplier=float(sample_size),
        within_df=sample_size - 1,
    )
    return _mean_result(
        statistic=statistic,
        pvalue=float(stats.norm.sf(statistic)),
        method="Srivastava-Du one-sample high-dimensional mean test (2008)",
    )


def sd_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the equal-covariance Srivastava-Du two-sample test."""
    first, second = _multivariate_pair(x, y)
    first_size = first.shape[0]
    second_size = second.shape[0]
    within_df = first_size + second_size - 2
    (first_scaled, second_scaled), _ = _feature_scaled_anchored_groups((first, second))
    diagonal, trace_r2 = _pooled_diagonal_and_correlation_trace_square(
        (first_scaled, second_scaled)
    )
    statistic = _sd_standardized_statistic(
        np.mean(first_scaled, axis=0) - np.mean(second_scaled, axis=0),
        diagonal=diagonal,
        trace_r2=trace_r2,
        mean_multiplier=first_size * second_size / (first_size + second_size),
        within_df=within_df,
    )
    return _mean_result(
        statistic=statistic,
        pvalue=float(stats.norm.sf(statistic)),
        method="Srivastava-Du two-sample high-dimensional mean test (2008)",
        data_name="x and y",
    )


def _behrens_fisher_inputs(
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    int,
    int,
]:
    first, second = _multivariate_pair(x, y)
    (first_scaled, second_scaled), _ = _feature_scaled_anchored_groups((first, second))
    first_size = first.shape[0]
    second_size = second.shape[0]
    difference = np.mean(first_scaled, axis=0) - np.mean(second_scaled, axis=0)
    weighted_residuals = np.vstack(
        tuple(
            (group - np.mean(group, axis=0))
            / math.sqrt(group.shape[0] * (group.shape[0] - 1))
            for group in (first_scaled, second_scaled)
        )
    )
    left, singular_values, right, scales = _residual_svd(
        weighted_residuals, name="estimated covariance of the mean difference"
    )
    # The weighted residual Gram is Sx/nx + Sy/ny. Its SVD whitens that
    # covariance directly, preserving every affine-invariant quadratic and
    # trace weight without forming an ill-conditioned normal-equations matrix.
    whitened_difference = (right @ (difference / scales)) / singular_values
    first_mean_covariance = left[:first_size].T @ left[:first_size]
    second_mean_covariance = left[first_size:].T @ left[first_size:]
    total_mean_covariance = np.eye(first.shape[1], dtype=np.float64)
    return (
        whitened_difference,
        first_mean_covariance,
        second_mean_covariance,
        total_mean_covariance,
        whitened_difference.copy(),
        first_size,
        second_size,
    )


def _behrens_fisher_result(
    statistic: float,
    numerator_df: int,
    denominator_df: float,
    adjustment: float,
    *,
    method: str,
) -> HypothesisTestResult:
    if not math.isfinite(denominator_df) or denominator_df <= 0.0:
        raise ValueError("the approximate denominator degrees of freedom are undefined")
    _require_positive(adjustment, name="F-statistic adjustment")
    pvalue = float(
        stats.f.sf(statistic / adjustment, float(numerator_df), denominator_df)
    )
    return _mean_result(
        statistic=statistic,
        pvalue=pvalue,
        method=method,
        statistic_name="T2",
        calibration="approximate F distribution",
        data_name="x and y",
        df=(float(numerator_df), denominator_df),
    )


def yao_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform Yao's approximate multivariate Behrens-Fisher test.

    Both group covariance matrices may differ.  The covariance of the mean
    difference must be positive definite.
    """
    (
        difference,
        first_covariance,
        second_covariance,
        _total_covariance,
        solved,
        first_size,
        second_size,
    ) = _behrens_fisher_inputs(x, y)
    feature_count = difference.size
    statistic = _require_nonnegative(float(np.dot(difference, solved)), name="T2")
    if statistic == 0.0:
        # Yao's direction-dependent degrees of freedom contain T2 in a
        # denominator, but the limiting upper-tail probability at T2 = 0 is
        # unambiguously one.
        return _mean_result(
            statistic=0.0,
            pvalue=1.0,
            method="Yao two-sample multivariate mean test (1965)",
            statistic_name="T2",
            calibration="limit of Yao's approximate F distribution",
            data_name="x and y",
        )
    first_fraction = float(solved @ first_covariance @ solved) / statistic
    second_fraction = float(solved @ second_covariance @ solved) / statistic
    inverse_df = first_fraction * first_fraction / (
        first_size - 1
    ) + second_fraction * second_fraction / (second_size - 1)
    _require_positive(inverse_df, name="Yao degrees-of-freedom denominator")
    v = 1.0 / inverse_df
    denominator_df = v - feature_count + 1.0
    adjustment = v * feature_count / denominator_df
    return _behrens_fisher_result(
        statistic,
        feature_count,
        denominator_df,
        adjustment,
        method="Yao two-sample multivariate mean test (1965)",
    )


def nvm_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the Nel-Van der Merwe multivariate Behrens-Fisher test."""
    first, second = _multivariate_pair(x, y)
    first_size = first.shape[0]
    second_size = second.shape[0]
    feature_count = first.shape[1]

    # T2 itself is affine-invariant, even though Nel--Van der Merwe's
    # trace-based degrees-of-freedom approximation is not.  Evaluate the
    # quadratic form after a harmless diagonal normalization, while retaining
    # one common scale for the trace formula so its unit dependence is not
    # changed.
    difference, _, _, _, solved, _, _ = _behrens_fisher_inputs(first, second)
    statistic = _require_nonnegative(float(np.dot(difference, solved)), name="T2")

    (first_scaled, second_scaled), _ = _globally_scaled_anchored_groups((first, second))
    first_covariance = _sample_covariance(first_scaled) / first_size
    second_covariance = _sample_covariance(second_scaled) / second_size
    total_covariance = first_covariance + second_covariance
    numerator = _trace_square(total_covariance) + float(np.trace(total_covariance)) ** 2
    denominator = (
        _trace_square(first_covariance) + float(np.trace(first_covariance)) ** 2
    ) / (first_size - 1) + (
        _trace_square(second_covariance) + float(np.trace(second_covariance)) ** 2
    ) / (second_size - 1)
    _require_positive(
        denominator, name="Nel-Van der Merwe degrees-of-freedom denominator"
    )
    v = numerator / denominator
    denominator_df = v - feature_count + 1.0
    adjustment = v * feature_count / denominator_df
    return _behrens_fisher_result(
        statistic,
        feature_count,
        denominator_df,
        adjustment,
        method="Nel-Van der Merwe two-sample multivariate mean test (1986)",
    )


def ky_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the Krishnamoorthy-Yu modified Nel-Van der Merwe test."""
    (
        difference,
        first_covariance,
        second_covariance,
        total_covariance,
        solved,
        first_size,
        second_size,
    ) = _behrens_fisher_inputs(x, y)
    feature_count = difference.size
    statistic = _require_nonnegative(float(np.dot(difference, solved)), name="T2")
    identity = np.eye(feature_count, dtype=np.float64)
    total_inverse = _solve_positive_definite(
        total_covariance,
        identity,
        name="estimated covariance of the mean difference",
    )
    first_weight = first_covariance @ total_inverse
    second_weight = second_covariance @ total_inverse
    denominator = (_trace_square(first_weight) + float(np.trace(first_weight)) ** 2) / (
        first_size - 1
    ) + (_trace_square(second_weight) + float(np.trace(second_weight)) ** 2) / (
        second_size - 1
    )
    _require_positive(
        denominator,
        name="Krishnamoorthy-Yu degrees-of-freedom denominator",
    )
    v = feature_count * (feature_count + 1.0) / denominator
    denominator_df = v - feature_count + 1.0
    adjustment = v * feature_count / denominator_df
    return _behrens_fisher_result(
        statistic,
        feature_count,
        denominator_df,
        adjustment,
        method="Krishnamoorthy-Yu two-sample multivariate mean test (2004)",
    )


def johansen_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform Johansen's Welch-James multivariate mean test."""
    first, second = _multivariate_pair(x, y)
    scaled_groups, _ = _feature_scaled_anchored_groups((first, second))
    for group, name in zip(
        scaled_groups,
        ("x covariance contribution", "y covariance contribution"),
        strict=True,
    ):
        _require_full_covariance_rank(group, name=name)
    (
        difference,
        first_covariance,
        second_covariance,
        _total_covariance,
        solved,
        first_size,
        second_size,
    ) = _behrens_fisher_inputs(x, y)
    feature_count = difference.size
    statistic = _require_nonnegative(float(np.dot(difference, solved)), name="T2")
    # In the Sx/nx + Sy/ny whitened coordinates, the Welch--James weights
    # Ci @ (C1+C2)^(-1) are simply Ci. This avoids inverting each group
    # covariance while retaining the same traces and correction.
    first_a = first_covariance
    second_a = second_covariance
    d_value = 0.5 * (
        (_trace_square(first_a) + float(np.trace(first_a)) ** 2) / (first_size - 1)
        + (_trace_square(second_a) + float(np.trace(second_a)) ** 2) / (second_size - 1)
    )
    _require_positive(d_value, name="Johansen correction")
    denominator_df = feature_count * (feature_count + 2.0) / (3.0 * d_value)
    adjustment = feature_count + 2.0 * d_value - 6.0 * d_value / (feature_count + 2.0)
    return _behrens_fisher_result(
        statistic,
        feature_count,
        denominator_df,
        adjustment,
        method="Johansen Welch-James two-sample multivariate mean test (1980)",
    )


def schott_ksamp(*samples: ArrayLike) -> HypothesisTestResult:
    """Perform Schott's high-dimensional one-way MANOVA mean test.

    The error sum-of-products matrix uses ``(n_i - 1) S_i`` for every
    group.  This is the defining MANOVA error matrix; legacy SHT instead used
    ``n_i S_i`` and thereby mismatched its claimed ``N - k`` error degrees of
    freedom.
    """
    groups = _multivariate_groups(samples)
    scaled, scale = _globally_scaled_anchored_groups(groups)
    sizes = np.asarray([group.shape[0] for group in scaled], dtype=np.float64)
    total_size = int(np.sum(sizes))
    group_count = len(scaled)
    error_df = total_size - group_count
    hypothesis_df = group_count - 1
    if error_df <= 1:
        raise ValueError(
            "Schott's calibration requires more than one error degree of freedom"
        )
    means = tuple(np.mean(group, axis=0, dtype=np.float64) for group in scaled)
    centered = tuple(group - mean for group, mean in zip(scaled, means, strict=True))
    stacked_centered = np.vstack(centered)
    trace_error = float(np.sum(stacked_centered * stacked_centered, dtype=np.float64))
    trace_error_squared = _cross_product_trace_square(stacked_centered)
    grand_mean = (
        sum(size * mean for size, mean in zip(sizes, means, strict=True)) / total_size
    )
    trace_hypothesis = math.fsum(
        float(size) * float(np.dot(mean - grand_mean, mean - grand_mean))
        for size, mean in zip(sizes, means, strict=True)
    )
    statistic = (trace_hypothesis / hypothesis_df - trace_error / error_df) / math.sqrt(
        total_size - 1.0
    )
    a_value = (trace_error_squared - trace_error * trace_error / error_df) / (
        (error_df + 2.0) * (error_df - 1.0)
    )
    variance = 2.0 * a_value / (hypothesis_df * error_df)
    _require_positive(variance, name="Schott variance estimate")
    standardized = statistic / math.sqrt(variance)
    return _mean_result(
        statistic=_rescale_square(statistic, scale),
        pvalue=float(stats.norm.sf(standardized)),
        method="Schott high-dimensional one-way MANOVA mean test (2007)",
        statistic_name="Tnp",
        data_name="samples",
        alternative="at least one true mean vector differs",
        diagnostics=(("standardized statistic", standardized),),
    )


def _cph_statistic(groups: tuple[NDArray[np.float64], ...]) -> float:
    sizes = np.asarray([group.shape[0] for group in groups], dtype=np.float64)
    total_size = float(np.sum(sizes))
    means = tuple(np.mean(group, axis=0, dtype=np.float64) for group in groups)
    # Algebraically, the paper's difference of raw Gram sums equals a
    # weighted sum of squared mean contrasts minus centered within-group
    # sums of squares.  Evaluating that identity directly avoids subtracting
    # O(location**2) terms after a large common translation.
    mean_contrasts = 0.0
    for first_index, (first_size, first_mean) in enumerate(
        zip(sizes[:-1], means[:-1], strict=True)
    ):
        for second_size, second_mean in zip(
            sizes[first_index + 1 :],
            means[first_index + 1 :],
            strict=True,
        ):
            difference = first_mean - second_mean
            mean_contrasts += (
                first_size
                * second_size
                / total_size
                * float(np.dot(difference, difference))
            )
    within_correction = 0.0
    for group, size, mean in zip(groups, sizes, means, strict=True):
        centered = group - mean
        within_correction += (
            (total_size - size)
            / (total_size * (size - 1.0))
            * float(np.sum(centered * centered, dtype=np.float64))
        )
    return mean_contrasts - within_correction


def _cph_variance_original(groups: tuple[NDArray[np.float64], ...]) -> float:
    sizes = np.asarray([group.shape[0] for group in groups], dtype=np.float64)
    total_size = float(np.sum(sizes))
    diagonal_term = 0.0
    for group, size in zip(groups, sizes, strict=True):
        # Cao, Park, and He define n_l1 = floor(n_l / 2) + 1 and
        # n_l2 = n_l - n_l1.  The legacy implementation used a different
        # split and then scaled both unbiased covariances as if they were MLEs.
        first_size = group.shape[0] // 2 + 1
        second_size = group.shape[0] - first_size
        if first_size < 2 or second_size < 2:
            raise ValueError(
                "the original CPH variance estimator requires at least five observations per group"
            )
        diagonal_term += (
            size
            * (total_size - size) ** 2
            / (size - 1.0)
            * _trace_sample_covariance_product(group[:first_size], group[first_size:])
        )
    cross_term = 0.0
    for first_index, (first_size, first_group) in enumerate(
        zip(sizes[:-1], groups[:-1], strict=True)
    ):
        for second_size, second_group in zip(
            sizes[first_index + 1 :],
            groups[first_index + 1 :],
            strict=True,
        ):
            cross_term += (
                2.0
                * first_size
                * second_size
                * _trace_sample_covariance_product(first_group, second_group)
            )
    return 2.0 * (diagonal_term + cross_term) / total_size**2


def _cph_variance_hu(groups: tuple[NDArray[np.float64], ...]) -> float:
    sizes = np.asarray([group.shape[0] for group in groups], dtype=np.float64)
    total_size = float(np.sum(sizes))
    diagonal_term = 0.0
    for size, group in zip(sizes, groups, strict=True):
        if size <= 2:
            raise ValueError(
                "the Hu CPH variance estimator requires at least three observations per group"
            )
        trace, trace_squared = _covariance_trace_moments(group)
        diagonal_term += (
            size
            * (total_size - size) ** 2
            * (size - 1.0)
            / ((size + 1.0) * (size - 2.0))
            * (trace_squared - trace * trace / (size - 1.0))
        )
    cross_term = 0.0
    for first_index, (first_size, first_group) in enumerate(
        zip(sizes[:-1], groups[:-1], strict=True)
    ):
        for second_size, second_group in zip(
            sizes[first_index + 1 :],
            groups[first_index + 1 :],
            strict=True,
        ):
            cross_term += (
                2.0
                * first_size
                * second_size
                * _trace_sample_covariance_product(first_group, second_group)
            )
    return 2.0 * (diagonal_term + cross_term) / total_size**2


def cph_ksamp(
    *samples: ArrayLike,
    variance_estimator: Literal["original", "hu"] = "original",
) -> HypothesisTestResult:
    """Perform the Cao-Park-He high-dimensional k-sample mean test.

    ``variance_estimator`` selects the paper's split-sample estimator or the
    Hu estimator.  Both implementations use the current group in every loop;
    legacy SHT accidentally reused the last group in the split-sample branch.
    """
    selected = _choice(
        variance_estimator,
        name="variance_estimator",
        choices=("original", "hu"),
    )
    minimum_rows = 5 if selected == "original" else 3
    groups = _multivariate_groups(samples, minimum_rows=minimum_rows)
    scaled, scale = _globally_scaled_anchored_groups(groups)
    statistic = _cph_statistic(scaled)
    if selected == "original":
        variance = _cph_variance_original(scaled)
    else:
        variance = _cph_variance_hu(scaled)
    _require_positive(variance, name="CPH variance estimate")
    standardized = statistic / math.sqrt(variance)
    estimator_name = "split-sample" if selected == "original" else "Hu"
    return _mean_result(
        statistic=_rescale_square(statistic, scale),
        pvalue=float(stats.norm.sf(standardized)),
        method=(
            "Cao-Park-He high-dimensional k-sample mean test (2019), "
            f"{estimator_name} variance"
        ),
        statistic_name="T",
        data_name="samples",
        alternative="at least one true mean vector differs",
        diagnostics=(
            ("standardized statistic", standardized),
            ("variance estimator", selected),
        ),
    )


def _zx_transformed_sample(
    groups: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    ordered = tuple(sorted(groups, key=lambda group: group.shape[0]))
    reference = ordered[0]
    reference_size = reference.shape[0]
    blocks: list[NDArray[np.float64]] = []
    for group in ordered[1:]:
        full_mean = np.mean(group, axis=0, dtype=np.float64)
        partial = group[:reference_size]
        partial_mean = np.mean(partial, axis=0, dtype=np.float64)
        block = (reference - full_mean) + math.sqrt(reference_size / group.shape[0]) * (
            partial - partial_mean
        )
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def zx_ksamp(
    *samples: ArrayLike,
    base_test: Literal["bai-saranadasa", "hotelling"] = "bai-saranadasa",
) -> HypothesisTestResult:
    """Perform the Zhang-Xu k-sample Behrens-Fisher mean test.

    Scheffe's transformation reduces the unequal-covariance k-sample problem
    to one sample.  ``base_test="bai-saranadasa"`` applies the corrected
    Bai-Saranadasa calibration; ``base_test="hotelling"`` applies the exact
    Gaussian Hotelling calibration when the transformed dimension permits it.

    The paper's transformation uses the first ``n_min`` observations from
    larger groups.  Their row order must therefore be arbitrary with respect
    to the measurements, as it is for an ordinary i.i.d. sample. If groups tie
    for the smallest size, their input order fixes the reference group and is
    part of the realized Scheffe transformation.
    """
    selected = _choice(
        base_test,
        name="base_test",
        choices=("bai-saranadasa", "hotelling"),
    )
    groups = _multivariate_groups(samples)
    scaled, _ = _globally_scaled_anchored_groups(groups)
    transformed = _zx_transformed_sample(scaled)
    if selected == "bai-saranadasa":
        base = bs_1samp(transformed)
        calibration = base.calibration
    else:
        base = hotelling_1samp(transformed)
        calibration = base.calibration
    return _mean_result(
        statistic=base.statistic,
        pvalue=base.pvalue,
        method=f"Zhang-Xu k-sample Behrens-Fisher mean test (2009), {selected}",
        statistic_name=base.statistic_name,
        calibration=calibration or _NORMAL_CALIBRATION,
        data_name="samples",
        df=base.df,
        alternative="at least one true mean vector differs",
        diagnostics=(("base test", selected),),
    )


def _projected_hotelling_statistic(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    projection: NDArray[np.float64],
) -> float:
    # Identical allocations must use identical reduction orders, including
    # when a random permutation only reorders rows within either group.
    first, second = _canonical_randomization_pair(first, second)
    first_size = first.shape[0]
    second_size = second.shape[0]
    # Projection commutes with centering and covariance formation.  Project
    # the observations first so the random-projection test never constructs
    # either sample's ambient p-by-p covariance matrix.
    if projection.shape[0] == projection.shape[1]:
        # An invertible full-dimensional projection leaves Hotelling T2
        # unchanged. Avoid numerically mixing features with disparate units
        # only to invert that same mixing immediately afterward.
        _residual_svd(projection, name="Gaussian projection")
        first_projected, second_projected = first, second
    else:
        first_projected = first @ projection
        second_projected = second @ projection
    difference = np.mean(first_projected, axis=0) - np.mean(second_projected, axis=0)
    quadratic = _hotelling_quadratic(
        difference,
        first_projected,
        second_projected,
        name="projected pooled covariance",
    )
    return float(first_size * second_size / (first_size + second_size) * quadratic)


def ljw_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    calibration: Literal["asymptotic", "monte-carlo"] = "asymptotic",
    n_resamples: int = 999,
    rng: np.random.Generator | int | np.integer | None = None,
) -> HypothesisTestResult | ResamplingTestResult:
    """Perform the Lopes-Jacob-Wainwright random-projection mean test.

    A single Gaussian projection is drawn independently of the data, as in the
    defining procedure.  Permutation calibration, when requested, holds that
    projection fixed for the observed and every permuted statistic. The
    pooled observations are canonicalized before seeded randomization, so row
    reordering or swapping sample labels cannot change a fixed-seed result.

    The conditional F calibration assumes Gaussian sampling with a common
    covariance matrix. Unrestricted permutation calibration is exact under
    the stronger null that the pooled observations are exchangeable; equality
    of means alone is not sufficient when the distributions differ.
    """
    first, second = _multivariate_pair(x, y)
    first, second = _canonical_randomization_pair(first, second)
    selected = _choice(
        calibration,
        name="calibration",
        choices=("asymptotic", "monte-carlo"),
    )
    generator = _random_generator(rng)
    within_df = first.shape[0] + second.shape[0] - 2
    projected_dimension = within_df // 2
    feature_count = first.shape[1]
    if feature_count < projected_dimension:
        raise ValueError("ljw_2samp requires p >= floor((n_x + n_y - 2) / 2)")
    if feature_count == projected_dimension:
        (first_scaled, second_scaled), _ = _feature_scaled_anchored_groups(
            (first, second)
        )
    else:
        (first_scaled, second_scaled), _ = _globally_scaled_anchored_groups(
            (first, second)
        )
    projection = generator.standard_normal((feature_count, projected_dimension))
    observed = _projected_hotelling_statistic(
        first_scaled,
        second_scaled,
        projection,
    )
    denominator_df = float(within_df - projected_dimension + 1)
    if selected == "asymptotic":
        f_statistic = denominator_df / (projected_dimension * within_df) * observed
        pvalue = float(
            stats.f.sf(f_statistic, float(projected_dimension), denominator_df)
        )
        return _mean_result(
            statistic=observed,
            pvalue=pvalue,
            method="Lopes-Jacob-Wainwright random-projection mean test (2011)",
            statistic_name="T2",
            calibration="F distribution conditional on one Gaussian projection",
            data_name="x and y",
            df=(float(projected_dimension), denominator_df),
            diagnostics=(("projected dimension", projected_dimension),),
        )

    resamples = _positive_integer(n_resamples, name="n_resamples")
    combined = np.vstack((first_scaled, second_scaled))
    first_size = first.shape[0]
    exceedances = 0
    for _ in range(resamples):
        order = generator.permutation(combined.shape[0])
        permuted = _projected_hotelling_statistic(
            combined[order[:first_size]],
            combined[order[first_size:]],
            projection,
        )
        exceedances += int(permuted >= upper_tail_threshold(observed))
    pvalue, standard_error, interval = monte_carlo_calibration(
        exceedances,
        resamples,
    )
    return ResamplingTestResult(
        statistic=observed,
        pvalue=pvalue,
        method="Lopes-Jacob-Wainwright random-projection mean test (2011)",
        alternative="true mean vectors differ",
        data_name="x and y",
        statistic_name="T2",
        calibration="Monte Carlo permutation conditional on one Gaussian projection",
        diagnostics=(("projected dimension", projected_dimension),),
        n_resamples=resamples,
        exceedances=exceedances,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
    )


def _subspace_hotelling_statistic(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    subspaces: tuple[NDArray[np.intp], ...],
) -> float:
    first, second = _canonical_randomization_pair(first, second)
    first_size = first.shape[0]
    second_size = second.shape[0]
    multiplier = first_size * second_size / (first_size + second_size)
    statistics = np.empty(len(subspaces), dtype=np.float64)
    for index, columns in enumerate(subspaces):
        first_selected = first[:, columns]
        second_selected = second[:, columns]
        difference = np.mean(first_selected, axis=0) - np.mean(second_selected, axis=0)
        quadratic = _hotelling_quadratic(
            difference,
            first_selected,
            second_selected,
            name="subspace pooled covariance",
        )
        statistics[index] = multiplier * quadratic
    return float(np.mean(statistics))


def thulin_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_subspaces: int = 100,
    n_resamples: int = 999,
    rng: np.random.Generator | int | np.integer | None = None,
) -> ResamplingTestResult:
    """Perform Thulin's random-subspaces two-sample mean test.

    The sampled feature subsets are drawn once and kept fixed throughout the
    permutation test.  This both follows the conditional randomization design
    and removes the auxiliary-randomness bug in legacy SHT. The pooled rows
    and sample labels are canonicalized before seeded randomization.

    Unrestricted permutation calibration is exact only when observations are
    exchangeable under the null. In particular, equality of means without a
    common distribution is not enough for exact finite-sample inference.
    """
    first, second = _multivariate_pair(x, y)
    first, second = _canonical_randomization_pair(first, second)
    subspace_count = _positive_integer(n_subspaces, name="n_subspaces")
    resamples = _positive_integer(n_resamples, name="n_resamples")
    generator = _random_generator(rng)
    feature_count = first.shape[1]
    within_df = first.shape[0] + second.shape[0] - 2
    selected_dimension = within_df // 2
    if selected_dimension > feature_count:
        raise ValueError("thulin_2samp requires p >= floor((n_x + n_y - 2) / 2)")
    (first_scaled, second_scaled), _ = _feature_scaled_anchored_groups((first, second))
    subspaces = tuple(
        np.asarray(
            generator.choice(
                feature_count,
                size=selected_dimension,
                replace=False,
            ),
            dtype=np.intp,
        )
        for _ in range(subspace_count)
    )
    observed = _subspace_hotelling_statistic(
        first_scaled,
        second_scaled,
        subspaces,
    )
    combined = np.vstack((first_scaled, second_scaled))
    first_size = first.shape[0]
    exceedances = 0
    for _ in range(resamples):
        order = generator.permutation(combined.shape[0])
        permuted = _subspace_hotelling_statistic(
            combined[order[:first_size]],
            combined[order[first_size:]],
            subspaces,
        )
        exceedances += int(permuted >= upper_tail_threshold(observed))
    pvalue, standard_error, interval = monte_carlo_calibration(
        exceedances,
        resamples,
    )
    return ResamplingTestResult(
        statistic=observed,
        pvalue=pvalue,
        method="Thulin random-subspaces two-sample mean test (2014)",
        alternative="true mean vectors differ",
        data_name="x and y",
        statistic_name="T2",
        calibration="Monte Carlo permutation conditional on fixed subspaces",
        diagnostics=(
            ("subspace dimension", selected_dimension),
            ("subspaces", subspace_count),
        ),
        n_resamples=resamples,
        exceedances=exceedances,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
    )


def _xy_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_resamples: int = 999,
    rng: np.random.Generator | int | np.integer | None = None,
) -> ResamplingTestResult:
    """Evaluate the validation-blocked Xue--Yao mean test.

    The statistic is the maximum absolute coordinate of the normalized mean
    difference.  Its critical law is estimated by the Gaussian multiplier
    bootstrap in Xue and Yao (2020).  Rows and group labels are canonicalized
    before a seeded bootstrap, making integer-seed results reproducible under
    row reordering and sample exchange without touching global RNG state.
    It remains private because the requested 999-draw Monte Carlo calibration
    is too discrete to satisfy the 0.01 release gate and fresh null simulations
    were conservative in practical regimes.
    """
    first, second = _multivariate_pair(x, y)
    first, second = _canonical_randomization_pair(first, second)
    resamples = _positive_integer(n_resamples, name="n_resamples")
    generator = _random_generator(rng)
    (first_scaled, second_scaled), scale = _globally_scaled_anchored_groups(
        (first, second)
    )
    first_size = first.shape[0]
    second_size = second.shape[0]
    root_ratio = math.sqrt(first_size / second_size)
    observed_scaled = math.sqrt(first_size) * float(
        np.max(
            np.abs(
                np.mean(first_scaled, axis=0, dtype=np.float64)
                - np.mean(second_scaled, axis=0, dtype=np.float64)
            )
        )
    )
    first_centered = first_scaled - np.mean(first_scaled, axis=0, dtype=np.float64)
    second_centered = second_scaled - np.mean(second_scaled, axis=0, dtype=np.float64)
    exceedances = 0
    completed = 0
    # Batching keeps the kernel in optimized matrix multiplication while
    # bounding temporary storage by roughly ``256 * (n_x + n_y + p)`` floats.
    while completed < resamples:
        batch_size = min(256, resamples - completed)
        multipliers = generator.standard_normal((batch_size, first_size + second_size))
        first_sums = (
            multipliers[:, :first_size] @ first_centered / math.sqrt(first_size)
        )
        second_sums = (
            multipliers[:, first_size:] @ second_centered / math.sqrt(second_size)
        )
        bootstrap = np.max(np.abs(first_sums - root_ratio * second_sums), axis=1)
        exceedances += int(np.count_nonzero(bootstrap >= observed_scaled))
        completed += batch_size
    pvalue, standard_error, interval = monte_carlo_calibration(exceedances, resamples)
    return ResamplingTestResult(
        statistic=_rescale(observed_scaled, scale),
        pvalue=pvalue,
        method="Xue-Yao distribution/correlation-free two-sample mean test (2020)",
        alternative="true mean vectors differ",
        data_name="x and y",
        statistic_name="T infinity",
        calibration="Gaussian multiplier bootstrap",
        diagnostics=(("bootstrap draws", resamples),),
        n_resamples=resamples,
        exceedances=exceedances,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
    )


def _numeric_square_matrix(
    value: ArrayLike, *, name: str, size: int
) -> NDArray[np.float64]:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric matrix") from exc
    if raw.dtype == np.dtype(bool) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric values")
    try:
        matrix = np.asarray(raw, dtype=np.float64, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return matrix


def _clime_precision(
    covariance: NDArray[np.float64], sample_size: int
) -> NDArray[np.float64]:
    feature_count = covariance.shape[0]
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0.0):
        raise ValueError(
            "CLIME requires every feature to have positive pooled variance"
        )
    standard_deviation = np.sqrt(diagonal)
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    tuning = math.sqrt(math.log(feature_count) / sample_size)
    identity = np.eye(feature_count, dtype=np.float64)
    constraint_matrix = np.block(
        [
            [correlation, -correlation],
            [-correlation, correlation],
        ]
    )
    objective = np.ones(2 * feature_count, dtype=np.float64)
    estimate = np.empty((feature_count, feature_count), dtype=np.float64)
    for column in range(feature_count):
        target = identity[:, column]
        bounds = np.concatenate((target + tuning, tuning - target))
        fitted = optimize.linprog(
            objective,
            A_ub=constraint_matrix,
            b_ub=bounds,
            bounds=(0.0, None),
            method="highs",
        )
        if not fitted.success or fitted.x is None:
            raise ValueError(f"CLIME optimization failed for column {column}")
        estimate[:, column] = fitted.x[:feature_count] - fitted.x[feature_count:]
    symmetric = np.empty_like(estimate)
    for row in range(feature_count):
        for column in range(feature_count):
            left = estimate[row, column]
            right = estimate[column, row]
            symmetric[row, column] = left if abs(left) <= abs(right) else right
    inverse_sd = 1.0 / standard_deviation
    return symmetric * np.outer(inverse_sd, inverse_sd)


def _adaptive_threshold_precision(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    pooled: NDArray[np.float64],
    delta: float,
) -> NDArray[np.float64]:
    first_size = first.shape[0]
    second_size = second.shape[0]
    effective_size = first_size * second_size / (first_size + second_size)
    feature_count = first.shape[1]
    first_centered = first - np.mean(first, axis=0)
    second_centered = second - np.mean(second, axis=0)
    thresholds = np.zeros_like(pooled)
    for first_column in range(feature_count - 1):
        for second_column in range(first_column + 1, feature_count):
            covariance_entry = pooled[first_column, second_column]
            first_products = (
                first_centered[:, first_column] * first_centered[:, second_column]
                - covariance_entry
            )
            second_products = (
                second_centered[:, first_column] * second_centered[:, second_column]
                - covariance_entry
            )
            theta = (
                float(np.dot(first_products, first_products))
                + float(np.dot(second_products, second_products))
            ) / (first_size + second_size)
            threshold = delta * math.sqrt(
                max(theta, 0.0) * math.log(feature_count) / effective_size
            )
            thresholds[first_column, second_column] = threshold
            thresholds[second_column, first_column] = threshold
    thresholded = pooled * (np.abs(pooled) >= thresholds)
    eigenvalues = np.linalg.eigvalsh(thresholded)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    minimum_allowed = math.sqrt(np.finfo(np.float64).eps) * spectral_scale
    if eigenvalues[0] < minimum_allowed:
        thresholded = thresholded + (minimum_allowed - eigenvalues[0]) * np.eye(
            feature_count
        )
    return _solve_positive_definite(
        thresholded,
        np.eye(feature_count, dtype=np.float64),
        name="adaptive-threshold covariance estimate",
    )


def _clx_pvalue(statistic: float, feature_count: int) -> float:
    centered = (
        statistic - 2.0 * math.log(feature_count) + math.log(math.log(feature_count))
    )
    log_intensity = -0.5 * centered - 0.5 * math.log(math.pi)
    if log_intensity >= math.log(np.finfo(np.float64).max):
        return 1.0
    intensity = math.exp(log_intensity)
    return -math.expm1(-intensity)


def _clx_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    precision: Literal["clime", "adaptive-threshold"] | ArrayLike = "clime",
    delta: float = 2.0,
) -> HypothesisTestResult:
    """Perform the Cai-Liu-Xia maximum-type two-sample mean test.

    ``precision`` may be a positive-definite precision matrix, ``"clime"``,
    or ``"adaptive-threshold"``.  The estimated branches assume a common
    covariance matrix.  The CLIME program is solved column-by-column and
    symmetrized by the standard minimum-magnitude rule.
    """
    first, second = _multivariate_pair(x, y)
    feature_count = first.shape[1]
    if feature_count < 2:
        raise ValueError("_clx_2samp requires at least two features")
    selected_delta = validate_real_scalar(delta, name="delta")
    if selected_delta <= 0.0:
        raise ValueError("delta must be greater than 0")
    (first_scaled, second_scaled), feature_scale = _feature_scaled_anchored_groups(
        (first, second)
    )
    first_size = first.shape[0]
    second_size = second.shape[0]
    pooled = (
        (first_size - 1) * _sample_covariance(first_scaled)
        + (second_size - 1) * _sample_covariance(second_scaled)
    ) / (first_size + second_size)

    if isinstance(precision, str):
        selected = _choice(
            precision,
            name="precision",
            choices=("clime", "adaptive-threshold"),
        )
        if selected == "clime":
            omega = _clime_precision(pooled, first_size + second_size)
            precision_label = "CLIME-estimated precision"
        else:
            omega = _adaptive_threshold_precision(
                first_scaled,
                second_scaled,
                pooled,
                selected_delta,
            )
            precision_label = "adaptive-threshold precision"
    else:
        supplied = _numeric_square_matrix(
            precision,
            name="precision",
            size=feature_count,
        )
        if not np.all(np.isfinite(feature_scale)):
            raise ValueError(
                "supplied precision cannot be transformed because an observed "
                "feature range exceeds the float64 domain"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            omega = supplied * np.outer(feature_scale, feature_scale)
        if not np.all(np.isfinite(omega)):
            raise ValueError(
                "supplied precision cannot be transformed within the float64 domain"
            )
        precision_label = "supplied precision"

    difference = np.mean(first_scaled, axis=0) - np.mean(second_scaled, axis=0)
    transformed_difference = omega @ difference
    if isinstance(precision, str):
        first_transformed = first_scaled @ omega
        second_transformed = second_scaled @ omega
        transformed_covariance = (
            (first_size - 1) * _sample_covariance(first_transformed)
            + (second_size - 1) * _sample_covariance(second_transformed)
        ) / (first_size + second_size)
        variance = np.diag(transformed_covariance)
    else:
        variance = np.diag(omega)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("the transformed marginal variances must be positive")
    effective_size = first_size * second_size / (first_size + second_size)
    statistic = effective_size * float(
        np.max(transformed_difference * transformed_difference / variance)
    )
    return _mean_result(
        statistic=statistic,
        pvalue=_clx_pvalue(statistic, feature_count),
        method=f"Cai-Liu-Xia maximum-type two-sample mean test (2014), {precision_label}",
        statistic_name="CLX",
        calibration="asymptotic type-I extreme-value distribution",
        data_name="x and y",
        diagnostics=(
            ("precision", precision_label),
            ("delta", selected_delta),
        ),
    )


def _lyl_centered_variance(values: NDArray[np.float64]) -> float:
    centered = values - np.mean(values, dtype=np.float64)
    component = float(np.dot(centered, centered)) / values.size
    if component < 0.0 and abs(component) <= 100.0 * np.finfo(np.float64).eps:
        return 0.0
    if not math.isfinite(component) or component < 0.0:
        raise ValueError("Lee-You-Lin variance component could not be evaluated")
    return component


def maximum_pairwise_bayes_factor_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    a0: float = 0.0,
    b0: float = 0.0,
    alpha: float = 2.01,
    gamma: float | None = None,
) -> BayesFactorTestResult:
    """Compute the Lee-You-Lin maximum pairwise Bayes factor for means.

    The returned statistic is the maximum *log* Bayes factor.  Component log
    Bayes factors are retained, and no frequentist p-value or automatic
    evidence threshold is manufactured.  Residual sums of squares use the
    ordinary centered maximum-likelihood variances in Equation (4) of the
    primary paper.  By default, ``gamma=max(n_x+n_y, p)**(-alpha)`` with the
    paper's numerical choice ``alpha=2.01``.  A supplied ``gamma`` overrides
    that rate.  Nonzero ``a0`` or ``b0`` selects pySHT's optional shared
    inverse-gamma extension rather than the published Equation (4).
    """
    first, second = _multivariate_pair(x, y)
    if first.shape[1] < 2:
        raise ValueError(
            "maximum_pairwise_bayes_factor_2samp requires at least two features"
        )
    shape = validate_real_scalar(a0, name="a0")
    prior_scale = validate_real_scalar(b0, name="b0")
    exponent = validate_real_scalar(alpha, name="alpha")
    if shape < 0.0:
        raise ValueError("a0 must be non-negative")
    if prior_scale < 0.0:
        raise ValueError("b0 must be non-negative")
    if exponent <= 0.0:
        raise ValueError("alpha must be greater than 0")
    first_size = first.shape[0]
    second_size = second.shape[0]
    total_size = first_size + second_size
    feature_count = first.shape[1]
    if gamma is None:
        log_gamma = -exponent * math.log(max(total_size, feature_count))
        shrinkage = math.exp(log_gamma)
        if shrinkage == 0.0:
            raise ValueError(
                "alpha produces a gamma below the positive float64 range; "
                "supply gamma explicitly"
            )
        gamma_source = "paper rate"
        gamma_diagnostics: tuple[tuple[str, bool | int | float | str], ...] = (
            ("alpha", exponent),
            ("gamma", shrinkage),
            ("gamma source", gamma_source),
        )
    else:
        shrinkage = validate_real_scalar(gamma, name="gamma")
        if shrinkage <= 0.0:
            raise ValueError("gamma must be greater than 0")
        log_gamma = math.log(shrinkage)
        gamma_source = "explicit"
        gamma_diagnostics = (
            ("gamma", shrinkage),
            ("gamma source", gamma_source),
        )
    prior_data_scale = math.sqrt(prior_scale) if prior_scale > 0.0 else 0.0
    # When ``a0=b0=0`` the Bayes factor is exactly invariant to a common
    # positive change of units.  Do not clamp the working scale to one: doing
    # so would square subnormal observations directly and could turn a
    # nonconstant feature into an artificial zero-variance boundary.
    (first_scaled, second_scaled), scale = _globally_scaled_anchored_groups(
        (first, second), extra_scale=prior_data_scale
    )
    scaled_prior = (prior_scale / scale) / scale
    log_shrinkage = log_gamma - math.log1p(shrinkage)
    log_factors = np.empty(feature_count, dtype=np.float64)
    for column in range(feature_count):
        first_values = first_scaled[:, column]
        second_values = second_scaled[:, column]
        combined = np.concatenate((first_values, second_values))
        numerator = 2.0 * scaled_prior + total_size * _lyl_centered_variance(combined)
        denominator = (
            2.0 * scaled_prior
            + first_size * _lyl_centered_variance(first_values)
            + second_size * _lyl_centered_variance(second_values)
        )
        if numerator == 0.0 and denominator == 0.0:
            raise ValueError(
                "Lee-You-Lin Bayes factors are undefined for a constant feature"
            )
        _require_positive(numerator, name="Lee-You-Lin numerator")
        if denominator == 0.0:
            log_factors[column] = math.inf
        else:
            _require_positive(denominator, name="Lee-You-Lin denominator")
            log_factors[column] = 0.5 * log_shrinkage + (total_size / 2.0 + shape) * (
                math.log(numerator) - math.log(denominator)
            )
    components = tuple(float(value) for value in log_factors)
    return BayesFactorTestResult(
        statistic=max(components),
        method="Lee-You-Lin maximum pairwise Bayes-factor mean test (2024)",
        alternative="true mean vectors differ",
        data_name="x and y",
        statistic_name="maximum log BF",
        calibration="closed-form Gaussian Bayes factors",
        diagnostics=(
            ("a0", shape),
            ("b0", prior_scale),
            (
                "prior specification",
                "paper Equation (4)"
                if shape == 0.0 and prior_scale == 0.0
                else "pySHT inverse-gamma extension",
            ),
        )
        + gamma_diagnostics,
        component_log_bayes_factors=components,
    )

"""Tests for one, two, and several population covariance matrices.

The implementations in this module follow the defining papers rather than
preserving computational defects in the legacy R package.  Observations are
rows and variables are columns.  Every randomized procedure uses a local
NumPy generator and therefore leaves NumPy's global random state untouched.
"""

from __future__ import annotations

import math
import operator
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats

from ._results import BayesFactorTestResult, HypothesisTestResult
from ._validation import (
    RngLike,
    make_generator,
    validate_2d_sample,
    validate_covariance_matrix,
    validate_multivariate_groups,
    validate_positive_integer,
    validate_real_scalar,
)

__all__ = [
    "clx_2samp",
    "lc_2samp",
    "lyl_2samp",
    "schott_2001_ksamp",
    "schott_2007_ksamp",
    "wl_1samp",
    "wl_2samp",
]

_NORMAL_CALIBRATION: Final = "asymptotic standard normal distribution"
_TWO_SIDED_MAX_NORMAL: Final = (
    "asymptotic independent two-sided standard-normal maximum"
)


def _common_scale(*arrays: NDArray[np.float64]) -> float:
    """Return a finite common scale that avoids overflow in scatter products."""
    scale = max((float(np.max(np.abs(array))) for array in arrays), default=0.0)
    return scale if scale > 0.0 else 1.0


def _center_together(
    *arrays: NDArray[np.float64], extra_scale: float = 0.0
) -> tuple[tuple[NDArray[np.float64], ...], float]:
    """Center each group before applying one common overflow-safe scale."""
    anchors = tuple(np.min(array, axis=0) for array in arrays)
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = tuple(
            np.asarray(array - anchor, dtype=np.float64)
            for array, anchor in zip(arrays, anchors, strict=True)
        )
    base_scale = 1.0
    if not all(np.all(np.isfinite(array)) for array in shifted):
        base_scale = _common_scale(*arrays)
        shifted = tuple(
            np.asarray(array / base_scale - anchor / base_scale, dtype=np.float64)
            for array, anchor in zip(arrays, anchors, strict=True)
        )
        if not all(np.all(np.isfinite(array)) for array in shifted):
            raise ValueError("the centered observations could not be represented")

    normalized_extra = extra_scale / base_scale
    local_scale = max(_common_scale(*shifted), normalized_extra)
    maximum = np.finfo(np.float64).max
    if local_scale <= 1.0 or base_scale <= maximum / local_scale:
        scale = base_scale * local_scale
        scaled = tuple(array / local_scale for array in shifted)
    else:
        scale = base_scale
        scaled = shifted
    centered = tuple(
        array - np.mean(array, axis=0, dtype=np.float64) for array in scaled
    )
    return centered, scale


def _sample_covariance(centered: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        centered.T @ centered / float(centered.shape[0] - 1), dtype=np.float64
    )


def _validate_pair(
    x: ArrayLike,
    y: ArrayLike,
    *,
    minimum_rows: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    first = validate_2d_sample(x, name="x", minimum_rows=minimum_rows)
    second = validate_2d_sample(y, name="y", minimum_rows=minimum_rows)
    if first.shape[1] != second.shape[1]:
        raise ValueError("x and y must have the same number of features")
    return first, second


def _positive_variant(value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("variant must be the integer 1 or 2, not bool")
    try:
        selected = operator.index(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("variant must be the integer 1 or 2") from exc
    if selected not in (1, 2):
        raise ValueError("variant must be 1 or 2")
    return selected


def _whiten_against_null(
    values: NDArray[np.float64], popcov: ArrayLike | None
) -> NDArray[np.float64]:
    feature_count = values.shape[1]
    covariance = (
        np.eye(feature_count, dtype=np.float64)
        if popcov is None
        else validate_covariance_matrix(
            popcov, name="popcov", size=feature_count, positive_definite=True
        )
    )
    covariance_scale = float(np.max(np.abs(covariance)))
    if covariance_scale <= 0.0:
        raise ValueError("popcov must be positive definite")
    normalized_covariance = covariance / covariance_scale
    try:
        root = np.linalg.cholesky(normalized_covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("popcov must be numerically positive definite") from exc

    (centered,), data_scale = _center_together(values)
    multiplier = data_scale / math.sqrt(covariance_scale)
    with np.errstate(over="ignore", invalid="ignore"):
        whitened = np.linalg.solve(root, centered.T).T * multiplier
    if not np.all(np.isfinite(whitened)):
        raise ValueError("whitening could not be evaluated in float64")
    return np.asarray(whitened, dtype=np.float64, order="C")


def _fisher_estimators(
    covariance: NDArray[np.float64], *, degrees: int
) -> tuple[float, float, float, float]:
    p = covariance.shape[0]
    n = float(degrees)
    square = covariance @ covariance
    cube = square @ covariance
    fourth = cube @ covariance
    tr1 = float(np.trace(covariance))
    tr2 = float(np.trace(square))
    tr3 = float(np.trace(cube))
    tr4 = float(np.trace(fourth))

    tau = n**4 / ((n - 1.0) * (n - 2.0) * (n + 2.0) * (n + 4.0))
    gamma = (
        n**5
        * (n * n + n + 2.0)
        / (
            (n + 1.0)
            * (n + 2.0)
            * (n + 4.0)
            * (n + 6.0)
            * (n - 1.0)
            * (n - 2.0)
            * (n - 3.0)
        )
    )
    a1 = tr1 / p
    a2 = n * n / ((n - 1.0) * (n + 2.0) * p) * (tr2 - tr1 * tr1 / n)
    a3 = tau / p * (tr3 - 3.0 * tr2 * tr1 / n + 2.0 * tr1**3 / n**2)
    a4_inner = (
        tr4
        - 4.0 * tr3 * tr1 / n
        - (2.0 * n * n + 3.0 * n - 6.0) / (n * (n * n + n + 2.0)) * tr2**2
        + 2.0 * (5.0 * n + 6.0) * tr2 * tr1**2 / (n * (n * n + n + 2.0))
        - (5.0 * n + 6.0) * tr1**4 / (n**4 + n**3 + 2.0 * n**2)
    )
    a4 = gamma * a4_inner / p
    estimators = (a1, a2, a3, a4)
    if not all(math.isfinite(value) for value in estimators):
        raise ValueError("Fisher trace estimators could not be evaluated in float64")
    return estimators


def _fisher_statistic_from_estimators(
    estimators: tuple[float, float, float, float],
    *,
    degrees: int,
    feature_count: int,
    variant: int,
) -> float:
    """Insert trace estimators into Fisher's two published statistics."""
    a1, a2, a3, a4 = estimators
    aspect_ratio = feature_count / degrees
    if variant == 1:
        return float(
            degrees
            / (aspect_ratio * math.sqrt(8.0))
            * (a4 - 4.0 * a3 + 6.0 * a2 - 4.0 * a1 + 1.0)
        )
    return float(
        degrees
        / math.sqrt(8.0 * (aspect_ratio**2 + 12.0 * aspect_ratio + 8.0))
        * (a4 - 2.0 * a2 + 1.0)
    )


def _fisher_log_scaled_statistic(
    normalized_estimators: tuple[float, float, float, float],
    *,
    log_covariance_scale: float,
    degrees: int,
    feature_count: int,
    variant: int,
) -> float:
    """Evaluate Fisher's polynomial loss as a signed log-scale sum."""
    a1, a2, a3, a4 = normalized_estimators
    terms = (
        ((4, a4), (3, -4.0 * a3), (2, 6.0 * a2), (1, -4.0 * a1), (0, 1.0))
        if variant == 1
        else ((4, a4), (2, -2.0 * a2), (0, 1.0))
    )
    log_terms = tuple(
        (power * log_covariance_scale + math.log(abs(coefficient)), coefficient)
        for power, coefficient in terms
        if coefficient != 0.0
    )
    largest_log = max(log_value for log_value, _ in log_terms)
    scaled_sum = math.fsum(
        math.copysign(math.exp(log_value - largest_log), coefficient)
        for log_value, coefficient in log_terms
    )
    if scaled_sum == 0.0:
        return 0.0

    aspect_ratio = feature_count / degrees
    multiplier = (
        degrees / (aspect_ratio * math.sqrt(8.0))
        if variant == 1
        else degrees / math.sqrt(8.0 * (aspect_ratio**2 + 12.0 * aspect_ratio + 8.0))
    )
    log_statistic = largest_log + math.log(abs(scaled_sum)) + math.log(multiplier)
    if log_statistic > math.log(float(np.finfo(np.float64).max)):
        return math.copysign(math.inf, scaled_sum)
    if log_statistic < math.log(float(np.nextafter(0.0, 1.0))):
        return math.copysign(0.0, scaled_sum)
    return math.copysign(math.exp(log_statistic), scaled_sum)


def _fisher_1samp(
    x: ArrayLike,
    *,
    popcov: ArrayLike | None = None,
    variant: int = 1,
) -> HypothesisTestResult:
    """Evaluate Fisher's withheld one-sample high-dimensional covariance test.

    This implementation remains private until its public-API null-calibration
    and power release gates can be reproduced in the advertised regime.
    ``variant=1`` uses the fourth-power loss and ``variant=2`` uses the
    squared quadratic loss from Fisher (2012).  The data are centered and
    whitened by ``popcov`` *before* the trace estimators are evaluated.
    """
    values = validate_2d_sample(x, name="x", minimum_rows=5)
    selected = _positive_variant(variant)
    whitened = _whiten_against_null(values, popcov)
    degrees = values.shape[0] - 1
    feature_count = values.shape[1]
    whitened_scale = float(np.max(np.abs(whitened)))
    if whitened_scale <= 1.0e30:
        with np.errstate(over="ignore", invalid="ignore"):
            covariance = whitened.T @ whitened / degrees
        statistic = _fisher_statistic_from_estimators(
            _fisher_estimators(covariance, degrees=degrees),
            degrees=degrees,
            feature_count=feature_count,
            variant=selected,
        )
    else:
        normalized_whitened = whitened / whitened_scale
        normalized_covariance = normalized_whitened.T @ normalized_whitened / degrees
        local_scale = float(np.max(np.abs(normalized_covariance)))
        if local_scale <= 0.0 or not math.isfinite(local_scale):
            raise ValueError("Fisher covariance scale could not be evaluated")
        normalized_covariance /= local_scale
        statistic = _fisher_log_scaled_statistic(
            _fisher_estimators(normalized_covariance, degrees=degrees),
            log_covariance_scale=(
                2.0 * math.log(whitened_scale) + math.log(local_scale)
            ),
            degrees=degrees,
            feature_count=feature_count,
            variant=selected,
        )
    if math.isnan(statistic):
        raise ValueError("Fisher statistic could not be evaluated")
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=float(stats.norm.sf(statistic)),
        method=f"Fisher one-sample covariance test (2012), variant {selected}",
        alternative="true covariance matrix differs from popcov",
        data_name="x",
        statistic_name=f"T{selected}",
        calibration=_NORMAL_CALIBRATION,
        diagnostics=(("variant", selected),),
    )


def _unit_projections(
    feature_count: int, n_projections: int, rng: RngLike
) -> NDArray[np.float64]:
    generator = make_generator(rng)
    projections = generator.standard_normal((n_projections, feature_count))
    lengths = np.linalg.norm(projections, axis=1)
    if np.any(lengths == 0.0) or not np.all(np.isfinite(lengths)):
        raise ValueError("random projection vectors could not be normalized")
    return projections / lengths[:, None]


def _two_sided_max_normal_pvalue(statistic: float, count: int) -> float:
    # Forming 2*Phi(M)-1 through erf rounds to one near M=9, producing a
    # spurious zero even when the maximum-test tail is still representable.
    # Work from the small two-sided marginal tail and use log1p instead.
    marginal_tail = 2.0 * float(special.ndtr(-statistic))
    if marginal_tail >= 1.0:
        return 1.0
    if marginal_tail <= 0.0:
        return 0.0
    return float(-math.expm1(count * math.log1p(-marginal_tail)))


def wl_1samp(
    x: ArrayLike,
    *,
    popcov: ArrayLike | None = None,
    n_projections: int = 25,
    rng: RngLike = None,
) -> HypothesisTestResult:
    """Perform the two-sided Wu--Li one-sample projection test.

    The procedure reports the maximum absolute square-root-transformed
    projected chi-square statistic.  Because the mean is estimated, the
    transformation uses ``n - 1`` degrees of freedom.
    """
    values = validate_2d_sample(x, name="x", minimum_rows=3)
    count = validate_positive_integer(n_projections, name="n_projections")
    whitened = _whiten_against_null(values, popcov)
    projections = _unit_projections(values.shape[1], count, rng)
    projected = whitened @ projections.T
    degrees = values.shape[0] - 1
    sums = np.sum(projected * projected, axis=0, dtype=np.float64)
    transformed = np.sqrt(2.0 * sums) - math.sqrt(2.0 * degrees - 1.0)
    statistic = float(np.max(np.abs(transformed)))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=_two_sided_max_normal_pvalue(statistic, count),
        method="Wu-Li two-sided one-sample covariance test (2015)",
        alternative="true covariance matrix differs from popcov",
        data_name="x",
        statistic_name="T1,m",
        calibration=_TWO_SIDED_MAX_NORMAL,
        diagnostics=(
            ("n_projections", count),
            ("projection distribution", "Gaussian unit vectors"),
        ),
    )


def _lc_a_unbiased(values: NDArray[np.float64]) -> float:
    """Evaluate Li--Chen Equation (2.1) via exact ordered-sum identities."""
    n = values.shape[0]
    gram = values @ values.T
    np.fill_diagonal(gram, 0.0)
    sum1 = float(np.sum(gram * gram, dtype=np.float64))
    row_sums = np.sum(gram, axis=1, dtype=np.float64)
    row_squares = np.sum(gram * gram, axis=1, dtype=np.float64)
    sum2 = float(np.sum(row_sums * row_sums - row_squares, dtype=np.float64))
    total = float(np.sum(row_sums, dtype=np.float64))
    complement = total - 2.0 * row_sums[:, None] - 2.0 * row_sums[None, :] + 2.0 * gram
    sum3 = float(np.sum(gram * complement, dtype=np.float64))
    nf = float(n)
    return (
        sum1 / (nf * (nf - 1.0))
        - 2.0 * sum2 / (nf * (nf - 1.0) * (nf - 2.0))
        + sum3 / (nf * (nf - 1.0) * (nf - 2.0) * (nf - 3.0))
    )


def _lc_c_unbiased(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """Evaluate Li--Chen Equation (2.2) via exact ordered-sum identities."""
    n1 = int(first.shape[0])
    n2 = int(second.shape[0])
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
    return term1 - term2 - term3 + term4


def lc_2samp(
    x: ArrayLike,
    y: ArrayLike,
) -> HypothesisTestResult:
    """Perform the Li--Chen two-sample high-dimensional covariance test.

    The implementation evaluates the literal order-two, order-three, and
    order-four U-statistics in Equations (2.1)--(2.2).  Each group therefore
    needs at least four observations.
    """
    first, second = _validate_pair(x, y, minimum_rows=4)
    (first_centered, second_centered), _ = _center_together(first, second)
    a1 = _lc_a_unbiased(first_centered)
    a2 = _lc_a_unbiased(second_centered)
    cross = _lc_c_unbiased(first_centered, second_centered)
    statistic_unscaled = a1 + a2 - 2.0 * cross
    standard_deviation = 2.0 * (a1 / second.shape[0] + a2 / first.shape[0])
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError("Li-Chen null standard-deviation estimate must be positive")
    statistic = statistic_unscaled / standard_deviation
    if not math.isfinite(statistic):
        raise ValueError("Li-Chen statistic could not be evaluated in float64")
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=float(stats.norm.sf(statistic)),
        method="Li-Chen two-sample high-dimensional covariance test (2012)",
        alternative="true covariance matrices differ",
        data_name="x and y",
        statistic_name="LC",
        calibration=_NORMAL_CALIBRATION,
    )


def _clx_tail(statistic: float, feature_count: int) -> float:
    centered = (
        statistic - 4.0 * math.log(feature_count) + math.log(math.log(feature_count))
    )
    log_rate = -0.5 * math.log(8.0 * math.pi) - 0.5 * centered
    if log_rate > math.log(np.finfo(np.float64).max):
        return 1.0
    rate = math.exp(log_rate)
    return float(-math.expm1(-rate))


def clx_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the Cai--Liu--Xia maximum covariance test (2013)."""
    first, second = _validate_pair(x, y)
    feature_count = first.shape[1]
    if feature_count < 2:
        raise ValueError("clx_2samp requires at least two features")
    (first_centered, second_centered), _ = _center_together(first, second)
    covariance1 = first_centered.T @ first_centered / first.shape[0]
    covariance2 = second_centered.T @ second_centered / second.shape[0]
    products1 = first_centered[:, :, None] * first_centered[:, None, :]
    products2 = second_centered[:, :, None] * second_centered[:, None, :]
    theta1 = np.mean((products1 - covariance1) ** 2, axis=0, dtype=np.float64)
    theta2 = np.mean((products2 - covariance2) ** 2, axis=0, dtype=np.float64)
    denominator = theta1 / first.shape[0] + theta2 / second.shape[0]
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("every CLX entrywise variance estimate must be positive")
    ratios = (covariance1 - covariance2) ** 2 / denominator
    statistic = float(np.max(ratios))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=_clx_tail(statistic, feature_count),
        method="Cai-Liu-Xia maximum two-sample covariance test (2013)",
        alternative="true covariance matrices differ",
        data_name="x and y",
        statistic_name="CLX",
        calibration="asymptotic type-I extreme-value distribution",
    )


def wl_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    n_projections: int = 50,
    rng: RngLike = None,
) -> HypothesisTestResult:
    """Perform the two-sided Wu--Li two-sample projection test."""
    first, second = _validate_pair(x, y, minimum_rows=3)
    count = validate_positive_integer(n_projections, name="n_projections")
    # Scale the centered groups separately, then restore their scalar units in
    # log variance.  A common computational scale can make the smaller
    # group's squared projections underflow when the variance ratio is extreme
    # even though its log ratio is finite and is precisely the quantity needed
    # by this test.
    (first_centered,), first_scale = _center_together(first)
    (second_centered,), second_scale = _center_together(second)
    projections = _unit_projections(first.shape[1], count, rng)
    first_projected = first_centered @ projections.T
    second_projected = second_centered @ projections.T
    df1 = first.shape[0] - 1
    df2 = second.shape[0] - 1
    variance1 = np.sum(first_projected**2, axis=0, dtype=np.float64) / df1
    variance2 = np.sum(second_projected**2, axis=0, dtype=np.float64) / df2
    if np.any(variance1 <= 0.0) or np.any(variance2 <= 0.0):
        raise ValueError("every projected sample variance must be positive")
    scale_ratio = first_scale / second_scale
    log_scale_ratio = (
        math.log(scale_ratio)
        if math.isfinite(scale_ratio) and scale_ratio > 0.0
        else math.log(first_scale) - math.log(second_scale)
    )
    log_variance_ratio = np.log(variance1) - np.log(variance2) + 2.0 * log_scale_ratio
    transformed = log_variance_ratio / math.sqrt(2.0 / df1 + 2.0 / df2)
    statistic = float(np.max(np.abs(transformed)))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=_two_sided_max_normal_pvalue(statistic, count),
        method="Wu-Li two-sided two-sample covariance test (2015)",
        alternative="true covariance matrices differ",
        data_name="x and y",
        statistic_name="T2,m",
        calibration=_TWO_SIDED_MAX_NORMAL,
        diagnostics=(
            ("n_projections", count),
            ("projection distribution", "Gaussian unit vectors"),
        ),
    )


def _log_prior_plus_half_residual_sum(
    response: NDArray[np.float64],
    predictor: NDArray[np.float64],
    *,
    log_prior_scale: float,
) -> float:
    """Evaluate ``log(b0 + RSS / 2)`` without squaring data in their units."""
    response_scale = float(np.max(np.abs(response)))
    predictor_scale = float(np.max(np.abs(predictor)))
    if predictor_scale <= 0.0 or not math.isfinite(predictor_scale):
        raise ValueError("LYL conditional regressions require variable predictors")
    if response_scale == 0.0:
        return log_prior_scale
    if not math.isfinite(response_scale):
        raise ValueError("LYL conditional regressions require finite responses")
    scaled_response = response / response_scale
    scaled_predictor = predictor / predictor_scale
    denominator = float(np.dot(scaled_predictor, scaled_predictor))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("LYL conditional regressions require variable predictors")
    coefficient = float(np.dot(scaled_response, scaled_predictor)) / denominator
    residual = scaled_response - coefficient * scaled_predictor
    residual_sum = float(np.dot(residual, residual))
    if residual_sum < 0.0 or not math.isfinite(residual_sum):
        raise ValueError("LYL residual sum of squares could not be evaluated")
    if residual_sum == 0.0:
        return log_prior_scale
    log_half_residual_sum = (
        math.log(0.5) + 2.0 * math.log(response_scale) + math.log(residual_sum)
    )
    return float(np.logaddexp(log_prior_scale, log_half_residual_sum))


def lyl_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    a0: float = 0.01,
    b0: float = 0.01,
    alpha: float = 2.01,
    gamma: float | None = None,
) -> BayesFactorTestResult:
    """Compute Lee--You--Lin maximum pairwise covariance Bayes factors.

    The statistic is the largest ordered-pair *log* Bayes factor from
    Equations (12)--(15).  No p-value or universal evidence threshold is
    manufactured.  This is the paper's known-zero-mean model: observations
    are used as supplied, without automatic centering or an intercept.
    Conditional residual variances use ordinary least squares.  By default,
    ``gamma = max(n1 + n2, p) ** (-alpha)`` as recommended by the authors;
    an explicit ``gamma`` overrides that rule.
    """
    first, second = _validate_pair(x, y)
    feature_count = first.shape[1]
    if feature_count < 2:
        raise ValueError("lyl_2samp requires at least two features")
    shape = validate_real_scalar(a0, name="a0")
    prior_scale = validate_real_scalar(b0, name="b0")
    exponent = validate_real_scalar(alpha, name="alpha")
    if shape <= 0.0:
        raise ValueError("a0 must be greater than 0")
    if prior_scale <= 0.0:
        raise ValueError("b0 must be greater than 0")
    if exponent <= 0.0:
        raise ValueError("alpha must be greater than 0")

    n1 = first.shape[0]
    n2 = second.shape[0]
    total = n1 + n2
    if gamma is None:
        shrinkage = math.exp(-exponent * math.log(max(total, feature_count)))
        gamma_source = "derived from alpha"
        if shrinkage <= 0.0:
            raise ValueError("derived gamma underflowed; supply an explicit gamma")
    else:
        shrinkage = validate_real_scalar(gamma, name="gamma")
        gamma_source = "supplied"
        if shrinkage <= 0.0:
            raise ValueError("gamma must be greater than 0")

    pooled = np.concatenate((first, second), axis=0)
    log_prior_scale = math.log(prior_scale)
    constant = (
        0.5 * (math.log(shrinkage) - math.log1p(shrinkage))
        + float(special.gammaln(n1 / 2.0 + shape))
        + float(special.gammaln(n2 / 2.0 + shape))
        - float(special.gammaln(total / 2.0 + shape))
        + shape * log_prior_scale
        - float(special.gammaln(shape))
    )
    factors = np.full((feature_count, feature_count), -math.inf, dtype=np.float64)
    for i in range(feature_count):
        for j in range(feature_count):
            if i == j:
                continue
            log_term1 = _log_prior_plus_half_residual_sum(
                first[:, i], first[:, j], log_prior_scale=log_prior_scale
            )
            log_term2 = _log_prior_plus_half_residual_sum(
                second[:, i], second[:, j], log_prior_scale=log_prior_scale
            )
            log_term0 = _log_prior_plus_half_residual_sum(
                pooled[:, i], pooled[:, j], log_prior_scale=log_prior_scale
            )
            term1 = (n1 / 2.0 + shape) * log_term1
            term2 = (n2 / 2.0 + shape) * log_term2
            term0 = (total / 2.0 + shape) * log_term0
            factors[i, j] = constant - term1 - term2 + term0
    components = tuple(tuple(float(value) for value in row) for row in factors)
    statistic = float(np.max(factors))
    return BayesFactorTestResult(
        statistic=statistic,
        method="Lee-You-Lin maximum pairwise Bayes-factor covariance test (2024)",
        alternative="true covariance matrices differ",
        data_name="x and y",
        statistic_name="maximum log BF",
        calibration="closed-form Gaussian conditional-regression Bayes factors",
        diagnostics=(
            ("a0", shape),
            ("b0", prior_scale),
            ("alpha", exponent),
            ("gamma", shrinkage),
            ("gamma source", gamma_source),
            ("mean assumption", "known zero"),
        ),
        component_log_bayes_factors=components,
    )


def _whiten_covariance(
    covariance: NDArray[np.float64], root: NDArray[np.float64]
) -> NDArray[np.float64]:
    left = np.linalg.solve(root, covariance)
    whitened = np.linalg.solve(root, left.T).T
    return 0.5 * (whitened + whitened.T)


def schott_2001_ksamp(*samples: ArrayLike) -> HypothesisTestResult:
    """Perform Schott's classical Wald test for covariance homogeneity."""
    groups = validate_multivariate_groups(samples, minimum_rows=2)
    centered_groups, _ = _center_together(*groups)
    degrees = np.asarray([group.shape[0] - 1 for group in groups], dtype=np.float64)
    covariances = tuple(_sample_covariance(group) for group in centered_groups)
    total_degrees = float(np.sum(degrees))
    pooled = sum(
        (degree / total_degrees) * covariance
        for degree, covariance in zip(degrees, covariances, strict=True)
    )
    eigenvalues = np.linalg.eigvalsh(pooled)
    spectral_scale = float(eigenvalues[-1])
    rank_tolerance = np.finfo(np.float64).eps * pooled.shape[0] * spectral_scale
    if eigenvalues[0] <= rank_tolerance:
        raise ValueError(
            "the pooled covariance must be numerically positive definite for "
            "Schott (2001)"
        )
    try:
        root = np.linalg.cholesky(pooled)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the pooled covariance must be positive definite for Schott (2001)"
        ) from exc
    feature_count = groups[0].shape[1]
    identity = np.eye(feature_count, dtype=np.float64)
    statistic_sum = 0.0
    for degree, covariance in zip(degrees, covariances, strict=True):
        whitened = _whiten_covariance(covariance, root)
        difference = whitened - identity
        statistic_sum += (
            degree
            / total_degrees
            * float(np.sum(difference * difference, dtype=np.float64))
        )
    statistic = 0.5 * total_degrees * statistic_sum
    df = float((len(groups) - 1) * feature_count * (feature_count + 1) / 2)
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=float(stats.chi2.sf(statistic, df)),
        method="Schott k-sample covariance homogeneity test (2001)",
        alternative="at least one true covariance matrix differs",
        data_name="samples",
        statistic_name="W",
        calibration="asymptotic chi-square distribution",
        df=df,
    )


def schott_2007_ksamp(*samples: ArrayLike) -> HypothesisTestResult:
    """Perform Schott's high-dimensional covariance homogeneity test."""
    groups = validate_multivariate_groups(samples, minimum_rows=3)
    centered_groups, _ = _center_together(*groups)
    degrees = np.asarray([group.shape[0] - 1 for group in groups], dtype=np.float64)
    covariances = tuple(_sample_covariance(group) for group in centered_groups)
    total_degrees = float(np.sum(degrees))
    pooled = sum(
        (degree / total_degrees) * covariance
        for degree, covariance in zip(degrees, covariances, strict=True)
    )
    traces = np.asarray([np.trace(covariance) for covariance in covariances])
    squared_traces = np.asarray(
        [np.sum(covariance * covariance) for covariance in covariances]
    )
    tnm = 0.0
    inner1 = 0.0
    for i in range(len(groups) - 1):
        ni = degrees[i]
        ei = (ni + 2.0) * (ni - 1.0)
        for j in range(i + 1, len(groups)):
            nj = degrees[j]
            ej = (nj + 2.0) * (nj - 1.0)
            cross_trace = float(np.sum(covariances[i] * covariances[j]))
            tnm += (
                (1.0 - (ni - 2.0) / ei) * squared_traces[i]
                + (1.0 - (nj - 2.0) / ej) * squared_traces[j]
                - 2.0 * cross_trace
                - ni / ei * traces[i] ** 2
                - nj / ej * traces[j] ** 2
            )
            inner1 += ((ni + nj) / (ni * nj)) ** 2
    feature_count = groups[0].shape[1]
    del feature_count  # the formula uses dimension only through covariance traces
    pooled_trace = float(np.trace(pooled))
    pooled_square_trace = float(np.sum(pooled * pooled))
    a_estimate = (
        total_degrees**2
        / ((total_degrees + 2.0) * (total_degrees - 1.0))
        * (pooled_square_trace - pooled_trace**2 / total_degrees)
    )
    inner2 = (len(groups) - 1) * (len(groups) - 2) * float(np.sum(1.0 / degrees**2))
    theta = 2.0 * math.sqrt(inner1 + inner2) * a_estimate
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("Schott (2007) variance estimate must be positive")
    statistic = tnm / theta
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=float(stats.norm.sf(statistic)),
        method="Schott high-dimensional k-sample covariance test (2007)",
        alternative="at least one true covariance matrix differs",
        data_name="samples",
        statistic_name="T",
        calibration=_NORMAL_CALIBRATION,
    )

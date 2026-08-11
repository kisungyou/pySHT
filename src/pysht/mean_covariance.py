"""Joint tests of multivariate population means and covariance matrices."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._results import HypothesisTestResult
from ._validation import (
    validate_1d_sample,
    validate_2d_sample,
    validate_covariance_matrix,
)

__all__ = ["hn_2samp", "llzs_1samp", "lrt_1samp"]

_JOINT_ALTERNATIVE_1SAMP = (
    "at least one of the population mean vector and covariance matrix "
    "differs from its null value"
)
_JOINT_ALTERNATIVE_2SAMP = (
    "at least one of the two population mean vectors and covariance matrices differs"
)
_LOG_FLOAT_MAX = math.log(float(np.finfo(np.float64).max))
_LOG_SMALLEST = math.log(float(np.nextafter(0.0, 1.0)))


def _null_mean(popmean: ArrayLike | None, p: int) -> NDArray[np.float64]:
    if popmean is None:
        return np.zeros(p, dtype=np.float64)
    result = validate_1d_sample(popmean, name="popmean", minimum_size=1)
    if result.size != p:
        raise ValueError("popmean must contain one value per feature")
    return result


def _null_covariance(popcov: ArrayLike | None, p: int) -> NDArray[np.float64]:
    if popcov is None:
        return np.eye(p, dtype=np.float64)
    return validate_covariance_matrix(
        popcov,
        name="popcov",
        size=p,
        positive_definite=True,
    )


def _whiten_against_null(
    values: NDArray[np.float64],
    popmean: ArrayLike | None,
    popcov: ArrayLike | None,
) -> NDArray[np.float64]:
    p = values.shape[1]
    null_mean = _null_mean(popmean, p)
    null_covariance = _null_covariance(popcov, p)
    try:
        factor = np.linalg.cholesky(null_covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("popcov must be positive definite") from exc

    # Preserve differences of a few ulps around a huge common null location by
    # subtracting first.  Fall back to scaled subtraction only when opposite
    # extreme endpoints make the direct difference overflow.
    with np.errstate(over="ignore", invalid="ignore"):
        differences = values - null_mean
    if np.all(np.isfinite(differences)):
        scale = max(
            float(np.max(np.abs(differences))),
            float(np.max(np.abs(factor))),
        )
        if scale == 0.0:
            scale = 1.0
        centered_scaled = differences / scale
    else:
        scale = max(
            float(np.max(np.abs(values))),
            float(np.max(np.abs(null_mean))),
            float(np.max(np.abs(factor))),
        )
        centered_scaled = values / scale - null_mean / scale
    factor_scaled = factor / scale
    try:
        whitened = np.linalg.solve(factor_scaled, centered_scaled.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("popcov could not be used to whiten the sample") from exc
    if not np.all(np.isfinite(whitened)):
        raise ValueError("the observations cannot be represented on the null scale")
    return np.asarray(whitened, dtype=np.float64, order="C")


def _mle_mean_covariance(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = np.mean(values, axis=0, dtype=np.float64)
    centered = values - mean
    covariance = centered.T @ centered / values.shape[0]
    return mean, covariance


def llzs_1samp(
    x: ArrayLike,
    *,
    popmean: ArrayLike | None = None,
    popcov: ArrayLike | None = None,
) -> HypothesisTestResult:
    """Perform the Liu--Liu--Zheng--Shi one-sample joint test.

    The procedure targets high-dimensional observations with dimension and
    sample size increasing proportionally.  It requires independent rows,
    coordinate-wise fourth moments, and the spectral regularity assumptions
    stated by Liu, Liu, Zheng, and Shi (2017).
    """
    values = validate_2d_sample(x, name="x")
    standardized = _whiten_against_null(values, popmean, popcov)
    n, p = standardized.shape
    aspect_ratio = p / n
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(standardized, axis=0, dtype=np.float64)
    # LLZS uses the second moment about the *known null mean*, not the
    # covariance centered at the sample mean.  SHT 0.1.9 centered here, which
    # shifts the null statistic by an order-one amount when p/n has a positive
    # limit.
    if p <= n:
        with np.errstate(over="ignore", invalid="ignore"):
            second_moment = standardized.T @ standardized / n
            covariance_difference = second_moment - np.eye(p, dtype=np.float64)
            squared_covariance_departure = float(
                np.sum(covariance_difference * covariance_difference)
            )
    else:
        # tr((M-I)^2) = tr(M^2) - 2 tr(M) + p.  The observation Gram
        # matrix avoids allocating a p-by-p matrix in the intended p > n
        # regime.
        with np.errstate(over="ignore", invalid="ignore"):
            gram = standardized @ standardized.T
            trace_second_moment = float(np.sum(standardized * standardized) / n)
            trace_second_moment_squared = float(np.sum(gram * gram) / n**2)
        squared_covariance_departure = (
            trace_second_moment_squared - 2.0 * trace_second_moment + p
        )
    with np.errstate(over="ignore", invalid="ignore"):
        raw_statistic = float(np.dot(mean, mean) + squared_covariance_departure)

    with np.errstate(over="ignore", invalid="ignore"):
        fourth_moment = float(np.mean(np.square(np.square(standardized))))
    excess_kurtosis = fourth_moment - 3.0
    if not math.isfinite(raw_statistic) or not math.isfinite(excess_kurtosis):
        return HypothesisTestResult(
            statistic=math.inf,
            pvalue=0.0,
            method="Liu-Liu-Zheng-Shi mean and covariance test (2017)",
            alternative=_JOINT_ALTERNATIVE_1SAMP,
            data_name="x",
            statistic_name="Z",
            calibration="asymptotic standard normal distribution (upper tail)",
            diagnostics=(
                ("aspect ratio p/n", aspect_ratio),
                ("estimated marginal excess kurtosis", "outside float64 range"),
            ),
        )
    null_center = aspect_ratio * (p + excess_kurtosis + 2.0)
    null_variance = (
        4.0 * aspect_ratio**2 * (aspect_ratio * (2.0 + excess_kurtosis) + 1.0)
    )
    if (
        not math.isfinite(raw_statistic)
        or not math.isfinite(null_center)
        or not math.isfinite(null_variance)
        or null_variance <= 0.0
    ):
        raise ValueError("the LLZS null calibration could not be estimated")

    statistic = (raw_statistic - null_center) / math.sqrt(null_variance)
    # Both squared departures have non-negative population targets, so local
    # alternatives shift the standardized statistic to the right.
    pvalue = float(stats.norm.sf(statistic))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Liu-Liu-Zheng-Shi mean and covariance test (2017)",
        alternative=_JOINT_ALTERNATIVE_1SAMP,
        data_name="x",
        statistic_name="Z",
        calibration="asymptotic standard normal distribution (upper tail)",
        diagnostics=(
            ("aspect ratio p/n", aspect_ratio),
            ("estimated marginal excess kurtosis", excess_kurtosis),
        ),
    )


def lrt_1samp(
    x: ArrayLike,
    *,
    popmean: ArrayLike | None = None,
    popcov: ArrayLike | None = None,
) -> HypothesisTestResult:
    """Perform the classical joint multivariate likelihood-ratio test.

    The test uses Wilks' fixed-dimension chi-square limit for a multivariate
    normal sample.  A positive-definite maximum-likelihood covariance requires
    more observations than features and full centered column rank.
    """
    values = validate_2d_sample(x, name="x")
    standardized = _whiten_against_null(values, popmean, popcov)
    n, p = standardized.shape
    if n <= p:
        raise ValueError("lrt_1samp requires more observations than features")

    with np.errstate(over="ignore", invalid="ignore"):
        mean, covariance = _mle_mean_covariance(standardized)
    eigenvalues: NDArray[np.float64] | None
    if np.all(np.isfinite(covariance)):
        try:
            eigenvalues = np.linalg.eigvalsh(covariance)
        except np.linalg.LinAlgError:
            eigenvalues = None
    else:
        eigenvalues = None

    if eigenvalues is not None:
        if np.any(eigenvalues <= 0.0) or not np.all(np.isfinite(eigenvalues)):
            raise ValueError(
                "the maximum-likelihood covariance must be positive definite"
            )
        eigenvalue_errors = eigenvalues - 1.0
        with np.errstate(over="ignore", invalid="ignore"):
            divergence = float(
                np.sum(eigenvalue_errors - np.log1p(eigenvalue_errors))
                + np.dot(mean, mean)
            )
    else:
        magnitude = float(np.max(np.abs(standardized)))
        if magnitude == 0.0 or not math.isfinite(magnitude):
            raise ValueError("the standardized observations are numerically singular")
        scaled = standardized / magnitude
        scaled_mean, scaled_covariance = _mle_mean_covariance(scaled)
        try:
            scaled_eigenvalues = np.linalg.eigvalsh(scaled_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "the maximum-likelihood covariance could not be diagonalized"
            ) from exc
        if np.any(scaled_eigenvalues <= 0.0) or not np.all(
            np.isfinite(scaled_eigenvalues)
        ):
            raise ValueError(
                "the maximum-likelihood covariance must be positive definite"
            )
        quadratic_coefficient = float(
            np.sum(scaled_eigenvalues) + np.dot(scaled_mean, scaled_mean)
        )
        log_quadratic = 2.0 * math.log(magnitude) + math.log(quadratic_coefficient)
        if log_quadratic > _LOG_FLOAT_MAX:
            divergence = math.inf
        else:
            quadratic = (
                0.0 if log_quadratic < _LOG_SMALLEST else math.exp(log_quadratic)
            )
            log_determinant = float(
                np.sum(np.log(scaled_eigenvalues)) + 2.0 * p * math.log(magnitude)
            )
            divergence = quadratic - log_determinant - p
    statistic = math.inf if not math.isfinite(divergence) else max(0.0, n * divergence)
    degrees_of_freedom = 0.5 * p * (p + 3)
    pvalue = float(stats.chi2.sf(statistic, degrees_of_freedom))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="One-sample likelihood-ratio test of mean and covariance",
        alternative=_JOINT_ALTERNATIVE_1SAMP,
        data_name="x",
        statistic_name="-2 log Lambda",
        calibration="asymptotic chi-square distribution",
        df=float(degrees_of_freedom),
    )


def _unbiased_trace_covariance_squared(
    centered: NDArray[np.float64],
    *,
    trace_covariance: float,
    trace_covariance_squared: float,
) -> float:
    """Estimate ``tr(Sigma**2)`` without a normality assumption."""
    n = centered.shape[0]
    row_squared_norms = np.einsum("ij,ij->i", centered, centered)
    fourth_norm_average = float(np.dot(row_squared_norms, row_squared_norms) / (n - 1))
    return float(
        (n - 1)
        / (n * (n - 2) * (n - 3))
        * (
            (n - 1) * (n - 2) * trace_covariance_squared
            + trace_covariance**2
            - n * fourth_norm_average
        )
    )


def _rescale_distance_estimate(
    value: float, scale: float, *, power: int
) -> float | None:
    """Restore an HN distance estimate to its original measurement units.

    HN's statistic is evaluated after a common scale transformation.  Mean
    and covariance squared distances have scale powers two and four,
    respectively.  A result outside finite float64 range is omitted rather
    than mislabeled in the normalized computational units.
    """
    if value == 0.0:
        return math.copysign(0.0, value)
    log_magnitude = math.log(abs(value)) + power * math.log(scale)
    if log_magnitude > _LOG_FLOAT_MAX:
        return None
    if log_magnitude < _LOG_SMALLEST:
        return math.copysign(0.0, value)
    return math.copysign(math.exp(log_magnitude), value)


def hn_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the Hyodo--Nishiyama two-sample joint test.

    The test combines unbiased squared-distance estimators for the means and
    covariance matrices.  Its standard-normal upper-tail calibration is
    asymptotic in dimension and both sample sizes and requires the moment and
    trace conditions of Hyodo and Nishiyama (2018).
    """
    first = validate_2d_sample(x, name="x", minimum_rows=4)
    second = validate_2d_sample(y, name="y", minimum_rows=4)
    if first.shape[1] != second.shape[1]:
        raise ValueError("x and y must contain the same number of features")

    # The statistic is invariant under a common scalar rescaling and shift.
    # Applying both first keeps fourth-order calculations in a safe range.
    anchor = first[0].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        first_shifted = first - anchor
        second_shifted = second - anchor
    if np.all(np.isfinite(first_shifted)) and np.all(np.isfinite(second_shifted)):
        restoration_scale = max(
            float(np.max(np.abs(first_shifted))),
            float(np.max(np.abs(second_shifted))),
        )
        if restoration_scale == 0.0:
            raise ValueError(
                "the HN variance estimators are undefined for constant data"
            )
        first_scaled = first_shifted / restoration_scale
        second_scaled = second_shifted / restoration_scale
    else:
        # Opposite extreme endpoints require a scaled-subtraction fallback.
        restoration_scale = max(
            float(np.max(np.abs(first))), float(np.max(np.abs(second)))
        )
        first_scaled = first / restoration_scale - anchor / restoration_scale
        second_scaled = second / restoration_scale - anchor / restoration_scale

    n1 = first.shape[0]
    n2 = second.shape[0]
    mean1 = np.mean(first_scaled, axis=0, dtype=np.float64)
    mean2 = np.mean(second_scaled, axis=0, dtype=np.float64)
    centered1 = first_scaled - mean1
    centered2 = second_scaled - mean2
    trace_covariance1 = float(np.sum(centered1 * centered1) / (n1 - 1))
    trace_covariance2 = float(np.sum(centered2 * centered2) / (n2 - 1))
    p = first.shape[1]
    if p <= min(n1, n2):
        scatter1 = centered1.T @ centered1
        scatter2 = centered2.T @ centered2
        trace_covariance_squared1 = float(np.sum(scatter1 * scatter1) / (n1 - 1) ** 2)
        trace_covariance_squared2 = float(np.sum(scatter2 * scatter2) / (n2 - 1) ** 2)
        trace_cross = float(np.sum(scatter1 * scatter2) / ((n1 - 1) * (n2 - 1)))
    else:
        gram1 = centered1 @ centered1.T
        gram2 = centered2 @ centered2.T
        cross_gram = centered1 @ centered2.T
        trace_covariance_squared1 = float(np.sum(gram1 * gram1) / (n1 - 1) ** 2)
        trace_covariance_squared2 = float(np.sum(gram2 * gram2) / (n2 - 1) ** 2)
        trace_cross = float(np.sum(cross_gram * cross_gram) / ((n1 - 1) * (n2 - 1)))

    trace_square1 = _unbiased_trace_covariance_squared(
        centered1,
        trace_covariance=trace_covariance1,
        trace_covariance_squared=trace_covariance_squared1,
    )
    trace_square2 = _unbiased_trace_covariance_squared(
        centered2,
        trace_covariance=trace_covariance2,
        trace_covariance_squared=trace_covariance_squared2,
    )
    estimated_mean_distance = float(
        np.dot(mean1 - mean2, mean1 - mean2)
        - trace_covariance1 / n1
        - trace_covariance2 / n2
    )
    estimated_covariance_distance = trace_square1 + trace_square2 - 2.0 * trace_cross

    mean_null_variance = (
        2.0 * trace_square1 / n1**2
        + 2.0 * trace_square2 / n2**2
        + 4.0 * trace_cross / (n1 * n2)
    )
    covariance_null_variance = (
        4.0 * trace_square1**2 / n1**2
        + 4.0 * trace_square2**2 / n2**2
        + 8.0 * trace_cross**2 / (n1 * n2)
    )
    if (
        not math.isfinite(mean_null_variance)
        or not math.isfinite(covariance_null_variance)
        or mean_null_variance <= 0.0
        or covariance_null_variance <= 0.0
    ):
        raise ValueError("the HN null variance estimators must be positive")

    statistic = estimated_mean_distance / math.sqrt(
        mean_null_variance
    ) + estimated_covariance_distance / math.sqrt(covariance_null_variance)
    pvalue = float(stats.norm.sf(statistic / math.sqrt(2.0)))
    restored_mean_distance = _rescale_distance_estimate(
        estimated_mean_distance, restoration_scale, power=2
    )
    restored_covariance_distance = _rescale_distance_estimate(
        estimated_covariance_distance, restoration_scale, power=4
    )
    estimates = tuple(
        (name, value)
        for name, value in (
            ("estimated squared mean distance", restored_mean_distance),
            (
                "estimated squared covariance distance",
                restored_covariance_distance,
            ),
        )
        if value is not None
    )
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Hyodo-Nishiyama two-sample mean and covariance test (2018)",
        alternative=_JOINT_ALTERNATIVE_2SAMP,
        data_name="x and y",
        statistic_name="T",
        calibration="T / sqrt(2) has an asymptotic standard normal distribution",
        estimates=estimates,
    )

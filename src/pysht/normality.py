"""Univariate goodness-of-fit tests for a normal distribution.

All procedures test the composite null that a finite real sample comes from
some normal distribution.  The implementations are location- and
scale-invariant, reject constant samples explicitly, and never use NumPy's
global random state.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._resampling import monte_carlo_calibration
from ._results import HypothesisTestResult, ResamplingTestResult
from ._validation import (
    make_generator,
    validate_1d_sample,
    validate_choice,
    validate_positive_integer,
    validate_real_scalar,
)

__all__ = [
    "adjusted_jarque_bera",
    "jarque_bera",
    "robust_jarque_bera",
    "shapiro_francia",
    "shapiro_wilk",
]


type _Calibration = Literal["asymptotic", "monte-carlo"]
type _RowStatistic = Callable[[NDArray[np.float64]], NDArray[np.float64]]

_NORMALITY_ALTERNATIVE: Final = "the distribution is not normal"
_MONTE_CARLO_BATCH_VALUES: Final = 1_000_000


def _validate_calibration(value: str) -> _Calibration:
    """Validate a normality-test calibration without accepting R initials."""
    return validate_choice(
        value,
        name="calibration",
        choices=("asymptotic", "monte-carlo"),
    )


def _standardized_sample(
    x: ArrayLike, *, minimum_size: int, maximum_size: int | None = None
) -> NDArray[np.float64]:
    """Validate and standardize a nonconstant sample with stable arithmetic."""
    values = validate_1d_sample(x, name="x", minimum_size=minimum_size)
    if maximum_size is not None and values.size > maximum_size:
        raise ValueError(f"x must contain at most {maximum_size} observations")

    # Centering only after division by a large common location can discard
    # several bits of an otherwise representable within-sample difference.
    # Subtract a permutation-invariant sample anchor first.  For samples that
    # span opposite float64 endpoints the subtraction can overflow, so retain
    # scale-first centering as the fallback for that distinct regime.
    anchor = float(np.min(values))
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = values - anchor
    if np.all(np.isfinite(shifted)):
        scale = float(np.max(np.abs(shifted)))
        if scale == 0.0:
            raise ValueError("a normality test is undefined for a constant sample")
        scaled = shifted / scale
    else:
        scale = float(np.max(np.abs(values)))
        if scale == 0.0:
            raise ValueError("a normality test is undefined for a constant sample")
        scaled = values / scale
    centered = scaled - float(np.mean(scaled, dtype=np.float64))
    second_moment = float(np.mean(centered * centered, dtype=np.float64))
    if second_moment == 0.0:
        raise ValueError("a normality test is undefined for a constant sample")
    return centered / math.sqrt(second_moment)


def _sample_moments(
    standardized: NDArray[np.float64],
) -> tuple[float, float]:
    """Return the standardized third and fourth central sample moments."""
    squared = standardized * standardized
    skewness = float(np.mean(squared * standardized, dtype=np.float64))
    kurtosis = float(np.mean(squared * squared, dtype=np.float64))
    return skewness, kurtosis


def _jarque_bera_statistic(standardized: NDArray[np.float64]) -> float:
    n = float(standardized.size)
    skewness, kurtosis = _sample_moments(standardized)
    return n * (skewness * skewness / 6.0 + (kurtosis - 3.0) ** 2 / 24.0)


def _adjusted_jarque_bera_statistic(
    standardized: NDArray[np.float64],
) -> float:
    n = float(standardized.size)
    skewness, kurtosis = _sample_moments(standardized)
    expected_kurtosis = 3.0 * (n - 1.0) / (n + 1.0)
    skewness_variance = 6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0))
    kurtosis_variance = (
        24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) ** 2 * (n + 3.0) * (n + 5.0))
    )
    return (
        skewness * skewness / skewness_variance
        + (kurtosis - expected_kurtosis) ** 2 / kurtosis_variance
    )


def _robust_jarque_bera_statistic(
    standardized: NDArray[np.float64], *, c1: float, c2: float
) -> float:
    n = float(standardized.size)
    skewness_moment, kurtosis_moment = _sample_moments(standardized)
    median = float(np.median(standardized))
    robust_scale = math.sqrt(math.pi / 2.0) * float(
        np.mean(np.abs(standardized - median), dtype=np.float64)
    )
    if robust_scale == 0.0:
        raise ValueError("the robust scale is zero; the statistic is undefined")
    robust_skewness = skewness_moment / robust_scale**3
    robust_kurtosis = kurtosis_moment / robust_scale**4
    statistic = (
        n * robust_skewness * robust_skewness / c1
        + n * (robust_kurtosis - 3.0) ** 2 / c2
    )
    if not math.isfinite(statistic):
        raise ValueError("the robust Jarque-Bera statistic is not finite")
    return statistic


def _standardize_rows(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = samples - np.mean(samples, axis=1, keepdims=True, dtype=np.float64)
    second_moments = np.mean(centered * centered, axis=1, keepdims=True)
    return centered / np.sqrt(second_moments)


def _jb_rows(standardized: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(standardized.shape[1])
    squared = standardized * standardized
    skewness = np.mean(squared * standardized, axis=1)
    kurtosis = np.mean(squared * squared, axis=1)
    return n * (skewness * skewness / 6.0 + (kurtosis - 3.0) ** 2 / 24.0)


def _ajb_rows(standardized: NDArray[np.float64]) -> NDArray[np.float64]:
    n = float(standardized.shape[1])
    squared = standardized * standardized
    skewness = np.mean(squared * standardized, axis=1)
    kurtosis = np.mean(squared * squared, axis=1)
    expected_kurtosis = 3.0 * (n - 1.0) / (n + 1.0)
    skewness_variance = 6.0 * (n - 2.0) / ((n + 1.0) * (n + 3.0))
    kurtosis_variance = (
        24.0 * n * (n - 2.0) * (n - 3.0) / ((n + 1.0) ** 2 * (n + 3.0) * (n + 5.0))
    )
    return (
        skewness * skewness / skewness_variance
        + (kurtosis - expected_kurtosis) ** 2 / kurtosis_variance
    )


def _rjb_rows(
    standardized: NDArray[np.float64], *, c1: float, c2: float
) -> NDArray[np.float64]:
    n = float(standardized.shape[1])
    squared = standardized * standardized
    third_moments = np.mean(squared * standardized, axis=1)
    fourth_moments = np.mean(squared * squared, axis=1)
    medians = np.median(standardized, axis=1, keepdims=True)
    robust_scales = math.sqrt(math.pi / 2.0) * np.mean(
        np.abs(standardized - medians), axis=1
    )
    robust_skewness = third_moments / robust_scales**3
    robust_kurtosis = fourth_moments / robust_scales**4
    return (
        n * robust_skewness * robust_skewness / c1
        + n * (robust_kurtosis - 3.0) ** 2 / c2
    )


def _monte_carlo_exceedances(
    *,
    observed: float,
    sample_size: int,
    n_resamples: int,
    generator: np.random.Generator,
    row_statistic: _RowStatistic,
) -> int:
    """Count simulated null statistics at least as large as the observed one."""
    remaining = n_resamples
    exceedances = 0
    batch_size = max(1, _MONTE_CARLO_BATCH_VALUES // sample_size)
    while remaining:
        current = min(remaining, batch_size)
        samples = generator.standard_normal((current, sample_size))
        standardized = _standardize_rows(samples)
        simulated = row_statistic(standardized)
        exceedances += int(np.count_nonzero(simulated >= observed))
        remaining -= current
    return exceedances


def _moment_test_result(
    standardized: NDArray[np.float64],
    *,
    statistic: float,
    statistic_name: str,
    method: str,
    calibration: str,
    n_resamples: object,
    rng: int | np.integer | np.random.Generator | None,
    row_statistic: _RowStatistic,
) -> HypothesisTestResult | ResamplingTestResult:
    """Calibrate one moment statistic and construct its immutable result."""
    selected_calibration = _validate_calibration(calibration)
    resamples = validate_positive_integer(n_resamples, name="n_resamples")
    generator = make_generator(rng)

    if selected_calibration == "asymptotic":
        return HypothesisTestResult(
            statistic=statistic,
            pvalue=float(stats.chi2.sf(statistic, df=2)),
            method=method,
            alternative=_NORMALITY_ALTERNATIVE,
            data_name="x",
            statistic_name=statistic_name,
            calibration="asymptotic chi-square approximation",
            df=2.0,
        )

    exceedances = _monte_carlo_exceedances(
        observed=statistic,
        sample_size=standardized.size,
        n_resamples=resamples,
        generator=generator,
        row_statistic=row_statistic,
    )
    pvalue, standard_error, interval = monte_carlo_calibration(exceedances, resamples)
    return ResamplingTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method=method,
        alternative=_NORMALITY_ALTERNATIVE,
        data_name="x",
        statistic_name=statistic_name,
        calibration="Monte Carlo normal-null calibration",
        n_resamples=resamples,
        exceedances=exceedances,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
    )


def shapiro_wilk(x: ArrayLike) -> HypothesisTestResult:
    """Perform the Shapiro--Wilk test of univariate normality.

    Parameters
    ----------
    x
        One-dimensional sample of 3 to 5,000 finite real observations. The
        sample must not be constant.

    Returns
    -------
    HypothesisTestResult
        The Shapiro--Wilk ``W`` statistic and its approximate p-value.

    Notes
    -----
    SciPy's implementation of the Shapiro--Wilk statistic and Royston p-value
    approximation is used after a location/scale normalization. The 5,000
    observation limit is enforced because the p-value approximation is not
    validated beyond it.

    References
    ----------
    Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance test for
    normality (complete samples). *Biometrika*, 52, 591--611.
    """
    standardized = _standardized_sample(x, minimum_size=3, maximum_size=5_000)
    scipy_result = stats.shapiro(standardized)
    return HypothesisTestResult(
        statistic=float(scipy_result.statistic),
        pvalue=float(scipy_result.pvalue),
        method="Shapiro-Wilk test for normality (1965)",
        alternative=_NORMALITY_ALTERNATIVE,
        data_name="x",
        statistic_name="W",
        calibration="Royston p-value approximation",
    )


def shapiro_francia(x: ArrayLike) -> HypothesisTestResult:
    """Perform the Shapiro--Francia test of univariate normality.

    Parameters
    ----------
    x
        One-dimensional sample of 5 to 5,000 finite real observations. The
        sample must not be constant.

    Returns
    -------
    HypothesisTestResult
        The squared normal-score correlation ``W`` and Royston's approximate
        p-value.

    References
    ----------
    Shapiro, S. S. and Francia, R. S. (1972). An approximate analysis of
    variance test for normality. *JASA*, 67, 215--216.
    """
    standardized = _standardized_sample(x, minimum_size=5, maximum_size=5_000)
    n = standardized.size
    probabilities = (np.arange(1, n + 1, dtype=np.float64) - 3.0 / 8.0) / (
        n + 1.0 / 4.0
    )
    normal_scores = stats.norm.ppf(probabilities)
    ordered = np.sort(standardized)
    numerator = float(np.dot(ordered, normal_scores))
    denominator = math.sqrt(
        float(np.dot(ordered, ordered)) * float(np.dot(normal_scores, normal_scores))
    )
    statistic = min(1.0, max(0.0, (numerator / denominator) ** 2))

    log_n = math.log(n)
    log_log_n = math.log(log_n)
    mean = -1.2725 + 1.0521 * (log_log_n - log_n)
    standard_deviation = 1.0308 - 0.26758 * (log_log_n + 2.0 / log_n)
    log_complement = -math.inf if statistic == 1.0 else math.log1p(-statistic)
    transformed = (log_complement - mean) / standard_deviation
    pvalue = float(stats.norm.sf(transformed))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Shapiro-Francia test for normality (1972)",
        alternative=_NORMALITY_ALTERNATIVE,
        data_name="x",
        statistic_name="W",
        calibration="Royston normal approximation",
    )


def jarque_bera(
    x: ArrayLike,
    *,
    calibration: str = "monte-carlo",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> HypothesisTestResult | ResamplingTestResult:
    """Perform the Jarque--Bera omnibus test of univariate normality.

    Parameters
    ----------
    x
        One-dimensional sample with at least three finite real observations.
    calibration
        ``"monte-carlo"`` (the default) simulates the finite-sample normal
        null. ``"asymptotic"`` explicitly uses the limiting chi-square
        approximation with two degrees of freedom.
    n_resamples
        Positive number of simulated normal samples for Monte Carlo
        calibration.
    rng
        ``None``, an integer seed, or a NumPy generator.

    Notes
    -----
    Monte Carlo calibration reports the corrected p-value
    ``(b + 1) / (B + 1)``. The asymptotic option has no finite-sample size
    guarantee and is not the authoritative default.

    References
    ----------
    Jarque, C. M. and Bera, A. K. (1980). Efficient tests for normality,
    homoscedasticity and serial independence of regression residuals.
    *Economics Letters*, 6, 255--259.
    """
    standardized = _standardized_sample(x, minimum_size=3)
    statistic = _jarque_bera_statistic(standardized)
    return _moment_test_result(
        standardized,
        statistic=statistic,
        statistic_name="JB",
        method="Jarque-Bera test for normality (1980)",
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
        row_statistic=_jb_rows,
    )


def adjusted_jarque_bera(
    x: ArrayLike,
    *,
    calibration: str = "monte-carlo",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> HypothesisTestResult | ResamplingTestResult:
    """Perform Urzúa's finite-sample adjusted Jarque--Bera test.

    Parameters
    ----------
    x
        One-dimensional sample with at least four finite real observations.
    calibration
        ``"monte-carlo"`` (the default) or ``"asymptotic"``.
    n_resamples
        Positive number of simulated normal samples for Monte Carlo
        calibration.
    rng
        ``None``, an integer seed, or a NumPy generator.

    Notes
    -----
    The statistic centers kurtosis at its exact normal-sample expectation and
    scales skewness and kurtosis by their exact normal-sample variances. A
    sample size of at least four is required. Monte Carlo calibration uses the
    corrected p-value ``(b + 1) / (B + 1)``.

    References
    ----------
    Urzúa, C. M. (1996). On the correct use of omnibus tests for normality.
    *Economics Letters*, 53, 247--251.
    """
    standardized = _standardized_sample(x, minimum_size=4)
    statistic = _adjusted_jarque_bera_statistic(standardized)
    return _moment_test_result(
        standardized,
        statistic=statistic,
        statistic_name="AJB",
        method="Adjusted Jarque-Bera test for normality (Urzúa, 1996)",
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
        row_statistic=_ajb_rows,
    )


def robust_jarque_bera(
    x: ArrayLike,
    *,
    c1: float = 6.0,
    c2: float = 64.0,
    calibration: str = "monte-carlo",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> HypothesisTestResult | ResamplingTestResult:
    """Perform the Gel--Gastwirth robust Jarque--Bera normality test.

    Parameters
    ----------
    x
        One-dimensional sample with at least three finite real observations.
    c1, c2
        Positive standardizing constants. The defaults 6 and 64 are the
        constants recommended by Gel and Gastwirth. Legacy SHT used 24 for
        ``c2``; that value does not reproduce the published procedure. Custom
        constants require Monte Carlo calibration.
    calibration
        ``"monte-carlo"`` (the default) or ``"asymptotic"``. The latter is
        an explicit chi-square approximation without a finite-sample size
        guarantee.
    n_resamples
        Number of simulated normal samples for Monte Carlo calibration.
    rng
        ``None``, an integer seed, or a NumPy generator.

    References
    ----------
    Gel, Y. R. and Gastwirth, J. L. (2008). A robust modification of the
    Jarque--Bera test of normality. *Economics Letters*, 99, 30--32.
    """
    standardized = _standardized_sample(x, minimum_size=3)
    constant1 = validate_real_scalar(c1, name="c1")
    constant2 = validate_real_scalar(c2, name="c2")
    if constant1 <= 0.0:
        raise ValueError("c1 must be greater than 0")
    if constant2 <= 0.0:
        raise ValueError("c2 must be greater than 0")
    selected_calibration = _validate_calibration(calibration)
    if selected_calibration == "asymptotic" and (constant1 != 6.0 or constant2 != 64.0):
        raise ValueError("asymptotic calibration requires the published c1=6 and c2=64")
    statistic = _robust_jarque_bera_statistic(standardized, c1=constant1, c2=constant2)
    return _moment_test_result(
        standardized,
        statistic=statistic,
        statistic_name="RJB",
        method="Robust Jarque-Bera test for normality (Gel-Gastwirth, 2008)",
        calibration=selected_calibration,
        n_resamples=n_resamples,
        rng=rng,
        row_statistic=lambda rows: _rjb_rows(rows, c1=constant1, c2=constant2),
    )

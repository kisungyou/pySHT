"""Goodness-of-fit tests for multivariate rectangular uniformity."""

from __future__ import annotations

import math
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, special, stats
from scipy.spatial import distance

from ._resampling import monte_carlo_calibration
from ._results import HypothesisTestResult, ResamplingTestResult
from ._validation import (
    make_generator,
    validate_2d_sample,
    validate_choice,
    validate_positive_integer,
    validate_real_scalar,
)

__all__ = ["ehy", "ym_interpoint", "ym_quantile"]


type _InterpointStatistic = Literal["q1", "q2", "q3"]
type _Calibration = Literal["asymptotic", "monte-carlo"]

_ALTERNATIVE: Final = "the distribution is not uniform on the specified hyperrectangle"
_MONTE_CARLO_BATCH_VALUES: Final = 1_000_000
_EHY_LOG_TIE_RTOL: Final = 100.0 * np.finfo(np.float64).eps


def _validate_statistic(value: str) -> _InterpointStatistic:
    return validate_choice(
        value,
        name="statistic",
        choices=("q1", "q2", "q3"),
    )


def _validate_calibration(value: str) -> _Calibration:
    return validate_choice(
        value,
        name="calibration",
        choices=("asymptotic", "monte-carlo"),
    )


def _validate_bound(
    value: ArrayLike | None,
    *,
    name: str,
    dimension: int,
    default: float,
) -> NDArray[np.float64]:
    if value is None:
        return np.full(dimension, default, dtype=np.float64)
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite real vector") from exc
    if raw.dtype == np.dtype(bool) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be a finite real vector")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"{name} must be a finite real vector")
    if raw.ndim != 1 or raw.size != dimension:
        raise ValueError(f"{name} must be a vector of length {dimension}")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real vector") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _unit_hyperrectangle(
    x: ArrayLike,
    *,
    lower: ArrayLike | None,
    upper: ArrayLike | None,
    strict_interior: bool,
    minimum_dimension: int = 2,
) -> NDArray[np.float64]:
    """Validate observations and map their declared support to the unit cube."""
    values = validate_2d_sample(x, name="x", minimum_rows=2)
    dimension = values.shape[1]
    if dimension < minimum_dimension:
        feature_word = "feature" if minimum_dimension == 1 else "features"
        raise ValueError(f"x must contain at least {minimum_dimension} {feature_word}")
    lower_bound = _validate_bound(lower, name="lower", dimension=dimension, default=0.0)
    upper_bound = _validate_bound(upper, name="upper", dimension=dimension, default=1.0)
    if np.any(lower_bound >= upper_bound):
        raise ValueError("every lower bound must be strictly less than its upper bound")

    if strict_interior:
        if np.any(values <= lower_bound) or np.any(values >= upper_bound):
            raise ValueError(
                "all observations must lie strictly inside the declared bounds"
            )
    elif np.any(values < lower_bound) or np.any(values > upper_bound):
        raise ValueError("all observations must lie within the declared bounds")

    with np.errstate(over="ignore", invalid="ignore"):
        widths = upper_bound - lower_bound
    transformed = np.empty_like(values)
    finite_width = np.isfinite(widths)
    if np.any(finite_width):
        transformed[:, finite_width] = (
            values[:, finite_width] - lower_bound[finite_width]
        ) / widths[finite_width]
    for column in np.flatnonzero(~finite_width):
        scale = max(abs(lower_bound[column]), abs(upper_bound[column]))
        scaled_lower = lower_bound[column] / scale
        scaled_upper = upper_bound[column] / scale
        transformed[:, column] = (values[:, column] / scale - scaled_lower) / (
            scaled_upper - scaled_lower
        )

    # A value already checked against the exact input bounds can stray by one
    # ulp during affine standardization. Clipping restores its mathematical
    # location without admitting an out-of-domain observation.
    np.clip(transformed, 0.0, 1.0, out=transformed)
    if strict_interior and (np.any(transformed <= 0.0) or np.any(transformed >= 1.0)):
        raise ValueError(
            "the affine transformation rounded an interior observation to a boundary"
        )
    return transformed


def _interpoint_variances(sample_size: int, dimension: int) -> tuple[float, float]:
    n = float(sample_size)
    d = float(dimension)
    variance_mean = d * (2.0 * n + 3.0) / (90.0 * n * (n - 1.0))
    if dimension == 2:
        variance_second = ((989.0 + 202.0 * (n - 2.0)) / 56_700.0) * (
            2.0 / (n * (n - 1.0))
        )
    elif dimension == 3:
        variance_second = ((37.0 + 6.0 * (n - 2.0)) / 1_050.0) * (2.0 / (n * (n - 1.0)))
    else:
        variance_second = (
            49.0 * d * d / 16_200.0
            + 101.0 * d / 37_800.0
            + 2.0 * (n - 2.0) * (d * d / 16_200.0 + 29.0 * d / 37_800.0)
        ) * (2.0 / (n * (n - 1.0)))
    return variance_mean, variance_second


def _interpoint_correlation(sample_size: int, dimension: int) -> float:
    """Return the exact null correlation of the two signed components.

    The covariance follows by applying the same overlapping-pair expansion as
    the two variances in Yang and Modarres (2017).  For one coordinate, the
    same-pair third central moment is ``11/945`` and the one-index-overlap
    covariance is ``2/945``.
    """
    n = float(sample_size)
    d = float(dimension)
    variance_mean, variance_second = _interpoint_variances(sample_size, dimension)
    covariance = (2.0 / (n * (n - 1.0))) * d * (4.0 * n + 3.0) / 945.0
    correlation = covariance / math.sqrt(variance_mean * variance_second)
    return min(float(np.nextafter(1.0, 0.0)), max(-1.0, correlation))


def _correlated_chisquare_sum_sf(statistic: float, correlation: float) -> float:
    """Survival probability of ``Z1**2 + Z2**2`` for correlated normals.

    Rotation diagonalizes the quadratic form into
    ``(1+rho) U**2 + (1-rho) V**2``.  Conditional on the polar angle of two
    independent standard normals, the squared radius is exponential, giving
    a stable bounded one-dimensional survival integral.
    """
    if statistic <= 0.0:
        return 1.0
    if math.isinf(statistic):
        return 0.0
    rho = abs(correlation)
    larger_weight = 1.0 + rho
    smaller_weight = 1.0 - rho

    def angular_survival(angle: float) -> float:
        radial_weight = (
            larger_weight * math.cos(angle) ** 2 + smaller_weight * math.sin(angle) ** 2
        )
        return math.exp(-statistic / (2.0 * radial_weight))

    integral, error = integrate.quad(
        angular_survival,
        0.0,
        0.5 * math.pi,
        epsabs=float(np.nextafter(0.0, 1.0)),
        epsrel=5.0e-13,
        limit=200,
    )
    if not math.isfinite(integral) or not math.isfinite(error) or integral < 0.0:
        raise ArithmeticError("the correlated chi-square probability is invalid")
    return float(min(1.0, max(0.0, (2.0 / math.pi) * integral)))


def _interpoint_components_from_distances(
    squared_distances: NDArray[np.float64],
    *,
    sample_size: int,
    dimension: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate Q1 and Q2 for one row or a batch of distance vectors."""
    d = float(dimension)
    expected_distance = d / 6.0
    mean_distance = np.mean(squared_distances, axis=-1, dtype=np.float64)
    mean_centered_square = np.mean(
        (squared_distances - expected_distance) ** 2,
        axis=-1,
        dtype=np.float64,
    )
    variance_mean, variance_second = _interpoint_variances(sample_size, dimension)
    q1 = (mean_distance - expected_distance) ** 2 / variance_mean
    q2 = (mean_centered_square - 7.0 * d / 180.0) ** 2 / variance_second
    return np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64)


def _select_interpoint_statistic(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
    selected: _InterpointStatistic,
) -> NDArray[np.float64]:
    if selected == "q1":
        return q1
    if selected == "q2":
        return q2
    return q1 + q2


def _interpoint_monte_carlo_exceedances(
    *,
    observed: float,
    sample_size: int,
    dimension: int,
    selected: _InterpointStatistic,
    n_resamples: int,
    generator: np.random.Generator,
) -> int:
    first, second = np.triu_indices(sample_size, k=1)
    pair_count = first.size
    values_per_replication = max(sample_size * dimension, pair_count * dimension)
    batch_size = max(1, _MONTE_CARLO_BATCH_VALUES // values_per_replication)
    remaining = n_resamples
    exceedances = 0
    while remaining:
        current = min(remaining, batch_size)
        samples = generator.random((current, sample_size, dimension))
        differences = samples[:, first, :] - samples[:, second, :]
        squared_distances = np.sum(differences * differences, axis=2)
        q1, q2 = _interpoint_components_from_distances(
            squared_distances,
            sample_size=sample_size,
            dimension=dimension,
        )
        simulated = _select_interpoint_statistic(q1, q2, selected)
        exceedances += int(np.count_nonzero(simulated >= observed))
        remaining -= current
    return exceedances


def _ehy_log_statistic(
    values: NDArray[np.float64], *, alpha: float, n_neighbors: int
) -> float:
    """Return log of the Ebner--Henze--Yukich volume-score statistic."""
    sample_size, dimension = values.shape
    differences = values[:, None, :] - values[None, :, :]
    # Sort the nonnegative coordinate contributions before reduction.  This
    # makes a fixed Monte Carlo stream exactly reproducible after a common
    # feature permutation, not merely equal up to a last-bit summation change.
    np.square(differences, out=differences)
    differences.sort(axis=2)
    squared_distances = np.sum(
        differences,
        axis=2,
        dtype=np.float64,
    )
    np.fill_diagonal(squared_distances, math.inf)
    nearest_squared = np.partition(squared_distances, n_neighbors - 1, axis=1)[
        :, :n_neighbors
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_radius = 0.5 * np.log(nearest_squared)
    log_unit_ball_volume = 0.5 * dimension * math.log(math.pi) - float(
        special.gammaln(0.5 * dimension + 1.0)
    )
    log_scores = alpha * (
        log_unit_ball_volume + math.log(sample_size) + dimension * log_radius
    )
    return float(special.logsumexp(np.sort(log_scores, axis=None)))


def _from_log_nonnegative(log_value: float) -> float:
    if log_value == -math.inf:
        return 0.0
    if log_value > math.log(np.finfo(np.float64).max):
        return math.inf
    return math.exp(log_value)


def _ehy_log_tail_contains(
    candidate: float,
    observed: float,
    *,
    lower_tail: bool,
) -> bool:
    """Include last-bit numerical ties in the selected EHY log tail."""
    direct = candidate <= observed if lower_tail else candidate >= observed
    if direct or not (math.isfinite(candidate) and math.isfinite(observed)):
        return direct
    tolerance = _EHY_LOG_TIE_RTOL * max(1.0, abs(candidate), abs(observed))
    return bool(
        candidate <= observed + tolerance
        if lower_tail
        else candidate >= observed - tolerance
    )


def ehy(
    x: ArrayLike,
    *,
    alpha: float,
    n_neighbors: int,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    calibration: str = "monte-carlo",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Perform the EHY nearest-neighbor test of rectangular uniformity.

    ``alpha`` and ``n_neighbors`` are scientifically consequential tuning
    parameters and therefore have no data-selected defaults.  The paper's
    statistic rejects in the lower tail for ``0 < alpha < 1`` and in the upper
    tail for ``alpha > 1``.  ``alpha=1`` is rejected because its probability
    limit is distribution-free and cannot identify non-uniform alternatives.
    """
    validate_choice(calibration, name="calibration", choices=("monte-carlo",))
    power = validate_real_scalar(alpha, name="alpha")
    if power <= 0.0:
        raise ValueError("alpha must be greater than 0")
    if power == 1.0:
        raise ValueError("alpha must not equal 1")
    neighbors = validate_positive_integer(n_neighbors, name="n_neighbors")
    values = _unit_hyperrectangle(
        x,
        lower=lower,
        upper=upper,
        strict_interior=False,
        minimum_dimension=1,
    )
    sample_size, dimension = values.shape
    if neighbors >= sample_size:
        raise ValueError("n_neighbors must be smaller than the sample size")
    resamples = validate_positive_integer(n_resamples, name="n_resamples")
    generator = make_generator(rng)

    observed_log = _ehy_log_statistic(values, alpha=power, n_neighbors=neighbors)
    lower_tail = power < 1.0
    exceedances = 0
    for _ in range(resamples):
        simulated = generator.random((sample_size, dimension))
        simulated_log = _ehy_log_statistic(
            simulated, alpha=power, n_neighbors=neighbors
        )
        exceedances += int(
            _ehy_log_tail_contains(
                simulated_log,
                observed_log,
                lower_tail=lower_tail,
            )
        )
    pvalue, standard_error, interval = monte_carlo_calibration(exceedances, resamples)
    return ResamplingTestResult(
        statistic=_from_log_nonnegative(observed_log),
        pvalue=pvalue,
        method=(
            "Ebner-Henze-Yukich nearest-neighbor test for rectangular uniformity (2018)"
        ),
        alternative=_ALTERNATIVE,
        data_name="x",
        statistic_name="T_alpha,n,J",
        calibration="Monte Carlo rectangular-uniform null calibration",
        n_resamples=resamples,
        exceedances=exceedances,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
        diagnostics=(
            ("alpha", power),
            ("n_neighbors", neighbors),
            ("rejection tail", "lower" if lower_tail else "upper"),
        ),
    )


def ym_interpoint(
    x: ArrayLike,
    *,
    statistic: str = "q1",
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
    calibration: str = "monte-carlo",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> HypothesisTestResult | ResamplingTestResult:
    """Perform a Yang--Modarres interpoint-distance uniformity test.

    Parameters
    ----------
    x
        An ``(n, d)`` matrix with ``n >= 2`` and ``d >= 2``. Rows are
        observations.
    statistic
        ``"q1"`` tests the mean squared interpoint distance, ``"q2"`` tests
        its centered second moment, and ``"q3"`` sums the two standardized
        components.
    lower, upper
        Finite length-``d`` support bounds. Defaults are zero and one in every
        coordinate. Observations may lie on the boundary.
    calibration
        ``"monte-carlo"`` (the default) simulates the finite-sample null.
        ``"asymptotic"`` uses the paper's one-component chi-square limits for
        ``"q1"`` and ``"q2"``. For ``"q3"`` it uses the correlated-square
        limit implied by the exact nonzero covariance of those components;
        it does not use the paper's invalid independence approximation.
    n_resamples
        Positive number of simulated uniform samples for Monte Carlo
        calibration.
    rng
        ``None``, an integer seed, or a NumPy generator. It never affects
        NumPy's global random state.

    Notes
    -----
    The one-component chi-square calibrations follow the U-process limits in
    Yang and Modarres (2017). For ``q3``, the two signed standardized
    components are not independent: pySHT uses their exact null covariance in
    the corresponding correlated-normal-square approximation instead of the
    paper's anti-conservative chi-square approximation. Monte Carlo remains
    the authoritative finite-sample default for all variants.

    References
    ----------
    Yang, M. and Modarres, R. (2017). Multivariate tests of uniformity.
    *Statistical Papers*, 58, 627--639.
    """
    selected = _validate_statistic(statistic)
    selected_calibration = _validate_calibration(calibration)
    resamples = validate_positive_integer(n_resamples, name="n_resamples")
    generator = make_generator(rng)
    values = _unit_hyperrectangle(x, lower=lower, upper=upper, strict_interior=False)
    sample_size, dimension_int = values.shape

    squared_distances = distance.pdist(values, metric="sqeuclidean")
    q1, q2 = _interpoint_components_from_distances(
        squared_distances,
        sample_size=sample_size,
        dimension=dimension_int,
    )
    selected_value = _select_interpoint_statistic(q1, q2, selected)
    test_statistic = float(selected_value)
    method_name = (
        "Yang-Modarres interpoint-distance test for rectangular uniformity (2017)"
    )
    if selected_calibration == "asymptotic":
        if selected == "q3":
            correlation = _interpoint_correlation(sample_size, dimension_int)
            return HypothesisTestResult(
                statistic=test_statistic,
                pvalue=_correlated_chisquare_sum_sf(test_statistic, correlation),
                method=method_name,
                alternative=_ALTERNATIVE,
                data_name="x",
                statistic_name=selected.upper(),
                calibration=("correlated-normal-square asymptotic approximation"),
                diagnostics=(("signed-component correlation", correlation),),
            )
        return HypothesisTestResult(
            statistic=test_statistic,
            pvalue=float(stats.chi2.sf(test_statistic, 1.0)),
            method=method_name,
            alternative=_ALTERNATIVE,
            data_name="x",
            statistic_name=selected.upper(),
            calibration="asymptotic chi-square approximation",
            df=1.0,
        )

    exceedances = _interpoint_monte_carlo_exceedances(
        observed=test_statistic,
        sample_size=sample_size,
        dimension=dimension_int,
        selected=selected,
        n_resamples=resamples,
        generator=generator,
    )
    pvalue, standard_error, interval = monte_carlo_calibration(exceedances, resamples)
    return ResamplingTestResult(
        statistic=test_statistic,
        pvalue=pvalue,
        method=method_name,
        alternative=_ALTERNATIVE,
        data_name="x",
        statistic_name=selected.upper(),
        calibration="Monte Carlo rectangular-uniform null calibration",
        n_resamples=resamples,
        exceedances=exceedances,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
    )


def ym_quantile(
    x: ArrayLike,
    *,
    lower: ArrayLike | None = None,
    upper: ArrayLike | None = None,
) -> HypothesisTestResult:
    """Perform the Yang--Modarres normal-quantile uniformity test.

    Every observation must lie strictly inside its declared bounds because a
    boundary point maps to an infinite normal quantile. Under rectangular
    uniformity, coordinatewise normal quantiles have mean zero and the
    statistic ``Cn = n ||mean(Z)||^2`` has a chi-square distribution with
    ``d`` degrees of freedom.

    References
    ----------
    Yang, M. and Modarres, R. (2017). Multivariate tests of uniformity.
    *Statistical Papers*, 58, 627--639.
    """
    values = _unit_hyperrectangle(x, lower=lower, upper=upper, strict_interior=True)
    sample_size, dimension = values.shape
    normal_quantiles = stats.norm.ppf(values)
    if not np.all(np.isfinite(normal_quantiles)):
        raise ValueError("normal-quantile transformation produced nonfinite values")
    mean_quantile = np.mean(normal_quantiles, axis=0, dtype=np.float64)
    test_statistic = float(sample_size * np.dot(mean_quantile, mean_quantile))
    degrees_of_freedom = float(dimension)
    return HypothesisTestResult(
        statistic=test_statistic,
        pvalue=float(stats.chi2.sf(test_statistic, degrees_of_freedom)),
        method=("Yang-Modarres normal-quantile test for rectangular uniformity (2017)"),
        alternative=_ALTERNATIVE,
        data_name="x",
        statistic_name="Cn",
        calibration="exact chi-square null after the normal-quantile transform",
        df=degrees_of_freedom,
    )

"""Tests for equality of probability distributions.

The public functions use exact or corrected Monte Carlo randomization.  A
private Ball Divergence research prototype is retained below, but is excluded
from :data:`__all__` because floating-point equality of computed ball radii
does not yet have a correctness-certified implementation.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._distance_kernel import (
    as_sample,
    calibrate_groups,
    canonical_groups,
    distance_geometry,
    kernel_matrix,
    pooled_groups,
)
from ._resampling import monte_carlo_calibration
from ._results import DistanceTestResult, ResamplingTestResult
from ._validation import (
    make_generator,
    validate_choice,
    validate_positive_integer,
    validate_real_scalar,
)

__all__ = ["bg_2samp", "energy_ksamp", "mmd_2samp"]


_METHOD: Final = "Biswas-Ghosh two-sample test (2014)"
_ALTERNATIVE: Final = "the two distributions are not equal"
_PERMUTATION_TIE_RTOL: Final = 100.0 * np.finfo(np.float64).eps


def _validate_sample(sample: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Validate one sample and return it as a two-dimensional float64 array."""
    raw = np.asarray(sample)
    if raw.dtype == np.dtype(bool) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric values")
    if raw.ndim == 1:
        raw = raw[:, None]
    elif raw.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if raw.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two observations")
    if raw.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")

    values = np.asarray(raw, dtype=np.float64, order="C")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")

    # Normalizing signed zeros makes the canonical byte key below independent
    # of a numerically irrelevant representation detail.
    if np.any(values == 0.0):
        values = values.copy()
        values[values == 0.0] = 0.0
    return values


def _sort_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return rows in a deterministic lexicographic order."""
    keys = tuple(values[:, column] for column in range(values.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    return np.ascontiguousarray(values[order])


def _canonical_groups(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Canonicalize group order so swapping the two samples changes no result."""
    x = _sort_rows(x)
    y = _sort_rows(y)
    if x.shape[0] > y.shape[0]:
        return y, x
    if x.shape[0] < y.shape[0]:
        return x, y
    if y.tobytes(order="C") < x.tobytes(order="C"):
        return y, x
    return x, y


def _pooled_data_and_labels(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Pool and canonicalize observations while retaining the first-group labels."""
    pooled = np.vstack((first, second))
    labels = np.zeros(pooled.shape[0], dtype=np.bool_)
    labels[: first.shape[0]] = True

    keys = tuple(pooled[:, column] for column in range(pooled.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    return np.ascontiguousarray(pooled[order]), labels[order]


def _pairwise_distances(
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Use pairwise subtraction before the common distance normalization.

    The Biswas--Ghosh statistic is homogeneous of degree two in distance, so
    the common normalization preserves its permutation ordering. Individual
    pair differences retain local spacings even in the presence of outliers.
    """
    geometry = distance_geometry(values)
    return geometry.distances, geometry.scale


def _restore_raw_statistic(normalized_statistic: float, distance_scale: float) -> float:
    """Restore the paper's raw statistic, allowing IEEE under/overflow."""
    if normalized_statistic == 0.0 or distance_scale == 0.0:
        return 0.0
    if math.isinf(distance_scale):
        return math.inf
    log_statistic = math.log(normalized_statistic) + 2.0 * math.log(distance_scale)
    maximum_log = math.log(np.finfo(np.float64).max)
    minimum_log = math.log(float(np.nextafter(0.0, 1.0)))
    if log_statistic > maximum_log:
        return math.inf
    if log_statistic < minimum_log:
        return 0.0
    return math.exp(log_statistic)


def _is_at_least_as_extreme(candidate: float, observed: float) -> bool:
    """Compare permutation statistics while retaining floating-point ties."""
    threshold = observed - abs(observed) * _PERMUTATION_TIE_RTOL
    return bool(candidate >= threshold)


def _statistic(
    distances: NDArray[np.float64],
    first_indices: NDArray[np.intp],
    second_indices: NDArray[np.intp],
) -> float:
    """Compute the Biswas-Ghosh squared distance-mean contrast."""
    first_size = first_indices.size
    second_size = second_indices.size

    first_block = distances[np.ix_(first_indices, first_indices)]
    second_block = distances[np.ix_(second_indices, second_indices)]
    cross_block = distances[np.ix_(first_indices, second_indices)]

    first_triangle = np.triu_indices(first_size, k=1)
    second_triangle = np.triu_indices(second_size, k=1)
    mean_first = float(np.mean(first_block[first_triangle], dtype=np.float64))
    mean_second = float(np.mean(second_block[second_triangle], dtype=np.float64))
    mean_cross = float(np.mean(cross_block, dtype=np.float64))

    difference_first = mean_first - mean_cross
    difference_second = mean_cross - mean_second
    with np.errstate(over="ignore", invalid="ignore"):
        statistic = float(
            difference_first * difference_first + difference_second * difference_second
        )
    if not math.isfinite(statistic):
        raise ValueError("the test statistic overflowed; rescale the observations")
    return statistic


def bg_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    calibration: str = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> DistanceTestResult:
    """Test whether two univariate or multivariate distributions are equal.

    The Biswas-Ghosh statistic compares the average within-sample and
    between-sample Euclidean distances.  Its permutation calibration is exact
    under exchangeability when all labelings are enumerated; this implementation
    uses Monte Carlo label permutations and the nonzero correction
    ``(exceedances + 1) / (n_resamples + 1)``.

    Parameters
    ----------
    x, y
        Independent samples. One-dimensional inputs represent univariate data;
        two-dimensional inputs use rows for observations and columns for
        features. The samples must have the same number of features and at
        least two observations each.
    calibration
        ``"permutation"`` (the default) automatically enumerates all labelings
        when their count does not exceed ``n_resamples`` and otherwise uses
        Monte Carlo permutations. Use ``"exact"`` to require enumeration or
        ``"monte-carlo"`` to require sampling. No asymptotic selector is
        exposed because the legacy variance calculation is not row-order
        invariant.
    n_resamples
        Monte Carlo sample size and computational budget for exact enumeration.
        Must be a positive integer.
    rng
        ``None``, an integer seed, or a :class:`numpy.random.Generator`. Exact
        enumeration validates but does not consume the generator.
    Returns
    -------
    DistanceTestResult
        An immutable result whose string form follows R's ``htest`` display.

    Notes
    -----
    The permutation null requires exchangeability of the pooled observations.
    A fixed integer seed gives identical results after reordering rows or
    swapping the two samples. Pairwise differences are formed before distances
    are divided by their maximum. This preserves local spacings around both
    huge common locations and distant outliers. Overflowing pairs use a
    logarithmic distance representation; the common normalization leaves
    permutation ordering unchanged in exact arithmetic.
    ``result.statistic`` remains the raw statistic from the paper;
    the dimensionless value and maximum-distance scale are retained as
    ``result.normalized_statistic`` and ``result.distance_scale``. The raw
    statistic may be zero or infinite when its scale lies outside float64,
    while the normalized calibration remains valid. Numerically tied
    permutation statistics are included in the upper tail using a relative
    tolerance of 100 double-precision epsilons. For Monte Carlo calibration,
    the reported standard error estimates the conditional standard deviation
    of the corrected estimator using ``q_hat = exceedances / B``. A 95 percent
    Clopper--Pearson interval records uncertainty in the underlying permutation
    tail probability.

    References
    ----------
    Biswas, M. and Ghosh, A. K. (2014). A nonparametric two-sample test
    applicable to high dimensional data. *Journal of Multivariate Analysis*,
    123, 160-171. https://doi.org/10.1016/j.jmva.2013.09.004
    """
    normalized_calibration = validate_choice(
        calibration,
        name="calibration",
        choices=("permutation", "exact", "monte-carlo"),
    )
    resamples = validate_positive_integer(n_resamples, name="n_resamples")

    first = _validate_sample(x, name="x")
    second = _validate_sample(y, name="y")
    if first.shape[1] != second.shape[1]:
        raise ValueError("x and y must have the same number of features")

    first, second = _canonical_groups(first, second)
    pooled, observed_labels = _pooled_data_and_labels(first, second)
    distances, distance_scale = _pairwise_distances(pooled)

    observed_first = np.flatnonzero(observed_labels)
    observed_second = np.flatnonzero(~observed_labels)
    observed_normalized = _statistic(distances, observed_first, observed_second)
    observed_raw = _restore_raw_statistic(observed_normalized, distance_scale)

    generator = make_generator(rng)
    pooled_size = pooled.shape[0]
    first_size = first.shape[0]
    total_labelings = math.comb(pooled_size, first_size)
    use_exact = normalized_calibration == "exact" or (
        normalized_calibration == "permutation" and total_labelings <= resamples
    )
    if normalized_calibration == "exact" and total_labelings > resamples:
        raise ValueError(
            f"exact calibration requires {total_labelings:,} labelings; "
            "increase n_resamples to at least that value"
        )

    exceedances = 0
    selected = np.zeros(pooled_size, dtype=np.bool_)
    if use_exact:
        for chosen in combinations(range(pooled_size), first_size):
            selected.fill(False)
            selected[np.fromiter(chosen, dtype=np.intp, count=first_size)] = True
            permuted = _statistic(
                distances, np.flatnonzero(selected), np.flatnonzero(~selected)
            )
            exceedances += int(_is_at_least_as_extreme(permuted, observed_normalized))
        effective_resamples = total_labelings
        pvalue = exceedances / total_labelings
        monte_carlo_standard_error = None
        tail_probability_interval = None
        calibration_label = "exact permutation"
    else:
        for _ in range(resamples):
            selected.fill(False)
            selected[generator.choice(pooled_size, size=first_size, replace=False)] = (
                True
            )
            permuted = _statistic(
                distances, np.flatnonzero(selected), np.flatnonzero(~selected)
            )
            exceedances += int(_is_at_least_as_extreme(permuted, observed_normalized))
        effective_resamples = resamples
        (
            pvalue,
            monte_carlo_standard_error,
            tail_probability_interval,
        ) = monte_carlo_calibration(
            exceedances,
            resamples,
        )
        calibration_label = "Monte Carlo permutation"

    return DistanceTestResult(
        statistic=observed_raw,
        normalized_statistic=observed_normalized,
        distance_scale=distance_scale,
        exact=use_exact,
        pvalue=pvalue,
        method=_METHOD,
        alternative=_ALTERNATIVE,
        data_name="x and y",
        statistic_name="T_mn",
        calibration=calibration_label,
        n_resamples=effective_resamples,
        exceedances=exceedances,
        monte_carlo_standard_error=monte_carlo_standard_error,
        tail_probability_interval=tail_probability_interval,
    )


def _energy_statistic(
    distances: NDArray[np.float64],
    groups: tuple[NDArray[np.intp], ...],
) -> float:
    """Use identical pair reductions for observed and resampled DISCO."""
    labels = np.empty((1, distances.shape[0]), dtype=np.intp)
    for label, indices in enumerate(groups):
        labels[0, indices] = label
    return float(
        _energy_statistic_batch(
            distances, labels, tuple(group.size for group in groups)
        )[0]
    )


def _energy_statistic_batch(
    distances: NDArray[np.float64],
    labels: NDArray[np.intp],
    sizes: tuple[int, ...],
) -> NDArray[np.float64]:
    """Evaluate DISCO from total and within dispersions with fixed sums."""
    total_size = sum(sizes)
    left, right = np.triu_indices(total_size, k=1)
    pairs = distances[left, right]
    total = float(np.sum(pairs, dtype=np.float64)) / total_size
    left_labels, right_labels = labels[:, left], labels[:, right]
    inverse_sizes = 1.0 / np.asarray(sizes, dtype=np.float64)
    weights = np.where(left_labels == right_labels, inverse_sizes[left_labels], 0.0)
    within = np.sum(np.ascontiguousarray(weights * pairs), axis=1, dtype=np.float64)
    between = total - within
    numerical_scale = np.maximum(within, max(total, np.finfo(np.float64).tiny))
    near_zero = np.abs(between) <= 500.0 * np.finfo(np.float64).eps * numerical_scale
    # Exact empirical equality implies zero dispersion, even if rounding of
    # the two positive sums would otherwise leave a small positive residue.
    # Multiplicity comparisons use integer cross-products, never a tolerance
    # that would erase a genuinely small difference between distributions.
    if np.any(near_zero):
        selected = labels[near_zero]
        same_point = distances == 0.0
        reference = np.sum((selected[:, None, :] == 0) & same_point[None, :, :], axis=2)
        identical = np.ones(selected.shape[0], dtype=np.bool_)
        for group in range(1, len(sizes)):
            counts = np.sum(
                (selected[:, None, :] == group) & same_point[None, :, :], axis=2
            )
            identical &= np.all(counts * sizes[0] == reference * sizes[group], axis=1)
        indices = np.flatnonzero(near_zero)
        between[indices[identical]] = 0.0
    between[(between < 0.0) & near_zero] = 0.0
    if np.any(between < 0.0):
        raise ValueError("the energy between-sample dispersion became negative")
    result = np.empty_like(between)
    zero_within = within == 0.0
    result[zero_within & (between == 0.0)] = 0.0
    result[zero_within & (between > 0.0)] = math.inf
    nonzero = ~zero_within
    result[nonzero] = (between[nonzero] / (len(sizes) - 1.0)) / (
        within[nonzero] / (total_size - len(sizes))
    )
    return result


def energy_ksamp(
    *samples: ArrayLike,
    exponent: float = 1.0,
    calibration: Literal["permutation", "exact", "monte-carlo"] = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Perform the DISCO energy test for equality of two or more distributions.

    The statistic is the distance-components pseudo-F ratio of Rizzo and
    Székely (2010), using Euclidean distance raised to ``exponent``.  Values
    strictly between zero and two characterize equality of distributions
    under the paper's moment conditions.  Calibration relabels the pooled
    observations while preserving every group size.

    Parameters
    ----------
    *samples
        Two or more independent samples. One-dimensional inputs are treated as
        one-feature observations. Every sample must have at least two rows and
        all samples must have the same feature count.
    exponent
        Distance exponent in the open interval ``(0, 2)``.
    calibration, n_resamples, rng
        Exact or corrected Monte Carlo fixed-size label permutation controls.

    References
    ----------
    Rizzo, M. L. and Székely, G. J. (2010). DISCO analysis: A nonparametric
    extension of analysis of variance. *Annals of Applied Statistics*, 4,
    1034--1055. https://doi.org/10.1214/09-AOAS245
    """
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    power = validate_real_scalar(exponent, name="exponent")
    if not 0.0 < power < 2.0:
        raise ValueError("exponent must be strictly between 0 and 2")
    groups = tuple(
        as_sample(sample, name=f"samples[{index}]")
        for index, sample in enumerate(samples)
    )
    feature_count = groups[0].shape[1]
    if any(group.shape[1] != feature_count for group in groups[1:]):
        raise ValueError("all samples must have the same number of features")
    canonical = canonical_groups(groups)
    pooled, observed_groups = pooled_groups(canonical)
    base_distances = distance_geometry(pooled).distances
    with np.errstate(under="ignore"):
        powered_distances = np.power(base_distances, power)
    observed = _energy_statistic(powered_distances, observed_groups)
    # F=A*(T/W-1), A=(N-K)/(K-1). Subtracting positive dispersions T,W
    # creates an absolute error proportional to A even when F is near zero.
    # Pairwise summation has logarithmic reduction depth; coordinate norms,
    # powers, weights, and the two compared ratios add bounded rounding steps.
    # Kernel statistics retain the default purely relative comparison.
    total_size = pooled.shape[0]
    reduction_depth = math.ceil(math.log2(max(2, total_size * (total_size - 1) // 2)))
    roundoff = 8.0 * np.finfo(np.float64).eps * (reduction_depth + feature_count + 4)
    factor = (total_size - len(groups)) / (len(groups) - 1.0)
    summary = calibrate_groups(
        observed=observed,
        statistic=lambda allocation: _energy_statistic(powered_distances, allocation),
        batch_statistic=lambda labels: _energy_statistic_batch(
            powered_distances,
            labels,
            tuple(group.shape[0] for group in canonical),
        ),
        sizes=tuple(group.shape[0] for group in canonical),
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
        absolute_tolerance=roundoff * factor,
        relative_tolerance=max(_PERMUTATION_TIE_RTOL, roundoff),
    )
    return ResamplingTestResult(
        statistic=observed,
        pvalue=summary.pvalue,
        method="DISCO energy k-sample test (2010)",
        alternative="at least one data-generating distribution differs",
        data_name="samples",
        statistic_name="F_alpha",
        calibration=summary.label,
        diagnostics=(
            ("groups", len(groups)),
            ("distance exponent", power),
        ),
        n_resamples=summary.n_resamples,
        exceedances=summary.exceedances,
        exact=summary.exact,
        monte_carlo_standard_error=summary.standard_error,
        tail_probability_interval=summary.interval,
    )


def _mmd_unbiased_statistic(
    gram: NDArray[np.float64],
    groups: tuple[NDArray[np.intp], ...],
) -> float:
    """Use the same pair reduction for observations and resampled labels."""
    labels = np.empty((1, gram.shape[0]), dtype=np.intp)
    for label, indices in enumerate(groups):
        labels[0, indices] = label
    return float(
        _mmd_unbiased_statistic_batch(gram, labels, (groups[0].size, groups[1].size))[0]
    )


def _mmd_unbiased_statistic_batch(
    gram: NDArray[np.float64],
    labels: NDArray[np.intp],
    sizes: tuple[int, int],
) -> NDArray[np.float64]:
    """Sum unordered pairs identically, independent of resampling batch size.

    A within-group pair appears twice in the ordered unbiased estimator;
    a cross-group pair has weight -2/(nm). Omitting the diagonal before
    summation preserves tiny kernels, and a fixed contiguous pair axis avoids
    different BLAS reduction paths for the observed and simulated statistics.
    """
    left, right = np.triu_indices(gram.shape[0], k=1)
    first_left = labels[:, left] == 0
    first_right = labels[:, right] == 0
    weights = np.where(
        first_left & first_right,
        2.0 / (sizes[0] * (sizes[0] - 1.0)),
        np.where(
            ~first_left & ~first_right,
            2.0 / (sizes[1] * (sizes[1] - 1.0)),
            -2.0 / (sizes[0] * sizes[1]),
        ),
    )
    weighted_pairs = np.ascontiguousarray(weights * gram[left, right])
    return np.sum(weighted_pairs, axis=1, dtype=np.float64)


def mmd_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    kernel: Literal["rbf", "laplacian"] = "rbf",
    bandwidth: str | float = "median",
    calibration: Literal["permutation", "exact", "monte-carlo"] = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Perform a characteristic-kernel maximum mean discrepancy test.

    pySHT reports the unequal-sample unbiased estimator ``MMD_u^2``.  The RBF
    or Laplacian Gram matrix, including a median-heuristic bandwidth when
    requested, is computed once from the pooled observations and held fixed
    across every label permutation.

    References
    ----------
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B. and Smola,
    A. (2012). A kernel two-sample test. *Journal of Machine Learning
    Research*, 13, 723--773. https://jmlr.org/papers/v13/gretton12a.html
    """
    groups = (as_sample(x, name="x"), as_sample(y, name="y"))
    if groups[0].shape[1] != groups[1].shape[1]:
        raise ValueError("x and y must have the same number of features")
    canonical = canonical_groups(groups)
    pooled, observed_groups = pooled_groups(canonical)
    gram, selected_kernel, bandwidth_mode, numeric_bandwidth = kernel_matrix(
        distance_geometry(pooled), kernel=kernel, bandwidth=bandwidth, offset="auto"
    )
    # The unbiased estimator excludes self pairs. Remove them before any
    # summation so their unit diagonal cannot erase tiny off-diagonal terms.
    np.fill_diagonal(gram, 0.0)
    observed = _mmd_unbiased_statistic(gram, observed_groups)
    summary = calibrate_groups(
        observed=observed,
        statistic=lambda allocation: _mmd_unbiased_statistic(gram, allocation),
        batch_statistic=lambda labels: _mmd_unbiased_statistic_batch(
            gram,
            labels,
            (canonical[0].shape[0], canonical[1].shape[0]),
        ),
        sizes=(canonical[0].shape[0], canonical[1].shape[0]),
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
    )
    diagnostics: tuple[tuple[str, int | float | str], ...] = (
        ("kernel", selected_kernel),
        ("bandwidth selection", bandwidth_mode),
    )
    if numeric_bandwidth is not None:
        diagnostics += (("bandwidth", numeric_bandwidth),)
    return ResamplingTestResult(
        statistic=observed,
        pvalue=summary.pvalue,
        method="maximum mean discrepancy two-sample test (2012)",
        alternative="the two distributions are not equal",
        data_name="x and y",
        statistic_name="MMD_u^2",
        calibration=summary.label,
        diagnostics=diagnostics,
        n_resamples=summary.n_resamples,
        exceedances=summary.exceedances,
        exact=summary.exact,
        monte_carlo_standard_error=summary.standard_error,
        tail_probability_interval=summary.interval,
    )


def _ball_divergence_statistic_literal(
    distances: NDArray[np.float64],
    groups: tuple[NDArray[np.intp], ...],
) -> float:
    """Evaluate Ball Divergence literally in cubic time (oracle only)."""
    first, second = groups
    total = 0.0
    for centers, radii_endpoints in ((first, first), (second, second)):
        component = 0.0
        for center in centers:
            center_distances = distances[center]
            for endpoint in radii_endpoints:
                radius = center_distances[endpoint]
                first_mass = float(
                    np.count_nonzero(center_distances[first] <= radius) / first.size
                )
                second_mass = float(
                    np.count_nonzero(center_distances[second] <= radius) / second.size
                )
                component += (first_mass - second_mass) ** 2
        total += component / (centers.size * radii_endpoints.size)
    return float(total)


def _ball_distance_orders(
    distances: NDArray[np.float64],
    *,
    feature_count: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Precompute stable distance orders and certified closed-ball tie ranks."""
    size = distances.shape[0]
    orders = np.empty((size, size), dtype=np.intp)
    final_ranks = np.empty((size, size), dtype=np.intp)
    for center in range(size):
        row = distances[center]
        order = np.argsort(row, kind="stable")
        sorted_distances = row[order]
        orders[center] = order
        sorted_final_ranks = np.empty(size, dtype=np.intp)
        start = 0
        while start < size:
            end = start
            anchor = float(sorted_distances[start])
            while end + 1 < size:
                candidate = float(sorted_distances[end + 1])
                tolerance = (
                    4.0
                    * np.finfo(np.float64).eps
                    * max(1, feature_count)
                    * max(1.0, abs(anchor), abs(candidate))
                )
                if candidate - anchor > tolerance:
                    break
                end += 1
            sorted_final_ranks[start : end + 1] = end
            start = end + 1
        inverse = np.empty(size, dtype=np.intp)
        inverse[order] = np.arange(size, dtype=np.intp)
        final_ranks[center] = sorted_final_ranks[inverse]
    return orders, final_ranks


def _ball_divergence_statistic(
    orders: NDArray[np.intp],
    final_ranks: NDArray[np.intp],
    groups: tuple[NDArray[np.intp], ...],
) -> float:
    """Evaluate Ball Divergence in quadratic time from fixed distance ranks."""
    first, second = groups
    size = orders.shape[0]
    first_labels = np.zeros(size, dtype=np.float64)
    first_labels[first] = 1.0
    second_labels = 1.0 - first_labels
    ordered_labels = first_labels[orders]
    cumulative_first = np.cumsum(ordered_labels, axis=1, dtype=np.float64)
    first_counts = np.take_along_axis(cumulative_first, final_ranks, axis=1)
    ball_sizes = final_ranks + 1
    differences = first_counts / first.size - (ball_sizes - first_counts) / second.size
    pair_weights = (
        np.multiply.outer(first_labels, first_labels) / first.size**2
        + np.multiply.outer(second_labels, second_labels) / second.size**2
    )
    return float(np.sum(pair_weights * differences**2, dtype=np.float64))


def _ball_divergence_statistic_batch(
    orders: NDArray[np.intp],
    final_ranks: NDArray[np.intp],
    labels: NDArray[np.intp],
    sizes: tuple[int, int],
) -> NDArray[np.float64]:
    """Evaluate a batch of Ball Divergence labelings in quadratic time each."""
    first = labels == 0
    second = ~first
    ordered_labels = first[:, orders]
    cumulative_first = np.cumsum(ordered_labels, axis=2, dtype=np.float64)
    ranks = np.broadcast_to(final_ranks, cumulative_first.shape)
    first_counts = np.take_along_axis(cumulative_first, ranks, axis=2)
    ball_sizes = final_ranks + 1
    differences = (
        first_counts / sizes[0] - (ball_sizes[None, :, :] - first_counts) / sizes[1]
    )
    pair_weights = (
        first[:, :, None] * first[:, None, :] / sizes[0] ** 2
        + second[:, :, None] * second[:, None, :] / sizes[1] ** 2
    )
    return np.sum(pair_weights * differences**2, axis=(1, 2), dtype=np.float64)


def _ball_divergence_2samp(
    x: ArrayLike,
    y: ArrayLike,
    *,
    calibration: Literal["permutation", "exact", "monte-carlo"] = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Evaluate the private, correctness-blocked Ball Divergence prototype.

    The empirical probability of every closed ball centered at an observation
    is compared between samples.  Tied points remain inside a closed ball;
    no random jitter is introduced. Stable distance orders and closed-ball
    tie-end ranks are computed once, after which each labeling costs quadratic
    time in the pooled sample size. This implementation is intentionally
    private: its roundoff-based tie classifier can merge distinguishable radii,
    while exact float comparisons can split mathematically equal radii after an
    isometry. A certified predicate is required before public exposure.

    References
    ----------
    Pan, W., Tian, Y., Wang, X. and Zhang, H. (2018). Ball Divergence:
    Nonparametric two sample test. *Annals of Statistics*, 46, 1109--1137.
    https://doi.org/10.1214/17-AOS1579
    """
    groups = (as_sample(x, name="x"), as_sample(y, name="y"))
    if groups[0].shape[1] != groups[1].shape[1]:
        raise ValueError("x and y must have the same number of features")
    canonical = canonical_groups(groups)
    pooled, observed_groups = pooled_groups(canonical)
    distances = distance_geometry(pooled).distances
    orders, final_ranks = _ball_distance_orders(
        distances, feature_count=pooled.shape[1]
    )
    observed = _ball_divergence_statistic(orders, final_ranks, observed_groups)
    summary = calibrate_groups(
        observed=observed,
        statistic=lambda allocation: _ball_divergence_statistic(
            orders, final_ranks, allocation
        ),
        batch_statistic=lambda labels: _ball_divergence_statistic_batch(
            orders,
            final_ranks,
            labels,
            (canonical[0].shape[0], canonical[1].shape[0]),
        ),
        sizes=(canonical[0].shape[0], canonical[1].shape[0]),
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
    )
    return ResamplingTestResult(
        statistic=observed,
        pvalue=summary.pvalue,
        method="Ball Divergence two-sample test (2018)",
        alternative="the two distributions are not equal",
        data_name="x and y",
        statistic_name="BD_nm",
        calibration=summary.label,
        diagnostics=(("metric", "Euclidean"), ("balls", "closed")),
        n_resamples=summary.n_resamples,
        exceedances=summary.exceedances,
        exact=summary.exact,
        monte_carlo_standard_error=summary.standard_error,
        tail_probability_interval=summary.interval,
    )

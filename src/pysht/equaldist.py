"""Tests for equality of two probability distributions.

This module currently provides the distance-based two-sample test of Biswas
and Ghosh (2014).  Only permutation calibration is exposed: the asymptotic
variance estimator in the legacy SHT implementation is order-dependent and is
therefore not used as a correctness oracle here.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._resampling import monte_carlo_calibration
from ._results import DistanceTestResult
from ._validation import make_generator, validate_choice, validate_positive_integer

__all__ = ["bg_2samp"]


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
    """Compute normalized Euclidean distances and their original scale.

    The Biswas--Ghosh statistic is homogeneous of degree two in distance, so a
    positive common scaling cannot affect its permutation ordering.  Dividing
    coordinates before distance evaluation avoids overflow in subtraction;
    dividing again by the largest resulting distance prevents squared
    contrasts from overflowing or underflowing during calibration.
    """
    coordinate_scale = float(np.max(np.abs(values)))
    scaled_values = values if coordinate_scale == 0.0 else values / coordinate_scale
    try:
        distances = _core.pairwise_distances(scaled_values, scaled_values)
    except OverflowError as exc:
        raise ValueError(
            "pairwise Euclidean distances overflowed; rescale the observations"
        ) from exc
    scaled_distance_max = float(np.max(distances))
    if scaled_distance_max > 0.0:
        distances /= scaled_distance_max
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        distance_scale = float(
            np.float64(coordinate_scale) * np.float64(scaled_distance_max)
        )
    return distances, distance_scale


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
    swapping the two samples. Calibration uses coordinates and distances
    divided by positive common scales. This leaves the permutation ordering
    unchanged in exact arithmetic and prevents scale-dependent overflow and
    underflow. ``result.statistic`` remains the raw statistic from the paper;
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

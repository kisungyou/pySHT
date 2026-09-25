"""Tests for circular uniformity and equality of circular distributions.

Angles are accepted in arbitrary units through the explicit ``period``
parameter and are reduced modulo that period before calculation.  The
one-sample procedures use finite-sample Monte Carlo calibration under the
continuous circular-uniform null.  The multi-sample procedure conditions on
the pooled directions and permutes group labels.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from itertools import combinations
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._resampling import exact_pvalue, monte_carlo_calibration
from ._results import ResamplingTestResult
from ._validation import (
    make_generator,
    validate_1d_sample,
    validate_choice,
    validate_positive_integer,
    validate_real_scalar,
)

__all__ = [
    "hermans_rasson",
    "mardia_watson_wheeler_ksamp",
    "rayleigh",
    "watson",
]


_TAU: Final = 2.0 * math.pi
_UNIFORM_ALTERNATIVE: Final = "the circular distribution is not uniform"
_EQUALITY_ALTERNATIVE: Final = "at least two circular distributions differ"
_TIE_RTOL: Final = 100.0 * np.finfo(np.float64).eps


def _angles(x: ArrayLike, *, name: str, period: object) -> NDArray[np.float64]:
    values = validate_1d_sample(x, name=name, minimum_size=2)
    cycle = validate_real_scalar(period, name="period")
    if cycle <= 0.0:
        raise ValueError("period must be greater than 0")
    # Divide before multiplying.  Forming ``2*pi / period`` first overflows
    # for a subnormal period and underflows for a period near float64 max.
    normalized = (np.remainder(values, cycle) / cycle) * _TAU
    np.minimum(normalized, np.nextafter(_TAU, 0.0), out=normalized)
    # Canonical order makes seeded calibration independent of row order and
    # normalizes the irrelevant distinction between positive and negative zero.
    normalized[normalized == 0.0] = 0.0
    return np.sort(normalized)


def _rayleigh_statistic(theta: NDArray[np.float64]) -> float:
    resultant_cosine = float(np.sum(np.cos(theta), dtype=np.float64))
    resultant_sine = float(np.sum(np.sin(theta), dtype=np.float64))
    return (
        resultant_cosine * resultant_cosine + resultant_sine * resultant_sine
    ) / theta.size


def _watson_statistic(theta: NDArray[np.float64]) -> float:
    uniform_scores = np.sort(np.remainder(theta, _TAU) / _TAU)
    n = uniform_scores.size
    expected = (2.0 * np.arange(1, n + 1, dtype=np.float64) - 1.0) / (2.0 * n)
    centered = uniform_scores - expected
    centered -= float(np.mean(centered, dtype=np.float64))
    return float(np.dot(centered, centered) + 1.0 / (12.0 * n))


def _hermans_rasson_statistic(theta: NDArray[np.float64]) -> float:
    differences = np.abs(theta[:, None] - theta[None, :])
    kernel = (
        np.abs(differences - math.pi)
        - math.pi / 2.0
        - 2.895 * (np.abs(np.sin(differences)) - 2.0 / math.pi)
    )
    return float(np.sum(kernel, dtype=np.float64) / theta.size)


def _uniform_monte_carlo(
    theta: NDArray[np.float64],
    *,
    statistic: Callable[[NDArray[np.float64]], float],
    statistic_name: str,
    method: str,
    alternative: str = _UNIFORM_ALTERNATIVE,
    n_resamples: object,
    rng: int | np.integer | np.random.Generator | None,
    estimates: tuple[tuple[str, float], ...] = (),
    diagnostics: tuple[tuple[str, bool | int | float | str], ...] = (),
) -> ResamplingTestResult:
    resamples = validate_positive_integer(n_resamples, name="n_resamples")
    generator = make_generator(rng)
    observed = statistic(theta)
    exceedances = 0
    for _ in range(resamples):
        simulated = generator.uniform(0.0, _TAU, size=theta.size)
        candidate = statistic(simulated)
        # The public input is wrapped and canonically sorted whereas a null
        # draw need not be.  Include last-bit summation/reduction differences
        # in the mathematical tie rather than occasionally excluding an
        # identical seeded null sample from its own upper tail.
        tolerance = _TIE_RTOL * max(1.0, abs(candidate), abs(observed))
        exceedances += int(candidate >= observed - tolerance)
    pvalue, standard_error, interval = monte_carlo_calibration(exceedances, resamples)
    return ResamplingTestResult(
        statistic=observed,
        pvalue=pvalue,
        method=method,
        alternative=alternative,
        data_name="x",
        statistic_name=statistic_name,
        calibration="Monte Carlo circular-uniform null calibration",
        n_resamples=resamples,
        exceedances=exceedances,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
        estimates=estimates,
        diagnostics=diagnostics,
    )


def rayleigh(
    x: ArrayLike,
    *,
    period: float = _TAU,
    calibration: str = "monte-carlo",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Perform Rayleigh's test against first-harmonic circular alternatives.

    This is a targeted test for a nonzero first trigonometric moment, not an
    omnibus test of circular uniformity.  Large values of ``n Rbar**2`` are
    significant.  Monte Carlo calibration is exact up to simulation error for
    every sample size under the continuous circular-uniform null.
    """
    validate_choice(calibration, name="calibration", choices=("monte-carlo",))
    cycle = validate_real_scalar(period, name="period")
    if cycle <= 0.0:
        raise ValueError("period must be greater than 0")
    theta = _angles(x, name="x", period=cycle)
    cosine_mean = float(np.mean(np.cos(theta), dtype=np.float64))
    sine_mean = float(np.mean(np.sin(theta), dtype=np.float64))
    resultant_length = math.hypot(cosine_mean, sine_mean)
    direction_defined = resultant_length > 64.0 * np.finfo(np.float64).eps
    estimates: tuple[tuple[str, float], ...]
    diagnostics: tuple[tuple[str, bool | int | float | str], ...]
    if direction_defined:
        mean_direction_radians = math.atan2(sine_mean, cosine_mean) % _TAU
        mean_direction = (mean_direction_radians / _TAU) * cycle
        estimates = (
            ("mean direction", mean_direction),
            ("mean resultant length", resultant_length),
        )
        diagnostics = ()
    else:
        estimates = (("mean resultant length", resultant_length),)
        diagnostics = (("mean direction", "undefined (zero resultant)"),)
    return _uniform_monte_carlo(
        theta,
        statistic=_rayleigh_statistic,
        statistic_name="Z",
        method="Rayleigh test for circular uniformity",
        alternative=(
            "the first trigonometric moment is nonzero "
            "(a preferred first-harmonic direction exists)"
        ),
        n_resamples=n_resamples,
        rng=rng,
        estimates=estimates,
        diagnostics=diagnostics,
    )


def watson(
    x: ArrayLike,
    *,
    period: float = _TAU,
    calibration: str = "monte-carlo",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Perform Watson's rotation-invariant ``U^2`` uniformity test (1961)."""
    validate_choice(calibration, name="calibration", choices=("monte-carlo",))
    theta = _angles(x, name="x", period=period)
    return _uniform_monte_carlo(
        theta,
        statistic=_watson_statistic,
        statistic_name="U^2",
        method="Watson U^2 test for circular uniformity (1961)",
        n_resamples=n_resamples,
        rng=rng,
    )


def hermans_rasson(
    x: ArrayLike,
    *,
    period: float = _TAU,
    calibration: str = "monte-carlo",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Perform the modified Hermans--Rasson omnibus test (1985).

    The pairwise Sobolev kernel uses the paper's recommended coefficient
    ``2.895``.  This version is sensitive to both unimodal and multimodal
    departures and rejects for large statistic values.
    """
    validate_choice(calibration, name="calibration", choices=("monte-carlo",))
    theta = _angles(x, name="x", period=period)
    return _uniform_monte_carlo(
        theta,
        statistic=_hermans_rasson_statistic,
        statistic_name="T_HR",
        method="Modified Hermans-Rasson circular-uniformity test (1985)",
        n_resamples=n_resamples,
        rng=rng,
    )


def _canonical_circular_groups(
    samples: tuple[ArrayLike, ...], *, period: object
) -> tuple[NDArray[np.float64], ...]:
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    groups = tuple(
        _angles(sample, name=f"samples[{index}]", period=period)
        for index, sample in enumerate(samples)
    )
    return tuple(
        sorted(groups, key=lambda group: (group.size, group.tobytes(order="C")))
    )


def _mww_statistic(
    cosine_scores: NDArray[np.float64],
    sine_scores: NDArray[np.float64],
    labels: NDArray[np.intp],
    sizes: tuple[int, ...],
) -> float:
    statistic = 0.0
    for group_index, size in enumerate(sizes):
        selected = labels == group_index
        cosine_sum = float(np.sum(cosine_scores[selected], dtype=np.float64))
        sine_sum = float(np.sum(sine_scores[selected], dtype=np.float64))
        statistic += (cosine_sum * cosine_sum + sine_sum * sine_sum) / size
    return 2.0 * statistic


def _allocation_count(sizes: tuple[int, ...]) -> int:
    remaining = sum(sizes)
    count = 1
    for size in sizes[:-1]:
        count *= math.comb(remaining, size)
        remaining -= size
    return count


def _label_allocations(sizes: tuple[int, ...]) -> Iterator[NDArray[np.intp]]:
    total = sum(sizes)
    labels = np.empty(total, dtype=np.intp)

    def visit(remaining: tuple[int, ...], group: int) -> Iterator[NDArray[np.intp]]:
        if group == len(sizes) - 1:
            labels[np.fromiter(remaining, dtype=np.intp)] = group
            yield labels.copy()
            return
        for chosen in combinations(remaining, sizes[group]):
            chosen_set = set(chosen)
            labels[np.fromiter(chosen, dtype=np.intp)] = group
            next_remaining = tuple(
                value for value in remaining if value not in chosen_set
            )
            yield from visit(next_remaining, group + 1)

    return visit(tuple(range(total)), 0)


def mardia_watson_wheeler_ksamp(
    *samples: ArrayLike,
    period: float = _TAU,
    calibration: str = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Test equality of two or more continuous circular distributions.

    Pooled circular ranks are converted to uniform trigonometric scores and
    group labels are permuted with their original sizes fixed.  Exact mode
    enumerates every ordered allocation.  Tied directions are rejected because
    the continuous-rank statistic has no uniquely specified grouped-data
    correction in this API.
    """
    selected_calibration = validate_choice(
        calibration,
        name="calibration",
        choices=("permutation", "exact", "monte-carlo"),
    )
    resamples = validate_positive_integer(n_resamples, name="n_resamples")
    groups = _canonical_circular_groups(samples, period=period)
    sizes = tuple(group.size for group in groups)
    pooled = np.concatenate(groups)
    order = np.argsort(pooled, kind="stable")
    ordered = pooled[order]
    if np.any(np.diff(ordered) == 0.0):
        raise ValueError(
            "pooled directions must be distinct; grouped circular ties require "
            "an explicit correction that this test does not apply"
        )

    original_labels = np.concatenate(
        [np.full(size, index, dtype=np.intp) for index, size in enumerate(sizes)]
    )[order]
    total = pooled.size
    ranks = np.arange(1, total + 1, dtype=np.float64)
    cosine_scores = np.cos(_TAU * ranks / total)
    sine_scores = np.sin(_TAU * ranks / total)
    observed = _mww_statistic(cosine_scores, sine_scores, original_labels, sizes)

    total_allocations = _allocation_count(sizes)
    generator = make_generator(rng)
    use_exact = selected_calibration == "exact" or (
        selected_calibration == "permutation" and total_allocations <= resamples
    )
    if selected_calibration == "exact" and total_allocations > resamples:
        raise ValueError(
            f"exact calibration requires {total_allocations:,} allocations; "
            "increase n_resamples to at least that value"
        )

    exceedances = 0
    # Trigonometric score cancellation can leave a mathematically zero W a
    # few ulps above zero after a rotation. Use a statistic-scale tolerance,
    # not one purely relative to the observed value, so equivalent null ties
    # receive the same tail count.
    threshold = observed - _TIE_RTOL * max(1.0, abs(observed))
    if use_exact:
        for labels in _label_allocations(sizes):
            statistic = _mww_statistic(cosine_scores, sine_scores, labels, sizes)
            exceedances += int(statistic >= threshold)
        effective_resamples = total_allocations
        pvalue = exact_pvalue(exceedances, total_allocations)
        standard_error = None
        interval = None
        calibration_label = "exact permutation"
    else:
        base_labels = np.concatenate(
            [np.full(size, index, dtype=np.intp) for index, size in enumerate(sizes)]
        )
        for _ in range(resamples):
            labels = generator.permutation(base_labels)
            statistic = _mww_statistic(cosine_scores, sine_scores, labels, sizes)
            exceedances += int(statistic >= threshold)
        effective_resamples = resamples
        pvalue, standard_error, interval = monte_carlo_calibration(
            exceedances, resamples
        )
        calibration_label = "Monte Carlo permutation"

    return ResamplingTestResult(
        statistic=observed,
        pvalue=pvalue,
        method="Mardia-Watson-Wheeler circular k-sample test (1972)",
        alternative=_EQUALITY_ALTERNATIVE,
        data_name="samples",
        statistic_name="W",
        calibration=calibration_label,
        n_resamples=effective_resamples,
        exceedances=exceedances,
        exact=use_exact,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
        diagnostics=(("groups", len(groups)),),
    )

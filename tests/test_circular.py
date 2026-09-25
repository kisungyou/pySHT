"""Formula, invariance, calibration, and boundary tests for circular data."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pytest

from pysht._results import ResamplingTestResult
from pysht.circular import (
    hermans_rasson,
    mardia_watson_wheeler_ksamp,
    rayleigh,
    watson,
)


def _watson_literal(theta: np.ndarray) -> float:
    scores = np.sort(np.remainder(theta, 2.0 * math.pi) / (2.0 * math.pi))
    n = scores.size
    centered = scores - (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n)
    return float(np.sum((centered - np.mean(centered)) ** 2) + 1.0 / (12.0 * n))


def _hr_literal(theta: np.ndarray) -> float:
    total = 0.0
    for first in theta:
        for second in theta:
            difference = abs(first - second)
            total += (
                abs(difference - math.pi)
                - math.pi / 2.0
                - 2.895 * (abs(math.sin(first - second)) - 2.0 / math.pi)
            )
    return total / theta.size


def test_one_sample_statistics_match_literal_equations() -> None:
    theta = np.array([0.0, 0.4, 1.7, 3.2, 5.8])
    cosine = float(np.sum(np.cos(theta)))
    sine = float(np.sum(np.sin(theta)))
    expected_rayleigh = (cosine * cosine + sine * sine) / theta.size

    rayleigh_result = rayleigh(theta, n_resamples=17, rng=1)
    watson_result = watson(theta, n_resamples=17, rng=1)
    hr_result = hermans_rasson(theta, n_resamples=17, rng=1)

    np.testing.assert_allclose(rayleigh_result.statistic, expected_rayleigh, rtol=2e-15)
    np.testing.assert_allclose(watson_result.statistic, _watson_literal(theta))
    np.testing.assert_allclose(hr_result.statistic, _hr_literal(theta), rtol=2e-15)
    assert rayleigh_result.statistic_name == "Z"
    assert "first trigonometric moment" in rayleigh_result.alternative
    assert "not uniform" not in rayleigh_result.alternative
    assert dict(rayleigh_result.estimates)["mean resultant length"] <= 1.0


@pytest.mark.parametrize("function", [rayleigh, watson, hermans_rasson])
def test_one_sample_tests_preserve_circular_units_and_symmetries(
    function: object,
) -> None:
    theta = np.array([0.1, 0.6, 1.4, 2.9, 4.2, 5.7])
    baseline = function(theta, n_resamples=79, rng=812)  # type: ignore[operator]
    variants = (
        function(theta[::-1], n_resamples=79, rng=812),  # type: ignore[operator]
        function(theta + 17.0 * math.pi, n_resamples=79, rng=812),  # type: ignore[operator]
        function(-theta, n_resamples=79, rng=812),  # type: ignore[operator]
        function(np.degrees(theta), period=360.0, n_resamples=79, rng=812),  # type: ignore[operator]
    )
    for actual in variants:
        np.testing.assert_allclose(actual.statistic, baseline.statistic, atol=2e-13)
        assert actual.exceedances == baseline.exceedances
        assert actual.pvalue == baseline.pvalue


@pytest.mark.parametrize(
    "function,seed",
    [(rayleigh, 2), (watson, 2), (hermans_rasson, 0)],
)
def test_identical_seeded_null_draw_is_counted_as_a_tie(
    function: object,
    seed: int,
) -> None:
    # The observed path canonicalizes and wraps its input, whereas the null
    # path receives generator output directly.  Their last-bit reductions may
    # differ, but an identical mathematical sample must remain in its own tail.
    sample = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi, size=37)
    result = function(sample, n_resamples=1, rng=seed)  # type: ignore[operator]
    assert result.exceedances == 1
    assert result.pvalue == 1.0


def test_period_normalization_handles_huge_and_subnormal_cycles() -> None:
    fractions = np.array([0.0, 0.125, 0.375, 0.75])
    baseline = rayleigh(fractions, period=1.0, n_resamples=31, rng=4)

    huge_period = float(np.finfo(np.float64).max)
    huge = rayleigh(
        fractions * huge_period,
        period=huge_period,
        n_resamples=31,
        rng=4,
    )
    unit = float(np.nextafter(0.0, 1.0))
    tiny_period = 8.0 * unit
    tiny = rayleigh(
        np.array([0.0, unit, 3.0 * unit, 6.0 * unit]),
        period=tiny_period,
        n_resamples=31,
        rng=4,
    )
    np.testing.assert_allclose(huge.statistic, baseline.statistic)
    np.testing.assert_allclose(tiny.statistic, baseline.statistic)
    assert huge.exceedances == tiny.exceedances == baseline.exceedances
    assert math.isfinite(dict(huge.estimates)["mean direction"])


def test_rayleigh_omits_undefined_mean_direction() -> None:
    result = rayleigh([0.0, math.pi], n_resamples=11, rng=2)
    assert "mean direction" not in dict(result.estimates)
    assert dict(result.diagnostics)["mean direction"] == ("undefined (zero resultant)")


def _mww_literal(theta: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(theta)
    ordered_labels = labels[order]
    ranks = np.arange(1, theta.size + 1)
    cosine = np.cos(2.0 * math.pi * ranks / theta.size)
    sine = np.sin(2.0 * math.pi * ranks / theta.size)
    total = 0.0
    for group in np.unique(labels):
        selected = ordered_labels == group
        total += (np.sum(cosine[selected]) ** 2 + np.sum(sine[selected]) ** 2) / np.sum(
            selected
        )
    return float(2.0 * total)


def test_mww_statistic_and_exact_tail_match_literal_enumeration() -> None:
    first = np.array([0.0, 0.5, 1.0])
    second = np.array([2.0, 2.5, 3.0])
    pooled = np.concatenate((first, second))
    observed_labels = np.array([0, 0, 0, 1, 1, 1])
    expected = _mww_literal(pooled, observed_labels)
    statistics = []
    for chosen in itertools.combinations(range(6), 3):
        labels = np.ones(6, dtype=int)
        labels[list(chosen)] = 0
        statistics.append(_mww_literal(pooled, labels))
    expected_exceedances = sum(value >= expected - 1e-13 for value in statistics)

    result = mardia_watson_wheeler_ksamp(
        first, second, calibration="exact", n_resamples=20
    )
    np.testing.assert_allclose(result.statistic, expected)
    assert result.exceedances == expected_exceedances
    assert result.pvalue == expected_exceedances / 20.0
    assert result.exact


def test_mww_is_group_rotation_reflection_and_unit_invariant() -> None:
    first = np.array([0.1, 0.8, 1.2, 2.0])
    second = np.array([2.5, 3.1, 4.4, 5.6])
    baseline = mardia_watson_wheeler_ksamp(
        first, second, calibration="monte-carlo", n_resamples=99, rng=91
    )
    variants = (
        mardia_watson_wheeler_ksamp(
            second[::-1], first[::-1], calibration="monte-carlo", n_resamples=99, rng=91
        ),
        mardia_watson_wheeler_ksamp(
            first + 4.7,
            second + 4.7,
            calibration="monte-carlo",
            n_resamples=99,
            rng=91,
        ),
        mardia_watson_wheeler_ksamp(
            -first, -second, calibration="monte-carlo", n_resamples=99, rng=91
        ),
        mardia_watson_wheeler_ksamp(
            np.degrees(first),
            np.degrees(second),
            period=360.0,
            calibration="monte-carlo",
            n_resamples=99,
            rng=91,
        ),
    )
    for actual in variants:
        np.testing.assert_allclose(actual.statistic, baseline.statistic, atol=2e-14)
        assert actual.exceedances == baseline.exceedances
        assert actual.pvalue == baseline.pvalue


@pytest.mark.parametrize(
    "calibration,n_resamples",
    [("exact", 70), ("monte-carlo", 99)],
)
def test_mww_zero_statistic_tail_is_rotation_stable(
    calibration: str, n_resamples: int
) -> None:
    first = np.array([6.20546726, 29.07897202, -8.30783528, 18.78956075])
    second = np.array([-10.89605293, 17.95281133, 6.04403513, -17.01865829])
    baseline = mardia_watson_wheeler_ksamp(
        first,
        second,
        calibration=calibration,
        n_resamples=n_resamples,
        rng=51539,
    )
    shifted = mardia_watson_wheeler_ksamp(
        second[::-1] - 18.23643656364383,
        first[::-1] - 18.23643656364383,
        calibration=calibration,
        n_resamples=n_resamples,
        rng=51539,
    )
    assert baseline.exceedances == shifted.exceedances
    assert baseline.pvalue == shifted.pvalue == 1.0


def test_circular_controls_and_domains_are_strict() -> None:
    with pytest.raises(ValueError, match="distinct"):
        mardia_watson_wheeler_ksamp([0.0, 1.0], [0.0, 2.0])
    with pytest.raises(ValueError, match="at least two samples"):
        mardia_watson_wheeler_ksamp([0.0, 1.0])
    with pytest.raises(ValueError, match="requires 20"):
        mardia_watson_wheeler_ksamp(
            [0.0, 0.5, 1.0], [2.0, 2.5, 3.0], calibration="exact", n_resamples=19
        )
    for invalid in (0.0, -1.0, math.inf):
        with pytest.raises(ValueError):
            rayleigh([0.0, 1.0], period=invalid)
    with pytest.raises(ValueError, match="calibration"):
        watson([0.0, 1.0], calibration="asymptotic")
    with pytest.raises(TypeError):
        hermans_rasson([0.0, 1.0], n_resamples=True)  # type: ignore[arg-type]


def test_seeded_result_contract_and_clear_alternatives() -> None:
    clustered = np.linspace(-0.04, 0.04, 20)
    result = rayleigh(clustered, n_resamples=499, rng=72)
    assert isinstance(result, ResamplingTestResult)
    assert result.pvalue < 0.01
    assert result.pvalue == (result.exceedances + 1) / 500
    assert result.tail_probability_interval is not None

    bimodal = np.concatenate(
        (np.linspace(-0.05, 0.05, 12), np.linspace(3.09, 3.19, 12))
    )
    assert watson(bimodal, n_resamples=499, rng=72).pvalue < 0.05
    assert hermans_rasson(bimodal, n_resamples=499, rng=72).pvalue < 0.05

    separated = mardia_watson_wheeler_ksamp(
        np.linspace(-0.08, 0.08, 5),
        np.linspace(math.pi - 0.08, math.pi + 0.08, 5),
        calibration="exact",
        n_resamples=252,
    )
    assert separated.pvalue < 0.05


def test_default_monte_carlo_budget_is_practical() -> None:
    sample = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
    started = time.perf_counter()
    result = hermans_rasson(sample, rng=10)
    elapsed = time.perf_counter() - started
    assert result.n_resamples == 9_999
    # This is intentionally generous for cross-platform CI while guarding an
    # accidental cubic implementation or one Python allocation per pair.
    assert elapsed < 5.0

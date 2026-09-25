"""Equation, exact-orbit, invariance, and zero tests for simplex additions."""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pytest
from scipy import special
from scipy.spatial import distance

from pysht import simplex
from pysht.simplex import _alpha_transform, alpha_energy_ksamp, ehy_uniformity


def _ehy_literal(x: np.ndarray, *, alpha: float, neighbors: int) -> float:
    n, components = x.shape
    dimension = components - 1
    radii = np.sort(distance.squareform(distance.pdist(x)), axis=1)[
        :, 1 : neighbors + 1
    ]
    unit_ball = math.pi ** (dimension / 2.0) / special.gamma(dimension / 2.0 + 1.0)
    density = math.factorial(dimension) / math.sqrt(components)
    return float(np.sum((unit_ball * n * density * radii**dimension) ** alpha))


def _alpha_coordinates(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0.0:
        logs = np.log(x)
        return logs - np.mean(logs, axis=1, keepdims=True)
    powered = x**alpha
    u = powered / np.sum(powered, axis=1, keepdims=True)
    return (x.shape[1] * u - 1.0) / alpha


def _energy(x: np.ndarray, labels: np.ndarray, sizes: tuple[int, ...]) -> float:
    distances = distance.squareform(distance.pdist(x))
    total = 0.0
    for first in range(len(sizes) - 1):
        left = np.flatnonzero(labels == first)
        for second in range(first + 1, len(sizes)):
            right = np.flatnonzero(labels == second)
            total += (
                sizes[first]
                * sizes[second]
                / (sizes[first] + sizes[second])
                * (
                    2.0 * np.mean(distances[np.ix_(left, right)])
                    - np.mean(distances[np.ix_(left, left)])
                    - np.mean(distances[np.ix_(right, right)])
                )
            )
    return float(total)


def test_simplex_ehy_matches_eq_1_and_4_with_hausdorff_density() -> None:
    x = np.array([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]])
    result = ehy_uniformity(x, alpha=2.0, n_neighbors=2, n_resamples=11, rng=3)
    np.testing.assert_allclose(
        result.statistic,
        _ehy_literal(x, alpha=2.0, neighbors=2),
        rtol=3e-14,
    )


def test_simplex_ehy_seeded_stream_and_component_invariance() -> None:
    x = np.array([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]])
    seed, resamples = 33, 29
    result = ehy_uniformity(
        x, alpha=0.5, n_neighbors=1, n_resamples=resamples, rng=seed
    )
    generator = np.random.default_rng(seed)
    exceedances = sum(
        _ehy_literal(
            generator.dirichlet(np.ones(3), size=4),
            alpha=0.5,
            neighbors=1,
        )
        <= result.statistic
        for _ in range(resamples)
    )
    assert result.exceedances == exceedances
    permuted = ehy_uniformity(
        x[::-1, [2, 0, 1]],
        alpha=0.5,
        n_neighbors=1,
        n_resamples=resamples,
        rng=seed,
    )
    np.testing.assert_allclose(permuted.statistic, result.statistic, rtol=2e-14)
    assert permuted.exceedances == result.exceedances


@pytest.mark.parametrize("alpha,seed", [(0.5, 53), (2.0, 6)])
def test_simplex_ehy_identical_seeded_null_draw_is_a_tie(
    alpha: float,
    seed: int,
) -> None:
    x = np.random.default_rng(seed).dirichlet(np.ones(4), size=17)
    result = ehy_uniformity(
        x,
        alpha=alpha,
        n_neighbors=3,
        n_resamples=1,
        rng=seed,
    )
    assert result.exceedances == 1
    assert result.pvalue == 1.0


@pytest.mark.parametrize("alpha", [0.0, 0.4, 1.0, -0.5])
def test_alpha_energy_statistic_matches_literal_transform(alpha: float) -> None:
    first = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    second = np.array([[0.6, 0.2, 0.2], [0.2, 0.7, 0.1]])
    pooled = _alpha_coordinates(np.vstack((first, second)), alpha)
    labels = np.array([0, 0, 1, 1])
    expected = _energy(pooled, labels, (2, 2))
    result = alpha_energy_ksamp(
        first,
        second,
        alpha=alpha,
        calibration="exact",
        n_resamples=6,
    )
    np.testing.assert_allclose(result.statistic, expected, rtol=3e-14, atol=2e-15)


@pytest.mark.parametrize("duplicate", [False, True])
def test_large_alpha_energy_orbit_uses_uniform_labels_with_repeated_rows(
    duplicate: bool,
) -> None:
    generator = np.random.default_rng(0)
    first = generator.dirichlet(np.ones(3), size=10)
    second = generator.dirichlet(np.ones(3), size=10)
    if duplicate:
        first[1] = first[0]
    # Replay only the deterministic row ordering; the alpha transform and
    # energy statistic are independently calculated from SciPy distances.
    groups = simplex._canonical_simplex_groups((first, second), interior=False)
    pooled = _alpha_coordinates(np.vstack(groups), 0.5)
    labels = np.repeat(np.arange(2), 10)
    observed = _energy(pooled, labels, (10, 10))
    generator = np.random.default_rng(17)
    exceedances = sum(
        _energy(pooled, generator.permutation(labels), (10, 10))
        >= observed - abs(observed) * 1e-13
        for _ in range(99)
    )
    result = alpha_energy_ksamp(
        first, second, alpha=0.5, calibration="monte-carlo", n_resamples=99, rng=17
    )
    assert math.comb(20, 10) == 184_756
    assert result.statistic == pytest.approx(observed, rel=3e-14)
    assert result.exceedances == exceedances
    assert result.pvalue == (exceedances + 1) / 100
    assert not result.exact
    assert result.n_resamples == 99


def test_alpha_energy_small_monte_carlo_budget_does_not_enumerate_orbit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prohibit_enumeration(sizes: tuple[int, ...]) -> None:
        raise AssertionError("a small Monte Carlo request must not enumerate its orbit")

    monkeypatch.setattr(simplex, "_simplex_label_allocations", prohibit_enumeration)
    first = np.asarray(list(itertools.permutations([0.1, 0.2, 0.7])))
    second = np.asarray(list(itertools.permutations([0.05, 0.35, 0.6])))
    result = alpha_energy_ksamp(
        first, second, alpha=0.4, calibration="monte-carlo", n_resamples=1, rng=1
    )
    assert math.comb(12, 6) == 924
    assert result.n_resamples == 1
    assert result.pvalue == (result.exceedances + 1) / 2


def test_alpha_energy_exact_tail_enumerates_all_allocations() -> None:
    first = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    second = np.array([[0.6, 0.2, 0.2], [0.2, 0.7, 0.1]])
    pooled = _alpha_coordinates(np.vstack((first, second)), 0.5)
    observed = _energy(pooled, np.array([0, 0, 1, 1]), (2, 2))
    statistics = []
    for chosen in itertools.combinations(range(4), 2):
        labels = np.ones(4, dtype=int)
        labels[list(chosen)] = 0
        statistics.append(_energy(pooled, labels, (2, 2)))
    exceedances = sum(value >= observed - abs(observed) * 1e-13 for value in statistics)
    result = alpha_energy_ksamp(
        first, second, alpha=0.5, calibration="exact", n_resamples=6
    )
    assert result.exact
    assert result.exceedances == exceedances
    assert result.pvalue == exceedances / 6.0


def test_alpha_energy_seed_is_row_group_and_component_permutation_invariant() -> None:
    # The first two rows are component permutations, exercising the
    # distance-signature tie-break in canonicalization.
    first = np.array(
        [[0.1, 0.2, 0.7], [0.7, 0.1, 0.2], [0.25, 0.5, 0.25], [0.4, 0.35, 0.25]]
    )
    second = np.array(
        [[0.55, 0.3, 0.15], [0.2, 0.65, 0.15], [0.15, 0.25, 0.6], [0.3, 0.2, 0.5]]
    )
    baseline = alpha_energy_ksamp(
        first,
        second,
        alpha=0.4,
        calibration="monte-carlo",
        n_resamples=199,
        rng=20260907,
    )
    variant = alpha_energy_ksamp(
        second[::-1, [2, 0, 1]],
        first[[1, 3, 0, 2]][:, [2, 0, 1]],
        alpha=0.4,
        calibration="monte-carlo",
        n_resamples=199,
        rng=20260907,
    )
    np.testing.assert_allclose(variant.statistic, baseline.statistic, rtol=2e-14)
    assert variant.exceedances == baseline.exceedances
    assert variant.pvalue == baseline.pvalue


def test_alpha_energy_adversarial_last_bit_invariance_regression() -> None:
    """A component permutation must not move a borderline null statistic."""
    pooled = np.array(
        [
            [
                0.17409854351555692,
                0.4138574823727938,
                0.20456633629959325,
                0.2074776378120561,
            ],
            [
                0.08680282772708188,
                0.7050958752801263,
                0.13774068343883117,
                0.07036061355396064,
            ],
            [
                0.12882570326663279,
                0.41554661022204986,
                0.01592513943744833,
                0.4397025470738692,
            ],
            [
                0.07429610502937296,
                0.1405006165685218,
                0.19091992190986162,
                0.5942833564922437,
            ],
            [
                0.08997207113869458,
                0.08207174296104695,
                0.8175705625232856,
                0.01038562337697276,
            ],
            [
                0.0059812234645219,
                0.36607962752872353,
                0.5506477317795837,
                0.07729141722717085,
            ],
            [
                0.17879379461521394,
                0.06760727342928369,
                0.7085614909732133,
                0.0450374409822891,
            ],
            [
                0.06811212782483823,
                0.20894144143417262,
                0.5038362378317025,
                0.21911019290928663,
            ],
            [
                0.02744010179283256,
                0.00817038447617036,
                0.45146634106716665,
                0.5129231726638304,
            ],
            [
                0.02440512465644104,
                0.36860053016706235,
                0.24020145183363106,
                0.36679289334286547,
            ],
            [
                0.03686030466965504,
                0.18998636574182845,
                0.00592523939627048,
                0.767228090192246,
            ],
            [
                0.10039030867620878,
                0.46470667695918383,
                0.05120591526839961,
                0.3836970990962077,
            ],
            [
                0.16674687873751282,
                0.21988719697539075,
                0.23979591111270063,
                0.373570013174396,
            ],
            [
                0.13419390088388697,
                0.3528752181255913,
                0.00538080463810102,
                0.5075500763524208,
            ],
            [
                0.08701113836708986,
                0.2207302368650647,
                0.08616140245497941,
                0.606097222312866,
            ],
        ]
    )
    first, second, third = np.split(pooled, (4, 9))
    baseline = alpha_energy_ksamp(
        first,
        second,
        third,
        alpha=0.4,
        calibration="monte-carlo",
        n_resamples=83,
        rng=88,
    )
    columns = [1, 0, 3, 2]
    variant = alpha_energy_ksamp(
        second[[4, 3, 0, 1, 2]][:, columns],
        third[[5, 4, 1, 2, 0, 3]][:, columns],
        first[::-1, columns],
        alpha=0.4,
        calibration="monte-carlo",
        n_resamples=83,
        rng=88,
    )
    assert variant.statistic == baseline.statistic
    assert variant.exceedances == baseline.exceedances == 1
    assert variant.pvalue == baseline.pvalue


def test_alpha_energy_clamps_certified_negative_roundoff() -> None:
    # This named construction makes two empirical distributions differ only
    # near float64 resolution.  Direct subtraction of their three nonnegative
    # distance means used to report a negative energy statistic.
    generator = np.random.default_rng(121)
    for _ in range(28):
        sample_size = int(generator.integers(2, 15))
        components = int(generator.integers(2, 8))
        alpha = float(generator.uniform(-1.0, 1.0))
        first = generator.dirichlet(np.ones(components), size=sample_size)
        epsilon = 10.0 ** float(generator.uniform(-18.0, -8.0))
        second = np.maximum(
            first * (1.0 + epsilon * generator.normal(size=first.shape)),
            np.nextafter(0.0, 1.0),
        )
        second /= np.sum(second, axis=1, keepdims=True)

    result = alpha_energy_ksamp(
        first,
        second,
        alpha=alpha,
        calibration="monte-carlo",
        n_resamples=1,
        rng=0,
    )
    assert result.statistic == 0.0


def test_alpha_transform_is_continuous_at_zero() -> None:
    values = np.array([[0.1, 0.2, 0.7], [1.0e-200, 0.25, 0.75 - 1.0e-200]])
    limit = _alpha_transform(values, alpha=0.0)
    for alpha in (1.0e-16, 1.0e-50, -1.0e-300):
        actual = _alpha_transform(values, alpha=alpha)
        # The finite-alpha first correction is proportional to alpha times
        # squared log ratios, so it remains a few e-12 for the deliberately
        # extreme 1e-200 component even when alpha=1e-16.
        np.testing.assert_allclose(actual, limit, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(np.sum(actual, axis=1), 0.0, atol=4e-14)


def test_alpha_transform_retains_small_finite_alpha_correction() -> None:
    values = np.array([[0.1, 0.2, 0.7], [0.7, 0.25, 0.05], [0.33, 0.33, 0.34]])
    alpha = 1.0e-8
    logs = np.log(values)
    centered = logs - np.mean(logs, axis=1, keepdims=True)
    squared = centered * centered
    expected_first_order = centered + 0.5 * alpha * (
        squared - np.mean(squared, axis=1, keepdims=True)
    )
    actual = _alpha_transform(values, alpha=alpha)

    assert not np.array_equal(actual, centered)
    np.testing.assert_allclose(actual, expected_first_order, rtol=0.0, atol=2e-16)


def test_alpha_transform_handles_extreme_negative_power_without_overflow() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    values = np.array([[tiny, 0.5, 0.5 - tiny]])
    actual = _alpha_transform(values, alpha=-1.0)
    np.testing.assert_allclose(actual, [[-2.0, 1.0, 1.0]], atol=2e-14)
    assert np.all(np.isfinite(actual))


def test_symmetric_orbit_has_invariant_seeded_monte_carlo_plan() -> None:
    first = np.asarray(list(itertools.permutations([0.1, 0.2, 0.7])))
    second = np.asarray(list(itertools.permutations([0.05, 0.35, 0.6])))
    baseline = alpha_energy_ksamp(
        first,
        second,
        alpha=0.4,
        calibration="monte-carlo",
        n_resamples=99,
        rng=1,
    )
    variant = alpha_energy_ksamp(
        second[::-1, [2, 0, 1]],
        first[[3, 0, 5, 1, 4, 2]][:, [2, 0, 1]],
        alpha=0.4,
        calibration="monte-carlo",
        n_resamples=99,
        rng=1,
    )
    assert baseline.exceedances == variant.exceedances
    assert baseline.pvalue == variant.pvalue
    np.testing.assert_allclose(baseline.statistic, variant.statistic, rtol=0, atol=0)


def test_alpha_domain_and_structural_zero_policy() -> None:
    positive = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    with_zero = np.array([[0.0, 0.3, 0.7], [0.2, 0.5, 0.3]])
    supported = alpha_energy_ksamp(
        with_zero, positive, alpha=0.2, calibration="exact", n_resamples=6
    )
    assert math.isfinite(supported.statistic)
    for alpha in (0.0, -0.2):
        with pytest.raises(ValueError, match="strictly inside"):
            alpha_energy_ksamp(with_zero, positive, alpha=alpha, n_resamples=6)
    for alpha in (-1.01, 1.01):
        with pytest.raises(ValueError, match="between -1 and 1"):
            alpha_energy_ksamp(positive, positive, alpha=alpha, n_resamples=6)
    with pytest.raises(ValueError, match="finite"):
        alpha_energy_ksamp(positive, positive, alpha=math.inf, n_resamples=6)
    with pytest.raises(ValueError, match="requires 6"):
        alpha_energy_ksamp(
            positive,
            positive,
            alpha=0.5,
            calibration="exact",
            n_resamples=5,
        )


def test_simplex_ehy_domains_and_named_power() -> None:
    x = np.array([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]])
    for alpha in (0.0, 1.0, math.inf):
        with pytest.raises(ValueError, match="alpha"):
            ehy_uniformity(x, alpha=alpha, n_neighbors=1, n_resamples=3)
    with pytest.raises(ValueError, match="smaller than"):
        ehy_uniformity(x, alpha=2.0, n_neighbors=4, n_resamples=3)

    rng = np.random.default_rng(20260908)
    concentrated = rng.dirichlet(np.full(3, 50.0), size=45)
    result = ehy_uniformity(
        concentrated,
        alpha=0.5,
        n_neighbors=1,
        n_resamples=499,
        rng=20260909,
    )
    assert result.pvalue < 0.05


def test_simplex_ehy_9999_draw_default_completes() -> None:
    x = np.array([[0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]])
    started = time.perf_counter()
    result = ehy_uniformity(x, alpha=2.0, n_neighbors=1, rng=20260910)
    assert result.n_resamples == 9_999
    assert time.perf_counter() - started < 30.0


def test_alpha_energy_default_budget_reuses_small_orbit() -> None:
    rng = np.random.default_rng(20260911)
    first = rng.dirichlet(np.ones(3), size=5)
    second = rng.dirichlet(np.ones(3), size=5)
    started = time.perf_counter()
    result = alpha_energy_ksamp(
        first,
        second,
        alpha=0.4,
        calibration="monte-carlo",
        rng=20260912,
    )
    assert result.n_resamples == 9_999
    assert time.perf_counter() - started < 10.0

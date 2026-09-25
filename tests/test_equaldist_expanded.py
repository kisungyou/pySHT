"""Independent checks for expanded distribution-equality procedures."""

from __future__ import annotations

import math
import time
from itertools import combinations, product

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import pysht._distance_kernel as distance_kernel
from pysht._distance_kernel import calibrate_groups
from pysht.equaldist import (
    _ball_divergence_2samp,
    energy_ksamp,
    mmd_2samp,
)

_TIE_RTOL = 100.0 * np.finfo(np.float64).eps


def _matrix(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result[:, None] if result.ndim == 1 else result


def _energy_oracle(samples: tuple[np.ndarray, ...], exponent: float = 1.0) -> float:
    groups = tuple(_matrix(sample) for sample in samples)
    sizes = tuple(group.shape[0] for group in groups)
    total = sum(sizes)
    within_means = tuple(
        float(np.mean(cdist(group, group) ** exponent)) for group in groups
    )
    within = sum(n * value / 2.0 for n, value in zip(sizes, within_means))
    between = 0.0
    for j in range(len(groups) - 1):
        for k in range(j + 1, len(groups)):
            cross = float(np.mean(cdist(groups[j], groups[k]) ** exponent))
            energy = 2.0 * cross - within_means[j] - within_means[k]
            between += sizes[j] * sizes[k] * energy / (2.0 * total)
    if within == 0.0:
        return 0.0 if between == 0.0 else math.inf
    return (between / (len(groups) - 1.0)) / (within / (total - len(groups)))


def _rbf_gram(values: np.ndarray, bandwidth: float) -> np.ndarray:
    distances = cdist(values, values)
    return np.exp(-(distances**2) / (2.0 * bandwidth**2))


def _mmd_oracle(gram: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    kxx = gram[np.ix_(first, first)]
    kyy = gram[np.ix_(second, second)]
    kxy = gram[np.ix_(first, second)]
    n = first.size
    m = second.size
    return float(
        (np.sum(kxx) - np.trace(kxx)) / (n * (n - 1))
        + (np.sum(kyy) - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * np.mean(kxy)
    )


def _ball_oracle(values: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    distances = cdist(values, values)
    total = 0.0
    for centers, endpoints in ((first, first), (second, second)):
        component = 0.0
        for center in centers:
            for endpoint in endpoints:
                radius = distances[center, endpoint]
                first_mass = np.mean(distances[center, first] <= radius)
                second_mass = np.mean(distances[center, second] <= radius)
                component += (first_mass - second_mass) ** 2
        total += component / (centers.size * endpoints.size)
    return float(total)


def _two_group_exact(
    values: np.ndarray,
    first_size: int,
    statistic: object,
) -> tuple[float, int, int]:
    assert callable(statistic)
    observed_first = np.arange(first_size, dtype=np.intp)
    observed_second = np.arange(first_size, values.shape[0], dtype=np.intp)
    observed = float(statistic(values, observed_first, observed_second))
    threshold = observed - _TIE_RTOL * abs(observed)
    exceedances = 0
    total = math.comb(values.shape[0], first_size)
    for chosen_tuple in combinations(range(values.shape[0]), first_size):
        chosen = np.fromiter(chosen_tuple, dtype=np.intp, count=first_size)
        selected = np.zeros(values.shape[0], dtype=bool)
        selected[chosen] = True
        candidate = float(
            statistic(values, np.flatnonzero(selected), np.flatnonzero(~selected))
        )
        exceedances += int(candidate >= threshold)
    return observed, exceedances, total


def test_symmetric_hypercube_uses_bounded_preprocessing() -> None:
    # Sixty-four entirely ordinary Euclidean points previously triggered an
    # exponential graph search even with a single requested permutation.
    vertices = np.asarray(list(product((0.0, 1.0), repeat=6)))
    first, second = np.split(vertices, 2)
    result = energy_ksamp(first, second, n_resamples=1, rng=0)
    assert result.statistic == pytest.approx(_energy_oracle((first, second)), rel=3e-14)
    assert result.n_resamples == 1
    assert result.pvalue == (result.exceedances + 1) / 2


def test_canonicalization_fallback_preserves_complete_label_orbit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force the bounded fallback and check its complete tail independently;
    # valid inference must never require a geometric canonical ordering.
    monkeypatch.setattr(distance_kernel, "_CANONICAL_WORK_BUDGET", 0)
    first = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    second = np.array([[1.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    pooled = np.vstack((first, second))
    observed, exceedances, total = _two_group_exact(
        pooled,
        3,
        lambda values, left, right: _energy_oracle((values[left], values[right])),
    )
    result = energy_ksamp(first, second, calibration="exact", n_resamples=20)
    assert result.statistic == pytest.approx(observed, rel=3e-14)
    assert result.exceedances == exceedances
    assert result.pvalue == exceedances / total


def test_energy_matches_literal_two_and_three_group_formulas() -> None:
    x = np.array([0.0, 1.0, 3.0])
    y = np.array([2.0, 4.0])
    z = np.array([7.0, 8.0, 10.0])
    for samples in ((x, y), (x, y, z)):
        total = math.factorial(sum(len(sample) for sample in samples))
        for sample in samples:
            total //= math.factorial(len(sample))
        result = energy_ksamp(
            *samples, exponent=1.3, calibration="exact", n_resamples=total
        )
        assert result.statistic == pytest.approx(
            _energy_oracle(tuple(samples), 1.3), rel=2e-14
        )
        assert result.exact
        assert result.n_resamples == total


def test_energy_constant_and_separated_boundaries_are_defined() -> None:
    null = energy_ksamp(np.zeros(2), np.zeros(3), calibration="exact", n_resamples=10)
    separated = energy_ksamp(
        np.zeros(2), np.ones(2), calibration="exact", n_resamples=6
    )
    assert null.statistic == 0.0
    assert null.pvalue == 1.0
    assert separated.statistic == math.inf
    assert separated.pvalue == pytest.approx(1.0 / 3.0)


def test_mmd_matches_unbiased_formula_and_exhaustive_oracle() -> None:
    x = np.array([[-1.0, 0.0], [0.5, 1.0], [2.0, -0.5]])
    y = np.array([[1.0, 2.0], [3.0, 0.25]])
    pooled = np.vstack((x, y))
    bandwidth = 1.7
    gram = _rbf_gram(pooled, bandwidth)

    def statistic(_: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        return _mmd_oracle(gram, first, second)

    observed, exceedances, total = _two_group_exact(pooled, len(x), statistic)
    result = mmd_2samp(
        x,
        y,
        bandwidth=bandwidth,
        calibration="exact",
        n_resamples=total,
    )
    assert result.statistic == pytest.approx(observed, rel=5e-14, abs=5e-15)
    assert result.exceedances == exceedances
    assert result.pvalue == exceedances / total


def test_mmd_median_bandwidth_is_fixed_and_scale_equivariant() -> None:
    x = np.array([[-2.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    y = np.array([[2.0, 0.5], [3.0, -2.0], [4.0, 1.5]])
    baseline = mmd_2samp(x, y, calibration="exact", n_resamples=20)
    scaled = mmd_2samp(1e200 * x, 1e200 * y, calibration="exact", n_resamples=20)
    assert scaled.statistic == pytest.approx(baseline.statistic, rel=2e-14)
    assert scaled.exceedances == baseline.exceedances
    assert scaled.pvalue == baseline.pvalue


def test_mmd_constant_sample_has_defined_explicit_bandwidth_boundary() -> None:
    x = np.zeros((2, 2))
    y = np.zeros((3, 2))
    result = mmd_2samp(
        x,
        y,
        bandwidth=1.0,
        calibration="exact",
        n_resamples=10,
    )
    assert result.statistic == pytest.approx(0.0, abs=1e-15)
    assert result.pvalue == 1.0
    assert result.exceedances == 10
    with pytest.raises(ValueError, match="median bandwidth"):
        mmd_2samp(x, y, calibration="exact", n_resamples=10)


def test_ball_optimized_result_matches_literal_formula_and_exact_orbit() -> None:
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    y = np.array([[1.0, 1.0], [2.0, 0.0], [2.0, 1.0]])
    pooled = np.vstack((x, y))

    def statistic(values: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        return _ball_oracle(values, first, second)

    observed, exceedances, total = _two_group_exact(pooled, len(x), statistic)
    result = _ball_divergence_2samp(x, y, calibration="exact", n_resamples=total)
    assert result.statistic == pytest.approx(observed, abs=1e-15)
    assert result.exceedances == exceedances
    assert result.pvalue == exceedances / total


def test_ball_closed_balls_include_all_zero_distance_ties() -> None:
    x = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, 2.0, 2.0])
    pooled = np.concatenate((x, y))[:, None]
    expected = _ball_oracle(pooled, np.arange(3), np.arange(3, 6))
    result = _ball_divergence_2samp(x, y, calibration="exact", n_resamples=20)
    assert result.statistic == pytest.approx(expected, abs=1e-15)


def test_ball_does_not_merge_near_but_unequal_distances() -> None:
    delta = 32.0 * np.finfo(np.float64).eps
    x = np.array([0.0, 1.0, 3.0])
    y = np.array([1.0 + delta, 2.0, 4.0])
    pooled = np.concatenate((x, y))[:, None]
    expected = _ball_oracle(pooled, np.arange(3), np.arange(3, 6))
    distances = cdist(pooled, pooled)

    def tolerant_oracle() -> float:
        total = 0.0
        first = np.arange(3)
        second = np.arange(3, 6)
        for centers, endpoints in ((first, first), (second, second)):
            component = 0.0
            for center in centers:
                for endpoint in endpoints:
                    radius = distances[center, endpoint]
                    threshold = radius + _TIE_RTOL * radius
                    component += (
                        np.mean(distances[center, first] <= threshold)
                        - np.mean(distances[center, second] <= threshold)
                    ) ** 2
            total += component / (centers.size * endpoints.size)
        return float(total)

    assert tolerant_oracle() != expected
    result = _ball_divergence_2samp(x, y, calibration="exact", n_resamples=20)
    assert result.statistic == pytest.approx(expected, abs=1e-15)


def test_ball_private_prototype_records_uncertified_radius_equality_blocker() -> None:
    """A tolerance cannot safely distinguish all equal and unequal radii."""
    x = np.array([6.66133815e-16, 3.0, -6.66133815e-16])
    y = np.array([-1.0, 2.0, 2.0])
    pooled = np.concatenate((x, y))[:, None]
    literal, literal_exceedances, orbit_size = _two_group_exact(pooled, 3, _ball_oracle)
    prototype = _ball_divergence_2samp(
        x, y, calibration="exact", n_resamples=orbit_size
    )

    assert literal == pytest.approx(0.38271604938271603, abs=1e-15)
    assert literal_exceedances == 6
    assert prototype.statistic == pytest.approx(0.4691358024691358, abs=1e-15)
    assert prototype.exceedances == 4
    assert prototype.pvalue == 0.2
    assert prototype.statistic != pytest.approx(literal, abs=1e-15)


def test_regular_polygon_closed_ball_ties_are_rotation_invariant() -> None:
    angles = np.arange(5) * (2.0 * np.pi / 5.0)
    x = np.column_stack((np.cos(angles), np.sin(angles)))
    y = 1.7 * np.column_stack((np.cos(angles + 0.2), np.sin(angles + 0.2)))

    def rotate(values: np.ndarray, angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        return values @ np.array([[cosine, -sine], [sine, cosine]])

    baseline = _ball_divergence_2samp(x, y, calibration="exact", n_resamples=252)
    transformed = _ball_divergence_2samp(
        rotate(x, 1.5759765498682703),
        rotate(y, 1.5759765498682703),
        calibration="exact",
        n_resamples=252,
    )
    assert baseline.statistic == pytest.approx(0.144, abs=1e-15)
    assert transformed.statistic == baseline.statistic
    assert transformed.exceedances == baseline.exceedances
    assert transformed.pvalue == baseline.pvalue


@pytest.mark.parametrize("function", [energy_ksamp, mmd_2samp])
def test_perturbed_symmetric_geometry_has_rotation_stable_seeded_plan(
    function: object,
) -> None:
    assert callable(function)
    angles = np.arange(5) * (2.0 * np.pi / 5.0)
    x = np.column_stack((np.cos(angles), np.sin(angles)))
    y = 1.7 * np.column_stack((np.cos(angles + 0.2), np.sin(angles + 0.2)))
    x[0, 0] += 1e-14
    y[2, 1] -= 1e-14
    cosine, sine = np.cos(1.5759765498682703), np.sin(1.5759765498682703)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    baseline = function(
        x,
        y,
        calibration="monte-carlo",
        n_resamples=99,
        rng=0,
    )
    transformed = function(
        x @ rotation,
        y @ rotation,
        calibration="monte-carlo",
        n_resamples=99,
        rng=0,
    )
    assert transformed.statistic == pytest.approx(
        baseline.statistic, rel=5e-13, abs=5e-15
    )
    assert transformed.exceedances == baseline.exceedances
    assert transformed.pvalue == baseline.pvalue


@pytest.mark.parametrize("function", [energy_ksamp, mmd_2samp])
def test_approximate_rank_boundary_keeps_exact_inference_invariant(
    function: object,
) -> None:
    """Rounded rotations may change an MC plan, never the exact orbit."""
    assert callable(function)
    x = np.array(
        [
            [0.7071067811865391, 0.70710678118655],
            [-1.0000000000000004, -1.1400532426543547e-15],
            [0.9999999999999973, -5.697851580338857e-15],
            [4.0897053526598126e-15, 1.000000000000004],
        ]
    )
    y = np.array(
        [
            [9.796369932685894e-16, -1.0000000000000058],
            [0.7071067811865469, -0.7071067811865429],
            [-0.7071067811865507, 0.7071067811865508],
            [-0.7071067811865451, -0.7071067811865472],
        ]
    )
    angle = -0.28138809174674684
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    exact = function(x, y, calibration="exact", n_resamples=70)
    transformed_exact = function(
        y @ rotation,
        x @ rotation,
        calibration="exact",
        n_resamples=70,
    )
    assert transformed_exact.exceedances == exact.exceedances
    assert transformed_exact.pvalue == exact.pvalue

    # Both forced-MC plans retain the corrected-p contract.  No equality is
    # required here: the data deliberately sit on the approximate distance-
    # rank boundary used only to canonicalize a fixed-seed plan.
    for first, second in ((x, y), (y @ rotation, x @ rotation)):
        monte_carlo = function(
            first,
            second,
            calibration="monte-carlo",
            n_resamples=41,
            rng=15487,
        )
        assert monte_carlo.pvalue == (monte_carlo.exceedances + 1) / 42


@pytest.mark.parametrize("function", [energy_ksamp, mmd_2samp, _ball_divergence_2samp])
def test_symmetric_group_partition_has_canonical_seeded_plan(
    function: object,
) -> None:
    assert callable(function)
    angles = np.arange(8) * (2.0 * np.pi / 8.0)
    vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    x, y = vertices[:4], vertices[4:]
    cosine, sine = np.cos(0.732), np.sin(0.732)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    baseline = function(
        x,
        y,
        calibration="monte-carlo",
        n_resamples=37,
        rng=0,
    )
    transformed = function(
        (y @ rotation)[[1, 3, 0, 2]],
        (x @ rotation)[[2, 0, 3, 1]],
        calibration="monte-carlo",
        n_resamples=37,
        rng=0,
    )
    assert transformed.statistic == pytest.approx(
        baseline.statistic, rel=5e-13, abs=5e-15
    )
    assert transformed.exceedances == baseline.exceedances
    assert transformed.pvalue == baseline.pvalue


@pytest.mark.parametrize(
    "function",
    [energy_ksamp, mmd_2samp, _ball_divergence_2samp],
)
def test_label_methods_have_seeded_row_and_group_order_replay(function: object) -> None:
    assert callable(function)
    x = np.array([[-1.0, 2.0], [0.5, 0.0], [2.0, -1.0], [3.0, 1.0]])
    y = np.array([[4.0, 2.0], [1.0, 3.0], [-2.0, 1.0], [2.5, -3.0]])
    baseline = function(x, y, calibration="monte-carlo", n_resamples=39, rng=2026)
    reordered = function(
        y[[2, 0, 3, 1]],
        x[[3, 1, 0, 2]],
        calibration="monte-carlo",
        n_resamples=39,
        rng=2026,
    )
    assert reordered.statistic == pytest.approx(baseline.statistic, rel=2e-14)
    assert reordered.exceedances == baseline.exceedances
    assert reordered.pvalue == baseline.pvalue


@pytest.mark.parametrize(
    "function",
    [energy_ksamp, mmd_2samp, _ball_divergence_2samp],
)
def test_exact_inference_is_euclidean_invariant(function: object) -> None:
    assert callable(function)
    x = np.array([[-2.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    y = np.array([[2.0, 0.5], [3.0, -2.0], [4.0, 1.5]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    baseline = function(x, y, calibration="exact", n_resamples=20)
    offset = np.array([1e14, -1e14]) if function is not _ball_divergence_2samp else 0.0
    transformed = function(
        x @ rotation + offset,
        y @ rotation + offset,
        calibration="exact",
        n_resamples=20,
    )
    assert transformed.statistic == pytest.approx(
        baseline.statistic, rel=0.02, abs=5e-15
    )
    assert transformed.exceedances == baseline.exceedances
    assert transformed.pvalue == baseline.pvalue


@pytest.mark.parametrize(
    "function",
    [energy_ksamp, mmd_2samp, _ball_divergence_2samp],
)
def test_seeded_plan_uses_distance_signatures_under_isometries(
    function: object,
) -> None:
    assert callable(function)
    x = np.array([[0.0, 0.0], [2.0, 1.0], [5.0, 3.0], [9.0, 7.0]])
    y = np.array([[1.0, 4.0], [3.0, 0.0], [8.0, 2.0], [13.0, 9.0]])
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    baseline = function(x, y, calibration="monte-carlo", n_resamples=79, rng=739)
    transformed_samples = (
        (x @ rotation, y @ rotation),
        (x[:, ::-1], y[:, ::-1]),
        (x + 1e14, y + 1e14),
    )
    for transformed_x, transformed_y in transformed_samples:
        transformed = function(
            transformed_x,
            transformed_y,
            calibration="monte-carlo",
            n_resamples=79,
            rng=739,
        )
        assert transformed.statistic == pytest.approx(baseline.statistic, rel=5e-14)
        assert transformed.exceedances == baseline.exceedances
        assert transformed.pvalue == baseline.pvalue


def test_extreme_ranges_and_subnormal_scales_preserve_exact_inference() -> None:
    baseline_x = np.array([-1.0, -0.5])
    baseline_y = np.array([0.5, 1.0])
    huge_x = np.array([-1e308, -5e307])
    huge_y = np.array([5e307, 1e308])
    tiny = float(np.nextafter(0.0, 1.0))
    tiny_x = np.array([0.0, tiny])
    tiny_y = np.array([2.0 * tiny, 3.0 * tiny])
    tiny_baseline_x = np.array([0.0, 1.0])
    tiny_baseline_y = np.array([2.0, 3.0])
    for function in (energy_ksamp, mmd_2samp, _ball_divergence_2samp):
        baseline = function(baseline_x, baseline_y, calibration="exact", n_resamples=6)
        huge = function(huge_x, huge_y, calibration="exact", n_resamples=6)
        tiny_result = function(tiny_x, tiny_y, calibration="exact", n_resamples=6)
        tiny_baseline = function(
            tiny_baseline_x,
            tiny_baseline_y,
            calibration="exact",
            n_resamples=6,
        )
        assert huge.pvalue == baseline.pvalue
        assert tiny_result.pvalue == tiny_baseline.pvalue


def test_resampling_controls_and_input_boundaries() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([2.0, 3.0, 5.0])
    for function in (energy_ksamp, mmd_2samp, _ball_divergence_2samp):
        with pytest.raises(ValueError, match="requires 20"):
            function(x, y, calibration="exact", n_resamples=19)
        result = function(x, y, calibration="monte-carlo", n_resamples=19, rng=7)
        assert result.pvalue == (result.exceedances + 1) / 20
        assert not result.exact
        assert result.monte_carlo_standard_error is not None
    with pytest.raises(ValueError, match="strictly between"):
        energy_ksamp(x, y, exponent=2.0, n_resamples=1)
    with pytest.raises(ValueError, match="same number of features"):
        mmd_2samp(np.ones((3, 2)), np.ones((3, 1)), n_resamples=1)
    with pytest.raises(ValueError, match="median bandwidth"):
        mmd_2samp(np.zeros(3), np.zeros(3), n_resamples=1)
    with pytest.raises(ValueError, match="kernel"):
        mmd_2samp(x, y, kernel="linear", n_resamples=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bandwidth"):
        mmd_2samp(x, y, bandwidth=0.0, n_resamples=1)


def test_functions_do_not_touch_numpy_global_rng() -> None:
    x = np.arange(5.0)
    y = np.arange(5.0) + 0.5
    np.random.seed(9182)
    expected = np.random.random(4)
    np.random.seed(9182)
    energy_ksamp(x, y, calibration="monte-carlo", n_resamples=3, rng=1)
    mmd_2samp(x, y, calibration="monte-carlo", n_resamples=3, rng=1)
    _ball_divergence_2samp(x, y, calibration="monte-carlo", n_resamples=3, rng=1)
    np.testing.assert_array_equal(np.random.random(4), expected)


def test_ball_quadratic_permutation_kernel_has_modest_runtime() -> None:
    generator = np.random.default_rng(20260812)
    x = generator.normal(size=(20, 4))
    y = generator.normal(size=(20, 4))
    started = time.perf_counter()
    _ball_divergence_2samp(x, y, calibration="monte-carlo", n_resamples=199, rng=2026)
    # This loose ceiling catches accidental restoration of the literal cubic
    # kernel without making ordinary shared-CI load a source of test failures.
    assert time.perf_counter() - started < 5.0


def test_resampling_batch_bound_shrinks_for_moderate_sample_sizes() -> None:
    observed_batch_sizes: list[int] = []

    def batched(labels: np.ndarray) -> np.ndarray:
        observed_batch_sizes.append(labels.shape[0])
        return np.zeros(labels.shape[0])

    summary = calibrate_groups(
        observed=0.0,
        statistic=lambda allocation: 0.0,
        batch_statistic=batched,
        sizes=(125, 125),
        calibration="monte-carlo",
        n_resamples=19,
        rng=7,
    )
    assert summary.n_resamples == 19
    assert max(observed_batch_sizes) <= 8

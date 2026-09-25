"""Independent formula, orbit, and invariance checks for independence tests."""

from __future__ import annotations

import math
from itertools import permutations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import pysht._distance_kernel as distance_kernel
from pysht._distance_kernel import (
    at_least_as_extreme,
    calibrate_blocks,
    calibrate_groups,
)
from pysht.independence import (
    dhsic,
    distance_covariance,
    distance_multivariance,
    hsic,
)

_TIE_RTOL = 100.0 * np.finfo(np.float64).eps


def test_shared_upper_tail_handles_extended_real_boundaries() -> None:
    assert at_least_as_extreme(-math.inf, -math.inf)
    assert at_least_as_extreme(0.0, -math.inf)
    assert at_least_as_extreme(math.inf, -math.inf)
    assert not at_least_as_extreme(0.0, math.inf)
    assert at_least_as_extreme(math.inf, math.inf)

    group = calibrate_groups(
        observed=-math.inf,
        statistic=lambda allocation: 0.0,
        batch_statistic=lambda labels: np.zeros(labels.shape[0]),
        sizes=(2, 2),
        calibration="exact",
        n_resamples=6,
        rng=1,
    )
    blocks = calibrate_blocks(
        observed=-math.inf,
        statistic=lambda indices: 0.0,
        batch_statistic=lambda plans: np.zeros(plans.shape[0]),
        sample_size=3,
        block_count=2,
        calibration="exact",
        n_resamples=6,
        rng=1,
    )
    assert (group.exceedances, group.pvalue) == (6, 1.0)
    assert (blocks.exceedances, blocks.pvalue) == (6, 1.0)


def test_shared_scalar_and_batch_tails_agree_for_infinite_candidates() -> None:
    def group_statistic(groups: tuple[np.ndarray, ...]) -> float:
        return -math.inf if 0 in groups[0] else math.inf

    group_scalar = calibrate_groups(
        observed=0.0,
        statistic=group_statistic,
        sizes=(2, 2),
        calibration="exact",
        n_resamples=6,
        rng=1,
    )
    group_batch = calibrate_groups(
        observed=0.0,
        statistic=group_statistic,
        batch_statistic=lambda labels: np.where(labels[:, 0] == 0, -math.inf, math.inf),
        sizes=(2, 2),
        calibration="exact",
        n_resamples=6,
        rng=1,
    )

    def block_statistic(indices: tuple[np.ndarray, ...]) -> float:
        return -math.inf if indices[1][0] == 0 else math.inf

    block_scalar = calibrate_blocks(
        observed=0.0,
        statistic=block_statistic,
        sample_size=3,
        block_count=2,
        calibration="exact",
        n_resamples=6,
        rng=1,
    )
    block_batch = calibrate_blocks(
        observed=0.0,
        statistic=block_statistic,
        batch_statistic=lambda plans: np.where(
            plans[:, 1, 0] == 0, -math.inf, math.inf
        ),
        sample_size=3,
        block_count=2,
        calibration="exact",
        n_resamples=6,
        rng=1,
    )

    assert group_scalar == group_batch
    assert block_scalar == block_batch
    assert group_batch.exceedances == 3
    assert block_batch.exceedances == 4


@pytest.mark.parametrize("calibrator", [calibrate_groups, calibrate_blocks])
def test_shared_calibration_rejects_nan_statistics(calibrator: object) -> None:
    assert callable(calibrator)
    common = {
        "observed": 0.0,
        "statistic": lambda indices: 0.0,
        "calibration": "monte-carlo",
        "n_resamples": 2,
        "rng": 1,
    }
    if calibrator is calibrate_groups:
        controls = {"sizes": (2, 2)}
        batch = {"batch_statistic": lambda labels: np.full(labels.shape[0], np.nan)}
    else:
        controls = {"sample_size": 3, "block_count": 2}
        batch = {"batch_statistic": lambda plans: np.full(plans.shape[0], np.nan)}
    with pytest.raises(ValueError, match="must not contain NaN"):
        calibrator(**common, **controls, **batch)
    with pytest.raises(ValueError, match="observed statistic must not be NaN"):
        calibrator(**(common | {"observed": math.nan}), **controls)


def test_independence_resampling_batch_is_memory_bounded() -> None:
    observed_batch_sizes: list[int] = []

    def batched(plans: np.ndarray) -> np.ndarray:
        observed_batch_sizes.append(plans.shape[0])
        return np.zeros(plans.shape[0])

    summary = calibrate_blocks(
        observed=0.0,
        statistic=lambda indices: 0.0,
        batch_statistic=batched,
        sample_size=250,
        block_count=3,
        calibration="monte-carlo",
        n_resamples=19,
        rng=7,
    )
    assert summary.n_resamples == 19
    assert max(observed_batch_sizes) <= 4


def _matrix(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result[:, None] if result.ndim == 1 else result


def _center(matrix: np.ndarray) -> np.ndarray:
    return (
        matrix
        - np.mean(matrix, axis=0, keepdims=True)
        - np.mean(matrix, axis=1, keepdims=True)
        + np.mean(matrix)
    )


def _raw_dcov(values_x: np.ndarray, values_y: np.ndarray) -> tuple[float, ...]:
    first = _center(cdist(_matrix(values_x), _matrix(values_x)))
    second = _center(cdist(_matrix(values_y), _matrix(values_y)))
    squared_covariance = float(np.mean(first * second))
    squared_variance_x = float(np.mean(first * first))
    squared_variance_y = float(np.mean(second * second))
    squared_correlation = squared_covariance / math.sqrt(
        squared_variance_x * squared_variance_y
    )
    return (
        values_x.shape[0] * squared_covariance,
        math.sqrt(max(0.0, squared_covariance)),
        math.sqrt(max(0.0, squared_correlation)),
        math.sqrt(squared_variance_x),
        math.sqrt(squared_variance_y),
    )


def _rbf(values: np.ndarray, bandwidth: float) -> np.ndarray:
    distances = cdist(_matrix(values), _matrix(values))
    return np.exp(-(distances**2) / (2.0 * bandwidth**2))


def _dhsic_oracle(grams: tuple[np.ndarray, ...]) -> float:
    n = grams[0].shape[0]
    first = np.ones((n, n), dtype=np.float64)
    for gram in grams:
        first *= gram
    term_one = np.sum(first) / n**2
    term_two = math.prod(float(np.sum(gram)) for gram in grams) / n ** (2 * len(grams))
    row_product = np.ones(n, dtype=np.float64)
    for gram in grams:
        row_product *= np.sum(gram, axis=1)
    term_three = 2.0 * np.sum(row_product) / n ** (len(grams) + 1)
    return float(term_one + term_two - term_three)


def _normalized_distance_matrices(
    samples: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    result = []
    for sample in samples:
        distances = cdist(_matrix(sample), _matrix(sample))
        result.append(-_center(distances) / np.mean(distances))
    return tuple(result)


def _total_multivariance_oracle(samples: tuple[np.ndarray, ...]) -> float:
    matrices = _normalized_distance_matrices(samples)
    if len(matrices) == 2:
        return float(np.mean(matrices[0] * matrices[1]))
    multiplied = np.ones_like(matrices[0])
    for matrix in matrices:
        multiplied *= 1.0 + matrix
    return float(
        (np.mean(multiplied) - 1.0) / (2 ** len(matrices) - len(matrices) - 1.0)
    )


def _exact_pairwise_count(
    observed: float,
    statistic: object,
    sample_size: int,
) -> int:
    assert callable(statistic)
    threshold = observed - _TIE_RTOL * abs(observed)
    return sum(
        float(statistic(np.asarray(order, dtype=np.intp))) >= threshold
        for order in permutations(range(sample_size))
    )


def test_distance_covariance_matches_literal_formula_and_estimates() -> None:
    x = np.array([[-1.0, 0.5], [0.0, 2.0], [1.5, -0.5], [3.0, 1.0]])
    y = np.array([[2.0], [-1.0], [0.5], [4.0]])
    expected = _raw_dcov(x, y)
    result = distance_covariance(
        x, y, calibration="exact", n_resamples=math.factorial(len(x))
    )
    assert result.statistic == pytest.approx(expected[0], rel=3e-14)
    estimates = dict(result.estimates)
    assert estimates["distance covariance"] == pytest.approx(expected[1], rel=3e-14)
    assert estimates["distance correlation"] == pytest.approx(expected[2], rel=3e-14)
    assert estimates["distance variance x"] == pytest.approx(expected[3], rel=3e-14)
    assert estimates["distance variance y"] == pytest.approx(expected[4], rel=3e-14)


def test_constant_marginal_boundaries_are_explicit() -> None:
    constant = np.zeros(4)
    varying = np.array([-2.0, 0.0, 1.0, 4.0])

    dcov = distance_covariance(constant, varying, calibration="exact", n_resamples=24)
    assert dcov.statistic == 0.0
    assert dcov.pvalue == 1.0
    assert dict(dcov.estimates) == {
        "distance covariance": 0.0,
        "distance correlation": 0.0,
        "distance variance x": 0.0,
        "distance variance y": pytest.approx(_raw_dcov(varying, varying)[1]),
    }

    explicit_hsic = hsic(
        constant,
        varying,
        bandwidth_x=1.0,
        bandwidth_y=1.0,
        calibration="exact",
        n_resamples=24,
    )
    assert explicit_hsic.statistic == pytest.approx(0.0, abs=1e-15)
    assert explicit_hsic.pvalue == 1.0
    with pytest.raises(ValueError, match="median bandwidth"):
        hsic(constant, varying, calibration="exact", n_resamples=24)
    with pytest.raises(ValueError, match="requires every marginal to vary"):
        distance_multivariance(constant, varying, calibration="exact", n_resamples=24)


@pytest.mark.parametrize("force_fallback", [False, True])
def test_distance_covariance_exact_orbit_matches_independent_enumeration(
    monkeypatch: pytest.MonkeyPatch,
    force_fallback: bool,
) -> None:
    if force_fallback:
        monkeypatch.setattr(distance_kernel, "_CANONICAL_WORK_BUDGET", 0)
    x = np.array([0.0, 1.0, 4.0, 7.0])
    y = np.array([3.0, -1.0, 2.0, 8.0])
    observed = _raw_dcov(x, y)[0]
    exceedances = _exact_pairwise_count(
        observed,
        lambda order: _raw_dcov(x, y[order])[0],
        len(x),
    )
    result = distance_covariance(
        x, y, calibration="exact", n_resamples=math.factorial(len(x))
    )
    assert result.exceedances == exceedances
    assert result.pvalue == exceedances / math.factorial(len(x))


def test_hsic_and_two_block_dhsic_match_literal_formula_and_pvalue() -> None:
    x = np.array([-2.0, 0.0, 1.0, 3.0])
    y = np.array([1.0, -1.0, 4.0, 0.5])
    bandwidth_x = 1.25
    bandwidth_y = 2.5
    gram_x = _rbf(x, bandwidth_x)
    gram_y = _rbf(y, bandwidth_y)
    expected = _dhsic_oracle((gram_x, gram_y))
    hsic_result = hsic(
        x,
        y,
        bandwidth_x=bandwidth_x,
        bandwidth_y=bandwidth_y,
        calibration="exact",
        n_resamples=24,
    )
    dhsic_result = dhsic(
        x,
        y,
        bandwidth=(bandwidth_x, bandwidth_y),
        calibration="exact",
        n_resamples=24,
    )
    assert hsic_result.statistic == pytest.approx(expected, rel=3e-14)
    assert dhsic_result.statistic == pytest.approx(len(x) * expected, rel=3e-14)
    assert dhsic_result.exceedances == hsic_result.exceedances
    assert dhsic_result.pvalue == hsic_result.pvalue


def test_three_block_dhsic_matches_definition_four_v_statistic() -> None:
    x = np.array([0.0, 1.0, 2.0, 4.0, 7.0, 8.0])
    y = np.array([2.0, -1.0, 3.0, 0.5, 4.0, 1.5])
    z = np.array([1.0, 5.0, 0.0, 2.0, -2.0, 3.0])
    bandwidths = (1.1, 1.7, 2.2)
    expected = _dhsic_oracle(
        tuple(
            _rbf(sample, bandwidth) for sample, bandwidth in zip((x, y, z), bandwidths)
        )
    )
    result = dhsic(
        x,
        y,
        z,
        bandwidth=bandwidths,
        calibration="monte-carlo",
        n_resamples=7,
        rng=4,
    )
    assert result.statistic == pytest.approx(len(x) * expected, rel=5e-14)


def test_total_distance_multivariance_matches_equation_429() -> None:
    samples = (
        np.array([0.0, 1.0, 3.0, 5.0]),
        np.array([2.0, -1.0, 4.0, 0.5]),
        np.array([1.0, 3.0, -2.0, 2.0]),
    )
    expected = _total_multivariance_oracle(samples)
    result = distance_multivariance(
        *samples, calibration="monte-carlo", n_resamples=7, rng=9
    )
    assert result.statistic == pytest.approx(4.0 * expected, rel=5e-14)


def test_two_block_multivariance_has_monotone_dcov_relation_and_same_orbit() -> None:
    x = np.array([-2.0, 0.0, 1.0, 3.0])
    y = np.array([1.0, -1.0, 4.0, 0.5])
    dcov = distance_covariance(x, y, calibration="exact", n_resamples=24)
    multivariance = distance_multivariance(x, y, calibration="exact", n_resamples=24)
    mean_x = float(np.mean(cdist(x[:, None], x[:, None])))
    mean_y = float(np.mean(cdist(y[:, None], y[:, None])))
    assert dcov.statistic == pytest.approx(
        multivariance.statistic * mean_x * mean_y, rel=3e-14
    )
    assert multivariance.exceedances == dcov.exceedances
    assert multivariance.pvalue == dcov.pvalue


def test_mutual_tests_respond_to_pairwise_independent_xor() -> None:
    x = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), 10)
    y = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), 10)
    z = np.mod(x + y, 2.0)
    dhsic_result = dhsic(x, y, z, calibration="monte-carlo", n_resamples=199, rng=21)
    multivariance_result = distance_multivariance(
        x, y, z, calibration="monte-carlo", n_resamples=199, rng=21
    )
    assert dhsic_result.pvalue <= 0.05
    assert multivariance_result.pvalue <= 0.05


def test_joint_row_reordering_has_bitwise_seeded_replay() -> None:
    x = np.array([0.0, 2.0, 1.0, 5.0, 4.0, 3.0])
    y = np.array([3.0, 0.0, 4.0, 1.0, 5.0, 2.0])
    z = np.array([2.0, 4.0, 1.0, 0.0, 3.0, 5.0])
    order = np.array([3, 0, 5, 1, 4, 2])
    calls = (
        lambda a, b, c: distance_covariance(a, b, n_resamples=29, rng=17),
        lambda a, b, c: hsic(a, b, n_resamples=29, rng=17),
        lambda a, b, c: dhsic(a, b, c, n_resamples=29, rng=17),
        lambda a, b, c: distance_multivariance(a, b, c, n_resamples=29, rng=17),
    )
    for call in calls:
        baseline = call(x, y, z)
        reordered = call(x[order], y[order], z[order])
        assert reordered.statistic == baseline.statistic
        assert reordered.exceedances == baseline.exceedances
        assert reordered.pvalue == baseline.pvalue


def test_pair_swap_and_block_reordering_preserve_seeded_inference() -> None:
    x = np.array([0.0, 2.0, 1.0, 5.0, 4.0, 3.0])
    y = np.array([3.0, 0.0, 4.0, 1.0, 5.0, 2.0])
    z = np.array([2.0, 4.0, 1.0, 0.0, 3.0, 5.0])
    pair_calls = (
        (
            distance_covariance(x, y, n_resamples=59, rng=71),
            distance_covariance(y, x, n_resamples=59, rng=71),
        ),
        (
            hsic(
                x,
                y,
                kernel_x="rbf",
                kernel_y="laplacian",
                bandwidth_x=1.2,
                bandwidth_y=2.1,
                n_resamples=59,
                rng=71,
            ),
            hsic(
                y,
                x,
                kernel_x="laplacian",
                kernel_y="rbf",
                bandwidth_x=2.1,
                bandwidth_y=1.2,
                n_resamples=59,
                rng=71,
            ),
        ),
    )
    for baseline, reordered in pair_calls:
        assert reordered.statistic == baseline.statistic
        assert reordered.exceedances == baseline.exceedances
        assert reordered.pvalue == baseline.pvalue

    dhsic_baseline = dhsic(
        x,
        y,
        z,
        kernel=("rbf", "laplacian", "rbf"),
        bandwidth=(1.2, 2.1, 0.9),
        n_resamples=59,
        rng=71,
    )
    dhsic_reordered = dhsic(
        z,
        x,
        y,
        kernel=("rbf", "rbf", "laplacian"),
        bandwidth=(0.9, 1.2, 2.1),
        n_resamples=59,
        rng=71,
    )
    multivariance_baseline = distance_multivariance(x, y, z, n_resamples=59, rng=71)
    multivariance_reordered = distance_multivariance(z, x, y, n_resamples=59, rng=71)
    for baseline, reordered in (
        (dhsic_baseline, dhsic_reordered),
        (multivariance_baseline, multivariance_reordered),
    ):
        assert reordered.statistic == baseline.statistic
        assert reordered.exceedances == baseline.exceedances
        assert reordered.pvalue == baseline.pvalue


def test_exact_statistics_and_pvalues_are_isometry_invariant() -> None:
    x = np.array([[-2.0, 0.0], [0.0, 1.0], [1.0, -1.0], [3.0, 2.0]])
    y = np.array([[1.0, -1.0], [0.5, 2.0], [4.0, 1.0], [-2.0, 3.0]])
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    calls = (
        distance_covariance,
        hsic,
        distance_multivariance,
    )
    for function in calls:
        baseline = function(x, y, calibration="exact", n_resamples=24)
        transformed = function(
            x @ rotation,
            y[:, ::-1],
            calibration="exact",
            n_resamples=24,
        )
        assert transformed.statistic == pytest.approx(baseline.statistic, rel=5e-14)
        assert transformed.exceedances == baseline.exceedances
        assert transformed.pvalue == baseline.pvalue


def test_seeded_plans_use_distance_signatures_under_isometries() -> None:
    x = np.array(
        [[0.0, 0.0], [2.0, 1.0], [5.0, 3.0], [9.0, 7.0], [14.0, 11.0], [20.0, 18.0]]
    )
    y = np.array(
        [[1.0, 4.0], [3.0, 0.0], [8.0, 2.0], [13.0, 9.0], [18.0, 5.0], [25.0, 16.0]]
    )
    z = np.array(
        [[0.0, 2.0], [4.0, 1.0], [7.0, 6.0], [12.0, 3.0], [19.0, 14.0], [27.0, 10.0]]
    )
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    calls = (
        (distance_covariance, (x, y)),
        (hsic, (x, y)),
        (dhsic, (x, y, z)),
        (distance_multivariance, (x, y, z)),
    )
    for function, samples in calls:
        baseline = function(*samples, n_resamples=79, rng=739)
        transformations = (
            tuple(sample @ rotation for sample in samples),
            tuple(sample[:, ::-1] for sample in samples),
            tuple(sample + 1e14 for sample in samples),
        )
        for transformed_samples in transformations:
            transformed = function(*transformed_samples, n_resamples=79, rng=739)
            assert transformed.statistic == pytest.approx(baseline.statistic, rel=5e-14)
            assert transformed.exceedances == baseline.exceedances
            assert transformed.pvalue == baseline.pvalue


def test_symmetric_metric_profiles_have_isometry_canonical_seeded_plans() -> None:
    angles = np.arange(6) * (2.0 * np.pi / 6.0)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    x = circle
    y = circle[[0, 2, 4, 1, 3, 5]]
    z = circle[[1, 3, 5, 0, 2, 4]]

    def rotate(values: np.ndarray, angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        matrix = np.array([[cosine, -sine], [sine, cosine]])
        return values @ matrix

    cases = (
        (distance_covariance, (x, y)),
        (hsic, (x, y)),
        (dhsic, (x, y, z)),
        (distance_multivariance, (x, y, z)),
    )
    for function, samples in cases:
        baseline = function(
            *samples,
            calibration="monte-carlo",
            n_resamples=99,
            rng=0,
        )
        transformed_samples = tuple(
            rotate(sample, 0.23 + 0.94 * index) for index, sample in enumerate(samples)
        )
        transformed = function(
            *transformed_samples,
            calibration="monte-carlo",
            n_resamples=99,
            rng=0,
        )
        assert transformed.statistic == pytest.approx(
            baseline.statistic, rel=5e-13, abs=5e-15
        )
        assert transformed.exceedances == baseline.exceedances
        assert transformed.pvalue == baseline.pvalue


@pytest.mark.parametrize(
    "function",
    [distance_covariance, hsic, dhsic, distance_multivariance],
)
def test_symmetric_exact_orbits_are_stable_to_independent_rotations(
    function: object,
) -> None:
    assert callable(function)
    angles = np.arange(6) * (2.0 * np.pi / 6.0)
    x = np.column_stack((np.cos(angles), np.sin(angles)))
    y = x[[0, 2, 4, 1, 3, 5]]
    cosine_x, sine_x = np.cos(0.23), np.sin(0.23)
    cosine_y, sine_y = np.cos(1.17), np.sin(1.17)
    rotated_x = x @ np.array([[cosine_x, -sine_x], [sine_x, cosine_x]])
    rotated_y = y @ np.array([[cosine_y, -sine_y], [sine_y, cosine_y]])
    baseline = function(x, y, calibration="exact", n_resamples=720)
    transformed = function(
        rotated_x,
        rotated_y,
        calibration="exact",
        n_resamples=720,
    )
    assert transformed.statistic == pytest.approx(
        baseline.statistic, rel=5e-13, abs=5e-15
    )
    assert transformed.exceedances == baseline.exceedances
    assert transformed.pvalue == baseline.pvalue


def test_separate_rescaling_has_declared_statistic_behavior() -> None:
    x = np.array([0.0, 1.0, 3.0, 7.0])
    y = np.array([2.0, -1.0, 4.0, 0.5])
    dcov = distance_covariance(x, y, calibration="exact", n_resamples=24)
    scaled_dcov = distance_covariance(
        3.0 * x, 5.0 * y, calibration="exact", n_resamples=24
    )
    assert scaled_dcov.statistic == pytest.approx(15.0 * dcov.statistic, rel=3e-14)
    assert scaled_dcov.pvalue == dcov.pvalue
    multivariance = distance_multivariance(x, y, calibration="exact", n_resamples=24)
    scaled_multivariance = distance_multivariance(
        3.0 * x, 5.0 * y, calibration="exact", n_resamples=24
    )
    assert scaled_multivariance.statistic == pytest.approx(
        multivariance.statistic, rel=3e-14
    )
    assert scaled_multivariance.pvalue == multivariance.pvalue


def test_extreme_scales_preserve_normalized_distance_inference() -> None:
    baseline_x = np.array([-1.0, -0.5, 0.5])
    baseline_y = np.array([0.5, -1.0, 1.0])
    huge_x = 1e308 * baseline_x
    huge_y = 1e308 * baseline_y
    baseline = distance_covariance(
        baseline_x, baseline_y, calibration="exact", n_resamples=6
    )
    huge = distance_covariance(huge_x, huge_y, calibration="exact", n_resamples=6)
    assert huge.statistic == math.inf
    assert huge.exceedances == baseline.exceedances
    assert huge.pvalue == baseline.pvalue
    huge_diagnostic = dict(huge.diagnostics)["normalized V_n^2"]
    baseline_diagnostic = dict(baseline.diagnostics)["normalized V_n^2"]
    assert huge_diagnostic == pytest.approx(baseline_diagnostic, rel=3e-14)


def test_kernel_controls_support_per_marginal_values_and_reject_bad_shapes() -> None:
    x = np.arange(6.0)
    y = np.array([0.0, 2.0, 1.0, 5.0, 3.0, 4.0])
    z = np.array([3.0, 0.0, 4.0, 1.0, 5.0, 2.0])
    hsic_result = hsic(
        x,
        y,
        kernel_x="rbf",
        kernel_y="laplacian",
        bandwidth_x=1.0,
        bandwidth_y=2.0,
        n_resamples=3,
        rng=1,
    )
    assert dict(hsic_result.diagnostics)["kernel x"] == "rbf"
    assert dict(hsic_result.diagnostics)["kernel y"] == "laplacian"
    dhsic_result = dhsic(
        x,
        y,
        z,
        kernel=("rbf", "laplacian", "rbf"),
        bandwidth=(1.0, "median", 2.0),
        n_resamples=3,
        rng=1,
    )
    assert dhsic_result.statistic >= 0.0
    diagnostics = dict(dhsic_result.diagnostics)
    assert {diagnostics[f"kernel {index}"] for index in (1, 2, 3)} == {
        "rbf",
        "laplacian",
    }
    assert sorted(
        value
        for name, value in diagnostics.items()
        if name.startswith("bandwidth ") and isinstance(value, float)
    ) == [1.0, 2.0]
    assert sum(value == "median" for value in diagnostics.values()) == 1
    with pytest.raises(ValueError, match="exactly 3"):
        dhsic(x, y, z, kernel=("rbf", "laplacian"), n_resamples=1)
    with pytest.raises(ValueError, match="exactly 3"):
        dhsic(x, y, z, bandwidth=(1.0, 2.0), n_resamples=1)


def test_boundaries_invalid_controls_and_exact_budgets() -> None:
    x = np.arange(4.0)
    y = np.array([0.0, 2.0, 1.0, 3.0])
    for function in (distance_covariance, hsic, distance_multivariance):
        with pytest.raises(ValueError, match="requires 24"):
            function(x, y, calibration="exact", n_resamples=23)
    with pytest.raises(ValueError, match="same number of rows"):
        distance_covariance(np.arange(3.0), np.arange(4.0), n_resamples=1)
    assert (
        distance_covariance(np.zeros(4), y, calibration="exact", n_resamples=24).pvalue
        == 1.0
    )
    with pytest.raises(ValueError, match="median bandwidth"):
        hsic(np.zeros(4), y, n_resamples=1)
    with pytest.raises(ValueError, match=r"n >= 2\*d"):
        dhsic(
            np.arange(5.0),
            np.array([0.0, 2.0, 1.0, 3.0, 4.0]),
            np.arange(5.0),
            n_resamples=1,
        )
    with pytest.raises(ValueError, match="kernel"):
        hsic(x, y, kernel_x="linear", n_resamples=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bandwidth"):
        hsic(x, y, bandwidth_y=-1.0, n_resamples=1)


def test_monte_carlo_contract_and_global_rng_isolation() -> None:
    x = np.arange(6.0)
    y = np.array([0.0, 2.0, 1.0, 5.0, 3.0, 4.0])
    result = distance_covariance(x, y, calibration="monte-carlo", n_resamples=19, rng=8)
    assert result.pvalue == (result.exceedances + 1) / 20
    assert result.monte_carlo_standard_error is not None
    assert result.tail_probability_interval is not None
    np.random.seed(227)
    expected = np.random.random(4)
    np.random.seed(227)
    distance_covariance(x, y, n_resamples=3, rng=9)
    hsic(x, y, n_resamples=3, rng=9)
    dhsic(x, y, n_resamples=3, rng=9)
    distance_multivariance(x, y, n_resamples=3, rng=9)
    np.testing.assert_array_equal(np.random.random(4), expected)

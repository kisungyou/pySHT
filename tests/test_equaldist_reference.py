"""Independent reference checks for the Biswas--Ghosh two-sample test.

The oracles in this file intentionally use only the public pySHT API and
SciPy's distance routines.  They do not import implementation helpers from
``pysht.equaldist`` or reuse its pooled distance matrix.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist, pdist

from pysht.equaldist import bg_2samp

_TIE_RTOL = 100.0 * np.finfo(np.float64).eps


def _as_matrix(sample: ArrayLike) -> NDArray[np.float64]:
    """Convert a test fixture to the observation-by-feature convention."""
    values = np.asarray(sample, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return values


def _reference_statistic(x: ArrayLike, y: ArrayLike) -> float:
    """Evaluate the published distance-mean contrast literally with SciPy."""
    first = _as_matrix(x)
    second = _as_matrix(y)

    mean_within_first = float(np.mean(pdist(first, metric="euclidean")))
    mean_between = float(np.mean(cdist(first, second, metric="euclidean")))
    mean_within_second = float(np.mean(pdist(second, metric="euclidean")))

    return (mean_within_first - mean_between) ** 2 + (
        mean_between - mean_within_second
    ) ** 2


def _exhaustive_permutation_oracle(
    x: ArrayLike,
    y: ArrayLike,
) -> tuple[float, int, int, float]:
    """Enumerate every fixed-size labeling using the literal reference statistic."""
    first = _as_matrix(x)
    second = _as_matrix(y)
    pooled = np.vstack((first, second))
    first_size = first.shape[0]
    total = math.comb(pooled.shape[0], first_size)
    observed = _reference_statistic(first, second)

    # This is the documented public tie rule, stated here rather than imported
    # from the implementation under test.
    threshold = observed - abs(observed) * _TIE_RTOL
    exceedances = 0
    for chosen_tuple in combinations(range(pooled.shape[0]), first_size):
        chosen = np.fromiter(chosen_tuple, dtype=np.intp, count=first_size)
        in_first = np.zeros(pooled.shape[0], dtype=np.bool_)
        in_first[chosen] = True
        candidate = _reference_statistic(pooled[in_first], pooled[~in_first])
        exceedances += int(candidate >= threshold)

    return observed, exceedances, total, exceedances / total


def test_fixed_univariate_fixture_has_hand_calculated_values() -> None:
    x = np.array([0.0, 2.0])
    y = np.array([1.0, 5.0, 8.0])

    result = bg_2samp(
        x,
        y,
        calibration="exact",
        n_resamples=math.comb(x.size + y.size, x.size),
    )

    # Within means are 2 and 14/3, while the between-sample mean is 4.
    expected_raw = 40.0 / 9.0
    assert _reference_statistic(x, y) == pytest.approx(expected_raw, rel=1e-15)
    assert result.statistic == pytest.approx(expected_raw, rel=1e-14)
    assert result.distance_scale == pytest.approx(8.0, rel=1e-15)
    assert result.normalized_statistic == pytest.approx(5.0 / 72.0, rel=1e-14)
    assert result.exceedances == 8
    assert result.n_resamples == 10
    assert result.pvalue == pytest.approx(4.0 / 5.0)


def test_fixed_rectangle_fixture_has_hand_calculated_values_and_ties() -> None:
    x = np.array([[0.0, 0.0], [0.0, 2.0]])
    y = np.array([[3.0, 0.0], [3.0, 2.0]])

    result = bg_2samp(
        x,
        y,
        calibration="exact",
        n_resamples=6,
    )

    # Both within means are 2 and the cross mean is (3 + sqrt(13)) / 2.
    expected_raw = 7.0 - math.sqrt(13.0)
    assert _reference_statistic(x, y) == pytest.approx(expected_raw, rel=1e-15)
    assert result.statistic == pytest.approx(expected_raw, rel=1e-14)
    assert result.distance_scale == pytest.approx(math.sqrt(13.0), rel=1e-14)
    assert result.normalized_statistic == pytest.approx(
        expected_raw / 13.0,
        rel=1e-14,
    )
    # The observed labeling and its complementary labeling are equal upper-tail
    # ties; all six fixed-size labelings remain distinct randomization outcomes.
    assert result.exceedances == 2
    assert result.n_resamples == 6
    assert result.pvalue == pytest.approx(1.0 / 3.0)


def test_exact_calibration_matches_independent_exhaustive_oracle() -> None:
    x = np.array([[-2.0, 0.5], [0.25, -1.0], [1.5, 2.0]])
    y = np.array(
        [
            [-1.25, -2.0],
            [0.75, 0.1],
            [2.25, -0.75],
            [3.0, 1.25],
        ]
    )
    expected_statistic, expected_exceedances, total, expected_pvalue = (
        _exhaustive_permutation_oracle(x, y)
    )

    result = bg_2samp(
        x,
        y,
        calibration="exact",
        n_resamples=total,
    )

    assert result.statistic == pytest.approx(expected_statistic, rel=5e-14)
    assert result.exceedances == expected_exceedances
    assert result.n_resamples == total
    assert result.pvalue == pytest.approx(expected_pvalue, rel=0.0, abs=0.0)
    assert result.calibration == "exact permutation"
    assert result.monte_carlo_standard_error is None


def test_translation_rotation_and_feature_permutation_are_invariant() -> None:
    x = np.array(
        [
            [-1.0, 0.25, 2.0],
            [0.5, -2.0, 0.75],
            [2.0, 1.0, -1.5],
        ]
    )
    y = np.array(
        [
            [-2.0, 1.5, 0.25],
            [1.25, 0.5, 2.5],
            [3.0, -1.0, 0.5],
        ]
    )
    total = math.comb(x.shape[0] + y.shape[0], x.shape[0])
    baseline = bg_2samp(
        x,
        y,
        calibration="exact",
        n_resamples=total,
    )

    angle = 0.713
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shift = np.array([7.0, -11.0, 2.5])
    transformed_samples = (
        (x + shift, y + shift),
        (x @ rotation, y @ rotation),
        (x[:, [2, 0, 1]], y[:, [2, 0, 1]]),
    )

    for transformed_x, transformed_y in transformed_samples:
        expected_statistic, expected_exceedances, _, expected_pvalue = (
            _exhaustive_permutation_oracle(transformed_x, transformed_y)
        )
        transformed = bg_2samp(
            transformed_x,
            transformed_y,
            calibration="exact",
            n_resamples=total,
        )

        assert transformed.statistic == pytest.approx(
            expected_statistic,
            rel=5e-14,
            abs=5e-15,
        )
        assert transformed.statistic == pytest.approx(
            baseline.statistic,
            rel=5e-14,
            abs=5e-15,
        )
        assert transformed.normalized_statistic == pytest.approx(
            baseline.normalized_statistic,
            rel=5e-14,
            abs=5e-15,
        )
        assert transformed.exceedances == expected_exceedances
        assert transformed.exceedances == baseline.exceedances
        assert transformed.pvalue == expected_pvalue
        assert transformed.pvalue == baseline.pvalue


def test_raw_and_normalized_statistics_have_declared_scale_relation() -> None:
    x = np.array([[-1.0, 0.0], [0.0, 2.0], [1.5, -0.5]])
    y = np.array([[-2.0, 1.0], [0.5, 0.75], [3.0, 2.5]])
    total = math.comb(x.shape[0] + y.shape[0], x.shape[0])
    result = bg_2samp(
        x,
        y,
        calibration="exact",
        n_resamples=total,
    )

    pooled = np.vstack((x, y))
    maximum_distance = float(np.max(pdist(pooled, metric="euclidean")))
    reference = _reference_statistic(x, y)
    assert result.statistic == pytest.approx(reference, rel=5e-14)
    assert result.distance_scale == pytest.approx(maximum_distance, rel=5e-14)
    assert result.normalized_statistic == pytest.approx(
        reference / maximum_distance**2,
        rel=5e-14,
    )

    factor = 2.75
    scaled = bg_2samp(
        factor * x,
        factor * y,
        calibration="exact",
        n_resamples=total,
    )
    assert scaled.statistic == pytest.approx(factor**2 * result.statistic, rel=5e-14)
    assert scaled.normalized_statistic == pytest.approx(
        result.normalized_statistic,
        rel=5e-14,
    )
    assert scaled.exceedances == result.exceedances
    assert scaled.pvalue == result.pvalue


@pytest.mark.parametrize(
    ("x_offsets", "y_offsets", "expected_exceedances"),
    (
        (
            np.array([-15.0, -8.0, 1.0, -48.0]),
            np.array([29.0, 18.0, -41.0, -27.0, 0.0]),
            87,
        ),
        (
            np.array([[-15.0, 2.0], [-8.0, -7.0], [1.0, 5.0], [-48.0, 11.0]]),
            np.array(
                [
                    [29.0, -4.0],
                    [18.0, 9.0],
                    [-41.0, 8.0],
                    [-27.0, -12.0],
                    [0.0, 3.0],
                ]
            ),
            74,
        ),
    ),
)
def test_huge_common_offset_preserves_exact_orbit_geometry(
    x_offsets: NDArray[np.float64],
    y_offsets: NDArray[np.float64],
    expected_exceedances: int,
) -> None:
    """Retain ULP-scale geometry before applying numerical normalization."""
    shift = 1e300
    spacing = np.spacing(shift)
    x = shift + spacing * x_offsets
    y = shift + spacing * y_offsets

    # Every intended displacement is exactly representable at this location.
    np.testing.assert_array_equal((x - shift) / spacing, x_offsets)
    np.testing.assert_array_equal((y - shift) / spacing, y_offsets)

    total = math.comb(x.shape[0] + y.shape[0], x.shape[0])
    reference = bg_2samp(
        x_offsets,
        y_offsets,
        calibration="exact",
        n_resamples=total,
    )
    translated = bg_2samp(
        x,
        y,
        calibration="exact",
        n_resamples=total,
    )

    assert reference.exceedances == expected_exceedances
    assert translated.exceedances == expected_exceedances
    assert translated.pvalue == reference.pvalue
    assert translated.normalized_statistic == pytest.approx(
        reference.normalized_statistic,
        rel=5e-14,
        abs=0.0,
    )
    assert translated.distance_scale == pytest.approx(
        spacing * reference.distance_scale,
        rel=5e-14,
        abs=0.0,
    )
    assert math.isinf(translated.statistic)

"""Independent DISCO references for near-zero dispersion and permutation ties."""

from __future__ import annotations

from decimal import Decimal, localcontext
from itertools import combinations
from unittest.mock import patch

import numpy as np
import pytest

from pysht import equaldist
from pysht._distance_kernel import canonical_groups, distance_geometry, pooled_groups


def _literal_one_dimensional_orbit(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, int, int]:
    """Enumerate the original within/cross formula on the exact float inputs."""
    with localcontext() as context:
        context.prec = 80
        points = [Decimal.from_float(float(v)) for v in np.r_[first, second]]
        distances = [[abs(a - b) for b in points] for a in points]

        def statistic(indices: tuple[int, ...]) -> Decimal:
            other = tuple(j for j in range(6) if j not in indices)
            within_first = sum(distances[i][j] for i in indices for j in indices) / 9
            within_second = sum(distances[i][j] for i in other for j in other) / 9
            cross = sum(distances[i][j] for i in indices for j in other) / 9
            within = Decimal("1.5") * (within_first + within_second)
            between = Decimal("0.75") * (2 * cross - within_first - within_second)
            return between / (within / 4)

        observed = statistic((0, 1, 2))
        orbit = [statistic(indices) for indices in combinations(range(6), 3)]
        # Eight allocations select one point from each adjacent close pair.
        # Their squared empirical-CDF differences are identical on every
        # interval, hence they are exact tied minima. This tolerance covers
        # only last-place rounding in the 80-digit independent arithmetic.
        tolerance = Decimal("1e-70")
        exceedances = sum(value >= observed - tolerance for value in orbit)
        ties = sum(abs(value - observed) <= tolerance for value in orbit)
        return float(observed), exceedances, ties


@pytest.mark.parametrize("separation", [1e-8, 1e-12])
@pytest.mark.parametrize("batch_size", [1, 3, 20])
def test_exact_energy_includes_algebraically_tied_close_group_allocations(
    separation: float, batch_size: int
) -> None:
    first = np.array([0.0, 1.0, 3.0])
    second = first + separation
    statistic, exceedances, ties = _literal_one_dimensional_orbit(first, second)
    assert exceedances == 20
    assert ties == 8
    with patch("pysht._distance_kernel._quadratic_batch_size", return_value=batch_size):
        actual = equaldist.energy_ksamp(
            first, second, exponent=1.0, calibration="exact", n_resamples=20
        )
    assert actual.statistic == pytest.approx(
        statistic, rel=1e-7, abs=8 * np.finfo(float).eps
    )
    assert actual.exceedances == exceedances
    assert actual.pvalue == 1.0


def test_energy_observed_allocation_is_identical_across_batch_sizes() -> None:
    generator = np.random.default_rng(29)
    groups = canonical_groups(
        tuple(generator.normal(size=(30 + j, 2)) for j in range(3))
    )
    pooled, indices = pooled_groups(groups)
    distances = distance_geometry(pooled).distances ** 1.99
    labels = np.empty(pooled.shape[0], dtype=np.intp)
    for label, selected in enumerate(indices):
        labels[selected] = label
    observed = equaldist._energy_statistic(distances, indices)
    sizes = tuple(map(len, groups))
    for batch_size in (1, 4, 37):
        actual = equaldist._energy_statistic_batch(
            distances, np.tile(labels, (batch_size, 1)), sizes
        )
        np.testing.assert_array_equal(actual, np.full(batch_size, observed))


@pytest.mark.parametrize("exponent", [0.1, 1.0, 1.99])
@pytest.mark.parametrize("reordered", [False, True])
def test_identical_empirical_groups_have_zero_energy_and_unit_pvalue(
    exponent: float, reordered: bool
) -> None:
    first = np.array([0.2, -1.5, 2.1])
    second = first[::-1] if reordered else first.copy()
    actual = equaldist.energy_ksamp(
        first, second, exponent=exponent, calibration="exact", n_resamples=20
    )
    assert actual.statistic == 0.0
    assert actual.exceedances == 20
    assert actual.pvalue == 1.0

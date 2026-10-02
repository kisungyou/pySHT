"""Independent regression oracles for extreme geometry and likelihood inputs."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from itertools import combinations

import numpy as np
import pytest
from scipy import stats

from pysht import equaldist, independence, simplex, uniformity
from pysht._distance_kernel import distance_geometry, kernel_matrix
from pysht.independence import (
    _dhsic_statistic,
    _dhsic_statistic_batch,
)


def test_mmd_retains_local_spacings_beside_a_distant_outlier() -> None:
    """Enumerate direct-kernel MMD without centering or global scaling."""
    points = [0.0, 1.0, 3.0, 1e17]
    with localcontext() as context:
        context.prec = 70
        z = list(map(Decimal.from_float, points))
        gram = [[(-((a - b) ** 2) / 2).exp() for b in z] for a in z]

        def statistic(first: tuple[int, ...]) -> Decimal:
            second = [j for j in range(4) if j not in first]
            return (
                gram[first[0]][first[1]]
                + gram[second[0]][second[1]]
                - sum(gram[i][j] for i in first for j in second) / 2
            )

        expected = statistic((0, 1))
        count = sum(statistic(first) >= expected for first in combinations(range(4), 2))
    actual = equaldist.mmd_2samp(
        points[:2], points[2:], bandwidth=1.0, calibration="exact", n_resamples=6
    )
    assert actual.statistic == pytest.approx(float(expected), rel=3e-15)
    assert actual.pvalue == count / 6 == 1 / 3


@pytest.mark.parametrize("bandwidth", [1e4, 1e8, 1e10])
@pytest.mark.parametrize("kernel", ["rbf", "laplacian"])
def test_mmd_nearly_constant_kernels_preserve_small_signal(
    bandwidth: float, kernel: str
) -> None:
    # Literal original kernels at high precision; enumerate all labelings.
    with localcontext() as context:
        context.prec = 75
        z = list(map(Decimal, [0, 1, 3, 4]))
        width = Decimal.from_float(bandwidth)
        gram = [
            [
                (-(((a - b) / width) ** 2) / 2).exp()
                if kernel == "rbf"
                else (-abs(a - b) / width).exp()
                for b in z
            ]
            for a in z
        ]

        def statistic(first: tuple[int, ...]) -> Decimal:
            second = [i for i in range(4) if i not in first]
            return (
                gram[first[0]][first[1]]
                + gram[second[0]][second[1]]
                - sum(gram[i][j] for i in first for j in second) / 2
            )

        expected = statistic((0, 1))
        count = sum(statistic(first) >= expected for first in combinations(range(4), 2))
    actual = equaldist.mmd_2samp(
        [0, 1],
        [3, 4],
        bandwidth=bandwidth,
        kernel=kernel,
        calibration="exact",
        n_resamples=6,
    )
    assert actual.statistic == pytest.approx(float(expected), rel=2e-14, abs=0)
    assert actual.pvalue == count / 6 == 1 / 3


def test_mmd_retains_tiny_nonzero_off_diagonal_kernels() -> None:
    actual = equaldist.mmd_2samp(
        [0, 10], [30, 40], bandwidth=1.0, calibration="exact", n_resamples=6
    )
    expected = (
        2 * math.exp(-50) - (math.exp(-200) + 2 * math.exp(-450) + math.exp(-800)) / 2
    )
    assert actual.statistic == pytest.approx(expected, rel=1e-13, abs=0)
    assert actual.pvalue == 1 / 3


def test_overflowing_global_range_retains_finite_local_kernel_distances() -> None:
    points = np.array([[-1e308], [0.0], [1.0], [3.0], [1e308]])
    geometry = distance_geometry(points)
    gram, *_ = kernel_matrix(geometry, kernel="rbf", bandwidth=1.0)
    assert np.all(np.isfinite(geometry.distances))
    assert geometry.distances[0, -1] == pytest.approx(1.0)
    assert gram[1, 2] == pytest.approx(math.exp(-0.5), rel=2e-15)
    assert gram[2, 3] == pytest.approx(math.exp(-2.0), rel=2e-15)
    assert gram[0, -1] == 0.0


@pytest.mark.parametrize("bandwidth", [3000.0, 10000.0])
def test_hsic_small_statistic_preserves_exact_permutation_order(
    bandwidth: float,
) -> None:
    # 90-digit direct exp(-d²/(2h²)), double-centering, and all 720
    # permutations give exactly two maxima: identity and reversal.
    expected = {
        3000.0: 1.0502388762678090082007107135559407212351950240208e-13,
        10000.0: 8.5069435852431115046123218291851643616854333364386e-16,
    }[bandwidth]
    x = np.arange(6.0)
    h = independence.hsic(
        x,
        x,
        bandwidth_x=bandwidth,
        bandwidth_y=bandwidth,
        calibration="exact",
        n_resamples=720,
    )
    dh = independence.dhsic(
        x, x, bandwidth=bandwidth, calibration="exact", n_resamples=720
    )
    assert h.statistic == pytest.approx(expected, rel=1e-13, abs=0)
    assert dh.statistic == pytest.approx(6 * expected, rel=1e-13, abs=0)
    assert h.pvalue == dh.pvalue == 2 / 720


def _decimal_dhsic(blocks: tuple[np.ndarray, ...], bandwidth: float) -> float:
    """Original three-expectation V-statistic at 70 digits; no offsets."""
    with localcontext() as context:
        context.prec = 70
        n = len(blocks[0])
        width = Decimal.from_float(bandwidth)
        grams = []
        for block in blocks:
            values = [Decimal.from_float(float(v)) for v in block[:, 0]]
            grams.append(
                [
                    [(-(((a - b) / width) ** 2) / 2).exp() for b in values]
                    for a in values
                ]
            )
        first = (
            sum(
                math.prod(gram[i][j] for gram in grams)
                for i in range(n)
                for j in range(n)
            )
            / n**2
        )
        second = math.prod(sum(map(sum, gram)) / n**2 for gram in grams)
        third = (
            2 * sum(math.prod(sum(gram[i]) / n for gram in grams) for i in range(n)) / n
        )
        return float(first + second - third)


@pytest.mark.parametrize("block_count", [3, 4])
@pytest.mark.parametrize("bandwidth", [2.0, 10000.0])
def test_multiblock_dhsic_matches_decimal_original_formula(
    block_count: int, bandwidth: float
) -> None:
    n = 2 * block_count
    blocks = tuple(
        np.ascontiguousarray(((np.arange(n) * (j + 1)) % (n + 1))[:, None], dtype=float)
        for j in range(block_count)
    )
    grams = tuple(
        kernel_matrix(
            distance_geometry(block), kernel="rbf", bandwidth=bandwidth, offset=True
        )[0]
        for block in blocks
    )
    generator = np.random.default_rng(29)
    plans = np.array(
        [[generator.permutation(n) for _ in blocks] for _ in range(5)], dtype=np.intp
    )
    batched = _dhsic_statistic_batch(grams, plans)
    for plan, batch_value in zip(plans, batched, strict=True):
        expected = _decimal_dhsic(
            tuple(block[index] for block, index in zip(blocks, plan, strict=True)),
            bandwidth,
        )
        scalar = _dhsic_statistic(grams, tuple(plan))
        assert scalar == pytest.approx(expected, rel=2e-12, abs=0)
        assert batch_value == pytest.approx(expected, rel=2e-12, abs=0)
    public = independence.dhsic(*blocks, bandwidth=bandwidth, n_resamples=5, rng=1)
    assert public.statistic == pytest.approx(
        n * _decimal_dhsic(blocks, bandwidth), rel=2e-12, abs=0
    )


@pytest.mark.parametrize("model", ["symmetric", "general"])
@pytest.mark.parametrize(
    "exponent,expected",
    [
        (-16, 38.68566542488045398596897617707649476114),
        (-20, 49.77602031198420974387145184345470971183),
        (-24, 60.86637520093608715864239511863917934309),
        (-26, 66.41155264541562298862766108627940901276),
    ],
)
def test_concentrated_dirichlet_matches_bracketed_high_precision_score(
    model: str, exponent: int, expected: float
) -> None:
    # Each input is an exactly represented dyadic rational. Independent
    # 90-digit bisection of psi(2a)-psi(a)+mean(log(x))=0, followed by
    # 4[logGamma(2a)-2logGamma(a)+2(a-1)mean(log(x))], gives these values.
    delta = 2.0**exponent
    values = [[0.5 - delta, 0.5 + delta], [0.5 + delta, 0.5 - delta]]
    actual = simplex.uniformity(values, model=model)
    assert actual.statistic == pytest.approx(expected, rel=2e-14)
    assert actual.pvalue == pytest.approx(
        stats.chi2.sf(expected, 1 if model == "symmetric" else 2), rel=3e-13
    )


@pytest.mark.parametrize(
    "exponent,expected",
    [
        (-12, 76.06780279602606818440285507097194812362),
        (-24, 175.8805448417668741970705012297851144158),
    ],
)
def test_asymmetric_concentrated_dirichlet_matches_independent_score_root(
    exponent: int, expected: float
) -> None:
    # Independent 85-digit three-variable digamma score solution and direct
    # log-gamma likelihood, using the exact dyadic inputs (rows sum to one).
    d = 2.0**exponent
    x = np.array(
        [
            [0.25 + d, 0.25, 0.5 - d],
            [0.25 - d, 0.25 + 2 * d, 0.5 - d],
            [0.25, 0.25 - 2 * d, 0.5 + 2 * d],
        ]
    )
    actual = simplex.uniformity(x, model="general")
    assert actual.statistic == pytest.approx(expected, rel=3e-14)
    reordered = simplex.uniformity(x[::-1, ::-1], model="general")
    assert reordered.statistic == pytest.approx(expected, rel=3e-14)


def test_concentrated_fit_honors_iteration_limit() -> None:
    d = 2.0**-12
    x = [
        [0.25 + d, 0.25, 0.5 - d],
        [0.25 - d, 0.25 + 2 * d, 0.5 - d],
        [0.25, 0.25 - 2 * d, 0.5 + 2 * d],
    ]
    with pytest.raises(RuntimeError, match="did not converge"):
        simplex.uniformity(x, model="general", max_iter=1)


def test_yang_modarres_counts_identical_null_draw() -> None:
    observed = np.random.default_rng(0).random((5, 10))
    actual = uniformity.ym_interpoint(observed, statistic="q2", n_resamples=1, rng=0)
    assert actual.exceedances == 1
    assert actual.pvalue == 1.0


def test_ehy_does_not_square_a_representable_tiny_distance_to_zero() -> None:
    actual = uniformity.ehy(
        [[0.0], [1e-200]], alpha=0.01, n_neighbors=1, n_resamples=1, rng=0
    )
    # n=2, J=1, one-dimensional unit-ball volume=2: T=2(4r)^alpha.
    expected = 2 * math.exp(0.01 * (math.log(4) + math.log(1e-200)))
    assert actual.statistic == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize("domain", ["rectangle", "simplex"])
def test_ehy_huge_positive_power_preserves_pair_spacing_order(domain: str) -> None:
    generator = np.random.default_rng(0)
    if domain == "rectangle":
        observed = [[0.0], [0.9]]
        draws = generator.random((999, 2, 1))
        expected = np.count_nonzero(np.abs(draws[:, 0, 0] - draws[:, 1, 0]) >= 0.9)
        method = uniformity.ehy
    else:
        observed = [[0.1, 0.9], [0.9, 0.1]]
        draws = generator.dirichlet(np.ones(2), size=(999, 2))
        expected = np.count_nonzero(np.abs(draws[:, 0, 0] - draws[:, 1, 0]) >= 0.8)
        method = simplex.ehy_uniformity
    for alpha in (2.0, 1.7e308):
        actual = method(observed, alpha=alpha, n_neighbors=1, n_resamples=999, rng=0)
        assert actual.exceedances == expected
        assert actual.pvalue == (expected + 1) / 1000


def test_mmd_observed_allocation_is_identical_in_every_batch_shape() -> None:
    from pysht.equaldist import (
        _mmd_unbiased_statistic,
        _mmd_unbiased_statistic_batch,
    )

    generator = np.random.default_rng(95)
    x, y = generator.normal(size=(2, 1)), generator.normal(size=(3, 1))
    pooled = np.vstack((x, y))
    gram, *_ = kernel_matrix(
        distance_geometry(pooled), kernel="rbf", bandwidth=10.0, offset="auto"
    )
    allocations = list(combinations(range(5), 2))
    plans = np.ones((len(allocations), 5), dtype=np.intp)
    scalar = []
    for i, first in enumerate(allocations):
        plans[i, list(first)] = 0
        scalar.append(
            _mmd_unbiased_statistic(
                gram,
                (np.array(first), np.array([j for j in range(5) if j not in first])),
            )
        )
    for batch_size in (1, 2, 3, 10):
        batched = np.concatenate(
            [
                _mmd_unbiased_statistic_batch(
                    gram, plans[start : start + batch_size], (2, 3)
                )
                for start in range(0, len(plans), batch_size)
            ]
        )
        np.testing.assert_array_equal(batched, scalar)

    with localcontext() as context:
        context.prec = 75
        z = [Decimal.from_float(float(v)) for v in pooled[:, 0]]
        kernels = [[(-(((a - b) / 10) ** 2) / 2).exp() for b in z] for a in z]

        def statistic(first: tuple[int, ...]) -> Decimal:
            second = [j for j in range(5) if j not in first]
            return (
                kernels[first[0]][first[1]]
                + sum(kernels[i][j] for i, j in combinations(second, 2)) / 3
                - sum(kernels[i][j] for i in first for j in second) / 3
            )

        observed = statistic((0, 1))
        count = sum(statistic(first) >= observed for first in allocations)
    actual = equaldist.mmd_2samp(
        x, y, bandwidth=10.0, calibration="exact", n_resamples=10
    )
    assert actual.statistic == pytest.approx(float(observed), rel=5e-13)
    assert actual.exceedances == count
    assert actual.pvalue == count / 10
    assert actual.pvalue >= 1 / 10

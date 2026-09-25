"""Formula-independent regressions from the September 2026 correctness audit."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from pysht import mean


def _hotelling_reference(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    delta = x.mean(axis=0) - y.mean(axis=0)
    pooled = (
        (len(x) - 1) * np.cov(x, rowvar=False) + (len(y) - 1) * np.cov(y, rowvar=False)
    ) / (len(x) + len(y) - 2)
    return float(
        len(x) * len(y) / (len(x) + len(y)) * delta @ np.linalg.solve(pooled, delta)
    )


@pytest.mark.parametrize("dimension", [1, 2, 3, 5])
def test_johansen_general_contrast_formula(dimension: int) -> None:
    rng = np.random.default_rng(83)
    x = rng.normal(size=(10, dimension))
    y = rng.normal(size=(12, dimension)) * 1.7
    sigma = np.zeros((2 * dimension, 2 * dimension))
    sigma[:dimension, :dimension] = np.cov(x, rowvar=False) / len(x)
    sigma[dimension:, dimension:] = np.cov(y, rowvar=False) / len(y)
    contrast = np.hstack((np.eye(dimension), -np.eye(dimension)))
    covariance = contrast @ sigma @ contrast.T
    delta = x.mean(axis=0) - y.mean(axis=0)
    statistic = float(delta @ np.linalg.solve(covariance, delta))
    weight = sigma @ contrast.T @ np.linalg.solve(covariance, contrast)
    correction = 0.0
    for index, size in enumerate((len(x), len(y))):
        block = weight[
            index * dimension : (index + 1) * dimension,
            index * dimension : (index + 1) * dimension,
        ]
        correction += (
            0.5 * (np.trace(block @ block) + np.trace(block) ** 2) / (size - 1)
        )
    degrees = dimension * (dimension + 2) / (3 * correction)
    divisor = dimension + 2 * correction - 6 * correction / (dimension + 2)
    result = mean.johansen_2samp(x, y)
    assert result.statistic == pytest.approx(statistic, rel=1e-11)
    assert result.df == pytest.approx((dimension, degrees), rel=1e-11)
    assert result.pvalue == pytest.approx(
        stats.f.sf(statistic / divisor, dimension, degrees), rel=1e-11
    )


def test_johansen_univariate_welch_identity() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    expected = stats.ttest_ind(x, y, equal_var=False)
    result = mean.johansen_2samp(x[:, None], y[:, None])
    assert result.statistic == pytest.approx(expected.statistic**2, rel=1e-12)
    assert result.df == pytest.approx((1.0, expected.df), rel=1e-12)
    assert result.pvalue == pytest.approx(expected.pvalue, rel=1e-12)


def _dependent_samples() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(0)
    base_x = rng.integers(-10, 11, size=(10, 2)).astype(float)
    base_y = rng.integers(-10, 11, size=(12, 2)).astype(float)
    return (
        np.column_stack((base_x, base_x.sum(axis=1))),
        np.column_stack((base_y, base_y.sum(axis=1))),
    )


@pytest.mark.parametrize("units", [np.ones(3), np.array([1e-150, 1.0, 1e150])])
def test_hotelling_rejects_exact_dependency_in_all_designs(
    units: NDArray[np.float64],
) -> None:
    x, y = _dependent_samples()
    x, y = x * units, y * units
    with pytest.raises(ValueError, match="dependent features"):
        mean.hotelling_1samp(x)
    with pytest.raises(ValueError, match="dependent features"):
        mean.hotelling_2samp(x, y)
    with pytest.raises(ValueError, match="dependent features"):
        mean.hotelling_2samp(x, y[: len(x)], paired=True)


@pytest.mark.parametrize(
    "name", ["yao_2samp", "nvm_2samp", "ky_2samp", "johansen_2samp"]
)
def test_behrens_fisher_rejects_exact_dependent_directions(name: str) -> None:
    with pytest.raises(ValueError, match="dependent features"):
        getattr(mean, name)(*_dependent_samples())


def test_johansen_rejects_singular_individual_covariance_with_full_rank_union() -> None:
    x, _ = _dependent_samples()
    y = np.random.default_rng(2).normal(size=(12, 3))
    with pytest.raises(ValueError, match="x covariance contribution"):
        mean.johansen_2samp(x, y)
    # Other Behrens--Fisher approximations require only the covariance union.
    for function in (mean.yao_2samp, mean.nvm_2samp, mean.ky_2samp):
        assert 0.0 <= function(x, y).pvalue <= 1.0


def test_hotelling_rank_check_preserves_extreme_feature_units() -> None:
    rng = np.random.default_rng(91)
    x, y = rng.normal(size=(12, 3)), rng.normal(size=(14, 3))
    units = np.array([1e-150, 1.0, 1e150])
    for baseline, scaled in (
        (mean.hotelling_1samp(x), mean.hotelling_1samp(x * units)),
        (mean.hotelling_2samp(x, y), mean.hotelling_2samp(x * units, y * units)),
        (
            mean.hotelling_2samp(x, y[:12], paired=True),
            mean.hotelling_2samp(x * units, y[:12] * units, paired=True),
        ),
    ):
        assert scaled.statistic == pytest.approx(baseline.statistic, rel=1e-12)
        assert scaled.pvalue == pytest.approx(baseline.pvalue, rel=1e-12)


@pytest.mark.parametrize("name", ["ljw_2samp", "thulin_2samp"])
def test_randomized_mean_rejects_dependent_selected_covariance(name: str) -> None:
    x, y = _dependent_samples()
    # Four observations per group select all three directions.
    options = {"rng": 123, "n_resamples": 1}
    if name == "thulin_2samp":
        options["n_subspaces"] = 1
    with pytest.raises(ValueError, match="dependent features"):
        getattr(mean, name)(x[:4], y[:4], **options)


@pytest.mark.parametrize("name,seed", [("thulin_2samp", 4), ("ljw_2samp", 9)])
def test_full_dimension_permutation_tail_includes_all_ties(
    name: str, seed: int
) -> None:
    pool = np.random.default_rng(seed).normal(size=(6, 2))
    allocations = list(itertools.combinations(range(6), 3))
    statistics = []
    for allocation in allocations:
        complement = [index for index in range(6) if index not in allocation]
        statistics.append(
            _hotelling_reference(pool[list(allocation)], pool[complement])
        )
    selected = allocations[int(np.argmax(statistics))]
    x = pool[list(selected)]
    y = pool[[index for index in range(6) if index not in selected]]
    groups = sorted(
        (x[np.lexsort((x[:, 1], x[:, 0]))], y[np.lexsort((y[:, 1], y[:, 0]))]),
        key=lambda group: tuple(group.ravel()),
    )
    ordered = np.vstack(groups)
    observed = _hotelling_reference(*groups)
    lookup = {}
    for allocation in allocations:
        complement = [index for index in range(6) if index not in allocation]
        lookup[frozenset(allocation)] = _hotelling_reference(
            ordered[list(allocation)], ordered[complement]
        )
    tolerance = 1e-10 * observed
    assert sum(statistic >= observed - tolerance for statistic in lookup.values()) == 2
    generator = np.random.default_rng(123)
    budget = 9999
    options = {"rng": 123, "n_resamples": budget}
    if name == "thulin_2samp":
        options["n_subspaces"] = 1
        generator.choice(2, size=2, replace=False)
    else:
        options["calibration"] = "monte-carlo"
        generator.standard_normal((2, 2))
    expected = sum(
        lookup[frozenset(generator.permutation(6)[:3])] >= observed - tolerance
        for _ in range(budget)
    )
    assert expected == 1018
    result = getattr(mean, name)(x, y, **options)
    assert result.exceedances == expected
    assert result.pvalue == 0.1019
    # Deterministic canonicalization must preserve the same random stream.
    for first, second in ((y, x), (x[::-1], y[[1, 2, 0]])):
        assert getattr(mean, name)(first, second, **options) == result


def test_chen_qin_trace_variant_matches_independent_pair_differences() -> None:
    # Four distinct observations split into two independent differences have
    # covariances 2 Sigma, so E[((Xi-Xj)'(Xk-Xl))**2 / 4] = tr(Sigma**2).
    values = np.random.default_rng(13).normal(size=(8, 5))
    independent = math.fsum(
        float((values[i] - values[j]) @ (values[k] - values[ell])) ** 2 / 4
        for i, j, k, ell in itertools.permutations(range(len(values)), 4)
    ) / math.perm(len(values), 4)
    for transformed in (values, values - values.mean(axis=0), values + np.arange(5)):
        assert mean._trace_covariance_square_u(transformed) == pytest.approx(
            independent, rel=1e-12
        )
    # Preserve the precise provenance distinction established in the audit.
    centered = values - values.mean(axis=0)
    literal_cq = 0.0
    for i, j in itertools.permutations(range(len(values)), 2):
        omitted = [index for index in range(len(values)) if index not in (i, j)]
        leave_two_out = centered[omitted].mean(axis=0)
        literal_cq += float(centered[i] @ (centered[j] - leave_two_out)) * float(
            centered[j] @ (centered[i] - leave_two_out)
        )
    literal_cq /= len(values) * (len(values) - 1)
    assert not math.isclose(independent, literal_cq, rel_tol=1e-4)


@pytest.mark.parametrize("name", ["ljw_2samp", "thulin_2samp"])
def test_every_observed_allocation_matches_independent_orbit_tail(name: str) -> None:
    pool = np.random.default_rng(52).normal(size=(6, 2))
    allocations = list(itertools.combinations(range(6), 3))
    for observed_allocation in allocations:
        x = pool[list(observed_allocation)]
        y = pool[[index for index in range(6) if index not in observed_allocation]]
        ordered_groups = sorted(
            (x[np.lexsort((x[:, 1], x[:, 0]))], y[np.lexsort((y[:, 1], y[:, 0]))]),
            key=lambda group: tuple(group.ravel()),
        )
        ordered = np.vstack(ordered_groups)
        observed = _hotelling_reference(*ordered_groups)
        tail = {}
        for allocation in allocations:
            complement = [index for index in range(6) if index not in allocation]
            candidate = _hotelling_reference(
                ordered[list(allocation)], ordered[complement]
            )
            tail[frozenset(allocation)] = candidate >= observed - 1e-10 * abs(observed)
        generator = np.random.default_rng(123)
        options = {"rng": 123, "n_resamples": 39}
        if name == "thulin_2samp":
            options["n_subspaces"] = 1
            generator.choice(2, size=2, replace=False)
        else:
            options["calibration"] = "monte-carlo"
            generator.standard_normal((2, 2))
        expected = sum(tail[frozenset(generator.permutation(6)[:3])] for _ in range(39))
        result = getattr(mean, name)(x, y, **options)
        assert result.exceedances == expected, (name, observed_allocation)
        assert getattr(mean, name)(y[::-1], x[[1, 2, 0]], **options) == result


@pytest.mark.parametrize("scale", [1e-8, 1e-150])
def test_full_dimension_ljw_avoids_ill_conditioned_projection_mixing(
    scale: float,
) -> None:
    pool = np.random.default_rng(9).normal(size=(6, 2))
    x, y = pool[:3], pool[3:]
    expected = mean.hotelling_2samp(x, y)
    result = mean.ljw_2samp(x * [1.0, scale], y * [1.0, scale], rng=123)
    assert result.statistic == pytest.approx(expected.statistic, rel=2e-12)
    assert result.pvalue == pytest.approx(expected.pvalue, rel=2e-12)


def test_hotelling_and_behrens_fisher_handle_near_collinearity_from_observations() -> (
    None
):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(12, 3))
    y = rng.normal(size=(15, 3))
    # Invertible mixing makes covariance formation lose the small direction.
    transform = np.array([[1.0, 1.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1.0]])
    for function in (
        mean.hotelling_2samp,
        mean.yao_2samp,
        mean.ky_2samp,
        mean.johansen_2samp,
    ):
        expected = function(x, y)
        observed = function(x @ transform, y @ transform)
        # Mixing the input itself loses O(eps/1e-8) relative information.
        assert observed.statistic == pytest.approx(expected.statistic, rel=5e-7)
        assert observed.pvalue == pytest.approx(expected.pvalue, rel=5e-7)


def test_feature_fallback_preserves_other_columns_ulp_variation() -> None:
    rng = np.random.default_rng(73)
    raw = rng.uniform(-1.0, 1.0, size=(13, 2))
    raw[:, 0] *= 1.6e308
    origin = 1e200
    step = np.spacing(origin)
    raw[:, 1] = origin + rng.integers(0, 20, size=13) * step
    x, y = raw[:6], raw[6:]
    # The first feature requires divide-before-subtract, while the second
    # must subtract its own anchor first to preserve the encoded increments.
    reference = np.column_stack((raw[:, 0] / 1e308, (raw[:, 1] - origin) / step))
    for function in (
        mean.hotelling_2samp,
        mean.yao_2samp,
        mean.ky_2samp,
        mean.johansen_2samp,
    ):
        result = function(x, y)
        expected = function(reference[:6], reference[6:])
        assert result.statistic == pytest.approx(expected.statistic, rel=2e-12)
        assert result.pvalue == pytest.approx(expected.pvalue, rel=2e-12)

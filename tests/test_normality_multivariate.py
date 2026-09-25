"""Literal, calibration, invariance, and boundary checks for MVN tests."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy import special
from scipy.spatial import distance

from pysht.normality import energy, henze_zirkler


def _whiten(x: np.ndarray, *, denominator: int) -> np.ndarray:
    centered = x - np.mean(x, axis=0)
    covariance = centered.T @ centered / denominator
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return centered @ (eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T)


def _hz_literal(x: np.ndarray) -> float:
    n, dimension = x.shape
    y = _whiten(x, denominator=n)
    beta = (n * (2.0 * dimension + 1.0) / 4.0) ** (1.0 / (dimension + 4.0)) / math.sqrt(
        2.0
    )
    squared_distances = distance.squareform(distance.pdist(y, metric="sqeuclidean"))
    squared_norms = np.sum(y * y, axis=1)
    return float(
        n
        * (
            np.mean(np.exp(-0.5 * beta**2 * squared_distances))
            - 2.0
            * (1.0 + beta**2) ** (-dimension / 2.0)
            * np.mean(np.exp(-(beta**2) * squared_norms / (2.0 * (1.0 + beta**2))))
            + (1.0 + 2.0 * beta**2) ** (-dimension / 2.0)
        )
    )


def _energy_literal(x: np.ndarray) -> float:
    n, dimension = x.shape
    # The 2005 procedure uses the ordinary sample covariance (n - 1).
    y = _whiten(x, denominator=n - 1)
    gamma_ratio = math.exp(
        special.gammaln((dimension + 1.0) / 2.0) - special.gammaln(dimension / 2.0)
    )
    expected_to_normal = (
        math.sqrt(2.0)
        * gamma_ratio
        * special.hyp1f1(
            -0.5,
            dimension / 2.0,
            -0.5 * np.sum(y * y, axis=1),
        )
    )
    return float(
        n
        * (
            2.0 * np.mean(expected_to_normal)
            - 2.0 * gamma_ratio
            - 2.0 * np.sum(distance.pdist(y)) / n**2
        )
    )


@pytest.fixture
def sample() -> np.ndarray:
    return np.array(
        [
            [-1.2, 0.3, 1.1],
            [-0.8, -0.6, 0.4],
            [-0.2, 1.4, -0.7],
            [0.1, -1.1, 0.8],
            [0.5, 0.7, -1.3],
            [0.9, -0.2, 0.2],
            [1.4, 1.0, 1.5],
            [2.1, -1.4, -0.4],
        ]
    )


def test_statistics_match_independent_literal_equations(sample: np.ndarray) -> None:
    hz = henze_zirkler(sample, n_resamples=19, rng=3)
    sr = energy(sample, n_resamples=19, rng=3)
    np.testing.assert_allclose(hz.statistic, _hz_literal(sample), rtol=2e-13)
    np.testing.assert_allclose(sr.statistic, _energy_literal(sample), rtol=3e-13)
    assert hz.statistic_name == "HZ"
    assert sr.statistic_name == "E"


def test_energy_matches_authors_reference_iris_fixture() -> None:
    # Fisher's iris Setosa measurements; energy::mvnorm.e reports 1.203397.
    iris_setosa = np.array(
        [
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5.0, 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [5.4, 3.7, 1.5, 0.2],
            [4.8, 3.4, 1.6, 0.2],
            [4.8, 3.0, 1.4, 0.1],
            [4.3, 3.0, 1.1, 0.1],
            [5.8, 4.0, 1.2, 0.2],
            [5.7, 4.4, 1.5, 0.4],
            [5.4, 3.9, 1.3, 0.4],
            [5.1, 3.5, 1.4, 0.3],
            [5.7, 3.8, 1.7, 0.3],
            [5.1, 3.8, 1.5, 0.3],
            [5.4, 3.4, 1.7, 0.2],
            [5.1, 3.7, 1.5, 0.4],
            [4.6, 3.6, 1.0, 0.2],
            [5.1, 3.3, 1.7, 0.5],
            [4.8, 3.4, 1.9, 0.2],
            [5.0, 3.0, 1.6, 0.2],
            [5.0, 3.4, 1.6, 0.4],
            [5.2, 3.5, 1.5, 0.2],
            [5.2, 3.4, 1.4, 0.2],
            [4.7, 3.2, 1.6, 0.2],
            [4.8, 3.1, 1.6, 0.2],
            [5.4, 3.4, 1.5, 0.4],
            [5.2, 4.1, 1.5, 0.1],
            [5.5, 4.2, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.2],
            [5.0, 3.2, 1.2, 0.2],
            [5.5, 3.5, 1.3, 0.2],
            [4.9, 3.6, 1.4, 0.1],
            [4.4, 3.0, 1.3, 0.2],
            [5.1, 3.4, 1.5, 0.2],
            [5.0, 3.5, 1.3, 0.3],
            [4.5, 2.3, 1.3, 0.3],
            [4.4, 3.2, 1.3, 0.2],
            [5.0, 3.5, 1.6, 0.6],
            [5.1, 3.8, 1.9, 0.4],
            [4.8, 3.0, 1.4, 0.3],
            [5.1, 3.8, 1.6, 0.2],
            [4.6, 3.2, 1.4, 0.2],
            [5.3, 3.7, 1.5, 0.2],
            [5.0, 3.3, 1.4, 0.2],
        ]
    )
    result = energy(iris_setosa, n_resamples=7, rng=1)
    np.testing.assert_allclose(result.statistic, 1.2033967029263737, rtol=2e-13)


@pytest.mark.parametrize("function", [henze_zirkler, energy])
def test_full_rank_affine_and_row_invariance(
    function: object, sample: np.ndarray
) -> None:
    matrix = np.array([[1.2, -0.4, 0.3], [0.1, 0.8, 0.7], [-0.2, 0.5, 1.4]])
    baseline = function(sample, n_resamples=59, rng=812)  # type: ignore[operator]
    transformed = function(  # type: ignore[operator]
        sample[::-1] @ matrix + np.array([10.0, -4.0, 7.0]),
        n_resamples=59,
        rng=812,
    )
    np.testing.assert_allclose(transformed.statistic, baseline.statistic, rtol=2e-12)
    assert transformed.exceedances == baseline.exceedances
    assert transformed.pvalue == baseline.pvalue


@pytest.mark.parametrize(
    "function,literal", [(henze_zirkler, _hz_literal), (energy, _energy_literal)]
)
def test_seeded_monte_carlo_refits_each_null_sample(
    function: object, literal: object, sample: np.ndarray
) -> None:
    seed, resamples = 91, 37
    result = function(sample, n_resamples=resamples, rng=seed)  # type: ignore[operator]
    generator = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(resamples):
        simulated = generator.standard_normal(sample.shape)
        exceedances += int(literal(simulated) >= result.statistic)  # type: ignore[operator]
    assert result.exceedances == exceedances
    assert result.pvalue == (exceedances + 1) / (resamples + 1)
    assert "parameter refitting" in result.calibration


@pytest.mark.parametrize("function", [henze_zirkler, energy])
def test_rank_and_shape_boundaries(function: object) -> None:
    with pytest.raises(ValueError, match="at least two features"):
        function(np.arange(8.0)[:, None], n_resamples=3)  # type: ignore[operator]
    with pytest.raises(ValueError, match="more observations than features"):
        function(np.eye(3), n_resamples=3)  # type: ignore[operator]
    collinear = np.column_stack((np.arange(6.0), 2.0 * np.arange(6.0)))
    with pytest.raises(ValueError, match="positive definite"):
        function(collinear, n_resamples=3)  # type: ignore[operator]
    with pytest.raises(ValueError, match="calibration"):
        function(np.arange(18.0).reshape(6, 3), calibration="asymptotic")  # type: ignore[operator]


@pytest.mark.parametrize("function", [henze_zirkler, energy])
def test_named_heavy_tail_power(function: object) -> None:
    rng = np.random.default_rng(20260901)
    alternative = rng.standard_t(df=2.0, size=(80, 2))
    result = function(alternative, n_resamples=499, rng=20260902)  # type: ignore[operator]
    assert result.pvalue < 0.05


@pytest.mark.parametrize("function", [henze_zirkler, energy])
def test_9999_draw_default_completes_with_quadratic_storage(function: object) -> None:
    sample = np.array([[-1.0, 0.2], [-0.2, -0.7], [0.4, 1.1], [1.2, -0.1]])
    started = time.perf_counter()
    result = function(sample, rng=20260903)  # type: ignore[operator]
    assert result.n_resamples == 9_999
    assert time.perf_counter() - started < 30.0

"""Primary-formula checks for Chen--Zhang--Zhong covariance tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from pysht import covariance
from pysht.covariance import czz_identity_1samp, czz_sphericity_1samp


def _literal_trace_square(values: NDArray[np.float64]) -> float:
    n = len(values)
    first = second = third = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gij = float(values[i] @ values[j])
            first += gij * gij
            for k in range(n):
                if k in (i, j):
                    continue
                second += gij * float(values[j] @ values[k])
                for ell in range(n):
                    if ell in (i, j, k):
                        continue
                    third += gij * float(values[k] @ values[ell])
    return (
        first / (n * (n - 1))
        - 2 * second / (n * (n - 1) * (n - 2))
        + third / (n * (n - 1) * (n - 2) * (n - 3))
    )


@pytest.fixture
def sample() -> NDArray[np.float64]:
    return np.array(
        [
            [0.2, -0.4, 1.1],
            [1.3, 0.7, -0.2],
            [-0.8, 0.1, 0.5],
            [0.4, -1.2, 0.3],
            [1.0, 0.5, -0.9],
            [-0.3, 1.4, 0.8],
        ]
    )


def test_czz_identity_matches_equations_2_2_and_2_3(
    sample: NDArray[np.float64],
) -> None:
    centered = sample - np.mean(sample, axis=0)
    t1 = float(np.sum(centered * centered)) / (len(sample) - 1)
    t2 = _literal_trace_square(centered)
    expected = len(sample) / 2 * (t2 / sample.shape[1] - 2 * t1 / sample.shape[1] + 1)
    actual = czz_identity_1samp(sample)

    np.testing.assert_allclose(actual.statistic, expected, rtol=3e-14)
    np.testing.assert_allclose(actual.pvalue, stats.norm.sf(expected), rtol=3e-14)


def test_czz_identity_whitening_equivalence(sample: NDArray[np.float64]) -> None:
    factor = np.array([[1.3, 0.0, 0.0], [0.2, 0.9, 0.0], [-0.1, 0.3, 1.1]])
    covariance_null = factor @ factor.T
    transformed = sample @ factor.T + np.array([4.0, -2.0, 1.5])
    baseline = czz_identity_1samp(sample)
    actual = czz_identity_1samp(transformed, popcov=covariance_null)
    np.testing.assert_allclose(actual.statistic, baseline.statistic, rtol=2e-13)
    np.testing.assert_allclose(actual.pvalue, baseline.pvalue, rtol=2e-13)


def test_czz_identity_whitens_heterogeneous_covariance_units() -> None:
    standardized = np.random.default_rng(7003).normal(size=(100, 3))
    scales = np.array([1.0e-150, 1.0, 1.0e150])
    values = standardized * scales
    popcov = np.diag(scales * scales)

    baseline = czz_identity_1samp(standardized)
    actual = czz_identity_1samp(values, popcov=popcov)

    np.testing.assert_allclose(actual.statistic, baseline.statistic, rtol=3e-13)
    np.testing.assert_allclose(actual.pvalue, baseline.pvalue, rtol=3e-13)


def test_czz_identity_retains_extreme_finite_scalar_alternatives() -> None:
    values = np.random.default_rng(7000).normal(size=(100, 20))

    tiny = czz_identity_1samp(1.0e-300 * values)
    huge = czz_identity_1samp(1.0e100 * values)

    # As Sigma tends to zero, V_n tends to one and n V_n / 2 tends to n/2.
    np.testing.assert_allclose(tiny.statistic, 50.0, rtol=0.0, atol=0.0)
    assert tiny.pvalue == stats.norm.sf(50.0)
    # The normalized order-four estimate is positive for this fixed fixture,
    # so restoring the fourth power gives an overwhelming upper-tail result.
    assert huge.statistic == math.inf
    assert huge.pvalue == 0.0


def test_czz_default_identity_does_not_allocate_a_feature_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.random.default_rng(7001).normal(size=(6, 5_000))

    def forbidden_identity(*args: object, **kwargs: object) -> NDArray[np.float64]:
        del args, kwargs
        raise AssertionError("the default identity null must remain implicit")

    monkeypatch.setattr(covariance.np, "eye", forbidden_identity)

    result = czz_identity_1samp(values)

    assert math.isfinite(result.statistic)


def test_czz_sphericity_matches_equation_2_1(sample: NDArray[np.float64]) -> None:
    centered = sample - np.mean(sample, axis=0)
    t1 = float(np.sum(centered * centered)) / (len(sample) - 1)
    t2 = _literal_trace_square(centered)
    expected = len(sample) / 2 * (sample.shape[1] * t2 / t1**2 - 1)
    actual = czz_sphericity_1samp(sample)
    np.testing.assert_allclose(actual.statistic, expected, rtol=3e-14)
    np.testing.assert_allclose(actual.pvalue, stats.norm.sf(expected), rtol=3e-14)


def test_czz_sphericity_invariances(sample: NDArray[np.float64]) -> None:
    rotation, _ = np.linalg.qr(
        np.array([[1.0, 2.0, -0.4], [0.3, -0.7, 1.2], [1.1, 0.2, 0.6]])
    )
    baseline = czz_sphericity_1samp(sample)
    transformed = czz_sphericity_1samp(
        1e130 * (sample @ rotation + np.array([8.0, -4.0, 3.0]))
    )
    permuted = czz_sphericity_1samp(sample[::-1])
    np.testing.assert_allclose(transformed.statistic, baseline.statistic, rtol=3e-13)
    np.testing.assert_allclose(permuted.statistic, baseline.statistic, rtol=3e-13)


def test_czz_boundaries_are_explicit(sample: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="at least 4 observations"):
        czz_identity_1samp(sample[:3])
    with pytest.raises(ValueError, match="at least two features"):
        czz_sphericity_1samp(sample[:, :1])
    with pytest.raises(ValueError, match="positive definite"):
        czz_identity_1samp(sample, popcov=np.diag([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="covariance trace"):
        czz_sphericity_1samp(np.ones((5, 3)))


def test_advanced_candidates_remain_private_until_their_gates_pass() -> None:
    assert "ylx_2samp" not in covariance.__all__
    assert "jwjwz_2samp" not in covariance.__all__
    assert hasattr(covariance, "_ylx_2samp")
    assert not hasattr(covariance, "jwjwz_2samp")
    assert not hasattr(covariance, "lyl_2samp")


def test_ylx_private_formula_combines_log_tails() -> None:
    rng = np.random.default_rng(901)
    x = rng.normal(size=(10, 5))
    y = rng.normal(size=(11, 5))
    dense = covariance.lc_2samp(x, y)
    sparse = covariance.clx_2samp(x, y)
    actual = covariance._ylx_2samp(x, y)
    expected = -2 * (
        float(stats.norm.logsf(dense.statistic))
        + covariance._clx_log_tail(sparse.statistic, x.shape[1])
    )
    np.testing.assert_allclose(actual.statistic, expected, rtol=2e-14)
    np.testing.assert_allclose(actual.pvalue, stats.chi2.sf(expected, 4), rtol=2e-14)
    assert math.isfinite(actual.statistic)

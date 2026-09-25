"""Independent mathematical boundary regressions from the September audit."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import special, stats

from pysht import covariance, normality
from pysht._resampling import upper_tail_threshold


def _saturated_normality_statistic(name: str, dimension: int) -> float:
    """Use the fixed regular-simplex geometry, without whitening a sample."""
    n = dimension + 1
    if name == "henze_zirkler":
        beta2 = (n * (2 * dimension + 1) / 4) ** (2 / (dimension + 4)) / 2
        return n * (
            (1 + (n - 1) * math.exp(-beta2 * n)) / n
            - 2
            * (1 + beta2) ** (-dimension / 2)
            * math.exp(-beta2 * dimension / (2 * (1 + beta2)))
            + (1 + 2 * beta2) ** (-dimension / 2)
        )
    gamma_ratio = math.exp(
        special.gammaln((dimension + 1) / 2) - special.gammaln(dimension / 2)
    )
    squared_norm = (n - 1) * dimension / n
    expectation = (
        math.sqrt(2)
        * gamma_ratio
        * special.hyp1f1(-0.5, dimension / 2, -squared_norm / 2)
    )
    return n * (
        2 * expectation - 2 * gamma_ratio - (n - 1) / n * math.sqrt(2 * (n - 1))
    )


@pytest.mark.parametrize("name", ["henze_zirkler", "energy"])
@pytest.mark.parametrize("dimension", [2, 3, 5, 10])
def test_saturated_normality_is_a_point_mass(name: str, dimension: int) -> None:
    function = getattr(normality, name)
    generator = np.random.default_rng(20260915)
    expected = _saturated_normality_statistic(name, dimension)
    for _ in range(10):
        # Non-Gaussian samples have the same geometry too: this case has no power.
        sample = generator.standard_t(3, size=(dimension + 1, dimension))
        sample *= np.geomspace(1e-120, 1e120, dimension)
        before = sample.copy()
        stream = np.random.default_rng(91)
        replay = np.random.default_rng(91)
        result = function(sample, n_resamples=37, rng=stream)
        assert result.statistic == pytest.approx(expected, rel=2e-13, abs=2e-13)
        assert result.pvalue == 1
        assert result.exceedances == result.n_resamples == 37
        assert result.monte_carlo_standard_error == 0
        assert dict(result.diagnostics)["degenerate affine geometry"] is True
        for _ in range(37):
            replay.standard_normal(sample.shape)
        assert stream.bit_generator.state == replay.bit_generator.state
        np.testing.assert_array_equal(sample, before)


@pytest.mark.parametrize(
    "name,dimension,index", [("henze_zirkler", 2, 41), ("energy", 5, 9)]
)
def test_previously_false_normality_rejections_are_ties(
    name: str, dimension: int, index: int
) -> None:
    generator = np.random.default_rng(20260910 + dimension)
    for _ in range(index + 1):
        sample = generator.normal(size=(dimension + 1, dimension))
    result = getattr(normality, name)(sample, n_resamples=9_999, rng=42)
    assert result.pvalue == 1
    assert result.exceedances == 9_999


@pytest.mark.parametrize("name", ["henze_zirkler", "energy"])
def test_saturated_normality_still_validates_rank_and_controls(name: str) -> None:
    function = getattr(normality, name)
    with pytest.raises(ValueError, match="positive definite"):
        function([[0, 0], [1, 1], [2, 2]], n_resamples=3, rng=0)
    triangle = [[0, 0], [1, 0], [0, 1]]
    with pytest.raises(TypeError):
        function(triangle, n_resamples=True, rng=0)
    with pytest.raises(TypeError):
        function(triangle, n_resamples=3, rng=True)


@pytest.mark.parametrize("name", ["henze_zirkler", "energy"])
def test_normality_overflow_fallback_preserves_other_columns(name: str) -> None:
    first = np.array([-1, 1, 0, 0.5, -0.5, 0.2])
    base = 1e300
    second = base + np.spacing(base) * np.array([8, 6, 5, 2, 3, 0])
    raw = np.column_stack([first * np.finfo(float).max, second])
    safe = np.column_stack([first, (second - base) / np.spacing(base)])
    function = getattr(normality, name)
    actual = function(raw, n_resamples=99, rng=0)
    expected = function(safe, n_resamples=99, rng=0)
    assert actual.statistic == pytest.approx(expected.statistic, rel=2e-13)
    assert actual.pvalue == expected.pvalue


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_upper_tail_tolerance_retains_scale_and_order(scale: float) -> None:
    threshold = upper_tail_threshold(scale)
    assert np.nextafter(scale, 0) >= threshold
    assert scale * (1 - 1e-12) < threshold
    assert threshold == pytest.approx(upper_tail_threshold(1) * scale, rel=1e-15)
    assert upper_tail_threshold(0) == 0
    assert upper_tail_threshold(math.inf) == math.inf


@pytest.mark.parametrize(
    "units", [[1, 1e-8], [1e-150, 1e150], [-1e200, 1e-200], [1, 1]]
)
def test_schott_wald_respects_independent_feature_units(units: list[float]) -> None:
    generator = np.random.default_rng(5)
    x, y = generator.normal(size=(30, 2)), generator.normal(size=(40, 2))
    s1, s2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    pooled = (29 * s1 + 39 * s2) / 68
    inverse = np.linalg.inv(pooled)
    expected = (
        sum(
            degree
            * np.trace((sample @ inverse - np.eye(2)) @ (sample @ inverse - np.eye(2)))
            for degree, sample in ((29, s1), (39, s2))
        )
        / 2
    )
    result = covariance.schott_2001_ksamp(x * units, y * units)
    assert result.statistic == pytest.approx(expected, rel=2e-13)
    assert result.pvalue == pytest.approx(stats.chi2.sf(expected, 3), rel=2e-13)


def test_schott_wald_preserves_mixed_units_when_one_column_spans_endpoints() -> None:
    x = np.array([[-1, 1], [1, 0], [-1, -1], [1, 2]], dtype=float)
    y = np.array([[-1, 2], [1, -1], [1, 3], [-1, 0], [1, -2]], dtype=float)
    expected = covariance.schott_2001_ksamp(x, y)
    units = [np.finfo(float).max, 1e-200]
    actual = covariance.schott_2001_ksamp(x * units, y * units)
    assert actual.statistic == pytest.approx(expected.statistic, rel=2e-13)


def test_schott_wald_rank_is_pooled_within_group_rank() -> None:
    # Each group is singular, but their pooled within-group scatter is full rank.
    x = np.array([[-1, 0], [0, 0], [1, 0]], dtype=float)
    y = np.array([[0, -1], [0, 0], [0, 1]], dtype=float)
    baseline = covariance.schott_2001_ksamp(x, y)
    transformed = covariance.schott_2001_ksamp(x * [1e-100, 1e100], y * [1e-100, 1e100])
    assert transformed.statistic == pytest.approx(baseline.statistic)
    generator = np.random.default_rng(0)
    sample = generator.integers(-10, 11, size=(10, 2)).astype(float)
    singular = np.column_stack([sample, sample.sum(axis=1)])
    with pytest.raises(ValueError, match="positive definite"):
        covariance.schott_2001_ksamp(singular, singular[::-1] + [0, 100, 0])


def test_schott_wald_is_invariant_to_nearly_dependent_but_resolvable_features() -> None:
    generator = np.random.default_rng(581)
    x, y = generator.normal(size=(30, 2)), generator.normal(size=(40, 2))
    transform = np.array([[1, 1], [0, 1e-7]])
    baseline = covariance.schott_2001_ksamp(x, y)
    actual = covariance.schott_2001_ksamp(x @ transform, y @ transform)
    # The transformed data themselves retain only about eight useful digits.
    assert actual.statistic == pytest.approx(baseline.statistic, rel=1e-7)

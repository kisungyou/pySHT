"""Primary-equation and calibration tests for rectangular EHY uniformity."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from scipy import special
from scipy.spatial import distance

from pysht.uniformity import ehy


def _literal(x: np.ndarray, *, alpha: float, neighbors: int) -> float:
    n, dimension = x.shape
    radii = np.sort(distance.squareform(distance.pdist(x)), axis=1)[
        :, 1 : neighbors + 1
    ]
    unit_ball = math.pi ** (dimension / 2.0) / special.gamma(dimension / 2.0 + 1.0)
    return float(np.sum((unit_ball * n * radii**dimension) ** alpha))


def test_eq_1_and_4_sum_every_one_of_first_j_neighbors() -> None:
    x = np.array([[0.05], [0.22], [0.49], [0.71], [0.93]])
    result = ehy(x, alpha=2.0, n_neighbors=2, n_resamples=11, rng=1)
    np.testing.assert_allclose(
        result.statistic,
        _literal(x, alpha=2.0, neighbors=2),
        rtol=2e-14,
    )
    first_only = _literal(x, alpha=2.0, neighbors=1)
    assert result.statistic > first_only
    assert dict(result.diagnostics)["n_neighbors"] == 2


def test_seeded_monte_carlo_matches_literal_null_stream() -> None:
    x = np.array([[0.08, 0.17], [0.25, 0.81], [0.52, 0.39], [0.76, 0.62]])
    seed, resamples = 413, 31
    result = ehy(x, alpha=0.5, n_neighbors=2, n_resamples=resamples, rng=seed)
    generator = np.random.default_rng(seed)
    exceedances = sum(
        _literal(generator.random(x.shape), alpha=0.5, neighbors=2) <= result.statistic
        for _ in range(resamples)
    )
    assert result.exceedances == exceedances
    assert result.pvalue == (exceedances + 1) / (resamples + 1)
    assert dict(result.diagnostics)["rejection tail"] == "lower"


@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_identical_seeded_null_draw_is_counted_as_a_tie(alpha: float) -> None:
    seed = 53
    x = np.random.default_rng(seed).random((17, 4))
    result = ehy(
        x,
        alpha=alpha,
        n_neighbors=3,
        n_resamples=1,
        rng=seed,
    )
    assert result.exceedances == 1
    assert result.pvalue == 1.0


def test_row_feature_and_rectangular_affine_invariance() -> None:
    x = np.array([[0.08, 0.17], [0.25, 0.81], [0.52, 0.39], [0.76, 0.62], [0.91, 0.28]])
    baseline = ehy(x, alpha=2.0, n_neighbors=2, n_resamples=79, rng=5)
    lower = np.array([-20.0, 4.0])
    upper = np.array([3.0, 4.25])
    mapped = lower + x * (upper - lower)
    variants = (
        ehy(x[::-1], alpha=2.0, n_neighbors=2, n_resamples=79, rng=5),
        ehy(x[:, ::-1], alpha=2.0, n_neighbors=2, n_resamples=79, rng=5),
        ehy(
            mapped,
            alpha=2.0,
            n_neighbors=2,
            lower=lower,
            upper=upper,
            n_resamples=79,
            rng=5,
        ),
    )
    for actual in variants:
        np.testing.assert_allclose(actual.statistic, baseline.statistic, rtol=2e-14)
        assert actual.exceedances == baseline.exceedances


def test_seeded_feature_invariance_is_bitwise_at_last_bit_boundary() -> None:
    x = np.random.default_rng(123).random((7, 5))
    baseline = ehy(x, alpha=2.0, n_neighbors=3, n_resamples=127, rng=991)
    variant = ehy(
        x[::-1, [1, 3, 4, 2, 0]],
        alpha=2.0,
        n_neighbors=3,
        n_resamples=127,
        rng=991,
    )
    assert variant.statistic == baseline.statistic
    assert variant.exceedances == baseline.exceedances
    assert variant.pvalue == baseline.pvalue


def test_alpha_tail_and_domains_are_explicit() -> None:
    x = np.array([[0.1], [0.3], [0.6], [0.9]])
    upper = ehy(x, alpha=1.5, n_neighbors=1, n_resamples=9, rng=2)
    assert dict(upper.diagnostics)["rejection tail"] == "upper"
    for invalid in (-1.0, 0.0, 1.0, math.inf):
        with pytest.raises(ValueError, match="alpha"):
            ehy(x, alpha=invalid, n_neighbors=1, n_resamples=3)
    with pytest.raises(ValueError, match="smaller than"):
        ehy(x, alpha=2.0, n_neighbors=4, n_resamples=3)
    with pytest.raises(ValueError, match="within"):
        ehy(np.array([[-0.1], [0.2]]), alpha=2.0, n_neighbors=1, n_resamples=3)
    with pytest.raises(TypeError):
        ehy(x, alpha=2.0, n_neighbors=True, n_resamples=3)  # type: ignore[arg-type]


def test_duplicate_points_have_documented_zero_radius_boundary() -> None:
    x = np.tile([0.4, 0.6], (4, 1))
    result = ehy(x, alpha=0.5, n_neighbors=1, n_resamples=19, rng=8)
    assert result.statistic == 0.0
    assert result.pvalue == 1.0 / 20.0


def test_named_clustered_alternative_has_power() -> None:
    rng = np.random.default_rng(20260904)
    clustered = np.clip(rng.normal(0.2, 0.015, size=(45, 2)), 0.0, 1.0)
    result = ehy(
        clustered,
        alpha=0.5,
        n_neighbors=1,
        n_resamples=499,
        rng=20260905,
    )
    assert result.pvalue < 0.05


def test_9999_draw_default_completes_for_small_quadratic_problem() -> None:
    x = np.array([[0.05], [0.22], [0.49], [0.71], [0.93]])
    started = time.perf_counter()
    result = ehy(x, alpha=2.0, n_neighbors=2, rng=20260906)
    assert result.n_resamples == 9_999
    assert time.perf_counter() - started < 30.0

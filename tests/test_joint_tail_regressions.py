"""Independent references for joint-tail, rank, and confidence-bound fixes."""

from __future__ import annotations

import itertools
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import stats

from pysht import mean_covariance, mean_variance, variance


def test_joint_large_sample_tail_matches_positive_binomial_identity() -> None:
    amplitude = float(np.sqrt(3.0 / 7.0))
    sample = np.r_[-np.ones(5000), 0.0, np.ones(5000)]
    # For integer beta shapes, I_z(a,b)=P(Binomial(a+b-1,z)>=a).
    # Decimal's positive sum is independent of the production beta algorithm.
    with localcontext() as context:
        context.prec = 70
        ratio = Decimal.from_float(amplitude) ** 2
        z = ratio / (1 + ratio)
        term = Decimal(math.comb(9999, 5000)) * z**5000 * (1 - z) ** 4999
        tail = term
        for k in range(5000, 9999):
            term *= Decimal(9999 - k) / (k + 1) * z / (1 - z)
            tail += term
        expected_statistic = float(-2 * (2 * tail).ln())
        # Under H0, (U,V,W)~Dirichlet(5000,5000,1/2). Markov applied
        # to 1-4UV <= (U-V)^2+2W rigorously bounds the exact LRT tail.
        total_shape = Decimal("10000.5")
        gap = 1 - 4 * ratio / (1 + ratio) ** 2
        bound = float(
            (1 / total_shape + 10000 / (total_shape * (total_shape + 1))) / gap
        )
    for x, y in ((amplitude * sample, sample), (sample, amplitude * sample)):
        pl = mean_variance.pl_2samp(x, y)
        assert pl.statistic == pytest.approx(expected_statistic, rel=2e-12)
        assert pl.pvalue == 0.0  # The actual tail is below float64's range.
        exact = mean_variance.exact_lrt_2samp(x, y)
        assert 0.0 <= exact.pvalue <= bound


@pytest.mark.parametrize("shape", [0.5, 1.0, 5000.0])
@pytest.mark.parametrize("log_x", [-1000.0, -40.0, -1.0, -1e-20])
def test_beta_log_tail_retains_closed_form_endpoints(
    shape: float, log_x: float
) -> None:
    # Beta(a,1) has exact CDF x**a, including underflowed displayed x.
    actual = mean_variance._log_beta_lower(log_x, shape, 1.0)
    assert actual == pytest.approx(shape * log_x, rel=2e-12, abs=0.0)


def test_pl_retains_a_partially_subnormal_scale_ratio() -> None:
    a, b = 2.5e-301, 1e23
    # F(1,1) is a squared Cauchy. Here q=4/pi*atan(a/b); the omitted
    # cubic term in atan is below 1e-970, far below the asserted precision.
    log_q = math.log(4 / math.pi) + math.log(a) - math.log(b)
    expected_statistic = -2 * log_q
    expected_p = math.exp(log_q + math.log1p(-log_q))
    for x, y in (([-a, a], [-b, b]), ([-b, b], [-a, a])):
        result = mean_variance.pl_2samp(x, y)
        assert result.statistic == pytest.approx(expected_statistic, rel=2e-15)
        assert result.pvalue == expected_p


@pytest.mark.parametrize("level", [1e-20, 1e-100, 0.1, 0.9])
def test_one_sided_variance_bounds_match_degree_two_closed_forms(level: float) -> None:
    sample = [-1.0, 0.0, 1.0]
    chi_less = variance.chisquare_1samp(
        sample, alternative="less", confidence_level=level
    )
    chi_greater = variance.chisquare_1samp(
        sample, alternative="greater", confidence_level=level
    )
    f_less = variance.f_2samp(
        sample, sample, alternative="less", confidence_level=level
    )
    f_greater = variance.f_2samp(
        sample, sample, alternative="greater", confidence_level=level
    )
    # chi2_2 CDF=1-exp(-x/2); F_2,2 CDF=x/(1+x).
    assert chi_less.confidence_interval == pytest.approx(
        (0.0, 1 / -math.log(level)), rel=5e-14
    )
    assert chi_greater.confidence_interval == pytest.approx(
        (1 / -math.log1p(-level), math.inf), rel=5e-14
    )
    assert f_less.confidence_interval == pytest.approx(
        (0.0, level / (1 - level)), rel=5e-14, abs=0.0
    )
    assert f_greater.confidence_interval == pytest.approx(
        ((1 - level) / level, math.inf), rel=5e-14
    )


@pytest.mark.parametrize("level", [1e-200, 1e-320])
def test_chi_square_underflowed_quantile_preserves_finite_bound(level: float) -> None:
    amplitude = 1e-200
    # chi2_1's lower p quantile is pi*p**2/2 + O(p**4), from erf's
    # series. Both the sample sum of squares and quantile underflow, while
    # their ratio is representable. The omitted relative error is <1e-399.
    expected = math.exp(
        math.log(4 / math.pi) + 2 * (math.log(amplitude) - math.log(level))
    )
    result = variance.chisquare_1samp(
        [-amplitude, amplitude], alternative="greater", confidence_level=level
    )
    assert result.confidence_interval is not None
    assert result.confidence_interval[0] == pytest.approx(expected, rel=5e-12)


@pytest.mark.parametrize("dfs", [(1.0, 3.0), (2.0, 2.0), (30.0, 80.0)])
@pytest.mark.parametrize("probability", [0.001, 0.05, 0.5, 0.95, 0.999])
def test_f_quantile_preserves_ordinary_tail_inversion(
    dfs: tuple[float, float], probability: float
) -> None:
    q = variance._f_quantile(probability, *dfs)
    assert stats.f.cdf(q, *dfs) == pytest.approx(probability, rel=2e-12)
    upper = variance._f_upper_quantile(probability, *dfs)
    assert stats.f.sf(upper, *dfs) == pytest.approx(probability, rel=2e-12)


def test_joint_lrt_rejects_exact_dependence_for_every_column_order() -> None:
    sample = np.array(
        [[0, 0, 0], [3, 5, 8], [-5, -4, -9], [4, 5, 9], [-3, -2, -5], [4, -1, 3]],
        dtype=float,
    )
    assert np.array_equal(sample[:, 2], sample[:, 0] + sample[:, 1])
    for order in itertools.permutations(range(3)):
        for scales in (np.ones(3), np.array([1e-150, 1.0, 1e150])):
            with pytest.raises(ValueError, match="positive definite"):
                mean_covariance.lrt_1samp(sample[:, order] * scales)


@pytest.mark.parametrize("null", [0.0, 1e15, 1e20])
@pytest.mark.parametrize("null_variance", [1e-100, 1.0, 1e100])
def test_joint_lrt_estimates_variation_independently_of_null(
    null: float, null_variance: float
) -> None:
    sample = np.array([[-1.0], [0.0], [1.0]])
    ratio = (2 / 3) / null_variance
    expected = 3 * (ratio - 1 - math.log(ratio) + null * null / null_variance)
    result = mean_covariance.lrt_1samp(sample, popmean=[null], popcov=[[null_variance]])
    assert result.statistic == pytest.approx(expected, rel=3e-14)
    assert result.pvalue == pytest.approx(stats.chi2.sf(expected, 2), rel=3e-14)

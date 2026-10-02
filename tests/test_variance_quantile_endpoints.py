"""Closed-form F references beyond the range of intermediate quantiles."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

from pysht.variance import f_2samp


def _f_two_df_bound(
    level: float,
    second_df: int,
    first_amplitude: float,
    second_amplitude: float,
    *,
    upper: bool,
) -> float:
    """Use F(2,d)'s exact CDF and exact float sample amplitudes."""
    with localcontext() as context:
        context.prec = 380
        confidence = Decimal.from_float(level)
        d = Decimal(second_df)
        # x=(-a,0,a), y=(-b,0,...,0,b) have variances a**2 and 2*b**2/d.
        ratio = (
            Decimal.from_float(first_amplitude) ** 2
            * d
            / (2 * Decimal.from_float(second_amplitude) ** 2)
        )
        survival = confidence if upper else 1 - confidence
        quantile = d / 2 * ((-2 / d * survival.ln()).exp() - 1)
        return float(ratio / quantile)


@pytest.mark.parametrize("second_df", [1, 2, 3, 10_000])
@pytest.mark.parametrize("level", [1e-20, 1e-300, 1e-320])
def test_tiny_confidence_f_bounds_match_exact_f_two_df_inverse(
    second_df: int, level: float
) -> None:
    first_amplitude = 1e-150
    second_amplitude = math.sqrt(second_df / 2)
    x = np.array([-first_amplitude, 0.0, first_amplitude])
    y = np.zeros(second_df + 1)
    y[0], y[-1] = -second_amplitude, second_amplitude
    result = f_2samp(x, y, alternative="greater", confidence_level=level)
    expected = _f_two_df_bound(
        level, second_df, first_amplitude, second_amplitude, upper=False
    )
    assert result.confidence_interval is not None
    assert result.confidence_interval[0] == pytest.approx(expected, rel=5e-12)
    assert result.confidence_interval[1] == math.inf


@pytest.mark.parametrize("swapped", [False, True])
def test_out_of_range_f_quantile_preserves_finite_confidence_bound(
    swapped: bool,
) -> None:
    # For F(2,1), the upper 1e-200 quantile is about5e399. Its ratio with
    # the sample variance ratio about1e300 is still a finite bound near2e-100.
    # Swapping groups exercises the reciprocal quantile below float64 range.
    level = 1e-200
    first_amplitude = 1e150
    second_amplitude = math.sqrt(0.5)
    x = np.array([-first_amplitude, 0.0, first_amplitude])
    y = np.array([-second_amplitude, second_amplitude])
    expected_upper = _f_two_df_bound(
        level, 1, first_amplitude, second_amplitude, upper=True
    )
    result = (
        f_2samp(y, x, alternative="greater", confidence_level=level)
        if swapped
        else f_2samp(x, y, alternative="less", confidence_level=level)
    )
    assert result.confidence_interval is not None
    actual = result.confidence_interval[0 if swapped else 1]
    expected = 1.0 / expected_upper if swapped else expected_upper
    assert math.isfinite(actual) and actual > 0
    assert actual == pytest.approx(expected, rel=5e-12, abs=0.0)

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from pysht._resampling import exact_pvalue, monte_carlo_calibration


def test_exact_pvalue_uses_literal_enumeration_fraction() -> None:
    assert exact_pvalue(np.int64(8), np.int64(10)) == 0.8
    assert exact_pvalue(0, 10) == 0.0


def test_positive_tail_probabilities_must_be_representable() -> None:
    with pytest.raises(ValueError, match="positive but cannot be represented"):
        exact_pvalue(1, 10**1000)
    with pytest.raises(ValueError, match="cannot be represented"):
        monte_carlo_calibration(0, 10**1000)
    with pytest.raises(ValueError, match="less than one"):
        exact_pvalue(10**1000 - 1, 10**1000)
    with pytest.raises(ValueError, match="less than one"):
        monte_carlo_calibration(10**1000 - 1, 10**1000)


def test_unrepresentable_beta_shapes_have_a_stable_public_error() -> None:
    # SciPy's beta quantile kernel cannot resolve intervals once integer
    # shapes are far beyond float64 precision.  Do not leak its internal
    # dtype error from this public calibration helper.
    with pytest.raises(ValueError, match="binomial interval could not be represented"):
        monte_carlo_calibration(10**19, 10**20)
    with pytest.raises(ValueError, match="binomial interval could not be represented"):
        monte_carlo_calibration(10**400, 10**400)


def test_monte_carlo_calibration_matches_documented_estimators() -> None:
    pvalue, standard_error, interval = monte_carlo_calibration(8, 100)

    assert pvalue == 9 / 101
    assert standard_error == pytest.approx(math.sqrt(100 * 0.08 * 0.92) / 101)
    assert interval == pytest.approx(
        (
            stats.beta.ppf(0.025, 8, 93),
            stats.beta.ppf(0.975, 9, 92),
        )
    )


@pytest.mark.parametrize(
    ("exceedances", "expected_interval"),
    [(0, (0.0, 1.0 - 0.025 ** (1.0 / 20.0))), (20, (0.025 ** (1 / 20), 1.0))],
)
def test_monte_carlo_interval_handles_boundary_counts(
    exceedances: int,
    expected_interval: tuple[float, float],
) -> None:
    pvalue, standard_error, interval = monte_carlo_calibration(exceedances, 20)

    assert pvalue == (exceedances + 1) / 21
    assert standard_error == 0.0
    assert interval == pytest.approx(expected_interval)


@pytest.mark.parametrize(
    ("exceedances", "n_resamples", "error"),
    [
        (True, 10, TypeError),
        (1.5, 10, TypeError),
        (-1, 10, ValueError),
        (11, 10, ValueError),
        (1, False, TypeError),
        (1, 0, ValueError),
    ],
)
def test_resampling_counts_are_strict(
    exceedances: object,
    n_resamples: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        monte_carlo_calibration(exceedances, n_resamples)


@pytest.mark.parametrize("confidence_level", [True, np.bool_(False), "0.95"])
def test_monte_carlo_confidence_level_rejects_coercive_values(
    confidence_level: object,
) -> None:
    with pytest.raises(TypeError):
        monte_carlo_calibration(2, 10, confidence_level=confidence_level)  # type: ignore[arg-type]


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, math.nan, math.inf])
def test_monte_carlo_confidence_level_has_open_unit_interval(
    confidence_level: float,
) -> None:
    with pytest.raises(ValueError):
        monte_carlo_calibration(2, 10, confidence_level=confidence_level)

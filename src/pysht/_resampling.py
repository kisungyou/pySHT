"""Shared exact and Monte Carlo calibration utilities."""

from __future__ import annotations

import math
import operator
from typing import SupportsIndex, cast

import numpy as np
from scipy import stats

from ._validation import validate_confidence_level, validate_positive_integer

_MAX_EXACT_FLOAT_INTEGER = 2**53 - 1


def exact_pvalue(exceedances: object, n_resamples: object) -> float:
    """Return ``b / B`` after validating exact-enumeration counts."""
    total = validate_positive_integer(n_resamples, name="n_resamples")
    count = _validate_exceedances(exceedances, total)
    pvalue = count / total
    if count > 0 and pvalue == 0.0:
        raise ValueError(
            "the exact p-value is positive but cannot be represented in float64"
        )
    if count < total and pvalue == 1.0:
        raise ValueError(
            "the exact p-value is less than one but cannot be represented in float64"
        )
    return pvalue


def monte_carlo_calibration(
    exceedances: object,
    n_resamples: object,
    *,
    confidence_level: float = 0.95,
) -> tuple[float, float, tuple[float, float]]:
    """Return corrected p-value, conditional MCSE estimate, and exact interval.

    The p-value is the nonzero estimator ``(b + 1) / (B + 1)``.  The reported
    standard error estimates the conditional standard deviation of that
    estimator using ``b / B``.  The interval is the equal-tail
    Clopper--Pearson interval for the underlying exceedance probability.
    """
    total = validate_positive_integer(n_resamples, name="n_resamples")
    count = _validate_exceedances(exceedances, total)
    level = validate_confidence_level(confidence_level)

    tail_estimate = count / total
    pvalue = (count + 1) / (total + 1)
    if pvalue == 0.0:
        raise ValueError(
            "the corrected Monte Carlo p-value cannot be represented in float64"
        )
    if count < total and pvalue == 1.0:
        raise ValueError(
            "the corrected Monte Carlo p-value is less than one but cannot be "
            "represented in float64"
        )
    # SciPy's beta quantile kernel accepts binary64 shape parameters.  Beyond
    # this boundary adjacent integer counts collapse to the same shape, so an
    # interval advertised as count-consistent would no longer be auditable.
    if total > _MAX_EXACT_FLOAT_INTEGER:
        raise ValueError(
            "the binomial interval could not be represented for these counts"
        )
    # Algebraically this is sqrt(B*q_hat*(1-q_hat))/(B+1), but this
    # arrangement never converts a huge Python integer to an intermediate
    # float before reducing its scale.
    standard_error = math.sqrt(tail_estimate * (1.0 - tail_estimate) / total) * (
        total / (total + 1)
    )
    alpha = 1.0 - level
    try:
        lower = (
            0.0
            if count == 0
            else float(
                stats.beta.ppf(
                    alpha / 2.0,
                    float(count),
                    float(total - count + 1),
                )
            )
        )
        upper = (
            1.0
            if count == total
            else float(
                stats.beta.ppf(
                    1.0 - alpha / 2.0,
                    float(count + 1),
                    float(total - count),
                )
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "the binomial interval could not be represented for these counts"
        ) from exc
    if not all(math.isfinite(bound) for bound in (lower, upper)):
        raise ValueError(
            "the binomial interval could not be represented for these counts"
        )
    return pvalue, standard_error, (lower, upper)


def _validate_exceedances(value: object, total: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("exceedances must be an integer, not bool")
    try:
        count = operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise TypeError("exceedances must be an integer") from exc
    if not 0 <= count <= total:
        raise ValueError("exceedances must be between 0 and n_resamples")
    return count


__all__ = ["exact_pvalue", "monte_carlo_calibration"]

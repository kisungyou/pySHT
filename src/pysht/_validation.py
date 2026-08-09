"""Shared validation for public statistical procedures.

Validation is intentionally strict.  Statistical routines should fail before
calculation when an input cannot satisfy the assumptions needed to define the
reported statistic or null distribution.
"""

from __future__ import annotations

import math
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Alternative = Literal["two-sided", "less", "greater"]


def validate_alternative(value: str) -> Alternative:
    """Return a normalized one- or two-sided alternative."""
    if not isinstance(value, str):
        raise TypeError("alternative must be a string")
    normalized = value.strip().lower().replace("_", "-")
    if normalized not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    return cast(Alternative, normalized)


def validate_confidence_level(value: object) -> float:
    """Validate a confidence coefficient strictly between zero and one."""
    result = validate_real_scalar(value, name="confidence_level")
    if not 0.0 < result < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    return result


def validate_real_scalar(value: object, *, name: str) -> float:
    """Validate one finite real scalar, rejecting booleans and complex values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not bool")
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_bool(value: object, *, name: str) -> bool:
    """Validate a genuine Python or NumPy boolean."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _numeric_array(sample: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert an array-like object to finite real float64 values."""
    try:
        raw = np.asarray(sample)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric array") from exc
    if raw.dtype == np.dtype(bool) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric values")
    try:
        values = np.asarray(raw, dtype=np.float64, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def validate_1d_sample(
    sample: ArrayLike, *, name: str, minimum_size: int = 2
) -> NDArray[np.float64]:
    """Validate a one-dimensional sample with a minimum observation count."""
    values = _numeric_array(sample, name=name)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if values.size < minimum_size:
        raise ValueError(f"{name} must contain at least {minimum_size} observations")
    return values


def validate_2d_sample(
    sample: ArrayLike, *, name: str, minimum_rows: int = 2
) -> NDArray[np.float64]:
    """Validate a row-observation, column-feature sample matrix."""
    values = _numeric_array(sample, name=name)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if values.shape[0] < minimum_rows:
        raise ValueError(f"{name} must contain at least {minimum_rows} observations")
    if values.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    return values


def validate_groups(samples: tuple[ArrayLike, ...]) -> tuple[NDArray[np.float64], ...]:
    """Validate at least two independent univariate samples."""
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    return tuple(
        validate_1d_sample(sample, name=f"samples[{index}]")
        for index, sample in enumerate(samples)
    )


__all__ = [
    "Alternative",
    "validate_1d_sample",
    "validate_2d_sample",
    "validate_alternative",
    "validate_bool",
    "validate_confidence_level",
    "validate_groups",
    "validate_real_scalar",
]

"""Shared validation for public statistical procedures.

Validation is intentionally strict.  Statistical routines should fail before
calculation when an input cannot satisfy the assumptions needed to define the
reported statistic or null distribution.
"""

from __future__ import annotations

import math
import operator
from typing import Literal, SupportsIndex, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Alternative = Literal["two-sided", "less", "greater"]
type RngLike = int | np.integer | np.random.Generator | None


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
    if isinstance(value, (str, bytes, complex, np.complexfloating)):
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


def validate_positive_integer(value: object, *, name: str) -> int:
    """Validate a strictly positive integer, rejecting Boolean values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return result


def validate_choice[Choice: str](
    value: object, *, name: str, choices: tuple[Choice, ...]
) -> Choice:
    """Validate a lowercase string option against an explicit finite set."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower().replace("_", "-")
    for choice in choices:
        if normalized == choice:
            return choice
    allowed = ", ".join(repr(choice) for choice in choices)
    raise ValueError(f"{name} must be one of {allowed}")


def make_generator(rng: RngLike) -> np.random.Generator:
    """Return a local NumPy generator without touching global RNG state."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is not None and (
        isinstance(rng, (bool, np.bool_)) or not isinstance(rng, (int, np.integer))
    ):
        raise TypeError("rng must be None, an integer seed, or a NumPy Generator")
    try:
        return np.random.default_rng(rng)
    except ValueError as exc:
        raise ValueError("rng integer seed is outside the supported range") from exc


def _numeric_array(sample: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert an array-like object to finite real float64 values."""
    try:
        raw = np.asarray(sample)
    except (TypeError, ValueError, OverflowError) as exc:
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


def validate_multivariate_groups(
    samples: tuple[ArrayLike, ...],
    *,
    minimum_groups: int = 2,
    minimum_rows: int = 2,
) -> tuple[NDArray[np.float64], ...]:
    """Validate multivariate samples with a shared feature dimension."""
    if len(samples) < minimum_groups:
        raise ValueError(f"at least {minimum_groups} samples are required")
    groups = tuple(
        validate_2d_sample(sample, name=f"samples[{index}]", minimum_rows=minimum_rows)
        for index, sample in enumerate(samples)
    )
    features = groups[0].shape[1]
    if any(group.shape[1] != features for group in groups[1:]):
        raise ValueError("all samples must have the same number of features")
    return groups


def validate_square_matrix(
    matrix: ArrayLike,
    *,
    name: str,
    size: int | None = None,
    symmetric: bool = True,
) -> NDArray[np.float64]:
    """Validate a finite real square matrix with optional symmetry and size."""
    values = _numeric_array(matrix, name=name)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row and column")
    if size is not None and values.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    if symmetric:
        with np.errstate(over="ignore", invalid="ignore"):
            asymmetry = np.abs(values - values.T)
            tolerance = 1.0e-12 * np.maximum(np.abs(values), np.abs(values.T))
        if np.any(~np.isfinite(asymmetry)) or np.any(asymmetry > tolerance):
            raise ValueError(f"{name} must be symmetric")
        # Downstream factorizations must not depend on which triangle an
        # accepted, roundoff-level asymmetry happened to occupy.
        upper = np.triu_indices(values.shape[0], k=1)
        with np.errstate(over="ignore", invalid="ignore"):
            midpoints = values[upper] + 0.5 * (
                values[(upper[1], upper[0])] - values[upper]
            )
        if not np.all(np.isfinite(midpoints)):
            raise ValueError(f"{name} must be symmetric")
        if upper[0].size:
            values = values.copy()
            values[upper] = midpoints
            values[(upper[1], upper[0])] = midpoints
    return values


def validate_covariance_matrix(
    matrix: ArrayLike,
    *,
    name: str,
    size: int | None = None,
    positive_definite: bool = True,
) -> NDArray[np.float64]:
    """Validate a symmetric covariance matrix and its definiteness."""
    values = validate_square_matrix(matrix, name=name, size=size, symmetric=True)
    diagonal = np.diag(values)
    if positive_definite:
        if np.any(diagonal <= 0.0):
            raise ValueError(f"{name} must be positive definite")
        active = np.ones(values.shape[0], dtype=bool)
    else:
        if np.any(diagonal < 0.0):
            raise ValueError(f"{name} must be positive semidefinite")
        active = diagonal > 0.0
        inactive = ~active
        if np.any(values[inactive, :] != 0.0) or np.any(values[:, inactive] != 0.0):
            raise ValueError(f"{name} must be positive semidefinite")
        if not np.any(active):
            return values

    # Diagonal equilibration is a congruence transformation and therefore
    # preserves definiteness.  Unlike division by one global maximum, it does
    # not erase a valid small-variance coordinate when marginal units span the
    # full float64 exponent range.
    principal = values[np.ix_(active, active)]
    scales = np.sqrt(np.diag(principal))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        correlation = principal / scales[:, None]
        correlation = correlation / scales[None, :]
    if not np.all(np.isfinite(correlation)):
        raise ValueError(f"{name} eigenvalues could not be evaluated")
    upper = np.triu_indices(correlation.shape[0], k=1)
    correlation_midpoints = correlation[upper] + 0.5 * (
        correlation[(upper[1], upper[0])] - correlation[upper]
    )
    correlation[upper] = correlation_midpoints
    correlation[(upper[1], upper[0])] = correlation_midpoints
    np.fill_diagonal(correlation, 1.0)
    try:
        eigenvalues = np.linalg.eigvalsh(correlation)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} eigenvalues could not be evaluated") from exc
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{name} eigenvalues could not be evaluated")
    if positive_definite:
        if np.any(eigenvalues <= 0.0):
            raise ValueError(f"{name} must be positive definite")
    elif np.any(eigenvalues < -100.0 * np.finfo(np.float64).eps):
        raise ValueError(f"{name} must be positive semidefinite")
    return values


def validate_bounds(
    lower: ArrayLike | None,
    upper: ArrayLike | None,
    *,
    size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate per-feature lower and upper bounds."""
    low = (
        np.zeros(size, dtype=np.float64)
        if lower is None
        else validate_1d_sample(lower, name="lower", minimum_size=1)
    )
    high = (
        np.ones(size, dtype=np.float64)
        if upper is None
        else validate_1d_sample(upper, name="upper", minimum_size=1)
    )
    if low.size != size or high.size != size:
        raise ValueError("lower and upper must contain one value per feature")
    if np.any(low >= high):
        raise ValueError("every lower bound must be strictly less than its upper bound")
    return low, high


def validate_simplex_sample(
    sample: ArrayLike,
    *,
    name: str = "x",
    interior: bool = True,
) -> NDArray[np.float64]:
    """Validate rows as compositional observations on a probability simplex."""
    values = validate_2d_sample(sample, name=name)
    if values.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two components")
    if interior:
        if np.any(values <= 0.0):
            raise ValueError(f"{name} must lie strictly inside the probability simplex")
    elif np.any(values < 0.0):
        raise ValueError(f"{name} must contain non-negative components")
    # Component labels are scientifically arbitrary.  A fixed magnitude order
    # keeps acceptance at the tolerance boundary invariant to a common
    # component permutation instead of exposing NumPy's reduction order.
    row_sums = np.sum(np.sort(values, axis=1), axis=1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"each row of {name} must sum to 1")
    return values


__all__ = [
    "Alternative",
    "RngLike",
    "make_generator",
    "validate_1d_sample",
    "validate_2d_sample",
    "validate_alternative",
    "validate_bool",
    "validate_bounds",
    "validate_choice",
    "validate_confidence_level",
    "validate_covariance_matrix",
    "validate_groups",
    "validate_multivariate_groups",
    "validate_positive_integer",
    "validate_real_scalar",
    "validate_simplex_sample",
    "validate_square_matrix",
]

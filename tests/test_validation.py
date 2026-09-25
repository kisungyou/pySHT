from __future__ import annotations

import numpy as np
import pytest

from pysht._validation import (
    make_generator,
    validate_2d_sample,
    validate_bool,
    validate_bounds,
    validate_choice,
    validate_covariance_matrix,
    validate_groups,
    validate_multivariate_groups,
    validate_positive_integer,
    validate_real_scalar,
    validate_simplex_sample,
    validate_square_matrix,
)


def test_positive_integer_rejects_boolean_and_coercion() -> None:
    assert validate_positive_integer(np.int64(7), name="count") == 7
    for value, error in ((True, TypeError), (2.0, TypeError), (0, ValueError)):
        with pytest.raises(error):
            validate_positive_integer(value, name="count")


def test_real_scalar_is_finite_and_noncoercive() -> None:
    assert validate_real_scalar(np.float64(1.25), name="value") == 1.25
    for value, error in (
        (True, TypeError),
        ("1.25", TypeError),
        (1 + 0j, TypeError),
        (np.inf, ValueError),
    ):
        with pytest.raises(error):
            validate_real_scalar(value, name="value")

    class InvalidFloat:
        def __float__(self) -> float:
            raise ValueError("not convertible")

    with pytest.raises(TypeError, match="real number"):
        validate_real_scalar(InvalidFloat(), name="value")


def test_boolean_and_choice_controls_reject_unrelated_types() -> None:
    assert validate_bool(np.bool_(True), name="flag") is True
    with pytest.raises(TypeError, match="boolean"):
        validate_bool(1, name="flag")
    with pytest.raises(TypeError, match="string"):
        validate_choice(1, name="mode", choices=("exact",))


def test_choice_returns_canonical_lowercase_hyphenated_value() -> None:
    assert (
        validate_choice(
            "Monte_Carlo",
            name="calibration",
            choices=("asymptotic", "monte-carlo"),
        )
        == "monte-carlo"
    )
    with pytest.raises(ValueError, match="one of"):
        validate_choice("permutation", name="calibration", choices=("exact",))


def test_generator_is_local_replayable_and_accepts_existing_generator() -> None:
    first = make_generator(919).normal(size=5)
    second = make_generator(np.int64(919)).normal(size=5)
    np.testing.assert_array_equal(first, second)

    supplied = np.random.default_rng(12)
    assert make_generator(supplied) is supplied
    for value in (True, 1.5, "seed"):
        with pytest.raises(TypeError):
            make_generator(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outside the supported range"):
        make_generator(-1)


def test_multivariate_groups_require_shared_feature_count() -> None:
    groups = validate_multivariate_groups(
        (np.ones((3, 2)), np.zeros((4, 2))), minimum_rows=3
    )
    assert [group.shape for group in groups] == [(3, 2), (4, 2)]
    with pytest.raises(ValueError, match="same number of features"):
        validate_multivariate_groups((np.ones((3, 2)), np.ones((3, 3))))
    with pytest.raises(ValueError, match="at least 3 samples"):
        validate_multivariate_groups(
            (np.ones((3, 2)), np.ones((3, 2))), minimum_groups=3
        )


def test_sample_and_group_shape_boundaries_are_explicit() -> None:
    with pytest.raises(ValueError, match="at least one feature"):
        validate_2d_sample(np.empty((2, 0)), name="x")
    with pytest.raises(ValueError, match="at least two samples"):
        validate_groups((np.ones(3),))


def test_array_coercion_failures_have_a_stable_public_error() -> None:
    class OverflowingArray:
        def __array__(self) -> np.ndarray:
            raise OverflowError("forced conversion overflow")

    with pytest.raises(TypeError, match="x must be a real numeric array"):
        validate_2d_sample(OverflowingArray(), name="x")  # type: ignore[arg-type]


def test_square_and_covariance_matrix_validation() -> None:
    covariance = np.array([[2.0, 0.5], [0.5, 1.0]])
    np.testing.assert_array_equal(
        validate_covariance_matrix(covariance, name="popcov", size=2), covariance
    )
    for scale in (1.0e-300, 1.0e300):
        scaled = covariance * scale
        np.testing.assert_array_equal(
            validate_covariance_matrix(scaled, name="popcov", size=2), scaled
        )
    heterogeneous = np.array([[1.0e-300, 0.5], [0.5, 1.0e300]])
    np.testing.assert_array_equal(
        validate_covariance_matrix(heterogeneous, name="popcov"), heterogeneous
    )
    permutation = np.array([1, 0])
    permuted_heterogeneous = heterogeneous[np.ix_(permutation, permutation)]
    np.testing.assert_array_equal(
        validate_covariance_matrix(permuted_heterogeneous, name="popcov"),
        permuted_heterogeneous,
    )
    np.testing.assert_array_equal(
        validate_covariance_matrix(
            heterogeneous,
            name="popcov",
            positive_definite=False,
        ),
        heterogeneous,
    )
    with pytest.raises(ValueError, match="symmetric"):
        validate_square_matrix([[1.0, 2.0], [0.0, 1.0]], name="matrix")
    with pytest.raises(ValueError, match="shape"):
        validate_square_matrix(np.eye(2), name="matrix", size=3)
    with pytest.raises(ValueError, match="at least one row and column"):
        validate_covariance_matrix(np.empty((0, 0)), name="popcov")
    np.testing.assert_array_equal(
        validate_square_matrix(
            [[1.0, 2.0], [0.0, 1.0]], name="matrix", symmetric=False
        ),
        [[1.0, 2.0], [0.0, 1.0]],
    )
    with pytest.raises(ValueError, match="symmetric"):
        validate_square_matrix(
            [[1.0e-300, 1.0e-300], [0.0, 1.0e-300]],
            name="matrix",
        )
    with pytest.raises(ValueError, match="symmetric"):
        validate_square_matrix(
            [[1.0e300, 1.0], [0.0, 1.0]],
            name="matrix",
        )
    with pytest.raises(ValueError, match="positive definite"):
        validate_covariance_matrix([[1.0, 1.0], [1.0, 1.0]], name="popcov")
    validate_covariance_matrix(
        [[1.0, 1.0], [1.0, 1.0]],
        name="popcov",
        positive_definite=False,
    )
    np.testing.assert_array_equal(
        validate_covariance_matrix(
            np.zeros((2, 2)), name="popcov", positive_definite=False
        ),
        np.zeros((2, 2)),
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_covariance_matrix(
            [[1.0, 2.0], [2.0, 1.0]],
            name="popcov",
            positive_definite=False,
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_covariance_matrix(
            [[0.0, np.nextafter(0.0, 1.0)], [np.nextafter(0.0, 1.0), 1.0]],
            name="popcov",
            positive_definite=False,
        )


def test_covariance_eigensolver_failure_has_public_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    with pytest.raises(ValueError, match="eigenvalues could not be evaluated"):
        validate_covariance_matrix(np.eye(2), name="popcov")


def test_nonfinite_covariance_eigenvalues_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda _: np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="eigenvalues could not be evaluated"):
        validate_covariance_matrix(np.eye(2), name="popcov")


def test_bounds_default_and_feature_validation() -> None:
    lower, upper = validate_bounds(None, None, size=2)
    np.testing.assert_array_equal(lower, [0.0, 0.0])
    np.testing.assert_array_equal(upper, [1.0, 1.0])
    with pytest.raises(ValueError, match="one value per feature"):
        validate_bounds([0.0], [1.0], size=2)
    with pytest.raises(ValueError, match="strictly less"):
        validate_bounds([0.0, 1.0], [1.0, 1.0], size=2)


def test_simplex_validation_requires_interior_and_unit_row_sums() -> None:
    values = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    np.testing.assert_array_equal(validate_simplex_sample(values), values)
    with pytest.raises(ValueError, match="strictly inside"):
        validate_simplex_sample([[0.0, 0.5, 0.5], [0.2, 0.3, 0.5]])
    with pytest.raises(ValueError, match="sum to 1"):
        validate_simplex_sample([[0.2, 0.3, 0.6], [0.1, 0.6, 0.3]])


def test_simplex_sum_tolerance_is_component_order_invariant() -> None:
    first = np.array([0.7, 0.2, 0.1, 5.0e-13, 5.0e-13])
    permuted = np.array([0.7, 0.2, 5.0e-13, 0.1, 5.0e-13])

    # Ordinary reductions land on opposite sides of the 1e-12 acceptance
    # boundary.  A component label permutation must not change the domain.
    assert abs(float(np.sum(first)) - 1.0) <= 1.0e-12
    assert abs(float(np.sum(permuted)) - 1.0) > 1.0e-12
    for row in (first, permuted):
        with pytest.raises(ValueError, match="sum to 1"):
            validate_simplex_sample(np.vstack((row, row)))

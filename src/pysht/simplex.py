"""Goodness-of-fit tests for observations on a probability simplex."""

from __future__ import annotations

import math
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special, stats

from ._results import HypothesisTestResult
from ._validation import (
    validate_choice,
    validate_positive_integer,
    validate_real_scalar,
    validate_simplex_sample,
)

__all__ = ["uniformity"]


type _DirichletAlternative = Literal["symmetric", "general"]

_ALTERNATIVE: Final = "the distribution is not uniform on the simplex"
_ARMIJO_CONSTANT: Final = 1.0e-4
_MINIMUM_LINE_SEARCH_STEP: Final = 2.0**-50


def _validate_model(value: str) -> _DirichletAlternative:
    return validate_choice(
        value,
        name="model",
        choices=("symmetric", "general"),
    )


def _validate_simplex(x: ArrayLike) -> NDArray[np.float64]:
    """Validate rows in the strict interior of a probability simplex."""
    values = validate_simplex_sample(x, name="x", interior=True)
    if np.any(values >= 1.0):
        raise ValueError("x must lie strictly inside the probability simplex")
    dimension = values.shape[1]
    with np.errstate(over="ignore", invalid="ignore"):
        row_sums = np.sum(values, axis=1, dtype=np.float64)
    tolerance = 64.0 * np.finfo(np.float64).eps * dimension
    if np.any(~np.isfinite(row_sums)) or np.any(np.abs(row_sums - 1.0) > tolerance):
        raise ValueError("every row of x must sum to 1 within floating-point tolerance")

    return values / row_sums[:, None]


def _dirichlet_log_likelihood_per_observation(
    alpha: NDArray[np.float64], mean_log: NDArray[np.float64]
) -> float:
    return float(
        special.gammaln(float(np.sum(alpha)))
        - np.sum(special.gammaln(alpha))
        + np.dot(alpha - 1.0, mean_log)
    )


def _symmetric_mle(
    mean_log_all: float,
    *,
    dimension: int,
    tolerance: float,
    max_iter: int,
) -> float:
    """Find the unique finite symmetric Dirichlet MLE by bracketing its score."""

    def score(alpha: float) -> float:
        return float(
            special.digamma(dimension * alpha) - special.digamma(alpha) + mean_log_all
        )

    # AM--GM implies mean(log(X_j)) + log(k) <= 0. Equality corresponds to
    # observations fixed at the barycenter and an infinite concentration MLE.
    limiting_score = math.log(dimension) + mean_log_all
    if limiting_score >= -64.0 * np.finfo(np.float64).eps:
        raise ValueError("the symmetric Dirichlet concentration MLE is unbounded")

    lower = 1.0e-12
    lower_score = score(lower)
    if not math.isfinite(lower_score) or lower_score <= 0.0:
        raise RuntimeError("failed to establish the lower Dirichlet MLE bracket")
    upper = 1.0
    upper_score = score(upper)
    bracket_iterations = 0
    while upper_score > 0.0 and bracket_iterations < max_iter:
        upper *= 2.0
        if not math.isfinite(upper):
            break
        upper_score = score(upper)
        bracket_iterations += 1
    if not math.isfinite(upper) or upper_score > 0.0:
        raise RuntimeError("failed to bracket the symmetric Dirichlet MLE")

    solution = optimize.root_scalar(
        score,
        bracket=(lower, upper),
        method="brentq",
        xtol=tolerance,
        rtol=max(tolerance, 4.0 * np.finfo(np.float64).eps),
        maxiter=max_iter,
    )
    if not solution.converged or solution.root <= 0.0:
        raise RuntimeError("symmetric Dirichlet maximum likelihood did not converge")
    return float(solution.root)


def _initial_general_alpha(values: NDArray[np.float64]) -> NDArray[np.float64]:
    means = np.mean(values, axis=0, dtype=np.float64)
    variances = np.var(values, axis=0, ddof=1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        candidates = means * (1.0 - means) / variances - 1.0
    usable = candidates[np.isfinite(candidates) & (candidates > 0.0)]
    concentration = float(np.median(usable)) if usable.size else float(values.shape[1])
    concentration = min(1.0e6, max(1.0e-2, concentration))
    return np.maximum(means * concentration, 1.0e-6)


def _general_mle(
    values: NDArray[np.float64],
    mean_log: NDArray[np.float64],
    *,
    tolerance: float,
    max_iter: int,
) -> NDArray[np.float64]:
    """Fit a general Dirichlet distribution with verified score equations."""
    alpha = _initial_general_alpha(values)
    current_log_likelihood = _dirichlet_log_likelihood_per_observation(alpha, mean_log)
    for _ in range(max_iter):
        alpha_sum = float(np.sum(alpha))
        gradient = special.digamma(alpha_sum) - special.digamma(alpha) + mean_log
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= tolerance:
            return alpha

        diagonal = special.polygamma(1, alpha)
        shared = float(special.polygamma(1, alpha_sum))
        hessian = np.full((alpha.size, alpha.size), shared, dtype=np.float64)
        hessian[np.diag_indices_from(hessian)] -= diagonal
        try:
            direction = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("the Dirichlet likelihood Hessian is singular") from exc
        directional_derivative = float(np.dot(gradient, direction))
        if not math.isfinite(directional_derivative) or directional_derivative <= 0.0:
            break

        step = 1.0
        negative = direction < 0.0
        if np.any(negative):
            step = min(
                step,
                0.99 * float(np.min(-alpha[negative] / direction[negative])),
            )
        accepted = False
        while step >= _MINIMUM_LINE_SEARCH_STEP:
            candidate = alpha + step * direction
            candidate_log_likelihood = _dirichlet_log_likelihood_per_observation(
                candidate, mean_log
            )
            if (
                math.isfinite(candidate_log_likelihood)
                and candidate_log_likelihood
                >= current_log_likelihood
                + _ARMIJO_CONSTANT * step * directional_derivative
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

        relative_step = float(np.max(np.abs(step * direction) / np.maximum(1.0, alpha)))
        alpha = candidate
        current_log_likelihood = candidate_log_likelihood
        if relative_step <= tolerance:
            final_gradient = (
                special.digamma(float(np.sum(alpha)))
                - special.digamma(alpha)
                + mean_log
            )
            if float(np.max(np.abs(final_gradient))) <= tolerance:
                return alpha

    # Very asymmetric compositions can make the alpha-coordinate Hessian
    # poorly conditioned even though the unique finite score root is regular
    # in log(alpha).  Use an independently parameterized trust-region solve as
    # a fallback, and accept it only after checking the original score.
    lower_log_alpha = -700.0
    upper_log_alpha = 700.0

    def log_score(log_alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        candidate = np.exp(log_alpha)
        return np.asarray(
            special.digamma(float(np.sum(candidate)))
            - special.digamma(candidate)
            + mean_log,
            dtype=np.float64,
        )

    def log_score_jacobian(log_alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        candidate = np.exp(log_alpha)
        shared = float(special.polygamma(1, float(np.sum(candidate))))
        jacobian = np.broadcast_to(
            shared * candidate,
            (candidate.size, candidate.size),
        ).copy()
        diagonal = np.diag_indices_from(jacobian)
        jacobian[diagonal] -= special.polygamma(1, candidate) * candidate
        return jacobian

    solver_tolerance = max(tolerance, 32.0 * np.finfo(np.float64).eps)
    solution = optimize.least_squares(
        log_score,
        np.clip(np.log(alpha), lower_log_alpha, upper_log_alpha),
        jac=log_score_jacobian,
        bounds=(lower_log_alpha, upper_log_alpha),
        ftol=solver_tolerance,
        xtol=solver_tolerance,
        gtol=solver_tolerance,
        max_nfev=max_iter,
    )
    fitted = np.asarray(np.exp(solution.x), dtype=np.float64)
    score_error = float(np.max(np.abs(log_score(solution.x))))
    roundoff_floor = (
        512.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(mean_log))))
    )
    if (
        not solution.success
        or np.any(solution.active_mask != 0)
        or score_error > max(tolerance, roundoff_floor)
    ):
        raise RuntimeError(
            "general Dirichlet maximum likelihood did not converge "
            f"in {max_iter} iterations"
        )
    return fitted


def uniformity(
    x: ArrayLike,
    *,
    model: str = "symmetric",
    tolerance: float = 1.0e-10,
    max_iter: int = 200,
) -> HypothesisTestResult:
    """Test uniformity on a probability simplex with a Dirichlet LRT.

    Parameters
    ----------
    x
        An ``(n, k)`` matrix whose rows lie in the strict interior of the
        probability simplex. At least two rows and two components are needed.
        Rows whose sums differ from one by no more than a documented
        floating-point tolerance are normalized before fitting.
    model
        ``"symmetric"`` fits one common Dirichlet concentration parameter;
        ``"general"`` fits one positive parameter per component.
    tolerance
        Positive convergence tolerance for the score equations.
    max_iter
        Positive maximum number of bracketing/root or Newton iterations.

    Returns
    -------
    HypothesisTestResult
        The likelihood-ratio statistic calibrated by Wilks' chi-square
        approximation with one or ``k`` degrees of freedom.

    Notes
    -----
    The null is Dirichlet ``alpha=(1, ..., 1)``, the uniform distribution with
    respect to simplex volume. Boundary components are rejected instead of
    being silently perturbed. The general-model fit rejects identical rows,
    for which its concentration MLE is unbounded. The symmetric fit is
    unbounded only when every row is the simplex barycenter.
    """
    values = _validate_simplex(x)
    selected_model = _validate_model(model)
    convergence_tolerance = validate_real_scalar(tolerance, name="tolerance")
    if convergence_tolerance <= 0.0:
        raise ValueError("tolerance must be greater than 0")
    iterations = validate_positive_integer(max_iter, name="max_iter")

    sample_size, dimension = values.shape
    log_values = np.log(values)
    mean_log = np.mean(log_values, axis=0, dtype=np.float64)
    null_alpha = np.ones(dimension, dtype=np.float64)
    null_log_likelihood = _dirichlet_log_likelihood_per_observation(
        null_alpha, mean_log
    )

    if selected_model == "symmetric":
        fitted_scalar = _symmetric_mle(
            float(np.mean(log_values, dtype=np.float64)),
            dimension=dimension,
            tolerance=convergence_tolerance,
            max_iter=iterations,
        )
        fitted_alpha = np.full(dimension, fitted_scalar, dtype=np.float64)
        degrees_of_freedom = 1.0
        method_name = "Simplex uniformity LRT against a symmetric Dirichlet model"
    else:
        if np.all(values == values[0]):
            raise ValueError(
                "the general Dirichlet maximum-likelihood estimate is unbounded "
                "for identical rows"
            )
        fitted_alpha = _general_mle(
            values,
            mean_log,
            tolerance=convergence_tolerance,
            max_iter=iterations,
        )
        degrees_of_freedom = float(dimension)
        method_name = "Simplex uniformity LRT against a general Dirichlet model"

    fitted_log_likelihood = _dirichlet_log_likelihood_per_observation(
        fitted_alpha, mean_log
    )
    statistic = 2.0 * sample_size * (fitted_log_likelihood - null_log_likelihood)
    roundoff_tolerance = (
        1.0e-10
        * sample_size
        * max(1.0, abs(null_log_likelihood), abs(fitted_log_likelihood))
    )
    if statistic < -roundoff_tolerance:
        raise RuntimeError(
            "the fitted Dirichlet likelihood is below the null likelihood"
        )
    statistic = max(0.0, statistic)

    return HypothesisTestResult(
        statistic=statistic,
        pvalue=float(stats.chi2.sf(statistic, degrees_of_freedom)),
        method=method_name,
        alternative=_ALTERNATIVE,
        data_name="x",
        statistic_name="LR",
        calibration="Wilks asymptotic chi-square approximation",
        df=degrees_of_freedom,
    )

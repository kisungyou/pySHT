"""Goodness-of-fit tests for observations on a probability simplex."""

from __future__ import annotations

import math
from collections.abc import Iterator
from decimal import Decimal, localcontext
from itertools import combinations
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, special, stats

from ._resampling import exact_pvalue, monte_carlo_calibration
from ._results import HypothesisTestResult, ResamplingTestResult
from ._validation import (
    make_generator,
    validate_choice,
    validate_positive_integer,
    validate_real_scalar,
    validate_simplex_sample,
)
from .uniformity import _scaled_log_power_sum

__all__ = ["alpha_energy_ksamp", "ehy_uniformity", "uniformity"]


type _DirichletAlternative = Literal["symmetric", "general"]

_ALTERNATIVE: Final = "the distribution is not uniform on the simplex"
_ARMIJO_CONSTANT: Final = 1.0e-4
_MINIMUM_LINE_SEARCH_STEP: Final = 2.0**-50
_PERMUTATION_TIE_RTOL: Final = 100.0 * np.finfo(np.float64).eps
_EHY_LOG_TIE_RTOL: Final = 100.0 * np.finfo(np.float64).eps
_INVARIANT_ORBIT_MONTE_CARLO_LIMIT: Final = 100_000
_INVARIANT_ORBIT_WORK_MULTIPLIER: Final = 10


def _stable_row_sums(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum nonnegative compositional rows independently of component order."""
    return np.sum(np.sort(values, axis=1), axis=1, dtype=np.float64)


def _stable_logsumexp_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate row log-sum-exp with component-order-stable reduction."""
    maxima = np.max(values, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        shifted = values - maxima
    exponentials = np.exp(shifted)
    totals = np.sum(
        np.sort(exponentials, axis=1), axis=1, keepdims=True, dtype=np.float64
    )
    return maxima + np.log(totals)


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
        row_sums = _stable_row_sums(values)
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


def _concentrated_dirichlet_lrt(
    values: NDArray[np.float64], *, symmetric: bool, tolerance: float, max_iter: int
) -> float:
    """Resolve concentrated fits beyond binary64 score/likelihood precision.

    Absolute alpha-coordinate scores vanish as concentration increases, even
    far from the optimum. Here Newton steps must be small relative to every
    parameter. Decimal arithmetic retains both the Jensen gap of nearly
    identical compositions and cancellations in the log-gamma normalizer.
    """
    with localcontext() as context:
        context.prec = 80
        zero, one, two = Decimal(0), Decimal(1), Decimal(2)
        count, dimension = values.shape
        n, d = Decimal(count), Decimal(dimension)
        rows = [[Decimal.from_float(float(value)) for value in row] for row in values]
        rows = [[value / sum(row, zero) for value in row] for row in rows]
        means = [sum((row[j] for row in rows), zero) / n for j in range(dimension)]
        logs = [sum((row[j].ln() for row in rows), zero) / n for j in range(dimension)]
        # Bernoulli B_2 through B_24; at x>=128 the next omitted term in
        # digamma is below 2e-49. Large concentration arguments need no shift.
        bernoulli = [
            Decimal(a) / Decimal(b)
            for a, b in (
                (1, 6),
                (-1, 30),
                (1, 42),
                (-1, 30),
                (5, 66),
                (-691, 2730),
                (7, 6),
                (-3617, 510),
                (43867, 798),
                (-174611, 330),
                (854513, 138),
                (-236364091, 2730),
            )
        ]
        log_two_pi = Decimal(
            "6.283185307179586476925286766559005768394338798750211641949889"
        ).ln()

        def special_values(a: Decimal) -> tuple[Decimal, Decimal, Decimal]:
            shifted = a
            psi_shift = zero
            trigamma_shift = zero
            gamma_shift = zero
            while shifted < 128:
                psi_shift -= one / shifted
                trigamma_shift += one / shifted**2
                gamma_shift -= shifted.ln()
                shifted += one
            inverse = one / shifted
            psi = shifted.ln() - inverse / two
            trigamma = inverse + inverse**2 / two
            gamma = (shifted - one / two) * shifted.ln() - shifted + log_two_pi / two
            for k, b in enumerate(bernoulli, start=1):
                psi -= b * inverse ** (2 * k) / Decimal(2 * k)
                trigamma += b * inverse ** (2 * k + 1)
                gamma += b * inverse ** (2 * k - 1) / Decimal(2 * k * (2 * k - 1))
            return psi + psi_shift, trigamma + trigamma_shift, gamma + gamma_shift

        if symmetric:
            gap = -d.ln() - sum(logs, zero) / d
            if gap <= 0:
                raise ValueError(
                    "the symmetric Dirichlet concentration MLE is unbounded"
                )
            initial = (d - one) / (two * d * gap)
            alpha = [initial] * dimension
        else:
            gap = sum(
                (
                    mean * (mean.ln() - log)
                    for mean, log in zip(means, logs, strict=True)
                ),
                zero,
            )
            if gap <= 0:
                raise ValueError("the general Dirichlet concentration MLE is unbounded")
            concentration = (d - one) / (two * gap)
            alpha = [mean * concentration for mean in means]

        def likelihood(parameters: list[Decimal]) -> Decimal:
            return special_values(sum(parameters, zero))[2] - sum(
                (
                    special_values(a)[2] - (a - one) * log
                    for a, log in zip(parameters, logs, strict=True)
                ),
                zero,
            )

        current = likelihood(alpha)
        requested = Decimal.from_float(tolerance)
        for _ in range(max_iter):
            psi_sum, trigamma_sum, _ = special_values(sum(alpha, zero))
            marginal = [special_values(a) for a in alpha]
            gradient = [
                psi_sum - item[0] + log
                for item, log in zip(marginal, logs, strict=True)
            ]
            if symmetric:
                step = -(sum(gradient, zero) / d) / (d * trigamma_sum - marginal[0][1])
                direction = [step] * dimension
            else:
                inverse_diagonal = [one / item[1] for item in marginal]
                shared = sum(
                    (
                        g * inverse
                        for g, inverse in zip(gradient, inverse_diagonal, strict=True)
                    ),
                    zero,
                ) / (one / trigamma_sum - sum(inverse_diagonal, zero))
                direction = [
                    (g + shared) * inverse
                    for g, inverse in zip(gradient, inverse_diagonal, strict=True)
                ]
            relative_step = max(
                abs(step / a) for step, a in zip(direction, alpha, strict=True)
            )
            if relative_step <= requested:
                null = special_values(d)[2]
                return float(two * n * (current - null))
            fraction = one
            accepted = False
            for _ in range(80):
                candidate = [
                    a + fraction * step
                    for a, step in zip(alpha, direction, strict=True)
                ]
                if min(candidate) > 0:
                    candidate_likelihood = likelihood(candidate)
                    if candidate_likelihood >= current:
                        alpha, current = candidate, candidate_likelihood
                        accepted = True
                        break
                fraction /= two
            if not accepted:
                break
        raise RuntimeError("concentrated Dirichlet maximum likelihood did not converge")


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

        diagonal = special.polygamma(1, alpha)
        shared = float(special.polygamma(1, alpha_sum))
        hessian = np.full((alpha.size, alpha.size), shared, dtype=np.float64)
        hessian[np.diag_indices_from(hessian)] -= diagonal
        try:
            direction = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("the Dirichlet likelihood Hessian is singular") from exc
        # An absolute score alone can be tiny far from a concentrated MLE.
        # Verify that the full Newton correction is relatively small as well.
        if (
            gradient_norm <= tolerance
            and float(np.max(np.abs(direction) / alpha)) <= tolerance
        ):
            return alpha
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
    if selected_model == "symmetric":
        if np.all(values == values[0, 0]):
            raise ValueError("the symmetric Dirichlet concentration MLE is unbounded")
        concentrated = abs(math.log(dimension) + float(np.mean(mean_log))) < 1e-5
    else:
        if np.all(values == values[0]):
            raise ValueError(
                "the general Dirichlet concentration MLE is unbounded for identical rows"
            )
        concentrated = float(np.sum(_initial_general_alpha(values))) >= 1e5
    if concentrated:
        statistic = _concentrated_dirichlet_lrt(
            values,
            symmetric=selected_model == "symmetric",
            tolerance=convergence_tolerance,
            max_iter=iterations,
        )
        degrees_of_freedom = 1.0 if selected_model == "symmetric" else float(dimension)
        return HypothesisTestResult(
            statistic=statistic,
            pvalue=float(stats.chi2.sf(statistic, degrees_of_freedom)),
            method=f"Simplex uniformity LRT against a {selected_model} Dirichlet model",
            alternative=_ALTERNATIVE,
            data_name="x",
            statistic_name="LR",
            calibration="Wilks asymptotic chi-square approximation",
            df=degrees_of_freedom,
        )
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


def _closed_simplex(x: ArrayLike, *, name: str) -> NDArray[np.float64]:
    values = validate_simplex_sample(x, name=name, interior=False)
    row_sums = _stable_row_sums(values)
    return np.asarray(values / row_sums[:, None], dtype=np.float64, order="C")


def _ehy_simplex_log_statistic(
    values: NDArray[np.float64], *, alpha: float, n_neighbors: int
) -> float:
    sample_size, components = values.shape
    manifold_dimension = components - 1
    distances = _component_invariant_distances(values)
    np.fill_diagonal(distances, math.inf)
    nearest = np.partition(distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_radius = np.log(nearest)
    log_unit_ball_volume = 0.5 * manifold_dimension * math.log(math.pi) - float(
        special.gammaln(0.5 * manifold_dimension + 1.0)
    )
    # The standard simplex has Hausdorff volume sqrt(D)/(D-1)!, so its
    # probability density relative to that volume is (D-1)!/sqrt(D).
    log_uniform_density = float(special.gammaln(components)) - 0.5 * math.log(
        components
    )
    log_volumes = (
        log_unit_ball_volume
        + math.log(sample_size)
        + manifold_dimension * log_radius
        + log_uniform_density
    )
    return _scaled_log_power_sum(log_volumes, alpha)


def _nonnegative_from_log(log_value: float) -> float:
    if log_value == -math.inf:
        return 0.0
    if log_value > math.log(np.finfo(np.float64).max):
        return math.inf
    return math.exp(log_value)


def _ehy_log_tail_contains(
    candidate: float,
    observed: float,
    *,
    lower_tail: bool,
) -> bool:
    """Include last-bit numerical ties in the selected EHY log tail."""
    direct = candidate <= observed if lower_tail else candidate >= observed
    if direct or not (math.isfinite(candidate) and math.isfinite(observed)):
        return direct
    tolerance = _EHY_LOG_TIE_RTOL * max(1.0, abs(candidate), abs(observed))
    return bool(
        candidate <= observed + tolerance
        if lower_tail
        else candidate >= observed - tolerance
    )


def ehy_uniformity(
    x: ArrayLike,
    *,
    alpha: float,
    n_neighbors: int,
    calibration: str = "monte-carlo",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Perform the EHY nearest-neighbor test of simplex uniformity.

    Distances are intrinsic Euclidean distances on the flat
    ``(D-1)``-dimensional simplex.  Boundary compositions are allowed.  The
    test rejects in the lower tail when ``0 < alpha < 1`` and in the upper
    tail when ``alpha > 1``; ``alpha=1`` is not an identifiable test.
    """
    validate_choice(calibration, name="calibration", choices=("monte-carlo",))
    power = validate_real_scalar(alpha, name="alpha")
    if power <= 0.0:
        raise ValueError("alpha must be greater than 0")
    if power == 1.0:
        raise ValueError("alpha must not equal 1")
    neighbors = validate_positive_integer(n_neighbors, name="n_neighbors")
    values = _closed_simplex(x, name="x")
    sample_size, components = values.shape
    if neighbors >= sample_size:
        raise ValueError("n_neighbors must be smaller than the sample size")
    resamples = validate_positive_integer(n_resamples, name="n_resamples")
    generator = make_generator(rng)
    observed_log = _ehy_simplex_log_statistic(
        values, alpha=power, n_neighbors=neighbors
    )
    lower_tail = power < 1.0
    exceedances = 0
    for _ in range(resamples):
        simulated = generator.dirichlet(np.ones(components), size=sample_size)
        simulated_log = _ehy_simplex_log_statistic(
            simulated, alpha=power, n_neighbors=neighbors
        )
        exceedances += int(
            _ehy_log_tail_contains(
                simulated_log,
                observed_log,
                lower_tail=lower_tail,
            )
        )
    pvalue, standard_error, interval = monte_carlo_calibration(exceedances, resamples)
    return ResamplingTestResult(
        statistic=_nonnegative_from_log(observed_log * max(1.0, power)),
        pvalue=pvalue,
        method="Ebner-Henze-Yukich nearest-neighbor simplex-uniformity test (2018)",
        alternative=_ALTERNATIVE,
        data_name="x",
        statistic_name="T_alpha,n,J",
        calibration="Monte Carlo simplex-uniform null calibration",
        n_resamples=resamples,
        exceedances=exceedances,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
        diagnostics=(
            ("alpha", power),
            ("n_neighbors", neighbors),
            ("rejection tail", "lower" if lower_tail else "upper"),
        ),
    )


def _row_invariant_key(row: NDArray[np.float64]) -> bytes:
    return np.sort(row).tobytes(order="C")


def _canonical_simplex_groups(
    samples: tuple[ArrayLike, ...], *, interior: bool
) -> tuple[NDArray[np.float64], ...]:
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    groups: list[NDArray[np.float64]] = []
    components: int | None = None
    for index, sample in enumerate(samples):
        values = validate_simplex_sample(
            sample, name=f"samples[{index}]", interior=interior
        )
        if components is None:
            components = values.shape[1]
        elif values.shape[1] != components:
            raise ValueError("all samples must have the same number of components")
        row_sums = _stable_row_sums(values)
        values = np.asarray(values / row_sums[:, None], dtype=np.float64, order="C")
        groups.append(np.ascontiguousarray(values))

    def group_key(
        group: NDArray[np.float64],
    ) -> tuple[int, tuple[bytes, ...], bytes]:
        row_keys = tuple(sorted(_row_invariant_key(row) for row in group))
        # ``scipy.spatial.distance`` accumulates squared coordinate
        # differences in their current column order.  A common component
        # permutation can therefore move the last bit of a distance and, in
        # turn, change a seeded label plan.  Sort the nonnegative summands
        # before reduction so this canonicalization key is bitwise invariant
        # to the scientifically irrelevant component order.
        within_matrix = _component_invariant_distances(group)
        within_distances = np.sort(within_matrix[np.triu_indices(group.shape[0], k=1)])
        return group.shape[0], row_keys, within_distances.tobytes(order="C")

    ordered_groups = sorted(groups, key=group_key)

    # Sorted component values alone tie whenever two observations are
    # component permutations.  Break such ties with the row's sorted distance
    # signature to every canonical group.  The signature is invariant to row,
    # group, and common component permutations. When the signatures resolve
    # all nonidentical rows, a fixed RNG stream assigns the same label plans
    # after those scientifically irrelevant reorderings. Unresolved metric
    # symmetries do not prevent valid uniform label permutations.
    canonical_groups: list[NDArray[np.float64]] = []
    for group in ordered_groups:

        def row_key(row: NDArray[np.float64]) -> tuple[bytes, tuple[bytes, ...]]:
            signatures: list[bytes] = []
            for target in ordered_groups:
                squared_differences = np.sort((target - row) ** 2, axis=1)
                distances = np.sqrt(
                    np.sum(squared_differences, axis=1, dtype=np.float64)
                )
                signatures.append(np.sort(distances).tobytes(order="C"))
            return _row_invariant_key(row), tuple(signatures)

        order = sorted(range(group.shape[0]), key=lambda index: row_key(group[index]))
        canonical_groups.append(np.ascontiguousarray(group[order]))

    return tuple(canonical_groups)


def _alpha_transform(
    values: NDArray[np.float64], *, alpha: float
) -> NDArray[np.float64]:
    components = values.shape[1]
    with np.errstate(divide="ignore"):
        log_values = np.log(values)
    if np.any(values == 0.0):
        if alpha <= 0.0:
            raise ValueError("zero components require alpha greater than 0")
        powered_logs = alpha * log_values
        log_power_composition = powered_logs - _stable_logsumexp_rows(powered_logs)
        transformed = np.expm1(math.log(components) + log_power_composition) / alpha
        if not np.all(np.isfinite(transformed)):
            raise ValueError("the alpha transformation is not finite")
        return np.asarray(transformed, dtype=np.float64, order="C")
    # A sorted reduction makes a common component permutation bitwise
    # reproducible, rather than merely equal within rounding error.
    mean_logs = (
        np.sum(np.sort(log_values, axis=1), axis=1, keepdims=True, dtype=np.float64)
        / components
    )
    centered_logs = log_values - mean_logs
    if alpha == 0.0:
        return np.asarray(centered_logs, dtype=np.float64, order="C")

    scaled_logs = alpha * centered_logs
    maximum_scaled_log = float(np.max(np.abs(scaled_logs)))
    if maximum_scaled_log <= math.sqrt(np.finfo(np.float64).eps):
        # Dividing D*softmax(alpha*l)-1 by alpha catastrophically cancels as
        # alpha -> 0. Expand the quotient instead. With mean(l)=0,
        #
        # T_i(a) = l_i + a/2 (l_i^2-m2)
        #          + a^2 (l_i^3/6-l_i*m2/2-m3/6) + O(a^3).
        #
        # This preserves a representable finite-alpha correction rather than
        # replacing every sufficiently small nonzero alpha by the CLR limit.
        squared_logs = centered_logs * centered_logs
        second_moment = (
            np.sum(
                np.sort(squared_logs, axis=1),
                axis=1,
                keepdims=True,
                dtype=np.float64,
            )
            / components
        )
        cubed_logs = squared_logs * centered_logs
        third_moment = (
            np.sum(
                np.sort(cubed_logs, axis=1),
                axis=1,
                keepdims=True,
                dtype=np.float64,
            )
            / components
        )
        transformed = (
            centered_logs
            + 0.5 * alpha * (squared_logs - second_moment)
            + alpha**2
            * (
                cubed_logs / 6.0
                - 0.5 * centered_logs * second_moment
                - third_moment / 6.0
            )
        )
        transformed -= (
            np.sum(
                np.sort(transformed, axis=1),
                axis=1,
                keepdims=True,
                dtype=np.float64,
            )
            / components
        )
        return np.asarray(transformed, dtype=np.float64, order="C")

    largest_safe_exponent = (
        math.log(np.finfo(np.float64).max) - math.log(components) - 2.0
    )
    if maximum_scaled_log >= largest_safe_exponent:
        # The centered exponential representation below can overflow even
        # though the normalized power composition is finite (for example,
        # alpha=-1 with a subnormal positive component). A log-softmax
        # evaluates that extreme regime without forming x**alpha.
        log_power_composition = scaled_logs - _stable_logsumexp_rows(scaled_logs)
        transformed = np.expm1(math.log(components) + log_power_composition) / alpha
        transformed -= (
            np.sum(
                np.sort(transformed, axis=1),
                axis=1,
                keepdims=True,
                dtype=np.float64,
            )
            / components
        )
        if not np.all(np.isfinite(transformed)):
            raise ValueError("the alpha transformation is not finite")
        return np.asarray(transformed, dtype=np.float64, order="C")

    # expm1/log1p preserve the small offset of the power composition from
    # 1/D. Centering the logs first also prevents overflow for negative alpha.
    exponential_offsets = np.expm1(scaled_logs)
    mean_exponential_offset = (
        np.sum(
            np.sort(exponential_offsets, axis=1),
            axis=1,
            keepdims=True,
            dtype=np.float64,
        )
        / components
    )
    log_mean_exponential = np.log1p(mean_exponential_offset)
    transformed = np.expm1(scaled_logs - log_mean_exponential) / alpha
    transformed_mean = (
        np.sum(
            np.sort(transformed, axis=1),
            axis=1,
            keepdims=True,
            dtype=np.float64,
        )
        / components
    )
    transformed -= transformed_mean
    if not np.all(np.isfinite(transformed)):
        raise ValueError("the alpha transformation is not finite")
    return np.asarray(transformed, dtype=np.float64, order="C")


def _component_invariant_distances(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return Euclidean distances with component-order-stable summation."""
    sample_size = values.shape[0]
    result = np.zeros((sample_size, sample_size), dtype=np.float64)
    for first in range(sample_size - 1):
        differences = values[first + 1 :] - values[first]
        distances = np.hypot.reduce(np.sort(np.abs(differences), axis=1), axis=1)
        result[first, first + 1 :] = distances
        result[first + 1 :, first] = distances
    return result


def _nonnegative_energy_contrast(
    *,
    cross_mean: float,
    first_mean: float,
    second_mean: float,
) -> float:
    """Return a Euclidean energy contrast with roundoff certification."""
    contrast = 2.0 * cross_mean - first_mean - second_mean
    scale = 2.0 * abs(cross_mean) + abs(first_mean) + abs(second_mean)
    tolerance = 500.0 * np.finfo(np.float64).eps * max(scale, np.finfo(np.float64).tiny)
    if contrast < 0.0 and abs(contrast) <= tolerance:
        return 0.0
    if contrast < 0.0:
        raise ArithmeticError("the Euclidean energy contrast became negative")
    return contrast


def _energy_statistic_invariant(
    distances: NDArray[np.float64], labels: NDArray[np.intp], sizes: tuple[int, ...]
) -> float:
    """Evaluate energy with row-order-stable nonnegative reductions."""

    def stable_mean(values: NDArray[np.float64]) -> float:
        ordered = np.sort(values, axis=None)
        return float(np.sum(ordered, dtype=np.float64) / ordered.size)

    group_indices = tuple(
        np.flatnonzero(labels == group) for group in range(len(sizes))
    )
    within_means = tuple(
        stable_mean(distances[np.ix_(indices, indices)]) for indices in group_indices
    )
    components: list[float] = []
    for first in range(len(sizes) - 1):
        first_indices = group_indices[first]
        first_mean = within_means[first]
        for second in range(first + 1, len(sizes)):
            second_indices = group_indices[second]
            second_mean = within_means[second]
            cross_mean = stable_mean(distances[np.ix_(first_indices, second_indices)])
            first_size = sizes[first]
            second_size = sizes[second]
            contrast = _nonnegative_energy_contrast(
                cross_mean=cross_mean,
                first_mean=first_mean,
                second_mean=second_mean,
            )
            components.append(
                first_size * second_size / (first_size + second_size) * contrast
            )
    return float(np.sum(np.sort(np.asarray(components)), dtype=np.float64))


def _simplex_allocation_count(sizes: tuple[int, ...]) -> int:
    remaining = sum(sizes)
    count = 1
    for size in sizes[:-1]:
        count *= math.comb(remaining, size)
        remaining -= size
    return count


def _simplex_label_allocations(
    sizes: tuple[int, ...],
) -> Iterator[NDArray[np.intp]]:
    total = sum(sizes)
    labels = np.empty(total, dtype=np.intp)

    def visit(remaining: tuple[int, ...], group: int) -> Iterator[NDArray[np.intp]]:
        if group == len(sizes) - 1:
            labels[np.fromiter(remaining, dtype=np.intp)] = group
            yield labels.copy()
            return
        for chosen in combinations(remaining, sizes[group]):
            labels[np.fromiter(chosen, dtype=np.intp)] = group
            chosen_set = set(chosen)
            next_remaining = tuple(
                value for value in remaining if value not in chosen_set
            )
            yield from visit(next_remaining, group + 1)

    yield from visit(tuple(range(total)), 0)


def alpha_energy_ksamp(
    *samples: ArrayLike,
    alpha: float,
    calibration: str = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Test equality of compositional distributions with alpha-energy.

    ``alpha`` is fixed before labels are inspected and must lie in ``[-1, 1]``.
    At zero the transform is the isometric log-ratio metric (represented in
    centered log-ratio coordinates); it therefore requires strictly positive
    components.  Structural zeros are supported only when ``alpha > 0``.
    """
    selected_calibration = validate_choice(
        calibration,
        name="calibration",
        choices=("permutation", "exact", "monte-carlo"),
    )
    power = validate_real_scalar(alpha, name="alpha")
    if not -1.0 <= power <= 1.0:
        raise ValueError("alpha must be between -1 and 1")
    groups = _canonical_simplex_groups(samples, interior=power <= 0.0)
    if power <= 0.0 and any(np.any(group == 0.0) for group in groups):
        raise ValueError("zero components require alpha greater than 0")
    sizes = tuple(group.shape[0] for group in groups)
    transformed = tuple(_alpha_transform(group, alpha=power) for group in groups)
    pooled = np.vstack(transformed)
    coordinate_scale = float(np.max(np.abs(pooled)))
    normalized = pooled if coordinate_scale == 0.0 else pooled / coordinate_scale
    distances = _component_invariant_distances(normalized)
    base_labels = np.concatenate(
        [np.full(size, index, dtype=np.intp) for index, size in enumerate(sizes)]
    )
    observed_normalized = _energy_statistic_invariant(distances, base_labels, sizes)
    observed = observed_normalized * coordinate_scale

    resamples = validate_positive_integer(n_resamples, name="n_resamples")
    generator = make_generator(rng)
    total_allocations = _simplex_allocation_count(sizes)
    use_exact = selected_calibration == "exact" or (
        selected_calibration == "permutation" and total_allocations <= resamples
    )
    if selected_calibration == "exact" and total_allocations > resamples:
        raise ValueError(
            f"exact calibration requires {total_allocations:,} allocations; "
            "increase n_resamples to at least that value"
        )

    exceedances = 0
    threshold = observed_normalized - abs(observed_normalized) * _PERMUTATION_TIE_RTOL
    if use_exact:
        for labels in _simplex_label_allocations(sizes):
            exceedances += int(
                _energy_statistic_invariant(distances, labels, sizes) >= threshold
            )
        effective_resamples = total_allocations
        pvalue = exact_pvalue(exceedances, total_allocations)
        standard_error = None
        interval = None
        calibration_label = "exact permutation"
    else:
        invariant_orbit_budget = min(
            _INVARIANT_ORBIT_MONTE_CARLO_LIMIT,
            _INVARIANT_ORBIT_WORK_MULTIPLIER * resamples,
        )
        if total_allocations <= invariant_orbit_budget:
            # Sampling indices from the sorted exact permutation distribution
            # is equivalent to drawing label allocations uniformly with
            # replacement.  Unlike applying random labels to arbitrary row
            # indices, it remains seeded-replay invariant even when the pooled
            # geometry has nontrivial automorphisms (for example, every
            # component permutation of one composition).
            null_statistics = np.fromiter(
                (
                    _energy_statistic_invariant(distances, labels, sizes)
                    for labels in _simplex_label_allocations(sizes)
                ),
                dtype=np.float64,
                count=total_allocations,
            )
            null_statistics.sort()
            selected = generator.integers(
                0, total_allocations, size=resamples, endpoint=False
            )
            exceedances = int(np.count_nonzero(null_statistics[selected] >= threshold))
        else:
            # Every fixed-size label vector has the same product(n_g!)
            # preimages under a uniform row permutation, including when rows
            # coincide. Geometric symmetries can affect bitwise seed replay
            # after a transformation, but never justify refusing inference.
            for _ in range(resamples):
                labels = generator.permutation(base_labels)
                exceedances += int(
                    _energy_statistic_invariant(distances, labels, sizes) >= threshold
                )
        effective_resamples = resamples
        pvalue, standard_error, interval = monte_carlo_calibration(
            exceedances, resamples
        )
        calibration_label = "Monte Carlo permutation"

    return ResamplingTestResult(
        statistic=observed,
        pvalue=pvalue,
        method="Alpha-energy k-sample test for compositional distributions",
        alternative="at least two compositional distributions differ",
        data_name="samples",
        statistic_name="alpha-EBT",
        calibration=calibration_label,
        n_resamples=effective_resamples,
        exceedances=exceedances,
        exact=use_exact,
        monte_carlo_standard_error=standard_error,
        tail_probability_interval=interval,
        diagnostics=(("alpha", power), ("groups", len(groups))),
    )

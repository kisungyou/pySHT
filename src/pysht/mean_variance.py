"""Joint tests of univariate population means and variances.

All procedures in this module assume independent observations from normal
populations.  The likelihood-ratio calculations use maximum-likelihood
variances (denominator ``n``), as required by their derivations, rather than
the unbiased sample variances used by the component t and F tests.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize, special, stats

from ._results import HypothesisTestResult
from ._validation import validate_1d_sample, validate_real_scalar

__all__ = [
    "as_1samp",
    "lrt_2samp",
    "muirhead_2samp",
    "pl_2samp",
    "pn_2samp",
    "zxc_2samp",
]

_CHI_SQUARED_2 = "asymptotic chi-square distribution with 2 degrees of freedom"
_JOINT_ALTERNATIVE_1SAMP = (
    "at least one of the population mean and variance differs from its null value"
)
_JOINT_ALTERNATIVE_2SAMP = (
    "at least one of the two population means and variances differs"
)
_LOG_SMALLEST = math.log(float(np.nextafter(0.0, 1.0)))
_LOG_LARGEST = math.log(float(np.finfo(np.float64).max))


@dataclass(frozen=True, slots=True)
class _TwoSampleSummary:
    """Scale-free sufficient statistics for two normal samples."""

    n: int
    m: int
    mean_x: float
    mean_y: float
    ss_x: float
    ss_y: float
    pooled_null_ss: float
    log_lambda: float
    log_ss_x: float
    log_ss_y: float
    log_abs_mean_difference: float


def _softplus(value: float) -> float:
    """Evaluate log(1 + exp(value)) without overflow."""
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _log_beta_lower(log_value: float, shape1: float, shape2: float) -> float:
    """Return a beta lower-tail log probability from log(x)."""
    if log_value >= 0.0:
        return 0.0
    if log_value > -30.0:
        probability = float(special.betainc(shape1, shape2, math.exp(log_value)))
        if probability > 0.0:
            return math.log(probability)

    # I_x(a,b) = x**a 2F1(a,1-b;a+1;x) / (a B(a,b)).
    # At x <= exp(-30), the hypergeometric factor is close to one but is
    # retained when SciPy can evaluate it, avoiding an avoidable cutoff error
    # for large beta shapes.
    log_probability = (
        shape1 * log_value - math.log(shape1) - float(special.betaln(shape1, shape2))
    )
    if log_value >= _LOG_SMALLEST:
        value = math.exp(log_value)
        correction = float(special.hyp2f1(shape1, 1.0 - shape2, shape1 + 1.0, value))
        if math.isfinite(correction) and correction > 0.0:
            log_probability += math.log(correction)
    return min(0.0, log_probability)


def _scaled_sample_moments(
    values: NDArray[np.float64],
) -> tuple[float, float]:
    """Return a mean and centered sum of squares on a common finite scale."""
    mean = float(np.mean(values, dtype=np.float64))
    centered = values - mean
    sum_squares = float(np.dot(centered, centered))
    return mean, sum_squares


def _log_centered_sum_squares(
    values: NDArray[np.float64],
    *,
    reference_scale: float,
) -> float:
    """Return log centered sum of squares in reference-scale units."""
    anchor = float(np.min(values))
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = values - anchor
    if np.all(np.isfinite(shifted)):
        scale = float(np.max(np.abs(shifted)))
        scaled = shifted / scale
    else:
        # A difference of opposite-sign float64 endpoints can overflow even
        # though division by their common absolute scale is well defined.
        scale = float(np.max(np.abs(values)))
        scaled = values / scale

    _, scaled_sum_squares = _scaled_sample_moments(scaled)
    if scaled_sum_squares <= 0.0 or not math.isfinite(scaled_sum_squares):
        raise ArithmeticError("the sample variance could not be evaluated")
    scale_ratio = scale / reference_scale
    if scale_ratio > 0.0:
        log_scale_ratio = math.log(scale_ratio)
    else:
        # The ratio alone can underflow when the samples span more than the
        # full exponent range, although both individual scales are finite.
        log_scale_ratio = math.log(scale) - math.log(reference_scale)
    return 2.0 * log_scale_ratio + math.log(scaled_sum_squares)


def _log_squared_mean_difference(
    values: NDArray[np.float64], null_mean: float, *, reference_scale: float = 1.0
) -> float:
    """Return a squared mean-difference log in reference-scale units.

    Direct subtraction is preferred because it preserves ulp-sized departures
    around a large common null.  Scaling first is reserved for the case where
    opposite-sign finite endpoints make that subtraction overflow.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        differences = values - null_mean
    if np.all(np.isfinite(differences)):
        scale = float(np.max(np.abs(differences)))
        if scale == 0.0:
            return -math.inf
        scaled_difference = float(np.mean(differences / scale, dtype=np.float64))
        if scaled_difference == 0.0:
            return -math.inf
        scale_ratio = scale / reference_scale
        log_scale_ratio = (
            math.log(scale_ratio)
            if scale_ratio > 0.0
            else math.log(scale) - math.log(reference_scale)
        )
        log_absolute_difference = log_scale_ratio + math.log(abs(scaled_difference))
    else:
        scale = max(float(np.max(np.abs(values))), abs(null_mean))
        scaled_difference = float(
            np.mean(values / scale - null_mean / scale, dtype=np.float64)
        )
        if scaled_difference == 0.0:
            return -math.inf
        scale_ratio = scale / reference_scale
        log_scale_ratio = (
            math.log(scale_ratio)
            if scale_ratio > 0.0
            else math.log(scale) - math.log(reference_scale)
        )
        log_absolute_difference = log_scale_ratio + math.log(abs(scaled_difference))
    return 2.0 * log_absolute_difference


def _two_sample_summary(x: ArrayLike, y: ArrayLike) -> _TwoSampleSummary:
    first = validate_1d_sample(x, name="x")
    second = validate_1d_sample(y, name="y")
    if np.all(first == first[0]):
        raise ValueError("x must have positive sample variance")
    if np.all(second == second[0]):
        raise ValueError("y must have positive sample variance")

    # Subtract one exactly representable shared anchor before scaling.  If a
    # large common location is divided away first, within-group differences
    # of only a few ulps can be destroyed before the means are subtracted.
    # Make the common coordinate system independent of group order.  This is
    # observable for samples separated by hundreds of orders of magnitude,
    # where choosing either group's first observation can round differently.
    anchor = min(float(np.min(first)), float(np.min(second)))
    with np.errstate(over="ignore", invalid="ignore"):
        first_shifted = first - anchor
        second_shifted = second - anchor
    if np.all(np.isfinite(first_shifted)) and np.all(np.isfinite(second_shifted)):
        scale = max(
            float(np.max(np.abs(first_shifted))),
            float(np.max(np.abs(second_shifted))),
        )
        if scale == 0.0:
            raise ValueError("x and y must have positive sample variances")
        first_scaled = first_shifted / scale
        second_scaled = second_shifted / scale
    else:
        # Opposite extreme float64 endpoints can overflow on subtraction.
        # Scaling first is safe in that fallback regime because there is no
        # nearly cancelling common offset to preserve.
        scale = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))))
        first_scaled = first / scale
        second_scaled = second / scale

    mean_x, ss_x = _scaled_sample_moments(first_scaled)
    mean_y, ss_y = _scaled_sample_moments(second_scaled)
    n = first.size
    m = second.size
    total = n + m
    mean_difference = mean_x - mean_y
    log_abs_mean_difference = (
        -math.inf if mean_difference == 0.0 else math.log(abs(mean_difference))
    )
    pooled_null_ss = ss_x + ss_y + (n * m / total) * mean_difference**2
    if not math.isfinite(pooled_null_ss) or pooled_null_ss <= 0.0:
        raise ValueError("the pooled null variance could not be evaluated")

    # Evaluate each group's variance on its own scale.  A common-scale sum of
    # squares can be nonzero but entirely numerical noise after a tiny group
    # is translated by a much larger group, so checking only for exact
    # underflow is insufficient.  Returning these logs in units of the shared
    # scale also retains invariance around a huge common location.
    log_ss_x = _log_centered_sum_squares(first, reference_scale=scale)
    log_ss_y = _log_centered_sum_squares(second, reference_scale=scale)
    if log_abs_mean_difference == -math.inf:
        log_between_ss = -math.inf
    else:
        log_between_ss = math.log(n * m / total) + 2.0 * log_abs_mean_difference
    log_pooled_null_ss = float(
        special.logsumexp(tuple(sorted((log_ss_x, log_ss_y, log_between_ss))))
    )
    log_lambda = (
        0.5 * n * (log_ss_x - math.log(n))
        + 0.5 * m * (log_ss_y - math.log(m))
        - 0.5 * total * (log_pooled_null_ss - math.log(total))
    )
    # The likelihood ratio is at most one.  A tiny positive value can arise
    # solely from cancellation when the two fitted normal models coincide.
    tolerance = 64.0 * np.finfo(np.float64).eps * total
    if log_lambda > tolerance:
        raise ArithmeticError("the likelihood ratio exceeded one numerically")

    return _TwoSampleSummary(
        n=n,
        m=m,
        mean_x=mean_x,
        mean_y=mean_y,
        ss_x=ss_x,
        ss_y=ss_y,
        pooled_null_ss=pooled_null_ss,
        log_lambda=min(0.0, log_lambda),
        log_ss_x=log_ss_x,
        log_ss_y=log_ss_y,
        log_abs_mean_difference=log_abs_mean_difference,
    )


def _likelihood_ratio(log_lambda: float) -> float:
    if log_lambda < _LOG_SMALLEST:
        return 0.0
    return math.exp(log_lambda)


def as_1samp(
    x: ArrayLike,
    *,
    popmean: float = 0.0,
    variance: float = 1.0,
) -> HypothesisTestResult:
    """Test one normal population mean and variance jointly.

    This is the likelihood-ratio procedure discussed by Arnold and Shavelle
    (1998).  It represents both ``mvar1.1998AS`` and ``mvar1.LRT`` from SHT,
    whose statistics are algebraically identical.

    Parameters
    ----------
    x
        One-dimensional sample with positive sample variance.
    popmean
        Population mean under the joint null hypothesis.
    variance
        Positive population variance under the joint null hypothesis.

    Notes
    -----
    The chi-square calibration is asymptotic.  Normality and independent
    observations are required.
    """
    values = validate_1d_sample(x, name="x")
    if np.all(values == values[0]):
        raise ValueError("x must have positive sample variance")
    null_mean = validate_real_scalar(popmean, name="popmean")
    null_variance = validate_real_scalar(variance, name="variance")
    if null_variance <= 0.0:
        raise ValueError("variance must be greater than 0")

    n = values.size
    log_null_variance = math.log(null_variance)
    null_standard_deviation = math.sqrt(null_variance)
    reference_correction = 2.0 * math.log(null_standard_deviation) - log_null_variance
    log_mle_variance_ratio = (
        _log_centered_sum_squares(values, reference_scale=null_standard_deviation)
        - math.log(n)
        + reference_correction
    )
    if log_mle_variance_ratio > _LOG_LARGEST:
        variance_term = math.inf
    elif abs(log_mle_variance_ratio) < 0.5:
        # exp(a) - 1 - a is the nonnegative likelihood contribution.  expm1
        # avoids turning a near-null variance fit into a negative statistic.
        variance_term = math.expm1(log_mle_variance_ratio) - log_mle_variance_ratio
    else:
        variance_term = math.exp(log_mle_variance_ratio) - 1.0 - log_mle_variance_ratio

    log_mean_square_ratio = (
        _log_squared_mean_difference(
            values,
            null_mean,
            reference_scale=null_standard_deviation,
        )
        + reference_correction
    )
    if log_mean_square_ratio == -math.inf:
        mean_term = 0.0
    elif log_mean_square_ratio > _LOG_LARGEST:
        mean_term = math.inf
    else:
        mean_term = math.exp(log_mean_square_ratio)

    statistic = n * (variance_term + mean_term)
    if math.isnan(statistic):
        raise ArithmeticError("the likelihood-ratio statistic is not defined")
    statistic = max(0.0, statistic)
    pvalue = float(stats.chi2.sf(statistic, 2.0))

    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Arnold-Shavelle one-sample mean and variance test (1998)",
        alternative=_JOINT_ALTERNATIVE_1SAMP,
        data_name="x",
        statistic_name="-2 log Lambda",
        calibration=_CHI_SQUARED_2,
        df=2.0,
    )


def pn_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the Pearson--Neyman two-sample normal-population test.

    The null distribution of the likelihood ratio is approximated by a beta
    distribution whose parameters match its first two exact null moments.
    Small likelihood ratios contradict equality, so the p-value is the lower
    beta tail.  This corrects the reversed tail in SHT 0.1.9.
    """
    summary = _two_sample_summary(x, y)
    n = summary.n
    m = summary.m
    total = n + m

    log_first_moment = (
        0.5 * total * math.log(total)
        - 0.5 * n * math.log(n)
        - 0.5 * m * math.log(m)
        + special.gammaln(n - 0.5)
        + special.gammaln(m - 0.5)
        - special.gammaln(total - 0.5)
        + special.gammaln(0.5 * (total - 1))
        - special.gammaln(0.5 * (n - 1))
        - special.gammaln(0.5 * (m - 1))
    )
    log_second_moment = (
        total * math.log(total)
        - n * math.log(n)
        - m * math.log(m)
        + special.gammaln(0.5 * (3 * n - 1))
        + special.gammaln(0.5 * (3 * m - 1))
        - special.gammaln(0.5 * (3 * total - 1))
        + special.gammaln(0.5 * (total - 1))
        - special.gammaln(0.5 * (n - 1))
        - special.gammaln(0.5 * (m - 1))
    )

    mean = math.exp(log_first_moment)
    second_moment = math.exp(log_second_moment)
    moment_gap = 2.0 * log_first_moment - log_second_moment
    variance = second_moment * (-math.expm1(moment_gap))
    if not 0.0 < mean < 1.0 or not math.isfinite(variance) or variance <= 0.0:
        raise ArithmeticError("the Pearson-Neyman beta moments are invalid")

    concentration = mean * (1.0 - mean) / variance - 1.0
    shape1 = mean * concentration
    shape2 = (1.0 - mean) * concentration
    if shape1 <= 0.0 or shape2 <= 0.0:
        raise ArithmeticError("the Pearson-Neyman beta shapes are invalid")

    likelihood_ratio = _likelihood_ratio(summary.log_lambda)
    pvalue = float(stats.beta.cdf(likelihood_ratio, shape1, shape2))
    return HypothesisTestResult(
        statistic=likelihood_ratio,
        pvalue=pvalue,
        method="Pearson-Neyman two-sample mean and variance test (1930)",
        alternative=_JOINT_ALTERNATIVE_2SAMP,
        data_name="x and y",
        statistic_name="Lambda",
        calibration="moment-matched beta distribution (lower tail)",
        diagnostics=(("beta shape 1", shape1), ("beta shape 2", shape2)),
    )


def pl_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the Perng--Littell joint two-sample test.

    Under independent normal samples and the joint null, the equal-variance
    pooled t statistic is independent of the sample-variance ratio.  Fisher's
    method therefore combines their two-sided p-values with a chi-square law
    having four degrees of freedom.
    """
    summary = _two_sample_summary(x, y)
    n = summary.n
    m = summary.m
    residual_df = n + m - 2
    log_pooled_variance = float(
        special.logsumexp(tuple(sorted((summary.log_ss_x, summary.log_ss_y))))
    ) - math.log(residual_df)
    log_standard_error = 0.5 * (log_pooled_variance + math.log(1.0 / n + 1.0 / m))
    if summary.log_abs_mean_difference == -math.inf:
        log_t_pvalue = 0.0
    else:
        log_abs_t = summary.log_abs_mean_difference - log_standard_error
        log_t_beta_argument = -_softplus(2.0 * log_abs_t - math.log(residual_df))
        # The two-sided Student t tail is exactly this regularized beta tail.
        log_t_pvalue = _log_beta_lower(log_t_beta_argument, 0.5 * residual_df, 0.5)

    first_df = n - 1
    second_df = m - 1
    log_variance_ratio = (
        summary.log_ss_x - math.log(first_df) - summary.log_ss_y + math.log(second_df)
    )
    f_logit = log_variance_ratio + math.log(first_df) - math.log(second_df)
    log_f_lower = _log_beta_lower(-_softplus(-f_logit), 0.5 * first_df, 0.5 * second_df)
    log_f_upper = _log_beta_lower(-_softplus(f_logit), 0.5 * second_df, 0.5 * first_df)
    log_f_pvalue = min(0.0, math.log(2.0) + min(log_f_lower, log_f_upper))

    log_combined_probability = min(0.0, log_t_pvalue + log_f_pvalue)
    statistic = -2.0 * log_combined_probability
    # chi2_4.sf(-2 log q) = q (1 - log q).  Evaluating this identity in the
    # log domain retains subnormal combined probabilities when q itself does
    # not survive exponentiation.
    log_pvalue = log_combined_probability + math.log1p(-log_combined_probability)
    pvalue = 0.0 if log_pvalue < _LOG_SMALLEST else math.exp(log_pvalue)
    t_pvalue = 0.0 if log_t_pvalue < _LOG_SMALLEST else math.exp(log_t_pvalue)
    f_pvalue = 0.0 if log_f_pvalue < _LOG_SMALLEST else math.exp(log_f_pvalue)

    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Perng-Littell two-sample mean and variance test (1976)",
        alternative=_JOINT_ALTERNATIVE_2SAMP,
        data_name="x and y",
        statistic_name="Fisher combination",
        calibration="chi-square distribution with 4 degrees of freedom",
        df=4.0,
        diagnostics=(("pooled t p-value", t_pvalue), ("F-test p-value", f_pvalue)),
    )


def muirhead_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform Muirhead's corrected likelihood-ratio approximation.

    The second-order null approximation is evaluated in the upper tail.  The
    finite expansion can leave the probability interval in extreme tails, so
    the reported approximation is truncated to ``[0, 1]``.
    """
    summary = _two_sample_summary(x, y)
    n = summary.n
    m = summary.m
    total = n + m
    ratio_sum = total / n + total / m - 1.0
    rho = 1.0 - (22.0 / (24.0 * total)) * ratio_sum
    gamma = (
        0.5 * ((total / n) ** 2 + (total / m) ** 2 - 1.0)
        - (121.0 / 96.0) * ratio_sum**2
    )
    if rho <= 0.0:
        raise ArithmeticError("Muirhead's correction factor is not positive")

    statistic = -2.0 * rho * summary.log_lambda
    correction = gamma / (rho * rho * total * total)
    sf2 = float(stats.chi2.sf(statistic, 2.0))
    sf6 = float(stats.chi2.sf(statistic, 6.0))
    raw_pvalue = sf2 + correction * (sf6 - sf2)
    pvalue = min(1.0, max(0.0, raw_pvalue))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Muirhead two-sample mean and variance test (1982)",
        alternative=_JOINT_ALTERNATIVE_2SAMP,
        data_name="x and y",
        statistic_name="-2 rho log Lambda",
        calibration="second-order corrected chi-square approximation (upper tail)",
        df=2.0,
        diagnostics=(("rho", rho), ("second-order coefficient", correction)),
    )


def lrt_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the asymptotic likelihood-ratio equality test.

    The procedure tests equality of both parameters of two independent normal
    populations.  Its statistic is ``-2 log(Lambda)`` and its chi-square law
    is asymptotic.
    """
    summary = _two_sample_summary(x, y)
    statistic = -2.0 * summary.log_lambda
    pvalue = float(stats.chi2.sf(statistic, 2.0))
    return HypothesisTestResult(
        statistic=statistic,
        pvalue=pvalue,
        method="Two-sample likelihood-ratio test of mean and variance",
        alternative=_JOINT_ALTERNATIVE_2SAMP,
        data_name="x and y",
        statistic_name="-2 log Lambda",
        calibration=_CHI_SQUARED_2,
        df=2.0,
    )


def _exact_lrt_pvalue(log_lambda: float, n: int, m: int) -> float:
    """Evaluate Zhang--Xu--Chen's exact lower tail in stable coordinates.

    Under the null, three normalized sums of squares have a Dirichlet law
    with parameters ``((n-1)/2, (m-1)/2, 1/2)``.  Conditioning on the first
    coordinate reduces the legacy two-dimensional quadrature to one integral
    involving beta probabilities.  Roots, tail masses, and the quadrature
    itself remain in logit/log coordinates, so representable probabilities
    are retained even when the likelihood ratio has underflowed to zero.
    """
    # The probability is symmetric in the sample sizes.  Assigning the
    # smaller shape to the conditional beta variable avoids a poorly resolved
    # boundary layer when the samples are highly unbalanced, and makes that
    # symmetry exact in floating-point arithmetic.
    n, m = max(n, m), min(n, m)
    if log_lambda >= -64.0 * np.finfo(np.float64).eps * (n + m):
        return 1.0
    if log_lambda == -math.inf:
        return 0.0
    if not math.isfinite(log_lambda):
        raise ArithmeticError("the log likelihood ratio is not finite")

    total = n + m
    first_shape = 0.5 * (n - 1)
    second_shape = 0.5 * (m - 1)
    residual_shape = 0.5
    marginal_second_shape = second_shape + residual_shape
    log_constant = (2.0 / m) * (
        log_lambda
        + 0.5 * n * math.log(n)
        + 0.5 * m * math.log(m)
        - 0.5 * total * math.log(total)
    )

    def log_ratio_logit(logit: float) -> float:
        # With x=expit(u), -log(x)=softplus(-u) and
        # -log(1-x)=softplus(u).  Solving in u retains roots many orders of
        # magnitude below a fixed absolute tolerance in x-space.
        return log_constant + (n / m) * _softplus(-logit) + _softplus(logit)

    midpoint_logit = math.log(n / m)

    def find_root(direction: float) -> float:
        step = 1.0
        outer = midpoint_logit + direction * step
        while log_ratio_logit(outer) <= 0.0:
            step *= 2.0
            outer = midpoint_logit + direction * step
            if not math.isfinite(outer):
                raise ArithmeticError("the exact likelihood-ratio roots were not found")
        left, right = sorted((midpoint_logit, outer))
        return float(
            optimize.brentq(
                log_ratio_logit,
                left,
                right,
                xtol=1e-12,
                rtol=4.0 * np.finfo(np.float64).eps,
            )
        )

    lower_logit = find_root(-1.0)
    upper_logit = find_root(1.0)
    log_beta_normalizer = float(special.betaln(first_shape, marginal_second_shape))

    def log_conditional_probability(log_threshold: float) -> float:
        if log_threshold >= 0.0:
            return 0.0
        # Very near one, compute the complementary beta tail from log(1-z)
        # so exp(log_threshold) rounding to one cannot erase it.
        if log_threshold > -1e-8:
            log_complement = math.log(-math.expm1(log_threshold))
            log_upper = _log_beta_lower(log_complement, residual_shape, second_shape)
            if log_upper < 0.0:
                upper = math.exp(log_upper)
                if upper < 1.0:
                    return math.log1p(-upper)
        return _log_beta_lower(log_threshold, second_shape, residual_shape)

    def log_marginal_integrand(logit: float) -> float:
        log_value = -_softplus(-logit)
        log_complement = -_softplus(logit)
        return (
            first_shape * log_value
            + marginal_second_shape * log_complement
            - log_beta_normalizer
            + log_conditional_probability(log_ratio_logit(logit))
        )

    def log_integral(left: float, right: float) -> float:
        grid = np.linspace(left, right, 33, dtype=np.float64)
        grid_values = np.array(
            [log_marginal_integrand(float(point)) for point in grid],
            dtype=np.float64,
        )
        maximum_index = int(np.argmax(grid_values))
        maximum = float(grid_values[maximum_index])
        local_left = float(grid[max(0, maximum_index - 1)])
        local_right = float(grid[min(grid.size - 1, maximum_index + 1)])
        if local_left < local_right:
            optimum = optimize.minimize_scalar(
                lambda value: -log_marginal_integrand(float(value)),
                bounds=(local_left, local_right),
                method="bounded",
                options={"xatol": 1e-11},
            )
            if optimum.success:
                maximum = max(maximum, -float(optimum.fun))

        def scaled_integrand(logit: float) -> float:
            log_scaled = log_marginal_integrand(logit) - maximum
            if log_scaled < _LOG_SMALLEST:
                return 0.0
            return math.exp(log_scaled)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            absolute_error_floor = 128.0 * np.finfo(np.float64).eps
            integral, error = integrate.quad(
                scaled_integrand,
                left,
                right,
                epsabs=0.5 * absolute_error_floor,
                epsrel=2e-11,
                limit=300,
            )
        if integral <= 0.0 or not math.isfinite(integral):
            raise ArithmeticError("the exact likelihood-ratio integral is invalid")
        if error > max(absolute_error_floor, 2e-8 * integral):
            raise ArithmeticError(
                "the exact likelihood-ratio integral did not converge"
            )
        return maximum + math.log(integral)

    lower_log_value = -_softplus(-lower_logit)
    upper_log_complement = -_softplus(upper_logit)
    log_components = (
        _log_beta_lower(lower_log_value, first_shape, marginal_second_shape),
        _log_beta_lower(upper_log_complement, marginal_second_shape, first_shape),
        log_integral(lower_logit, midpoint_logit),
        log_integral(midpoint_logit, upper_logit),
    )
    log_pvalue = float(special.logsumexp(log_components))
    if not math.isfinite(log_pvalue):
        raise ArithmeticError("the exact likelihood-ratio probability is not finite")
    if log_pvalue < _LOG_SMALLEST:
        return 0.0
    return min(1.0, math.exp(log_pvalue))


def zxc_2samp(x: ArrayLike, y: ArrayLike) -> HypothesisTestResult:
    """Perform the exact Zhang--Xu--Chen two-sample normal test.

    The likelihood ratio itself is reported.  Its exact p-value is the lower
    tail because smaller ratios provide stronger evidence against equality.
    Computation stays in the log domain and uses the conditional-beta form of
    the null probability rather than exponentiating a two-dimensional
    quadrature integrand.
    """
    summary = _two_sample_summary(x, y)
    likelihood_ratio = _likelihood_ratio(summary.log_lambda)
    pvalue = _exact_lrt_pvalue(summary.log_lambda, summary.n, summary.m)
    return HypothesisTestResult(
        statistic=likelihood_ratio,
        pvalue=pvalue,
        method="Zhang-Xu-Chen exact two-sample normal-population test (2012)",
        alternative=_JOINT_ALTERNATIVE_2SAMP,
        data_name="x and y",
        statistic_name="Lambda",
        calibration="exact Dirichlet null distribution (lower tail)",
    )

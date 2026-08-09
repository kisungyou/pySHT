# Classical mean tests: formula and validation ledger

This ledger covers the first independently verifiable mean-testing slice in
`pysht.mean`. Legacy SHT output is used only as a regression comparator; the
formulas and SciPy distribution functions below define the implementation.

## Univariate t tests

For a one-sample test, pySHT reports

\[
t = \frac{\bar X-\mu_0}{S/\sqrt n}, \qquad \nu=n-1.
\]

For two independent samples it reports either the pooled-variance statistic
with \(\nu=n_x+n_y-2\), or Welch's statistic

\[
t = \frac{\bar X-\bar Y}
         {\sqrt{S_x^2/n_x+S_y^2/n_y}},
\]

with the Welch--Satterthwaite degrees of freedom. A paired test is the
one-sample test of the row-wise differences. P-values use the requested lower,
upper, or doubled symmetric t tail. Confidence intervals use the same degrees
of freedom and alternative.

The calculations use a common positive scale before forming means and
standard deviations. This leaves the t statistic unchanged and prevents
avoidable overflow for data near the limits of float64.

Validation evidence:

- all three alternatives agree with `scipy.stats.ttest_1samp`,
  `ttest_ind`, and `ttest_rel` on fixed non-degenerate fixtures;
- Welch and pooled branches are checked separately;
- the test remains defined when one independent group is constant and the
  other has positive variance;
- common scaling through approximately \(10^{200}\) preserves the statistic
  and p-value;
- zero standard error, non-finite values, undersized samples, and inconsistent
  paired designs fail explicitly.

## One-way ANOVA

For \(k\) groups and total size \(N\), the implementation forms

\[
F = \frac{SS_B/(k-1)}{SS_W/(N-k)}
\]

and uses the upper \(F_{k-1,N-k}\) tail. A shared positive scale is applied
before the sums of squares, so equality testing is invariant to measurement
units without squaring extremely large observations.

The fixed fixture agrees with `scipy.stats.f_oneway`; group means and both
degrees of freedom are retained in the result. A zero pooled within-group
variance is rejected because the stated F calibration is then undefined.

## Hotelling tests

The one-sample statistic is

\[
T^2=n(\bar X-\mu_0)^\mathsf{T}S^{-1}(\bar X-\mu_0),
\qquad
\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}.
\]

For independent samples with a shared covariance matrix,

\[
T^2=\frac{n_xn_y}{n_x+n_y}
(\bar X-\bar Y)^\mathsf{T}S_p^{-1}(\bar X-\bar Y),
\]

and

\[
\frac{n_x+n_y-p-1}{p(n_x+n_y-2)}T^2
\sim F_{p,n_x+n_y-p-1}.
\]

Covariance systems are evaluated through a Cholesky factor rather than an
explicit inverse. Every feature is positively rescaled before covariance
formation; Hotelling's statistic is invariant to this nonsingular diagonal
transformation. The implementation requires the sample-size conditions for
positive denominator degrees of freedom and a positive-definite covariance
estimate.

Validation evidence:

- fixed one- and two-sample values agree with literal matrix formulas;
- swapping independent groups preserves statistic and p-value;
- a nonsingular linear transformation preserves the one-sample result;
- the paired procedure agrees with a one-sample test of differences;
- singular covariance and insufficient-sample designs fail explicitly.

## Deliberate scope boundary

The independent two-sample Hotelling procedure currently exposes only the
equal-covariance exact test. Legacy SHT's unequal-covariance branch is an
approximate multivariate Behrens--Fisher procedure and will be exposed under a
method-specific name only after a separate paper-level derivation and
calibration study.

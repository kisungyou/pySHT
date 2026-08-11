# Data and assumptions

pySHT validates whether inputs can define a statistic, but software cannot
verify whether a sampling design justifies the reported null distribution.
Both parts of the contract matter. Passing validation means that a computation
is defined, not that its scientific assumptions hold.

## Accepted data shapes

Public procedures accept array-like, real numeric data representable as NumPy
`float64` values.

| Family | Expected shape |
|---|---|
| Univariate mean, variance, joint mean-and-variance, and normality | One-dimensional sample vectors |
| Multivariate mean, covariance, and joint mean-and-covariance | Observation-by-feature matrices |
| Equality of distributions | Two vectors or two observation-by-feature matrices with matching feature counts |
| Rectangular uniformity | One observation-by-coordinate matrix |
| Simplex uniformity | One observation-by-component matrix whose rows are compositions |

Every value must be finite. `NaN`, positive or negative infinity, complex
values, Boolean arrays, empty feature dimensions, and ragged or nonnumeric
inputs are rejected. Minimum sample sizes are method-specific and can exceed
two observations.

pySHT does not silently remove missing values, impute them, align labeled
records, or convert categorical variables. Perform those operations before
calling a test, and document how they affect the sampling design.

## Dependence and sampling units

Except in an explicit paired mode, rows are treated as independent sampling
units. Measurements from the same participant, household, site, batch, time
series, pedigree, or cluster generally are not independent merely because they
appear on separate rows.

For `paired=True`, row `i` of `x` is paired with row `i` of `y`, and the inputs
must contain the same number of observations. Pairing is appropriate only when
that row-wise correspondence comes from the design. Sorting or matching
records is the caller's responsibility.

The public API has no weights, blocks, strata, clusters, repeated-measures
model, or user-defined restricted permutation scheme. A method that accepts
several samples assumes those samples are mutually independent unless its
documentation says otherwise.

## Conditions behind calibration

| Family | Conditions to justify before using its reported calibration |
|---|---|
| Classical t, ANOVA, Hotelling, chi-square, F, and Bartlett tests | Independent normal observations in the analyzed unit; pooled procedures also require their stated common variance or covariance; paired procedures apply assumptions to paired differences |
| Levene and Brown--Forsythe | Independent groups; the F law for absolute deviations is an approximation |
| High-dimensional mean tests | The paper-specific moment, trace, sparsity, factor, aspect-ratio, and covariance conditions; these differ materially across procedures |
| Multivariate Behrens--Fisher tests | Independent multivariate samples, method-specific covariance rank, and the approximation used for effective degrees of freedom |
| Covariance tests | Independent rows and the paper-specific Gaussian or moment assumptions; pooled-inverse procedures additionally need an invertible pooled estimate; `covariance.lyl_2samp` uses a known-zero-mean model |
| One- and two-sample mean-and-variance tests | Independent observations from normal populations with positive within-sample variance |
| `mean_covariance.lrt_1samp` | Multivariate normality, more observations than features, and a positive-definite fitted covariance |
| `mean_covariance.llzs_1samp` and `hn_2samp` | Their high-dimensional moment, aspect-ratio, and trace conditions rather than a generic small-sample guarantee |
| Equality-of-distributions permutation test | Exchangeability of pooled observations under equality of the complete distributions |
| Normality tests | Independent identically distributed scalar observations under a composite normal null; moment tests use Monte Carlo by default because their chi-square law is only asymptotic |
| Rectangular-uniformity tests | Independent observations from a uniform law on the declared, fixed hyperrectangle |
| Simplex uniformity | Independent interior compositions under a Dirichlet `(1, ..., 1)` null; the likelihood-ratio p-value uses Wilks' approximation |

An asymptotic p-value is not an exact finite-sample guarantee. A method's
public availability means that its implementation and advertised regimes are
documented and tested; it does not extend the source theorem to arbitrary
`n`, `p`, tails, or data distributions.

## Exchangeability and auxiliary randomness

Exchangeability is stronger than having the same mean, variance, or
covariance. Under a two-sample permutation null, labels must be freely
interchangeable for the actual sampling design. Unrestricted relabeling is not
valid for paired, clustered, stratified, or serially dependent observations.

Random-projection and random-subspace tests introduce auxiliary randomness in
addition to sampling variation. pySHT draws the projection or subspace plan
once and holds it fixed across the observed and permuted statistics. Record
`rng`, `n_projections` or `n_subspaces`, and `n_resamples` when applicable.
Changing them can change a Monte Carlo result without changing the data.

## Support and boundary assumptions

- Rectangular-uniformity bounds are part of the null. `ym_interpoint` accepts
  observations on the declared boundary; `ym_quantile` requires strict
  interior points because endpoints map to infinite normal quantiles.
- Simplex rows must be nonnegative compositions summing to one within the
  documented numerical tolerance. Every component must be strictly positive;
  zeros are not perturbed or replaced.
- A supplied `popcov` must be finite, symmetric, positive definite, and have
  the same dimension as the observations. It is a fixed null quantity, not an
  estimate formed by the function.
- `covariance.lyl_2samp` uses observations as supplied in no-intercept
  conditional regressions. It does not silently center the groups; a zero mean
  is part of that published null model.

Estimating support bounds or null parameters from the same observations can
change the null distribution. Do so only when the chosen method explicitly
accounts for that estimation.

## Sample size, rank, and degeneracy

Mathematically undefined cases fail explicitly rather than returning a `NaN`
result.

- A one-sample t test needs positive sample variance; a paired t test needs
  positive variance of the row-wise differences.
- Welch's independent t test needs at least one positive sample variance. Its
  pooled counterpart and one-way ANOVA need positive pooled within-group
  variance.
- The two-sample F test and Bartlett's test need positive sample variance in
  every relevant group.
- Classical one-sample and paired Hotelling tests require `n > p`. The
  independent two-sample version requires sufficient pooled residual degrees
  of freedom. Every covariance matrix used in a quadratic solve must be
  positive definite.
- Trace and U-statistic estimators can require three or four observations per
  group even though their target matrices need not be invertible.
- Projection procedures still require positive projected variances; increasing
  the number of projections does not repair a degenerate sample.
- Joint mean-and-variance methods require positive within-sample variance.
- Normality statistics reject constant samples; Shapiro methods also enforce
  the sample-size range supported by their p-value approximations.
- The distribution-equality statistic needs at least two observations in each
  group and matching feature counts.
- Dirichlet alternatives can have an unbounded maximum at degenerate samples;
  optimizer nonconvergence and boundary optima fail loudly.

Consult the exact function documentation before treating these examples as a
complete set of limits.

## Transformations and numerical range

Several implementations center, whiten, rescale, solve linear systems, or use
logarithmic arithmetic to avoid unnecessary `float64` overflow and underflow.
These transformations preserve the intended statistic or its ordering. They
do not make an invalid sampling model valid, repair influential outliers, or
establish normality, sparsity, or exchangeability.

The mathematical invariances differ by test. A mean test may be translation
equivariant, a covariance-equality test may permit common scaling, and a test
against a fixed null covariance must transform that null together with the
data. Read the method ledger before applying a transformation as though it
were harmless.

Do not standardize, transform, trim, or remove observations solely to obtain a
preferred p-value. If preprocessing is scientifically justified, define it
before inspecting the outcome and include it in the reproducible analysis.

## Before calling a test

- State the null, alternative, sampling unit, and significance or evidence
  rule.
- Decide whether groups are independent or paired.
- Check dimensions, feature units, row correspondence, support, and rank.
- Justify normality, common dispersion, high-dimensional conditions,
  exchangeability, or domain assumptions as required.
- Choose asymptotic, exact, Monte Carlo, or Bayesian evidence deliberately.
- Plan missing-data handling, exclusions, transformations, random controls,
  and multiplicity adjustments in advance.

Use [Choose a test](choose-a-test.md) for the procedure map and the
[validation ledgers](../validation/index.md) for formula-level assumptions and
advertised regimes.

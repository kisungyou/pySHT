# [4] Tests for Covariance

The functions in `pysht.covariance` compare one covariance matrix with a
specified null, compare two population covariances, or test homogeneity across
several groups. Observations are rows and variables are columns. The
frequentist procedures treat location as a nuisance parameter and center each
group. The Lee--You--Lin Bayes-factor procedure instead follows its published
known-zero-mean model and uses observations exactly as supplied.

```{currentmodule} pysht.covariance
```

## Choosing a procedure

| Design and working assumptions | Function | R routine |
|---|---|---|
| One sample; Gaussian; randomized one-dimensional projections | `wl_1samp` | `cov1.2015WL` |
| Two samples; high-dimensional factor model with finite eighth moments; dense differences | `lc_2samp` | `cov2.2012LC` |
| Two samples; high-dimensional tails and sparse entrywise differences | `clx_2samp` | `cov2.2013CLX` |
| Two samples; Gaussian; randomized projected variance ratios | `wl_2samp` | `cov2.2015WL` |
| Two samples; known zero means; Gaussian; sparse conditional-regression differences; Bayesian evidence | `lyl_2samp` | `cov2.mxPBF` |
| Several samples; Gaussian; fixed dimension and invertible pooled covariance | `schott_2001_ksamp` | `covk.2001Schott` |
| Several samples; Gaussian; dimension grows with sample sizes | `schott_2007_ksamp` | `covk.2007Schott` |

These calibrations target different asymptotic regimes. A convenient function
name does not replace checking normality, dimensional growth, moment, or
invertibility assumptions. The [covariance validation
ledger](../validation/covariance.md) records the exact formulas, finite-sample
audits, and limitations.

## One-sample tests

The corrected Fisher implementation is withheld from 0.1.0 because its former
20,000-run evidence was not reproducible through the public path and smaller
fresh probes failed calibration. See the validation ledger; there is no public
`fisher_1samp` placeholder or fallback.

```{eval-rst}
.. autofunction:: wl_1samp
```

`wl_1samp` draws `n_projections` Gaussian unit vectors from a local generator.
An integer `rng` reproduces the projection plan without changing NumPy's
global random state. The statistic is the maximum absolute transformed
projected variance; both unusually small and unusually large variances count
against equality.

## Two-sample tests

```{eval-rst}
.. autofunction:: lc_2samp
```

pySHT evaluates every order-two, order-three, and order-four term in Li--Chen
Equations (2.1)--(2.2), using algebraically exact ordered-sum identities. It
needs at least four observations per group. A former centered leading-term
shortcut was removed before 0.1.0 because it was badly anti-conservative in
the advertised finite null regime.

```{eval-rst}
.. autofunction:: clx_2samp
```

The CLX procedure is a maximum test for sparse covariance differences. It
requires at least two variables and positive entrywise variance estimates.
The reported p-value uses its type-I extreme-value limit.

```{eval-rst}
.. autofunction:: wl_2samp
```

The same projection plan is applied to both samples. pySHT uses the absolute
log variance ratio and a two-sided maximum-normal probability, making the
answer invariant to exchanging the groups for a fixed seed. Its fresh null
gate passed at `N1=300, N2=360, p=30, n_projections=50`; the smaller
`N1=100, N2=120` regime failed and is not advertised as calibrated.

```{eval-rst}
.. autofunction:: lyl_2samp
```

This procedure returns a `BayesFactorTestResult`. Its primary statistic is the
maximum ordered-pair **log** Bayes factor; the complete immutable matrix is in
`component_log_bayes_factors`, with `-inf` on the diagonal. It has no p-value
and pySHT supplies no universal evidence cutoff. The published model assumes
known zero means, so pySHT neither centers the samples nor fits an intercept.
The paper's recommended defaults are `a0=b0=0.01` and
`gamma=max(n1+n2, p)**(-alpha)` with `alpha=2.01`; supplying `gamma` overrides
that rule. `b0` is a variance-scale hyperparameter, so it must be multiplied by
the square of any change in the data's physical units if the same prior is
intended. If population means are unknown, this paper-specific calibration is
not established; choose another covariance procedure or justify preprocessing
separately.

## Multi-sample tests

```{eval-rst}
.. autofunction:: schott_2001_ksamp
```

The 2001 Wald statistic requires a numerically positive-definite pooled
covariance. pySHT rejects a singular design instead of substituting a
pseudoinverse, because that would change the statistic and its chi-square
reference law.

```{eval-rst}
.. autofunction:: schott_2007_ksamp
```

The 2007 trace procedure permits singular group covariance matrices and is
intended for growing dimension. It requires at least three observations per
group so that every finite-sample correction is defined.

## Interpreting output

Frequentist functions return immutable `HypothesisTestResult` values. Small
p-values are evidence against covariance equality; the raw statistic is not
an effect size and should not be compared across methods. Wu--Li results also
report the number and distribution of random projections. Reuse the same
integer seed when an analysis must be replayed.

The LYL output is evidence on a different scale: positive log Bayes factors
favor a pairwise alternative over its null under the selected priors. It must
not be interpreted as `-log(p)`.

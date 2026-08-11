# Choose a test

Choose a procedure from the scientific question and sampling design before
looking at the observed results. Start by naming the population quantity in
the null hypothesis, then determine the number of samples, whether observations
are independent or paired, and whether each observation is scalar,
multivariate, rectangular, or compositional.

## Start with the null hypothesis

| Null concerns | Data and design | API category |
|---|---|---|
| One or more scalar means | One, two, or several univariate samples | [Univariate mean](../api/univariate-mean.md) |
| One or more mean vectors | One, two, or several multivariate samples | [Multivariate mean](../api/multivariate-mean.md) |
| One or more scalar variances | One, two, or several univariate samples | [Variance](../api/variance.md) |
| One or more covariance matrices | One, two, or several multivariate samples | [Covariance](../api/covariance.md) |
| Mean and variance together | One or two univariate normal samples | [Mean and variance](../api/mean-variance.md) |
| Mean vector and covariance together | One or two multivariate samples | [Mean and covariance](../api/mean-covariance.md) |
| An entire distribution | Two independent samples | [Equality of distributions](../api/equaldist.md) |
| Fit to a normal distribution | One univariate sample | [Normality](../api/normality.md) |
| Fit to a rectangular uniform distribution | One multivariate sample | [Rectangular uniformity](../api/uniformity.md) |
| Fit to uniformity on a probability simplex | One compositional sample | [Simplex uniformity](../api/simplex.md) |

A test of a mean, variance, or covariance answers a narrower question than a
test of an entire distribution. Failure to reject one equality does not
establish another. A joint test answers whether at least one component of its
joint null fails; it does not identify which component changed.

## Scalar parameters and goodness of fit

| Question and design | Procedure choices | Main distinction |
|---|---|---|
| One scalar mean | `ttest_1samp` | Normal-theory one-sample inference |
| Two scalar means | `ttest_2samp` | Welch is the independent-sample default; pooled and paired modes encode different designs |
| Several scalar means | `anova_oneway` | Classical independent-groups ANOVA with a common variance |
| One scalar variance | `chisquare_1samp` | Normal-theory test against a specified variance |
| Two scalar variances | `f_2samp` | Normal-theory variance-ratio test |
| Several scalar variances or spreads | `bartlett`, `levene`, `brown_forsythe` | Bartlett is normal-theory; Levene and Brown--Forsythe use deviations from means and medians |
| One mean and variance jointly | `as_1samp` | Asymptotic likelihood-ratio test against specified normal parameters |
| Two means and variances jointly | `pn_2samp`, `pl_2samp`, `muirhead_2samp`, `zxc_2samp`, `lrt_2samp` | Respectively beta approximation, combined component tests, corrected approximation, exact calibration, and asymptotic LRT |
| Univariate normality | `shapiro_wilk`, `shapiro_francia`, `jarque_bera`, `adjusted_jarque_bera`, `robust_jarque_bera` | Shapiro tests use order statistics; moment tests target skewness and kurtosis and default to finite-sample Monte Carlo calibration |

Choose a joint mean-and-variance test only when the scientific null really
specifies both parameters. Do not use it as a substitute for inspecting which
parameter matters. Likewise, select among the normality tests from the kinds of
departures and sample-size regime relevant to the analysis, not by reporting
the smallest of several p-values.

## Multivariate means

Here `n` denotes sample size and `p` the number of features.

| Design or regime | Procedure choices | Important condition |
|---|---|---|
| Classical one-sample mean vector | `hotelling_1samp` | Requires `n > p` and an invertible covariance estimate |
| Classical two-sample or paired mean vector | `hotelling_2samp` | Independent mode uses a common covariance; the fitted covariance must be invertible |
| Dense shift when covariance inversion is unsuitable | `dempster_1samp`, `dempster_2samp`, `bs_1samp`, `bs_2samp` | Use the paper-specific dimensional and covariance regimes |
| Coordinate-standardized dense shift | `sd_1samp`, `sd_2samp` | Requires usable marginal variance estimates |
| Unequal-covariance two-sample mean vector | `yao_2samp`, `johansen_2samp`, `nvm_2samp`, `ky_2samp` | Low-dimensional Behrens--Fisher approximations with method-specific degrees of freedom |
| Randomized high-dimensional comparison | `ljw_2samp`, `thulin_2samp` | Record projection or subspace controls and `rng`; Monte Carlo mode fixes auxiliary randomness across permutations |
| Coordinatewise Bayesian evidence | `lyl_2samp` | Returns maximum and component log Bayes factors, not a p-value |
| Several groups | `schott_ksamp`, `zx_ksamp`, `cph_ksamp` | Match common- versus unequal-covariance assumptions and the advertised dimensional regime |

High-dimensional does not mean assumption-free. Trace, diagonal,
random-projection, sparse, and factor-model tests target different alternatives
and require different spectral or moment conditions. Use the method-specific
[validation ledger](../validation/index.md) before selecting one from sample
size and dimension alone.

## Covariance and joint multivariate parameters

| Question and regime | Procedure choices | Main distinction |
|---|---|---|
| One covariance against a specified matrix | `wl_1samp` | Random-projection method; record `rng` and verify the advertised Gaussian regime |
| Two high-dimensional covariances | `lc_2samp`, `clx_2samp`, `wl_2samp` | Li--Chen targets a global Frobenius departure, CLX a maximum standardized entry, and Wu--Li projected variance ratios |
| Two covariances, Bayesian evidence | `lyl_2samp` | Uses the paper's known-zero-mean Gaussian model and returns conditional-regression log Bayes-factor evidence without a universal threshold |
| Several covariance matrices | `schott_2001_ksamp`, `schott_2007_ksamp` | The 2001 procedure uses a pooled inverse; the 2007 procedure targets a high-dimensional regime |
| One mean vector and covariance against specified values | `llzs_1samp`, `lrt_1samp` | LLZS is high-dimensional; the LRT is fixed-dimensional and needs an invertible fitted covariance |
| Two mean vectors and covariances jointly | `hn_2samp` | High-dimensional joint test under its source paper's moment and trace conditions |

The covariance CLX procedure is public, but Fisher's covariance procedure and
SHT's distinct Cai--Liu--Xia mean
test is validation-blocked and therefore does not appear in the selection
table. The two public `lyl_2samp` functions are likewise distinct procedures;
module qualification is part of each public name. In particular, the
covariance version does not estimate or remove a mean: center observations
externally only when a scientific design justifies treating the resulting
values as observations from the paper's fixed zero-mean model.

## Distributional and structured-domain questions

| Null hypothesis | Procedure | Design boundary |
|---|---|---|
| Two independent samples have the same distribution | `equaldist.bg_2samp` | Pooled observations must be exchangeable; use exact enumeration when feasible or corrected Monte Carlo calibration |
| Observations are uniform on declared rectangular bounds | `uniformity.ym_interpoint` | Monte Carlo is the default calibration for `q1`, `q2`, and `q3`; boundary points are allowed |
| Observations are uniform on declared rectangular bounds | `uniformity.ym_quantile` | Every coordinate must be strictly inside its bounds before the normal-quantile transform |
| Compositions are uniform on a probability simplex | `simplex.uniformity` | Rows must lie in the strict simplex interior; choose a symmetric or general Dirichlet alternative |

The bounds in a rectangular-uniformity test and the simplex itself are part of
the null model. Estimating support limits from the same data changes that
model and is not handled automatically.

## A practical decision sequence

1. **Write the null and alternative in words.** Decide whether the target is a
   mean, dispersion parameter, joint model, full distribution, or domain-
   specific uniform law.
2. **Identify sampling units and dependence.** Matched observations require an
   explicit paired procedure. Unrestricted tests here do not model clusters,
   repeated measures, or time dependence.
3. **Determine the dimensional regime.** Check `n`, `p`, covariance rank, group
   balance, and whether the reference is fixed-dimensional, high-dimensional,
   exact, asymptotic, or simulation based.
4. **Match the alternative.** Global squared-distance, maximum-coordinate,
   randomized-projection, and Bayes-factor methods can have very different
   power against the same broad null.
5. **Choose calibration and random controls in advance.** When using Monte
   Carlo calibration, set `n_resamples` from the required precision and pass a
   reproducible `rng` when exact replay matters.
6. **Read the result according to its contract.** A p-value, a Monte Carlo
   estimate with uncertainty, and a log Bayes factor are not interchangeable.

## Call the selected procedure

Functions return a result object; they do not print or make a reject/retain
decision automatically. Supply design choices explicitly and retain the
result for both reporting and programmatic use:

```python
from pysht.mean import ttest_2samp

result = ttest_2samp(
    treatment,
    control,
    alternative="two-sided",
    equal_var=False,
)

print(result)
```

This requests Welch's independent-samples t test. Setting `equal_var=True`
changes the statistical model; setting `paired=True` changes the sampling
design. Arguments should follow the study design rather than the observed
data. Consult the category page in the [API reference](../api/index.md) for
exact signatures and defaults.

## Scope boundaries

The SHT compatibility catalog does not provide regression, generalized linear
models, survival methods, repeated-measures models, survey weights, arbitrary
missing-data handling, user-defined restricted permutations, or automatic
multiple-testing adjustments. Use a library designed for those models rather
than substituting a superficially similar standalone test.

Before analysis, review the [data and assumptions guide](data-assumptions.md).
After computing a test, use the [result-object guide](results.md) to interpret
what was returned.

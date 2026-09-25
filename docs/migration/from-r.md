# Migrating from SHT for R

pySHT is an independently validated successor to SHT, not a line-by-line
translation. The compatibility baseline is SHT 0.1.9 at commit
`4e29cda1257f86dd0237d37329af358b54d04f2b`. A primary paper and the stated
hypotheses take precedence whenever the legacy implementation disagrees with
the mathematics.

The pinned R namespace contains 56 exports: 54 statistical routine identities
and two dynamic adapters. The pySHT compatibility subset has 51 canonical functions
covering 52 of those 54 identities: `mvar1.1998AS` and `mvar1.LRT` share one
algebraically identical implementation, while `mean2.2014CLX` and
`cov1.2012Fisher` are validation-blocked and have no public Python mapping. The
adapters are unnecessary in Python.

This page intentionally remains limited to the SHT compatibility target. The
21 validated research methods added by pySHT are listed separately in the
[pySHT-native method catalog](../methods/native-methods.md), so a reader can
distinguish migration coverage from new scientific functionality.

```{index} single: migration from R
```
```{index} single: SHT 0.1.9; function crosswalk
```

## Complete statistical crosswalk

### [1] Tests for univariate mean

| SHT 0.1.9 | pySHT |
|---|---|
| `mean1.ttest` | `pysht.mean.ttest_1samp` |
| `mean2.ttest` | `pysht.mean.ttest_2samp` |
| `meank.anova` | `pysht.mean.anova_oneway` |

### [2] Tests for multivariate mean

| SHT 0.1.9 | pySHT |
|---|---|
| `mean1.1931Hotelling` | `pysht.mean.hotelling_1samp` |
| `mean1.1958Dempster` | `pysht.mean.dempster_1samp` |
| `mean1.1996BS` | `pysht.mean.bs_1samp` |
| `mean1.2008SD` | `pysht.mean.sd_1samp` |
| `mean2.1931Hotelling` | `pysht.mean.hotelling_2samp` |
| `mean2.1958Dempster` | `pysht.mean.dempster_2samp` |
| `mean2.1965Yao` | `pysht.mean.yao_2samp` |
| `mean2.1980Johansen` | `pysht.mean.johansen_2samp` |
| `mean2.1986NVM` | `pysht.mean.nvm_2samp` |
| `mean2.1996BS` | `pysht.mean.bs_2samp` |
| `mean2.2004KY` | `pysht.mean.ky_2samp` |
| `mean2.2008SD` | `pysht.mean.sd_2samp` |
| `mean2.2011LJW` | `pysht.mean.ljw_2samp` |
| `mean2.2014CLX` | Validation-blocked; no public pySHT callable |
| `mean2.2014Thulin` | `pysht.mean.thulin_2samp` |
| `mean2.mxPBF` | `pysht.mean.maximum_pairwise_bayes_factor_2samp` |
| `meank.2007Schott` | `pysht.mean.schott_ksamp` |
| `meank.2009ZX` | `pysht.mean.zx_ksamp` |
| `meank.2019CPH` | `pysht.mean.cph_ksamp` |

### [3] Tests for variance

| SHT 0.1.9 | pySHT |
|---|---|
| `var1.chisq` | `pysht.variance.chisquare_1samp` |
| `var2.F` | `pysht.variance.f_2samp` |
| `vark.1937Bartlett` | `pysht.variance.bartlett` |
| `vark.1960Levene` | `pysht.variance.levene` |
| `vark.1974BF` | `pysht.variance.brown_forsythe` |

### [4] Tests for covariance

| SHT 0.1.9 | pySHT |
|---|---|
| `cov1.2012Fisher` | Withheld: corrected private implementation has not passed reproducible release gates |
| `cov1.2015WL` | `pysht.covariance.wl_1samp` |
| `cov2.2012LC` | `pysht.covariance.lc_2samp` |
| `cov2.2013CLX` | `pysht.covariance.clx_2samp` |
| `cov2.2015WL` | `pysht.covariance.wl_2samp` |
| `cov2.mxPBF` | `pysht.covariance.maximum_pairwise_bayes_factor_2samp` (published known-zero-mean model) |
| `covk.2001Schott` | `pysht.covariance.schott_2001_ksamp` |
| `covk.2007Schott` | `pysht.covariance.schott_2007_ksamp` |

The exported-but-pkgdown-hidden `cov2.mxPBF` is included because it is in the
0.1.9 namespace. The source-only `cov1.mxPBF` is excluded: it is marked
`@noRd`, absent from `NAMESPACE`, and is not one of the 54 public tests.

### [5] Simultaneous tests for mean and variance

| SHT 0.1.9 | pySHT | Note |
|---|---|---|
| `mvar1.1998AS` | `pysht.mean_variance.lrt_1samp` | Arnold--Shavelle form |
| `mvar1.LRT` | `pysht.mean_variance.lrt_1samp` | Same algebra as the preceding R routine |
| `mvar2.1930PN` | `pysht.mean_variance.pn_2samp` | Lower likelihood-ratio tail corrected |
| `mvar2.1976PL` | `pysht.mean_variance.pl_2samp` | Fisher combination |
| `mvar2.1982Muirhead` | `pysht.mean_variance.muirhead_2samp` | Upper rejection tail corrected |
| `mvar2.2012ZXC` | `pysht.mean_variance.exact_lrt_2samp` | Stable exact calculation |
| `mvar2.LRT` | `pysht.mean_variance.lrt_2samp` | Asymptotic LRT |

### [6] Simultaneous tests for mean and covariance

| SHT 0.1.9 | pySHT |
|---|---|
| `sim1.2017Liu` | `pysht.mean_covariance.llzs_1samp` |
| `sim1.LRT` | `pysht.mean_covariance.lrt_1samp` |
| `sim2.2018HN` | `pysht.mean_covariance.hn_2samp` |

### [7] Tests for equality of distributions

| SHT 0.1.9 | pySHT |
|---|---|
| `eqdist.2014BG` | `pysht.equaldist.bg_2samp` |

Only exact and corrected Monte Carlo permutation calibration are available.
The invalid legacy asymptotic branch is outside the compatibility target.

### [8] Goodness-of-fit: normal distribution

| SHT 0.1.9 | pySHT |
|---|---|
| `norm.1965SW` | `pysht.normality.shapiro_wilk` |
| `norm.1972SF` | `pysht.normality.shapiro_francia` |
| `norm.1980JB` | `pysht.normality.jarque_bera` |
| `norm.1996AJB` | `pysht.normality.adjusted_jarque_bera` |
| `norm.2008RJB` | `pysht.normality.robust_jarque_bera` |

### [9] Goodness-of-fit: uniform distribution

| SHT 0.1.9 | pySHT |
|---|---|
| `unif.2017YMi` | `pysht.uniformity.ym_interpoint` |
| `unif.2017YMq` | `pysht.uniformity.ym_quantile` |

### [10] Tests on special domains

| SHT 0.1.9 | pySHT |
|---|---|
| `simplex.uniform` | `pysht.simplex.uniformity` |

## The two adapter exports

`usek1d` and `useknd` accept an R function and a list of samples. Python users
call an ordinary callable directly and pass multi-group samples positionally,
so pySHT deliberately provides no adapter aliases:

```python
from pysht.variance import brown_forsythe

result = brown_forsythe(group_a, group_b, group_c)
```

## Argument translation

| R convention | Python convention |
|---|---|
| `mu0` | `popmean` |
| `Sigma0` | `popcov` |
| `var0` | `variance` |
| `var.equal` | `equal_var` |
| `alternative="two.sided"` | `alternative="two-sided"` |
| `dlist=list(x, y, z)` | positional groups: `function(x, y, z)` |
| `method`, `nreps` | descriptive lowercase options, `calibration`, `n_resamples` |
| ambient R random state | `rng=None`, an integer seed, or a NumPy `Generator` |
| `m` for random projections | `n_projections` or `n_subspaces`, as applicable |
| `nthreads` | no counterpart; the current runtime is NumPy/SciPy only |
| matrices `X`, `Y` | arrays `x`, `y`; rows remain observations |

Short R option initials such as `"L"` and `"T"` are not accepted. Use the
documented lowercase word, such as `"bai-saranadasa"`, `"hotelling"`,
`"clime"`, or `"monte-carlo"`. Public controls are keyword-only when a
positional value would be ambiguous.

## Result translation

| R `htest` component | pySHT field |
|---|---|
| `statistic` | `result.statistic` |
| `p.value` | `result.pvalue` or the read-only `result.p_value` alias |
| `alternative` | `result.alternative` |
| `method` | `result.method` |
| `data.name` | `result.data_name` |
| named parameters | `result.df`, `result.estimates`, or `result.diagnostics` |
| `conf.int` | `result.confidence_interval`, `result.confidence_level` |

`print(result)` gives an `htest`-style summary, but the Python object is
immutable. Resampling results retain counts, exactness, Monte Carlo standard
error, and a binomial tail-probability interval.

The two LYL functions return `BayesFactorTestResult`, not an object pretending
to be a frequentist `htest`. They expose the maximum and component **log** Bayes
factors and intentionally have no p-value. pySHT never exponentiates the
maximum internally or supplies an automatic evidence threshold. Their default
`gamma` is derived from `alpha=2.01`, sample size, and dimension. Unlike the
legacy covariance kernel's implicit centering, the Python covariance function
uses the primary paper's known-zero-mean, no-intercept model.

## Deliberate correctness differences

- The corrected Fisher covariance implementation whitens by the supplied null
  covariance, but remains private until reproducible null and power gates pass;
  the legacy code calculated that transform and then ignored it.
- Li--Chen uses literal scale-equivariant U-statistics.
- Wu--Li equality tests account for both directions instead of applying a
  one-sided maximum to a two-sided hypothesis.
- Pearson--Neyman and Muirhead use the rejection tails implied by their
  likelihood-ratio statistics.
- Cao--Park--He evaluates each group in both variance-estimator branches rather
  than reusing a stale loop index.
- Adjusted and robust Jarque--Bera default calls are defined, and finite-sample
  Monte Carlo calibration reports its simulation uncertainty.
- Random projections and subspaces are drawn once and held fixed across the
  observed and permuted statistics.
- Zhang--Xu--Chen and all Bayes-factor kernels use log-domain arithmetic where
  direct exponentiation would underflow or overflow.
- Biswas--Ghosh uses floating-point tie handling, exact enumeration when
  feasible, stable distance normalization, and no legacy asymptotic branch.

See the [method-name glossary](../methods/names-and-acronyms.md) for expanded
surnames and the [validation ledgers](../validation/index.md) for the evidence
behind these departures.

# API reference

The scientific categories below follow the order used by the
[SHT reference](https://www.kisungyou.com/SHT/reference/index.html). The public
API contains 51 canonical pySHT functions covering 52 of the 54 statistical
routine identities in SHT 0.1.9. Two one-sample mean-and-variance identities
share one algebraically identical implementation; `mean2.2014CLX` remains a
private validation target after failing its practical-size calibration gate,
and `cov1.2012Fisher` is withheld because its release audit is not reproducible.
Every public Python function uses lowercase `snake_case` and lives in its
subject module; test functions are not re-exported by `pysht`.

## [0] Utilities

SHT's dynamic `usek1d` and `useknd` adapters have no direct pySHT counterpart.
Python functions can be passed and called directly.

## [1] Tests for Univariate Mean

| Function | Purpose |
|---|---|
| [`pysht.mean.ttest_1samp`](univariate-mean.md) | Test one population mean against a reference value |
| [`pysht.mean.ttest_2samp`](univariate-mean.md) | Compare two independent or paired population means |
| [`pysht.mean.anova_oneway`](univariate-mean.md) | Compare means across two or more independent groups |

## [2] Tests for Multivariate Mean

| Function | Purpose |
|---|---|
| [`pysht.mean.hotelling_1samp`](multivariate-mean.md) | One-sample Hotelling mean-vector test |
| [`pysht.mean.hotelling_2samp`](multivariate-mean.md) | Independent or paired Hotelling mean-vector test |
| [`pysht.mean.dempster_1samp`](multivariate-mean.md) | One-sample Dempster test |
| [`pysht.mean.bs_1samp`](multivariate-mean.md) | One-sample Bai--Saranadasa test |
| [`pysht.mean.sd_1samp`](multivariate-mean.md) | One-sample Srivastava--Du test |
| [`pysht.mean.dempster_2samp`](multivariate-mean.md) | Two-sample Dempster test |
| [`pysht.mean.yao_2samp`](multivariate-mean.md) | Yao unequal-covariance test |
| [`pysht.mean.johansen_2samp`](multivariate-mean.md) | Johansen unequal-covariance test |
| [`pysht.mean.nvm_2samp`](multivariate-mean.md) | Nel--Van der Merwe unequal-covariance test |
| [`pysht.mean.bs_2samp`](multivariate-mean.md) | Two-sample Bai--Saranadasa test |
| [`pysht.mean.ky_2samp`](multivariate-mean.md) | Krishnamoorthy--Yu unequal-covariance test |
| [`pysht.mean.sd_2samp`](multivariate-mean.md) | Two-sample Srivastava--Du test |
| [`pysht.mean.ljw_2samp`](multivariate-mean.md) | Lopes--Jacob--Wainwright random-projection test |
| [`pysht.mean.thulin_2samp`](multivariate-mean.md) | Thulin random-subspace test |
| [`pysht.mean.lyl_2samp`](multivariate-mean.md) | Lee--You--Lin maximum log Bayes-factor test |
| [`pysht.mean.schott_ksamp`](multivariate-mean.md) | Schott multi-sample mean test |
| [`pysht.mean.zx_ksamp`](multivariate-mean.md) | Zhang--Xu transformed multi-sample test |
| [`pysht.mean.cph_ksamp`](multivariate-mean.md) | Cao--Park--He high-dimensional multi-sample test |

## [3] Tests for Variance

| Function | Purpose |
|---|---|
| [`pysht.variance.chisquare_1samp`](variance.md) | Test one population variance against a reference value |
| [`pysht.variance.f_2samp`](variance.md) | Compare two normal-population variances |
| [`pysht.variance.bartlett`](variance.md) | Compare variances across normal populations |
| [`pysht.variance.levene`](variance.md) | Compare group spreads around their means |
| [`pysht.variance.brown_forsythe`](variance.md) | Compare group spreads around their medians |

## [4] Tests for Covariance

| Function | Purpose |
|---|---|
| [`pysht.covariance.wl_1samp`](covariance.md) | Random-projection one-sample covariance test |
| [`pysht.covariance.lc_2samp`](covariance.md) | Li--Chen two-sample covariance test |
| [`pysht.covariance.clx_2samp`](covariance.md) | Cai--Liu--Xia maximum covariance test |
| [`pysht.covariance.wl_2samp`](covariance.md) | Random-projection two-sample covariance test |
| [`pysht.covariance.lyl_2samp`](covariance.md) | Lee--You--Lin known-zero-mean maximum log Bayes-factor test |
| [`pysht.covariance.schott_2001_ksamp`](covariance.md) | Schott (2001) multi-sample covariance test |
| [`pysht.covariance.schott_2007_ksamp`](covariance.md) | Schott (2007) high-dimensional multi-sample test |

## [5] Simultaneous Tests for Mean and Variance

| Function | Purpose |
|---|---|
| [`pysht.mean_variance.as_1samp`](mean-variance.md) | Test one normal mean and variance jointly |
| [`pysht.mean_variance.pn_2samp`](mean-variance.md) | Pearson--Neyman two-sample joint test |
| [`pysht.mean_variance.pl_2samp`](mean-variance.md) | Perng--Littell component-test combination |
| [`pysht.mean_variance.muirhead_2samp`](mean-variance.md) | Muirhead corrected likelihood-ratio approximation |
| [`pysht.mean_variance.zxc_2samp`](mean-variance.md) | Zhang--Xu--Chen exact likelihood-ratio calibration |
| [`pysht.mean_variance.lrt_2samp`](mean-variance.md) | Asymptotic two-sample likelihood-ratio test |

## [6] Simultaneous Tests for Mean and Covariance

| Function | Purpose |
|---|---|
| [`pysht.mean_covariance.llzs_1samp`](mean-covariance.md) | Liu--Liu--Zheng--Shi high-dimensional one-sample test |
| [`pysht.mean_covariance.lrt_1samp`](mean-covariance.md) | Classical one-sample likelihood-ratio test |
| [`pysht.mean_covariance.hn_2samp`](mean-covariance.md) | Hyodo--Nishiyama high-dimensional two-sample test |

## [7] Tests for Equality of Distributions

| Function | Purpose |
|---|---|
| [`pysht.equaldist.bg_2samp`](equaldist.md) | Compare two univariate or multivariate distributions |

## [8] Goodness-of-Fit: Normal Distribution

| Function | Purpose |
|---|---|
| [`pysht.normality.shapiro_wilk`](normality.md) | Shapiro--Wilk normality test |
| [`pysht.normality.shapiro_francia`](normality.md) | Shapiro--Francia normality test |
| [`pysht.normality.jarque_bera`](normality.md) | Jarque--Bera moment test |
| [`pysht.normality.adjusted_jarque_bera`](normality.md) | Adjusted Jarque--Bera moment test |
| [`pysht.normality.robust_jarque_bera`](normality.md) | Robust Jarque--Bera moment test |

## [9] Goodness-of-Fit: Uniform Distribution

| Function | Purpose |
|---|---|
| [`pysht.uniformity.ym_interpoint`](uniformity.md) | Yang--Modarres interpoint-distance test |
| [`pysht.uniformity.ym_quantile`](uniformity.md) | Yang--Modarres normal-quantile test |

## [10] Tests on Special Domains

| Function | Purpose |
|---|---|
| [`pysht.simplex.uniformity`](simplex.md) | Likelihood-ratio test of probability-simplex uniformity |

## Result objects

Every public test returns an immutable
[`StatisticalTestResult`](results.md) subtype. Frequentist tests report an
`htest`-style statistic and p-value; resampling results add exactness and
Monte Carlo diagnostics. The two Lee--You--Lin functions instead return a
maximum log Bayes factor and have no p-value.

For R names, years, and expanded author acronyms, use the
[migration crosswalk](../migration/from-r.md) and searchable
[method-name glossary](../methods/names-and-acronyms.md).

```{toctree}
:hidden:
:maxdepth: 1

univariate-mean
multivariate-mean
variance
covariance
mean-variance
mean-covariance
equaldist
normality
uniformity
simplex
results
../methods/index
```

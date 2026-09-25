# API reference

Categories [0]--[10] preserve the order used by the
[SHT reference](https://www.kisungyou.com/SHT/reference/index.html). Categories
[11]--[12] contain pySHT-native independence and circular-data methods. The R
crosswalk remains limited to SHT identities; native additions have their own
[catalog](../methods/native-methods.md). The candidate catalog contains 72
canonical public functions, including 21 validated pySHT-native additions.
Every public function uses lowercase
`snake_case`, lives in its scientific module, and is not re-exported from
top-level `pysht`.

## [0] Utilities

SHT's `usek1d` and `useknd` adapters have no Python counterpart. Python
functions can be passed and called directly.

## [1] Tests for Univariate Mean

| Function | Purpose |
|---|---|
| [`pysht.mean.ttest_1samp`](univariate-mean.md) | One population mean |
| [`pysht.mean.ttest_2samp`](univariate-mean.md) | Two independent or paired means |
| [`pysht.mean.anova_oneway`](univariate-mean.md) | Two or more independent means |

## [2] Tests for Multivariate Mean

| Function | Purpose |
|---|---|
| [`pysht.mean.hotelling_1samp`](multivariate-mean.md) | One-sample Hotelling test |
| [`pysht.mean.hotelling_2samp`](multivariate-mean.md) | Independent or paired Hotelling test |
| [`pysht.mean.dempster_1samp`](multivariate-mean.md), [`dempster_2samp`](multivariate-mean.md) | Dempster mean tests |
| [`pysht.mean.bs_1samp`](multivariate-mean.md), [`bs_2samp`](multivariate-mean.md) | Bai--Saranadasa tests |
| [`pysht.mean.sd_1samp`](multivariate-mean.md), [`sd_2samp`](multivariate-mean.md) | Srivastava--Du tests |
| [`pysht.mean.yao_2samp`](multivariate-mean.md) | Yao unequal-covariance test |
| [`pysht.mean.johansen_2samp`](multivariate-mean.md) | Johansen unequal-covariance test |
| [`pysht.mean.nvm_2samp`](multivariate-mean.md) | Nel--Van der Merwe test |
| [`pysht.mean.ky_2samp`](multivariate-mean.md) | Krishnamoorthy--Yu test |
| [`pysht.mean.ljw_2samp`](multivariate-mean.md) | Random-projection test |
| [`pysht.mean.thulin_2samp`](multivariate-mean.md) | Random-subspace test |
| [`pysht.mean.cq_2samp`](multivariate-mean.md) | Chen--Qin dense high-dimensional test |
| [`pysht.mean.li_1samp`](multivariate-mean.md), [`li_2samp`](multivariate-mean.md), [`li_ksamp`](multivariate-mean.md) | Li fixed-small-sample tests |
| [`pysht.mean.maximum_pairwise_bayes_factor_2samp`](multivariate-mean.md) | Lee--You--Lin maximum pairwise log Bayes factor |
| [`pysht.mean.schott_ksamp`](multivariate-mean.md) | Schott multi-sample test |
| [`pysht.mean.zx_ksamp`](multivariate-mean.md) | Zhang--Xu transformed test |
| [`pysht.mean.cph_ksamp`](multivariate-mean.md) | Cao--Park--He test |

## [3] Tests for Variance

| Function | Purpose |
|---|---|
| [`pysht.variance.chisquare_1samp`](variance.md) | One normal-population variance |
| [`pysht.variance.f_2samp`](variance.md) | Two normal-population variances |
| [`pysht.variance.bartlett`](variance.md) | Normal-population variance homogeneity |
| [`pysht.variance.levene`](variance.md) | Spread around group means |
| [`pysht.variance.brown_forsythe`](variance.md) | Spread around group medians |

## [4] Tests for Covariance

| Function | Purpose |
|---|---|
| [`pysht.covariance.wl_1samp`](covariance.md), [`wl_2samp`](covariance.md) | Wu--Li random-projection tests |
| [`pysht.covariance.czz_identity_1samp`](covariance.md) | Chen--Zhang--Zhong identity test |
| [`pysht.covariance.czz_sphericity_1samp`](covariance.md) | Chen--Zhang--Zhong sphericity test |
| [`pysht.covariance.lc_2samp`](covariance.md) | Li--Chen dense covariance test |
| [`pysht.covariance.clx_2samp`](covariance.md) | Cai--Liu--Xia sparse maximum test |
| [`pysht.covariance.maximum_pairwise_bayes_factor_2samp`](covariance.md) | Lee--You--Lin known-zero-mean log Bayes factor |
| [`pysht.covariance.schott_2001_ksamp`](covariance.md) | Fixed-dimensional covariance homogeneity |
| [`pysht.covariance.schott_2007_ksamp`](covariance.md) | High-dimensional covariance homogeneity |

## [5] Simultaneous Tests for Mean and Variance

| Function | Purpose |
|---|---|
| [`pysht.mean_variance.lrt_1samp`](mean-variance.md) | One normal mean and variance jointly |
| [`pysht.mean_variance.pn_2samp`](mean-variance.md) | Pearson--Neyman approximation |
| [`pysht.mean_variance.pl_2samp`](mean-variance.md) | Perng--Littell component combination |
| [`pysht.mean_variance.muirhead_2samp`](mean-variance.md) | Corrected likelihood-ratio approximation |
| [`pysht.mean_variance.exact_lrt_2samp`](mean-variance.md) | Exact Zhang--Xu--Chen calibration |
| [`pysht.mean_variance.lrt_2samp`](mean-variance.md) | Asymptotic two-sample LRT |

## [6] Simultaneous Tests for Mean and Covariance

| Function | Purpose |
|---|---|
| [`pysht.mean_covariance.llzs_1samp`](mean-covariance.md) | High-dimensional one-sample joint test |
| [`pysht.mean_covariance.lrt_1samp`](mean-covariance.md) | Classical one-sample joint LRT |
| [`pysht.mean_covariance.hn_2samp`](mean-covariance.md) | High-dimensional two-sample joint test |

## [7] Tests for Equality of Distributions

| Function | Purpose |
|---|---|
| [`pysht.equaldist.bg_2samp`](equaldist.md) | Biswas--Ghosh two-sample test |
| [`pysht.equaldist.energy_ksamp`](equaldist.md) | DISCO/energy multi-sample test |
| [`pysht.equaldist.mmd_2samp`](equaldist.md) | Characteristic-kernel MMD test |

## [8] Goodness-of-Fit: Normal Distribution

| Function | Purpose |
|---|---|
| [`pysht.normality.shapiro_wilk`](normality.md), [`shapiro_francia`](normality.md) | Shapiro normality tests |
| [`pysht.normality.jarque_bera`](normality.md), [`adjusted_jarque_bera`](normality.md), [`robust_jarque_bera`](normality.md) | Moment normality tests |
| [`pysht.normality.henze_zirkler`](normality.md) | Multivariate Henze--Zirkler test |
| [`pysht.normality.energy`](normality.md) | Multivariate energy test |

## [9] Goodness-of-Fit: Uniform Distribution

| Function | Purpose |
|---|---|
| [`pysht.uniformity.ym_interpoint`](uniformity.md) | Yang--Modarres interpoint test |
| [`pysht.uniformity.ym_quantile`](uniformity.md) | Yang--Modarres quantile test |
| [`pysht.uniformity.ehy`](uniformity.md) | EHY nearest-neighbor rectangular-uniformity test |

## [10] Tests on Special Domains

| Function | Purpose |
|---|---|
| [`pysht.simplex.uniformity`](simplex.md) | Dirichlet-alternative simplex-uniformity LRT |
| [`pysht.simplex.ehy_uniformity`](simplex.md) | Omnibus EHY simplex-uniformity test |
| [`pysht.simplex.alpha_energy_ksamp`](simplex.md) | Alpha-transformed compositional equality test |

## [11] Independence

| Function | Purpose |
|---|---|
| [`pysht.independence.distance_covariance`](independence.md) | Distance-covariance pairwise independence |
| [`pysht.independence.hsic`](independence.md) | Kernel pairwise independence |
| [`pysht.independence.dhsic`](independence.md) | Kernel joint independence of two or more blocks |
| [`pysht.independence.distance_multivariance`](independence.md) | Distance-multivariance mutual independence |

## [12] Circular Data

| Function | Purpose |
|---|---|
| [`pysht.circular.rayleigh`](circular.md) | First-harmonic circular non-uniformity |
| [`pysht.circular.watson`](circular.md) | Watson $U^2$ omnibus circular uniformity |
| [`pysht.circular.hermans_rasson`](circular.md) | Modified Hermans--Rasson omnibus test |
| [`pysht.circular.mardia_watson_wheeler_ksamp`](circular.md) | Circular multi-sample equality |

## Correctness-gated research candidates

Ball Divergence, the Xue--Yao mean test, and the Yu--Li--Xue covariance
combination have private
research implementations but no public callable because their planned
finite-resampling or joint-component calibration did not pass the release
gate. The JWJWZ covariance and YLXL joint procedures remain formula-audit
targets with no callable. A blocked method is never replaced by a silent
fallback.

## Result objects

Every public test returns an immutable
[`StatisticalTestResult`](results.md) subtype. Frequentist tests report a
statistic and p-value; resampling results add exactness and Monte Carlo
diagnostics. The two maximum-pairwise-Bayes-factor functions instead report
maximum and component log Bayes factors and have no p-value.

For R names, years, and expanded acronyms, use the
[migration crosswalk](../migration/from-r.md), the
[method-name glossary](../methods/names-and-acronyms.md), and the
[native-method catalog](../methods/native-methods.md).

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
independence
circular
results
../methods/index
```

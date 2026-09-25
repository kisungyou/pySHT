# Method names and acronyms

pySHT uses short, lowercase author tokens for paper-specific Python functions:
one author contributes a surname, while multiple authors contribute the initials
of their surnames. Capitalization and full names remain visible in prose and
printed results. Conventional statistical names remain conventional.

The qualified module is part of every public name. The covariance CLX test is
public, while the distinct mean CLX and one-sample Fisher covariance identities
are validation-blocked. The R names below are searchable migration metadata;
pySHT does not expose them as callables.

```{index} single: method acronyms
```
```{index} single: SHT; R-to-Python names
```

## Means and variances

| Token or name | Expanded method | Canonical Python function | SHT 0.1.9 |
|---|---|---|---|
| Student t | Student one- and two-sample t tests | `mean.ttest_1samp`, `mean.ttest_2samp` | `mean1.ttest`, `mean2.ttest` |
| ANOVA | One-way analysis of variance | `mean.anova_oneway` | `meank.anova` |
| Hotelling | Hotelling's $T^2$ (1931) | `mean.hotelling_1samp`, `mean.hotelling_2samp` | `mean1.1931Hotelling`, `mean2.1931Hotelling` |
| Dempster | Dempster non-exact mean tests (1958) | `mean.dempster_1samp`, `mean.dempster_2samp` | `mean1.1958Dempster`, `mean2.1958Dempster` |
| BS | Bai--Saranadasa (1996) | `mean.bs_1samp`, `mean.bs_2samp` | `mean1.1996BS`, `mean2.1996BS` |
| SD | Srivastava--Du (2008) | `mean.sd_1samp`, `mean.sd_2samp` | `mean1.2008SD`, `mean2.2008SD` |
| Yao | Yao (1965) | `mean.yao_2samp` | `mean2.1965Yao` |
| Johansen | Johansen (1980) | `mean.johansen_2samp` | `mean2.1980Johansen` |
| NVM | Nel--Van der Merwe (1986) | `mean.nvm_2samp` | `mean2.1986NVM` |
| KY | Krishnamoorthy--Yu (2004) | `mean.ky_2samp` | `mean2.2004KY` |
| LJW | Lopes--Jacob--Wainwright (2011) | `mean.ljw_2samp` | `mean2.2011LJW` |
| CLX | Cai--Liu--Xia maximum mean test (2014) | — (validation-blocked; no public callable) | `mean2.2014CLX` |
| Thulin | Thulin random-subspace test (2014) | `mean.thulin_2samp` | `mean2.2014Thulin` |
| mxPBF / LYL | Lee--You--Lin maximum pairwise Bayes factor | `mean.maximum_pairwise_bayes_factor_2samp` | `mean2.mxPBF` |
| Schott | Schott k-sample mean test (2007) | `mean.schott_ksamp` | `meank.2007Schott` |
| ZX | Zhang--Xu (2009) | `mean.zx_ksamp` | `meank.2009ZX` |
| CPH | Cao--Park--He (2019) | `mean.cph_ksamp` | `meank.2019CPH` |
| chi-square | One-sample normal variance test | `variance.chisquare_1samp` | `var1.chisq` |
| F | Two-sample normal variance test | `variance.f_2samp` | `var2.F` |
| Bartlett | Bartlett homogeneity test (1937) | `variance.bartlett` | `vark.1937Bartlett` |
| Levene | Levene homogeneity test (1960) | `variance.levene` | `vark.1960Levene` |
| Brown--Forsythe | Brown--Forsythe homogeneity test (1974) | `variance.brown_forsythe` | `vark.1974BF` |

## Covariance and joint hypotheses

| Token or name | Expanded method | Canonical Python function | SHT 0.1.9 |
|---|---|---|---|
| Fisher | Fisher covariance test (2012), withheld in 0.1.0 | no public function | `cov1.2012Fisher` |
| WL | Wu--Li random-projection covariance tests (2015) | `covariance.wl_1samp`, `covariance.wl_2samp` | `cov1.2015WL`, `cov2.2015WL` |
| LC | Li--Chen covariance test (2012) | `covariance.lc_2samp` | `cov2.2012LC` |
| CLX | Cai--Liu--Xia covariance test (2013) | `covariance.clx_2samp` | `cov2.2013CLX` |
| mxPBF / LYL | Lee--You--Lin maximum pairwise Bayes factor | `covariance.maximum_pairwise_bayes_factor_2samp` | `cov2.mxPBF` |
| Schott 2001 | Schott Wald covariance test | `covariance.schott_2001_ksamp` | `covk.2001Schott` |
| Schott 2007 | Schott high-dimensional covariance test | `covariance.schott_2007_ksamp` | `covk.2007Schott` |
| LRT / AS | One-sample normal likelihood-ratio test; Arnold--Shavelle form (1998) | `mean_variance.lrt_1samp` | `mvar1.1998AS`, `mvar1.LRT` |
| PN | Pearson--Neyman (1930) | `mean_variance.pn_2samp` | `mvar2.1930PN` |
| PL | Perng--Littell (1976) | `mean_variance.pl_2samp` | `mvar2.1976PL` |
| Muirhead | Muirhead (1982) | `mean_variance.muirhead_2samp` | `mvar2.1982Muirhead` |
| exact LRT / ZXC | Zhang--Xu--Chen exact likelihood-ratio test (2012) | `mean_variance.exact_lrt_2samp` | `mvar2.2012ZXC` |
| LRT | Two-sample likelihood-ratio test | `mean_variance.lrt_2samp` | `mvar2.LRT` |
| LLZS | Liu--Liu--Zheng--Shi (2017) | `mean_covariance.llzs_1samp` | `sim1.2017Liu` |
| LRT | One-sample mean/covariance likelihood-ratio test | `mean_covariance.lrt_1samp` | `sim1.LRT` |
| HN | Hyodo--Nishiyama (2018) | `mean_covariance.hn_2samp` | `sim2.2018HN` |

## Distributional goodness-of-fit and special domains

| Token or name | Expanded method | Canonical Python function | SHT 0.1.9 |
|---|---|---|---|
| BG | Biswas--Ghosh (2014) | `equaldist.bg_2samp` | `eqdist.2014BG` |
| Shapiro--Wilk | Shapiro--Wilk normality test (1965) | `normality.shapiro_wilk` | `norm.1965SW` |
| Shapiro--Francia | Shapiro--Francia normality test (1972) | `normality.shapiro_francia` | `norm.1972SF` |
| JB | Jarque--Bera normality test (1980) | `normality.jarque_bera` | `norm.1980JB` |
| AJB | Adjusted Jarque--Bera / Urzúa (1996) | `normality.adjusted_jarque_bera` | `norm.1996AJB` |
| RJB | Robust Jarque--Bera / Gel--Gastwirth (2008) | `normality.robust_jarque_bera` | `norm.2008RJB` |
| YM interpoint | Yang--Modarres interpoint test (2017) | `uniformity.ym_interpoint` | `unif.2017YMi` |
| YM quantile | Yang--Modarres quantile test (2017) | `uniformity.ym_quantile` | `unif.2017YMq` |
| simplex uniformity | Dirichlet likelihood-ratio test of uniformity | `simplex.uniformity` | `simplex.uniform` |

## pySHT-native research methods

These names have no SHT crosswalk. Expanded surnames and common search terms
are indexing metadata, not callable aliases.

| Token or search name | Expanded method | Canonical Python function |
|---|---|---|
| CQ | Chen--Qin dense high-dimensional mean test (2010) | `mean.cq_2samp` |
| Li | Li fixed-small-sample, increasing-dimension Student tests (2023) | `mean.li_1samp`, `mean.li_2samp`, `mean.li_ksamp` |
| CZZ | Chen--Zhang--Zhong covariance identity and sphericity tests (2010) | `covariance.czz_identity_1samp`, `covariance.czz_sphericity_1samp` |
| DISCO / energy | Rizzo--Székely energy multi-sample equality test | `equaldist.energy_ksamp` |
| MMD | Maximum mean discrepancy | `equaldist.mmd_2samp` |
| Ball Divergence | Pan--Tian--Wang--Zhang distribution-equality test | — (validation-blocked; private research implementation) |
| distance covariance / dCov | Székely--Rizzo--Bakirov independence test | `independence.distance_covariance` |
| HSIC | Hilbert--Schmidt independence criterion | `independence.hsic` |
| dHSIC | Joint Hilbert--Schmidt independence criterion | `independence.dhsic` |
| distance multivariance | Böttcher--Keller-Ressel--Schilling mutual-independence test | `independence.distance_multivariance` |
| HZ | Henze--Zirkler multivariate-normality test | `normality.henze_zirkler` |
| energy normality | Székely--Rizzo multivariate-normality test | `normality.energy` |
| EHY | Ebner--Henze--Yukich nearest-neighbor uniformity statistic | `uniformity.ehy`, `simplex.ehy_uniformity` |
| alpha-energy | Sevinç--Tsagris compositional equality test | `simplex.alpha_energy_ksamp` |
| Rayleigh | First-harmonic circular-uniformity test | `circular.rayleigh` |
| Watson $U^2$ | Omnibus circular-uniformity test | `circular.watson` |
| HR | Modified Hermans--Rasson Sobolev circular-uniformity test (1985) | `circular.hermans_rasson` |
| MWW | Mardia--Watson--Wheeler circular multi-sample test | `circular.mardia_watson_wheeler_ksamp` |

The [native-method catalog](native-methods.md) distinguishes public methods
from research candidates that remain behind scientific gates.

## Collision rule

Years appear in a callable only when author token and sampling design collide
inside one module. This is why the covariance module has
`schott_2001_ksamp` and `schott_2007_ksamp`, while the mean module needs only
`schott_ksamp`. If a future method still collides after adding its year, pySHT
will append a descriptive statistic token rather than an arbitrary `a` or `b`.

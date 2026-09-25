# pySHT-native method catalog

This catalog records research methods added beyond the SHT 0.1.9 migration
surface. They do not appear in the R crosswalk. “Public” means the callable,
formula ledger, boundary contract, and advertised calibration evidence have
passed the current release gate; a blocked method has no public callable.

```{index} single: pySHT-native methods
```

## Foundational cohort

| Method | Python function | Primary source | Status |
|---|---|---|---|
| Chen--Qin dense mean test | `mean.cq_2samp` | [Chen--Qin (2010)](https://doi.org/10.1214/09-AOS716) | Public |
| Li fixed-small-sample mean tests | `mean.li_1samp`, `li_2samp`, `li_ksamp` | [Li (2023)](https://doi.org/10.1016/j.jmva.2023.105183) | Public |
| CZZ covariance identity and sphericity | `covariance.czz_identity_1samp`, `czz_sphericity_1samp` | [Chen--Zhang--Zhong (2010)](https://doi.org/10.1198/jasa.2010.tm09560) | Public |
| DISCO/energy distribution equality | `equaldist.energy_ksamp` | [Rizzo--Székely (2010)](https://doi.org/10.1214/09-AOAS245) | Public |
| Distance-covariance independence | `independence.distance_covariance` | [Székely--Rizzo--Bakirov (2007)](https://doi.org/10.1214/009053607000000505) | Public |
| Rayleigh circular uniformity | `circular.rayleigh` | Rayleigh statistic; simulation-calibrated | Public |
| Watson circular uniformity | `circular.watson` | [Watson (1961)](https://doi.org/10.1093/biomet/48.1-2.109) | Public |

## Omnibus, kernel, and mutual-dependence cohort

| Method | Python function | Primary source | Status |
|---|---|---|---|
| Maximum mean discrepancy | `equaldist.mmd_2samp` | [Gretton et al. (2012)](https://jmlr.org/papers/v13/gretton12a.html) | Public |
| HSIC pairwise independence | `independence.hsic` | [Gretton et al. (2008)](https://papers.nips.cc/paper/3201-a-kernel-statistical-test-of-independence.pdf) | Public |
| dHSIC mutual independence | `independence.dhsic` | [Pfister et al. (2018)](https://doi.org/10.1111/rssb.12235) | Public |
| Distance multivariance | `independence.distance_multivariance` | [Böttcher et al. (2019)](https://doi.org/10.1214/18-AOS1764) | Public |
| Henze--Zirkler multivariate normality | `normality.henze_zirkler` | [Henze--Zirkler (1990)](https://doi.org/10.1080/03610929008830400) | Public |
| Energy multivariate normality | `normality.energy` | [Székely--Rizzo (2005)](https://doi.org/10.1016/j.jmva.2003.12.002) | Public |
| Modified Hermans--Rasson circular uniformity | `circular.hermans_rasson` | [Hermans--Rasson (1985)](https://doi.org/10.1093/biomet/72.3.698) | Public |
| Mardia--Watson--Wheeler circular equality | `circular.mardia_watson_wheeler_ksamp` | [Mardia (1972)](https://doi.org/10.1111/j.2517-6161.1972.tb00891.x) | Public |

## Structured-domain cohort

| Method | Python function | Primary source | Status |
|---|---|---|---|
| EHY rectangular uniformity | `uniformity.ehy` | [Ebner--Henze--Yukich (2018)](https://doi.org/10.1016/j.jmva.2017.12.009) | Public |
| EHY simplex uniformity | `simplex.ehy_uniformity` | [Ebner--Henze--Yukich (2018)](https://doi.org/10.1016/j.jmva.2017.12.009) | Public |
| Alpha-energy compositional equality | `simplex.alpha_energy_ksamp` | [Sevinç--Tsagris (2026)](https://doi.org/10.1080/03610918.2026.2636167) | Public |

## Advanced candidates held behind gates

| Method | Planned function | Reason it is not public |
|---|---|---|
| Ball Divergence | `equaldist.ball_divergence_2samp` | Floating-distance equality cannot yet preserve every exact closed-ball symmetry without merging a representably unequal near-tie |
| Xue--Yao multiplier mean test | `mean.xy_2samp` | The planned 999-draw calibration was too conservative in its audited regime |
| Yu--Li--Xue covariance combination | `covariance.ylx_2samp` | Joint component independence and a 20,000-dataset size gate remain incomplete |
| Jiang--Wen--Jiang--Wang--Zhang covariance test | `covariance.jwjwz_2samp` | Finite estimator and calibration have not been reconciled with a literal oracle |
| Yu--Li--Xue--Li joint test | `mean_covariance.ylxl_2samp` | The default Fisher/power-enhancement branch has not completed its equation and calibration gates |

These blocked names are documentation metadata, not imports. pySHT does not
publish placeholders or substitute another statistic under the planned name.

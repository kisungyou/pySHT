# Validation ledgers

Validation is a release requirement, not a retrospective exercise. These
records connect each public implementation to its hypotheses, primary-paper
formula, calibration, literal or trusted oracle, numerical edge cases,
invariants, and known differences from SHT 0.1.9.

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Mean tests
:link: classical-mean
:link-type: doc
:class-card: status-card status-available

Classical formulas, high-dimensional trace and diagonal statistics,
Behrens--Fisher approximations, randomization, Bayes factors, multi-group
procedures, and the validation-blocked sparse maximum-test audit.
:::

:::{grid-item-card} Variance tests
:link: classical-variance
:link-type: doc
:class-card: status-card status-available

Alternative-tail coverage, robust scale handling, degenerate samples, and
independent SciPy comparisons.
:::

:::{grid-item-card} Covariance tests
:link: covariance
:link-type: doc
:class-card: status-card status-available

Null-covariance whitening, literal U-statistics, two-sided projection tests,
multi-group formulas, and conditional-regression Bayes factors.
:::

:::{grid-item-card} Mean and variance
:link: mean-variance
:link-type: doc
:class-card: status-card status-available

Joint normal-population likelihood ratios, corrected rejection tails,
component combinations, and stable exact quadrature.
:::

:::{grid-item-card} Mean and covariance
:link: mean-covariance
:link-type: doc
:class-card: status-card status-available

Nonidentity-null whitening, high-dimensional centering, classical LRT checks,
and unequal-sample trace estimators.
:::

:::{grid-item-card} Distributional equality
:link: biswas-ghosh-2014
:link-type: doc
:class-card: status-card status-available

Exact enumeration, corrected Monte Carlo inference, floating-point ties,
scale stability, and a literal distance-statistic oracle.
:::

:::{grid-item-card} Normality
:link: normality
:link-type: doc
:class-card: status-card status-available

Shapiro approximations, moment formulas, finite-sample null-size audits, and
Monte Carlo defaults.
:::

:::{grid-item-card} Rectangular uniformity
:link: uniformity
:link-type: doc
:class-card: status-card status-available

Interpoint moments, quantile transforms, support boundaries, and asymptotic
versus Monte Carlo calibration.
:::

:::{grid-item-card} Simplex uniformity
:link: simplex-uniformity
:link-type: doc
:class-card: status-card status-available

Dirichlet likelihoods, symmetric and general optimization, strict boundaries,
and Wilks-regime checks.
:::
::::

## Multivariate-mean method ledgers

- [Dempster](highdim-mean/dempster.md)
- [Bai--Saranadasa](highdim-mean/bai-saranadasa.md)
- [Srivastava--Du](highdim-mean/srivastava-du.md)
- [Multivariate Behrens--Fisher procedures](highdim-mean/behrens-fisher.md)
- [Lopes--Jacob--Wainwright](highdim-mean/lopes-jacob-wainwright.md)
- [Thulin](highdim-mean/thulin.md)
- [Cai--Liu--Xia](highdim-mean/cai-liu-xia.md) — validation-blocked; not public
- [Lee--You--Lin](highdim-mean/lee-you-lin.md)
- [Schott](highdim-mean/schott.md)
- [Zhang--Xu](highdim-mean/zhang-xu.md)
- [Cao--Park--He](highdim-mean/cao-park-he.md)

## What the evidence labels mean

| Evidence | Question answered |
|---|---|
| Formula ledger | Does the code evaluate the intended finite-sample quantity? |
| Hand fixture | Can a reader reproduce at least one result independently? |
| Trusted oracle | Does a separate established implementation agree where contracts overlap? |
| Literal reference | Does an intentionally simple implementation reproduce the optimized path? |
| Invariance check | Does the result respect transformations implied by the mathematics? |
| Numerical stress test | Does finite precision preserve inferential ordering and physical units? |
| Null simulation | Is an approximation calibrated in the regime where it is advertised? |

No single row is sufficient by itself. The appropriate combination depends on
the method and its calibration. A public asymptotic option remains labeled as
an approximation when finite-sample simulation does not support a stronger
claim.

The [release-simulation runner](reproducing-simulations.md) records the
maintained scenario and random-stream contracts for the null and targeted
alternative audits, and explains why historical Fisher rows are excluded from
the 0.1.0 release evidence.

## Legacy corrections

The [legacy audit](legacy-audit.md) records confirmed defects and how pySHT
addresses them. Important corrections include null-covariance whitening,
literal scale-equivariant covariance U-statistics, genuine two-sided Wu--Li
tests, corrected likelihood-ratio tails, fixed CPH group indexing, defined
AJB/RJB defaults with Monte Carlo uncertainty, fixed auxiliary randomness
during permutation, log-domain exact and Bayesian calculations, and removal
of the invalid distribution-equality asymptotic branch.

```{toctree}
:hidden:
:maxdepth: 1

classical-mean
highdim-mean/dempster
highdim-mean/bai-saranadasa
highdim-mean/srivastava-du
highdim-mean/behrens-fisher
highdim-mean/lopes-jacob-wainwright
highdim-mean/thulin
highdim-mean/cai-liu-xia
highdim-mean/lee-you-lin
highdim-mean/schott
highdim-mean/zhang-xu
highdim-mean/cao-park-he
classical-variance
covariance
mean-variance
mean-covariance
biswas-ghosh-2014
normality
uniformity
simplex-uniformity
legacy-audit
reproducing-simulations
```

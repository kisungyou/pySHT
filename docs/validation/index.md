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

:::{grid-item-card} Native distribution tests
:link: distribution-equality
:link-type: doc
:class-card: status-card status-available

Energy and kernel MMD formulas, exact small orbits, Monte Carlo inference,
geometric invariants, performance boundaries, and the validation-blocked Ball
Divergence audit.
:::

:::{grid-item-card} Independence
:link: independence
:link-type: doc
:class-card: status-card status-available

Distance covariance, HSIC, dHSIC, and distance multivariance with fixed
distance/kernel geometry and marginal-permutation calibration.
:::

:::{grid-item-card} Normality
:link: normality
:link-type: doc
:class-card: status-card status-available

Shapiro approximations, moment formulas, finite-sample null-size audits, and
Monte Carlo defaults.
:::

:::{grid-item-card} Circular data
:link: circular-rayleigh
:link-type: doc
:class-card: status-card status-available

First-harmonic and omnibus uniformity statistics plus circular multi-sample
rank testing, with period-unit and rotation/reflection invariants.
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
- [Chen--Qin, Li, and Xue--Yao](highdim-mean/chen-qin-li-xue-yao.md)
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

## pySHT-native method ledgers

- [Distribution equality](distribution-equality.md)
- [Independence](independence.md)
- [Henze--Zirkler multivariate normality](henze-zirkler.md)
- [Energy multivariate normality](energy-normality.md)
- [EHY rectangular uniformity](ehy-rectangular.md)
- [EHY simplex uniformity](ehy-simplex.md)
- [Alpha-energy compositional equality](alpha-energy-simplex.md)
- [Rayleigh circular uniformity](circular-rayleigh.md)
- [Watson circular uniformity](circular-watson.md)
- [Modified Hermans--Rasson](circular-hermans-rasson.md)
- [Mardia--Watson--Wheeler](circular-mardia-watson-wheeler.md)

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

The simulation release gate is a regression screen, not a claim that true
finite-sample rejection probabilities equal their nominal levels. In the
independent-dataset audits its tolerance is
$\max\{0.005,4\sqrt{a(1-a)/R}\}$ for $R$ replications at level $a$.
At $R=20{,}000$ and $a=0.01$, the absolute 0.005 floor permits a 50% relative
size distortion. A passing row can therefore have a binomial confidence
interval that excludes the nominal level. The native mean/covariance runner
reports marginal 95% Clopper--Pearson intervals and nominal-level inclusion
separately from gate decisions. These intervals describe simulation
uncertainty for the specified design; they do not establish validity for
different dimensions, covariance spectra, sample sizes, or distributional
assumptions, and they are not adjusted for examining multiple scenarios.

For example, the CZZ identity-null design at $(n,p)=(100,100)$ rejects
260/20,000 datasets at nominal 0.01 (observed 0.013), and 1,090/20,000 at
nominal 0.05 (observed 0.0545). Both pass the release screen, while showing
finite-sample excess rejection. Such options remain explicitly asymptotic;
the gate does not make them exact tests or guarantee a user-specified error
rate in that regime.

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
highdim-mean/chen-qin-li-xue-yao
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
distribution-equality
independence
normality
henze-zirkler
energy-normality
uniformity
ehy-rectangular
simplex-uniformity
ehy-simplex
alpha-energy-simplex
circular-rayleigh
circular-watson
circular-hermans-rasson
circular-mardia-watson-wheeler
legacy-audit
reproducing-simulations
```

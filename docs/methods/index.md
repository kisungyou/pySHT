# Methods and implementation status

pySHT exposes 51 canonical functions covering 52 of the 54 public statistical
routine identities in SHT 0.1.9. `mvar1.1998AS` and `mvar1.LRT` are
algebraically the same test and share `pysht.mean_variance.as_1samp`.
`mean2.2014CLX` and `cov1.2012Fisher` are validation-blocked and have no public
callable. The two R callable adapters are unnecessary in Python, and no
compatibility aliases are added.

The [API reference](../api/index.md) is authoritative for callable names and
signatures. The [R migration crosswalk](../migration/from-r.md) records every
legacy entry point, while the searchable
[method names and acronyms](names-and-acronyms.md) glossary expands author
tokens, years, and spelling variants.

## Scientific categories

| SHT category | pySHT module | Public scope |
|---|---|---|
| [0] Utilities | — | R adapters are replaced by ordinary Python callables |
| [1] Univariate mean | `pysht.mean` | t tests and one-way ANOVA |
| [2] Multivariate mean | `pysht.mean` | classical, high-dimensional, unequal-covariance, randomized, Bayesian, and multi-group tests; mean CLX remains validation-blocked |
| [3] Variance | `pysht.variance` | one-, two-, and multi-sample variance or spread tests |
| [4] Covariance | `pysht.covariance` | validated one-, two-, and multi-sample covariance tests, including a Bayesian procedure; Fisher remains withheld |
| [5] Mean and variance | `pysht.mean_variance` | one- and two-sample joint tests for univariate normal parameters |
| [6] Mean and covariance | `pysht.mean_covariance` | one- and two-sample joint multivariate tests |
| [7] Equality of distributions | `pysht.equaldist` | exact or Monte Carlo two-sample permutation testing |
| [8] Normality | `pysht.normality` | Shapiro and moment-based univariate goodness-of-fit tests |
| [9] Rectangular uniformity | `pysht.uniformity` | interpoint-distance and normal-quantile tests |
| [10] Special domains | `pysht.simplex` | probability-simplex uniformity against Dirichlet alternatives |

Module qualification is part of a function's identity. For example, the
public covariance CLX test belongs to `pysht.covariance`, while the distinct
mean CLX and one-sample Fisher identities are not public. Python names use lowercase `snake_case`;
mathematical capitalization remains in method titles and statistic labels.

## A consistent method contract

Every procedure documents:

1. the null and alternative hypotheses;
2. sampling and dimensional assumptions;
3. the statistic and finite-sample denominators;
4. the null, randomization, or Bayes-factor calibration;
5. supported alternatives and confidence intervals;
6. numerical edge cases and mathematical invariants;
7. result fields and their units; and
8. primary references and independent validation evidence.

An asymptotic p-value remains an approximation even when a function is public.
The [test chooser](../user-guide/choose-a-test.md) explains how to select among
methods, and the [validation center](../validation/index.md) records the
evidence and advertised regimes behind each implementation.

```{toctree}
:hidden:
:maxdepth: 1

names-and-acronyms
```

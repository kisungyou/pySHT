# Thulin random-subspaces mean test

**Status:** primary-paper statistic, fixed-plan permutation, reproducibility,
feature-scale, and boundary gates pass.

The statistic averages two-sample Hotelling statistics over `n_subspaces`
feature subsets. The subset dimension is fixed by the paper at
`floor((n_x + n_y - 2) / 2)` and is reported in the result diagnostics.

The subspaces are drawn exactly once and held fixed for the observed and all
permuted statistics. This corrects the legacy implementation, which sampled
new subspaces for each permutation and mixed auxiliary randomness into the
randomization distribution. Seeded literal reconstruction verifies all
subspaces, permutation statistics, exceedances, the corrected Monte Carlo
p-value, its standard error, and its 95% binomial interval. Separate positive
feature rescalings leave the result unchanged when the seed is fixed.

Each allocation is placed in deterministic row/group order before computing
the subspace average. Upper-tail comparisons include relative roundoff ties
within $100\epsilon|T_{\mathrm{obs}}|$. A six-row, two-feature regression
enumerates all 20 allocations independently: the two mathematically tied
upper-tail allocations have exact probability 0.1. With one full-dimensional
subspace, 9,999 draws and seed 123, the corrected implementation counts 1,018
exceedances and reports 0.1019; swapping groups or reordering rows gives the
same result. Each quadratic is evaluated directly from an SVD of equilibrated
centered selected observations, simultaneously checking rank and avoiding a
scatter matrix that squares their condition number. Dependent directions
cannot become artificially positive definite through scatter-matrix rounding.

Each Hotelling statistic is formed directly from the selected $k$ columns.
The implementation never constructs a full $p\times p$ identity matrix or
pooled covariance for a $k$-column subspace, so working storage for this
kernel is governed by the selected dimension rather than quadratically by the
ambient feature count. A 5,000-feature allocation guard covers this property.

The direct paper locations are Algorithm 2 (permutation test, p. 6),
Algorithm 3 (random-subspace statistic, p. 8), Proposition 3 (conditional
marginal-scale and location invariance, p. 9), and Section 3.3 (the recommended
`floor((n_x + n_y - 2) / 2)` dimension, p. 9). Holding the sampled subspaces
fixed when Algorithm 2 evaluates the statistic is pySHT's explicit
conditional-randomization interpretation, supported by the paper's repeated
conditioning on the chosen subspaces; it is not a claim that Algorithm 2
spells out the random-number plumbing.

The unrestricted permutation null requires exchangeability of the pooled
rows. Section 2.1 (pp. 4–5) explains the common-distribution/common-covariance
null motivating this relabeling. A mean-only null with different distributions
does not give an exact finite-sample permutation test. Canonical pooled-row and
group ordering makes a fixed integer seed invariant to row order and sample
exchange without altering the set or distribution of label permutations.

## Targeted alternative-power gate

Seed 2026090320 initializes a `SeedSequence`. In each replication, its
persistent child-0 PCG64 stream first draws 12 rows from
$N_{25}(0.8\mathbf 1,I)$ and then 12 rows from $N_{25}(0,I)$. Its persistent
child-1 PCG64 stream is passed to the exact public call
`mean.thulin_2samp(x, y, n_subspaces=20, n_resamples=199,
rng=auxiliary_rng)`. Thus every dataset gets 20 freshly drawn subspaces, held
fixed across its observed statistic and 199 Monte Carlo label permutations.
Both streams advance through 300 outer replications and a rejection means
`pvalue < alpha`.

| alpha | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|
| Rejections / 300 | 291 | 300 | 300 |
| Rate | 0.970 | 1.000 | 1.000 |

The inner corrected p-values lie on a grid of width `1/200`; the outer power
estimate has worst-case binomial standard error 0.0289. This deliberately
strong alternative is a tractable response gate, not a precise or comparative
power estimate. `python -m tools.mean_power_audits` reproduces it under Python
3.12.13, NumPy 2.5.1, and SciPy 1.18.0.

Primary reference: M. Thulin, *A High-Dimensional Two-Sample Test for the
Mean Using Random Subspaces*, Computational Statistics & Data Analysis 74
(2014), 26–38, <https://arxiv.org/abs/1304.4564>.

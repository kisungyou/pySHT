# Distance and kernel independence tests: validation record

## Scope and status

This ledger covers `independence.distance_covariance`, `hsic`, `dhsic`, and
`distance_multivariance`. Each accepts row-paired random-vector blocks and
tests factorization of their joint distribution. `distance_covariance` and
`hsic` are pairwise tests; `dhsic` and `distance_multivariance` test mutual
independence and are not reducible to a collection of pairwise decisions.

Independent rows are required. Unrestricted marginal permutations are invalid
for time series, clusters, matched sets requiring restricted permutations, or
other dependent designs.

## Formula ledger

### Distance covariance

Let $B_X$ and $B_Y$ be Euclidean distance matrices and
$H=I-\mathbf1\mathbf1^T/n$. Define $A_X=HB_XH$ and $A_Y=HB_YH$. The public
statistic is the original biased empirical statistic in raw distance units,

$$
nV_n^2=\frac1n\sum_{j,k}(A_X)_{jk}(A_Y)_{jk}.
$$

The result also reports distance covariance $\sqrt{V_n^2}$, distance
correlation, and the two distance variances when their raw scales fit in
float64. Calibration divides each centered matrix by its positive mean
interpoint distance. This changes only the common positive scale of every
permutation statistic, so the ordering and p-value remain exact even when the
raw statistic overflows.

Primary source: Székely, Rizzo, and Bakirov, “Measuring and testing dependence
by correlation of distances,” *Annals of Statistics* 35 (2007), 2769--2794,
<https://doi.org/10.1214/009053607000000505>, Definitions 2--5.

### HSIC and dHSIC

For two fixed Gram matrices, `hsic` reports the biased V-statistic

$$
\mathrm{HSIC}_b=\frac1{n^2}\operatorname{tr}(KHLH).
$$

For $d$ Gram matrices $K^1,\ldots,K^d$, `dhsic` implements Definition 4 of
Pfister et al.:

$$
\widehat{\mathrm{dHSIC}}_n=
\frac1{n^2}\sum_{i_1,i_2}\prod_{r=1}^d K^r_{i_1i_2}
+\frac1{n^{2d}}\prod_{r=1}^d\sum_{i,j}K^r_{ij}
-\frac2{n^{d+1}}\sum_i\prod_{r=1}^d\sum_jK^r_{ij}.
$$

The public dHSIC statistic is $n\widehat{\mathrm{dHSIC}}_n$. When $d=2$,
this is $n$ times the public `hsic` value; consequently both functions have
identical permutation ordering, exceedance count, and p-value.

RBF and Laplacian kernels are characteristic. Each block has its own kernel
and bandwidth metadata. A median bandwidth is computed from strictly positive
within-block distances and frozen before permutation. `dhsic` requires
$n\ge2d$, matching the finite-sample regime of the defining V-statistic.

Primary sources: Gretton et al., “A kernel statistical test of independence,”
*NIPS 20* (2008), 585--592; Pfister et al., “Kernel-based tests for joint
independence,” *JRSS B* 80 (2018), 5--31,
<https://doi.org/10.1111/rssb.12235>, Definition 4 and Section 3.2.1.

### Normalized total distance multivariance

For each block let $B_r$ be its distance matrix,
$A_r=-HB_rH/\overline B_r$, where $\overline B_r=n^{-2}\sum B_r$. The public
statistic is $n$ times equation (4.29):

$$
\overline M_n^2=
\frac{n^{-2}\sum_{j,k}\prod_{r=1}^d(1+(A_r)_{jk})-1}
{2^d-d-1}.
$$

This normalized total statistic detects higher-order dependence, including
pairwise-independent alternatives. For $d=2$, it differs from raw $V_n^2$ by
the fixed positive factor $\overline B_X\overline B_Y$. The null hypothesis,
permutation ordering, exceedance count, and p-value therefore reduce exactly
to `distance_covariance`; the printed statistics intentionally retain their
different documented normalizations.

Primary source: Böttcher, Keller-Ressel, and Schilling, “Distance
multivariance: New dependence measures for random vectors,” *Annals of
Statistics* 47 (2019), 2757--2789,
<https://doi.org/10.1214/18-AOS1764>, Theorem 4.1 and equations (4.25), (4.29).

## Permutation and numerical contract

A common row permutation is redundant. Exact calibration fixes the first
canonical block and enumerates all permutations of each remaining block, an
orbit of size $(n!)^{d-1}$. Monte Carlo draws those marginal permutations
independently and uses $(b+1)/(B+1)$, MCSE, and a 95% binomial interval.

An isometry-invariant, metric-only canonical form fixes paired-row order and
full block/kernel/bandwidth bundles before a seeded plan is generated. The
canonicalizer uses stable approximate-equality ranks for computed distances,
colour refinement, and individualization of unresolved metric symmetries.
Each refinement pass and twin comparison charges $n^2$ times its number of
edge layers against a fixed 1,000,000-unit budget. Exhaustion switches to
deterministic coordinate ordering. Exact inference is invariant
under joint row reorderings, block reorderings, pair exchange, rotations,
reflections, feature permutations, separate distance rescaling, and exactly
representable translations. Fixed-seed Monte Carlo replay has these invariants
when canonicalization completes within budget, away from the floating
approximate-distance-rank boundary. At that boundary or after the coordinate
fallback, an isometry can choose another uniformly valid plan and hence another
finite-$B$ count; the exact orbit remains unchanged. Regular-polygon fixtures
exercise stable symmetric cases, while perturbed boundary searches verify
that the limitation affects reproducibility rather than calibration validity.

## Complexity

For $d$ marginal blocks, distance or Gram construction takes
$O(n^2\sum_r p_r)$ time and $O(dn^2)$ storage. Each marginal-permutation
evaluation costs $O(dn^2)$, giving $O(Bdn^2)$ calibration time; bounded batches
keep peak storage quadratic rather than proportional to $B$. The metric-only
canonicalizer uses colour refinement in ordinary cases and bounds its
refinement/twin work to avoid exponential searches on unresolved graph
symmetries. Distance-rank construction and edge-token storage remain quadratic,
with sorting costs for ranks and neighbor signatures; the work budget does
not cap required distance storage or requested resampling work.

## Independent validation

`tests/test_independence.py` contains literal SciPy-distance and Gram-matrix
oracles and establishes:

- raw $nV_n^2$, dCov, dCor, and both distance variances;
- Definition 4 dHSIC for two and three blocks;
- equation (4.29) for three blocks;
- the two-block dHSIC/HSIC and multivariance/dCov monotone reductions;
- exhaustive small marginal-permutation orbits;
- response to the pairwise-independent XOR alternative;
- row, block, pair, Euclidean, feature, translation, and scaling invariants;
- extreme-scale inference, per-block kernel metadata, exact budgets, corrected
  Monte Carlo uncertainty, and global-RNG isolation.

At $n=25$, 999 batched permutations took 0.05 seconds for distance covariance
and 0.04 seconds for HSIC on the development machine. With three blocks whose
feature dimensions were 5, 5, and 3, 999 permutations took 0.07 seconds for
dHSIC and 0.05 seconds for total distance multivariance. Permutation batches
are dynamically bounded by the sample size and block count rather than
materializing all requested plans at once.

## Named-seed release evidence

The committed runner `tools.distribution_independence_audits` generated 20,000
independent null data sets per method. Pairwise scenarios used independent
$N_2(0,I)$ blocks with $n=6$; mutual scenarios used three independent
univariate-normal blocks with $n=6$. Every data set received 999 fresh marginal
permutations from a separate spawned RNG stream. At levels 0.01, 0.05, and
0.10, the raw counts were:

| Method | Seed | at 0.01 | at 0.05 | at 0.10 |
|---|---:|---:|---:|---:|
| distance covariance | 2026091104 | 173 / 20,000 | 971 / 20,000 | 1,959 / 20,000 |
| HSIC | 2026091105 | 161 / 20,000 | 971 / 20,000 | 1,947 / 20,000 |
| dHSIC | 2026091106 | 185 / 20,000 | 1,027 / 20,000 | 2,000 / 20,000 |
| total distance multivariance | 2026091107 | 168 / 20,000 | 964 / 20,000 | 1,938 / 20,000 |

All twelve rates passed the preregistered release tolerance
$\max\{0.005,4\sqrt{\alpha(1-\alpha)/20000}\}$.

Targeted 300-data-set response checks use $Y=X+0.4\epsilon$ for distance
covariance, $Y=X^2+0.2\epsilon$ for HSIC, and the pairwise-independent
$Z=X\mathbin{\mathrm{xor}}Y$ construction for dHSIC and multivariance. These
checks establish directional response, not comparative or local power.

| Method and alternative | Seed | Rejections at 0.01 | at 0.05 | at 0.10 |
|---|---:|---:|---:|---:|
| dCov, $Y=X+0.4\epsilon$ | 2026091204 | 300 / 300 | 300 / 300 | 300 / 300 |
| HSIC, $Y=X^2+0.2\epsilon$ | 2026091205 | 281 / 300 | 300 / 300 | 300 / 300 |
| dHSIC, XOR | 2026091206 | 122 / 300 | 299 / 300 | 300 / 300 |
| total multivariance, XOR | 2026091207 | 300 / 300 | 300 / 300 | 300 / 300 |

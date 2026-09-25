# Energy and MMD, with blocked Ball Divergence: validation record

## Scope and status

This record covers public `equaldist.energy_ksamp` and `equaldist.mmd_2samp`,
plus the private Ball Divergence research implementation. Energy and MMD have
published statistics, fixed-size label orbits, numerical invariants, and
corrected Monte Carlo contracts with executable checks. They test

$$
H_0:F_1=\cdots=F_K
$$

against general distributional inequality. Unrestricted relabeling requires
independent observations and exchangeability of the pooled sample under the
null. Equality of means or covariances alone is insufficient.

## Formula ledger

### DISCO energy test

For groups $A_j$ of sizes $n_j$, total size $N$, and $0<\alpha<2$, define

$$
g_\alpha(A_j,A_k)=\frac{1}{n_jn_k}
\sum_{a\in A_j}\sum_{b\in A_k}\lVert a-b\rVert^\alpha.
$$

The implementation evaluates the Rizzo--Székely decomposition

$$
W_\alpha=\sum_j\frac{n_j}{2}g_\alpha(A_j,A_j),
$$

$$
S_\alpha=\sum_{j<k}\frac{n_jn_k}{2N}
\left\{2g_\alpha(A_j,A_k)-g_\alpha(A_j,A_j)-g_\alpha(A_k,A_k)\right\},
$$

and reports

$$
F_\alpha=\frac{S_\alpha/(K-1)}{W_\alpha/(N-K)}.
$$

The within-group sums include diagonal zeros and use the paper's V-statistic
denominators. When $W_\alpha=0$, the result is zero if $S_\alpha=0$ and
infinity otherwise. The open exponent interval is enforced because
$\alpha=2$ does not give the paper's omnibus consistency claim.

Primary source: Rizzo and Székely, “DISCO analysis: A nonparametric extension
of analysis of variance,” *Annals of Applied Statistics* 4 (2010),
1034--1055, <https://doi.org/10.1214/09-AOAS245>, Sections 2--3.

### Maximum mean discrepancy

For two samples of sizes $n$ and $m$, pySHT reports the unequal-size unbiased
estimator

$$
\widehat{\mathrm{MMD}}_u^2=
\frac{\sum_{i\ne j}k(X_i,X_j)}{n(n-1)}+
\frac{\sum_{i\ne j}k(Y_i,Y_j)}{m(m-1)}-
\frac{2\sum_{i,j}k(X_i,Y_j)}{nm}.
$$

Supported characteristic kernels are

$$
k_{\rm RBF}(x,y)=\exp\{-\lVert x-y\rVert^2/(2\sigma^2)\},
\qquad
k_{\rm Lap}(x,y)=\exp\{-\lVert x-y\rVert/\sigma\}.
$$

The bandwidth is either a positive scalar in data units or the median of the
strictly positive off-diagonal pooled distances. It is selected before labels
are permuted and is therefore part of the fixed test statistic. A constant
pooled sample is rejected because the median heuristic is undefined.

Primary source: Gretton et al., “A kernel two-sample test,” *JMLR* 13 (2012),
723--773, <https://jmlr.org/papers/v13/gretton12a.html>, Section 2 and Lemma 6.

### Ball Divergence (validation-blocked)

Let $\delta(u,v,z)=1\{d(u,z)\le d(u,v)\}$. For a ball centered at an $X$
observation with radius determined by another $X$ observation, let
$A^X_{ij}$ and $A^Y_{ij}$ be the empirical masses from the two samples;
define $C^X_{kl}$ and $C^Y_{kl}$ analogously for balls centered on $Y$. The
implemented statistic is

$$
\mathrm{BD}_{n,m}=\frac1{n^2}\sum_{i,j=1}^n
(A^X_{ij}-A^Y_{ij})^2+
\frac1{m^2}\sum_{k,l=1}^m(C^X_{kl}-C^Y_{kl})^2.
$$

The mathematical membership rule is the closed-ball comparison
`distance <= radius`. Stable distance orders and tie-end ranks reduce each
permuted evaluation from a literal cubic scan to quadratic time, but the
prototype is not public. Exact comparison of computed Euclidean distances can
split radii that are mathematically equal after a rotation, while a roundoff
tolerance can merge radii that are genuinely distinct. On the complete
20-label orbit for `x=[6.66133815e-16,3,-6.66133815e-16]` and
`y=[-1,2,2]`, the direct literal statistic is 0.38271604938271603 with 6
exceedances and p-value 0.3. The optimized prototype instead reports
0.4691358024691358 with 4 exceedances and p-value 0.2. No equality rule audited
so far passes both this fixture and exact geometric-symmetry fixtures.
`ball_divergence_2samp` therefore remains absent from the public API and audit
registry until radius equality has a correctness-certified implementation.

Primary source: Pan et al., “Ball Divergence: Nonparametric two sample test,”
*Annals of Statistics* 46 (2018), 1109--1137,
<https://doi.org/10.1214/17-AOS1579>, equations (2.3)--(2.6).

## Randomization and numerical contract

For group sizes $(n_1,\ldots,n_K)$, exact calibration enumerates all
$N!/\prod_j n_j!$ ordered fixed-size allocations. `calibration="exact"`
fails before calculation if that number exceeds `n_resamples`.
`"permutation"` enumerates within budget and otherwise samples exactly
`n_resamples` allocations. Monte Carlo results use $(b+1)/(B+1)$ and include
the conditional MCSE and 95% Clopper--Pearson tail-probability interval;
exact results use $b/B$.

Coordinates are centered at an overflow-safe per-feature midpoint, divided by
a common positive scale, and distances are divided by their maximum. Energy
inference is invariant to this scaling. Median-bandwidth MMD is also scale
invariant; explicit bandwidths retain their original data units through a
log-ratio calculation. A metric-only canonical form incorporates both the
pooled distance graph and the unlabeled group partition. Stable distance ranks,
colour refinement, and individualization attempt to resolve symmetric
geometries within a fixed work budget. Each refinement pass and twin comparison
charges $N^2$ times its number of edge layers against a 1,000,000-unit budget.
On exhaustion, deterministic coordinate ordering replaces the optional graph
search. Exact inference is invariant under row/group
reordering, orthogonal maps, feature permutations, and exactly representable
common translations. Fixed-seed Monte Carlo replay has the same invariance
when canonicalization completes within budget, away from the floating
approximate-distance-rank boundary. At that boundary or after the coordinate
fallback, isometric coordinates can select a different uniform plan; both plans
remain valid draws from the same conditional orbit, but need not give the same
finite-$B$ exceedance count. Exact calibration remains invariant.

The committed near-regular-octagon boundary fixture applies a rounded rotation
to coordinates perturbed at roughly $10^{-15}$. It requires the same statistic,
exact exceedance count, and exact p-value, while requiring each forced Monte
Carlo result separately to satisfy the corrected-p contract rather than
requiring two boundary-dependent seed realizations to coincide.

## Complexity

For $N$ pooled observations and $p$ features, distance or Gram construction is
$O(N^2p)$ time and $O(N^2)$ storage. Each of $B$ label allocations costs
$O(N^2)$ arithmetic, so calibration is $O(N^2p+BN^2)$ time. Dynamically
bounded batches keep peak storage $O(N^2)$. Metric canonicalization is fast
colour refinement for ordinary data. The bounded refinement/twin work avoids
exponential preprocessing on symmetric inputs, including the 64 vertices of a
six-dimensional hypercube. Distance-rank construction and edge-token storage
remain quadratic, with sorting costs for ranks and neighbor signatures. The
budget bounds optional canonicalization work, not distance storage or the
requested resampling cost.

## Independent validation and performance

`tests/test_equaldist_expanded.py` implements the public statistics
independently with SciPy `cdist`, enumerates every labeling of fixed fixtures,
and verifies:

- two- and three-group DISCO formulas, including constant/separated boundaries;
- literal unequal-size $\mathrm{MMD}_u^2$ and fixed bandwidths;
- the private Ball prototype's optimized-versus-literal fixtures, including
  the adversarial radius case that blocks public exposure;
- translation, orthogonal, feature, row, group, and common-scaling invariants;
- extreme $10^{308}$ and subnormal coordinate scales;
- exact budgets, corrected Monte Carlo metadata, deterministic seeds, and
  isolation from NumPy's global generator.

On the release-development machine at $N=50$ and $p=5$, 999 batched
permutations took 0.03 seconds for Energy and 0.05 seconds for MMD. Calibration
processes at most 512 labelings at once and shrinks the batch as $N$ grows.

## Named-seed release evidence

The reproducible runner is `python -m tools.distribution_independence_audits
<scenario>`. Each null row uses 20,000 independently generated null data sets,
not repeated draws from one conditional orbit. The null scenario uses two
independent $N_2(0,I)$ samples of sizes 4 and 6 and enumerates the complete
210-label orbit for every data set. At levels 0.01, 0.05, and 0.10, the raw
counts were:

| Method | Seed | at 0.01 | at 0.05 | at 0.10 |
|---|---:|---:|---:|---:|
| Energy | 2026091101 | 180 / 20,000 | 904 / 20,000 | 1,892 / 20,000 |
| MMD | 2026091102 | 175 / 20,000 | 914 / 20,000 | 1,907 / 20,000 |

All six calibration points for the two public methods passed the preregistered
release tolerance
$\max\{0.005,4\sqrt{\alpha(1-\alpha)/20000}\}$.

Targeted power uses 300 independent data sets, 199 permutations, and the
prespecified alternative $N_3(0,I)$ versus $N_3(1,2.25I)$. These are response
checks, not comparative power claims. Counts are recorded by the same runner
with `--power`.

| Method | Seed | Rejections at 0.01 | at 0.05 | at 0.10 |
|---|---:|---:|---:|---:|
| Energy | 2026091201 | 202 / 300 | 276 / 300 | 286 / 300 |
| MMD | 2026091202 | 193 / 300 | 275 / 300 | 292 / 300 |

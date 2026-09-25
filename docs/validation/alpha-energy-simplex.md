# Alpha-energy equality of compositional distributions

**Public verdict:** validated and exposed as
`pysht.simplex.alpha_energy_ksamp`. It is a pySHT-native addition with no SHT
0.1.9 crosswalk entry.

## Transform, null, and statistic

The null is equality of the complete compositional distributions in all input
groups. For a $D$-component composition $x$, primary equations (5)--(10) set

$$
u_{\alpha j}(x)=\frac{x_j^\alpha}{\sum_l x_l^\alpha},
\qquad
w_\alpha(x)=\frac{D u_\alpha(x)-\mathbf1}{\alpha},
$$

with the centered-log-ratio limit at $\alpha=0$. A Helmert submatrix in the
paper removes the redundant coordinate, but is an isometry on this zero-sum
subspace. pySHT therefore computes the same distances directly as
$\Delta_\alpha(x,y)=\lVert w_\alpha(x)-w_\alpha(y)\rVert$.

For two groups $A,B$, equation (15) is

$$
\frac{n_A n_B}{n_A+n_B}\left[
\frac{2}{n_A n_B}\sum_{i,j}\Delta_\alpha(A_i,B_j)
-\frac1{n_A^2}\sum_{i,j}\Delta_\alpha(A_i,A_j)
-\frac1{n_B^2}\sum_{i,j}\Delta_\alpha(B_i,B_j)
\right].
$$

For $k>2$, pySHT uses the standard energy extension: the sum of this quantity
over all unordered group pairs. Large values reject.

The paper explicitly restricts prespecified $\alpha$ to $[-1,1]$. At
$\alpha=0$ (and for negative alpha), every component must be strictly
positive. Structural zeros are valid only for $\alpha>0$. No zero replacement
and no data-selected alpha are performed. The source is [Sevinç and Tsagris
(2026)](https://doi.org/10.1080/03610918.2026.2636167), especially equations
(5)--(15) and the paragraph following equation (15).

## Permutation contract and invariants

The statistic conditions on all pooled transformed observations and preserves
group sizes. `calibration="exact"` enumerates all
$B=N!/\prod_g n_g!$ ordered allocations and returns $b/B$; it requires the
budget to cover the orbit. `"monte-carlo"` returns $(b+1)/(B+1)$ with MCSE and
a Clopper--Pearson interval. Automatic `"permutation"` chooses exact when the
complete orbit fits the budget. Exact mode validates but consumes no RNG draws.

Before a seeded Monte Carlo plan, groups and rows are canonicalized using
component-permutation-invariant row keys plus within- and cross-group distance
signatures. Tests include the difficult case in which rows are component
permutations of one another. When the orbit has at most
`min(100_000, 10*n_resamples)` allocations,
the engine forms and sorts its conditional statistic distribution once, then
uses seeded uniform orbit indices; this remains invariant even under exact
geometric automorphisms. Otherwise it draws uniform fixed-size label vectors
directly, without requiring a complete orbit calculation or rejecting
duplicated compositions. Every label vector has the same
$\prod_g n_g!$ preimages under a uniform permutation of row indices, so
duplicates and unresolved symmetries preserve valid conditional inference.
When nonidentical rows or groups have unresolved canonical signatures, row or
group reorderings and common component permutations can change the finite
Monte Carlo count for one seed. They preserve the mathematical statistic and
conditional null distribution; exact inference is unaffected. The pooled
distance matrix is not rebuilt during permutations.

## Complexity

For $N$ pooled rows, $D$ components, and $K$ groups, transformation and the
distance matrix cost $O(ND+N^2D\log D)$ time and $O(ND+N^2)$ storage, including
component-order-stable distance reductions. Each allocation uses sorted
nonnegative reductions, costing $O(N^2\log N)$ time and $O(N^2)$ temporary
storage; within-group means are calculated once per allocation. The optional
complete-orbit calculation evaluates and
stores at most `min(100_000, 10*n_resamples)` scalar statistics, so requesting
few draws no longer forces tens of thousands of extra allocations. Otherwise
Monte Carlo computes exactly `n_resamples` null statistics with bounded
storage. Exact enumeration is streamed.

The alpha transform uses a second-order series around its centered-log-ratio
limit when $|\alpha\log x|$ is within the cancellation regime. Regression
values $10^{-16}$, $10^{-50}$, and $-10^{-300}$ converge to `alpha=0`,
including a composition spanning 200 orders of magnitude, while retaining any
representable finite-alpha correction. Outside that regime, centered
`expm1`/`log1p` arithmetic evaluates the power transform.

## Release evidence

Regression checks cover the 184,756-allocation orbit of two ten-row samples
with a repeated composition. Its 99-draw Monte Carlo count agrees with an
independent replay using the literal alpha transform and SciPy distances. A
separate small-budget check prevents unrequested enumeration of the symmetric
924-allocation orbit when only one Monte Carlo draw is requested.

This method has a finite conditional permutation null, so its calibration gate
is an exhaustive orbit argument rather than a claim of 20,000 independent
datasets. Seed 20260920 generated one pooled ten-row, three-component sample.
For sizes (5,5), all 252 allocations were enumerated. At both $\alpha=0$ and
$\alpha=0.4$, the exact rejection counts across the orbit were 2/252
(0.00794), 12/252 (0.04762), and 24/252 (0.09524) at levels 0.01, 0.05, and
0.10. This is conservative where the orbit makes an exact nominal level
unattainable and passes the project tolerance.

For power, seed 20260930 generated 2,000 pairs of samples with sizes (5,5):
`Dirichlet(20,1,1)` versus `Dirichlet(1,20,1)`. Every dataset used its own
complete 252-allocation conditional null at $\alpha=0.4$; 2,000/2,000 rejected
at 0.05. This is a named strong alternative, not a general power guarantee.

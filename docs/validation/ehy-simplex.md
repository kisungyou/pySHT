# EHY simplex-uniformity test

**Public verdict:** validated and exposed as `pysht.simplex.ehy_uniformity`.
It is a pySHT-native addition with no SHT 0.1.9 crosswalk entry.

## Geometry and statistic

For $D$ compositional components the simplex is a flat
$m=D-1$ dimensional manifold embedded in $\mathbb R^D$. Ambient Euclidean
distance equals its intrinsic flat distance, its Hausdorff volume is
$\sqrt D/(D-1)!$, and its uniform density is
$f_0=(D-1)!/\sqrt D$. Consequently EHY equations (1) and (4) become

$$
T_{n,J}^{(\alpha)}=
\sum_{i=1}^n\sum_{k=1}^J
\left\{v_{D-1}n\lVert X_i-X_i^{(k)}\rVert^{D-1}
\frac{(D-1)!}{\sqrt D}\right\}^{\alpha}.
$$

The null is uniformity with respect to simplex volume. The parameter domain is
$\alpha>0$, $\alpha\ne1$, and $1\le J<n$. Boundary compositions are part of
the closed simplex and are allowed. All first $J$ neighbor contributions are
summed. Rejection is lower-tail for $0<\alpha<1$ and upper-tail for
$\alpha>1$, following Theorem 1 and its remarks in [Ebner, Henze, and Yukich
(2018)](https://doi.org/10.1016/j.jmva.2017.12.009).

## Calibration and checks

The null simulator draws `Dirichlet(1, ..., 1)` samples, which are exactly
uniform with respect to simplex volume. There are no fitted nuisance
parameters. The public stream consumes $nD$ gamma/exponential-derived
Dirichlet values per replicate from the supplied NumPy generator, after the
observed statistic; the precise low-level variate consumption is NumPy's
documented `Generator.dirichlet` stream and is replayed through that public
method rather than assumed from a gamma implementation. Corrected Monte Carlo
$p=(b+1)/(B+1)$ and uncertainty fields are reported.

An independent direct-distance formula verifies the Hausdorff-density factor
and every first-$J$ term. Tests cover row and common component permutations,
closed-boundary values, duplicates, both tails, stream replay, and the 9,999
draw performance budget.

## Complexity

With $n$ rows and $D$ components, one statistic costs $O(n^2D)$ time. The
distance matrix uses $O(n^2)$ storage and the row-at-a-time distance workspace
uses $O(nD)$, so a $B$-draw calibration costs $O(Bn^2D)$ time and
$O(n^2+nD)$ peak storage rather than storage proportional to $B$.

## Release evidence

At $(n,D,J)=(25,3,2)$, each independent 20,000-dataset stream was compared
with its own independently seeded 99,999-statistic reference bank:

| $\alpha$ | reference seed | dataset seed | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 20260914 | 20260915 | 200 (0.01000) | 971 (0.04855) | 1,944 (0.09720) |
| 0.5 | 20260932 | 20260916 | 188 (0.00940) | 943 (0.04715) | 1,955 (0.09775) |
| 2.0 | 20260914 | 20260915 | 227 (0.01135) | 1,071 (0.05355) | 2,003 (0.10015) |
| 2.0 | 20260932 | 20260916 | 200 (0.01000) | 942 (0.04710) | 1,927 (0.09635) |

Every cell passes the release tolerance. Because all observed datasets in one
stream share a finite reference bank, the tolerance is
$\max\{0.005,4\sqrt{a(1-a)(1/20000+1/99999)}\}$ at nominal level $a$ and
includes both dataset and reference-quantile uncertainty. Seed 20260926 generated 2,000
size-40 `Dirichlet(50,50,50)` alternatives; with $\alpha=0.5$, $J=1$, and a
19,999-statistic null bank from seed 20260925, 2,000/2,000 rejected at 0.05.
This is a named concentration-alternative check only.

The table uses empirical reference banks rather than 20,000 literal public
9,999-draw calls. Separate fixtures replay `Generator.dirichlet`, the selected
tail, tie handling, and corrected public p-value exactly.

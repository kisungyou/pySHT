# EHY rectangular-uniformity test

**Public verdict:** validated and exposed as `pysht.uniformity.ehy`. It is a
pySHT-native addition with no SHT 0.1.9 crosswalk entry.

## Primary equations and hypotheses

The fixed null is uniformity on the user-declared hyperrectangle. Coordinates
are mapped affinely to the unit cube, where the null density is $f_0=1$.
For intrinsic dimension $m$, let $v_m=\pi^{m/2}/\Gamma(m/2+1)$ and let
$X_i^{(k)}$ be the $k$th nearest neighbor of $X_i$. The implementation follows
EHY equations (1) and (4):

$$
\xi_{n,J}^{(\alpha)}(X_i)=
 \sum_{k=1}^{J}\left\{v_m n
 \lVert X_i-X_i^{(k)}\rVert^m\right\}^{\alpha},
\qquad
T_{n,J}^{(\alpha)}=
 \sum_{i=1}^n\xi_{n,J}^{(\alpha)}(X_i) f_0(X_i)^\alpha.
$$

Thus `n_neighbors=J` means *all* first $J$ terms are summed. It never means
only the $J$th distance. Theorem 1 and remarks (i), (iv), and (v) require
$\alpha>0$, identify $\alpha=1$ as distribution-free and unusable for this
test, and establish lower-tail rejection for $0<\alpha<1$ and upper-tail
rejection for $\alpha>1$. The domain is $1\le J<n$. The paper permits $m=1$;
pySHT therefore accepts one-feature rectangles rather than inheriting the
unrelated $d\ge2$ Yang--Modarres restriction.

The primary source is [Ebner, Henze, and Yukich
(2018)](https://doi.org/10.1016/j.jmva.2017.12.009).

## Numerical and calibration policy

The sum is evaluated in the log domain; coincident observations and extreme
positive powers therefore have explicit zero/infinite-statistic behavior
without intermediate overflow. Bounds are fixed, finite, and not estimated.
Monte Carlo replicates are iid uniform matrices on the standardized cube; no
nuisance parameter is refitted. A public call consumes $nmB$ uniform variates
from its isolated generator after evaluating the data. Ties enter the selected
tail and $p=(b+1)/(B+1)$.

Literal first-$J$ loops, one-dimensional fixtures, row/feature permutations,
per-coordinate affine maps, duplicates, tail direction, seeded replay, and a
9,999-draw performance test cover the implementation.

## Complexity

For $n$ observations in $m$ dimensions, one statistic evaluation costs
$O(n^2m)$ time and $O(n^2m)$ peak storage for the pairwise coordinate
differences. A $B$-draw calibration therefore costs $O(Bn^2m)$ time while
processing and releasing one null replicate at a time, so storage does not
grow with $B$.

## Release evidence

At $(n,m,J)=(25,2,2)$, each independent 20,000-dataset stream was compared
with its own independently seeded 99,999-statistic reference bank:

| $\alpha$ | reference seed | dataset seed | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 20260914 | 20260915 | 207 (0.01035) | 1,035 (0.05175) | 1,973 (0.09865) |
| 0.5 | 20260932 | 20260916 | 189 (0.00945) | 973 (0.04865) | 1,982 (0.09910) |
| 2.0 | 20260914 | 20260915 | 178 (0.00890) | 986 (0.04930) | 1,991 (0.09955) |
| 2.0 | 20260932 | 20260916 | 207 (0.01035) | 966 (0.04830) | 1,986 (0.09930) |

All pass the release tolerance. Because a finite reference bank is shared
within each stream, that tolerance is
$\max\{0.005,4\sqrt{a(1-a)(1/20000+1/99999)}\}$ at nominal level $a$; the
second reciprocal term accounts for reference-quantile uncertainty. Seed
20260924 generated 2,000 size-40 samples
with independent coordinates from a clipped $N(0.2,0.02^2)$ alternative.
With $\alpha=0.5$, $J=1$, and a 19,999-statistic null bank from seed 20260923,
2,000/2,000 rejected at 0.05. This is a strong clustered-alternative check,
not a universal power claim.

The table is an empirical-bank audit, not 20,000 literal public calls with
9,999 draws each. Focused tests independently replay the exact public null
stream, selected tail, tie rule, and plus-one correction.

# Henze--Zirkler multivariate normality

**Public verdict:** validated and exposed as `pysht.normality.henze_zirkler`.
This is a pySHT-native method, with no SHT 0.1.9 crosswalk entry.

## Null, statistic, and domain

The composite null is that the rows are iid from some nonsingular
$N_p(\mu,\Sigma)$. Let $Y_i=S_n^{-1/2}(X_i-\bar X)$, where $S_n$ uses
denominator $n$, and

$$
\beta=\frac{1}{\sqrt2}\left\{\frac{n(2p+1)}4\right\}^{1/(p+4)}.
$$

The implementation evaluates the paper's empirical-characteristic-function
statistic

$$
\begin{aligned}
HZ=n\bigg[&\frac1{n^2}\sum_{i,j}
 \exp\left\{-\frac{\beta^2}{2}\lVert Y_i-Y_j\rVert^2\right\}\\
&-\frac{2}{n(1+\beta^2)^{p/2}}\sum_i
 \exp\left\{-\frac{\beta^2}{2(1+\beta^2)}\lVert Y_i\rVert^2\right\}
 +(1+2\beta^2)^{-p/2}\bigg].
\end{aligned}
$$

A stable SVD evaluates the row Gram matrix without explicitly inverting
$S_n$. The sample must have $p\ge2$, $n>p$, and full centered column rank.
This gives full-rank affine invariance; a singular covariance is outside the
advertised null rather than silently regularized.

At the boundary $n=p+1$, $YY^T=nI-11^T$: every full-rank sample has the
same fitted regular-simplex geometry. The statistic is therefore constant
under both the null and all full-rank alternatives. pySHT returns $p=1$,
counts every null draw as a tie, and sets the `degenerate affine geometry`
diagnostic to `True`. This case has no power; informative testing requires
$n\ge p+2$. A fixed Helmert basis evaluates the boundary statistic after
the input rank check, avoiding spurious dependence on SVD rounding.

The primary source is [Henze and Zirkler
(1990)](https://doi.org/10.1080/03610929008830400).

## Calibration and reproducibility

Every Monte Carlo draw is an independent $n\times p$ standard-normal matrix
that is re-centered and re-whitened. Refitting the nuisance parameters in each
draw is essential for the composite null. A public call consumes exactly
$npB$ standard-normal variates from the supplied generator, after computing
the observed statistic, and does not touch NumPy's global RNG. Ties enter the
upper-tail count and $p=(b+1)/(B+1)$; the result also reports conditional MCSE
and a 95% Clopper--Pearson interval. Comparisons include numerical ties within
100 float64 epsilons relative to the observed statistic. At $n=p+1$, the
same $npB$ variates are consumed, but every comparison is resolved
analytically as a tie, without refitting simulated matrices. The usual
count-based interval is retained; the true conditional tail probability at
this boundary is known to be one.

Literal eigendecomposition fixtures, full affine transformations, row
permutations, singular cases, and seeded replay of every fitted null replicate
are tested independently of the SVD kernel.

## Complexity

For $n>p$, whitening costs $O(np^2)$ time and the pairwise kernel costs
$O(n^2p)$; each of the $B$ refitted null replicates has the same order. Peak
storage is $O(np+n^2)$ because null replicates are processed one at a time,
not retained as a $B\times n\times n$ tensor.

## Release evidence

At $(n,p)=(20,2)$, each 20,000-dataset stream was compared with its own
independently seeded 99,999-statistic reference bank. The resulting rejection
counts (rates) were:

| reference seed | dataset seed | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|---:|
| 20260911 | 20260912 | 183 (0.00915) | 996 (0.04980) | 2,009 (0.10045) |
| 20260931 | 20260913 | 203 (0.01015) | 975 (0.04875) | 1,938 (0.09690) |

Every entry passes the project's 20,000-dataset tolerance. Since a finite
reference bank is shared by the observed datasets within a stream, that
tolerance is
$\max\{0.005,4\sqrt{a(1-a)(1/20000+1/99999)}\}$ at nominal level $a$; it
includes reference-quantile as well as dataset uncertainty. This empirical-bank
audit is supplemented by public-call RNG replay; it is not represented as
20,000 independent 9,999-draw tests.

For a named power check, seed 20260922 generated 2,000 samples of size 40 from
a bivariate $t_2$ distribution. Against a 19,999-statistic normal reference
bank from seed 20260921, 1,798/2,000 rejected at 0.05. This checks direction and
sensitivity for one heavy-tailed alternative, not minimum power.

The default 9,999-draw path has a focused performance regression.

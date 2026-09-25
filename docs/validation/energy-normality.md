# Energy test of multivariate normality

**Public verdict:** validated and exposed as `pysht.normality.energy`. This is
a pySHT-native method, with no SHT 0.1.9 crosswalk entry.

## Null, statistic, and scale convention

The null is iid $N_p(\mu,\Sigma)$ with nonsingular, unknown covariance. Write
$Y_i=S^{-1/2}(X_i-\bar X)$ using the ordinary sample covariance with
denominator $n-1$, and let $Z,Z'$ be independent $N_p(0,I)$. The statistic is

$$
E_n=n\left\{
 \frac2n\sum_i E\lVert Y_i-Z\rVert
 -E\lVert Z-Z'\rVert
 -\frac1{n^2}\sum_{i,j}\lVert Y_i-Y_j\rVert
\right\}.
$$

The two population terms are evaluated without inner simulation:

$$
E\lVert Y_i-Z\rVert=\sqrt2\,
\frac{\Gamma((p+1)/2)}{\Gamma(p/2)}
{}_1F_1\left(-\frac12;\frac p2;-\frac{\lVert Y_i\rVert^2}{2}\right),
$$

and $E\lVert Z-Z'\rVert=2\Gamma((p+1)/2)/\Gamma(p/2)$.

The $n-1$ covariance convention is a finite-sample part of the published
procedure. As an external fixture, the authors' `energy::mvnorm.e` reference
value for the 50 Iris Setosa rows is 1.203397; pySHT computes
1.2033967029263737. Population-covariance whitening does not reproduce that
fixture and is not used. The primary source is [Székely and Rizzo
(2005)](https://doi.org/10.1016/j.jmva.2003.12.002).

The domain is $p\ge2$, $n>p$, with full centered column rank. Stable SVD
whitening supplies affine invariance and rejects singular covariance instead
of adding an undocumented ridge.

When $n=p+1$, the fitted residual Gram matrix is
$(n-1)(I-11^T/n)$. All full-rank samples have identical norms and pairwise
distances, including nonnormal alternatives. Consequently the statistic is
constant and the p-value is one. pySHT explicitly counts all null draws as
ties and flags `degenerate affine geometry=True`; informative testing requires
$n\ge p+2$. Input rank is still checked before a fixed Helmert basis is used
to evaluate this constant statistic.

## Calibration and reproducibility

The composite normal null is calibrated by standard-normal simulation with
the mean and ordinary sample covariance refitted in every replicate. The
public generator stream therefore consists of exactly $npB$ standard-normal
variates after the observed statistic; the global RNG is untouched. Upper-tail
ties count, $p=(b+1)/(B+1)$, and MCSE and the binomial interval are reported.
Literal covariance/eigendecomposition and hypergeometric fixtures independently
check the production statistic and full seeded null stream.

Numerical ties within 100 float64 epsilons relative to the observed statistic
enter the upper tail. At $n=p+1$, all $npB$ variates are still consumed, but
comparisons are resolved analytically as ties without numerical refitting.
The standard count-based interval remains available, although the underlying
conditional tail probability is known to be one at this boundary.

## Complexity

For $n>p$, SVD whitening costs $O(np^2)$ time and the pairwise energy term
costs $O(n^2p)$; this work repeats for all $B$ refitted null samples. Peak
storage is $O(np+n^2)$ because the implementation retains only one replicate
and its distance vector at a time.

## Release evidence

At $(n,p)=(20,2)$, independent 20,000-dataset streams were each compared with
their own independently seeded 99,999-statistic normal reference bank:

| reference seed | dataset seed | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|---:|
| 20260911 | 20260912 | 193 (0.00965) | 993 (0.04965) | 2,036 (0.10180) |
| 20260931 | 20260913 | 199 (0.00995) | 979 (0.04895) | 1,972 (0.09860) |

All cells pass the release tolerance. Since one independent finite reference
bank is shared within each stream, the gate uses
$\max\{0.005,4\sqrt{a(1-a)(1/20000+1/99999)}\}$ at nominal level $a$ and
accounts for both dataset and reference-quantile uncertainty. These are empirical-bank calibrations,
not 20,000 complete 9,999-draw public calls; focused tests separately replay
the exact public refitting, generator, tie, and plus-one path.

Seed 20260922 generated 2,000 size-40 bivariate $t_2$ alternatives. With a
19,999-statistic normal bank from seed 20260921, 1,845/2,000 rejected at 0.05.
This is a direction/power fixture for a named alternative only. A performance
test guards the default 9,999-draw path.

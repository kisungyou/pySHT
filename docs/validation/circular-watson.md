# Watson $U^2$ circular-uniformity test

**Public verdict:** validated and exposed as `pysht.circular.watson`. This is a
pySHT-native method with no SHT 0.1.9 crosswalk entry.

## Statistic and scope

Let $U_{(i)}$ be the ordered angles divided by $2\pi$, and define

$$
d_i=U_{(i)}-\frac{2i-1}{2n}.
$$

The rotation-invariant statistic is

$$
U^2=\sum_{i=1}^n(d_i-\bar d)^2+\frac1{12n}.
$$

Large values reject circular uniformity against omnibus nonuniform
alternatives. Unlike Rayleigh, the reported alternative is not restricted to
the first harmonic. The primary source is [Watson
(1961)](https://doi.org/10.1093/biomet/48.1-2.109).

Angles are wrapped by an explicit positive finite `period`. Literal ordered-CDF
fixtures plus row, rotation, reflection, wrapping, and degree/radian
transformations verify the statistic. Monte Carlo replicates are iid
circular-uniform samples with no nuisance fit; the public stream consumes
$nB$ uniform variates after the observed statistic. Ties are upper-tail,
$p=(b+1)/(B+1)$, and uncertainty fields are returned.

## Complexity

Sorting the $n$ circular scores makes each statistic $O(n\log n)$ time and
$O(n)$ storage. A direct $B$-draw finite-null calibration is therefore
$O(Bn\log n)$ time with peak storage independent of $B$.

## Release evidence

At $n=20$, each independent 20,000-dataset stream was compared with its own
independently seeded 99,999-statistic reference bank:

| reference seed | dataset seed | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|---:|
| 20260917 | 20260918 | 199 (0.00995) | 990 (0.04950) | 2,042 (0.10210) |
| 20260933 | 20260919 | 210 (0.01050) | 1,000 (0.05000) | 2,003 (0.10015) |

Every entry passes the release tolerance. Since each stream shares one
independent 99,999-statistic bank, the tolerance is
$\max\{0.005,4\sqrt{a(1-a)(1/20000+1/99999)}\}$ at nominal level $a$ and
includes reference-quantile as well as dataset uncertainty. Seed 20260928 generated 2,000
size-20 antipodal alternatives, with ten von Mises observations around zero
and ten around $\pi$, both at concentration 8. Against a 19,999-statistic bank
from seed 20260927, 1,051/2,000 rejected at 0.05. The deliberately
first-moment-free fixture distinguishes Watson's omnibus role from Rayleigh;
it does not state a minimum power guarantee.

The table is an empirical-bank audit, not 20,000 complete public calls. The
public generator sequence and corrected p-value are independently replayed by
focused fixtures.

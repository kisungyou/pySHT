# Modified Hermans--Rasson circular-uniformity test

**Public verdict:** validated and exposed as `pysht.circular.hermans_rasson`.
This is a pySHT-native method with no SHT 0.1.9 crosswalk entry.

## Primary kernel

For wrapped radians $\theta_1,\ldots,\theta_n$, the modified statistic is

$$
T_{HR}=\frac1n\sum_{i=1}^n\sum_{j=1}^n
\left[
\left|\,|\theta_i-\theta_j|-\pi\right|-\frac\pi2
-2.895\left\{|\sin(\theta_i-\theta_j)|-\frac2\pi\right\}
\right].
$$

The sign and coefficient 2.895 are preserved exactly; large values reject.
An explicit double-loop oracle checks the complete kernel, including diagonal
terms. This is an omnibus Sobolev statistic designed for sensitivity to both
unimodal and multimodal departures. The primary source is [Hermans and Rasson
(1985)](https://doi.org/10.1093/biomet/72.3.698).

The angle normalization and invariance contract is the same as Watson's:
positive finite period, wrapping, rotation, reflection, row order, and unit
conversion. Each Monte Carlo replicate is an iid circular-uniform sample with
no fitted nuisance parameters. The isolated public generator consumes $nB$
uniform variates after the observed statistic. Upper-tail ties count and
$p=(b+1)/(B+1)$ with reported MC uncertainty.

## Complexity

The complete pairwise kernel costs $O(n^2)$ time and $O(n^2)$ storage per
statistic. A 9,999-draw performance test guards the resulting $O(Bn^2)$
calibration time; replicates are released immediately, so storage does not
grow with $B$.

## Release evidence

At $n=20$, each independent 20,000-dataset stream was compared with its own
independently seeded 99,999-statistic reference bank:

| reference seed | dataset seed | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|---:|
| 20260917 | 20260918 | 209 (0.01045) | 974 (0.04870) | 2,008 (0.10040) |
| 20260933 | 20260919 | 211 (0.01055) | 1,006 (0.05030) | 1,985 (0.09925) |

All cells pass the release tolerance. Because a finite independent reference
bank is shared by the observed datasets in each stream, the tolerance is
$\max\{0.005,4\sqrt{a(1-a)(1/20000+1/99999)}\}$ at nominal level $a$ and
includes both sources of variation. On the same named antipodal fixture used
for Watson--2,000 size-20 samples from seed 20260928, concentration 8--the
test rejected 2,000/2,000 at 0.05 against a 19,999-statistic bank from seed
20260927. This confirms the intended multimodal direction, not uniform power.

These are empirical-bank counts rather than 20,000 literal public calls;
focused fixtures independently replay the complete public Monte Carlo stream.

# Rayleigh first-harmonic test

**Public verdict:** validated and exposed as `pysht.circular.rayleigh`. This is
a pySHT-native method with no SHT 0.1.9 crosswalk entry.

## Scientific scope and statistic

For normalized angles $\theta_i\in[0,2\pi)$, define

$$
\bar C=\frac1n\sum_i\cos\theta_i,
\qquad
\bar S=\frac1n\sum_i\sin\theta_i,
\qquad
\bar R=(\bar C^2+\bar S^2)^{1/2}.
$$

The reported statistic is the conventional $Z=n\bar R^2$. The null is
circular uniformity, but the alternative is specifically a nonzero first
trigonometric moment--a preferred first-harmonic direction. It is not described
as an omnibus test: antipodally symmetric nonuniform distributions can have
zero first moment.

The result reports $\bar R$. It reports
$\operatorname{atan2}(\bar S,\bar C)$ in the input-period units only when the
resultant is numerically nonzero; otherwise the direction is omitted and an
explicit diagnostic marks it undefined. The implementation and terminology
follow the standard Rayleigh circular-uniformity statistic.

## Units, calibration, and boundaries

Finite values are reduced modulo a positive finite `period` and divided by
that period before multiplication by $2\pi$. This order remains valid for
subnormal and near-maximum float64 periods. Tests cover radians/degrees,
rotation, reflection, wrapping, row order, huge and subnormal periods, and an
exactly antipodal zero-resultant fixture.

The finite-sample null is simulated directly: each replicate contains $n$
iid uniform angles, with no fitted nuisance parameters. A public call consumes
exactly $nB$ uniform variates after the observed statistic. Upper-tail ties
count and $p=(b+1)/(B+1)$; MCSE and a 95% binomial interval are returned.

## Complexity

Each resultant calculation is $O(n)$ time and $O(n)$ storage. Direct
finite-null calibration with $B$ replicates therefore costs $O(Bn)$ time and
retains only one size-$n$ replicate, so peak storage is independent of $B$.

## Release evidence

At $n=20$, each independent 20,000-dataset stream was compared with its own
independently seeded 99,999-statistic reference bank:

| reference seed | dataset seed | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|---:|
| 20260917 | 20260918 | 199 (0.00995) | 1,004 (0.05020) | 1,997 (0.09985) |
| 20260933 | 20260919 | 221 (0.01105) | 996 (0.04980) | 2,008 (0.10040) |

All pass the release tolerance. Because a single independent finite reference
bank is shared within a stream, the gate uses
$\max\{0.005,4\sqrt{a(1-a)(1/20000+1/99999)}\}$ at nominal level $a$; this
accounts for both the 20,000 datasets and reference-quantile variation. For a
targeted power audit, seed 20260928
generated 2,000 size-20 von Mises samples with concentration 2.5. Against a
19,999-statistic null bank from seed 20260927, 2,000/2,000 rejected at 0.05.
This supports the intended first-harmonic direction only.

The table is an empirical-bank audit rather than 20,000 complete public calls;
focused tests separately replay the literal public generator, tie, and
plus-one path.

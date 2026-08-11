# Rectangular-uniformity tests: formula and validation ledger

This ledger covers `pysht.uniformity`. It follows Yang and Modarres,
[Multivariate tests of
uniformity](https://doi.org/10.1007/s00362-015-0715-x), while correcting the
calibration policy where the paper's approximation fails the pySHT release
gate.

## Domain transformation

For declared bounds $a_j<b_j$, both tests map

$$
U_{ij}=\frac{X_{ij}-a_j}{b_j-a_j}.
$$

The implementation evaluates this affine map without overflowing even when a
width such as $10^{308}-(-10^{308})$ is not representable. Bounds are finite,
have one entry per feature, and are fixed under the null. Interpoint tests allow
closed-boundary observations. The normal-quantile test requires
$0<U_{ij}<1$; endpoints are rejected before applying `ppf`.

Tests cover arbitrary per-coordinate affine maps, coordinate permutations,
overflowing widths, all four boundary directions, and malformed support
vectors.

## Interpoint statistics

For $n$ observations in $d\ge2$ dimensions, define

$$
\bar\delta={n\choose2}^{-1}\sum_{i<j}\lVert U_i-U_j\rVert^2,
\qquad
\bar S={n\choose2}^{-1}\sum_{i<j}
\left(\lVert U_i-U_j\rVert^2-d/6\right)^2.
$$

The implementation uses the paper's exact null moments and variances:

$$
Q_1=\frac{(\bar\delta-d/6)^2}{\operatorname{Var}(\bar\delta)},
\qquad
Q_2=\frac{(\bar S-7d/180)^2}{\operatorname{Var}(\bar S)},
\qquad Q_3=Q_1+Q_2.
$$

Independent literal calculations cover the special $d=2$, $d=3$, and
general $d\ge4$ formulas. Fixed results agree with the statistic values in
SHT, but SHT p-values are not used as an oracle.

### The published Q3 independence claim is false

The paper argues that $Q_1$ and $Q_2$ become independent and therefore
$Q_3\Rightarrow\chi^2_2$. The overlapping-pair covariance does not vanish.
For one uniform coordinate, direct polynomial integration gives

$$
E\{(D^2-1/6)^3\}=\frac{11}{945},\qquad
\operatorname{Cov}\{D_{12}^2,(D_{13}^2-1/6)^2\}=\frac{2}{945}.
$$

Applying the paper's own U-statistic overlap counting from its equation (9)
therefore gives the exact finite-$n$ covariance

$$
\operatorname{Cov}(\bar\delta,\bar S)
=\binom n2^{-1}\frac{d(4n+3)}{945},
$$

which is strictly positive. If $Z_1$ and $Z_2$ denote the signed standardized
versions of $\bar\delta$ and $\bar S$, their correlation is this covariance
divided by the square root of the exact variances in the paper's equation (5).
For example, it is `0.884087939361298` at $(n,d)=(50,3)$ and converges to
`0.920087412456472` as $n\to\infty$ for $d=3$.

The paper's chi-square approximation is consequently materially liberal in a
seeded direct null diagnostic. Every cell below is the rejection count out of
20,000 followed by its rate at nominal 0.05:

| $n$ | $d$ | seed | Q1 count/rate | Q2 count/rate | legacy Q3 count/rate |
|---:|---:|---:|---:|---:|---:|
| 20 | 2 | 20260825 | 962 (0.04810) | 947 (0.04735) | 1,413 (0.07065) |
| 50 | 2 | 20260825 | 1,021 (0.05105) | 896 (0.04480) | 1,479 (0.07395) |
| 50 | 3 | 20260825 | 955 (0.04775) | 907 (0.04535) | 1,469 (0.07345) |
| 50 | 10 | 20260825 | 983 (0.04915) | 964 (0.04820) | 1,310 (0.06550) |
| 200 | 3 | 20260825 | 1,020 (0.05100) | 997 (0.04985) | 1,559 (0.07795) |

A fresh `numpy.random.default_rng(20260825)` was reset for each $(n,d)$ row
and generated independent unit-hypercube datasets in replication order. The
production batched component formulas were used, with the legacy
$\chi^2_2$ survival law applied only for this diagnostic; fixed draws were
checked against the public Q1, Q2, and Q3 statistics. The invalid legacy Q3
law is not exposed by pySHT. The paper's own table reports 0.067 for Q3 at
$(n,d)=(10,2)$, consistent with this finding.

pySHT retains the published $Q_3=Q_1+Q_2$ statistic but corrects its optional
asymptotic law. Under the joint U-statistic central limit theorem,

$$
Q_3\ \dot\sim\ (1+\rho)\chi^2_1+(1-\rho)\widetilde\chi^2_1,
$$

where the two chi-square variables are independent and $\rho$ is the exact
finite-$n$ correlation above. The survival probability is evaluated by an
independent polar-angle integral. A separate convolution of the two weighted
chi-square densities is the fixed-fixture oracle in the tests.

Two 20,000-replication audits of the corrected approximation gave the
following raw counts and rates:

| $(n,d)$ | seed | 0.01 count/rate | 0.05 count/rate | 0.10 count/rate | gate |
|---:|---:|---:|---:|---:|---:|
| (50, 2) | 20260825 | 222 (0.01110) | 971 (0.04855) | 1,894 (0.09470) | Pass |
| (50, 2) | 20260826 | 242 (0.01210) | 980 (0.04900) | 1,936 (0.09680) | Pass |
| (50, 3) | 20260825 | 229 (0.01145) | 955 (0.04775) | 1,887 (0.09435) | Pass |
| (50, 3) | 20260826 | 236 (0.01180) | 962 (0.04810) | 1,913 (0.09565) | Pass |
| (50, 10) | 20260825 | 227 (0.01135) | 936 (0.04680) | 1,967 (0.09835) | Pass |
| (50, 10) | 20260826 | 266 (0.01330) | 1,001 (0.05005) | 1,988 (0.09940) | Pass |
| (200, 3) | 20260825 | 218 (0.01090) | 989 (0.04945) | 1,996 (0.09980) | Pass |
| (200, 3) | 20260826 | 213 (0.01065) | 1,003 (0.05015) | 2,054 (0.10270) | Pass |

For every scenario and seed, the generator was reset before drawing
unit-hypercube datasets. The batched audit evaluated the literal public Q3
statistic formula and the production correlated-square survival law; fixed
draws were independently matched to the complete public-call path. Rejection
meant `pvalue < alpha`.

The smaller $(n,d)=(20,2)$ design failed at level 0.10 in both seeds
(1,822/20,000 = 0.09110 for seed 20260825 and 1,802/20,000 = 0.09010
for seed 20260826), so pySHT does not advertise the corrected asymptotic law
in that regime.

### Why Monte Carlo remains authoritative

Consequently, `ym_interpoint` defaults to a parametric Monte Carlo uniform
null for all three variants. It uses the corrected
$(b+1)/(B+1)$ p-value, counts upper-tail ties, reports Monte Carlo uncertainty,
and accepts an isolated/replayable `rng`. A literal seeded simulation
independently reproduces its statistic vector and exceedance count.

`calibration="asymptotic"` uses the corrected correlated-square law for `q3`.
The legacy $\chi^2_2$ calculation is not exposed because it is not the
asymptotic distribution of the statistic.

## Normal-quantile statistic

Set $Z_{ij}=\Phi^{-1}(U_{ij})$. Under the rectangular-uniform null, rows of
$Z$ are exactly $N_d(0,I)$, so

$$
C_N=n\lVert\bar Z\rVert^2\sim\chi^2_d.
$$

This is an exact null result, rather than an asymptotic approximation. Two
20,000-sample audits at $(n,d)=(50,3)$ gave:

| seed | 0.01 count/rate | 0.05 count/rate | 0.10 count/rate |
|---:|---:|---:|---:|
| 20260817 | 170 (0.00850) | 957 (0.04785) | 1,975 (0.09875) |
| 20260818 | 196 (0.00980) | 1,017 (0.05085) | 1,988 (0.09940) |

For each seed, a fresh generator produced 20,000 independent unit-hypercube
datasets in order and every replication called public `ym_quantile`; rejection
meant `result.pvalue < alpha`. Both satisfy the release gate.

## Targeted alternative-power audit

All public paths were checked against 2,000 samples at $(n,d)=(50,3)$ with
independent `Beta(5, 1)` coordinates, generated in row order by
`numpy.random.default_rng(20260829)`. For the authoritative Monte Carlo
interpoint calibration, `numpy.random.default_rng(20260830)` generated a
shared bank of 100,000 unit-hypercube samples. Production batched component
kernels counted upper-tail ties and rejection used
$(b+1)/(100000+1)<0.05$; fixed draws were matched to the complete public Monte
Carlo path. Public `ym_quantile` was called once per alternative dataset.

| Public path | Rejections/2,000 | Rate |
|---|---:|---:|
| `ym_interpoint(statistic="q1")` | 2,000 | 1.0000 |
| `ym_interpoint(statistic="q2")` | 1,711 | 0.8555 |
| `ym_interpoint(statistic="q3")` | 2,000 | 1.0000 |
| `ym_quantile` | 2,000 | 1.0000 |

This is a targeted sensitivity check for a strong skewed alternative, not a
uniform power guarantee over every departure from rectangular uniformity.

## Legacy mapping

| pySHT | SHT 0.1.9 |
|---|---|
| `ym_interpoint` | `unif.2017YMi` |
| `ym_quantile` | `unif.2017YMq` |

# Zhang–Xu k-sample Scheffé transformation

**Status:** primary-paper transformation, both base tests, dimension,
scale, and boundary gates pass. Row-pairing sensitivity is intrinsic and
documented.

After ordering groups by sample size, Zhang and Xu transform the unequal-
covariance k-sample problem into a one-sample problem with `n_min` rows and
`(k - 1) p` columns. `zx_ksamp` then applies either the Bai–Saranadasa test or,
when dimension permits, the exact Gaussian Hotelling test.

Section 2 of the primary paper (pp. 4–5) gives the transformation and the
resulting independent Gaussian sample in Equations (2.2)–(2.3), after assuming
`n_1 <= ... <= n_k`. pySHT evaluates that display literally.

For a larger group `l`, the transformation explicitly uses its first
`n_min` observations both in the partial mean and paired row term. Therefore,
permuting rows in that group can change the realized covariance and test
statistic. This is part of the published Scheffé construction, whose null
distribution remains valid for arbitrarily ordered i.i.d. rows; a row-order
invariance gate is consequently not applicable. The unit ledger deliberately
demonstrates this sensitivity. Users should ensure row order is unrelated to
the measurements and should not sort rows by an outcome before testing.
When several groups tie for the minimum sample size, their input order resolves
the paper's otherwise unspecified reference tie. Relabeling such tied groups
can therefore change the realized statistic. A data-dependent tie-break would
compromise the fixed-design Gaussian transformation, so pySHT does not
silently reorder tied groups by their observed values.

One shared original-coordinate feature anchor is removed from all groups
before constructing the transformation. This changes neither the Scheffé
blocks nor their units and prevents a huge common location from driving the
working scale.

## Null calibration gate for the default base test

The advertised scenario uses three independent balanced Gaussian groups with
`n_i=30`, `p=500`, and identity covariance. After the published transformation
the default Bai–Saranadasa branch has 30 rows and 1000 columns. The independent
batched implementation evaluates 20,000 complete null transformations per
seed.

| Seed | alpha=0.01 | alpha=0.05 | alpha=0.10 |
|---|---:|---:|---:|
| 20260810 | 0.01235 | 0.05105 | 0.10125 |
| 20260811 | 0.01100 | 0.05035 | 0.09935 |

Both rows pass the release tolerance at every level. The optional Hotelling
base has exact Gaussian F calibration when its dimension requirement holds
and is covered by formula fixtures rather than this asymptotic gate.

## Targeted alternative-power gate for the default base test

Seed 2026090318 initializes a `SeedSequence`; child 0 drives one persistent
PCG64 data stream and child 1 is a separate persistent PCG64 auxiliary stream,
unused here. In each of 1,000 replications, child 0 draws standard-normal
matrices with `(n,p)=(25,30)`, `(30,30)`, and `(35,30)` in call order. Their
columns are multiplied respectively by `linspace(0.8,1.2,30)`,
`linspace(1.0,1.5,30)`, and `linspace(0.7,1.1,30)`; the second and third group
means are then shifted by $+0.45\mathbf 1$ and $-0.45\mathbf 1$. The public
call is `mean.zx_ksamp(x, y, z)`, exercising the default Bai–Saranadasa base.
Streams advance in replication order. For `pvalue < alpha`:

| alpha | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|
| Rejections / 1,000 | 1,000 | 1,000 | 1,000 |
| Rate | 1.000 | 1.000 | 1.000 |

The intentionally strong dense alternative checks response under unequal
sample sizes and unequal diagonal covariance; it is not a broad power claim.
`python -m tools.mean_power_audits` reproduces the result under Python
3.12.13, NumPy 2.5.1, and SciPy 1.18.0. The worst-case binomial standard error
at 1,000 outer replications is 0.0159.

Primary reference: J.-T. Zhang and J. Xu, *On the k-Sample Behrens–Fisher
Problem for High-Dimensional Data*, Science in China Series A 52 (2009),
1285–1304, doi:10.1007/s11425-009-0091-x.

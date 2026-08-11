# Dempster non-exact mean tests

**Status:** formula, independent fixture, invariance, boundary, and advertised
Gaussian null-calibration gates pass.

The one- and two-sample functions implement Dempster's Euclidean mean-square
ratio. For within-group degrees of freedom `nu`, the effective rank is
estimated from `tr(Sigma)` and `tr(Sigma²)` and the statistic is calibrated by
`F(floor(r), floor(nu r))`. This calibration assumes multivariate normality;
it does not require an invertible sample covariance.

## Audit findings

- `dempster_1samp` agrees with the trace formula in the pinned SHT source and
  an independent NumPy/SciPy implementation.
- The legacy `mean2.1958Dempster` computed a positive F-type ratio but passed
  it to a standard-normal upper tail. `dempster_2samp` repairs that mismatch
  and uses Dempster's approximate F law.
- Both routines are invariant to a common nonzero scale; the two-sample
  routine is invariant to exchanging groups.
- Null centering and shared two-sample anchoring happen before numerical
  scaling. Tests at common locations up to `1e14` reproduce the attainable
  result from the translated float64 observations.
- Singular or high-dimensional covariance estimates are allowed, but the
  estimated trace and squared-trace correction must be positive.

## Null calibration gate

The advertised scenario is independent Gaussian data with identity
covariance, `p=3000`, and 50 covariance degrees of freedom: `n=51` for the
one-sample test and `n_x=n_y=26` for the two-sample test. Under this null the
two designs have the same independent chi-square mean component and Wishart
covariance component. We sampled those sufficient statistics directly and
cross-checked their formulas against the full-data fixture.

| Seed | alpha=0.01 | alpha=0.05 | alpha=0.10 |
|---|---:|---:|---:|
| 20260810 | 0.00885 | 0.04660 | 0.09875 |
| 20260811 | 0.01015 | 0.05025 | 0.09890 |

Each row uses 20,000 null datasets. Both one- and two-sample routines have
these rates and satisfy the preregistered scale-aware release tolerance at
all three levels. This gate advertises the stated Gaussian identity regime;
it is not a blanket finite-sample guarantee for arbitrary spectra.

## Targeted alternative-power gate

The public functions were also exercised end to end under dense Gaussian
mean shifts with `p=40 > n=30`. For `dempster_1samp`, child-0 PCG64 draws 30
rows from $N_{40}(0.35\mathbf 1,I)$ and the public call is
`mean.dempster_1samp(x)`. For `dempster_2samp`, it draws 30 rows from
$N_{40}(0.45\mathbf 1,I)$ and then 30 rows from $N_{40}(0,I)$ before
`mean.dempster_2samp(x, y)`. Each named seed initializes a `SeedSequence`;
child 0 is the persistent data stream and child 1 is a separate persistent
PCG64 auxiliary stream, unused here. Streams advance in replication order
and reset between methods. Rejection means `pvalue < alpha`.

| Public function | Seed | Replications | alpha=0.01 count/rate | alpha=0.05 count/rate | alpha=0.10 count/rate |
|---|---:|---:|---:|---:|---:|
| `dempster_1samp` | 2026090306 | 1,000 | 1,000 / 1.000 | 1,000 / 1.000 | 1,000 / 1.000 |
| `dempster_2samp` | 2026090307 | 1,000 | 1,000 / 1.000 | 1,000 / 1.000 | 1,000 / 1.000 |

These are strong-alternative detection checks, not comparative power claims.
They are reproduced by `python -m tools.mean_power_audits` under Python
3.12.13, NumPy 2.5.1, and SciPy 1.18.0. The worst-case binomial standard error
for 1,000 outer replications is 0.0159.

Primary references: A. P. Dempster, *A High Dimensional Two Sample
Significance Test* (1958), and *A Significance Test for the Separation of Two
Highly Multivariate Small Samples* (1960).

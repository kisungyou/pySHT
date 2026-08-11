# Srivastava–Du diagonal mean tests

**Status:** formula, independent fixture, scale, exchange, and boundary gates
pass in the advertised high-dimensional asymptotic regime.

These tests replace the full covariance inverse with marginal variance
standardization and estimate the calibration variance through `tr(R²)`, where
`R` is the sample correlation matrix. Every tested feature must have positive
variance. The two-sample form assumes a common covariance matrix.

The independent ledger recomputes the diagonal quadratic form, finite-sample
centering, `tr(R²)` correction, and standard-normal tail without using pySHT
helpers. It also checks invariance to separate positive feature rescalings and
to exchanging the two groups.

Null vectors are removed before scaling and two groups use one deterministic
feature-wise anchor. Regressions at common locations `1e8` and `1e14` compare
against the information still representable in the translated float64 input.

## Null calibration gate

The advertised high-dimensional scenario uses independent Gaussian
identity-covariance data with `p=100` and 50 covariance degrees of freedom:
`n=51` for `sd_1samp` and `n_x=n_y=26` for `sd_2samp`. The audit directly
samples the independent normal mean component and Wishart covariance
component, including the complete sample-correlation correction.

| Seed | alpha=0.01 | alpha=0.05 | alpha=0.10 |
|---|---:|---:|---:|
| 20260810 | 0.01245 | 0.05080 | 0.09605 |
| 20260811 | 0.01260 | 0.05015 | 0.09495 |

Each row uses 20,000 null datasets. The one- and two-sample designs share
these rates and pass the release tolerance at all three levels. This does not
extend the calibration claim to arbitrary covariance spectra.

## Targeted alternative-power gate

The public standardized-coordinate paths were run with `p=40 > n=30` and
identity covariance. For `sd_1samp`, child-0 PCG64 draws 30 rows from
$N_{40}(0.18\mathbf 1,I)$ before `mean.sd_1samp(x)`. For `sd_2samp`, it draws
30 rows from $N_{40}(0.25\mathbf 1,I)$ and then 30 rows from $N_{40}(0,I)$
before `mean.sd_2samp(x, y)`. Each named integer seed initializes a
`SeedSequence`; child 0 is the persistent data stream and child 1 is a
separate persistent PCG64 auxiliary stream, unused here. Streams advance in
replication order and reset between methods. Rejection means
`pvalue < alpha`.

| Public function | Seed | Replications | alpha=0.01 count/rate | alpha=0.05 count/rate | alpha=0.10 count/rate |
|---|---:|---:|---:|---:|---:|
| `sd_1samp` | 2026090310 | 1,000 | 848 / 0.848 | 934 / 0.934 | 963 / 0.963 |
| `sd_2samp` | 2026090311 | 1,000 | 824 / 0.824 | 935 / 0.935 | 958 / 0.958 |

These deliberately strong alternatives test response direction, not relative
efficiency. `python -m tools.mean_power_audits` reproduces the counts under
Python 3.12.13, NumPy 2.5.1, and SciPy 1.18.0. The worst-case binomial
standard error for 1,000 outer replications is 0.0159.

Primary reference: M. S. Srivastava and M. Du, *A Test for the Mean Vector
with Fewer Observations than the Dimension*, Journal of Multivariate Analysis
99 (2008), 386–402.

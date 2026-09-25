# Schott high-dimensional one-way MANOVA

**Status:** primary-paper formula, independent fixture, exchange, scaling,
and boundary gates pass in the advertised Gaussian asymptotic regime.

The MANOVA error sum-of-products matrix is

`E = sum_i (n_i - 1) S_i`,

with error degrees of freedom `N - k`. The raw `Tnp` statistic is returned;
its standardized normal statistic is included in diagnostics.

The pinned SHT implementation used `sum_i n_i S_i` while retaining `N-k` as
the error degrees of freedom. That is not the defining error SSP. pySHT uses
the paper formula. A literal NumPy implementation independently checks `E`,
the hypothesis SSP, variance estimator, standardized statistic, and upper
normal tail. Group exchange leaves the result unchanged; multiplying every
observation by `c` multiplies raw `Tnp` by `c²` and leaves its p-value
unchanged.

All groups are first expressed relative to one deterministic feature-wise
anchor. This preserves the raw statistic's squared measurement units while
avoiding location-driven cancellation; regressions cover common locations
through `1e14`.

The error trace and squared trace are evaluated from the smaller centered row
or feature Gram, while the hypothesis trace is a weighted sum of squared mean
contrasts. The high-dimensional implementation therefore does not allocate
either MANOVA $p\times p$ sum-of-products matrix, and a tall low-dimensional
input avoids a quadratic row Gram. A 5,000-feature regression guards the
high-dimensional storage property.

## Null calibration gate

The advertised Gaussian scenario has three independent groups of 20 rows,
`p=500`, and identity covariance. The audit uses exact normal/Wishart
sufficient-statistic algebra, independently matched to the full-data formula
fixture, for 20,000 null datasets per seed.

| Seed | alpha=0.01 | alpha=0.05 | alpha=0.10 |
|---|---:|---:|---:|
| 20260810 | 0.01285 | 0.05335 | 0.10265 |
| 20260811 | 0.01215 | 0.05275 | 0.10060 |

Both rows pass the release tolerance at all three levels. The claim is scoped
to the stated balanced Gaussian high-dimensional regime.

## Targeted alternative-power gate

Seed 2026090316 initializes a `SeedSequence`; child 0 drives one persistent
PCG64 data stream and child 1 drives a separate persistent PCG64 auxiliary
stream, unused by this deterministic test. In each of 1,000 replications,
child 0 draws, in call order, three 20-by-40 independent Gaussian matrices
with identity covariance and mean vectors $0$, $0.35\mathbf 1$, and
$-0.35\mathbf 1$. The public call is
`mean.schott_ksamp(x, y, z)`. Streams advance in replication order. The
counts for `pvalue < alpha` are:

| alpha | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|
| Rejections / 1,000 | 1,000 | 1,000 | 1,000 |
| Rate | 1.000 | 1.000 | 1.000 |

This deliberately dense and strong alternative checks response direction in
the `p>n_i` path; it is not a general power guarantee. It is reproduced by
`python -m tools.mean_power_audits` under Python 3.12.13, NumPy 2.5.1, and
SciPy 1.18.0. The worst-case binomial standard error at 1,000 outer
replications is 0.0159.

Primary reference: J. R. Schott, *Some High-Dimensional Tests for a One-Way
MANOVA*, Journal of Multivariate Analysis 98 (2007), 1825–1839,
doi:10.1016/j.jmva.2006.11.007.

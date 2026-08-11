# Bai–Saranadasa trace mean tests

**Status:** formula, independent fixture, transformation, and boundary gates
pass in the advertised high-dimensional asymptotic regime.

The statistic centers the squared Euclidean mean difference by `tr(S)` and
uses the unbiased Wishart correction for `tr(Sigma²)`. The two-sample version
assumes a common covariance matrix. Its upper-tail normal calibration is
asymptotic and is intended for increasing sample size and dimension.

## Audit findings

- Literal implementations independently reproduce both statistics and
  p-values.
- The legacy one-sample denominator cancelled an `n + 1` term where the
  squared-covariance correction requires `n + 2`. pySHT uses
  `2 n (n + 1) / ((n - 1)(n + 2))`.
- Common scaling and orthogonal feature transformations leave the result
  unchanged; exchanging the two samples leaves `bs_2samp` unchanged.
- Null centering and shared two-sample anchoring precede numerical scaling;
  large-location regressions include `1e8` and `1e14`.
- A nonpositive trace-variance estimate is rejected rather than clipped.

## Null calibration gate

For independent Gaussian identity-covariance data, we used `p=3000` and 50
covariance degrees of freedom (`n=51` one-sample; `n_x=n_y=26` two-sample).
The normal mean component and Wishart covariance component are independent,
so the audit samples those sufficient statistics directly; fixed full-data
fixtures verify the same implementation formula.

| Seed | alpha=0.01 | alpha=0.05 | alpha=0.10 |
|---|---:|---:|---:|
| 20260810 | 0.01025 | 0.04840 | 0.09975 |
| 20260811 | 0.01120 | 0.05165 | 0.10040 |

Each row represents 20,000 null datasets. The one- and two-sample designs
share these rates and pass the release tolerance at every level. Smaller
dimensions can retain visible chi-square skewness, so this is a deliberately
specific advertised asymptotic regime rather than a small-sample claim.

## Targeted alternative-power gate

The end-to-end alternative audit retains the high-dimensional design
`p=40 > n=30`. For `bs_1samp`, child-0 PCG64 draws 30 rows from
$N_{40}(0.18\mathbf 1,I)$ and calls `mean.bs_1samp(x)`. For `bs_2samp`, it
draws 30 rows from $N_{40}(0.25\mathbf 1,I)$ and then 30 rows from
$N_{40}(0,I)$ before `mean.bs_2samp(x, y)`. Each named integer seed starts a
`SeedSequence`; child 0 is one persistent data stream and child 1 is a
separate persistent PCG64 auxiliary stream, unused here. Streams advance in
replication order and reset between methods. Rejection means
`pvalue < alpha`.

| Public function | Seed | Replications | alpha=0.01 count/rate | alpha=0.05 count/rate | alpha=0.10 count/rate |
|---|---:|---:|---:|---:|---:|
| `bs_1samp` | 2026090308 | 1,000 | 878 / 0.878 | 946 / 0.946 | 968 / 0.968 |
| `bs_2samp` | 2026090309 | 1,000 | 862 / 0.862 | 948 / 0.948 | 979 / 0.979 |

This strong-alternative gate shows directional sensitivity in the advertised
high-dimensional path; it is not a ranking against other tests. Run
`python -m tools.mean_power_audits` to reproduce it. The recorded engine was
Python 3.12.13 with NumPy 2.5.1 and SciPy 1.18.0; the worst-case binomial
standard error for 1,000 outer replications is 0.0159.

Primary reference: Z. Bai and H. Saranadasa, *Effect of High Dimension: by an
Example of a Two Sample Problem*, Statistica Sinica 6 (1996), 311–329.

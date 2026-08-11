# Multivariate Behrens–Fisher approximations

**Status:** literal matrix-formula, univariate-limit, exchange, scale, and
boundary gates pass.

This ledger covers `yao_2samp`, `johansen_2samp`, `nvm_2samp`, and
`ky_2samp`. All four use

`T² = (xbar - ybar)' (Sx / nx + Sy / ny)^(-1) (xbar - ybar)`

and differ in their approximate degrees of freedom and F adjustment. The
estimated covariance of the mean difference must be positive definite;
Johansen's construction additionally needs each covariance contribution to
be positive definite.

## Independent checks

- Every trace, matrix weight, adjustment, degrees of freedom, and F tail is
  recomputed literally from the papers' displayed formulas.
- Yao, Nel–Van der Merwe, and Krishnamoorthy–Yu reduce to the squared Welch
  t statistic and Welch p-value when there is one feature.
- All four are invariant to exchanging samples and a common nonzero scale.
- Both samples are shifted by one shared, deterministic feature-wise anchor
  before scaling; this is covered at a common location of `1e14`.
- When sample means agree exactly, the statistic is zero and the upper-tail
  probability is one. Yao's direction-dependent degrees of freedom are then
  omitted because their ratio is undefined even though the limiting p-value
  is not.

## Null calibration gate

We simulated 20,000 full datasets per seed with `p=2`, `n_x=80`, `n_y=100`,
`Sigma_x=I`, and
`Sigma_y = A' A` for `A=[[1.5, 0.3], [0, 0.7]]`. Thus the gate exercises the
unequal-covariance case rather than only a pooled-covariance special case.

| Method | Seed | alpha=0.01 | alpha=0.05 | alpha=0.10 |
|---|---:|---:|---:|---:|
| Yao | 20260810 | 0.01020 | 0.04930 | 0.10130 |
| Yao | 20260811 | 0.01135 | 0.05115 | 0.10050 |
| Johansen | 20260810 | 0.01005 | 0.04930 | 0.10125 |
| Johansen | 20260811 | 0.01130 | 0.05120 | 0.10035 |
| Nel–Van der Merwe | 20260810 | 0.01025 | 0.04940 | 0.10150 |
| Nel–Van der Merwe | 20260811 | 0.01135 | 0.05140 | 0.10065 |
| Krishnamoorthy–Yu | 20260810 | 0.01020 | 0.04935 | 0.10130 |
| Krishnamoorthy–Yu | 20260811 | 0.01130 | 0.05125 | 0.10035 |

All rows pass the release tolerance at the three nominal levels. These are
low-dimensional approximation gates and do not waive each method's
positive-definiteness requirements.

## Targeted alternative-power gate

Each method uses a separately reset unequal-covariance Gaussian experiment.
Child-0 PCG64 first draws `x` as 50 rows from
$N_3((0.8,0.5,0)^\mathsf{T},I_3)$, then draws `y` as 60 independent standard
normal rows and multiplies its columns by $D=(1.5,0.7,1.2)$. Thus
$Y\sim N_3(0,D^2)$ and both the mean alternative and unequal covariance
regime are present. The public call is the named function with `(x, y)` and
no optional arguments. Each integer seed initializes `SeedSequence(seed)`;
child 0 is the persistent data PCG64 stream and child 1 is a separate
persistent PCG64 auxiliary stream, unused by these deterministic methods.
Streams advance in replication order and reset between methods. A rejection
is `pvalue < alpha`.

| Public function | Seed | Replications | alpha=0.01 count/rate | alpha=0.05 count/rate | alpha=0.10 count/rate |
|---|---:|---:|---:|---:|---:|
| `yao_2samp` | 2026090312 | 1,000 | 888 / 0.888 | 967 / 0.967 | 987 / 0.987 |
| `johansen_2samp` | 2026090313 | 1,000 | 882 / 0.882 | 974 / 0.974 | 986 / 0.986 |
| `nvm_2samp` | 2026090314 | 1,000 | 913 / 0.913 | 974 / 0.974 | 986 / 0.986 |
| `ky_2samp` | 2026090315 | 1,000 | 884 / 0.884 | 969 / 0.969 | 990 / 0.990 |

This is a strong-alternative response gate, not a statistically powered
comparison between the approximations. It is reproducible with
`python -m tools.mean_power_audits` under Python 3.12.13, NumPy 2.5.1, and
SciPy 1.18.0. The worst-case binomial standard error for 1,000 outer
replications is 0.0159.

Primary references: Yao (1965), Johansen (1980), Nel and Van der Merwe
(1986), and Krishnamoorthy and Yu (2004).

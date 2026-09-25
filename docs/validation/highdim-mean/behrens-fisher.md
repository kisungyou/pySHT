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
- Yao, Johansen, Nel–Van der Merwe, and Krishnamoorthy–Yu reduce to the squared Welch
  t statistic and Welch p-value when there is one feature.
- Johansen uses $c=p+2A-6A/(p+2)$ and $\nu_2=p(p+2)/(3A)$.
  The September 2026 correction replaced an incorrect denominator that
  happened to agree at $p=2$. A separate general contrast-matrix construction
  checks dimensions 1, 2, 3, and 5, including the scalar Welch identity.
  The generalized formula is displayed in the
  [welchADF mathematical formulation](https://journal.r-project.org/articles/RJ-2017-049/).
- All four are invariant to exchanging samples and a common nonzero scale.
  Yao, Johansen, and Krishnamoorthy--Yu are also affine-invariant; their
  implementation uses that exact invariance to normalize columns separately
  before solving covariance systems. A regression with column scales spanning
  approximately $10^{-150}$ through $10^{150}$ verifies that valid covariance
  directions are not lost merely because features use very different units.
  Nel--Van der Merwe is intentionally excluded from this transformation because
  its trace-based degrees-of-freedom approximation is not affine-invariant.
  Its affine-invariant $T^2$ component is nevertheless evaluated in normalized
  coordinates, while its trace formula retains one common scale; this prevents
  a spurious singular-matrix failure without changing the approximation's
  genuine dependence on relative feature units.
- Both samples are shifted by one shared, deterministic feature-wise anchor
  before scaling; this is covered at a common location of `1e14`.
- When sample means agree exactly, the statistic is zero and the upper-tail
  probability is one. Yao's direction-dependent degrees of freedom are then
  omitted because their ratio is undefined even though the limiting p-value
  is not.
- Rank is checked on column-equilibrated, independently centered observations
  using an SVD. This rejects dependent directions even if a
  rounded covariance happens to admit Cholesky factorization. The common
  covariance union is checked for all four procedures; Johansen additionally
  checks each group. A full-rank union of individually singular samples
  remains permitted for the other three approximations.
  The same factor directly whitens the weighted residuals defining
  $S_x/n_x+S_y/n_y$: quadratics and covariance weights are evaluated in those
  coordinates without squaring the residual matrix's condition number.
  Nel--Van der Merwe retains its original-coordinate trace formula for the
  degrees of freedom. Separate regressions cover near-collinear features and
  column-selective overflow fallback, where a very large range in one feature
  must not erase representable increments in another.

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
The historical $p=2$ Johansen rows are unchanged by the formula correction;
they cannot serve as evidence that its dimension-dependent adjustment was
implemented correctly.

After the correction, 20,000 fresh full public calls per row used independent
zero-mean Gaussian groups with $\Sigma_x=I$ and
$\Sigma_y=\operatorname{diag}\{\operatorname{linspace}(1.2,2,p)^2\}$.
The persistent PCG64 stream draws $x$ and then $y$ in each replication.
Every call was also compared with an independent general contrast-matrix
formula; all rejection counts agreed, and the largest absolute p-value
difference over the 80,000 datasets was $8.78\times10^{-15}$ after the
observation-SVD whitening correction.

| Dimension | Sample sizes | Seed | Counts at 0.01, 0.05, 0.10 | Rate at 0.05 | Exact 95% interval for that rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 5, 7 | 2026091001 | 166, 937, 1920 | 0.04685 | [0.04396, 0.04987] |
| 2 | 8, 10 | 2026091002 | 199, 1047, 2034 | 0.05235 | [0.04930, 0.05553] |
| 3 | 8, 10 | 2026091003 | 186, 1029, 2014 | 0.05145 | [0.04843, 0.05460] |
| 5 | 15, 20 | 2026091005 | 184, 1042, 2114 | 0.05210 | [0.04906, 0.05527] |

This is evidence for the corrected implementation and for these particular
finite-sample designs. The univariate interval excludes 0.05, illustrating
that Welch--James remains an approximation even when its formula is correct
and the broader release tolerance passes.

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
| `johansen_2samp` | 2026090313 | 1,000 | 884 / 0.884 | 975 / 0.975 | 987 / 0.987 |
| `nvm_2samp` | 2026090314 | 1,000 | 913 / 0.913 | 974 / 0.974 | 986 / 0.986 |
| `ky_2samp` | 2026090315 | 1,000 | 884 / 0.884 | 969 / 0.969 | 990 / 0.990 |

This is a strong-alternative response gate, not a statistically powered
comparison between the approximations. It is reproducible with
`python -m tools.mean_power_audits` under Python 3.12.13, NumPy 2.5.1, and
SciPy 1.18.0. The worst-case binomial standard error for 1,000 outer
replications is 0.0159.
The Johansen row was regenerated after the September 2026 adjustment
correction under Python 3.12.14 with the same NumPy, SciPy, and random-stream
contract; the other rows retain their original evidence.

Primary references: Yao (1965), Johansen (1980), Nel and Van der Merwe
(1986), and Krishnamoorthy and Yu (2004).

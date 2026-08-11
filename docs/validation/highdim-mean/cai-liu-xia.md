# Cai–Liu–Xia maximum mean test

**Status: validation-blocked; no public callable.** The oracle formula,
extreme-tail, transformation, estimator-program, and invalid-input gates pass,
but the practical-size Gumbel calibration does not. A reduced oracle null law
passes only at a very large dimension; no honest end-to-end release scenario
currently passes for the estimated-precision branches.

For a precision matrix `Omega`, the oracle statistic is

`n_x n_y / (n_x + n_y) * max_j ((Omega d)_j² / Omega_jj)`.

The p-value uses the limiting type-I extreme-value distribution and is
evaluated with `expm1` to avoid cancellation. A literal independent fixture
checks the statistic, log-tail, group exchange, and diagonal feature scaling
with the correspondingly transformed precision matrix.

The direct paper locations are Equation (2), p. 352, for the oracle statistic;
Equations (6)–(7), pp. 354–355, for the data-driven statistic and transformed
variance; and Theorem 1, pp. 355–356, for
`M - 2 log(p) + log(log(p))` and its type-I extreme-value limit. The CLIME
program and minimum-magnitude symmetrization are displayed on pp. 353–354.
These comparisons use the authors' article, not the legacy R output as a
formula oracle.

## Estimated precision

- `precision="clime"` solves the column-wise CLIME linear programs with
  SciPy, applies the paper's minimum-magnitude symmetrization, and uses
  `lambda = sqrt(log(p) / (n_x + n_y))`, i.e. the theoretical rate with
  constant one. The paper permits cross-validation of that constant; pySHT
  intentionally makes the deterministic rule explicit and does not silently
  run a data-dependent search.
- `precision="adaptive-threshold"` uses the entry-specific thresholds with
  default `delta=2`, followed by a documented positive-definite numerical
  floor before inversion.
- Both estimated branches assume a common covariance matrix. No unequal-
  covariance fallback is implied.

All groups use one deterministic feature-wise anchor before per-feature
scaling. Supplied precision matrices remain defined in the original column
units; an explicit domain error is raised if transforming such a matrix would
overflow float64.

## Null calibration audit and current limitation

Under independent Gaussian coordinates and a supplied identity precision,
the oracle statistic reduces exactly to the maximum of `p` independent
chi-square-one variables. This gives an exact, allocation-free simulator for
the implemented limiting p-value. With 20,000 null datasets per seed:

| p | Seed | alpha=0.01 | alpha=0.05 | alpha=0.10 | Gate |
|---:|---:|---:|---:|---:|---|
| 10,000 | 20260810 | 0.00770 | 0.04650 | 0.09595 | pass |
| 10,000 | 20260811 | 0.00810 | 0.04385 | 0.08920 | fail at 0.10 |
| 300,000 | 20260810 | 0.00810 | 0.04905 | 0.09825 | pass |
| 300,000 | 20260811 | 0.00825 | 0.04515 | 0.09170 | pass |

The extreme-value approximation converges slowly. The passing `p=300,000`
row uses the exact reduced oracle law; the current dense precision-matrix API
cannot make that an honest end-to-end execution scenario. No null-calibration
claim is therefore made for `precision="clime"` or
`precision="adaptive-threshold"`. The method remains private until a practical
sparse-covariance scenario passes the same gate; pySHT never substitutes a
different calibration silently.

## Alternative-power status

No public alternative-power result is reported. The implementation remains
private as `_clx_2samp`, is absent from `pysht.mean.__all__`, and cannot pass
the public release gate until an end-to-end sparse-covariance null scenario
passes. A strong-alternative result would not repair that missing type-I-error
evidence, so the power audit deliberately does not promote or advertise it.

Primary reference: T. T. Cai, W. Liu, and Y. Xia, *Two-Sample Test of High
Dimensional Means under Dependence*, JRSS B 76 (2014), 349–372,
doi:10.1111/rssb.12034; author copy
<https://faculty.wharton.upenn.edu/wp-content/uploads/2014/06/Two_Sample_Test_of_High_Dimensional_Means_Under_Dependence.pdf>.

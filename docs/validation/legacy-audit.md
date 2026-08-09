# Legacy SHT audit ledger

This document records the baseline findings that must be resolved before an
affected method can ship in pySHT. The executable baseline is SHT 0.1.9 at
Git commit `4e29cda1257f86dd0237d37329af358b54d04f2b`.

The legacy namespace contains 54 statistical tests and two dynamic adapters.
The adapters are unnecessary in Python because ordinary callables can be
passed directly. Implemented and independently validated slices are recorded
separately in the [Biswas--Ghosh](biswas-ghosh-2014.md),
[classical mean](classical-mean.md), and
[classical variance](classical-variance.md) ledgers.

Legacy parity is not evidence of correctness. A method is release-ready only
after its primary formula, calibration, numerical implementation, invariants,
and null behavior have been independently checked.

| Legacy method | Initial status | Confirmed concern | Required gate |
|---|---|---|---|
| `cov2.2012LC` | Red | Dimensionally inconsistent scale-dependent normalization | Re-derive from Li--Chen and prove common-scale invariance |
| `cov1.2012Fisher` | Red | Computation ignores the whitened data and supplied null covariance | Whitening-equivalence oracle |
| `cov1.2015WL`, `cov2.2015WL` | Red | Equality-facing API implements one directional tail | Re-derive all supported alternatives |
| `mvar2.1930PN`, `mvar2.1982Muirhead` | Red | Reversed p-value tails | Null calibration and power in both directions |
| `eqdist.2014BG` | Red | Legacy asymptotic variance changes under row permutation | Order-invariant derivation; corrected permutation implementation first |
| `meank.2019CPH` | Red | Stale loop index reuses the wrong group | Group-order and unequal-size tests |
| `mean2.1958Dempster` | Red | Raw positive ratio is incorrectly calibrated as standard normal | Primary-paper re-derivation |
| `mvar2.2012ZXC` | Red | Exact quadrature is unstable and underflows in stronger alternatives | Log-domain/high-precision oracle |
| `norm.1996AJB`, `norm.2008RJB` | Red | Documented default call fails | Default-call smoke tests |
| maximum pairwise BF kernels | Red | Double-to-float truncation and exponentiation overflow | Float64, log-domain implementation |

Classical one- and two-sample t tests, scalar chi-square/F variance tests,
one-way ANOVA, and Shapiro--Wilk matched trusted baseline implementations in
the initial fixed-data probes. The t, chi-square/F variance, and ANOVA methods
have now passed the pySHT validation gates; Shapiro--Wilk remains unimplemented.

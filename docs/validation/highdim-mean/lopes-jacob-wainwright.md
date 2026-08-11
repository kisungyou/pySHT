# Lopes–Jacob–Wainwright random projection

**Status:** primary-paper formula, fixed-plan, reproducibility, permutation,
and numerical boundary gates pass.

`ljw_2samp` draws one Gaussian `p x k` projection with
`k = floor((n_x + n_y - 2) / 2)`. Conditional on that projection, its
projected Hotelling statistic has the reported F calibration under Gaussian
sampling with a common covariance matrix.

This is direct primary-paper validation, not legacy parity. Section 2.2 of
the final arXiv version (pp. 4–5) defines one data-independent Gaussian
projection, the displayed projected Hotelling statistic, and its conditional
`k n / (n - k + 1) F(k, n - k + 1)` null law, where
`n = n_x + n_y - 2`. The implementation uses the algebraically equivalent F
statistic and degrees of freedom. The arXiv record explicitly marks version 2
as defunct because its Equation (4) had an erroneous variance formula; the
audit uses version 3, which the authors identify with version 1 and the final
NIPS paper.

For `calibration="monte-carlo"`, the projection is drawn once and held fixed
for the observed and every permuted statistic. This repairs the legacy SHT
implementation, which drew a new projection inside each call and therefore
changed the auxiliary random plan during calibration. The ledger reconstructs
the seeded projection and every permutation independently, verifies the
exceedance count, and checks the corrected `(b + 1) / (B + 1)` p-value,
Monte Carlo standard error, and binomial interval.

Unrestricted label permutation is exact under exchangeability of the pooled
observations. Equality of means by itself is insufficient if the two null
distributions differ. Before applying an integer-seeded random plan, pySHT
sorts rows within each group and fixes a canonical group order. This only
reindexes a uniform random labeling distribution; it does not change the
observed statistic or its conditional null law. Regression tests cover both
equal and unequal sample sizes and establish identical full results after row
reordering or group exchange.

## Targeted alternative-power gate

Seed 2026090319 initializes a `SeedSequence`. Child 0 drives a persistent
PCG64 stream that draws 20 rows from $N_{40}(0.45\mathbf 1,I)$ and then 20
rows from $N_{40}(0,I)$ per replication. Child 1 is one persistent PCG64
auxiliary stream passed directly to
`mean.ljw_2samp(x, y, rng=auxiliary_rng)`; it draws exactly one independent
Gaussian projection for each dataset. The public default conditional-F
calibration is used. Both streams advance through 1,000 replications, and a
rejection means `pvalue < alpha`.

| alpha | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|
| Rejections / 1,000 | 539 | 802 | 888 |
| Rate | 0.539 | 0.802 | 0.888 |

This strong dense alternative checks the complete data-plus-projection path;
it does not imply power conditional on every possible projection. Run
`python -m tools.mean_power_audits` to reproduce it under Python 3.12.13,
NumPy 2.5.1, and SciPy 1.18.0. The worst-case binomial standard error at 1,000
outer replications is 0.0159.

Primary reference: M. E. Lopes, L. Jacob, and M. J. Wainwright, *A More
Powerful Two-Sample Test in High Dimensions Using Random Projection*, NIPS
24 (2011), 1206–1214, final arXiv version 3,
<https://arxiv.org/abs/1108.2401>.

# Cao–Park–He k-sample mean test

**Status:** primary-paper U-statistic, both variance estimators, independent
fixtures, exchange, scaling, and boundary gates pass in the advertised
high-dimensional factor-model regime.

The test permits different covariance matrices. Its raw statistic uses the
full ordered sums over `i != j` within groups and `l != s` between groups. The
standardized normal statistic and selected variance estimator are retained as
diagnostics.

## Legacy defects repaired

- SHT divided the already off-diagonal within-group sum by two, although the
  paper's equation is an ordered sum. pySHT uses the full U-statistic.
- The split-sample branch reused a stale group index. pySHT evaluates each
  group independently.
- The paper defines split sizes `floor(n_l / 2) + 1` and the remainder and
  uses ordinary unbiased covariance matrices. pySHT follows those definitions
  and requires at least five observations per group for this branch.
- The Hu branch uses the displayed unbiased `tr(Sigma_l²)` correction, not an
  additional MLE rescaling.

These corrections are checked directly against Equation (6) (the ordered
U-statistic), Equation (8) (its null variance), and Lemmas 3.1–3.2 (the split
and Hu trace estimators) on pp. 4–6 of the authors' preprint. The split sizes
are defined on p. 3 as `floor(n_l / 2) + 1` and the remainder. In particular,
the second covariance has at least two rows only when `n_l >= 5`; the public
validator and its error message now state that exact boundary.

Literal tests recompute the raw statistic, both variance estimators, the
standardized statistic, and p-value without using implementation helpers.

The implementation removes one shared feature-wise anchor before scaling.
Translations through `1e12` are checked with tolerances tied to float64 input
spacing; the raw statistic remains in squared measurement units.

The original estimator intentionally uses the first and remaining rows of
each group, so its finite-sample result can change after a row permutation.
This is a property of the displayed split-sample estimator, not a reason to
sort the observations: a data-dependent sort would invalidate independence of
the two covariance estimates. The Hu branch has no such row-split dependence.

Every covariance-trace product uses whichever exact row- or feature-space
product has lower cost, and the Hu squared trace uses the smaller Gram. Thus
neither variance estimator allocates a dense $p\times p$ covariance matrix in
the high-dimensional regime or a dense row Gram for tall low-dimensional
inputs. A 5,000-feature regression guards both branches.

## Null calibration gate

The advertised factor-model special case uses three independent groups of 20
Gaussian rows, `p=500`, and identity covariance. Twenty thousand null datasets
are evaluated per seed with exact Gram/trace reductions independently matched
to the literal full-data fixture.

| Variance estimator | Seed | alpha=0.01 | alpha=0.05 | alpha=0.10 |
|---|---:|---:|---:|---:|
| Hu | 20260810 | 0.01310 | 0.05270 | 0.10280 |
| Hu | 20260811 | 0.01215 | 0.05300 | 0.10045 |
| Original split-sample | 20260810 | 0.01315 | 0.05285 | 0.10265 |
| Original split-sample | 20260811 | 0.01220 | 0.05345 | 0.09950 |

Every row passes the release tolerance at all three levels. This gate supports
the stated balanced Gaussian regime and does not assert uniform calibration
over all unequal-covariance factor models.

## Targeted alternative-power gate

The default split-sample public path uses seed 2026090317. Its
`SeedSequence` spawns a persistent child-0 PCG64 data stream and a separate
persistent child-1 PCG64 auxiliary stream, unused here. In each of 1,000
replications, child 0 draws, in call order, three independent 20-by-40
Gaussian matrices with identity covariance and mean vectors $0$,
$0.35\mathbf 1$, and $-0.35\mathbf 1$. The exact call is
`mean.cph_ksamp(x, y, z)`, so the documented default
`variance_estimator="original"` is exercised. Rejection means
`pvalue < alpha`.

| alpha | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|
| Rejections / 1,000 | 1,000 | 1,000 | 1,000 |
| Rate | 1.000 | 1.000 | 1.000 |

This dense strong-alternative gate is not a comparison with the Hu branch or
a general power claim. Run `python -m tools.mean_power_audits` to reproduce
it under Python 3.12.13, NumPy 2.5.1, and SciPy 1.18.0. The worst-case
binomial standard error at 1,000 outer replications is 0.0159.

Primary reference: M.-X. Cao, J. Park, and D.-J. He, *A Test for the k Sample
Behrens–Fisher Problem in High Dimensional Data*, Journal of Statistical
Planning and Inference 201 (2019), 86–102,
doi:10.1016/j.jspi.2018.12.002; preprint
<https://arxiv.org/abs/1710.07878>.

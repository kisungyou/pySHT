# [2] Tests for Multivariate Mean

The functions in `pysht.mean` cover classical low-dimensional inference,
high-dimensional trace and diagonal tests, unequal-covariance procedures,
sparse alternatives, randomized dimension reduction, and multi-group designs.
Observations are rows and variables are columns.

```{currentmodule} pysht.mean
```

## Choosing a procedure

| Design and working assumptions | Function | R routine |
|---|---|---|
| One sample; Gaussian; invertible covariance | `hotelling_1samp` | `mean1.1931Hotelling` |
| Two samples; Gaussian; common invertible covariance | `hotelling_2samp` | `mean2.1931Hotelling` |
| One or two samples; Gaussian; covariance may be singular | `dempster_1samp`, `dempster_2samp` | `mean1.1958Dempster`, `mean2.1958Dempster` |
| One or two samples; dense high-dimensional shift; common covariance for two samples | `bs_1samp`, `bs_2samp` | `mean1.1996BS`, `mean2.1996BS` |
| One or two samples; diagonal standardization; common covariance for two samples | `sd_1samp`, `sd_2samp` | `mean1.2008SD`, `mean2.2008SD` |
| Two samples; unequal covariance; more rows than variables | `yao_2samp`, `johansen_2samp`, `nvm_2samp`, `ky_2samp` | `mean2.1965Yao`, `mean2.1980Johansen`, `mean2.1986NVM`, `mean2.2004KY` |
| Two samples; high dimension; one Gaussian projection | `ljw_2samp` | `mean2.2011LJW` |
| Two samples; high dimension; averaged random subspaces | `thulin_2samp` | `mean2.2014Thulin` |
| Two samples; sparse shift; Bayesian evidence | `lyl_2samp` | `mean2.mxPBF` |
| Several groups; common covariance | `schott_ksamp` | `meank.2007Schott` |
| Several groups; unequal covariance; Gaussian Scheffé transformation | `zx_ksamp` | `meank.2009ZX` |
| Several groups; unequal covariance; high-dimensional factor model | `cph_ksamp` | `meank.2019CPH` |

An asymptotic p-value is not a guarantee that a test is suitable for a small
dataset. Check the dimensional regime and covariance assumptions in the
method ledger before interpreting the result. The reported `alternative`
describes the scientific direction: small p-values are evidence that at least
one mean vector differs. `lyl_2samp` instead returns maximum log Bayes-factor
evidence and deliberately has no p-value.

For numerical stability, null vectors are removed before scaling in
one-sample routines. Translation-invariant two- and multi-sample routines
remove one deterministic feature-wise anchor, shared by every group, in the
original input coordinates before choosing a global or per-feature scale.
For procedures whose definitions do not split or pair rows, this also makes
the deterministic arithmetic insensitive to row or group order and avoids
letting a huge common location erase representable variation. The documented
CPH split estimator and Zhang–Xu Scheffé transformation are exceptions because
row splitting or pairing is part of their definitions. No numerical method can
recover information already lost when values were rounded to float64.

## Classical and unequal-covariance tests

```{eval-rst}
.. autofunction:: hotelling_1samp
```

```{eval-rst}
.. autofunction:: hotelling_2samp
```

```{eval-rst}
.. autofunction:: dempster_1samp
```

```{eval-rst}
.. autofunction:: dempster_2samp
```

```{eval-rst}
.. autofunction:: yao_2samp
```

```{eval-rst}
.. autofunction:: johansen_2samp
```

```{eval-rst}
.. autofunction:: nvm_2samp
```

```{eval-rst}
.. autofunction:: ky_2samp
```

Validation: [classical mean](../validation/classical-mean.md),
[Dempster](../validation/highdim-mean/dempster.md), and
[multivariate Behrens–Fisher](../validation/highdim-mean/behrens-fisher.md).

## High-dimensional trace and diagonal tests

```{eval-rst}
.. autofunction:: bs_1samp
```

```{eval-rst}
.. autofunction:: bs_2samp
```

```{eval-rst}
.. autofunction:: sd_1samp
```

```{eval-rst}
.. autofunction:: sd_2samp
```

Validation: [Bai–Saranadasa](../validation/highdim-mean/bai-saranadasa.md)
and [Srivastava–Du](../validation/highdim-mean/srivastava-du.md).

## Randomized and sparse tests

```{eval-rst}
.. autofunction:: ljw_2samp
```

`calibration="asymptotic"` uses the conditional F law for the single
projection. `calibration="monte-carlo"` holds that same projection fixed for
the observed statistic and every permutation. The F law assumes Gaussian
samples with a common covariance. The permutation option requires pooled
exchangeability; equal means alone do not make unrestricted relabeling exact.
With an integer seed, canonical pooling makes the full Monte Carlo result
invariant to row reordering and sample exchange.

```{eval-rst}
.. autofunction:: thulin_2samp
```

The subspace dimension is the paper's
`floor((n_x + n_y - 2) / 2)` and is reported as a diagnostic. The sampled
subspaces are fixed across all permutations.
The same pooled-exchangeability requirement applies. Canonical pooling makes
fixed-integer-seed results invariant to row reordering and sample exchange.

```{eval-rst}
.. autofunction:: lyl_2samp
```

The primary statistic is the maximum component log Bayes factor. Positive
values favor a coordinate-wise alternative over the null, but pySHT does not
invent a universal evidence threshold. The published default is
`alpha=2.01` with
`gamma=max(n_x+n_y, p)**(-alpha)`; an explicit `gamma` overrides it. The
paper's Equation (4) corresponds to `a0=b0=0`. Nonzero `a0` or `b0` selects a
clearly labeled pySHT inverse-gamma extension. The paper's two-sample model
uses a common covariance matrix.

Validation: [Lopes–Jacob–Wainwright](../validation/highdim-mean/lopes-jacob-wainwright.md),
[Thulin](../validation/highdim-mean/thulin.md),
[Lee–You–Lin](../validation/highdim-mean/lee-you-lin.md).

## Validation-blocked SHT identity

SHT's `mean2.2014CLX` (the Cai--Liu--Xia maximum mean test) has no public pySHT
callable. Its practical-size Gumbel calibration failed the release gate, and
neither estimated-precision branch has a passing advertised
sparse-covariance scenario. The private implementation is retained only for
scientific audit work; see the
[Cai–Liu–Xia validation ledger](../validation/highdim-mean/cai-liu-xia.md).

## Multi-group tests

```{eval-rst}
.. autofunction:: schott_ksamp
```

```{eval-rst}
.. autofunction:: zx_ksamp
```

`base_test="bai-saranadasa"` is the high-dimensional default;
`base_test="hotelling"` requires more transformed observations than
transformed variables. The defining Scheffé construction uses the first
`n_min` rows from larger groups. Consequently, changing their row pairing can
change the finite-sample statistic even though its null law remains valid for
i.i.d. rows. If groups tie for the smallest size, input order selects the
reference group; changing the order of tied groups can also change the realized
statistic.

```{eval-rst}
.. autofunction:: cph_ksamp
```

`variance_estimator="original"` uses independent covariance estimates from
the paper's split; `"hu"` uses the corrected full-sample trace estimator. The
split branch requires five rows per group and is intentionally sensitive to
row order. Do not sort rows by the outcome before applying it.

Validation: [Schott](../validation/highdim-mean/schott.md),
[Zhang–Xu](../validation/highdim-mean/zhang-xu.md), and
[Cao–Park–He](../validation/highdim-mean/cao-park-he.md).

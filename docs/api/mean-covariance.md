# [6] Simultaneous Mean and Covariance

The functions in `pysht.mean_covariance` test a mean vector and a
covariance matrix in one decision. Observations are rows and variables are
columns.

| Design | Function | Calibration regime |
|---|---|---|
| One sample against a known mean and covariance | `llzs_1samp` | Proportional high-dimensional normal limit |
| One sample against a known mean and covariance | `lrt_1samp` | Fixed-dimensional Wilks limit |
| Two independent samples | `hn_2samp` | High-dimensional normal limit |

For a one-sample test, `popmean=None` and `popcov=None` mean
the zero vector and identity matrix. A supplied covariance matrix must be
finite, symmetric, and positive definite. pySHT whitens by a Cholesky solve;
it does not form a matrix inverse.

```python
from pysht.mean_covariance import llzs_1samp

result = llzs_1samp(x, popmean=null_mean, popcov=null_covariance)
print(result)
```

The alternative for every function is the complement of the joint null: a
mean departure, a covariance departure, or both can be significant.
`llzs_1samp` reports its aspect ratio and estimated marginal excess
kurtosis as immutable calibration diagnostics. `hn_2samp` reports its
two unbiased squared-distance estimates in the original measurement units
when those values are representable as finite float64 numbers.

These methods serve different asymptotic regimes. `llzs_1samp` is for
dimension and sample size growing proportionally. `hn_2samp` requires
both group sizes and dimension to grow under the paper's moment-factorization
and trace conditions. `lrt_1samp` assumes multivariate normality with
fixed dimension, more observations than variables, and full centered column
rank. None is advertised as a generic small-sample calibration.

The [validation ledger](../validation/mean-covariance.md) gives the exact
formula oracles, the LLZS correction relative to SHT 0.1.9, and the
20,000-dataset null-size gates for the advertised regimes.

```{currentmodule} pysht.mean_covariance
```

## Functions

```{eval-rst}
.. autofunction:: llzs_1samp
```

```{eval-rst}
.. autofunction:: lrt_1samp
```

```{eval-rst}
.. autofunction:: hn_2samp
```

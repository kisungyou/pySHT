# [5] Simultaneous Mean and Variance

The functions in `pysht.mean_variance` test the mean and variance of
one or two univariate normal populations in one decision. A significant result
can reflect a mean departure, a variance departure, or both.

| Question or calibration | Function |
|---|---|
| One population versus a specified mean and variance | `lrt_1samp` |
| Pearson--Neyman beta approximation for two populations | `pn_2samp` |
| Perng--Littell combination of pooled-t and F tests | `pl_2samp` |
| Muirhead second-order likelihood-ratio approximation | `muirhead_2samp` |
| Zhang--Xu--Chen exact likelihood-ratio calibration | `exact_lrt_2samp` |
| Wilks asymptotic likelihood-ratio calibration | `lrt_2samp` |

All methods require independent normal observations and positive within-sample
variance. The scalar null variance for `lrt_1samp` is supplied with the
keyword `variance`; for example:

```python
from pysht.mean_variance import lrt_1samp

result = lrt_1samp(x, popmean=2.0, variance=1.5)
print(result)
```

The result prints in the same compact style as an R `htest` object:
method, statistic, degrees of freedom where applicable, p-value, alternative,
and calibration. Pearson--Neyman beta shapes, Perng--Littell component
p-values, and Muirhead correction constants appear under immutable
`diagnostics`; they are calibration metadata, not parameter estimates.

For small samples, prefer `exact_lrt_2samp` when exact likelihood-ratio
calibration is required. `pn_2samp` is a highly accurate moment
approximation in the validated normal scenarios. The chi-square calibrations
in `lrt_1samp` and `lrt_2samp`, and the finite Muirhead
expansion, are asymptotic approximations. The
[validation ledger](../validation/mean-variance.md) records the finite-sample
regimes that passed the release-size gate and the smaller regimes that did not.

```{currentmodule} pysht.mean_variance
```

## Functions

```{eval-rst}
.. autofunction:: lrt_1samp
```

```{eval-rst}
.. autofunction:: pn_2samp
```

```{eval-rst}
.. autofunction:: pl_2samp
```

```{eval-rst}
.. autofunction:: muirhead_2samp
```

```{eval-rst}
.. autofunction:: exact_lrt_2samp
```

```{eval-rst}
.. autofunction:: lrt_2samp
```

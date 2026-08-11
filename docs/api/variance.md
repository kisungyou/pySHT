# [3] Tests for Variance

`pysht.variance` provides classical one-, two-, and multi-sample procedures for
scalar population variances.

| Question | Function |
|---|---|
| Is one variance equal to a reference value? | `chisquare_1samp` |
| Are two normal-population variances equal? | `f_2samp` |
| Are several normal-population variances equal? | `bartlett` |
| Are several group spreads equal around their means? | `levene` |
| Are several group spreads equal around their medians? | `brown_forsythe` |

See the [classical-variance validation
ledger](../validation/classical-variance.md) for assumptions, scale handling,
and SciPy comparisons.

## Functions

```{eval-rst}
.. automodule:: pysht.variance
   :members:
   :member-order: bysource
```

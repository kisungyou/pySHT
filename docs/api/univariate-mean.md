# [1] Tests for Univariate Mean

The univariate mean API contains one-sample and two-sample t tests together
with classical one-way analysis of variance. The functions live in
`pysht.mean`; this page groups them by the scientific category used in SHT.

| Question | Function |
|---|---|
| Is one population mean equal to a reference value? | `ttest_1samp` |
| Are two independent or paired means equal? | `ttest_2samp` |
| Are two or more independent-group means equal? | `anova_oneway` |

See the [classical-mean validation ledger](../validation/classical-mean.md) for
formula and oracle details.

```{currentmodule} pysht.mean
```

## Functions

```{eval-rst}
.. autofunction:: ttest_1samp
```

```{eval-rst}
.. autofunction:: ttest_2samp
```

```{eval-rst}
.. autofunction:: anova_oneway
```

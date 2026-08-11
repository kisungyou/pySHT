# Test results

All result objects are immutable, keyword-only data classes. The common base
contains the statistic and presentation metadata; specialized subclasses add
frequentist p-values, resampling evidence, distance normalization, or Bayes
factors.

For an interpretation-oriented introduction, see [Working with test
results](../user-guide/results.md). This page is the exact field-level API.

```{currentmodule} pysht
```

## Common result

```{eval-rst}
.. autoclass:: StatisticalTestResult
   :members:
   :show-inheritance:
```

Named `diagnostics` are immutable `(name, value)` pairs. They record quantities
such as a standardized statistic, estimator choice, projection dimension, or
number of random subspaces without turning each method into a new result type.

## Frequentist result

```{eval-rst}
.. autoclass:: HypothesisTestResult
   :members:
   :show-inheritance:
```

## Resampling result

This subclass adds the enumeration or simulation budget and the number of
statistics at least as extreme as the observed value. Exact enumeration uses
$b/B$. Monte Carlo tests use the nonzero correction $(b+1)/(B+1)$ and report
both a conditional Monte Carlo standard-error estimate and a 95% exact
binomial interval for the underlying tail probability.

```{eval-rst}
.. autoclass:: ResamplingTestResult
   :members:
   :show-inheritance:
```

## Distance-test result

This subclass additionally records the normalized statistic and distance
scale. These diagnostic fields preserve the distinction between the reported
scientific statistic and the dimensionless quantity used for stable
permutation ordering.

```{eval-rst}
.. autoclass:: DistanceTestResult
   :members:
   :show-inheritance:
```

## Bayes-factor result

```{eval-rst}
.. autoclass:: BayesFactorTestResult
   :members:
   :show-inheritance:
```

`BayesFactorTestResult` deliberately has no `pvalue`. Its primary statistic is
the maximum log Bayes factor and its immutable component values remain in log
space. pySHT does not exponentiate the maximum, manufacture a frequentist
p-value, or choose an evidence threshold for the analyst.

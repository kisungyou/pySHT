# Test results

All result objects are immutable, keyword-only data classes. The base class
contains the common inferential quantities; specialized subclasses add
resampling and distance-test diagnostics.

```{currentmodule} pysht
```

## Base result

```{autoclass} HypothesisTestResult
:members:
:show-inheritance:
```

## Resampling result

```{autoclass} ResamplingTestResult
:members:
:show-inheritance:
```

## Distance-test result

```{autoclass} DistanceTestResult
:members:
:show-inheritance:
```

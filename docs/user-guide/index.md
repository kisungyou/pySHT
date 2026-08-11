# User guide

The user guide explains the full pySHT workflow: translate a scientific
question into a test, prepare data that match the sampling design, call the
procedure, and interpret the returned result. The same workflow applies across
the package's mean, variance, covariance, joint-parameter, distributional,
goodness-of-fit, and structured-domain categories.

## A practical route through the guide

1. [Choose a test](choose-a-test.md) from the target quantity and sampling
   design.
2. Review [data and assumptions](data-assumptions.md), including independence,
   pairing, dimensionality, and distributional conditions.
3. Read [test results](results.md) to understand the R `htest`-style printed
   report, Monte Carlo uncertainty, log Bayes factors, and the fields available
   for programmatic use.
4. Follow [reproducible inference](reproducibility.md) when recording inputs,
   software versions, analysis options, and any random-number controls.

The [API reference](../api/index.md) is organized by statistical category and
contains the complete function signatures. The guide focuses on deciding what
to call and what the result means.

```{toctree}
:maxdepth: 1

choose-a-test
data-assumptions
results
reproducibility
```

## Reference material

- [Implemented methods](../methods/index.md)
- [Validation ledgers](../validation/index.md)
- [Migration from SHT for R](../migration/from-r.md)
- [Project information](../project/index.md)

```{toctree}
:hidden:
:maxdepth: 2

../validation/index
../project/index
```

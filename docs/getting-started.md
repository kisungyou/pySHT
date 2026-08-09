# Getting started

## Installation

pySHT currently requires Python 3.12 or newer. Once release wheels are
available, install it from PyPI with:

```console
python -m pip install pysht
```

For development from a source checkout, a C++17 compiler and CMake are also
required:

```console
python -m pip install -e ".[test]"
```

## Result objects

Every test returns an immutable result rather than printing as a side effect.
The object supports both programmatic access and a human-readable display:

```python
from pysht.mean import ttest_1samp

result = ttest_1samp([2.1, 2.4, 1.9, 2.2, 2.5], popmean=2.0)

print(result)          # R htest-style summary
result.statistic       # test statistic
result.pvalue          # p-value
result.alternative     # alternative hypothesis
result.method          # method name
```

Additional fields, such as degrees of freedom, confidence intervals,
estimates, resampling diagnostics, and distance normalization metadata, are
present when the procedure computes them. Result instances are frozen data
classes, so attempts to mutate a field raise an error.

## Reproducible resampling

Pass an integer seed to a resampling procedure when reproducibility matters:

```python
from pysht.equaldist import biswas_ghosh_2samp

result = biswas_ghosh_2samp(
    [[0.0, 0.2], [0.3, 0.1], [0.1, 0.4]],
    [[1.0, 1.2], [0.8, 1.1], [1.2, 0.9]],
    n_resamples=999,
    rng=42,
)
```

The equality-of-distributions test uses exact enumeration when the requested
budget covers every distinct labeling; otherwise it uses the corrected Monte
Carlo permutation p-value.

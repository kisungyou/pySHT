# pySHT

pySHT provides independently validated statistical hypothesis tests for
NumPy and SciPy workflows. Its result objects are immutable and display the
key inferential quantities in the concise style of an R `htest` object.

```{toctree}
:hidden:
:maxdepth: 2

getting-started
api/index
validation/index
```

## Current scope

- equality of two univariate or multivariate distributions;
- classical univariate and multivariate tests for population means;
- one-, two-, and multi-sample tests for population variances;
- a C++17/nanobind foundation for computational kernels.

Every port is treated as a new statistical implementation: legacy code is
audited, numerical behavior is tested, and unverified calibration formulas
are not exposed. See the [validation ledgers](validation/index.md) for the
findings and independent checks that guide this work.

## Quick example

```python
import numpy as np

from pysht.mean import ttest_1samp

x = np.array([1.2, 0.8, 1.4, 1.1, 0.9])
result = ttest_1samp(x, popmean=0.0)

print(result)
print(result.statistic, result.pvalue)
```

Begin with [installation and basic usage](getting-started.md), or go directly
to the [API reference](api/index.md).

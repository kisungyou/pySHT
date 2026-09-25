# [11] Tests of Independence

`pysht.independence` provides permutation tests for paired observations of
random vectors. Every input block uses rows for the same observational units;
different blocks may have different feature dimensions.

Exact marginal-permutation inference is invariant to Euclidean isometries.
Fixed-seed Monte Carlo replay also holds when metric canonicalization completes
within its fixed work budget and distance ranks are separated from their
floating equality boundary. A boundary case or symmetric geometry that
exhausts this budget can select a different, but still uniformly valid,
permutation plan after a rotation; use exact mode for transformation replay
when its orbit is tractable.

| Function | Null hypothesis | Primary statistic |
|---|---|---|
| `distance_covariance` | $X$ and $Y$ are independent | $nV_n^2$ in raw distance units |
| `hsic` | $X$ and $Y$ are independent | biased $\mathrm{HSIC}_b$ |
| `dhsic` | all supplied blocks are mutually independent | $n\widehat{\mathrm{dHSIC}}_n$ |
| `distance_multivariance` | all supplied blocks are mutually independent | normalized total $n\overline M_n^2$ |

Pairwise independence is not mutual independence. In particular, `dhsic` and
`distance_multivariance` can detect higher-order alternatives such as
$Z=X\mathbin{\mathrm{xor}}Y$, even though all three pairs are independent.

Kernel methods accept only built-in RBF and Laplacian kernels. `hsic` exposes
separate `kernel_x`, `kernel_y`, `bandwidth_x`, and `bandwidth_y` controls.
`dhsic` accepts either one value broadcast to every block or a tuple with one
value per block. All Gram matrices and median bandwidths are fixed before
permuting marginal rows.

```python
import numpy as np
from pysht import independence

rng = np.random.default_rng(2026)
x = rng.normal(size=(40, 2))
y = x**2 + 0.25 * rng.normal(size=(40, 2))

result = independence.hsic(x, y, n_resamples=999, rng=17)
result.pvalue
```

See the [independence validation ledger](../validation/independence.md) for
formula reductions, randomization semantics, numerical normalization, and
calibration evidence.

## Functions

```{eval-rst}
.. automodule:: pysht.independence
   :members:
   :member-order: bysource
```

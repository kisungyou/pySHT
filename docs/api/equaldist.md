# [7] Tests for Equality of Distributions

`pysht.equaldist` tests equality of complete data-generating distributions,
not only equality of a particular mean or covariance.

| Function | Design | Statistic | Calibration |
|---|---|---|---|
| `energy_ksamp` | two or more samples | DISCO $F_\alpha$ | exact/Monte Carlo label permutation |
| `mmd_2samp` | two samples | unbiased $\mathrm{MMD}_u^2$ | exact/Monte Carlo label permutation |
| `bg_2samp` | two samples | Biswas--Ghosh distance contrast | exact/Monte Carlo label permutation |

All inputs are Euclidean observations. One-dimensional arrays mean univariate
samples; matrices use rows as observations. Permutation inference assumes the
pooled observations are exchangeable under the null. It is not valid for
paired, clustered, stratified, or serially dependent samples without a design-
specific randomization scheme.

Exact calibration is invariant to Euclidean isometries. Fixed-seed Monte Carlo
replay also holds when metric canonicalization completes within its fixed work
budget and distance ranks are separated from their floating equality boundary.
At that boundary, or when symmetric geometry exhausts the budget and triggers
coordinate ordering, a rotation can select a different, still uniformly valid
Monte Carlo plan. Use exact mode when transformation replay is essential and
the orbit is tractable.

`mmd_2samp` deliberately accepts only the characteristic RBF and Laplacian
kernels. The median heuristic, when selected, is computed once from strictly
positive pooled distances and remains fixed for the observed and every
permuted labeling.

Ball Divergence is not public in this release. Its mathematical closed-ball
rule requires exact equality for equal radii, but ordinary Euclidean distance
evaluation can split those ties after an isometry; a tolerance can instead
merge genuinely distinct radii. The private research implementation remains
blocked until that distinction has a correctness-certified solution.

```python
import numpy as np
from pysht import equaldist

rng = np.random.default_rng(2026)
x = rng.normal(size=(30, 3))
y = rng.normal(loc=0.5, size=(35, 3))

result = equaldist.energy_ksamp(x, y, n_resamples=999, rng=17)
result.pvalue
```

See the [new distribution-test validation ledger](../validation/distribution-equality.md)
and the [Biswas--Ghosh record](../validation/biswas-ghosh-2014.md).

## Functions

```{eval-rst}
.. automodule:: pysht.equaldist
   :members:
   :member-order: bysource
```

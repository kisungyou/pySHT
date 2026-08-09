# pySHT

**pySHT** is a Python toolbox for literature-traceable statistical hypothesis
testing, with particular emphasis on high-dimensional, joint-parameter, and
structured-domain problems.

The distribution and import name is `pysht`. The project is an independently
validated successor to the R package
[SHT](https://github.com/kisungyou/SHT); it does not treat legacy numerical
output as a correctness oracle.

> **Status:** early implementation. The public API is not yet stable and the
> package has not yet been published to PyPI.

## Implemented namespaces

- `pysht.equaldist`: Biswas--Ghosh equality-of-distributions test with exact
  or corrected Monte Carlo permutation calibration.
- `pysht.mean`: one- and two-sample t tests, one-way ANOVA, and one- and
  two-sample Hotelling tests.
- `pysht.variance`: chi-square, F, Bartlett, Levene, and Brown--Forsythe
  variance tests.

For example:

```python
import numpy as np

from pysht.equaldist import biswas_ghosh_2samp

rng = np.random.default_rng(42)
x = rng.normal(size=(30, 3))
y = rng.normal(loc=0.5, size=(35, 3))

result = biswas_ghosh_2samp(x, y, rng=42)
print(result)
```

Results are immutable Python objects, but their text display follows the
human-readable style of R's `htest` objects: method title, data, statistic,
p-value, alternative, degrees of freedom, confidence interval, estimates, and
calibration details when applicable.

Small permutation spaces are enumerated exactly. Larger problems use the
nonzero correction `(exceedances + 1) / (n_resamples + 1)` and report Monte
Carlo uncertainty. The Biswas--Ghosh result retains both the raw statistic from
the paper and the normalized statistic used for stable numerical comparison.

## Correctness policy

Every shipped method must have a formula ledger, a literal reference
implementation, independent validation evidence, invariance checks, numerical
stress tests, and documented assumptions. Known differences from SHT are
recorded in the [validation documentation](docs/validation/index.md).

The current suite also builds the documentation with warnings treated as
errors, checks strict typing and formatting, and verifies installed stable-ABI
wheels across Python 3.12--3.14 in CI.

## License

MIT. See [LICENSE](LICENSE).

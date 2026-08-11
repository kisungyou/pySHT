# pySHT

**pySHT** is a Python toolbox for literature-traceable statistical hypothesis
testing, with particular emphasis on high-dimensional, joint-parameter, and
structured-domain problems.

Documentation: <https://kisungyou.com/pysht/>

The distribution and import name is `pysht`. The project is an independently
validated successor to the R package
[SHT](https://github.com/kisungyou/SHT); it does not treat legacy numerical
output as a correctness oracle.

> **Status:** version 0.1.0 is the initial public release. The scientific
> surface is correctness-gated, but APIs may still evolve before version 1.0.

Install the release from PyPI:

```console
python -m pip install pysht
```

## Statistical scope

pySHT's public API currently contains 51 canonical Python functions covering
52 of the 54 public statistical routine identities in SHT 0.1.9. Two
algebraically identical R entry points share one Python implementation. The
withheld identities are `mean2.2014CLX`, whose practical-size Gumbel
calibration failed, and `cov1.2012Fisher`, whose former audit is not
reproducible through the public path. The public functions are organized by
scientific question rather than re-exported from the top-level package:

- `pysht.mean`: univariate, multivariate, high-dimensional, randomized, sparse,
  and multi-group mean tests;
- `pysht.variance`: one-, two-, and multi-sample variance tests;
- `pysht.covariance`: one-, two-, and multi-sample covariance tests;
- `pysht.mean_variance` and `pysht.mean_covariance`: joint-parameter tests;
- `pysht.equaldist`: equality-of-distributions tests;
- `pysht.normality` and `pysht.uniformity`: goodness-of-fit tests; and
- `pysht.simplex`: tests for compositional data on the probability simplex.

The [API catalog](https://kisungyou.com/pysht/api/index.html) follows SHT's
original `[0]`--`[10]` category order and records every public name.

For example:

```python
from pysht.mean import ttest_1samp

result = ttest_1samp([2.1, 2.4, 1.9, 2.2, 2.5], popmean=2.0)
print(result)
```

Results are immutable Python objects, but their text display follows the
human-readable style of R's `htest` objects: method title, data, statistic,
p-value, alternative, degrees of freedom, confidence interval, estimates, and
calibration details when applicable.

Procedure-specific results can additionally report resampling diagnostics,
numerical metadata, or component log Bayes factors. The
[user guide](https://kisungyou.com/pysht/user-guide/index.html) explains how to
read and use this output.

## Correctness policy

Every shipped method must have a formula ledger, an appropriate literal
reference or trusted independent comparator, validation evidence, invariance
checks, numerical stress tests, and documented assumptions. Known differences
from SHT are recorded in the
[validation documentation](https://kisungyou.com/pysht/validation/index.html).

The current suite also builds the documentation with warnings treated as
errors, checks strict typing and formatting, and verifies installed stable-ABI
wheels across Python 3.12--3.14 in CI.

## License

MIT. See the [license](https://github.com/kisungyou/pySHT/blob/master/LICENSE)
and [third-party notices](https://github.com/kisungyou/pySHT/blob/master/THIRD_PARTY_LICENSES.md).

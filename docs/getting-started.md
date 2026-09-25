# Getting started

This guide takes you from installation to a result you can inspect, print,
interpret, and use programmatically. If you are unsure which procedure matches
your question and sampling design, start with the
[test chooser](user-guide/choose-a-test.md).

## Installation

pySHT requires Python 3.12 or newer. Install the current release from PyPI:

```console
python -m pip install pysht
```

For development from a source checkout, a C++17 compiler and CMake are also
required:

```console
python -m pip install -e ".[test]"
```

## Run a first test

```python
from pysht.mean import ttest_1samp

x = [2.1, 2.4, 1.9, 2.2, 2.5]
result = ttest_1samp(x, popmean=2.0)

print(result)
```

The printed summary contains the method, data label, statistic, degrees of
freedom, p-value, alternative hypothesis, calibration, confidence interval,
and estimate when those quantities apply. pySHT uses the same display contract
across its scientific modules; the exact fields depend on the procedure.

Before interpreting the p-value, confirm that the procedure, alternative, and
assumptions match the scientific question. See the
[data and assumptions guide](user-guide/data-assumptions.md).

## Result objects

Every test returns an immutable `StatisticalTestResult` subtype rather than
printing as a side effect. A frequentist result supports both programmatic
access and a human-readable display:

```python
from pysht.mean import ttest_1samp

result = ttest_1samp([2.1, 2.4, 1.9, 2.2, 2.5], popmean=2.0)

print(result)  # R htest-style summary
result.statistic  # test statistic
result.pvalue  # p-value
result.alternative  # alternative hypothesis
result.method  # method name
```

Additional fields, such as degrees of freedom, confidence intervals, estimates,
named diagnostics, and calibration metadata, are present when the procedure
computes them. Exact and Monte Carlo results also record simulation counts and
tail uncertainty. The two Lee--You--Lin procedures return maximum and component
log Bayes factors with no `pvalue`; they do not fabricate a frequentist
decision. Result instances are frozen data classes, so attempts to mutate a
field raise an error.

Use `repr(result)` for a compact developer-facing representation and
`print(result)` for the statistical report.

## Interpret the result

Treat the printed output as a compact report, not as a decision made by the
library. A p-value is calculated under the stated null model and calibration;
it is not the probability that the null hypothesis is true. pySHT therefore
does not add a `reject` field. Set the significance level before analysis and
report an estimate and confidence interval when the procedure provides them.

The [test-results guide](user-guide/results.md) explains each printed section
and the common structured fields. For exact signatures and parameter defaults,
use the category pages in the [API reference](api/index.md), beginning with
[univariate mean tests](api/univariate-mean.md) for the function above.

## Preserve the analysis

Record the data provenance, preprocessing, fully qualified function name, all
non-default arguments, software versions, and returned result. If a procedure
uses resampling, random projections, random subspaces, or numerical
optimization, also record its calibration mode, budget, random-number policy,
and convergence controls.
See [reproducible inference](user-guide/reproducibility.md) for a complete
checklist.

## Where to go next

- [Choose a test](user-guide/choose-a-test.md) from the scientific question.
- [Understand result objects](user-guide/results.md) and their interpretation.
- Inspect the complete [API reference](api/index.md): SHT-compatible categories
  `[0]`--`[10]` followed by pySHT-native independence and circular categories
  `[11]`--`[12]`.
- Use the [R migration crosswalk](migration/from-r.md) to translate an SHT
  routine into its lowercase Python name.
- Read the [validation center](validation/index.md) before relying on a method
  in a sensitive workflow.

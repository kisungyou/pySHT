# Test results

Every public test returns an immutable result object. Printing it produces a
statistical report modeled on R's `htest` output, while its fields provide the
same information without parsing text.

```python
from pysht.mean import ttest_1samp

result = ttest_1samp([2.1, 2.4, 1.9, 2.2, 2.5], popmean=2.0)

print(result)
print(result.statistic)
print(result.pvalue)
```

The printed report for this call is:

```text
One Sample t-test

data: x
t = 2.06049, df = 4, p-value = 0.108392
alternative hypothesis: true mean is not equal to 2
calibration: Student t distribution
95 percent confidence interval:
 1.92356 2.51644
sample estimates:
mean of x = 2.22
```

Read the report from top to bottom:

1. **Method and data** identify the procedure and inputs being summarized.
2. **Statistic, degrees of freedom, and p-value** describe the observed test
   statistic and its calibration under the null hypothesis.
3. **Alternative hypothesis** states the direction used to calculate the
   p-value.
4. **Calibration** names the reference distribution or resampling scheme.
5. **Confidence interval and estimates**, when available, put the test in the
   scale of the measured quantity.

Printing is presentation only: the test function does not print as a side
effect, and display rounding does not change the stored floating-point values.

## Common fields

Frequentist results are instances of
[`HypothesisTestResult`](../api/results.md) or one of its subclasses. The
Lee--You--Lin procedures instead return `BayesFactorTestResult`, because a
Bayes factor is not a p-value.

| Field | Meaning |
|---|---|
| `statistic` | Reported test statistic; its interpretation depends on the procedure |
| `pvalue` | Tail probability under the null model and stated calibration, in $[0,1]$ |
| `p_value` | Read-only alias for `pvalue` |
| `method` | Human-readable procedure name |
| `alternative` | The alternative hypothesis evaluated by the test |
| `data_name` | Compact description of the inputs, when available |
| `statistic_name` | Label used for the statistic in the text display |
| `calibration` | Reference distribution or resampling method |
| `diagnostics` | Immutable named method-specific quantities, such as an estimator or projection count |
| `df` | One degree of freedom or a tuple of degrees of freedom, when applicable |
| `confidence_interval` | `(lower, upper)`, when the procedure computes one |
| `confidence_level` | Confidence coefficient associated with that interval |
| `estimates` | Tuple of `(name, value)` sample estimates, when reported |

Optional fields are `None` or empty when they do not apply. For example,
Hotelling and omnibus variance results currently do not report confidence
intervals, while the t, chi-square, and two-sample F procedures do.

```python
if result.confidence_interval is not None:
    lower, upper = result.confidence_interval
    print(result.confidence_level, lower, upper)

for name, value in result.estimates:
    print(name, value)
```

## Printed output or structured fields?

Use `print(result)` in an interactive analysis or a human-readable report. Use
the fields when branching on availability, building a table, or serializing an
analysis record. Do not extract numbers from the printed text: its labels and
rounding are intended for readers, not as a machine-readable interface.

Use `repr(result)` for a compact developer-facing representation. The
scientific values are the same in all three forms.

## Procedure-specific fields

[`ResamplingTestResult`](../api/results.md) adds:

- `n_resamples`: the number of sampled permutations or, for an exact result,
  the number of enumerated labelings;
- `exceedances`: sampled statistics at least as extreme as the observation;
- `monte_carlo_standard_error`: simulation error for the corrected Monte Carlo
  p-value estimator, or `None` for exact enumeration;
- `tail_probability_interval`: a 95% exact binomial interval for the
  underlying exceedance probability, or `None` for exact enumeration; and
- `exact`: whether every member of the finite randomization space was
  enumerated.

Some subclasses add further diagnostics needed by a particular calibration or
statistic. Those fields are documented in the
[result-class API](../api/results.md); they are not part of every test result.

## Bayes factors are not p-values

`BayesFactorTestResult` stores the maximum log Bayes factor in `statistic` and
all component log Bayes factors in `component_log_bayes_factors`. It has no
`pvalue` attribute. Positive log Bayes factors favor the alternative relative
to the null under the stated priors; negative values favor the null. Their
magnitude is not a frequentist tail probability, and exponentiating very large
values is unnecessary and can overflow.

pySHT does not attach a universal evidence threshold. If an analysis uses a
decision threshold, choose and justify it before examining the result, record
it separately, and report the prior settings together with the log Bayes
factor.

## Immutability and export

Results are frozen, slotted data classes. Reassigning a field raises an error;
this prevents a reported p-value from drifting away from its statistic and
metadata.

Use `dataclasses.asdict` when a mutable copy is needed for tabular output or
serialization:

```python
from dataclasses import asdict

record = asdict(result)
```

The resulting dictionary can contain tuples and non-finite interval endpoints
such as `-inf` or `inf`. Convert those explicitly if a downstream format, such
as strict JSON, does not accept them.

## Interpretation boundaries

A p-value is a tail probability under the stated null model and calibration.
It is not the probability that the null hypothesis is true, the probability
that a result occurred by chance, an effect size, or a guarantee that the
assumptions hold. Report the statistic, its reference information, the p-value,
the alternative, and an interval or estimate when available—not the p-value
alone.

pySHT deliberately does not attach a `reject` field. Choose the significance
level before analysis, compare the p-value with that threshold if a binary
decision is required, and interpret the magnitude and uncertainty in the
scientific context. A result above the threshold does not prove equality or
the null hypothesis.

A confidence interval and p-value belong to the same procedure only when they
use the same alternative and confidence level. For multiple tests, selection
or multiplicity adjustments remain the analyst's responsibility.

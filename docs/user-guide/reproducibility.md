# Reproducible inference

Reproducibility starts before a function call. Preserve enough information to
reconstruct the analytical data, sampling design, selected test, arguments,
software environment, and reported result. A random seed alone does not make
an undocumented analysis reproducible.

## Record the complete analysis

For every test, retain:

- input provenance, preprocessing, exclusions, and missing-data decisions;
- the sampling unit and whether samples were independent or paired;
- the null and alternative hypotheses and the prespecified significance level;
- the fully qualified function name and every non-default argument;
- the pySHT, Python, NumPy, and SciPy versions;
- the statistic, reference information, p-value, alternative, confidence
  interval, and estimates that apply; and
- the source revision when using unreleased changes.

The immutable result object can be converted to a dictionary for a structured
analysis record:

```python
from dataclasses import asdict
from importlib.metadata import version
from pysht.mean import ttest_1samp

result = ttest_1samp(sample, popmean=0.0, alternative="two-sided")

record = {
    "pysht_version": version("pysht"),
    "function": "pysht.mean.ttest_1samp",
    "parameters": {"popmean": 0.0, "alternative": "two-sided"},
    "result": asdict(result),
}
```

The resulting dictionary may need explicit conversion of tuples or non-finite
interval endpoints before strict JSON serialization.

## Deterministic and resampling procedures

Classical pySHT procedures are deterministic for fixed inputs and arguments.
Resampling procedures additionally require the calibration mode, resampling
budget, and random-number policy to be recorded. For a Monte Carlo procedure,
also retain the exceedance count and Monte Carlo standard error.

When a procedure accepts `rng`, pass an integer seed for a standalone analysis
that must be replayed exactly. A NumPy `Generator` is more appropriate when a
larger workflow manages a deliberate random stream:

- an integer constructs a fresh generator for the call and is easiest to
  reproduce in isolation;
- a `Generator` advances as it is used, so call order and its initial state are
  part of the reproducibility record; and
- `None` initializes an uncontrolled stream and should not be used when exact
  replay matters.

An exact enumeration has no sampling variability, but its calibration mode and
enumeration budget should still be recorded.

## Monte Carlo p-values are estimates

With $B$ simulated null statistics and $b$ values at least as extreme as the
observation, pySHT reports

$$
\widehat p=\frac{b+1}{B+1}.
$$

The nonzero correction avoids claiming an impossible p-value of zero after a
finite simulation. Letting $\widehat q=b/B$ estimate the underlying exceedance
probability, `result.monte_carlo_standard_error` reports the plug-in
conditional standard deviation of the corrected estimator:

$$
\frac{\sqrt{B\widehat q(1-\widehat q)}}{B+1}.
$$

The result also contains a 95% Clopper--Pearson interval for the underlying
tail probability $q$. These quantities measure simulation noise in the p-value
calculation. They are not uncertainty about an effect size and do not repair a
wrong null simulation or invalid exchangeability assumption. Increase
`n_resamples` when Monte Carlo error is consequential relative to the decision
threshold. Repeating the computation with prespecified independent seeds is a
useful diagnostic, not a replacement for a sufficient budget.

Exact enumeration has no Monte Carlo standard error. Its p-value is exact for
the conditional label-randomization distribution under exchangeability, not
for a design that violates that assumption.

## A resampling call

Permutation-calibrated and parametric Monte Carlo procedures use the same
resampling metadata. For example:

```python
from pysht.equaldist import bg_2samp

result = bg_2samp(
    x,
    y,
    calibration="monte-carlo",
    n_resamples=99_999,
    rng=20260809,
)
```

Here the record should include all three calibration arguments, the returned
exceedance count, standard error, and tail-probability interval. Exact
enumeration does not consume a supplied generator. Random-projection and
random-subspace tests also record their fixed auxiliary plan in diagnostics;
reusing a seed is necessary to replay that conditional analysis. Each
category's API page documents its exact calibration contract.

## Reproducing Bayes factors and optimization

For Lee--You--Lin tests, record `a0`, `b0`, `alpha`, the effective `gamma`
reported in diagnostics, the maximum log Bayes factor, and the component log
Bayes factors. If `gamma` was supplied explicitly, record that override. For
the covariance procedure, also record why the observations satisfy the
known-zero-mean model. Do not replace a large log value with an overflowed
exponentiated value.

For the simplex likelihood-ratio test, record the selected Dirichlet model and
all exposed optimizer tolerances. A successful return means the score and
convergence checks passed; changing a tolerance is an analytical choice and
belongs in the reproducibility record.

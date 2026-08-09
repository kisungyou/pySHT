# Changelog

## Unreleased

- Established the Python 3.12+ package and native-build scaffold.
- Added immutable hypothesis-test result objects with `htest`-style display.
- Added `pysht.equaldist` and the permutation-calibrated Biswas--Ghosh test.
- Added exhaustive calibration for small permutation spaces, corrected
  Monte Carlo p-values and uncertainty, floating-point tie handling, and
  scale-stable raw/normalized Biswas--Ghosh statistics.
- Added `pysht.mean` with Student/Welch/paired t tests, one-way ANOVA, and
  exact-covariance Hotelling tests.
- Added `pysht.variance` with chi-square, F, Bartlett, Levene, and
  Brown--Forsythe procedures.
- Extended result objects with degrees of freedom, confidence intervals,
  named estimates, resampling diagnostics, and distance normalization data.
- Added the initial legacy SHT correctness ledger.
- Added formula ledgers, independent numerical oracles, cross-platform CI,
  strict typing/linting, warning-free documentation builds, branch coverage,
  distribution builds, and installed-wheel smoke tests.

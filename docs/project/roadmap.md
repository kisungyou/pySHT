# Roadmap

pySHT 0.1.0 is the initial public release. The combined 0.2--0.5 development
line preserves its validation-driven SHT compatibility core and adds
independently sourced research methods rather than treating a larger catalog
as sufficient evidence.
The project remains pre-1.0: changes to the public surface are possible, but
they must be intentional, tested, and recorded.

## Implemented expansion

- The 51-function compatibility core covers 52 of the 54 public SHT
  statistical routine identities; the one shared implementation and the
  validation-blocked mean CLX and Fisher identities are documented in the
  [migration crosswalk](../migration/from-r.md).
- Twenty-one validated pySHT-native methods expand the development API to 72
  canonical public functions. They add dense and fixed-small-sample mean
  tests, one-sample covariance tests, energy/kernel distribution tests,
  pairwise and mutual independence, multivariate normality, nearest-neighbor
  rectangular and simplex uniformity, compositional equality, and circular
  inference.
- Functions use lowercase `snake_case`, live in scientific category modules,
  and are not re-exported from top-level `pysht`.
- Immutable result types provide R `htest`-style frequentist output,
  resampling diagnostics, named numerical diagnostics, and log-domain Bayesian
  evidence where appropriate.
- The mean, variance, covariance, joint-parameter, equality-of-distributions,
  independence, normality, rectangular-uniformity, simplex, and circular
  families have public API and validation pages.
- Shared validators, local random-number handling, stable linear algebra,
  log-tail calculations, optimization checks, exact enumeration, and corrected
  Monte Carlo inference support the scientific functions.

The [API reference](../api/index.md) is authoritative for the implemented
surface. The [validation center](../validation/index.md) states the evidence
and advertised regime for each method.

## Release gates

A procedure is ready to ship only when it has:

1. hypotheses, assumptions, finite-sample formula, and calibration recorded in
   a primary-paper ledger;
2. an appropriate literal implementation or trusted independent comparator,
   plus fixed fixtures that exercise the production path;
3. scale-aware numerical comparisons, including log-tail comparisons where
   direct probabilities lose resolution;
4. every applicable order, exchange, scale, translation, feature, and
   transformation invariant;
5. singularity, minimum-sample, extreme-magnitude, malformed-input, and random-
   control boundary tests;
6. null-calibration and alternative-power simulations in each advertised
   regime; and
7. complete typing, result rendering, API documentation, migration mapping,
   source-distribution, and installed-wheel checks.

For an advertised asymptotic scenario, the release simulation uses 20,000 null
datasets and checks levels 0.01, 0.05, and 0.10 against the declared binomial
tolerance. Resampling methods emphasize exhaustive small cases, exact-versus-
Monte-Carlo comparisons, corrected tail counts, and fixed auxiliary
randomness. A method is not silently switched to a different statistic when a
gate fails.

## Current expansion status

The research program is organized by scientific dependency rather than by
catalog pressure:

- the foundational Chen--Qin, Li, CZZ, energy, distance-covariance, Rayleigh,
  and Watson methods are implemented and public;
- kernel distribution tests, mutual-independence, multivariate-normality,
  Hermans--Rasson, and circular multi-sample methods are implemented and
  public;
- Xue--Yao and Yu--Li--Xue have private research implementations but failed or
  have not completed the calibration evidence required for public exposure;
  JWJWZ and YLXL remain equation-audit targets with no callable; and
- EHY rectangular/simplex and alpha-energy compositional methods are
  implemented with explicitly prespecified tuning parameters.
- Ball Divergence remains private because a floating-distance equality rule
  has not yet reproduced literal closed-ball membership across both exact
  geometric symmetries and representably unequal near-ties.

The [native-method catalog](../methods/native-methods.md) is the authoritative
public-versus-blocked record. A blocked candidate does not receive a
`NotImplementedError` placeholder and does not delay an unrelated method that
has passed its own gates.

## Priorities after the current expansion

### Broaden the scientific evidence

- extend null-calibration and power studies beyond the deliberately narrow
  advertised regimes without implying a universal finite-sample guarantee;
- retain explicit warnings or narrow documented regimes whenever an
  asymptotic branch does not pass a new finite-sample calibration check;
- add independent high-precision fixtures for the most delicate quadrature,
  optimization, sparse-precision, and Bayes-factor kernels as new boundary
  cases are identified; and
- keep `mean2.2014CLX` private unless a practical end-to-end calibration gate
  passes for an estimator supported by the public interface.

### Improve reproducible validation

- add optimized, mechanically replayable runners for more of the expensive
  high-dimensional 20,000-dataset audits;
- retain dependency versions, integer seeds, raw rejection counts, and stream
  construction for every new validation table;
- extend wheel coverage when supported hosted runners become available; and
- keep the warning-clean website and combined GitHub Pages deployment current
  while preserving the simple navigation and single validation-ledger sidebar
  entry.

### Stabilize the public contract

- review function signatures, option vocabulary, result diagnostics, and
  statistic labels before the first API-stability commitment;
- benchmark exact enumeration, Monte Carlo calibration, random projections,
  CLIME optimization, and large covariance U-statistics; and
- define any future parallel-resampling design before adding a public
  parallelism control.

## Deliberate exclusions

The two R adapters `usek1d` and `useknd` have no Python counterpart. The
unexported experimental `cov1.mxPBF` and the invalid legacy Biswas--Ghosh
asymptotic branch remain outside the compatibility target. No public
`NotImplementedError` placeholder or R-shaped callable alias is planned.
The mean test `mean2.2014CLX` is a validation-blocked catalog identity rather
than a compatibility exclusion: its private implementation may become public
only after an honest end-to-end scenario passes the release-calibration gate.

Known corrections to legacy code are recorded in the
[legacy audit](../validation/legacy-audit.md), not hidden behind compatibility
switches.

## Stability policy during development

Before version 1.0, function names, signatures, return metadata, and supported
Python versions can change in a minor release. Changes must nevertheless be
intentional, tested, recorded in the changelog, and accompanied by migration
guidance when they affect users. Once a stable API is declared, incompatible
public changes are reserved for major releases.

## How to contribute evidence

Open a focused issue in the
[pySHT repository](https://github.com/kisungyou/pySHT/issues) with the
scientific use case, primary reference, target data regime, and an independent
comparison strategy. Reproducible fixtures and calibration scripts are more
useful than matching a legacy numerical value without establishing which
formula it represents.

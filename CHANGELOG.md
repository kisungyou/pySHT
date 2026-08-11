# Changelog

## Unreleased

## 0.1.0 - 2026-08-10

### Added

- Established the Python 3.12+ package, NumPy/SciPy runtime, C++17 native
  kernel, stable-ABI wheel, and typed-package scaffold.
- Implemented and exposed 51 canonical Python functions representing 52 of the
  54 public statistical routine identities in SHT 0.1.9. The catalog now covers
  univariate and multivariate means, variances, covariances, joint mean and
  variance, joint mean and covariance, equality of distributions, normality,
  rectangular uniformity, and simplex uniformity.
- Added immutable `StatisticalTestResult`, `HypothesisTestResult`,
  `ResamplingTestResult`, `DistanceTestResult`, and `BayesFactorTestResult`
  contracts. Frequentist results render in an R `htest`-style report;
  Bayesian results retain component log Bayes factors without fabricating a
  p-value.
- Added exact-enumeration and corrected Monte Carlo infrastructure, including
  `(b + 1) / (B + 1)` p-values, exceedance counts, conditional Monte Carlo
  standard errors, and 95% binomial tail-probability intervals.
- Added shared validation, local random-number handling, stable whitening and
  linear algebra, log-domain tail and Bayes-factor calculations, bounded
  optimization, and overflow-aware scaling utilities.
- Added the complete 56-export R migration crosswalk, searchable method-name
  glossary, category-organized API reference, interpretation-centered user
  guide, and method-specific validation ledgers.

### Changed

- Standardized every public function on lowercase `snake_case` author tokens
  and scientific category modules. Test functions are not re-exported from
  top-level `pysht`.
- Renamed the equality-of-distributions entry point directly to
  `pysht.equaldist.bg_2samp`; no obsolete `biswas_ghosh_2samp` alias is
  provided.
- Mapped both `mvar1.1998AS` and `mvar1.LRT` to the single algebraically
  identical `pysht.mean_variance.as_1samp` implementation.
- Withheld `mean2.2014CLX` from the public API after its practical-size Gumbel
  calibration failed the release gate. Its implementation and validation
  ledger remain private so the unresolved scientific limitation is visible.
- Withheld `cov1.2012Fisher` because its earlier optimized null-calibration
  evidence was not reproducible through the public path and practical fresh
  public-path regimes did not pass. The corrected implementation remains a
  private validation target.
- Made the Lee--You--Lin procedures return maximum and component log Bayes
  factors. Their published default shrinkage is derived from `alpha=2.01`,
  sample size, and dimension, with an explicit `gamma` override.
- Replaced the invalid Biswas--Ghosh asymptotic branch with exact or corrected
  Monte Carlo permutation calibration.
- Switched release-wheel construction to cibuildwheel so Linux artifacts are
  repaired to portable manylinux wheels while preserving the CPython 3.12+
  stable ABI.

### Corrected

- Repaired known SHT defects in null-covariance whitening, Li--Chen scaling,
  two-sided Wu--Li covariance tests, Pearson--Neyman and Muirhead rejection
  tails, Cao--Park--He group indexing and variance branches, adjusted and
  robust Jarque--Bera defaults, and two-sample Dempster calibration.
- Held random projection and subspace plans fixed across observed and
  permuted statistics, canonicalized seeded plans across row and group order,
  and avoided touching NumPy's global random state.
- Stabilized Zhang--Xu--Chen quadrature and roots, extreme-value tails,
  likelihood ratios, covariance traces, and Bayes-factor kernels for extreme
  float64 magnitudes.
- Centered samples against null or shared anchors before scaling throughout
  the mean, variance, covariance, normality, and joint-parameter paths, with
  explicit opposite-endpoint fallbacks. This preserves representable variation
  near very large common locations and across heterogeneous scales.
- Corrected the Yang--Modarres `Q3` asymptotic law for the nonzero correlation
  of its signed components, retained the published robust Jarque--Bera
  denominator constant 64 instead of SHT's erroneous 24, and made the
  Arnold--Shavelle calculation fully log-domain.
- Tightened exact-resampling count consistency, Bayes-factor component/max
  consistency, simplex-interior validation, optimizer convergence checks, and
  covariance eigensolver and extreme-tail boundaries.

### Validation and release engineering

- Added primary-paper formula ledgers, literal or trusted numerical oracles,
  fixed fixtures, transformation invariants, boundary tests, null-calibration
  audits, named-seed targeted alternative-power checks for every public
  frequentist procedure, Bayesian evidence-direction checks, and explicit
  legacy-correction records.
- Added strict typing, linting and formatting, branch-coverage enforcement,
  warning-as-error Sphinx builds, source-distribution checks, cross-platform
  portable wheel builds, and isolated installed-wheel smoke tests for the
  exact 51-function public surface on Python 3.12--3.14.
- Added a release-triggered, cross-platform GitHub Actions workflow for
  tokenless PyPI Trusted Publishing. The workflow assembles one source archive
  and four platform wheels, smoke-tests the exact artifacts, records SHA-256
  hashes, and grants OIDC permission only to the final protected publishing
  job.
- Added a minimum-supported NumPy/SciPy CI gate, replayable release-audit
  runners, and the BSD/MIT notices required by the native extension's statically
  linked nanobind and `tsl::robin_map` code.

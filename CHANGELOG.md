# Changelog

## Unreleased

## 0.5.0rc1 - 2026-09-25

Release candidate for testing before 0.5.0.

### Added

- Added 21 correctness-gated pySHT-native methods, bringing the candidate
  API to 72 canonical public functions. The additions comprise Chen--Qin
  and Li high-dimensional mean tests; Chen--Zhang--Zhong covariance tests;
  energy and MMD distribution tests; distance covariance,
  HSIC, dHSIC, and distance multivariance; multivariate Henze--Zirkler and
  energy normality tests; EHY rectangular and simplex uniformity; alpha-energy
  compositional equality; and four circular-data procedures.
- Added public `pysht.independence` and `pysht.circular` modules without
  top-level function re-exports, plus native-method API, chooser, glossary,
  formula-ledger, reproducible-audit, and installed-wheel coverage.
- Added shared distance, metric-canonicalization, Gram-matrix, exact-orbit,
  kernel-bandwidth, and multiplier-plan infrastructure. Built-in kernel
  choices are deliberately limited to RBF and Laplacian kernels with a fixed
  positive or observed-median bandwidth.

### Changed

- Renamed `mean_variance.as_1samp` to `lrt_1samp`,
  `mean_variance.zxc_2samp` to `exact_lrt_2samp`, and both Lee--You--Lin
  `lyl_2samp` functions to `maximum_pairwise_bayes_factor_2samp`. These are
  direct pre-1.0 breaking changes; no callable aliases are provided.
- Extended the API organization with native categories `[11] Independence`
  and `[12] Circular data`, while keeping the R migration crosswalk limited to
  SHT identities and recording new research methods in a separate catalog.

### Corrected

- Corrected Johansen's finite-sample F adjustment, including its exact
  reduction to the squared Welch test in one dimension.
- Included floating-point permutation ties in the LJW and Thulin upper tails,
  and rejected rank-deficient Hotelling inputs using equilibrated observation
  matrices instead of relying on Cholesky success alone.
- Treated saturated multivariate normality samples (`n = p + 1`) as the
  degenerate all-ties case with p-value one and an explicit diagnostic.
- Limited overflow fallback to affected features so normality and mean tests
  preserve representable small variations in other coordinates.
- Preserved Schott (2001) results across independent changes of feature units
  with observation-space SVD whitening and rank checks.
- Preserved representable variance-test tail probabilities when their F or
  chi-square pivots overflow or underflow in float64.
- Bounded metric canonicalization work and accepted duplicate compositions
  for Monte Carlo alpha-energy inference beyond small exact orbits. Bitwise
  seeded geometric replay is conditional on completing canonicalization;
  uniform randomization remains statistically valid after its fallback.
- Identified the Chen--Qin variance implementation as the Li--Chen unbiased
  trace-estimator variant and distinguished simulation release gates from
  exact finite-sample calibration, with binomial intervals in native audits.
- Made distance-, kernel-, independence-, circular-, and compositional
  statistics invariant to the transformations implied by each method and
  stabilized seeded resampling across tractable symmetric geometries and ties.
- Centered Biswas--Ghosh coordinates before numerical scaling so exact
  permutation inference preserves representable within-sample geometry near
  a huge common location instead of collapsing ULP-scale separations.
- Stabilized the alpha transformation continuously at `alpha=0`, including
  extremely small positive and negative values, while preserving the
  structural-zero rule for positive alpha.
- Made simplex domain validation and alpha-energy canonicalization invariant
  to a common component permutation at floating-point tolerance boundaries.
- Aligned circular Monte Carlo tie handling so an identical seeded null draw
  is always included in its own Rayleigh, Watson, or Hermans--Rasson upper
  tail despite harmless preprocessing-order roundoff.
- Removed the unsupported unique-minimum restriction from Li's multi-sample
  test, added literal tied-reference behavior, and validated a balanced
  three-group null design through a replayable 20,000-dataset scenario.
- Stabilized Srivastava--Du and the Yao, Krishnamoorthy--Yu, Johansen, and
  Nel--Van der Merwe mean tests for coordinates with radically different
  units, while replacing avoidable `p`-by-`p` trace, projection, and subspace
  work with observation-space equivalents where applicable.
- Preserved representable extreme Pearson--Neyman lower-tail probabilities
  when the likelihood ratio itself underflows, and bounded CLX entrywise
  variance evaluation so it no longer materializes an `n`-by-`p`-by-`p`
  tensor.
- Evaluated Student-t, chi-square, F, and Clopper--Pearson upper quantiles
  through survival functions so confidence bounds remain finite and
  representable when the requested confidence level is adjacent to one.
- Made shared resampling calibration reject NaN statistics explicitly and
  apply identical scalar and batched upper-tail ordering at positive and
  negative infinity.
- Stabilized the joint mean/covariance likelihood ratio for full-rank columns
  spanning hundreds of orders of magnitude, and removed unnecessary
  `p`-by-`p` identity Cholesky allocations from the default CZZ and LLZS
  paths.
- Replaced global-scale covariance definiteness checks with diagonal
  equilibration and entrywise symmetry checks, preserving valid null matrices
  whose marginal variances span the float64 exponent range.
- Stabilized Chen--Zhang--Zhong identity statistics when every coordinate is
  scaled to an extreme finite magnitude, reconstructing second- and
  fourth-order terms analytically instead of overflowing or underflowing
  intermediate powers.
- Made CI and release validation measure branch coverage explicitly and force
  a fresh, complete warning-as-error documentation build.

### Withheld by scientific gates

- Kept Xue--Yao `xy_2samp` private because the planned 999-draw calibration
  missed its audited null-size gate.
- Kept Yu--Li--Xue `ylx_2samp` private until the joint component-independence
  and 20,000-dataset combination gate pass. JWJWZ covariance and YLXL joint
  procedures have no public callable because their equation and calibration
  audits remain incomplete.
- Kept Ball Divergence private because the optimized floating-distance rank
  engine cannot yet distinguish every representable near-tie while preserving
  exact closed-ball symmetries; one adversarial case changed the literal exact
  p-value.

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

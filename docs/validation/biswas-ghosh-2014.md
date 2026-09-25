# Biswas--Ghosh (2014) two-sample test: validation record

## Scope and status

This record applies to `pysht.equaldist.bg_2samp`. It separates
four questions that must not be conflated:

1. whether the code evaluates the intended finite-sample statistic;
2. whether label permutations are calibrated correctly;
3. whether numerical safeguards preserve the mathematical test; and
4. whether an asymptotic approximation has been independently validated.

The first three have executable validation evidence. The fourth does not, so
asymptotic calibration is deliberately unavailable. This is a computational
validation record, not a new proof of the consistency or power results in the
paper.

Primary reference: M. Biswas and A. K. Ghosh, "A nonparametric two-sample test
applicable to high dimensional data," *Journal of Multivariate Analysis* 123
(2014), 160--171, <https://doi.org/10.1016/j.jmva.2013.09.004>. The authors'
preprint is available at
<https://www.isical.ac.in/~statmath/report/93114-3.pdf>.

The legacy comparison target is SHT 0.1.9 at Git commit
`4e29cda1257f86dd0237d37329af358b54d04f2b`, as recorded in the
[legacy audit](legacy-audit.md).

## Hypotheses and assumptions

The public alternative is two-sided in distribution space:

$$
H_0:F_X=F_Y,
\qquad
H_1:F_X\ne F_Y.
$$

The implemented randomization inference relies on the following conditions.

- `x` and `y` are independent samples, and observations are independent within
  each sample under the sampling design.
- Under the null, the pooled observations are exchangeable with respect to the
  two sample labels. Exactness is conditional on the pooled observations and
  follows from this exchangeability.
- Every observation is a finite real vector in a common Euclidean feature
  space. Rows are observations and columns are features. Univariate input is
  interpreted as a one-column matrix.
- Both sample sizes are at least two, because each within-sample distance mean
  has denominator $\binom{n}{2}$ or $\binom{m}{2}$.
- The two samples have the same number of features. Missing values, infinities,
  complex values, and Boolean arrays are outside the accepted data contract.

These assumptions exclude paired, clustered, stratified, weighted, and
serially dependent designs. Such designs require a permutation scheme that
respects their dependence structure; unrestricted relabeling is not valid for
them.

## Formula ledger

Let $X_1,\ldots,X_n\in\mathbb R^d$ and
$Y_1,\ldots,Y_m\in\mathbb R^d$. The distance is the ordinary Euclidean
metric $d(a,b)=\lVert a-b\rVert_2$.

| Symbol | Quantity evaluated | Finite-sample denominator |
|---|---|---:|
| $\bar d_{XX}$ | $\sum_{i<j}d(X_i,X_j)\big/\binom n2$ | $\binom n2$ |
| $\bar d_{YY}$ | $\sum_{i<j}d(Y_i,Y_j)\big/\binom m2$ | $\binom m2$ |
| $\bar d_{XY}$ | $\sum_i\sum_j d(X_i,Y_j)/(nm)$ | $nm$ |
| $T_{nm}$ | $(\bar d_{XX}-\bar d_{XY})^2+(\bar d_{XY}-\bar d_{YY})^2$ | not applicable |

Thus `result.statistic` is non-negative and has units of squared distance. No
diagonal zero is included in either within-sample mean, and each unordered
within-sample pair is included exactly once. Every cross-sample pair is
included exactly once.

This ledger is checked directly against Section 3, pp. 4–5 of the authors'
preprint, where the two estimated distance-mean vectors and
$T_{m,n}=\lVert\widehat\mu_{D_F}-\widehat\mu_{D_G}\rVert^2$ are displayed and
permutation calibration is prescribed for small samples. It is not inferred
only from the legacy R implementation.

The implementation uses the upper-tail ordering: a relabeling is at least as
extreme as the observation when its statistic is greater than or tied with the
observed statistic.

## Exact and Monte Carlo calibration

Write $N=n+m$, $C=\binom{N}{n}$, and let $T(A)$ be the statistic after
assigning the index subset $A$ of size $n$ to the first group. All fixed-size
labelings are equiprobable under the exchangeable null.

For exact calibration, all $C$ subsets are enumerated. If

$$
b=\#\{A:T(A)\mathrel{\geq}T_{\mathrm{obs}}\},
$$

then the reported p-value is $b/C$. The observed labeling is in the reference
set, so an exact p-value cannot be zero. Complementary labelings are not
deduplicated: they are separate randomization outcomes even when equal sample
sizes make their statistics identical. `n_resamples` is reported as $C$, and
the Monte Carlo standard error is `None`.

For Monte Carlo calibration, $B$ label subsets are sampled independently and
uniformly; sampling is without replacement *within* a labeling, but the same
labeling can occur in different draws. With $b$ sampled statistics at least
as extreme as the observation, the reported value is

$$
\widehat p=\frac{b+1}{B+1}.
$$

The correction prevents spuriously reporting a zero p-value from a finite
simulation. Let $\widehat q=b/B$ be the plug-in estimate of the underlying
permutation-tail probability. The accompanying diagnostic is

$$
\widehat{\operatorname{se}}(\widetilde p)
=\frac{\sqrt{B\widehat q(1-\widehat q)}}{B+1},
\qquad \widetilde p=\frac{b+1}{B+1}.
$$

It describes Monte Carlo sampling error; it is not a confidence interval for
the scientific effect and does not account for violations of exchangeability.

The calibration selector has these exact semantics:

| `calibration` | Behavior |
|---|---|
| `"exact"` | Enumerate all $C$ labelings; fail if `n_resamples < C`. |
| `"monte-carlo"` | Draw exactly `n_resamples` labelings, even when enumeration would be affordable. |
| `"permutation"` | Enumerate when $C$ does not exceed `n_resamples`; otherwise use Monte Carlo. |

The legacy `"asymptotic"` selector is intentionally not part of the Python
API. Passing it is an invalid option rather than a public placeholder or a
silent fallback.

## Numerical normalization and raw-statistic semantics

Directly subtracting coordinates near the extremes of float64 can overflow,
while squaring very small distance contrasts can underflow. The implementation
therefore performs calibration on a dimensionless distance matrix.

For each feature, the pooled coordinates are first centered at the
overflow-safe midpoint

$$
c_j=\frac{\min_i z_{ij}}2+\frac{\max_i z_{ij}}2.
$$

The centered coordinates are divided by their largest absolute value (unless
all are zero). Centering first is essential when observations differ by only
a few representable units around a huge common location: scaling the raw
coordinates first can round those differences away. Euclidean distances are
then evaluated and divided by their largest value. Conceptually, if
$d_{\max}$ is the largest pooled pairwise distance, calibration uses

$$
\widetilde d_{ij}=d_{ij}/d_{\max},
\qquad
\widetilde T=T_{nm}/d_{\max}^2.
$$

A common positive rescaling multiplies every permutation statistic by the
same positive constant and therefore cannot change their ordering in exact
arithmetic. Translation, orthogonal rotation, reflection, and feature
permutation leave Euclidean distances unchanged. These facts provide useful
metamorphic checks independent of any fixed numeric answer.

The result preserves both numerical and scientific units:

- `normalized_statistic` is $\widetilde T$, the dimensionless value used for
  all permutation comparisons;
- `distance_scale` is the reconstructed $d_{\max}$; and
- `statistic` is the reconstructed raw $T_{nm}$ in squared-distance units.

For moderate data, `statistic == normalized_statistic * distance_scale**2` up
to floating-point rounding. If the mathematically scaled raw result lies
outside the representable float64 range, `statistic` may be `0.0` or `inf`.
The normalized statistic and permutation p-value remain meaningful. If all
pooled observations coincide, `distance_scale`, `normalized_statistic`, and
`statistic` are all zero and every labeling is an upper-tail tie.

## Floating-point tie rule

Permutation distributions are discrete, and rigid transformations can turn a
theoretical equality into adjacent floating-point values. The comparison is
therefore made on normalized statistics using

$$
T_{\mathrm{perm}}
\ge T_{\mathrm{obs}}-|T_{\mathrm{obs}}|\,(100\epsilon),
$$

where $\epsilon$ is double-precision machine epsilon. This is a relative
allowance of about $2.22\times10^{-14}$, with no positive absolute allowance
when the observed statistic is zero. Ties are included in the upper tail. The
rule is intentionally narrow: it protects algebraic ties from roundoff but is
not a user-selectable scientific tolerance.

## Differences from legacy SHT

The finite-sample statistic above agrees algebraically with the statistic in
the legacy R function `eqdist.2014BG`. pySHT intentionally differs in its
calibration and numerical contract.

| Area | Legacy SHT 0.1.9 | pySHT |
|---|---|---|
| Public location | `eqdist.2014BG` | `pysht.equaldist.bg_2samp` |
| Monte Carlo p-value | $b/B$, which can be zero | $(b+1)/(B+1)$ plus reported Monte Carlo SE |
| Exhaustive calibration | Not exposed | Explicit `"exact"` mode and automatic enumeration within budget |
| Default simulation budget | 999 | 9,999 |
| Numerical scale | Raw pairwise distances | Dimensionless calibration plus reconstructed raw statistic |
| Reproducibility invariants | Random sequence depends on input order | Integer seed is stable under row reordering and group exchange |
| Inputs | Several values are silently coerced by R | Strict dimensions, numeric type, finiteness, and integer controls |
| Result | Mutable R list with class `htest` | Immutable result with an `htest`-style display and explicit diagnostics |

Most importantly, legacy asymptotic calibration is not reproduced. This is
not a claim that SHT simply mistranscribed the article. Section 4, Theorem 4.1
and the displayed estimators $S_1,S_2$ on pp. 12–13 define the fixed-dimension
large-sample chi-square calibration. As printed, each $i<j<k$ term uses only
the lowest-index observation as the shared distance vertex. The legacy helper
has the same finite-sample row-order dependence. A symmetrized U-statistic
could remove it, but that would be a new procedure requiring a derivation and
an independent calibration study. The public API therefore retains the
paper-supported permutation test and withholds the asymptotic selector rather
than presenting legacy parity as validation.

## Targeted alternative-power gate

This gate targets a scale-distribution difference with equal means, rather
than reducing the omnibus procedure to a mean-shift example. Seed 2026090321
initializes a `SeedSequence`. In each of 300 outer replications, persistent
child-0 PCG64 first draws 15 rows from $N_5(0,I)$ and then 15 standard-normal
rows multiplied by 1.8, giving $N_5(0,3.24I)$. Persistent child-1 PCG64 is
passed to the exact public call
`equaldist.bg_2samp(x, y, calibration="monte-carlo", n_resamples=199,
rng=auxiliary_rng)`. It supplies the 199 label permutations for that dataset;
both streams then advance to the next replication. Rejection means
`pvalue < alpha`.

| alpha | 0.01 | 0.05 | 0.10 |
|---:|---:|---:|---:|
| Rejections / 300 | 270 | 298 | 299 |
| Rate | 0.900 | 0.993333 | 0.996667 |

The corrected inner p-values have grid width `1/200`, and the outer estimate
has worst-case binomial standard error 0.0289. This deliberately strong
alternative is a tractable response check, not a comparative or local-power
claim. Reproduce it with `python -m tools.mean_power_audits
equaldist.bg_2samp` under Python 3.12.13, NumPy 2.5.1, and SciPy 1.18.0.

## Validation evidence

The independent checks live in `tests/test_equaldist_reference.py`. They use
SciPy's `pdist` and `cdist` directly, never a private pySHT helper or the native
distance matrix under test.

| Evidence | What it establishes |
|---|---|
| Hand fixture `x=[0,2]`, `y=[1,5,8]` | Raw $T=40/9$, $d_{\max}=8$, normalized $T=5/72$, and exact p-value $8/10$. |
| Two-dimensional rectangle fixture | Raw $T=7-\sqrt{13}$; observed/complement ties produce exact p-value $2/6$. |
| Exhaustive SciPy oracle | The public exact result matches a fresh evaluation of the literal formula for every one of $\binom73=35$ labelings. |
| Translation, rotation, feature permutation | Statistic and exhaustive upper-tail count respect Euclidean invariance. |
| Common rescaling | Raw statistic has degree two while normalized inference and the p-value are invariant. |
| Huge common location | Univariate and bivariate ULP-scale fixtures near $10^{300}$ reproduce the recentered geometry, raw-scale overflow, and every exact-orbit exceedance. |
| Existing implementation tests | Row/group-order invariance, constant samples, extreme scales, input rejection, deterministic seeds, and Monte Carlo correction remain covered. |

Reproduce the independent layer with:

```console
python -m pytest tests/test_equaldist_reference.py -q
```

A green test run is necessary but not sufficient for release. Future changes
to the statistic, distance metric, normalization, or tie rule require updating
this ledger only after the new behavior has an independent mathematical
oracle. Asymptotic calibration remains a separate blocked validation item.

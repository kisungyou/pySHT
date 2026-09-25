# Covariance tests: formula and validation ledger

This ledger covers the nine public functions in `pysht.covariance`, one
withheld Fisher implementation, and two correctness-gated research
candidates. The
pinned SHT 0.1.9 source was inspected for migration, but primary-paper
equations, intentionally literal implementations, algebraic invariants, and
null simulations are the correctness oracles.

## Common numerical policy

Except for the paper-specific Lee--You--Lin known-zero-mean model, each group
has a feature-wise anchor removed in the original coordinates before centering
and division by one common finite scale. An opposite-sign overflow fallback
performs the subtraction in normalized coordinates. Covariance products are
therefore evaluated without avoidable overflow, while standardized statistics
remain unchanged. Tests cover separately shifted locations as large as
$10^{14}$ while preserving exactly representable within-group variation,
common scales through $10^{100}$, group exchange, group order, and feature
permutation where the method implies those invariances. Undefined zero
variance estimates and singular matrices are rejected before a p-value is
formed.

## Fisher one-sample test (withheld)

`fisher_1samp` is not part of the 0.1.0 public API. The private
`_fisher_1samp` implementation is retained for continued validation. For $N$
observations, write $n=N-1$, $c=p/n$, and let $S$ be the unbiased
sample covariance **after** centering and whitening by
$\Sigma_0^{-1/2}$. Fisher's unbiased spectral-moment estimators
$a_1,\ldots,a_4$ are inserted into

$$
T_1=\frac{n}{c\sqrt 8}(a_4-4a_3+6a_2-4a_1+1)
$$

and

$$
T_2=\frac{n}{\sqrt{8(c^2+12c+8)}}(a_4-2a_2+1).
$$

Both use an upper standard-normal tail. A fixed nonidentity-null fixture is
checked against an independent trace-power implementation for both variants.
An invertible feature transformation applied simultaneously to the data and
null covariance leaves the result unchanged. When the fourth-order
polynomial lies outside float64, normalized trace estimators and a signed
log-sum evaluation preserve its limiting sign instead of leaking an arithmetic
overflow. The sign matters because an unbiased fourth-moment estimator need
not be nonnegative in a very small finite sample, even though its population
target is nonnegative.

**Legacy correction.** SHT calculated whitened observations but passed the
unwhitened sample covariance to the moment estimator. Consequently its stated
nonidentity null was not tested. pySHT computes every $a_r$ from the whitened
covariance.

The normal approximation is finite-sample sensitive. A historical optimized
scatter simulation at $(N,p)=(201,800)$ was not retained as a reproducible
public-call runner. Fresh public-path probes at the same aspect ratio but
$(N,p)=(51,200)$ and $(101,400)$ failed calibration for at least one variant
and level. Consequently neither null calibration nor targeted power has met
the project's reproducibility gate, and exposing the function would violate
the 0.1.0 release policy. No historical rate is used as release evidence.

This caution also has primary-author support independent of the failed pySHT
audit. Fisher's dissertation, Section 2.3, derives the identity-covariance
construction; on printed page 27 it says the fourth-moment estimator's large
variance can hinder the proposed statistic and that more work is necessary.
The conclusion on printed page 28 says the construction could not then be
justified. Those are PDF pages 36--37 in the repository copy. See
[Fisher (2009)](https://open.clemson.edu/all_dissertations/486/).

Primary source: {cite:t}`fisher-covariance-2012`.

## Chen--Zhang--Zhong identity and sphericity tests

Let $N$ be the number of rows, $p$ the number of features, and $X_i$ the
observations after centering. For identity testing, a supplied positive-
definite null covariance is first removed by a Cholesky solve, so the null is
$\Sigma=I_p$. Chen, Zhang, and Zhong's unbiased estimators are

$$
T_1=\frac{1}{N-1}\sum_i X_i^T X_i,
$$

and the order-four estimator $T_2$ of
$\operatorname{tr}(\Sigma^2)$, evaluated by the same exact mutually-distinct-
index Gram identities used by the Li--Chen oracle. The reported identity
statistic is

$$
\frac{N}{2}\left(\frac{T_2}{p}-\frac{2T_1}{p}+1\right),
$$

while the scale-free sphericity statistic is

$$
\frac{N}{2}\left(\frac{pT_2}{T_1^2}-1\right).
$$

Both use an upper standard-normal limit. Literal four-loop fixtures reproduce
$T_2$. The identity fixture verifies equivalence after simultaneously
transforming the data and `popcov`; the sphericity fixture covers row order,
translation, orthogonal rotation, and scales through $10^{130}$. At least four
rows and two features are required. Non-positive covariance trace, a singular
null covariance, and non-finite finite-sample estimates are rejected.

Both tests evaluate $T_2$ through an $N$-by-$N$ observation Gram matrix, with
$O(N^2p)$ time and $O(Np+N^2)$ storage. Identity-null whitening adds
$O(p^3+Np^2)$ time and $O(p^2)$ storage only when an explicit `popcov` is
supplied; the default identity null does not allocate or factor a $p$-by-$p$
matrix. Sphericity needs no feature-space matrix. The identity implementation
also carries a common data scale analytically through the second- and fourth-
order terms. Thus every finite scalar alternative remains evaluable even when
its covariance underflows or its fourth-order trace overflows in raw float64
arithmetic; a fixed fixture covers scales $10^{-300}$ and $10^{100}$.

Each null audit reset one PCG64 generator with the named integer seed and, in
each replication, drew one `standard_normal((100, 100))` array in C row-major
order before calling the public function once. No samples or fitted quantities
were reused across replications. The opt-in
`tools/native_null_audits.py` runner preserves this exact public-call and
random-stream contract.

| public function | seed | rejection counts at 0.01, 0.05, 0.10 | rates | gate |
|---|---:|---:|---:|---|
| `czz_identity_1samp` | 20260835 | 260, 1090, 2119 | 0.01300, 0.05450, 0.10595 | Pass |
| `czz_sphericity_1samp` | 20260836 | 231, 1068, 2025 | 0.01155, 0.05340, 0.10125 | Pass |

The targeted audits used 2,000 independent arrays from a freshly reset PCG64
stream. In each alternative the first coordinate had standard deviation 1.35
and the other 99 coordinates had standard deviation one.

| public function | seed | rejection counts at 0.01, 0.05, 0.10 | rates |
|---|---:|---:|---:|---|
| `czz_identity_1samp` | 20260945 | 72, 209, 356 | 0.0360, 0.1045, 0.1780 |
| `czz_sphericity_1samp` | 20260946 | 65, 218, 372 | 0.0325, 0.1090, 0.1860 |

These alternatives are intentionally modest checks of direction, not minimum-
power promises. Primary source: [Chen, Zhang, and Zhong
(2010)](https://doi.org/10.1198/jasa.2010.tm09560), Equations (2.1)--(2.3).

## Wu--Li random-projection tests

For each Gaussian unit vector $R_r$, the one-sample statistic uses

$$
Z_r=\sqrt{2\sum_i (R_r^T\widetilde X_i)^2}-\sqrt{2(N-1)-1}.
$$

The two-sample statistic uses the variance-stabilized log ratio

$$
Z_r=\frac{\log(s_{1r}^2/s_{2r}^2)}
{\sqrt{2/(N_1-1)+2/(N_2-1)}}.
$$

Each group's centered projections are normalized separately and their scalar
units are restored after taking logarithms. Consequently, a finite log
variance ratio remains evaluable even when forming $s_{1r}^2/s_{2r}^2$
directly would overflow or one group's squared working values would underflow.

Because covariance equality has an unrestricted alternative, pySHT reports
$M=\max_r|Z_r|$ and

$$
P\{\max_{1\le r\le m}|Z_r|\ge M\}
\approx 1-\{2\Phi(M)-1\}^m.
$$

The survival probability is evaluated from the small marginal tail
$2\Phi(-M)$ with `log1p`/`expm1`. This avoids rounding
$2\Phi(M)-1$ to one, which would otherwise report a zero p-value around
$M=9$ even though the maximum-test tail remains representable.

This is a genuine two-sided maximum law under the paper's asymptotic
independence argument. The paper derives its procedure for known zero means;
pySHT deliberately extends it to unknown means by centering and using $N-1$
rather than $N$ degrees of freedom. The two 20,000-run audits below validate
that extension in the stated Gaussian regimes. Seed replay, global RNG
isolation, group exchange, both low- and high-variance alternatives, and the
literal maximum-normal probability are tested.

**Legacy correction.** SHT returned only `max(Z_r)` with a one-sided tail
while describing a two-sided equality test. That loses power when projected
variances are smaller, and the two-sample answer can change after exchanging
the groups. pySHT uses `max(abs(Z_r))` and the corresponding two-sided tail.

Primary source: {cite:t}`wu-li-2015`.

Fresh 20,000-replication public-call audits use the scenario keys and stream
contract recorded below. The two-sample method failed at
$(N_1,N_2,p,m)=(100,120,30,50)$ and is therefore advertised only at the larger
validated sample-size regime.

| scenario key | seed | rejection counts at 0.01, 0.05, 0.10 | rates |
|---|---:|---:|---:|
| `covariance.wl-1samp.n100-p30-m25` | 20260831 | 222, 987, 1963 | 0.01110, 0.04935, 0.09815 |
| `covariance.wl-1samp.n100-p30-m25` | 20260901 | 205, 990, 2019 | 0.01025, 0.04950, 0.10095 |
| `covariance.wl-2samp.n300-n360-p30-m50` | 20260831 | 233, 1030, 2074 | 0.01165, 0.05150, 0.10370 |
| `covariance.wl-2samp.n300-n360-p30-m50` | 20260901 | 214, 1042, 2033 | 0.01070, 0.05210, 0.10165 |

## Li--Chen two-sample test

The full path evaluates the mutually-distinct-index U-statistics

$$
A_{nh}=\frac{\sum_{i\ne j}(X_{hi}^TX_{hj})^2}{n_h(n_h-1)}
-\frac{2\sum^*_{i,j,k}X_{hi}^TX_{hj}X_{hj}^TX_{hk}}
{n_h(n_h-1)(n_h-2)}
+\frac{\sum^*_{i,j,k,l}X_{hi}^TX_{hj}X_{hk}^TX_{hl}}
{n_h(n_h-1)(n_h-2)(n_h-3)}
$$

and Equation (2.2)'s four-term $C_{n_1n_2}$. The unstandardized statistic is

$$
T=A_{n1}+A_{n2}-2C_{n_1n_2}.
$$

Under the null, the paper's standard-deviation estimator is

$$
\widehat\sigma_0=2A_{n1}/n_2+2A_{n2}/n_1,
\qquad L=T/\widehat\sigma_0.
$$

An independent four-loop fixture reproduces every ordered sum. The optimized
implementation reduces the computation to Gram-matrix row sums without
downgrading the U-statistic.

A former `unbiased=False` centered leading-term shortcut is deliberately not
part of the 0.1.0 signature. In two independent 2,000-replication Gaussian
null probes at $(n_1,n_2,p)=(30,30,50)$ (integer seeds 20260831 and 20260901,
one reset stream per seed), its nominal 0.05 rejection rates were 0.2185 and
0.2095 (437/2000 and 419/2000). That branch therefore failed before the
release-scale gate; pySHT exposes only the literal U-statistic path that passed
the 20,000-run audits.

**Legacy correction and typesetting audit.** SHT divided $T$ by
$\sqrt{\widehat\sigma_0}$. Equation (2.7), physical units, scale invariance,
and the authors' published reference code all identify the displayed linear
combination as the estimated **standard deviation**, despite an inconsistent
square in nearby prose. pySHT divides by it directly. At the fixed ledger
fixture, the corrected and square-rooted values differ materially.

A fresh public-call normal-null audit with $(n_1,n_2,p)=(30,30,50)$ produced
counts `(190, 1031, 2060)` and rates `(0.00950, 0.05155, 0.10300)` for seed
20260831, and counts `(227, 1053, 2082)` and rates
`(0.01135, 0.05265, 0.10410)` for seed 20260901. Both 20,000-run streams pass.

Primary source: {cite:t}`li-chen-2012`.

## Cai--Liu--Xia maximum test

Let $\widehat\sigma_{k,ij}$ be the maximum-likelihood sample covariance and

$$
\widehat\theta_{k,ij}=n_k^{-1}\sum_l
\{(X_{kl,i}-\bar X_{k,i})(X_{kl,j}-\bar X_{k,j})
-\widehat\sigma_{k,ij}\}^2.
$$

pySHT reports

$$
M=\max_{i,j}\frac{(\widehat\sigma_{1,ij}-
\widehat\sigma_{2,ij})^2}
{\widehat\theta_{1,ij}/n_1+\widehat\theta_{2,ij}/n_2}
$$

with the published type-I extreme-value survival probability. The tail is
evaluated with `expm1` and a log rate, avoiding cancellation near zero. A
literal double-loop fixture and all group, location, scale, and feature-order
invariances pass. The implementation evaluates the literal centered-product
squares in bounded feature blocks. This avoids both cancellation in the
algebraic fourth-moment subtraction and an unbounded
$O((n_1+n_2)p^2)$ product tensor; working storage is
$O(n_1p+n_2p+p^2)$ plus a fixed-size block. Fresh 20,000-run public-call normal-null streams at
$(n_1,n_2,p)=(100,100,30)$ produced counts `(163, 994, 2037)` and rates
`(0.00815, 0.04970, 0.10185)` for seed 20260831, and counts
`(176, 958, 2044)` and rates `(0.00880, 0.04790, 0.10220)` for seed 20260901.
Both pass, while the output remains correctly labeled asymptotic.

Primary source: {cite:t}`cai-liu-xia-covariance-2013`.

## Lee--You--Lin maximum pairwise Bayes factor

For every ordered pair $i\ne j$, the method compares equality of the two
conditional regressions under the paper's known-zero-mean Gaussian model.
Writing $X_i$ and $X_j$ for the raw variable columns, pySHT calculates

$$
\widehat\tau_{k,ij}=n_k^{-1}
X_{k,i}^T(I-H_{X_{k,j}})
X_{k,i}
$$

with ordinary least squares and no intercept, and inserts the two group
residuals and the pooled residual into Equations (12)--(14). No automatic
centering is performed. `gammaln`, logarithms, pairwise scaling, and `logaddexp`
are used throughout. Equation (15)'s maximum is retained in log form; the full
ordered matrix is immutable, its diagonal is `-inf`, and no frequentist p-value
or automatic threshold is invented.

An independent implementation of Equations (12)--(15) reproduces every
component, including the recommended `a0=b0=0.01` and
`gamma=max(n_1+n_2,p)^(-2.01)` defaults. Tests cover group exchange, feature
permutation, explicit `gamma` overrides, extreme float64 scales, and
simultaneous conversion of `b0` with squared data units. Translation invariance
is intentionally not claimed because it would contradict the known-zero-mean
model.

**Legacy corrections.** The old C++ kernel cast log arguments to `float` and
weakened the OLS projection by `(1+gamma)`. The primary equations put `gamma`
only in the prior penalty. pySHT uses ordinary OLS residuals and float64
log-domain arithmetic.

The ordered conditional regressions take
$O((n_1+n_2)p^2)$ time. The input arrays and complete component matrix require
$O((n_1+n_2)p+p^2)$ storage; no three-dimensional product tensor is formed.

Primary source: {cite:t}`lee-you-lin-2024`.

## Schott multi-sample tests

For the 2001 Wald test, let $n_i=N_i-1$,
$S=\sum_i n_iS_i/\sum_i n_i$, and
$A_i=S^{-1/2}S_iS^{-1/2}$. The implemented nonnegative form is

$$
W=\frac{\sum_i n_i}{2}\sum_i\frac{n_i}{\sum_jn_j}
\|A_i-I\|_F^2,
\qquad
W\overset a\sim\chi^2_{(g-1)p(p+1)/2}.
$$

It is algebraically identical to the paper's double-trace expression and is
checked against that literal form. A singular pooled covariance is outside
the Wald statistic's domain. SHT silently fell back to a pseudoinverse;
pySHT rejects it. Rank is evaluated by SVD on the stacked, separately centered
observations after equilibration of each feature, and whitening is performed
directly through the left singular vectors. This preserves independent
changes of feature units without squaring the condition number in a pooled
scatter matrix. The threshold is relative to the largest singular value and
the dimensions of the equilibrated observation matrix. Exact linear
dependencies are rejected; individual groups may be singular if their pooled
within-group scatter is full rank.

The 2007 test uses the paper's pairwise bias-corrected Frobenius statistic,
pooled trace estimator, and normalizing $\theta$. A separate transcription of
every pairwise term agrees with the optimized implementation. Group order,
separate locations, and common scale do not affect either Schott result.
The 2001 statistic additionally has full nonsingular affine invariance;
regressions cover feature units from $10^{-200}$ to $10^{200}$ and a
resolvable nearly dependent transformation. The 2007 Frobenius statistic does
not claim invariance to arbitrary separate feature rescaling. Both current
implementations require feature-space $p\times p$ work: the 2001 SVD costs
$O(Np^2)$ when $N>p$, and the 2007 scatters cost $O(Np^2)$ time and
$O(gp^2+Np)$ storage. They are not bounded-memory streaming methods in $p$.

The fresh 20,000-replication public-call Gaussian-null audit was:

| scenario key | seed | rejection counts at 0.01, 0.05, 0.10 | rates |
|---|---:|---:|---:|
| `covariance.schott-2001.g3-n100-p3` | 20260831 | 193, 1003, 2005 | 0.00965, 0.05015, 0.10025 |
| `covariance.schott-2001.g3-n100-p3` | 20260901 | 183, 901, 1959 | 0.00915, 0.04505, 0.09795 |
| `covariance.schott-2007.g3-n100-p50` | 20260831 | 243, 1100, 2110 | 0.01215, 0.05500, 0.10550 |
| `covariance.schott-2007.g3-n100-p50` | 20260901 | 227, 1045, 2095 | 0.01135, 0.05225, 0.10475 |

Every entry satisfies the project's release tolerance. A preliminary
$p=30$ audit missed the 0.05 boundary by two rejections for one seed, so that
smaller-dimensional finite regime is not advertised as calibrated.

Primary sources: {cite:t}`schott-covariance-2001` and
{cite:t}`schott-covariance-2007`.

## Reproducible stream and power evidence

`python -m tools.covariance_release_audits --all` resets each scenario/seed
pair, constructs `SeedSequence(seed)`, spawns the data stream first and the
auxiliary-randomness stream second, and advances both in replication order.
The runs above used Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0, and pySHT
0.1.0. Every displayed count is from the public function named by its
scenario, not from a duplicated optimized formula.

Targeted alternatives used 2,000 replications, integer seed 20260902, and the
same reset/split policy with `SeedSequence([seed, method_index])`. The table
reports rejection counts and rates at level 0.05.

| public function | fully specified alternative | count | rate |
|---|---|---:|---:|
| `wl_1samp` | Gaussian, $N=100,p=30,\Sigma=1.5I$ | 2000 | 1.0000 |
| `wl_2samp` | Gaussian, $N_1=300,N_2=360,p=30,\Sigma_1=I,\Sigma_2=1.3I$ | 1997 | 0.9985 |
| `lc_2samp` | Gaussian, $n_1=n_2=30,p=50,\Sigma_1=I,\Sigma_2=2I$ | 1799 | 0.8995 |
| `clx_2samp` | Gaussian, $n_1=n_2=100,p=30$; second-group first SD 1.8 | 1817 | 0.9085 |
| `schott_2001_ksamp` | Gaussian, $g=3,N_i=100,p=3$; third-group SD $(1.6,1,1)$ | 1947 | 0.9735 |
| `schott_2007_ksamp` | Gaussian, $g=3,N_i=100,p=50$; third-group covariance $1.5I$ | 1968 | 0.9840 |

For `maximum_pairwise_bayes_factor_2samp`, the same seed and 2,000 paired draws used
$n_1=n_2=50,p=10$ and correlation 0.7 between the first two coordinates only
in the alternative second group. The median maximum log Bayes factor moved
from -5.38217 under the null comparator to 1.66247 under the alternative; the
alternative exceeded its paired null value in 1919/2000 draws (0.9595). This
is evidence direction, not a fabricated frequentist rejection rule.

## Validation-blocked advanced combinations

### Yu--Li--Xue covariance combination

The private `_ylx_2samp` implementation computes the Li--Chen log upper tail
$\ell_{LC}$ and the CLX extreme-value log upper tail $\ell_{CLX}$ before
forming the paper's Fisher statistic

$$
-2(\ell_{LC}+\ell_{CLX})
$$

and its $\chi^2_4$ reference probability. A fixed fixture independently
reconstructs both log tails and the combination, including cases where
exponentiating a component first would underflow. It is not in `__all__` and
there is no public `ylx_2samp` attribute. The paper's calibration depends on
the asymptotic independence of the dense and maximum components; a named-seed
20,000-null joint component-independence and size audit has not yet been
completed in an advertised regime. The method therefore fails the release
evidence gate even though the algebraic composition is implemented. Primary
source: [Yu, Li, and Xue
(2024)](https://doi.org/10.1080/01621459.2022.2126781).

The private composition inherits the two component costs: its time is
$O(\{n_1^2+n_2^2+n_1n_2\}p+(n_1+n_2)p^2)$ and its peak storage is
$O(n_1^2+n_2^2+n_1n_2+(n_1+n_2)p+p^2)$. Those bounds are recorded for audit
purposes only and do not relax the statistical release gate.

### Jiang--Wang--Jiang--Wang--Zhang random integration

`jwjwz_2samp` has no public or private callable. The primary paper was
obtained, but the random-integration target, finite-sample estimator, and
calibration have not yet been reconciled with a literal independent oracle.
Consequently there are no fixed fixtures, 20,000-null gate, or targeted-power
audit. Its implementable time and storage complexity therefore also remain
uncertified; publishing either a callable or a performance claim at this stage
would create the prohibited placeholder API. Primary source: [Jiang et al.
(2023)](https://doi.org/10.5705/ss.202020.0486).

## Legacy mapping

| pySHT | SHT 0.1.9 | Deliberate change |
|---|---|---|
| `_fisher_1samp` (withheld) | `cov1.2012Fisher` | Corrected implementation retained privately; reproducible release gates incomplete |
| `czz_identity_1samp` | -- | pySHT-native high-dimensional identity test; known SPD null allowed by whitening |
| `czz_sphericity_1samp` | -- | pySHT-native scale-free high-dimensional sphericity test |
| `wl_1samp` | `cov1.2015WL` | Genuine two-sided maximum and local RNG |
| `lc_2samp` | `cov2.2012LC` | Literal U-statistics; divide by estimated SD, not its square root |
| `clx_2samp` | `cov2.2013CLX` | Stable extreme-value tail and strict domains |
| `wl_2samp` | `cov2.2015WL` | Genuine two-sided, group-symmetric maximum |
| `maximum_pairwise_bayes_factor_2samp` | `cov2.mxPBF` | Published OLS residuals, log output, no fabricated p-value |
| `schott_2001_ksamp` | `covk.2001Schott` | Reject singular pooled covariance; no pseudoinverse fallback |
| `schott_2007_ksamp` | `covk.2007Schott` | Stable common scaling and literal correction factors |

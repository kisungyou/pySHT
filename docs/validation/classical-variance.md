# Classical variance tests: formula and validation ledger

This ledger covers `pysht.variance`. The implementations use normalized or
log-domain arithmetic and SciPy distribution functions; output from legacy SHT
is not treated as the correctness oracle.

## One-sample chi-square test

For independent normal observations and null variance $\sigma_0^2>0$,

$$
X^2=\frac{(n-1)S^2}{\sigma_0^2}\sim\chi^2_{n-1}.
$$

Lower, upper, and doubled minimum-tail p-values are supported. Confidence
intervals invert the same chi-square pivot. Sample variance is computed in the
log domain after scaling, avoiding direct squaring of very large observations.
A constant sample is represented by the mathematically limiting statistic and
interval zero; under a positive normal-theory null its two-sided p-value is
zero. A permutation-invariant sample anchor is subtracted before scaling, so
representable ulp-sized spreads near offsets such as $10^{100}$ are not
rounded a second time during normalization.

Validation includes a literal fixed-data formula, pivot-inversion checks for
two- and one-sided intervals, joint data/null rescaling through approximately
$10^{\pm100}$, strict control validation, and `htest`-style rendering.

The p-value retains the log pivot independently of its displayed statistic.
For $z=X^2/2$ below half of machine epsilon, the implementation evaluates
$\log P(a,z)=a\log z-\log\Gamma(a+1)$, where $a=(n-1)/2$.
The omitted factor is bounded between $e^{-z}$ and 1, so its relative
effect is below machine epsilon. This follows directly from the
[incomplete-gamma integral and series](https://dlmf.nist.gov/8.7).
For ordinary pivots, SciPy supplies both log tails. Subnormal probabilities
are reevaluated before SciPy's intermediate arithmetic can
underflow: a positive lower-gamma series or the finite upper-gamma recurrence
for integer and half-integer shapes retains the log prefactor. The upper
recurrence uses the exact exponential or scaled complementary-error-function
base case. Positive geometric bounds limit the omitted terms to machine
epsilon relative to the sum; failure to converge within 100,000 terms raises
an arithmetic error. The independent
one-degree-of-freedom identity for a sample $(0,d)$ is
$P(X^2\le d^2/2)=\operatorname{erf}(|d|/2)$; regression checks retain
the positive p-value for $d=10^{-200}$ even though $X^2$ is displayed as zero.

## Two-sample F test

For two independent normal samples,

$$
F=\frac{S_x^2}{S_y^2}\sim F_{n_x-1,n_y-1}
$$

under equality of variances. Both sample variances must be positive. The ratio
is formed by subtracting log variances, so common units cancel before
exponentiation. Confidence intervals invert the F pivot.

Validation includes literal F tails and quantiles, translation and common-scale
invariance, and the directional identity obtained by swapping samples:

$$
F(y,x)=1/F(x,y),\qquad
p_{\text{less}}(y,x)=p_{\text{greater}}(x,y).
$$

When group scales span the float64 exponent range, each positive spread keeps
an independently evaluated variance log if common normalization would place
any nonzero anchored entry in the subnormal range. This prevents partial
precision loss before the variance is logged, including when one group's
large common location far exceeds its spread. Tests compare a group near
$10^{300}$, separated by one floating-point spacing, against its explicitly
recentered version and a second group with spread $10^{-22}$. The F tail
agrees with its exact squared-Cauchy identity and Bartlett's statistic is
unchanged by the translation. Tests also cover a ratio below the smallest representable positive
float and its reciprocal above the largest one; the reported boundary
statistics remain zero and infinity with the correct tail probability.

The lower and upper tails use the complementary beta arguments
$x=\operatorname{logistic}(\log F+\log \nu_1-\log\nu_2)$ and $1-x$,
retaining the smaller argument in log space and swapping the beta shapes when
necessary. This avoids rounding an argument close to one before calculating
its complementary tail. For an extremely small argument,
$\log I_x(a,b)=a\log x-\log a-\log B(a,b)$ is used only when
$x\max(1,b)<\epsilon/2$. The omitted factor is a weighted mean of
$(1-t)^{b-1}$ over $0\le t\le x$, whose log magnitude is then below
machine epsilon. Ordinary arguments use SciPy's regularized incomplete beta
function and its directly evaluated complement, retaining the smaller tail
and calculating its complement with `log1p`. Subnormal probabilities
are reevaluated with the positive hypergeometric series in DLMF 8.17.8,
retaining the log prefactor. A geometric bound on all remaining terms controls
truncation; failure to converge within 100,000 terms raises an arithmetic
error. For arguments above 0.9, where that series converges slowly, a
modified-Lentz evaluation of the DLMF 8.17.22 continued fraction retains the
log prefactor and the small complementary argument. Three successive full
convergents must stabilize within four machine epsilons, with the same
iteration cap and an explicit error on nonconvergence. These fallbacks cover
subnormal tails even for arguments near $1/2$ or 1, where an endpoint
approximation would be inappropriate. If one beta shape is an integer
$m\le32$ and the other is at least 10,000, the normalizer uses the exact
short recurrence $\log B(a,m)=\log\Gamma(m)-\sum_{j=0}^{m-1}\log(a+j)$.
This avoids cancellation from subtracting large log-gamma values. A separate
finite beta-sum identity checks shape pairs $(2{,}000{,}000,3)$ and
$(10{,}000{,}000,25)$, including reciprocal F tails. Half-integer shapes up
32 use the companion base
$\log B(a,1/2)=\tfrac12(\log\pi-\log a)+1/(8a)-1/(192a^3)$
and the exact short beta recurrence. The next term is below
$1.6\times10^{-23}$ at the cutoff $a=10{,}000$; an exact central-binomial
coefficient identity independently checks that boundary. See the
[gamma-ratio expansion](https://dlmf.nist.gov/5.11) and
[beta integral and identities](https://dlmf.nist.gov/8.17).

Regression tests use the independent squared-Cauchy identity for $F_{1,1}$,
covering ratios whose displayed statistic overflows and reciprocal ratios
that underflow. They also check unequal degrees of freedom against SciPy,
continuity across normal/subnormal pivot ranges, subnormal p-values, exact
Poisson-sum identities for integer gamma shapes, and the $I_x(a,1)=x^a$
identity with $a=1075$ and $x$ near $1/2$.
Two-sided probabilities double the smaller tail in log space before rounding;
this preserves a representable doubled tail even when the single tail rounds
to zero. Tail probabilities below half the smallest positive float still
necessarily round to zero.

## Bartlett test

For $k$ normal populations, the implementation uses the classical corrected
log-variance statistic and the $\chi^2_{k-1}$ approximation. The pooled
variance is evaluated with log-sum-exp, and every group must have positive
sample variance.

The fixed fixture agrees with `scipy.stats.bartlett`. Tests also cover group and
row reordering, translation, common scaling, invalid groups, and degenerate
variances.

## Levene and Brown--Forsythe tests

Both procedures apply a one-way F statistic to absolute within-group
deviations. Levene centers at each sample mean; Brown--Forsythe centers at each
sample median. Their reported $F_{k-1,N-k}$ laws are finite-sample
approximations and do not require the same normality assumption as Bartlett's
derivation.

The fixed fixtures agree with `scipy.stats.levene(center="mean")` and
`scipy.stats.levene(center="median")`. Tests cover group and row order,
translation, scales from approximately $10^{-100}$ to $10^{100}$, and both
zero/zero and positive/zero ANOVA denominator limits. Within-group anchors are
removed before a common deviation scale is chosen, preventing large group
locations from contaminating the absolute deviations.

## Explicit null-size gate

The following Gaussian-null audit used three independent $N(0,1)$ groups of
100 observations and 20,000 datasets for each seed. Every entry satisfies the project
release tolerance simultaneously at nominal levels 0.01, 0.05, and 0.10.
For every procedure and seed, a fresh `numpy.random.default_rng(seed)` generated
the three groups in group order and the corresponding public function was
called once per replication. Rejection meant `result.pvalue < alpha`.

| Seed | Procedure | 0.01 count/rate | 0.05 count/rate | 0.10 count/rate |
|---:|---|---:|---:|---:|
| 20260823 | Bartlett | 196 (0.00980) | 1,028 (0.05140) | 2,070 (0.10350) |
| 20260824 | Bartlett | 182 (0.00910) | 957 (0.04785) | 1,947 (0.09735) |
| 20260823 | Levene (mean) | 198 (0.00990) | 1,060 (0.05300) | 2,068 (0.10340) |
| 20260824 | Levene (mean) | 194 (0.00970) | 1,019 (0.05095) | 2,045 (0.10225) |
| 20260823 | Brown--Forsythe (median) | 177 (0.00885) | 976 (0.04880) | 1,918 (0.09590) |
| 20260824 | Brown--Forsythe (median) | 169 (0.00845) | 938 (0.04690) | 1,916 (0.09580) |

This validates the displayed approximations in that advertised normal design;
it is not a claim of an exact F or chi-square law for arbitrary finite
samples or arbitrary nonnormal populations.

## Targeted alternative-power audit

Every public routine was also exercised on 2,000 strong normal alternatives at
nominal 0.05. A fresh `numpy.random.default_rng(20260829)` was reset for each
row, observations were generated in the displayed group order, and the public
function was called once per replication.

| Procedure | Alternative and design | Rejections/2,000 | Rate |
|---|---|---:|---:|
| `chisquare_1samp` | $N(0,4)$, $n=50$, tested against variance 1 | 2,000 | 1.0000 |
| `f_2samp` | $N(0,4)$ versus $N(0,1)$, $n=m=50$ | 1,993 | 0.9965 |
| `bartlett` | standard deviations $(1,1.5,2)$, three groups of 100 | 2,000 | 1.0000 |
| `levene` | standard deviations $(1,1.5,2)$, three groups of 100 | 2,000 | 1.0000 |
| `brown_forsythe` | standard deviations $(1,1.5,2)$, three groups of 100 | 2,000 | 1.0000 |

These targeted results verify direction and practical sensitivity for an
explicit alternative; they are not a minimum-power guarantee over all unequal
variances.

## Legacy mapping

| pySHT | SHT 0.1.9 |
|---|---|
| `chisquare_1samp` | `var1.chisq` |
| `f_2samp` | `var2.F` |
| `bartlett` | `vark.1937Bartlett` |
| `levene` | `vark.1960Levene` |
| `brown_forsythe` | `vark.1974BF` |

The Python names describe the statistical operation rather than encoding a
publication year. Exact assumptions and calibration are included in every
result display.

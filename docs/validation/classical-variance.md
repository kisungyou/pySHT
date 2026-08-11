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
an independently evaluated variance log if common normalization would
underflow it. Tests cover a ratio below the smallest representable positive
float and its reciprocal above the largest one; the reported boundary
statistics remain zero and infinity with the correct tail probability.

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

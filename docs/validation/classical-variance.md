# Classical variance tests: formula and validation ledger

This ledger covers `pysht.variance`. The implementations use normalized or
log-domain arithmetic and SciPy distribution functions; output from legacy SHT
is not treated as the correctness oracle.

## One-sample chi-square test

For independent normal observations and null variance \(\sigma_0^2>0\),

\[
X^2=\frac{(n-1)S^2}{\sigma_0^2}\sim\chi^2_{n-1}.
\]

Lower, upper, and doubled minimum-tail p-values are supported. Confidence
intervals invert the same chi-square pivot. Sample variance is computed in the
log domain after scaling, avoiding direct squaring of very large observations.
A constant sample is represented by the mathematically limiting statistic and
interval zero; under a positive normal-theory null its two-sided p-value is
zero.

Validation includes a literal fixed-data formula, pivot-inversion checks for
two- and one-sided intervals, joint data/null rescaling through approximately
\(10^{\pm100}\), strict control validation, and `htest`-style rendering.

## Two-sample F test

For two independent normal samples,

\[
F=\frac{S_x^2}{S_y^2}\sim F_{n_x-1,n_y-1}
\]

under equality of variances. Both sample variances must be positive. The ratio
is formed by subtracting log variances, so common units cancel before
exponentiation. Confidence intervals invert the F pivot.

Validation includes literal F tails and quantiles, translation and common-scale
invariance, and the directional identity obtained by swapping samples:

\[
F(y,x)=1/F(x,y),\qquad
p_{\text{less}}(y,x)=p_{\text{greater}}(x,y).
\]

## Bartlett test

For \(k\) normal populations, the implementation uses the classical corrected
log-variance statistic and the \(\chi^2_{k-1}\) approximation. The pooled
variance is evaluated with log-sum-exp, and every group must have positive
sample variance.

The fixed fixture agrees with `scipy.stats.bartlett`. Tests also cover group and
row reordering, translation, common scaling, invalid groups, and degenerate
variances.

## Levene and Brown--Forsythe tests

Both procedures apply a one-way F statistic to absolute within-group
deviations. Levene centers at each sample mean; Brown--Forsythe centers at each
sample median. Their reported \(F_{k-1,N-k}\) laws are finite-sample
approximations and do not require the same normality assumption as Bartlett's
derivation.

The fixed fixtures agree with `scipy.stats.levene(center="mean")` and
`scipy.stats.levene(center="median")`. Tests cover group and row order,
translation, scales from approximately \(10^{-100}\) to \(10^{100}\), and both
zero/zero and positive/zero ANOVA denominator limits.

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

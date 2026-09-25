# Lee–You–Lin maximum pairwise Bayes factor for means

**Status:** Equation (4), log-domain, translation, scaling, group-exchange,
large-magnitude, and degenerate-feature gates pass.

For each feature, `maximum_pairwise_bayes_factor_2samp` uses the ordinary centered maximum-likelihood
variances from Equation (4). The published model is Gaussian with a common
covariance matrix under the two samples. At `a0=b0=0`, its component log
Bayes factor is

`0.5 log(gamma / (1 + gamma)) + n / 2 * log(RSS_pooled / RSS_within)`.

The maximum component log Bayes factor is returned without exponentiation and
without a fabricated p-value or universal cutoff. All component values remain
available on the immutable result.

By default, `alpha=2.01` and
`gamma=max(n_x+n_y, p)**(-alpha)`, matching the paper's numerical choice and
dimension-dependent rate. Supplying `gamma` explicitly overrides that rule
and is recorded in diagnostics.

## Audit finding

The legacy C++ helper weakened centering by dividing the squared total by
`1 + gamma`. That made the result change after adding the same constant to
both groups and does not match Equation (4). pySHT uses exact centered residual
sums; `gamma` appears only in the prior penalty. A shared inverse-gamma
`a0,b0` extension is retained for compatibility and reduces algebraically to
Equation (4) at zero. It is explicitly labeled as a pySHT extension in the
result diagnostics and documentation, not attributed to Equation (4).
Scaling `b0` by the square of the data scale preserves the result.

The two samples use a shared deterministic feature-wise anchor before
working-scale selection. This preserves group exchange and translation
invariance at large common locations without changing the log Bayes factors.

## Targeted alternative-evidence gate

Because `maximum_pairwise_bayes_factor_2samp` returns a Bayes factor rather than a p-value, this gate
does not invent a rejection cutoff. Seed 2026090322 initializes a
`SeedSequence`; child 0 drives one persistent PCG64 data stream and child 1 is
reserved as an independent auxiliary stream. In each of 1,000 replications,
child 0 draws `x` and then `y` as independent 30-by-60 standard-normal
matrices. The matched null value is
`mean.maximum_pairwise_bayes_factor_2samp(x, y).statistic`; the alternative
value is `mean.maximum_pairwise_bayes_factor_2samp(x, y + shift).statistic`, where `shift`
equals 1.0 in the first three coordinates and zero elsewhere. The same
realized `(x,y)` is used in each matched pair, and the default published
`gamma` rate is retained.

| Replications | Null median maximum log BF | Alternative median maximum log BF | Alternative exceeds matched null |
|---:|---:|---:|---:|
| 1,000 | -0.817807 | 5.961739 | 992 / 1,000 = 0.992 |

The paired ordering and median displacement establish directional evidence
under this sparse strong alternative; neither number is proposed as a
universal evidence threshold. The audit is reproduced by
`python -m tools.mean_power_audits mean.maximum_pairwise_bayes_factor_2samp`
under Python 3.12.13,
NumPy 2.5.1, and SciPy 1.18.0.

Primary reference: K. Lee, K. You, and L. Lin, *Bayesian Optimal Two-Sample
Tests for High-Dimensional Gaussian Populations*, Bayesian Analysis 19
(2024), 869–893, doi:10.1214/23-BA1373.

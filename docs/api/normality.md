# [8] Tests for Normality

`pysht.normality` provides five univariate and two multivariate
goodness-of-fit procedures. Every function tests a composite normal null;
location, scale, and (for multivariate samples) covariance are not specified.

| Procedure | Function | Authoritative calibration |
|---|---|---|
| Shapiro--Wilk | `shapiro_wilk` | Royston approximation |
| Shapiro--Francia | `shapiro_francia` | Royston approximation |
| Jarque--Bera | `jarque_bera` | Monte Carlo normal null |
| Adjusted Jarque--Bera | `adjusted_jarque_bera` | Monte Carlo normal null |
| Robust Jarque--Bera | `robust_jarque_bera` | Monte Carlo normal null |
| Henze--Zirkler | `henze_zirkler` | Monte Carlo normal null with refitting |
| Energy | `energy` | Monte Carlo normal null with refitting |

The three moment tests default to `calibration="monte-carlo"`. Pass an integer
to `rng` for a replayable result. Their `calibration="asymptotic"` option is an
explicit chi-square approximation, not a finite-sample guarantee.

Constant samples are outside the domain of every normality statistic. The
Shapiro procedures also enforce the sample-size ranges over which their
p-value approximations are supported.

The multivariate functions require a two-dimensional sample with more rows
than columns and positive-definite sample covariance. They re-estimate the
mean and covariance in every simulated null sample. Thus their Monte Carlo
calibration tests the intended composite null rather than a known-parameter
standard-normal null. Henze--Zirkler uses maximum-likelihood covariance
scaling; the energy procedure uses the ordinary sample covariance, matching
the published procedure and its authors' reference implementation.

See the [normality validation ledger](../validation/normality.md) for formulas,
null-size audits, Monte Carlo semantics, and deliberate corrections to SHT.
The native multivariate additions have separate ledgers for
[Henze--Zirkler](../validation/henze-zirkler.md) and
[energy normality](../validation/energy-normality.md).

## Functions

```{eval-rst}
.. automodule:: pysht.normality
   :members:
   :member-order: bysource
```

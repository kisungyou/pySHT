# [8] Tests for Normality

`pysht.normality` provides five univariate goodness-of-fit procedures. Every
function tests the composite null that the observations follow some normal
distribution; location and scale are not specified under the null.

| Procedure | Function | Authoritative calibration |
|---|---|---|
| Shapiro--Wilk | `shapiro_wilk` | Royston approximation |
| Shapiro--Francia | `shapiro_francia` | Royston approximation |
| Jarque--Bera | `jarque_bera` | Monte Carlo normal null |
| Adjusted Jarque--Bera | `adjusted_jarque_bera` | Monte Carlo normal null |
| Robust Jarque--Bera | `robust_jarque_bera` | Monte Carlo normal null |

The three moment tests default to `calibration="monte-carlo"`. Pass an integer
to `rng` for a replayable result. Their `calibration="asymptotic"` option is an
explicit chi-square approximation, not a finite-sample guarantee.

Constant samples are outside the domain of every normality statistic. The
Shapiro procedures also enforce the sample-size ranges over which their
p-value approximations are supported.

See the [normality validation ledger](../validation/normality.md) for formulas,
null-size audits, Monte Carlo semantics, and deliberate corrections to SHT.

## Functions

```{eval-rst}
.. automodule:: pysht.normality
   :members:
   :member-order: bysource
```

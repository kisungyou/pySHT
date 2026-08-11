# [9] Tests for Rectangular Uniformity

`pysht.uniformity` tests whether multivariate observations are uniform on a
declared hyperrectangle. Bounds default to the unit hypercube and are treated
as part of the null hypothesis, not estimated from the data.

| Question | Function |
|---|---|
| Do squared interpoint distances have their uniform-null moments? | `ym_interpoint` |
| Do coordinatewise normal quantiles have mean zero? | `ym_quantile` |

`ym_interpoint` provides the Yang--Modarres `q1`, `q2`, and `q3` statistics.
Parametric Monte Carlo calibration is the finite-sample default. The optional
asymptotic calibration uses chi-square limits for `q1` and `q2`; for `q3`, it
uses the correlated-normal-square limit implied by the exact covariance of
the two signed components. It does not reproduce the paper's invalid
independence approximation. Integer `rng` seeds reproduce the complete Monte
Carlo calibration.

`ym_quantile` has an exact chi-square null after its transformation. It requires
every observation to lie strictly inside the declared bounds: endpoints map to
infinite normal quantiles and are rejected rather than clipped.

See the [rectangular-uniformity validation
ledger](../validation/uniformity.md) for the formulas, boundary policy, and
calibration audit.

## Functions

```{eval-rst}
.. automodule:: pysht.uniformity
   :members:
   :member-order: bysource
```

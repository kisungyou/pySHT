# [9] Tests for Rectangular Uniformity

`pysht.uniformity` tests whether multivariate observations are uniform on a
declared hyperrectangle. Bounds default to the unit hypercube and are treated
as part of the null hypothesis, not estimated from the data.

| Question | Function |
|---|---|
| Is the complete nearest-neighbor geometry compatible with uniformity? | `ehy` |
| Do squared interpoint distances have their uniform-null moments? | `ym_interpoint` |
| Do coordinatewise normal quantiles have mean zero? | `ym_quantile` |

`ehy` implements the Ebner--Henze--Yukich statistic. Both `alpha` and
`n_neighbors` are required because they affect power and must be fixed before
testing. Here `n_neighbors=J` means that the score sums the contributions from
each of the first $J$ neighbors, not only the $J$th distance. The supported
domain is $\alpha>0$, $\alpha\ne1$, and $1\le J<n$. It rejects in the lower
tail for $0<\alpha<1$ and in the upper tail for $\alpha>1$. Unlike the older
Yang--Modarres interpoint routine, EHY is also defined for one-dimensional
rectangles.

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
See the method-specific [EHY rectangular ledger](../validation/ehy-rectangular.md)
for its primary equations and release gates.

## Functions

```{eval-rst}
.. automodule:: pysht.uniformity
   :members:
   :member-order: bysource
```

# [7] Tests for Equality of Distributions

`pysht.equaldist` contains tests whose null hypothesis concerns the complete
data-generating distribution rather than a single parameter such as a mean or
variance.

The current public function is `bg_2samp`. It supports exact or
corrected Monte Carlo permutation calibration; the unverified legacy
asymptotic calibration is not exposed.

See the [validation record](../validation/biswas-ghosh-2014.md) for the formula,
exchangeability assumptions, tie handling, numerical normalization, and
independent exact-enumeration checks.

## Functions

```{eval-rst}
.. automodule:: pysht.equaldist
   :members:
   :member-order: bysource
```

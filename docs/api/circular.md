# [12] Circular Data

`pysht.circular` accepts angles in arbitrary units through `period`; values are
reduced modulo that period. The default is $2\pi$ radians.

| Scientific question | Function | Calibration |
|---|---|---|
| Is the first trigonometric moment nonzero? | `rayleigh` | Monte Carlo circular-uniform null |
| Is there any departure detectable by Watson $U^2$? | `watson` | Monte Carlo circular-uniform null |
| Is there an omnibus, possibly multimodal departure? | `hermans_rasson` | Monte Carlo circular-uniform null |
| Are two or more continuous circular distributions equal? | `mardia_watson_wheeler_ksamp` | exact or Monte Carlo label permutation |

Rayleigh is deliberately described as a targeted first-harmonic test: a
distribution can be nonuniform while its first trigonometric moment is zero.
Watson and modified Hermans--Rasson are omnibus tests. Rayleigh reports the
mean resultant length and, when it is numerically defined, the mean direction
in the same units as the input.

All one-sample routines use corrected Monte Carlo p-values and default to
9,999 draws. The Mardia--Watson--Wheeler routine conditions on the pooled
directions. Automatic `calibration="permutation"` is exact when the complete
ordered allocation orbit fits the resampling budget. Because this API follows
the continuous-rank statistic, pooled ties are rejected rather than silently
receiving an undocumented grouped-data correction.

See the method ledgers for [Rayleigh](../validation/circular-rayleigh.md),
[Watson](../validation/circular-watson.md), [modified
Hermans--Rasson](../validation/circular-hermans-rasson.md), and
[Mardia--Watson--Wheeler](../validation/circular-mardia-watson-wheeler.md).

## Functions

```{eval-rst}
.. automodule:: pysht.circular
   :members:
   :member-order: bysource
```

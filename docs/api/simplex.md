# [10] Simplex Methods

`pysht.simplex` provides two tests of uniformity with respect to simplex volume
and an equality-of-distributions test for compositional samples.

| Question | Function | Calibration |
|---|---|---|
| Is a sample uniform against a Dirichlet alternative? | `uniformity` | Wilks approximation |
| Is nearest-neighbor geometry compatible with simplex uniformity? | `ehy_uniformity` | Monte Carlo simplex null |
| Do two or more compositional distributions agree? | `alpha_energy_ksamp` | exact or Monte Carlo label permutation |

`ehy_uniformity` requires prespecified `alpha` and `n_neighbors`. It sums the
first $J$ nearest-neighbor contributions on the flat $(D-1)$-dimensional
simplex and uses the simplex's Hausdorff-volume density. Its parameter domain
is $\alpha>0$, $\alpha\ne1$, and $1\le J<n$; zeros on the simplex boundary are
allowed.

`alpha_energy_ksamp` requires one prespecified `alpha` in $[-1,1]$. At
`alpha=0` it uses the Aitchison metric and all components must be strictly
positive. Structural zeros are accepted only for `alpha>0`. Automatic
`calibration="permutation"` enumerates the complete fixed-size allocation
orbit when it fits `n_resamples`, and otherwise uses Monte Carlo permutation.
Repeated compositions are supported. For unresolved symmetries beyond the
bounded canonical-orbit calculation, reordering components or observations
can change a fixed seed's Monte Carlo count; the statistic and the uniform
permutation distribution remain the same.
No data-selected alpha is offered because that would require joint
calibration.

## Dirichlet likelihood-ratio alternatives

The `model` selector controls the alternative:

| `model` | Alternative model | Wilks degrees of freedom |
|---|---|---:|
| `"symmetric"` | one common positive concentration | 1 |
| `"general"` | one positive concentration per component | number of components |

Inputs must lie in the strict simplex interior. Boundary zeros are not
perturbed. The optimizer controls `tolerance` and `max_iter` are explicit and
fail loudly when a finite maximum cannot be established.

See the [simplex-uniformity validation
ledger](../validation/simplex-uniformity.md) for the likelihood, independent
optimization oracle, null calibration, and MLE boundary cases.
The native additions have separate ledgers for [EHY simplex
uniformity](../validation/ehy-simplex.md) and [alpha-energy compositional
equality](../validation/alpha-energy-simplex.md).

## Functions

```{eval-rst}
.. automodule:: pysht.simplex
   :members:
   :member-order: bysource
```

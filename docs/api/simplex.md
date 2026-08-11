# [10] Simplex Uniformity

`pysht.simplex.uniformity` tests whether compositional observations are uniform
with respect to volume on a probability simplex. The null is the Dirichlet
model with every concentration parameter equal to one.

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

## Functions

```{eval-rst}
.. automodule:: pysht.simplex
   :members:
   :member-order: bysource
```

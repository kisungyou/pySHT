# Mardia--Watson--Wheeler circular $k$-sample test

**Public verdict:** validated and exposed as
`pysht.circular.mardia_watson_wheeler_ksamp`. This is a pySHT-native method
with no SHT 0.1.9 crosswalk entry.

## Conditional-rank statistic

Pool $N$ distinct circular observations, assign ranks $R_{gi}$, and define

$$
C_g=\sum_{i=1}^{n_g}\cos(2\pi R_{gi}/N),
\qquad
S_g=\sum_{i=1}^{n_g}\sin(2\pi R_{gi}/N).
$$

The statistic

$$
W=2\sum_{g=1}^k\frac{C_g^2+S_g^2}{n_g}
$$

tests equality of the complete continuous circular distributions; large
values reject. The formulation and naming follow [Mardia
(1972)](https://doi.org/10.1111/j.2517-6161.1972.tb00891.x). Literal pooled-rank
and allocation loops independently verify the statistic and tail.

The API intentionally targets continuous circular data. Exact pooled ties make
the rank scores nonunique and are rejected; pySHT does not silently choose a
grouped-data correction. Inputs are otherwise invariant to row/group order,
common rotation and reflection, wrapping, and period-unit conversion.

## Exact and Monte Carlo calibration

Conditioning on pooled directions, group sizes remain fixed. There are
$B=N!/\prod_g n_g!$ ordered allocations. Exact mode enumerates every one and
uses $p=b/B$. Automatic `calibration="permutation"` uses exact mode when the
orbit fits `n_resamples`; otherwise Monte Carlo permutations use
$(b+1)/(B+1)$ and report MCSE and a binomial interval. Exact mode validates an
`rng` argument but consumes no random values. Canonical group and angle order
make a Monte Carlo seed replay under scientifically irrelevant reorderings.

## Complexity

Pooling and ranking costs $O(N\log N)$ time and $O(N)$ storage. With $K$
groups, each allocation statistic costs $O(KN)$ in the current literal score
reduction. Thus exact enumeration costs $O(AKN)$ for
$A=N!/\prod_g n_g!$ allocations and Monte Carlo costs $O(BKN)$; both stream
allocations and retain $O(N+K)$ working storage.

## Release evidence

The calibration gate is an exhaustive conditional orbit, not 20,000
independent circular datasets. For two groups of size eight, all
$\binom{16}{8}=12,870$ allocations were enumerated. The proportions of the
orbit whose exact p-values fell below 0.01, 0.05, and 0.10 were respectively
128/12,870 (0.00995), 640/12,870 (0.04973), and 1,248/12,870 (0.09697).
Discrete conditional tests need not attain nominal levels exactly; all three
pass the project tolerance.

For a power check, seed 20260929 generated 2,000 two-group datasets of size
(5,5), one group von Mises around zero and the other around $\pi$, each with
concentration 8. Every dataset used its complete 252-allocation exact null;
2,000/2,000 rejected at 0.05. This is a named separation alternative only.

# Simplex uniformity: formula and validation ledger

This ledger covers `pysht.simplex.uniformity`. The routine tests the fixed null

$$
H_0:\alpha=(1,\ldots,1),
$$

which is the uniform distribution with respect to volume on the
$(k-1)$-dimensional probability simplex.

The native additions have separate ledgers for [EHY simplex
uniformity](ehy-simplex.md) and [alpha-energy compositional
equality](alpha-energy-simplex.md).

## Likelihood-ratio statistics

For $n$ strict-interior compositions and positive Dirichlet parameter
$\alpha$, the log likelihood is

$$
\ell(\alpha)=n\left[\log\Gamma(\alpha_0)-
\sum_{j=1}^k\log\Gamma(\alpha_j)\right]
+\sum_{j=1}^k(\alpha_j-1)\sum_{i=1}^n\log x_{ij},
\qquad \alpha_0=\sum_j\alpha_j.
$$

The statistic is $LR=2\{\ell(\widehat\alpha)-\ell(1)\}$. The
`model="symmetric"` alternative sets every parameter to one fitted scalar and
uses the Wilks $\chi^2_1$ approximation. The `model="general"` alternative
fits all $k$ positive parameters and uses $\chi^2_k$.

## Optimization and domain policy

Every component must be strictly positive. Row sums may differ from one only
by the documented accumulation tolerance
$64k\epsilon_{64}$; accepted rows are normalized before logarithms are
taken. A component stored as exactly one is also rejected: with every other
component positive, such a row lies outside the strict real-valued simplex
even if floating-point summation rounds its total back to one. Unlike SHT
0.1.9, zeros are never replaced by $10^{-10}$, and
non-compositional rows are never silently projected onto the simplex.

The symmetric score is monotone and is solved with a bracketed Brent root. The
general log likelihood is concave in $\alpha$; its analytic score and Hessian
are first solved by damped Newton steps with an Armijo line search. If extreme
asymmetry makes that coordinate system ill-conditioned, a bounded trust-region
solve of the same score equations in $\log\alpha$ is used. The fallback has an
analytic Jacobian and is accepted only after the original alpha-coordinate
score meets the requested tolerance, up to a scale-aware floating-point floor.
`tolerance` and `max_iter` are public, deterministic controls. Failure to
bracket, preserve positivity, or verify the score raises an exception rather
than returning a partially optimized test.

The score, trigamma Hessian, and Newton system follow equations (10)--(18) of
[Minka's *Estimating a Dirichlet
distribution*](https://tminka.github.io/papers/dirichlet/). The log-coordinate
fallback changes only the numerical parameterization, not the likelihood or
estimating equations.

MLE boundaries are model-specific:

- identical rows make the general-model concentration unbounded;
- the symmetric concentration is unbounded only when every observation is the
  simplex barycenter;
- identical noncentral rows retain a finite symmetric optimum.

## Independent oracles

The symmetric fixed fixture is independently optimized in the scalar
log-concentration coordinate. The general fixture independently solves all
score equations in unconstrained log-parameter coordinates with SciPy's hybrid
root solver. These oracles use neither the production Newton iteration nor the
legacy R initializer. Resulting likelihood-ratio statistics agree within the
conditioning-appropriate $10^{-8}$ to $10^{-9}$ relative tolerances.

Tests additionally cover component permutation, nesting of the symmetric model
inside the general model, strict boundaries, roundoff-sized row-sum errors,
unbounded MLE cases, optimizer failure, immutable result rendering, and a
concentrated alternative. A two-observation, ten-component fixture spanning
about 39 orders of magnitude exercises the log-score fallback and agrees with
an independent Levenberg--Marquardt root of all score equations.

## Null calibration

Two independent 20,000-sample audits used $n=50$, $k=3$, and
$\operatorname{Dirichlet}(1,1,1)$ observations. There were no optimizer failures.
For each model and seed, a fresh `numpy.random.default_rng(seed)` generated
datasets in replication order and the public `uniformity` function was called.
Rejection meant `result.pvalue < alpha`.

| seed | model | 0.01 count/rate | 0.05 count/rate | 0.10 count/rate |
|---:|---|---:|---:|---:|
| 20260813 | symmetric | 195 (0.00975) | 1,011 (0.05055) | 2,004 (0.10020) |
| 20260813 | general | 206 (0.01030) | 1,027 (0.05135) | 2,031 (0.10155) |
| 20260814 | symmetric | 211 (0.01055) | 1,034 (0.05170) | 2,012 (0.10060) |
| 20260814 | general | 226 (0.01130) | 1,045 (0.05225) | 2,078 (0.10390) |

Every entry passes the project release tolerance, so Wilks calibration is the
validated default in this advertised regime. This audit does not imply an
exact finite-sample chi-square law or validate every possible $(n,k)$ pair.

## Targeted alternative-power audit

For a $\operatorname{Dirichlet}(20,20,20)$ alternative with the same
$(n,k)=(50,3)$, a fresh `numpy.random.default_rng(20260829)` was reset for
each model, generated 2,000 datasets in order, and called the public
`uniformity` function once per replication at nominal 0.05.

| Model | Rejections/2,000 | Rate |
|---|---:|---:|
| `"symmetric"` | 2,000 | 1.0000 |
| `"general"` | 2,000 | 1.0000 |

The 99% Wilson lower bound for either rate is 0.9967. This is a targeted
concentration-alternative check, not a uniform power guarantee over the
Dirichlet parameter space.

## Legacy mapping

| pySHT | SHT 0.1.9 |
|---|---|
| `uniformity(model="symmetric")` | `simplex.uniform(..., "LRTsym")` |
| `uniformity(model="general")` | `simplex.uniform(..., "LRT")` |

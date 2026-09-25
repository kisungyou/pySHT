# Joint mean and variance: formula and validation ledger

This ledger covers `pysht.mean_variance`. All six functions assume
independent observations from normal populations. SHT 0.1.9 was used only as
a migration audit; published formulas and independently evaluated probability
laws are the correctness oracles.

## Formula-source ledger

| pySHT function | Primary formula basis | Independent oracle | Status |
|---|---|---|---|
| `lrt_1samp` | Arnold and Shavelle (1998), pp. 133--140; normal likelihood ratio | Literal maximized log likelihood | Public; asymptotic calibration |
| `pn_2samp` | Pearson and Neyman (1930; reprinted 1967, pp. 99--115); exact first two moments of the likelihood ratio | Log-gamma moment calculation and exact ZXC comparison | Public; lower beta tail corrected |
| `pl_2samp` | Perng and Littell (1976), pp. 968--971 | Independent SciPy pooled-t and F p-values followed by Fisher's identity | Public |
| `muirhead_2samp` | Muirhead (1982), Theorem 10.8.4, p. 370, specialized to two univariate populations | Literal second-order survival expansion and null simulation | Public; approximation limits disclosed |
| `exact_lrt_2samp` | Zhang, Xu, and Chen (2012), pp. 180--184; exact Dirichlet probability | High-accuracy conditional-beta quadrature and direct Dirichlet simulation | Public; exact |
| `lrt_2samp` | Wilks likelihood-ratio limit applied to the same normal likelihood | Literal maximized log likelihood | Public; asymptotic calibration |

## Shared likelihood ratio

For samples of sizes $n$ and $m$, define

$$
A=\sum_i(X_i-\bar X)^2,\qquad
B=\sum_j(Y_j-\bar Y)^2,
$$

and

$$
C=A+B+\frac{nm}{n+m}(\bar X-\bar Y)^2.
$$

The likelihood ratio for equality of both normal-population parameters is

$$
\Lambda=
\frac{(A/n)^{n/2}(B/m)^{m/2}}
     {(C/(n+m))^{(n+m)/2}}.
$$

pySHT removes a shared location before numerical scaling. When both within-
group sums of squares fit comfortably on that common scale, their ratio is
formed directly. When their magnitudes differ by hundreds of orders, each
group instead retains its own log sum of squares and the pooled null sum is
formed with `logsumexp`. This makes every two-sample statistic invariant to
group exchange, common translation, and a change of units without erasing the
smaller group's positive variance. A fixture with one group near $10^{-87}$
and the other near $10^{87}$ retains the finite value
$\log\Lambda=-799.9133180008080$ even though $\Lambda$ itself underflows.

## One-sample Arnold--Shavelle likelihood ratio

For null values $(\mu_0,\sigma_0^2)$ and the maximum-likelihood variance
$\widehat\sigma^2=n^{-1}\sum_i(X_i-\bar X)^2$,

$$
-2\log\Lambda=n\left[
\log\frac{\sigma_0^2}{\widehat\sigma^2}
+\frac{\widehat\sigma^2}{\sigma_0^2}
+\frac{(\bar X-\mu_0)^2}{\sigma_0^2}-1
\right]\ \xrightarrow{d}\ \chi^2_2.
$$

The two R entries `mvar1.1998AS` and `mvar1.LRT` reduce
algebraically to this same statistic and therefore map to one Python function.
The fixed fixture is checked against the literal likelihood and after a joint
location-scale transformation near $10^{100}$.
An additional null-scale fixture verifies that a nonconstant sample near
$10^{-300}$ against a variance of $10^{300}$ retains the finite log-domain
likelihood-ratio statistic `8284.413760573307` and zero p-value. The earlier
standardize-then-square implementation returned infinity because both the
mean and variance ratios underflowed before their logarithmic contribution was
formed.

## Pearson--Neyman moment match

Let $a=E_0(\Lambda)$ and $b=\operatorname{Var}_0(\Lambda)$. The beta shapes
are

$$
p=-\frac{a}{b}(a^2-a+b),\qquad
q=\frac{a-1}{b}(a^2-a+b).
$$

The exact gamma-function expressions for $E_0(\Lambda)$ and
$E_0(\Lambda^2)$ are evaluated with `gammaln` and `expm1`.
Because small likelihood ratios contradict the null, the p-value is the lower
tail $I_{\lambda_{\rm obs}}(p,q)$. SHT 0.1.9 used the complementary tail and
also mislabeled this routine as Muirhead in its printed method.

On the fixed ledger fixture, the corrected beta p-value is
`0.6917901786656631`; the exact ZXC probability at the same likelihood
ratio is `0.6918190488406402`. A separate extreme-scale fixture has
$n=m=2$, for which the moment equations reduce to beta shapes $3/8$ and
$33/32$. It evaluates the beta lower tail from $\log\Lambda$ and retains the
representable probability `5.39880912287e-131` even though $\Lambda$
itself underflows to zero in float64. The 20,000-null release simulations were:

| $(n,m)$ | seed | 0.01 count/rate | 0.05 count/rate | 0.10 count/rate | gate |
|---:|---:|---:|---:|---:|---:|
| (3, 3) | 20260821 | 154 (0.00770) | 949 (0.04745) | 1,903 (0.09515) | Pass |
| (3, 3) | 20260822 | 170 (0.00850) | 938 (0.04690) | 1,907 (0.09535) | Pass |
| (7, 8) | 20260821 | 182 (0.00910) | 997 (0.04985) | 1,951 (0.09755) | Pass |
| (7, 8) | 20260822 | 190 (0.00950) | 962 (0.04810) | 1,955 (0.09775) | Pass |
| (20, 20) | 20260821 | 201 (0.01005) | 964 (0.04820) | 1,994 (0.09970) | Pass |
| (20, 20) | 20260822 | 185 (0.00925) | 970 (0.04850) | 1,902 (0.09510) | Pass |

## Perng--Littell combination

Under the joint normal null, the two-sided pooled-t p-value $p_t$ is
independent of the two-sided variance-ratio p-value $p_F$. Therefore

$$
-2(\log p_t+\log p_F)\sim\chi^2_4.
$$

The component p-values are recomputed independently in the tests and are
reported as calibration diagnostics, not parameter estimates. Both component
tails and Fisher's combination are evaluated in the log domain. On the public
heterogeneous-scale fixture $X=\{-10^{-87},10^{-87}\}$ and
$Y=\{-10^{87},10^{87}\}$, the combined p-value remains the representable
`5.110888469223042e-172` instead of collapsing to zero during an intermediate
F tail calculation.

## Muirhead second-order approximation

With $N=n+m$, set

$$
\rho=1-\frac{22}{24N}\left(\frac{N}{n}+\frac{N}{m}-1\right)
$$

and

$$
\gamma=\frac12\left[
\left(\frac{N}{n}\right)^2+\left(\frac{N}{m}\right)^2-1
\right]
-\frac{121}{96}\left(\frac{N}{n}+\frac{N}{m}-1\right)^2.
$$

For $W=-2\rho\log\Lambda$, pySHT evaluates the survival approximation

$$
\Pr_0(W\geq w)\approx
\bar G_2(w)+\frac{\gamma}{\rho^2N^2}
\{\bar G_6(w)-\bar G_2(w)\},
$$

where $\bar G_k$ is a chi-square survival function. SHT used the corresponding
CDF, which increases toward one under strong alternatives. The finite
expansion can leave $[0,1]$ in extreme tails, so pySHT truncates the result and
reports $\rho$ and the second-order coefficient as diagnostics.

This approximation is not advertised for very small samples. The complete
passing $(n,m)=(20,20)$ and failing $(7,8)$ audits appear in the explicit
tables below. Use `exact_lrt_2samp` for exact small-sample inference.

## Exact Zhang--Xu--Chen probability

Under the joint null, the normalized within-group and between-group sums of
squares have a Dirichlet distribution with parameters

$$
\left(\frac{n-1}{2},\frac{m-1}{2},\frac12\right).
$$

The exact p-value is the probability of a likelihood ratio no larger than the
observed ratio. pySHT conditions on one Dirichlet coordinate, reducing the
published two-dimensional region integral to beta tail masses plus one
one-dimensional conditional-beta integral. Roots are solved in logit
coordinates. Each half of the interior integrand is divided by its own log
maximum before quadrature, and the four probability pieces are recombined by
`logsumexp`. The smaller sample is assigned to the conditional coordinate,
which makes group symmetry exact in floating point and avoids a thin
unbalanced-sample boundary layer.

The fixed high-accuracy integral is `0.6918190488406402`; the legacy
500-by-500 nested rule gives `0.6921061`. A direct ten-million-draw
Dirichlet calculation gave `0.6917072` with Monte Carlo standard error
`0.000146`, independently corroborating the stable integral.

Three independent far-tail fixtures exercise behavior that ordinary
probability-space quadrature misses: $\log\Lambda=-50$ with $(n,m)=(2,2)$
gives `1.2412994009417087e-10`; the same log likelihood with
$(n,m)=(20,200)$ gives `2.2176768591905327e-21`; and
$\log\Lambda=-800$ with $(n,m)=(2,2)$ gives the still-representable
`2.4572424773752458e-172`. These values come from separate high-precision
log-coordinate integrations.

## Explicit asymptotic size gates

For every row below, 20,000 independent standard-normal null datasets were
generated for each of two seeds. The one-sample null was $N(0,1)$ with the
default null mean and variance; each two-sample row used two independent
$N(0,1)$ samples of the displayed sizes. For each method, scenario, and seed,
a fresh `numpy.random.default_rng(seed)` generated observations in sample and
replication order, the public function was called once, and rejection meant
`result.pvalue < alpha`. A row passes at all
$\alpha\in\{0.01,0.05,0.10\}$ when

$$
|\widehat\alpha-\alpha|
\leq
\max\left(0.005,\,
4\sqrt{\frac{\alpha(1-\alpha)}{20000}}\right).
$$

| Calibration | Scenario | Seed | 0.01 count/rate | 0.05 count/rate | 0.10 count/rate | Gate |
|---|---:|---:|---:|---:|---:|---|
| `lrt_1samp`, $\chi^2_2$ | $n=50$ | 20260819 | 204 (0.01020) | 1,029 (0.05145) | 2,078 (0.10390) | Pass |
| `lrt_1samp`, $\chi^2_2$ | $n=50$ | 20260820 | 221 (0.01105) | 1,022 (0.05110) | 2,036 (0.10180) | Pass |
| `lrt_2samp`, $\chi^2_2$ | $n=m=75$ | 20260819 | 184 (0.00920) | 1,016 (0.05080) | 2,051 (0.10255) | Pass |
| `lrt_2samp`, $\chi^2_2$ | $n=m=75$ | 20260820 | 204 (0.01020) | 1,002 (0.05010) | 2,082 (0.10410) | Pass |
| `muirhead_2samp` | $n=m=20$ | 20260819 | 210 (0.01050) | 1,030 (0.05150) | 2,065 (0.10325) | Pass |
| `muirhead_2samp` | $n=m=20$ | 20260820 | 206 (0.01030) | 1,063 (0.05315) | 2,050 (0.10250) | Pass |

Smaller scenarios are not claimed as release-validated. These replayable rows
show the failures rather than hiding them:

| Calibration | Scenario | Seed | 0.01 count/rate | 0.05 count/rate | 0.10 count/rate |
|---|---:|---:|---:|---:|---:|
| `lrt_1samp`, $\chi^2_2$ | $n=20$ | 20260819 | 251 (0.01255) | 1,192 (0.05960) | 2,268 (0.11340) |
| `lrt_1samp`, $\chi^2_2$ | $n=20$ | 20260820 | 244 (0.01220) | 1,121 (0.05605) | 2,192 (0.10960) |
| `lrt_2samp`, $\chi^2_2$ | $n=m=20$ | 20260819 | 263 (0.01315) | 1,203 (0.06015) | 2,358 (0.11790) |
| `lrt_2samp`, $\chi^2_2$ | $n=m=20$ | 20260820 | 260 (0.01300) | 1,241 (0.06205) | 2,352 (0.11760) |
| `muirhead_2samp` | $(n,m)=(7,8)$ | 20260819 | 474 (0.02370) | 1,441 (0.07205) | 2,556 (0.12780) |
| `muirhead_2samp` | $(n,m)=(7,8)$ | 20260820 | 435 (0.02175) | 1,374 (0.06870) | 2,492 (0.12460) |

The functions remain mathematically defined in these smaller designs, but
their reported p-values must be interpreted as rough asymptotic
approximations; `exact_lrt_2samp` supplies the exact two-sample alternative.

## Targeted alternative-power audit

At nominal 0.05, every public routine was run on 2,000 strong alternatives.
For `lrt_1samp`, observations were $N(0.75,1.5^2)$ and the tested null was the
default $(0,1)$ at $n=50$. Each two-sample row compared $N(0,1)$ with
$N(1,2^2)$; the sample size stays in the passing calibration regime of the
corresponding approximation. A fresh `numpy.random.default_rng(20260829)` was
reset for every row, generated $x$ then $y$ in each replication, and the
public function was called once.

| Procedure | Design | Rejections/2,000 | Rate |
|---|---:|---:|---:|
| `lrt_1samp` | $n=50$ | 2,000 | 1.0000 |
| `pn_2samp` | $n=m=20$ | 1,810 | 0.9050 |
| `pl_2samp` | $n=m=20$ | 1,818 | 0.9090 |
| `muirhead_2samp` | $n=m=20$ | 1,812 | 0.9060 |
| `exact_lrt_2samp` | $n=m=20$ | 1,810 | 0.9050 |
| `lrt_2samp` | $n=m=75$ | 2,000 | 1.0000 |

The near agreement among PN, Muirhead, and exact ZXC in the common design is
an additional directional check; no table entry asserts a uniform lower power
bound over all joint alternatives.

## Migration mapping

| pySHT | SHT 0.1.9 | Deliberate change |
|---|---|---|
| `lrt_1samp` | `mvar1.1998AS`, `mvar1.LRT` | Duplicate formulas share one function; null variance keyword is `variance` |
| `pn_2samp` | `mvar2.1930PN` | Lower beta tail, log-gamma moments, corrected method label |
| `pl_2samp` | `mvar2.1976PL` | Stable explicit component calculations |
| `muirhead_2samp` | `mvar2.1982Muirhead` | Survival tail and explicit truncation |
| `exact_lrt_2samp` | `mvar2.2012ZXC` | Stable conditional-beta integral |
| `lrt_2samp` | `mvar2.LRT` | Log-domain likelihood ratio |

Primary sources: [Arnold and Shavelle
(1998)](https://doi.org/10.1080/00031305.1998.10480552), [Pearson and Neyman
(1930)](https://errorstatistics.com/wp-content/uploads/2019/01/e.pearson26j.neyman1930282p29.pdf),
[Perng and Littell
(1976)](https://doi.org/10.1080/01621459.1976.10480978), Muirhead's
*Aspects of Multivariate Statistical Theory* (1982), and [Zhang, Xu, and Chen
(2012)](https://doi.org/10.1080/00031305.2012.707083).

# Joint mean and covariance: formula and validation ledger

This ledger covers `pysht.mean_covariance`. Each alternative is the
complement of the joint null: a departure in the mean, covariance, or both is
evidence against the null. SHT 0.1.9 was used only as migration evidence;
published formulas, literal independent implementations, and null simulations
are the scientific gates.

## Formula-source ledger

| pySHT function | Primary formula basis | Independent oracle | Status |
|---|---|---|---|
| `llzs_1samp` | Liu, Liu, Zheng, and Shi (2017), Section 2, pp. 84--87 | Literal raw-second-moment formula and high-dimensional null simulations | Public; legacy centering and tail corrected |
| `lrt_1samp` | Multivariate normal likelihood and Wilks' fixed-dimension limit | Original-coordinate solve and log-determinant identity | Public; fixed-dimensional asymptotic calibration |
| `hn_2samp` | Hyodo and Nishiyama (2018), Sections 2.1--2.2; equations (2.1)--(2.2) in the open technical report | Literal trace-estimator transcription and high-dimensional null simulations | Public |

## Stable null whitening

For the one-sample methods, define

$$
Z_i=\Sigma_0^{-1/2}(X_i-\mu_0).
$$

pySHT obtains $Z_i$ with a Cholesky solve, never by forming an inverse or
matrix square root. Tests generate $X_i=\mu_0+LZ_i$ with
$\Sigma_0=LL^\mathsf{T}$ and recover the identity-null statistic and p-value
to floating-point tolerance. A supplied null covariance must be finite,
symmetric, and positive definite.

## LLZS one-sample high-dimensional test

For $n$ whitened observations in $p$ dimensions, let

$$
\bar Z=n^{-1}\sum_i Z_i,\qquad
M_n=n^{-1}\sum_i Z_iZ_i^\mathsf{T},
$$

and define

$$
T_L=\|\bar Z\|^2+\operatorname{tr}(M_n-I_p)^2.
$$

With $y=p/n$ and marginal excess-kurtosis parameter $\beta$, Liu, Liu, Zheng,
and Shi derive

$$
\mu_0=y(p+\beta+2),\qquad
\sigma_0^2=4y^2\{y(2+\beta)+1\},
$$

and $(T_L-\mu_0)/\sigma_0\Rightarrow N(0,1)$. pySHT uses the paper's moment
estimator

$$
\widehat\beta=(np)^{-1}\sum_{i,j}Z_{ij}^4-3
$$

and the upper normal tail. The population targets behind both squared
departures are nonnegative, so alternatives move the statistic to the right.
The realized aspect ratio and estimated kurtosis are calibration diagnostics,
not scientific parameter estimates returned by the procedure.

When $p>n$, the implementation evaluates
$\operatorname{tr}(M_n^2)$ through the $n$-by-$n$ observation Gram matrix.
It therefore does not allocate a $p$-by-$p$ matrix in the regime for which the
method was designed. A separate fixture compares this route with the literal
feature-space formula.

SHT 0.1.9 replaced $M_n$ with the covariance matrix centered at the sample
mean while retaining the null center derived for $M_n$. When $p/n\to y>0$,
that changes the statistic by an order-one amount. It also used a two-sided
normal p-value. The fixed regression fixture distinguishes raw and centered
second moments and checks the corrected upper tail.

The advertised regime has independent rows, the paper's coordinate and
spectral regularity, a finite fourth moment, and $p/n$ bounded away from zero
and infinity. The explicit release gates below cover Gaussian, bounded
non-Gaussian, and finite-fourth-moment heavy-tailed coordinates.

## Classical one-sample likelihood-ratio test

Let $\bar Z$ be the whitened sample mean and

$$
S_n=n^{-1}\sum_i(Z_i-\bar Z)(Z_i-\bar Z)^\mathsf{T}.
$$

The normal likelihood gives

$$
-2\log\Lambda=n\left\{
\operatorname{tr}(S_n)-\log|S_n|-p+\|\bar Z\|^2
\right\}
\ \xrightarrow{d}\ \chi^2_{p(p+3)/2}.
$$

The implementation requires $n>p$ and a positive-definite fitted covariance;
it never substitutes a pseudoinverse. Eigenvalue contributions are evaluated
as $\lambda-1-\log1p(\lambda-1)$ to preserve accuracy near the null. The
fixed-data oracle computes the same likelihood directly in the original
coordinates with solves and determinant ratios.

This is a multivariate-normal, fixed-dimension limit. It is not a fallback for
a singular or proportional high-dimensional design.

## Hyodo--Nishiyama two-sample test

For group $g\in\{1,2\}$, let $S_g$ be the unbiased sample covariance and

$$
K_g=(n_g-1)^{-1}\sum_i\|X_{gi}-\bar X_g\|^4.
$$

The unbiased trace-square estimator is

$$
A_g=
\frac{n_g-1}{n_g(n_g-2)(n_g-3)}
\left[
(n_g-1)(n_g-2)\operatorname{tr}(S_g^2)
+\operatorname{tr}(S_g)^2-n_gK_g
\right].
$$

With $C=\operatorname{tr}(S_1S_2)$, the squared-distance estimators are

$$
\widehat d^2=
\|\bar X_1-\bar X_2\|^2
-\frac{\operatorname{tr}S_1}{n_1}
-\frac{\operatorname{tr}S_2}{n_2},
$$

$$
\widehat D^2=A_1+A_2-2C.
$$

Under the joint null, define

$$
V_1=
\frac{2A_1}{n_1^2}+\frac{2A_2}{n_2^2}
+\frac{4C}{n_1n_2},
$$

$$
V_2=
\frac{4A_1^2}{n_1^2}+\frac{4A_2^2}{n_2^2}
+\frac{8C^2}{n_1n_2}.
$$

Hyodo and Nishiyama's statistic is

$$
T=\frac{\widehat d^2}{\sqrt{V_1}}
+\frac{\widehat D^2}{\sqrt{V_2}},
\qquad
\frac{T}{\sqrt2}\Rightarrow N(0,1),
$$

with an upper-tail p-value. At least four rows per group are needed to define
$A_g$, and nonpositive finite-sample variance estimates are rejected.

The statistic is evaluated after a common translation and scaling to protect
fourth-order arithmetic. The two distance estimates are converted back to the
original squared and fourth-power measurement units. If an original-unit
estimate lies outside finite float64 range, that estimate is omitted rather
than mislabeled in computational units.

Trace products are evaluated adaptively in feature space when $p$ is small
and through observation Gram matrices when $p$ is large. Both paths are
checked against literal covariance-matrix calculations.

Tests independently transcribe the paper's equations and cover group
exchange, common translation and scaling, orthogonal feature transformations,
and alternatives that change only the mean or only the covariance. The
advertised null limit requires uniformly bounded eighth moments, the paper's
mixed-moment factorization for distinct latent coordinates, both group sizes
and dimension increasing, and assumptions A1--A2 on sample and trace growth.

## Explicit asymptotic size gates

Every release row uses 20,000 null datasets and is assessed simultaneously at
$\alpha\in\{0.01,0.05,0.10\}$ against

$$
|\widehat\alpha-\alpha|
\leq
\max\left(0.005,\,
4\sqrt{\frac{\alpha(1-\alpha)}{20000}}\right).
$$

### LLZS proportional-growth regime

| scenario key | seed | counts at 0.01, 0.05, 0.10 | rates | gate |
|---|---:|---:|---:|---|
| `mean-covariance.llzs.normal-n100-p50` | 20260831 | 235, 1047, 1964 | 0.01175, 0.05235, 0.09820 | Pass |
| `mean-covariance.llzs.normal-n100-p50` | 20260901 | 255, 1044, 1972 | 0.01275, 0.05220, 0.09860 | Pass |
| `mean-covariance.llzs.normal-n100-p100` | 20260831 | 198, 1007, 2009 | 0.00990, 0.05035, 0.10045 | Pass |
| `mean-covariance.llzs.normal-n100-p100` | 20260901 | 229, 989, 1917 | 0.01145, 0.04945, 0.09585 | Pass |
| `mean-covariance.llzs.normal-n100-p200` | 20260831 | 183, 963, 1961 | 0.00915, 0.04815, 0.09805 | Pass |
| `mean-covariance.llzs.normal-n100-p200` | 20260901 | 195, 958, 1939 | 0.00975, 0.04790, 0.09695 | Pass |
| `mean-covariance.llzs.t8-n100-p100` | 20260831 | 193, 938, 1871 | 0.00965, 0.04690, 0.09355 | Pass |
| `mean-covariance.llzs.t8-n100-p100` | 20260901 | 196, 987, 1915 | 0.00980, 0.04935, 0.09575 | Pass |
| `mean-covariance.llzs.uniform-n100-p100` | 20260831 | 221, 995, 1974 | 0.01105, 0.04975, 0.09870 | Pass |
| `mean-covariance.llzs.uniform-n100-p100` | 20260901 | 222, 1030, 2018 | 0.01110, 0.05150, 0.10090 | Pass |

The smaller $(n,p)=(50,25)$ Gaussian scenario failed one seed at level 0.01
with rejection rate `0.01515`; pySHT therefore does not claim broad
small-sample calibration from the asymptotic LLZS law.

### Fixed-dimensional likelihood-ratio regime

| scenario key | seed | counts at 0.01, 0.05, 0.10 | rates | gate |
|---|---:|---:|---:|---|
| `mean-covariance.lrt.normal-n200-p2` | 20260831 | 200, 1021, 2023 | 0.01000, 0.05105, 0.10115 | Pass |
| `mean-covariance.lrt.normal-n200-p2` | 20260901 | 222, 1042, 2056 | 0.01110, 0.05210, 0.10280 | Pass |
| `mean-covariance.lrt.normal-n300-p3` | 20260831 | 201, 997, 2021 | 0.01005, 0.04985, 0.10105 | Pass |
| `mean-covariance.lrt.normal-n300-p3` | 20260901 | 213, 1035, 2041 | 0.01065, 0.05175, 0.10205 | Pass |
| `mean-covariance.lrt.normal-n500-p5` | 20260831 | 194, 1041, 2084 | 0.00970, 0.05205, 0.10420 | Pass |
| `mean-covariance.lrt.normal-n500-p5` | 20260901 | 201, 1030, 2041 | 0.01005, 0.05150, 0.10205 | Pass |

At $(n,p)=(50,5)$, the uncorrected Wilks calibration yielded approximately
$(0.015,0.071,0.131)$ and failed. The documentation therefore treats
`lrt_1samp` as genuinely fixed-dimensional and large-sample.

### HN high-dimensional regime

| scenario key | seed | counts at 0.01, 0.05, 0.10 | rates | gate |
|---|---:|---:|---:|---|
| `mean-covariance.hn.normal-n100-n100-p200` | 20260831 | 215, 1083, 2055 | 0.01075, 0.05415, 0.10275 | Pass |
| `mean-covariance.hn.normal-n100-n100-p200` | 20260901 | 236, 1030, 1992 | 0.01180, 0.05150, 0.09960 | Pass |
| `mean-covariance.hn.normal-n150-n180-p300` | 20260831 | 261, 1074, 2084 | 0.01305, 0.05370, 0.10420 | Pass |
| `mean-covariance.hn.normal-n150-n180-p300` | 20260901 | 240, 1008, 2016 | 0.01200, 0.05040, 0.10080 | Pass |
| `mean-covariance.hn.uniform-n150-n180-p300` | 20260831 | 251, 1054, 2050 | 0.01255, 0.05270, 0.10250 | Pass |
| `mean-covariance.hn.uniform-n150-n180-p300` | 20260901 | 222, 1047, 2034 | 0.01110, 0.05235, 0.10170 | Pass |

The smaller unequal scenario $(100,120,200)$ narrowly failed one of two seeds
at level 0.05 (`0.05655` versus a tolerance endpoint of about
`0.05616`). This is evidence of finite-sample convergence error, not
a formula substitution. The public method is documented for the larger
high-dimensional regime rather than being described as a generic procedure
for all arrays with four rows.

### Stream contract and targeted power

All fresh tables above are produced by
`python -m tools.covariance_release_audits --all`. Each scenario/seed pair
resets `SeedSequence(seed)`, spawns the data stream first and auxiliary stream
second, and advances in replication order. The runs used Python 3.12.13,
NumPy 2.5.1, SciPy 1.18.0, and pySHT 0.1.0. Every p-value comes from the
public function itself.

Targeted alternatives used 2,000 replications and integer seed 20260902, with
`SeedSequence([seed, method_index])` reset for each function. At level 0.05:

| public function | fully specified alternative | count | rate |
|---|---|---:|---:|
| `llzs_1samp` | Gaussian, $n=100,p=100$; every mean coordinate shifted by 0.2 | 2000 | 1.0000 |
| `lrt_1samp` | Gaussian, $n=300,p=3$; mean shift $(0.3,0,0)$ | 1945 | 0.9725 |
| `hn_2samp` | Gaussian, $n_1=150,n_2=180,p=300$; every second-group mean coordinate shifted by 0.1 | 2000 | 1.0000 |

Run `python -m tools.covariance_power_audits` to reproduce these counts. This
is targeted power evidence for the advertised regimes, not a claim that the
same power holds for smaller samples or arbitrary alternatives.

## Migration mapping

| pySHT | SHT 0.1.9 | Deliberate change |
|---|---|---|
| `llzs_1samp` | `sim1.2017Liu` | Raw second moment, upper tail, correct logical alternative |
| `lrt_1samp` | `sim1.LRT` | Stable whitening, strict SPD/rank domain, correct logical alternative |
| `hn_2samp` | `sim2.2018HN` | Stable scaling, original-unit estimates, correct logical alternative |

Primary sources are [Liu, Liu, Zheng, and Shi
(2017)](https://doi.org/10.1016/j.jspi.2017.03.009) and [Hyodo and Nishiyama
(2018)](https://doi.org/10.1007/s11749-017-0567-x). The open
[Hyodo--Nishiyama technical report](https://www.math.sci.hiroshima-u.ac.jp/stat/TR/TR17/TR17-06.pdf)
provides the equation-numbered oracle used in the independent tests.

# Normality tests: formula and validation ledger

This ledger covers `pysht.normality`. The pinned SHT 0.1.9 implementation was
audited at commit `4e29cda`, but its output is used only for differential
investigation. Primary formulas, independent NumPy calculations, SciPy's
Shapiro implementation, invariance checks, and null simulation are the
correctness oracles.

The pySHT-native multivariate procedures have method-specific ledgers:
[Henze--Zirkler](henze-zirkler.md) and [energy normality](energy-normality.md).

## Shapiro--Wilk and Shapiro--Francia

`shapiro_wilk` delegates the statistic and p-value approximation to
`scipy.stats.shapiro` after a numerically safe location/scale normalization.
It accepts $3\le n\le5000$. This limit avoids presenting the Royston p-value
approximation beyond its validated range.

For ordered observations $x_{(i)}$, `shapiro_francia` uses

$$
W'=\operatorname{cor}^2\!\left(
  x_{(i)},\;\Phi^{-1}\!\left(\frac{i-3/8}{n+1/4}\right)
\right).
$$

For $5\le n\le5000$, its p-value follows Royston's normal approximation to
$\log(1-W')$, with

$$
\begin{aligned}
u&=\log n, & v&=\log u,\\
\mu&=-1.2725+1.0521(v-u), &
\sigma&=1.0308-0.26758(v+2/u).
\end{aligned}
$$

The fixed fixture agrees with an independent normal-score correlation to
approximately $10^{-14}$. Tests also cover the exact $W'=1$ boundary,
reflection, translation, scales near $10^{100}$, sample-size endpoints, and
constant samples. Standardization subtracts a sample anchor before dividing by
its scale. This ordering is necessary: scale-first centering changed a
Shapiro--Francia fixture near $10^{100}$ by more than `0.01`, even though all
within-sample differences remained representable.

A 20,000-sample $N(0,1)$ null audit produced these raw rejection counts and
rates:

| seed | $n$ | procedure | 0.01 count/rate | 0.05 count/rate | 0.10 count/rate |
|---:|---:|---|---:|---:|---:|
| 20260815 | 20 | Shapiro--Wilk | 184 (0.00920) | 1,034 (0.05170) | 2,030 (0.10150) |
| 20260815 | 20 | Shapiro--Francia | 216 (0.01080) | 1,056 (0.05280) | 2,078 (0.10390) |
| 20260815 | 100 | Shapiro--Wilk | 199 (0.00995) | 963 (0.04815) | 1,984 (0.09920) |
| 20260815 | 100 | Shapiro--Francia | 218 (0.01090) | 1,045 (0.05225) | 2,003 (0.10015) |
| 20260816 | 20 | Shapiro--Wilk | 162 (0.00810) | 925 (0.04625) | 2,011 (0.10055) |
| 20260816 | 20 | Shapiro--Francia | 192 (0.00960) | 997 (0.04985) | 2,060 (0.10300) |
| 20260816 | 100 | Shapiro--Wilk | 204 (0.01020) | 1,009 (0.05045) | 2,036 (0.10180) |
| 20260816 | 100 | Shapiro--Francia | 217 (0.01085) | 1,065 (0.05325) | 2,060 (0.10300) |

For each procedure and seed, a fresh `numpy.random.default_rng(seed)` generated
the $n=20$ block first and continued into the $n=100$ block. Every replication
called the corresponding public function, and rejection meant
`result.pvalue < alpha`. The repository release-simulation runner preserves
that stream prelude and reproduces these counts exactly.

Every entry satisfies the release tolerance specified by the project plan.

Primary sources are [Shapiro and Wilk
(1965)](https://doi.org/10.2307/2333709), [Shapiro and Francia
(1972)](https://doi.org/10.1080/01621459.1972.10481232), and the Royston
approximations used by the implementations.

## Moment statistics

Let

$$
m_r=\frac1n\sum_i(x_i-\bar x)^r,
\qquad
g_1=\frac{m_3}{m_2^{3/2}},
\qquad
b_2=\frac{m_4}{m_2^2}.
$$

The Jarque--Bera statistic is

$$
JB=n\left(\frac{g_1^2}{6}+\frac{(b_2-3)^2}{24}\right).
$$

Urzúa's adjusted statistic replaces the asymptotic centers and variances by

$$
\operatorname{Var}(g_1)=\frac{6(n-2)}{(n+1)(n+3)},\quad
E(b_2)=\frac{3(n-1)}{n+1},\quad
\operatorname{Var}(b_2)=
\frac{24n(n-2)(n-3)}{(n+1)^2(n+3)(n+5)}.
$$

The robust statistic uses

$$
J_n=\sqrt{\pi/2}\,n^{-1}\sum_i|x_i-\operatorname{median}(x)|
$$

in the moment denominators and

$$
RJB=\frac{n}{6}\left(\frac{m_3}{J_n^3}\right)^2+
    \frac{n}{64}\left(\frac{m_4}{J_n^4}-3\right)^2.
$$

The displayed Gel--Gastwirth statistic (2008, p. 31) uses 64. SHT 0.1.9
defaults to 24 in both its signature and help page; pySHT deliberately follows
the primary procedure rather than preserving that legacy defect. Custom positive constants remain
available with Monte Carlo calibration, but are rejected with the chi-square
option because the latter requires the published standardization. The AJB and RJB default calls in SHT
0.1.9 also fail before computation because a vector-valued default is compared
inside a scalar `if`; both Python defaults are executable.

Primary sources are [Jarque and Bera
(1980)](https://doi.org/10.1016/0165-1765(80)90024-5), [Urzúa
(1996)](https://doi.org/10.1016/S0165-1765(96)00923-8), and [Gel and Gastwirth
(2008)](https://doi.org/10.1016/j.econlet.2007.05.022).

## Calibration policy

All three moment tests default to a parametric Monte Carlo normal null. The
observed statistic is compared with $B$ independent null statistics using

$$
p=\frac{b+1}{B+1},
$$

where ties count in the upper tail. Results include $B$, $b$, the
conditional Monte Carlo standard error, and a 95% Clopper--Pearson interval for
the underlying tail probability. An integer `rng` seed replays the full
calibration without touching NumPy's global random state. Literal seeded
simulations independently reproduce the counts for JB, AJB, and RJB.

The explicit `calibration="asymptotic"` option reports a chi-square
approximation with two degrees of freedom. It has no advertised finite-sample
size guarantee. The following seeded $N(0,1)$ diagnostic at nominal 0.05
illustrates why it is not the default; every cell is count/20,000 followed by
the rate:

| $n$ | seed | JB | AJB | RJB |
|---:|---:|---:|---:|---:|
| 20 | 20260827 | 441 (0.02205) | 1,185 (0.05925) | 1,192 (0.05960) |
| 20 | 20260828 | 506 (0.02530) | 1,186 (0.05930) | 1,191 (0.05955) |
| 100 | 20260827 | 799 (0.03995) | 1,051 (0.05255) | 1,134 (0.05670) |
| 100 | 20260828 | 783 (0.03915) | 1,041 (0.05205) | 1,064 (0.05320) |
| 2,000 | 20260827 | 940 (0.04700) | 953 (0.04765) | 821 (0.04105) |
| 2,000 | 20260828 | 978 (0.04890) | 992 (0.04960) | 876 (0.04380) |

The generator was reset from the displayed integer seed for each $(n,$ seed$)$
design, then 20,000 samples were consumed in row order. The production batched
row-statistic kernels and chi-square survival function were used; fixed draws
were also checked against each public function's statistic and asymptotic
p-value. This is a replayable diagnostic, not a claimed release gate.

No asymptotic regime is marketed as validated until it passes both release
seeds at nominal levels 0.01, 0.05, and 0.10.

## Targeted alternative-power audit

All five public routines were evaluated on the same 2,000 samples of size 100
from a Student $t_3$ alternative, generated in row order by
`numpy.random.default_rng(20260829)`. The Shapiro procedures were called
publicly once per row. For the three authoritative Monte Carlo moment tests, a
separate `numpy.random.default_rng(20260830)` generated a shared bank of
100,000 $N(0,1)$ samples; ties entered the upper-tail exceedance count and
$(b+1)/(100000+1)<0.05$ defined rejection. Their production batched kernels
were used for tractability and fixed draws were matched to the complete public
Monte Carlo path.

| Procedure | Rejections/2,000 | Rate |
|---|---:|---:|
| `shapiro_wilk` | 1,749 | 0.8745 |
| `shapiro_francia` | 1,809 | 0.9045 |
| `jarque_bera` | 1,807 | 0.9035 |
| `adjusted_jarque_bera` | 1,810 | 0.9050 |
| `robust_jarque_bera` | 1,849 | 0.9245 |

This named-seed audit supplies a direction-and-power check for one heavy-tailed
alternative. It does not claim a minimum power over every nonnormal
distribution.

## Legacy mapping

| pySHT | SHT 0.1.9 |
|---|---|
| `shapiro_wilk` | `norm.1965SW` |
| `shapiro_francia` | `norm.1972SF` |
| `jarque_bera` | `norm.1980JB` |
| `adjusted_jarque_bera` | `norm.1996AJB` |
| `robust_jarque_bera` | `norm.2008RJB` |

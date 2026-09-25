# Chen--Qin, Li, and Xue--Yao mean tests

This ledger covers the pySHT-native dense and fixed-small-sample mean tests.
The Xue--Yao implementation is retained privately because its requested
calibration has not passed the release gate.

## Chen--Qin two-sample test

For samples of sizes $n_1,n_2$, Chen and Qin (2010), Equation (2.1), define

$$
T_n=\frac{\sum_{i\ne j}X_i^TX_j}{n_1(n_1-1)}
+\frac{\sum_{i\ne j}Y_i^TY_j}{n_2(n_2-1)}
-\frac{2\sum_{i,j}X_i^TY_j}{n_1n_2}.
$$

The null variance from Equation (3.9) is

$$
\widehat\sigma^2=
\frac{2\widehat{\operatorname{tr}(\Sigma_1^2)}}{n_1(n_1-1)}+
\frac{2\widehat{\operatorname{tr}(\Sigma_2^2)}}{n_2(n_2-1)}+
\frac{4\widehat{\operatorname{tr}(\Sigma_1\Sigma_2)}}{n_1n_2}.
$$

The implemented variance estimator is a **documented Chen--Qin variant**:
its within-sample trace estimates are $A_n$ from Li and Chen (2012), Equation
(2.1), and its cross-sample trace estimate is their Equation (2.2).
These are not the same finite-sample quantities as the leave-two-out
expression preceding Chen and Qin's Theorem 2. The result diagnostics name
the variance estimator explicitly.

An independent identity makes the within-sample estimator precise. If
$(n)_4=n(n-1)(n-2)(n-3)$, then

$$
A_n=\frac{1}{4(n)_4}\sum_{i,j,k,l\;\mathrm{distinct}}
\{(X_i-X_j)^T(X_k-X_l)\}^2.
$$

The two differences in each summand are independent, have mean zero and
covariance $2\Sigma$, so the expected squared inner product is
$4\operatorname{tr}(\Sigma^2)$. This proves unbiasedness and translation
invariance without identifying the estimator with Chen--Qin's different
formula. The cross-sample estimator equals $\operatorname{tr}(S_1S_2)$ and
is unbiased by independence. Tests compare the Gram reductions both with
literal ordered sums and with the independent pair-difference identity, and
retain a fixture distinguishing $A_n=5.5303652114$ from the original
leave-two-out value $5.4313033476$.

The standard-normal tail is an increasing-sample-and-dimension limit, not a
finite-sample exact calibration. Chen and Qin's Equations (3.1)--(3.3) assume
independent samples with a linear factor representation, standardized latent
coordinates that are pseudo-independent for products of distinct coordinates
through total degree eight, a finite common fourth moment, and
$n_1/(n_1+n_2)\to\kappa\in(0,1)$. Their Equation (3.6) additionally requires
each fourth-order covariance trace to be negligible relative to
$\operatorname{tr}^2\{(\Sigma_1+\Sigma_2)^2\}$. Heavy tails without the
required moments, strongly spiked covariance sequences that violate the trace
condition, severely unbalanced sample sizes, or small $n$ and $p$ are outside
the stated null approximation. pySHT also requires at least four rows per
group so the unbiased trace estimators exist and rejects a nonpositive
estimated null variance instead of manufacturing a normal statistic.

For this variance variant, sufficient additional conditions are Li--Chen
(2012) A1--A3: both sample sizes and dimension diverge, their ratio stays
away from zero and one, latent eighth moments are finite with factorized
mixed moments through degree eight, and all covariance-pair traces diverge
while the fourth-order traces satisfy their Equation (2.4). Section 6.3
proves the ratio consistency of each $A_n$ without needing equality of the
two covariances; Section 6.5 gives the corresponding cross-trace variance
bound. Consequently the positive weighted sum above is ratio consistent.
Combined with Chen--Qin's null central limit theorem, Slutsky's theorem
justifies this studentization. This sufficient envelope is intentionally
stronger than an assumption of merely finite fourth moments; simulations
alone do not establish it for arbitrary heavy-tailed data.

Let $n=n_1+n_2$. The Gram reductions take
$O((n_1^2+n_2^2+n_1n_2)p)=O(n^2p)$ time and $O(n^2)$ auxiliary storage.
They avoid both the literal order-four loops and any dense $p\times p$
covariance matrix; input and centered-data storage remain $O(np)$.

The release stream used four independently spawned PCG64 streams from
`SeedSequence(20260839)`. Each shard generated 5,000 replications in order;
every replication drew $X\sim N_{1000}(0,I)$ with 50 rows, then an independent
$Y\sim N_{1000}(0,I)$ with 60 rows, and called `cq_2samp` once. The combined
20,000-run counts were `(259, 1030, 2045)` and rates
`(0.01295, 0.05150, 0.10225)` at levels `(0.01, 0.05, 0.10)`, passing the
project gate.

The opt-in `tools/native_null_audits.py` runner records the exact public-call,
PCG64, parent-`SeedSequence`, child-stream, draw-order, and replication
contract behind this table.

The targeted alternative used integer seed `20260941`, 2,000 replications,
$X\sim N_{1000}(0.12\mathbf 1,I)$ with 50 rows, and
$Y\sim N_{1000}(0,I)$ with 60 rows. Counts were `(2000, 2000, 2000)` and rates
`(1.0000, 1.0000, 1.0000)`.

Primary source: [Chen and Qin (2010)](https://doi.org/10.1214/09-AOS716),
Equations (2.1), (3.9), and Theorems 1--2. Variance-estimator source:
[Li and Chen (2012)](https://arxiv.org/pdf/1206.0917), Equations (2.1)--(2.2),
Conditions A1--A3, and Sections 6.3 and 6.5.

## Li fixed-small-sample tests

For one sample, Li (2023), Equations (2)--(4), treats
$H_{ij}=X_i^TX_j$, $i<j$, as an asymptotically independent univariate sample
as dimension diverges. If $q=n(n-1)/2$, pySHT reports

$$
t=\frac{\bar H}{s_H/\sqrt q},\qquad df=q-1,
$$

with an upper Student tail. A nonzero null vector is subtracted before forming
the products.

For two samples with $n_1\le n_2$, Equation (6) constructs

$$
Y_i=X_{1i}-\sqrt{n_1/n_2}X_{2i}
+\sqrt{n_1/n_2}\bar X_{2,[1:n_1]}-\bar X_2,
$$

and applies the same pair-product statistic. The ANOVA extension sums
$Y_{\ell i}^TY_{\ell j}$ across every nonreference group before Studentization.
Literal fixtures reproduce all three forms. The Scheffé construction uses the
first rows of larger groups; row order must therefore be arbitrary with respect
to the observations. In the final paper, the ANOVA construction preceding
Theorem 5 assumes the non-strict ordering
$n_m\ge\cdots\ge n_2\ge n_1\ge3$ and uses sample 1 as the fixed reference.
Accordingly, pySHT accepts balanced designs and uses the first minimum-sized
group in caller order. Permuting only nonreference groups leaves the result
unchanged. If multiple groups tie for the minimum, moving a different tied
group into the first-minimum position can change the realized statistic; each
fixed reference choice retains Theorem 5's null calibration, but the choice can
affect power. pySHT neither selects a reference from observed values nor
averages reference-specific statistics, since either would define a different
procedure. A balanced Gaussian fixture independently reconstructs Equation
(7) for each reference contrast, their sum $W_m$, and its Studentization.

These Student laws are increasing-dimension limits, not exact small-$p$
calibrations. The paper's Equations (3), (8), and (12) use linear factor
representations with mutually independent, standardized latent coordinates
and a finite common fourth moment. Conditions (C1)--(C3) require fourth-order
covariance traces to be negligible relative to the square of the relevant
second-order trace as $p\to\infty$, while sample sizes remain fixed and at
least three. Gaussian data are a permitted special case, not a requirement.
Strong factors that violate those trace conditions, dependence among latent
coordinates outside the model, or modest dimension can invalidate the
advertised Student approximation.

For `li_1samp`, forming the row Gram takes $O(n^2p)$ time and
$O(np+n^2)$ auxiliary storage, including the scaled copy of the observations.
For `li_2samp`, let $r=\min(n_1,n_2)$; the Scheffé construction and
pair-product Gram together take $O((n_1+n_2)p+r^2p)$ time and
$O((n_1+n_2)p+r^2)$ auxiliary storage. For `li_ksamp`, if the fixed
first-minimum reference group has size $r$ and there are $k$ groups, the
corresponding costs are $O((\sum_l n_l)p+(k-1)r^2p)$ time and
$O((\sum_l n_l)p+(k-1)r^2)$ auxiliary storage. None of these paths constructs
a $p\times p$ covariance matrix.

Fresh 20,000-replication normal-null streams were generated in replication,
group, row, then feature order:

| function | scenario | seed | counts at 0.01, 0.05, 0.10 | rates | gate |
|---|---|---:|---:|---:|---|
| `li_1samp` | $n=6,p=1000$ | 20260840 | 209, 1053, 2116 | 0.01045, 0.05265, 0.10580 | Pass |
| `li_2samp` | $n_1=6,n_2=9,p=1000$ | 20260841 | 213, 1035, 2038 | 0.01065, 0.05175, 0.10190 | Pass |
| `li_ksamp` | $n=(6,8,10),p=500$ | 20260842 | 221, 1040, 2030 | 0.01105, 0.05200, 0.10150 | Pass |

These rows are also executable through `tools/native_null_audits.py`; the
runner calls the named public function once for every independently generated
dataset.

Because the original gate used a unique smallest group, the tie correction
also received a separate balanced-null gate. Integer seed `20260844` drove one
persistent PCG64 stream for 20,000 replications; each replication drew three
independent $6\times500$ standard-normal matrices in caller order and made the
single call `li_ksamp(x, y, z)`. Rejection counts were `(233, 1063, 2033)` and
rates `(0.01165, 0.05315, 0.10165)` at levels `(0.01, 0.05, 0.10)`, passing
the same release tolerance at every level. This gate exercises the first tied
minimum as the fixed reference; it does not average over reference choices.

Targeted 2,000-run alternatives used one fresh PCG64 stream per row:

| function | seed and alternative | counts at 0.01, 0.05, 0.10 | rates |
|---|---|---:|---:|
| `li_1samp` | 20260942; $N_{1000}(0.2\mathbf1,I)$, $n=6$ | 1921, 1993, 2000 | 0.9605, 0.9965, 1.0000 |
| `li_2samp` | 20260943; means $0.22\mathbf1,0$, $n=(6,9)$, $p=1000$ | 1555, 1883, 1948 | 0.7775, 0.9415, 0.9740 |
| `li_ksamp` | 20260944; means $(0,0.2,-0.2)\mathbf1$, $n=(6,8,10)$, $p=500$ | 963, 1521, 1718 | 0.4815, 0.7605, 0.8590 |

Primary source: [Li (2023)](https://doi.org/10.1016/j.jmva.2023.105183),
Equations (2), (4), (6)--(10), and the non-strictly ordered ANOVA construction
and $W_m/\widehat\sigma_{m,0}$ limit immediately preceding Theorem 5.

## Xue--Yao status: private

`_xy_2samp` implements the statistic and multiplier bootstrap in Xue and Yao
(2020), Equations (3) and (6). It draws Gaussian multipliers in deterministic
batches of at most 256 and holds the observed centered samples fixed. A
separate one-draw loop with the identical multiplier stream reproduces every
exceedance; runtime and global-RNG-isolation tests pass.

It is not exported. With the planned default of 999 bootstrap draws, corrected
Monte Carlo p-values have a grid of $1/1000$, leaving only ten attainable
values at or below 0.01. A fresh 5,000-null pilot at
$(n_1,n_2,p)=(30,36,80)$, data seed `20260843`, and per-replication bootstrap
seeds `3000000+i` produced counts `(26,198,447)` and rates
`(0.0052,0.0396,0.0894)`. The 0.01 and 0.05 rates are conservative beyond the
release tolerance. A public callable would therefore violate the
correctness-gated policy. The implementation can be reconsidered with a larger
default bootstrap budget and a passing 20,000-run calibration audit.

Primary source: [Xue and Yao (2020)](https://doi.org/10.1214/19-AOS1848).

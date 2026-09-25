# Reproducing release simulations

The 20,000-dataset release audits are opt-in scientific validation, not part
of the normal unit-test suite. Their scenario registry and runner live in
`tools/release_simulations.py`; they are repository tooling rather than public
`pysht` API.

List the registered designs:

```console
python -m tools.release_simulations --list
```

Run both recorded seeds for one design and retain machine-readable output:

```console
python -m tools.release_simulations variance.bartlett.g3-n100 --format json
```

During development, a small deterministic plumbing run is useful:

```console
python -m tools.release_simulations variance.bartlett.g3-n100 \
  --seed 20260823 --replications 100
```

Only the default 20,000-replication run is compared with a ledger table. Use
`--verify-documented` when exact reproduction of the rounded counts is the
goal. A shortened run exercises the same sample generator and public test but
reports only its own release-tolerance decision.

## Reproducibility contract

Each registry entry fixes the null distribution, array dimensions, public
function and options, nominal levels, replication count, and integer seeds.
The data use an isolated `numpy.random.Generator` in replication order.
Rejection means `pvalue < alpha`. The normality ledger generated its `n=20`
block before its `n=100` block from the same stream; that otherwise easy-to-
miss prelude is part of the two `n=100` scenarios.

Replacement audits recorded directly in a ledger state their own equally
binding stream policy. Unless a ledger explicitly says otherwise, the generator
is reset from the displayed integer seed for every method/scenario pair. The
joint mean/variance replacement audits call the public function once per
replication. The moment-normality and Yang--Modarres interpoint diagnostics use
the production batched statistic kernels for tractability and independently
match fixed draws to the complete public-call path; their tables say so
explicitly. Raw rejection counts, rather than rounded rates alone, are the
replay target.

JSON output records Python, NumPy, SciPy, and pySHT versions along with raw
counts, rates, tolerances, decisions, and differences from the counts implied
by five-decimal ledger rates. Reproduction claims should retain this output
and the pySHT commit. Exact counts are expected only for the same code,
dependency versions, random-stream contract, and floating-point platform; the
statistical release gate is the scientifically relevant decision across
supported platforms.

The tolerance is a regression screen with an absolute 0.005 floor, not a
confidence bound on size accuracy. The native mean/covariance runner also
reports marginal exact-binomial 95% intervals for rejection rates and whether
each nominal level lies in its interval. Gate passage and nominal-level
inclusion answer different questions; see the
[interpretation of validation evidence](index.md).

## Provenance audit

Registry entries labeled `ledger-replay` have an explicit design, public
evaluation path, seed, and recorded rates in the corresponding ledger.
Registered replay designs cover the normality Shapiro tests, the exact
Yang--Modarres quantile test, simplex likelihood-ratio tests, and classical
multi-group variance tests. This focused set is intentionally evaluated
through public functions rather than duplicating their formulas in a release
script.

The scalar/domain audit replaced the formerly anonymous joint mean/variance,
moment-normality, and Yang--Modarres interpoint summaries. Their ledgers now
record complete null designs, explicit seeds, stream-reset policy, evaluation
path, raw counts, and rates. In particular, the corrected Q3 audit records
seeds 20260825 and 20260826 across four advertised $(n,d)$ designs and also
records the failed $(20,2)$ boundary design. These replacement rows are
replayable evidence even though their batched runners are not yet generalized
as command-line registry entries. Every scalar/domain public routine also has
a 2,000-replication strong-alternative audit with seed 20260829; moment
normality and interpoint Monte Carlo banks use the separately recorded seed
20260830.

Mean and equality-of-distributions power checks have a dedicated public-call
runner:

```console
python -m tools.mean_power_audits
```

Scenarios use consecutive integer seeds 2026090301 through 2026090322. Each
seed spawns a PCG64 data stream and a separate auxiliary-randomness stream;
both advance in replication order and reset between methods. Deterministic and
asymptotic procedures use 1,000 outer replications. The more expensive Thulin
and Biswas--Ghosh resampling scenarios use 300, with their inner resampling
budgets stated in the ledgers. Lee--You--Lin records paired log-Bayes-factor
movement rather than inventing a rejection threshold.

The native Chen--Qin, Li, and CZZ null gates are independently replayable with
fresh public calls:

```console
python -m tools.native_null_audits
```

Chen--Qin uses the four child streams of `SeedSequence(20260839)` for four
5,000-dataset shards; the Li and CZZ scenarios use the integer streams printed
in their ledgers for 20,000 datasets each. A short `--replications` run checks
plumbing without claiming to reproduce a release table. Their named
2,000-dataset alternatives are registered in `tools.mean_power_audits` under
seeds 20260941 through 20260946.

The additional `mean.li_ksamp.balanced` scenario replays the 20,000-dataset
balanced-group gate with seed 20260844. It fixes the first caller-supplied
minimum-size group as the paper's indexed reference and therefore audits the
non-strict sample-size boundary separately from the original `(6,8,10)` row.

The native distribution-equality and independence methods have a separate
runner:

```console
python -m tools.distribution_independence_audits equaldist.energy
python -m tools.distribution_independence_audits independence.hsic
```

The equality scenarios use fresh null datasets and the complete 210-label
orbit for group sizes 4 and 6. Independence scenarios use fresh row-paired
null datasets and a separate auxiliary stream for 999 marginal permutations.
The runner prints raw counts and applies the same 20,000-dataset tolerance;
`--power` selects its named strong-alternative registry.

Domain-expansion null and power evidence is opt-in because of its simulation
cost:

```console
PYSHT_RUN_DOMAIN_RELEASE_AUDITS=1 python -m pytest \
  tests/test_domain_release_audits.py -s
```

For parametric-null normality, EHY, and circular methods, the audit factors the
20,000 independently generated observed statistics from a 99,999-statistic
null reference bank. It therefore validates the null statistic law and
reported size without pretending to be 20,000 separate 9,999-draw public
calls. Focused module tests independently replay the complete public RNG,
nuisance-refitting, and corrected-tail path. Mardia--Watson--Wheeler and
alpha-energy use exhaustive conditional permutation orbits instead.

Covariance and joint mean/covariance designs have dedicated public-call
runners:

```console
python -m tools.covariance_release_audits --all
python -m tools.covariance_power_audits
```

The null runner uses integer seeds 20260831 and 20260901 for 20,000 datasets;
the targeted-alternative runner uses seed 20260902 for 2,000 datasets. Each
scenario resets its seed, separates data and auxiliary streams, and records
raw counts, rates, software versions, and the exact stream policy in the
method ledger.

The repository audit still found historical summaries that cannot yet be
replayed exactly:

- the covariance Fisher table used an optimized direct-scatter path that was
  not retained, so Fisher is withheld rather than credited with that evidence.

The historical Fisher rows are not release evidence. They remain absent from
the public-call runner unless the missing provenance is recovered or a fresh
replacement audit is completed.

Future validation tables should always include the scenario key, both seeds,
raw rejection counts, dependency versions, and the pySHT commit.

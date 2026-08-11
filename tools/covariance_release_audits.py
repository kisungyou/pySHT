"""Run reproducible 20,000-dataset covariance release audits.

Every scenario evaluates a public pySHT function.  For each scenario and
integer seed, a fresh ``SeedSequence`` is split into independent data and
auxiliary-randomness streams.  The streams reset at the start of every
scenario/seed pair and advance once per replication.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from types import MappingProxyType
from typing import Final

import numpy as np
import scipy

from pysht import covariance, mean_covariance
from tools.release_simulations import DEFAULT_REPLICATIONS, LEVELS, release_tolerance

type Evaluator = Callable[[np.random.Generator, np.random.Generator], float]

SEEDS: Final = (20260831, 20260901)


@dataclass(frozen=True, slots=True)
class Scenario:
    """One fully specified public-API null-calibration design."""

    key: str
    description: str
    evaluate: Evaluator


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Raw counts, rates, tolerances, and decisions for one audit stream."""

    scenario: str
    description: str
    seed: int
    replications: int
    rejection_counts: tuple[int, ...]
    rejection_rates: tuple[float, ...]
    tolerances: tuple[float, ...]
    gate_passes: tuple[bool, ...]

    @property
    def passes(self) -> bool:
        return all(self.gate_passes)


def _normal(generator: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    return generator.standard_normal((rows, columns))


def _standardized_t8(
    generator: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    return generator.standard_t(8, size=(rows, columns)) * math.sqrt(6.0 / 8.0)


def _standardized_uniform(
    generator: np.random.Generator, rows: int, columns: int
) -> np.ndarray:
    return math.sqrt(12.0) * (generator.random((rows, columns)) - 0.5)


def _registry() -> dict[str, Scenario]:
    def wl1(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        return covariance.wl_1samp(
            _normal(data, 100, 30), n_projections=25, rng=auxiliary
        ).pvalue

    def wl2(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        return covariance.wl_2samp(
            _normal(data, 300, 30),
            _normal(data, 360, 30),
            n_projections=50,
            rng=auxiliary,
        ).pvalue

    def lc(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return covariance.lc_2samp(_normal(data, 30, 50), _normal(data, 30, 50)).pvalue

    def clx(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return covariance.clx_2samp(
            _normal(data, 100, 30), _normal(data, 100, 30)
        ).pvalue

    def schott_2001(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return covariance.schott_2001_ksamp(
            *(_normal(data, 100, 3) for _ in range(3))
        ).pvalue

    def schott_2007(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return covariance.schott_2007_ksamp(
            *(_normal(data, 100, 50) for _ in range(3))
        ).pvalue

    def llzs_normal_p50(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean_covariance.llzs_1samp(_normal(data, 100, 50)).pvalue

    def llzs_normal_p100(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean_covariance.llzs_1samp(_normal(data, 100, 100)).pvalue

    def llzs_normal_p200(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean_covariance.llzs_1samp(_normal(data, 100, 200)).pvalue

    def llzs_t8(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean_covariance.llzs_1samp(_standardized_t8(data, 100, 100)).pvalue

    def llzs_uniform(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean_covariance.llzs_1samp(_standardized_uniform(data, 100, 100)).pvalue

    def lrt_n200_p2(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean_covariance.lrt_1samp(_normal(data, 200, 2)).pvalue

    def lrt_n300_p3(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean_covariance.lrt_1samp(_normal(data, 300, 3)).pvalue

    def lrt_n500_p5(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean_covariance.lrt_1samp(_normal(data, 500, 5)).pvalue

    def hn_normal_100(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean_covariance.hn_2samp(
            _normal(data, 100, 200), _normal(data, 100, 200)
        ).pvalue

    def hn_normal_150_180(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean_covariance.hn_2samp(
            _normal(data, 150, 300), _normal(data, 180, 300)
        ).pvalue

    def hn_uniform_150_180(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean_covariance.hn_2samp(
            _standardized_uniform(data, 150, 300),
            _standardized_uniform(data, 180, 300),
        ).pvalue

    entries = (
        Scenario(
            "covariance.wl-1samp.n100-p30-m25",
            "Wu--Li one-sample Gaussian null; N=100, p=30, m=25",
            wl1,
        ),
        Scenario(
            "covariance.wl-2samp.n300-n360-p30-m50",
            "Wu--Li two-sample Gaussian null; N1=300, N2=360, p=30, m=50",
            wl2,
        ),
        Scenario(
            "covariance.lc.n30-n30-p50", "Li--Chen Gaussian null; n1=n2=30, p=50", lc
        ),
        Scenario(
            "covariance.clx.n100-n100-p30",
            "Cai--Liu--Xia Gaussian null; n1=n2=100, p=30",
            clx,
        ),
        Scenario(
            "covariance.schott-2001.g3-n100-p3",
            "Schott (2001) Gaussian null; g=3, Ni=100, p=3",
            schott_2001,
        ),
        Scenario(
            "covariance.schott-2007.g3-n100-p50",
            "Schott (2007) Gaussian null; g=3, Ni=100, p=50",
            schott_2007,
        ),
        Scenario(
            "mean-covariance.llzs.normal-n100-p50",
            "LLZS standard-normal null; n=100, p=50",
            llzs_normal_p50,
        ),
        Scenario(
            "mean-covariance.llzs.normal-n100-p100",
            "LLZS standard-normal null; n=100, p=100",
            llzs_normal_p100,
        ),
        Scenario(
            "mean-covariance.llzs.normal-n100-p200",
            "LLZS standard-normal null; n=100, p=200",
            llzs_normal_p200,
        ),
        Scenario(
            "mean-covariance.llzs.t8-n100-p100",
            "LLZS variance-one Student-t8 null; n=100, p=100",
            llzs_t8,
        ),
        Scenario(
            "mean-covariance.llzs.uniform-n100-p100",
            "LLZS variance-one uniform null; n=100, p=100",
            llzs_uniform,
        ),
        Scenario(
            "mean-covariance.lrt.normal-n200-p2",
            "joint LRT standard-normal null; n=200, p=2",
            lrt_n200_p2,
        ),
        Scenario(
            "mean-covariance.lrt.normal-n300-p3",
            "joint LRT standard-normal null; n=300, p=3",
            lrt_n300_p3,
        ),
        Scenario(
            "mean-covariance.lrt.normal-n500-p5",
            "joint LRT standard-normal null; n=500, p=5",
            lrt_n500_p5,
        ),
        Scenario(
            "mean-covariance.hn.normal-n100-n100-p200",
            "HN standard-normal null; n1=n2=100, p=200",
            hn_normal_100,
        ),
        Scenario(
            "mean-covariance.hn.normal-n150-n180-p300",
            "HN standard-normal null; n1=150, n2=180, p=300",
            hn_normal_150_180,
        ),
        Scenario(
            "mean-covariance.hn.uniform-n150-n180-p300",
            "HN variance-one uniform null; n1=150, n2=180, p=300",
            hn_uniform_150_180,
        ),
    )
    return {entry.key: entry for entry in entries}


SCENARIOS: Final = MappingProxyType(_registry())


def run_scenario(scenario: Scenario, *, seed: int, replications: int) -> AuditResult:
    seed_sequence = np.random.SeedSequence(seed)
    data_seed, auxiliary_seed = seed_sequence.spawn(2)
    data_rng = np.random.default_rng(data_seed)
    auxiliary_rng = np.random.default_rng(auxiliary_seed)
    counts = np.zeros(len(LEVELS), dtype=np.int64)
    levels = np.asarray(LEVELS)
    for replication in range(replications):
        pvalue = float(scenario.evaluate(data_rng, auxiliary_rng))
        if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
            raise RuntimeError(
                f"{scenario.key} returned invalid p-value {pvalue!r} "
                f"at replication {replication}"
            )
        counts += pvalue < levels
        if (replication + 1) % 5_000 == 0:
            print(
                f"{scenario.key} seed={seed}: {replication + 1}/{replications}",
                file=sys.stderr,
                flush=True,
            )
    raw_counts = tuple(int(count) for count in counts)
    rates = tuple(count / replications for count in raw_counts)
    tolerances = tuple(release_tolerance(level, replications) for level in LEVELS)
    decisions = tuple(
        abs(rate - level) <= tolerance
        for rate, level, tolerance in zip(rates, LEVELS, tolerances, strict=True)
    )
    return AuditResult(
        scenario=scenario.key,
        description=scenario.description,
        seed=seed,
        replications=replications,
        rejection_counts=raw_counts,
        rejection_rates=rates,
        tolerances=tolerances,
        gate_passes=decisions,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="*", choices=tuple(SCENARIOS))
    parser.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--replications", type=int, default=DEFAULT_REPLICATIONS)
    args = parser.parse_args(argv)
    if args.replications <= 0:
        parser.error("--replications must be positive")
    if args.run_all and args.scenario:
        parser.error("use either scenario keys or --all")
    selected = tuple(SCENARIOS) if args.run_all else tuple(args.scenario)
    if not selected:
        parser.error("select at least one scenario or use --all")
    seeds = tuple(args.seed) if args.seed else SEEDS
    if any(seed < 0 for seed in seeds):
        parser.error("--seed must be nonnegative")
    results = [
        run_scenario(SCENARIOS[key], seed=seed, replications=args.replications)
        for key in selected
        for seed in seeds
    ]
    payload = {
        "stream_contract": (
            "Each scenario/seed resets numpy SeedSequence(seed), spawns data then "
            "auxiliary streams, and advances both in replication order."
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pysht": version("pysht"),
        "levels": LEVELS,
        "results": [{**asdict(result), "passes": result.passes} for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(result.passes for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproduce opt-in null-calibration simulations from the validation ledgers.

This module is deliberately repository-local: release simulations are far too
expensive for the normal test suite and are not part of the public ``pysht``
API.  The small API below exists so its scenario and random-stream plumbing can
still be tested without running a 20,000-replication audit.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import scipy
from numpy.typing import NDArray

from pysht import normality, simplex, uniformity, variance

type Draw = tuple[NDArray[np.float64], ...]
type DrawFunction = Callable[[np.random.Generator], Draw]
type PrepareFunction = Callable[[np.random.Generator, int], None]
type TestFunction = Callable[[Draw], float]

LEVELS: Final = (0.01, 0.05, 0.10)
DEFAULT_REPLICATIONS: Final = 20_000


def _no_prepare(generator: np.random.Generator, replications: int) -> None:
    del generator, replications


@dataclass(frozen=True, slots=True)
class SimulationScenario:
    """A fully specified null-simulation design."""

    key: str
    description: str
    ledger: str
    seeds: tuple[int, ...]
    draw: DrawFunction
    test: TestFunction
    prepare: PrepareFunction = _no_prepare
    documented_rates: tuple[tuple[int, tuple[float, ...]], ...] = ()
    provenance: str = "canonical"
    replications: int = DEFAULT_REPLICATIONS
    levels: tuple[float, ...] = LEVELS

    def reference_rates(self, seed: int) -> tuple[float, ...] | None:
        """Return the ledger rates for ``seed``, when a table records them."""
        for candidate, rates in self.documented_rates:
            if candidate == seed:
                return rates
        return None


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Counts, rates, and release-gate decisions from one scenario seed."""

    scenario: str
    seed: int
    replications: int
    levels: tuple[float, ...]
    rejection_counts: tuple[int, ...]
    rejection_rates: tuple[float, ...]
    tolerances: tuple[float, ...]
    gate_passes: tuple[bool, ...]
    documented_rates: tuple[float, ...] | None
    documented_count_differences: tuple[int, ...] | None

    @property
    def passes(self) -> bool:
        """Whether every nominal level satisfies the release tolerance."""
        return all(self.gate_passes)

    @property
    def reproduces_documented_counts(self) -> bool | None:
        """Whether rounded ledger rates imply the observed counts."""
        if self.documented_count_differences is None:
            return None
        return all(difference == 0 for difference in self.documented_count_differences)


def release_tolerance(level: float, replications: int) -> float:
    """Return the project's binomial release tolerance."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    if isinstance(replications, bool) or not isinstance(replications, int):
        raise TypeError("replications must be an integer")
    if replications <= 0:
        raise ValueError("replications must be positive")
    return max(0.005, 4.0 * math.sqrt(level * (1.0 - level) / replications))


def _normal_1samp(sample_size: int) -> DrawFunction:
    def draw(generator: np.random.Generator) -> Draw:
        return (generator.standard_normal(sample_size),)

    return draw


def _normal_groups(group_count: int, sample_size: int) -> DrawFunction:
    def draw(generator: np.random.Generator) -> Draw:
        return tuple(generator.standard_normal(sample_size) for _ in range(group_count))

    return draw


def _uniform_cube(sample_size: int, dimension: int) -> DrawFunction:
    def draw(generator: np.random.Generator) -> Draw:
        return (generator.random((sample_size, dimension)),)

    return draw


def _uniform_simplex(sample_size: int, dimension: int) -> DrawFunction:
    def draw(generator: np.random.Generator) -> Draw:
        return (generator.dirichlet(np.ones(dimension), size=sample_size),)

    return draw


def _discard_normal_prefix(sample_size: int) -> PrepareFunction:
    """Reproduce a ledger stream whose smaller design was generated first."""

    def prepare(generator: np.random.Generator, replications: int) -> None:
        generator.standard_normal((replications, sample_size))

    return prepare


def _pvalue_1samp(function: Callable[..., Any], **kwargs: Any) -> TestFunction:
    def test(draw: Draw) -> float:
        return float(function(draw[0], **kwargs).pvalue)

    return test


def _pvalue_groups(function: Callable[..., Any], **kwargs: Any) -> TestFunction:
    def test(draw: Draw) -> float:
        return float(function(*draw, **kwargs).pvalue)

    return test


def _scenario_registry() -> dict[str, SimulationScenario]:
    scenarios = [
        SimulationScenario(
            key="normality.shapiro-wilk.n20",
            description="Shapiro--Wilk under N(0,1), n=20",
            ledger="docs/validation/normality.md",
            seeds=(20260815, 20260816),
            draw=_normal_1samp(20),
            test=_pvalue_1samp(normality.shapiro_wilk),
            documented_rates=(
                (20260815, (0.00920, 0.05170, 0.10150)),
                (20260816, (0.00810, 0.04625, 0.10055)),
            ),
            provenance="ledger-replay",
        ),
        SimulationScenario(
            key="normality.shapiro-wilk.n100",
            description="Shapiro--Wilk under N(0,1), n=100",
            ledger="docs/validation/normality.md",
            seeds=(20260815, 20260816),
            draw=_normal_1samp(100),
            test=_pvalue_1samp(normality.shapiro_wilk),
            prepare=_discard_normal_prefix(20),
            documented_rates=(
                (20260815, (0.00995, 0.04815, 0.09920)),
                (20260816, (0.01020, 0.05045, 0.10180)),
            ),
            provenance="ledger-replay",
        ),
        SimulationScenario(
            key="normality.shapiro-francia.n20",
            description="Shapiro--Francia under N(0,1), n=20",
            ledger="docs/validation/normality.md",
            seeds=(20260815, 20260816),
            draw=_normal_1samp(20),
            test=_pvalue_1samp(normality.shapiro_francia),
            documented_rates=(
                (20260815, (0.01080, 0.05280, 0.10390)),
                (20260816, (0.00960, 0.04985, 0.10300)),
            ),
            provenance="ledger-replay",
        ),
        SimulationScenario(
            key="normality.shapiro-francia.n100",
            description="Shapiro--Francia under N(0,1), n=100",
            ledger="docs/validation/normality.md",
            seeds=(20260815, 20260816),
            draw=_normal_1samp(100),
            test=_pvalue_1samp(normality.shapiro_francia),
            prepare=_discard_normal_prefix(20),
            documented_rates=(
                (20260815, (0.01090, 0.05225, 0.10015)),
                (20260816, (0.01085, 0.05325, 0.10300)),
            ),
            provenance="ledger-replay",
        ),
        SimulationScenario(
            key="uniformity.ym-quantile.n50-p3",
            description="Yang--Modarres quantile test under U(0,1)^3, n=50",
            ledger="docs/validation/uniformity.md",
            seeds=(20260817, 20260818),
            draw=_uniform_cube(50, 3),
            test=_pvalue_1samp(uniformity.ym_quantile),
            documented_rates=(
                (20260817, (0.00850, 0.04785, 0.09875)),
                (20260818, (0.00980, 0.05085, 0.09940)),
            ),
            provenance="ledger-replay",
        ),
        SimulationScenario(
            key="simplex.symmetric.n50-k3",
            description="symmetric Dirichlet LRT under Dirichlet(1,1,1), n=50",
            ledger="docs/validation/simplex-uniformity.md",
            seeds=(20260813, 20260814),
            draw=_uniform_simplex(50, 3),
            test=_pvalue_1samp(simplex.uniformity, model="symmetric"),
            documented_rates=(
                (20260813, (0.00975, 0.05055, 0.10020)),
                (20260814, (0.01055, 0.05170, 0.10060)),
            ),
            provenance="ledger-replay",
        ),
        SimulationScenario(
            key="simplex.general.n50-k3",
            description="general Dirichlet LRT under Dirichlet(1,1,1), n=50",
            ledger="docs/validation/simplex-uniformity.md",
            seeds=(20260813, 20260814),
            draw=_uniform_simplex(50, 3),
            test=_pvalue_1samp(simplex.uniformity, model="general"),
            documented_rates=(
                (20260813, (0.01030, 0.05135, 0.10155)),
                (20260814, (0.01130, 0.05225, 0.10390)),
            ),
            provenance="ledger-replay",
        ),
        SimulationScenario(
            key="variance.bartlett.g3-n100",
            description="Bartlett under three N(0,1) groups, n_i=100",
            ledger="docs/validation/classical-variance.md",
            seeds=(20260823, 20260824),
            draw=_normal_groups(3, 100),
            test=_pvalue_groups(variance.bartlett),
            documented_rates=(
                (20260823, (0.00980, 0.05140, 0.10350)),
                (20260824, (0.00910, 0.04785, 0.09735)),
            ),
            provenance="ledger-replay",
        ),
        SimulationScenario(
            key="variance.levene.g3-n100",
            description="mean-centered Levene under three N(0,1) groups, n_i=100",
            ledger="docs/validation/classical-variance.md",
            seeds=(20260823, 20260824),
            draw=_normal_groups(3, 100),
            test=_pvalue_groups(variance.levene),
            documented_rates=(
                (20260823, (0.00990, 0.05300, 0.10340)),
                (20260824, (0.00970, 0.05095, 0.10225)),
            ),
            provenance="ledger-replay",
        ),
        SimulationScenario(
            key="variance.brown-forsythe.g3-n100",
            description="Brown--Forsythe under three N(0,1) groups, n_i=100",
            ledger="docs/validation/classical-variance.md",
            seeds=(20260823, 20260824),
            draw=_normal_groups(3, 100),
            test=_pvalue_groups(variance.brown_forsythe),
            documented_rates=(
                (20260823, (0.00885, 0.04880, 0.09590)),
                (20260824, (0.00845, 0.04690, 0.09580)),
            ),
            provenance="ledger-replay",
        ),
    ]
    return {scenario.key: scenario for scenario in scenarios}


SCENARIOS: Final[Mapping[str, SimulationScenario]] = MappingProxyType(
    _scenario_registry()
)


def run_scenario(
    scenario: SimulationScenario,
    *,
    seed: int,
    replications: int | None = None,
) -> SimulationResult:
    """Run one scenario with an isolated and replayable random stream."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    total = scenario.replications if replications is None else replications
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError("replications must be an integer")
    if total <= 0:
        raise ValueError("replications must be positive")

    generator = np.random.default_rng(seed)
    scenario.prepare(generator, total)
    counts = np.zeros(len(scenario.levels), dtype=np.int64)
    for replication in range(total):
        draw = scenario.draw(generator)
        try:
            pvalue = scenario.test(draw)
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{scenario.key} failed at replication {replication}"
            ) from exc
        if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
            raise RuntimeError(
                f"{scenario.key} returned invalid p-value {pvalue!r} "
                f"at replication {replication}"
            )
        counts += pvalue < np.asarray(scenario.levels)

    rejection_counts = tuple(int(value) for value in counts)
    rates = tuple(value / total for value in rejection_counts)
    tolerances = tuple(release_tolerance(level, total) for level in scenario.levels)
    gate_passes = tuple(
        abs(rate - level) <= tolerance
        for level, rate, tolerance in zip(
            scenario.levels, rates, tolerances, strict=True
        )
    )
    documented = (
        scenario.reference_rates(seed) if total == scenario.replications else None
    )
    differences = (
        None
        if documented is None
        else tuple(
            observed - round(rate * scenario.replications)
            for observed, rate in zip(rejection_counts, documented, strict=True)
        )
    )
    return SimulationResult(
        scenario=scenario.key,
        seed=seed,
        replications=total,
        levels=scenario.levels,
        rejection_counts=rejection_counts,
        rejection_rates=rates,
        tolerances=tolerances,
        gate_passes=gate_passes,
        documented_rates=documented,
        documented_count_differences=differences,
    )


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pysht": version("pysht"),
    }


def _text_result(result: SimulationResult) -> str:
    lines = [
        f"scenario: {result.scenario}",
        f"seed: {result.seed}",
        f"replications: {result.replications}",
        "alpha  count  rate     tolerance  gate  documented_delta",
    ]
    for index, (level, count, rate, tolerance, passed) in enumerate(
        zip(
            result.levels,
            result.rejection_counts,
            result.rejection_rates,
            result.tolerances,
            result.gate_passes,
            strict=True,
        )
    ):
        difference = (
            "--"
            if result.documented_count_differences is None
            else f"{result.documented_count_differences[index]:+d}"
        )
        lines.append(
            f"{level:0.2f}   {count:5d}  {rate:0.5f}  {tolerance:0.5f}   "
            f"{'pass' if passed else 'FAIL':4s}  {difference}"
        )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", nargs="?", choices=tuple(SCENARIOS))
    parser.add_argument(
        "--list", action="store_true", help="list registered scenarios and exit"
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="seed to run; repeat for multiple seeds (defaults to ledger seeds)",
    )
    parser.add_argument(
        "--replications",
        type=int,
        help="override the release scenario's 20,000 replications",
    )
    parser.add_argument(
        "--format", choices=("text", "json"), default="text", help="output format"
    )
    parser.add_argument(
        "--verify-documented",
        action="store_true",
        help="fail unless a full run exactly reproduces rounded ledger counts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process status."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.list:
        for scenario in SCENARIOS.values():
            seed_text = ",".join(str(seed) for seed in scenario.seeds)
            print(f"{scenario.key}\t{scenario.provenance}\tseeds={seed_text}")
        return 0
    if arguments.scenario is None:
        parser.error("a scenario is required unless --list is used")

    scenario = SCENARIOS[arguments.scenario]
    seeds = scenario.seeds if arguments.seed is None else tuple(arguments.seed)
    results = tuple(
        run_scenario(
            scenario,
            seed=seed,
            replications=arguments.replications,
        )
        for seed in seeds
    )
    if arguments.format == "json":
        print(
            json.dumps(
                {
                    "environment": _environment(),
                    "scenario": {
                        "key": scenario.key,
                        "description": scenario.description,
                        "ledger": scenario.ledger,
                        "provenance": scenario.provenance,
                    },
                    "results": [asdict(result) for result in results],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("\n\n".join(_text_result(result) for result in results))

    if arguments.verify_documented:
        return (
            0
            if all(result.reproduces_documented_counts is True for result in results)
            else 2
        )
    return 0 if all(result.passes for result in results) else 1


if __name__ == "__main__":  # pragma: no cover - exercised through ``main``
    raise SystemExit(main())

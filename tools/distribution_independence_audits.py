"""Genuine named-seed null and power audits for expanded resampling tests."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np

from pysht.equaldist import energy_ksamp, mmd_2samp
from pysht.independence import (
    dhsic,
    distance_covariance,
    distance_multivariance,
    hsic,
)

type Generator = np.random.Generator
type ScenarioFunction = Callable[[Generator, Generator], float]

LEVELS: Final = (0.01, 0.05, 0.10)
DEFAULT_REPLICATIONS: Final = 20_000
DEFAULT_POWER_REPLICATIONS: Final = 300


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    seed: int
    description: str
    sampler: ScenarioFunction


@dataclass(frozen=True, slots=True)
class AuditResult:
    scenario: str
    seed: int
    description: str
    replications: int
    levels: tuple[float, ...]
    rejection_counts: tuple[int, ...]
    rejection_rates: tuple[float, ...]
    tolerances: tuple[float, ...]
    passes: tuple[bool, ...]


def _release_tolerance(alpha: float, replications: int) -> float:
    return max(0.005, 4.0 * math.sqrt(alpha * (1.0 - alpha) / replications))


def _groups(data: Generator) -> tuple[np.ndarray, np.ndarray]:
    return data.normal(size=(4, 2)), data.normal(size=(6, 2))


def _energy_null(data: Generator, auxiliary: Generator) -> float:
    del auxiliary
    x, y = _groups(data)
    return energy_ksamp(x, y, calibration="exact", n_resamples=210).pvalue


def _mmd_null(data: Generator, auxiliary: Generator) -> float:
    del auxiliary
    x, y = _groups(data)
    return mmd_2samp(x, y, calibration="exact", n_resamples=210).pvalue


def _paired(data: Generator) -> tuple[np.ndarray, np.ndarray]:
    return data.normal(size=(6, 2)), data.normal(size=(6, 2))


def _dcov_null(data: Generator, auxiliary: Generator) -> float:
    x, y = _paired(data)
    return distance_covariance(
        x, y, calibration="monte-carlo", n_resamples=999, rng=auxiliary
    ).pvalue


def _hsic_null(data: Generator, auxiliary: Generator) -> float:
    x, y = _paired(data)
    return hsic(x, y, calibration="monte-carlo", n_resamples=999, rng=auxiliary).pvalue


def _joint(data: Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        data.normal(size=(6, 1)),
        data.normal(size=(6, 1)),
        data.normal(size=(6, 1)),
    )


def _dhsic_null(data: Generator, auxiliary: Generator) -> float:
    samples = _joint(data)
    return dhsic(
        *samples, calibration="monte-carlo", n_resamples=999, rng=auxiliary
    ).pvalue


def _multivariance_null(data: Generator, auxiliary: Generator) -> float:
    samples = _joint(data)
    return distance_multivariance(
        *samples, calibration="monte-carlo", n_resamples=999, rng=auxiliary
    ).pvalue


def _energy_power(data: Generator, auxiliary: Generator) -> float:
    return energy_ksamp(
        data.normal(size=(15, 3)),
        data.normal(loc=1.0, scale=1.5, size=(15, 3)),
        calibration="monte-carlo",
        n_resamples=199,
        rng=auxiliary,
    ).pvalue


def _mmd_power(data: Generator, auxiliary: Generator) -> float:
    return mmd_2samp(
        data.normal(size=(15, 3)),
        data.normal(loc=1.0, scale=1.5, size=(15, 3)),
        calibration="monte-carlo",
        n_resamples=199,
        rng=auxiliary,
    ).pvalue


def _dcov_power(data: Generator, auxiliary: Generator) -> float:
    x = data.normal(size=(25, 2))
    y = x + 0.4 * data.normal(size=(25, 2))
    return distance_covariance(
        x, y, calibration="monte-carlo", n_resamples=199, rng=auxiliary
    ).pvalue


def _hsic_power(data: Generator, auxiliary: Generator) -> float:
    x = data.normal(size=30)
    y = x**2 + 0.2 * data.normal(size=30)
    return hsic(x, y, calibration="monte-carlo", n_resamples=199, rng=auxiliary).pvalue


def _xor_samples(data: Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = data.integers(0, 2, size=40).astype(np.float64)
    y = data.integers(0, 2, size=40).astype(np.float64)
    z = np.mod(x + y, 2.0)
    return x, y, z


def _dhsic_power(data: Generator, auxiliary: Generator) -> float:
    return dhsic(
        *_xor_samples(data),
        calibration="monte-carlo",
        n_resamples=199,
        rng=auxiliary,
    ).pvalue


def _multivariance_power(data: Generator, auxiliary: Generator) -> float:
    return distance_multivariance(
        *_xor_samples(data),
        calibration="monte-carlo",
        n_resamples=199,
        rng=auxiliary,
    ).pvalue


SCENARIOS: Final[dict[str, Scenario]] = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            "equaldist.energy",
            2026091101,
            "independent N_2(0,I) samples, n_x=4 and n_y=6; exact 210-label orbit",
            _energy_null,
        ),
        Scenario(
            "equaldist.mmd",
            2026091102,
            "independent N_2(0,I) samples, n_x=4 and n_y=6; exact 210-label orbit",
            _mmd_null,
        ),
        Scenario(
            "independence.distance-covariance",
            2026091104,
            "six independent paired N_2(0,I) vectors; 999 marginal permutations",
            _dcov_null,
        ),
        Scenario(
            "independence.hsic",
            2026091105,
            "six independent paired N_2(0,I) vectors; 999 marginal permutations",
            _hsic_null,
        ),
        Scenario(
            "independence.dhsic",
            2026091106,
            "three independent univariate normal blocks, n=6; 999 marginal permutations",
            _dhsic_null,
        ),
        Scenario(
            "independence.distance-multivariance",
            2026091107,
            "three independent univariate normal blocks, n=6; 999 marginal permutations",
            _multivariance_null,
        ),
    )
}

POWER_SCENARIOS: Final[dict[str, Scenario]] = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            "equaldist.energy",
            2026091201,
            "n_x=n_y=15, p=3; N(0,I) versus N(1,2.25I)",
            _energy_power,
        ),
        Scenario(
            "equaldist.mmd",
            2026091202,
            "n_x=n_y=15, p=3; N(0,I) versus N(1,2.25I)",
            _mmd_power,
        ),
        Scenario(
            "independence.distance-covariance",
            2026091204,
            "n=25, p=q=2; Y=X+0.4 epsilon",
            _dcov_power,
        ),
        Scenario(
            "independence.hsic",
            2026091205,
            "n=30; Y=X^2+0.2 epsilon",
            _hsic_power,
        ),
        Scenario(
            "independence.dhsic",
            2026091206,
            "n=40 Bernoulli blocks; Z=X xor Y (pairwise independent)",
            _dhsic_power,
        ),
        Scenario(
            "independence.distance-multivariance",
            2026091207,
            "n=40 Bernoulli blocks; Z=X xor Y (pairwise independent)",
            _multivariance_power,
        ),
    )
}


def run_scenario(
    scenario: Scenario,
    *,
    replications: int = DEFAULT_REPLICATIONS,
) -> AuditResult:
    if replications <= 0:
        raise ValueError("replications must be positive")
    seed_sequence = np.random.SeedSequence(scenario.seed)
    data_seed, auxiliary_seed = seed_sequence.spawn(2)
    data = np.random.default_rng(data_seed)
    auxiliary = np.random.default_rng(auxiliary_seed)
    levels = np.asarray(LEVELS)
    counts = np.zeros(levels.size, dtype=np.int64)
    for _ in range(replications):
        pvalue = scenario.sampler(data, auxiliary)
        counts += pvalue < levels
    rates = counts / replications
    tolerances = np.asarray(
        [_release_tolerance(level, replications) for level in levels]
    )
    passes = np.abs(rates - levels) <= tolerances
    return AuditResult(
        scenario=scenario.name,
        seed=scenario.seed,
        description=scenario.description,
        replications=replications,
        levels=LEVELS,
        rejection_counts=tuple(int(value) for value in counts),
        rejection_rates=tuple(float(value) for value in rates),
        tolerances=tuple(float(value) for value in tolerances),
        passes=tuple(bool(value) for value in passes),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=tuple(SCENARIOS))
    parser.add_argument("--replications", type=int)
    parser.add_argument("--power", action="store_true")
    args = parser.parse_args(argv)
    scenarios = POWER_SCENARIOS if args.power else SCENARIOS
    replications = args.replications
    if replications is None:
        replications = (
            DEFAULT_POWER_REPLICATIONS if args.power else DEFAULT_REPLICATIONS
        )
    result = run_scenario(scenarios[args.scenario], replications=replications)
    print(json.dumps(asdict(result)))
    return 0 if args.power or all(result.passes) else 1


if __name__ == "__main__":
    raise SystemExit(main())

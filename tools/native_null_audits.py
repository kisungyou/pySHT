"""Reproduce named-seed null gates for pySHT-native mean/covariance tests.

The default scenarios are intentionally expensive and are not part of the
ordinary test suite.  Each replication generates a fresh null dataset and
calls the public function exactly once.  Use ``--replications`` for a short
plumbing check; omitting it runs the documented release design.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Final

import numpy as np
from scipy import stats

from pysht import covariance, mean

type Evaluator = Callable[[np.random.Generator], float]
type StreamSpec = int | tuple[int, int]

LEVELS: Final = (0.01, 0.05, 0.10)
REPLICATIONS: Final = 20_000


@dataclass(frozen=True, slots=True)
class Scenario:
    """One independently replayable null-calibration design."""

    name: str
    description: str
    streams: tuple[StreamSpec, ...]
    replications: int
    evaluate: Evaluator


@dataclass(frozen=True, slots=True)
class Result:
    """Raw rejection counts and release-gate decisions."""

    scenario: str
    stream: str
    replications: int
    counts: tuple[int, ...]
    rates: tuple[float, ...]
    tolerances: tuple[float, ...]
    passes: tuple[bool, ...]
    rejection_rate_intervals_95: tuple[tuple[float, float], ...]
    nominal_in_interval_95: tuple[bool, ...]


def _normal(generator: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    return generator.standard_normal((rows, columns))


def _cq(generator: np.random.Generator) -> float:
    return mean.cq_2samp(
        _normal(generator, 50, 1_000),
        _normal(generator, 60, 1_000),
    ).pvalue


def _li_1samp(generator: np.random.Generator) -> float:
    return mean.li_1samp(_normal(generator, 6, 1_000)).pvalue


def _li_2samp(generator: np.random.Generator) -> float:
    return mean.li_2samp(
        _normal(generator, 6, 1_000),
        _normal(generator, 9, 1_000),
    ).pvalue


def _li_ksamp(generator: np.random.Generator) -> float:
    return mean.li_ksamp(
        _normal(generator, 6, 500),
        _normal(generator, 8, 500),
        _normal(generator, 10, 500),
    ).pvalue


def _li_ksamp_balanced(generator: np.random.Generator) -> float:
    return mean.li_ksamp(
        _normal(generator, 6, 500),
        _normal(generator, 6, 500),
        _normal(generator, 6, 500),
    ).pvalue


def _czz_identity(generator: np.random.Generator) -> float:
    return covariance.czz_identity_1samp(_normal(generator, 100, 100)).pvalue


def _czz_sphericity(generator: np.random.Generator) -> float:
    return covariance.czz_sphericity_1samp(_normal(generator, 100, 100)).pvalue


SCENARIOS: Final = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            "mean.cq_2samp",
            "X~N_1000(0,I), n=50; Y~N_1000(0,I), n=60",
            ((20260839, 0), (20260839, 1), (20260839, 2), (20260839, 3)),
            5_000,
            _cq,
        ),
        Scenario(
            "mean.li_1samp",
            "N_1000(0,I), n=6",
            (20260840,),
            REPLICATIONS,
            _li_1samp,
        ),
        Scenario(
            "mean.li_2samp",
            "independent N_1000(0,I), n=(6,9)",
            (20260841,),
            REPLICATIONS,
            _li_2samp,
        ),
        Scenario(
            "mean.li_ksamp",
            "independent N_500(0,I), n=(6,8,10)",
            (20260842,),
            REPLICATIONS,
            _li_ksamp,
        ),
        Scenario(
            "mean.li_ksamp.balanced",
            "independent N_500(0,I), n=(6,6,6); first group is fixed reference",
            (20260844,),
            REPLICATIONS,
            _li_ksamp_balanced,
        ),
        Scenario(
            "covariance.czz_identity_1samp",
            "N_100(0,I), n=100",
            (20260835,),
            REPLICATIONS,
            _czz_identity,
        ),
        Scenario(
            "covariance.czz_sphericity_1samp",
            "N_100(0,I), n=100",
            (20260836,),
            REPLICATIONS,
            _czz_sphericity,
        ),
    )
}


def release_tolerance(level: float, replications: int) -> float:
    """Return the release regression tolerance, including its 0.005 floor.

    Passing this gate is not evidence of exact finite-sample calibration.
    Separate binomial intervals describe sampling uncertainty in each rate.
    """
    return max(0.005, 4.0 * math.sqrt(level * (1.0 - level) / replications))


def _generator(stream: StreamSpec) -> tuple[np.random.Generator, str]:
    if isinstance(stream, tuple):
        entropy, child = stream
        sequence = np.random.SeedSequence(entropy, spawn_key=(child,))
        return np.random.Generator(np.random.PCG64(sequence)), (
            f"SeedSequence({entropy}).spawn(4)[{child}]"
        )
    return np.random.default_rng(stream), str(stream)


def run(scenario: Scenario, stream: StreamSpec) -> Result:
    """Run one scenario/seed with a fresh PCG64 stream."""
    generator, stream_label = _generator(stream)
    counts = np.zeros(len(LEVELS), dtype=np.int64)
    levels = np.asarray(LEVELS)
    for _ in range(scenario.replications):
        counts += float(scenario.evaluate(generator)) < levels
    raw_counts = tuple(int(value) for value in counts)
    rates = tuple(value / scenario.replications for value in raw_counts)
    tolerances = tuple(
        release_tolerance(level, scenario.replications) for level in LEVELS
    )
    intervals = tuple(
        stats.binomtest(count, scenario.replications).proportion_ci()
        for count in raw_counts
    )
    return Result(
        scenario=scenario.name,
        stream=stream_label,
        replications=scenario.replications,
        counts=raw_counts,
        rates=rates,
        tolerances=tolerances,
        passes=tuple(
            abs(rate - level) <= tolerance
            for rate, level, tolerance in zip(rates, LEVELS, tolerances, strict=True)
        ),
        rejection_rate_intervals_95=tuple(
            (float(interval.low), float(interval.high)) for interval in intervals
        ),
        nominal_in_interval_95=tuple(
            bool(interval.low <= level <= interval.high)
            for level, interval in zip(LEVELS, intervals, strict=True)
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected scenarios and print machine-readable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", choices=tuple(SCENARIOS))
    parser.add_argument("--replications", type=int)
    arguments = parser.parse_args(argv)
    selected = arguments.scenarios or list(SCENARIOS)
    results: list[Result] = []
    for name in selected:
        scenario = SCENARIOS[name]
        if arguments.replications is not None:
            if arguments.replications <= 0:
                parser.error("--replications must be positive")
            scenario = replace(scenario, replications=arguments.replications)
        results.extend(run(scenario, stream) for stream in scenario.streams)
    print(json.dumps([asdict(result) for result in results], indent=2))
    return int(any(not all(result.passes) for result in results))


if __name__ == "__main__":
    raise SystemExit(main())

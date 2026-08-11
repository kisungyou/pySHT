"""Run targeted public-API alternative-power audits for mean procedures.

The simulations in this module are release evidence rather than unit tests.
They intentionally call only public functions and use independent data and
auxiliary-randomness streams for each method.  The default run is moderately
expensive; its complete output is transcribed into the validation ledgers.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Final

import numpy as np
import scipy

from pysht import equaldist, mean
from tools.release_simulations import LEVELS

type FrequentistEvaluator = Callable[[np.random.Generator, np.random.Generator], float]

FIRST_SCENARIO_SEED: Final = 2_026_090_301
DEFAULT_REPLICATIONS: Final = 1_000
RESAMPLING_REPLICATIONS: Final = 300


@dataclass(frozen=True, slots=True)
class PowerScenario:
    """One fully specified alternative-power design."""

    method: str
    design: str
    public_call: str
    replications: int
    evaluate: FrequentistEvaluator


@dataclass(frozen=True, slots=True)
class PowerResult:
    """Rejection counts and rates from one alternative-power design."""

    method: str
    design: str
    public_call: str
    seed: int
    replications: int
    rejection_counts: tuple[int, ...]
    rejection_rates: tuple[float, ...]


def _normal(generator: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    return generator.standard_normal((rows, columns))


def _generator(seed: np.random.SeedSequence) -> np.random.Generator:
    """Construct the audit's pinned PCG64 stream."""
    return np.random.Generator(np.random.PCG64(seed))


def _scenario_registry() -> tuple[PowerScenario, ...]:
    def ttest_1samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.ttest_1samp(data.standard_normal(30) + 0.8).pvalue

    def ttest_2samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.ttest_2samp(
            data.standard_normal(30) + 0.8,
            data.standard_normal(30),
        ).pvalue

    def anova_oneway(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean.anova_oneway(
            data.standard_normal(20),
            data.standard_normal(20) + 0.8,
            data.standard_normal(20) + 1.6,
        ).pvalue

    def hotelling_1samp(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        shift = np.array([0.8, 0.0, 0.0])
        return mean.hotelling_1samp(_normal(data, 40, 3) + shift).pvalue

    def hotelling_2samp(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        shift = np.array([1.0, 0.0, 0.0])
        return mean.hotelling_2samp(
            _normal(data, 30, 3) + shift,
            _normal(data, 30, 3),
        ).pvalue

    def dempster_1samp(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean.dempster_1samp(_normal(data, 30, 40) + 0.35).pvalue

    def dempster_2samp(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean.dempster_2samp(
            _normal(data, 30, 40) + 0.45,
            _normal(data, 30, 40),
        ).pvalue

    def bs_1samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.bs_1samp(_normal(data, 30, 40) + 0.18).pvalue

    def bs_2samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.bs_2samp(
            _normal(data, 30, 40) + 0.25,
            _normal(data, 30, 40),
        ).pvalue

    def sd_1samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.sd_1samp(_normal(data, 30, 40) + 0.18).pvalue

    def sd_2samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.sd_2samp(
            _normal(data, 30, 40) + 0.25,
            _normal(data, 30, 40),
        ).pvalue

    unequal_scale = np.array([1.5, 0.7, 1.2])
    behrens_shift = np.array([0.8, 0.5, 0.0])

    def yao_2samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.yao_2samp(
            _normal(data, 50, 3) + behrens_shift,
            _normal(data, 60, 3) * unequal_scale,
        ).pvalue

    def johansen_2samp(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean.johansen_2samp(
            _normal(data, 50, 3) + behrens_shift,
            _normal(data, 60, 3) * unequal_scale,
        ).pvalue

    def nvm_2samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.nvm_2samp(
            _normal(data, 50, 3) + behrens_shift,
            _normal(data, 60, 3) * unequal_scale,
        ).pvalue

    def ky_2samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.ky_2samp(
            _normal(data, 50, 3) + behrens_shift,
            _normal(data, 60, 3) * unequal_scale,
        ).pvalue

    def schott_ksamp(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        del auxiliary
        return mean.schott_ksamp(
            _normal(data, 20, 40),
            _normal(data, 20, 40) + 0.35,
            _normal(data, 20, 40) - 0.35,
        ).pvalue

    def cph_ksamp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean.cph_ksamp(
            _normal(data, 20, 40),
            _normal(data, 20, 40) + 0.35,
            _normal(data, 20, 40) - 0.35,
        ).pvalue

    def zx_ksamp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        first_scale = np.linspace(0.8, 1.2, 30)
        second_scale = np.linspace(1.0, 1.5, 30)
        third_scale = np.linspace(0.7, 1.1, 30)
        return mean.zx_ksamp(
            _normal(data, 25, 30) * first_scale,
            _normal(data, 30, 30) * second_scale + 0.45,
            _normal(data, 35, 30) * third_scale - 0.45,
        ).pvalue

    def ljw_2samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        return mean.ljw_2samp(
            _normal(data, 20, 40) + 0.45,
            _normal(data, 20, 40),
            rng=auxiliary,
        ).pvalue

    def thulin_2samp(
        data: np.random.Generator, auxiliary: np.random.Generator
    ) -> float:
        return mean.thulin_2samp(
            _normal(data, 12, 25) + 0.8,
            _normal(data, 12, 25),
            n_subspaces=20,
            n_resamples=199,
            rng=auxiliary,
        ).pvalue

    def bg_2samp(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        return equaldist.bg_2samp(
            _normal(data, 15, 5),
            1.8 * _normal(data, 15, 5),
            calibration="monte-carlo",
            n_resamples=199,
            rng=auxiliary,
        ).pvalue

    return (
        PowerScenario(
            "mean.ttest_1samp",
            "N(0.8,1), n=30; null mean 0",
            "mean.ttest_1samp(x)",
            DEFAULT_REPLICATIONS,
            ttest_1samp,
        ),
        PowerScenario(
            "mean.ttest_2samp",
            "N(0.8,1) versus N(0,1), n_x=n_y=30; Welch default",
            "mean.ttest_2samp(x, y)",
            DEFAULT_REPLICATIONS,
            ttest_2samp,
        ),
        PowerScenario(
            "mean.anova_oneway",
            "three N(mu_i,1) groups, n_i=20, means (0,0.8,1.6)",
            "mean.anova_oneway(x, y, z)",
            DEFAULT_REPLICATIONS,
            anova_oneway,
        ),
        PowerScenario(
            "mean.hotelling_1samp",
            "N_3((0.8,0,0),I), n=40; null mean zero",
            "mean.hotelling_1samp(x)",
            DEFAULT_REPLICATIONS,
            hotelling_1samp,
        ),
        PowerScenario(
            "mean.hotelling_2samp",
            "N_3((1,0,0),I) versus N_3(0,I), n_x=n_y=30",
            "mean.hotelling_2samp(x, y)",
            DEFAULT_REPLICATIONS,
            hotelling_2samp,
        ),
        PowerScenario(
            "mean.dempster_1samp",
            "N_40(0.35*1,I), n=30; p>n and null mean zero",
            "mean.dempster_1samp(x)",
            DEFAULT_REPLICATIONS,
            dempster_1samp,
        ),
        PowerScenario(
            "mean.dempster_2samp",
            "N_40(0.45*1,I) versus N_40(0,I), n_x=n_y=30",
            "mean.dempster_2samp(x, y)",
            DEFAULT_REPLICATIONS,
            dempster_2samp,
        ),
        PowerScenario(
            "mean.bs_1samp",
            "N_40(0.18*1,I), n=30; p>n and null mean zero",
            "mean.bs_1samp(x)",
            DEFAULT_REPLICATIONS,
            bs_1samp,
        ),
        PowerScenario(
            "mean.bs_2samp",
            "N_40(0.25*1,I) versus N_40(0,I), n_x=n_y=30",
            "mean.bs_2samp(x, y)",
            DEFAULT_REPLICATIONS,
            bs_2samp,
        ),
        PowerScenario(
            "mean.sd_1samp",
            "N_40(0.18*1,I), n=30; p>n and null mean zero",
            "mean.sd_1samp(x)",
            DEFAULT_REPLICATIONS,
            sd_1samp,
        ),
        PowerScenario(
            "mean.sd_2samp",
            "N_40(0.25*1,I) versus N_40(0,I), n_x=n_y=30",
            "mean.sd_2samp(x, y)",
            DEFAULT_REPLICATIONS,
            sd_2samp,
        ),
        PowerScenario(
            "mean.yao_2samp",
            "N_3((0.8,0.5,0),I), n_x=50, versus N_3(0,D^2), n_y=60, D=(1.5,0.7,1.2)",
            "mean.yao_2samp(x, y)",
            DEFAULT_REPLICATIONS,
            yao_2samp,
        ),
        PowerScenario(
            "mean.johansen_2samp",
            "N_3((0.8,0.5,0),I), n_x=50, versus N_3(0,D^2), n_y=60, D=(1.5,0.7,1.2)",
            "mean.johansen_2samp(x, y)",
            DEFAULT_REPLICATIONS,
            johansen_2samp,
        ),
        PowerScenario(
            "mean.nvm_2samp",
            "N_3((0.8,0.5,0),I), n_x=50, versus N_3(0,D^2), n_y=60, D=(1.5,0.7,1.2)",
            "mean.nvm_2samp(x, y)",
            DEFAULT_REPLICATIONS,
            nvm_2samp,
        ),
        PowerScenario(
            "mean.ky_2samp",
            "N_3((0.8,0.5,0),I), n_x=50, versus N_3(0,D^2), n_y=60, D=(1.5,0.7,1.2)",
            "mean.ky_2samp(x, y)",
            DEFAULT_REPLICATIONS,
            ky_2samp,
        ),
        PowerScenario(
            "mean.schott_ksamp",
            "three N_40(mu_i,I) groups, n_i=20, means (0,0.35*1,-0.35*1)",
            "mean.schott_ksamp(x, y, z)",
            DEFAULT_REPLICATIONS,
            schott_ksamp,
        ),
        PowerScenario(
            "mean.cph_ksamp",
            "three N_40(mu_i,I) groups, n_i=20, means (0,0.35*1,-0.35*1); split-sample default",
            "mean.cph_ksamp(x, y, z)",
            DEFAULT_REPLICATIONS,
            cph_ksamp,
        ),
        PowerScenario(
            "mean.zx_ksamp",
            "three unequal-covariance Gaussian groups, n=(25,30,35), p=30, dense means (0,0.45*1,-0.45*1)",
            "mean.zx_ksamp(x, y, z)",
            DEFAULT_REPLICATIONS,
            zx_ksamp,
        ),
        PowerScenario(
            "mean.ljw_2samp",
            "N_40(0.45*1,I) versus N_40(0,I), n_x=n_y=20; one projection per dataset",
            "mean.ljw_2samp(x, y, rng=auxiliary_rng)",
            DEFAULT_REPLICATIONS,
            ljw_2samp,
        ),
        PowerScenario(
            "mean.thulin_2samp",
            "N_25(0.8*1,I) versus N_25(0,I), n_x=n_y=12; 20 subspaces and 199 permutations",
            "mean.thulin_2samp(x, y, n_subspaces=20, n_resamples=199, rng=auxiliary_rng)",
            RESAMPLING_REPLICATIONS,
            thulin_2samp,
        ),
        PowerScenario(
            "equaldist.bg_2samp",
            "N_5(0,I) versus N_5(0,3.24*I), n_x=n_y=15; 199 label permutations",
            'equaldist.bg_2samp(x, y, calibration="monte-carlo", n_resamples=199, rng=auxiliary_rng)',
            RESAMPLING_REPLICATIONS,
            bg_2samp,
        ),
    )


def run_scenario(scenario: PowerScenario, index: int) -> PowerResult:
    """Run one scenario with its documented independent stream pair."""
    scenario_seed = FIRST_SCENARIO_SEED + index
    data_seed, auxiliary_seed = np.random.SeedSequence(scenario_seed).spawn(2)
    data_rng = _generator(data_seed)
    auxiliary_rng = _generator(auxiliary_seed)
    counts = np.zeros(len(LEVELS), dtype=np.int64)
    levels = np.asarray(LEVELS)
    for _ in range(scenario.replications):
        pvalue = float(scenario.evaluate(data_rng, auxiliary_rng))
        counts += pvalue < levels
    raw_counts = tuple(int(value) for value in counts)
    return PowerResult(
        method=scenario.method,
        design=scenario.design,
        public_call=scenario.public_call,
        seed=scenario_seed,
        replications=scenario.replications,
        rejection_counts=raw_counts,
        rejection_rates=tuple(value / scenario.replications for value in raw_counts),
    )


def _run_lyl_direction() -> dict[str, object]:
    index = len(_scenario_registry())
    scenario_seed = FIRST_SCENARIO_SEED + index
    data_seed, _ = np.random.SeedSequence(scenario_seed).spawn(2)
    data_rng = _generator(data_seed)
    replications = DEFAULT_REPLICATIONS
    null_values = np.empty(replications)
    alternative_values = np.empty(replications)
    shift = np.zeros(60)
    shift[:3] = 1.0
    for replication in range(replications):
        first = _normal(data_rng, 30, 60)
        second = _normal(data_rng, 30, 60)
        null_values[replication] = mean.lyl_2samp(first, second).statistic
        alternative_values[replication] = mean.lyl_2samp(
            first,
            second + shift,
        ).statistic
    return {
        "method": "mean.lyl_2samp",
        "design": (
            "n_x=n_y=30,p=60; matched null is N(0,I) versus N(0,I), "
            "alternative adds 1.0 to the first three coordinates of y"
        ),
        "public_call": "mean.lyl_2samp(x, y)",
        "seed": scenario_seed,
        "replications": replications,
        "null_median_max_log_bf": float(np.median(null_values)),
        "alternative_median_max_log_bf": float(np.median(alternative_values)),
        "paired_alternative_exceeds_null_count": int(
            np.sum(alternative_values > null_values)
        ),
        "paired_alternative_exceeds_null_rate": float(
            np.mean(alternative_values > null_values)
        ),
    }


def _select_scenarios(names: Sequence[str]) -> tuple[tuple[int, PowerScenario], ...]:
    indexed = tuple(enumerate(_scenario_registry()))
    if not names:
        return indexed
    requested = set(names)
    available = {scenario.method for _, scenario in indexed}
    unknown = sorted(requested - available - {"mean.lyl_2samp"})
    if unknown:
        raise ValueError(f"unknown scenarios: {', '.join(unknown)}")
    return tuple(
        (index, scenario) for index, scenario in indexed if scenario.method in requested
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenarios",
        nargs="*",
        help="fully qualified method names; omit to run every scenario",
    )
    arguments = parser.parse_args(argv)
    selected = _select_scenarios(arguments.scenarios)
    payload: dict[str, object] = {
        "levels": LEVELS,
        "stream_contract": (
            "Scenario i has integer seed FIRST_SCENARIO_SEED+i. Its SeedSequence "
            "spawns child 0 for data and child 1 as one persistent PCG64 auxiliary "
            "Generator; both PCG64 streams advance in replication order and reset "
            "between methods."
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pysht": version("pysht"),
        "frequentist": [
            asdict(run_scenario(scenario, index)) for index, scenario in selected
        ],
    }
    if not arguments.scenarios or "mean.lyl_2samp" in arguments.scenarios:
        payload["bayesian"] = _run_lyl_direction()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run targeted public-API power audits for covariance-family procedures."""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Final

import numpy as np
import scipy

from pysht import covariance, mean_covariance
from tools.release_simulations import LEVELS

type FrequentistEvaluator = Callable[[np.random.Generator, np.random.Generator], float]

SEED: Final = 20260902
REPLICATIONS: Final = 2_000


@dataclass(frozen=True, slots=True)
class PowerResult:
    method: str
    design: str
    rejection_counts: tuple[int, ...]
    rejection_rates: tuple[float, ...]


def _normal(generator: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    return generator.standard_normal((rows, columns))


def _evaluators() -> tuple[tuple[str, str, FrequentistEvaluator], ...]:
    def wl1(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        return covariance.wl_1samp(
            math.sqrt(1.5) * _normal(data, 100, 30), rng=auxiliary
        ).pvalue

    def wl2(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        return covariance.wl_2samp(
            _normal(data, 300, 30),
            math.sqrt(1.3) * _normal(data, 360, 30),
            rng=auxiliary,
        ).pvalue

    def lc(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return covariance.lc_2samp(
            _normal(data, 30, 50), math.sqrt(2.0) * _normal(data, 30, 50)
        ).pvalue

    def clx(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        scale = np.ones(30)
        scale[0] = 1.8
        return covariance.clx_2samp(
            _normal(data, 100, 30), _normal(data, 100, 30) * scale
        ).pvalue

    def schott_2001(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        scale = np.array([1.6, 1.0, 1.0])
        return covariance.schott_2001_ksamp(
            _normal(data, 100, 3),
            _normal(data, 100, 3),
            _normal(data, 100, 3) * scale,
        ).pvalue

    def schott_2007(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return covariance.schott_2007_ksamp(
            _normal(data, 100, 50),
            _normal(data, 100, 50),
            math.sqrt(1.5) * _normal(data, 100, 50),
        ).pvalue

    def llzs(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean_covariance.llzs_1samp(_normal(data, 100, 100) + 0.2).pvalue

    def lrt(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        shift = np.array([0.3, 0.0, 0.0])
        return mean_covariance.lrt_1samp(_normal(data, 300, 3) + shift).pvalue

    def hn(data: np.random.Generator, auxiliary: np.random.Generator) -> float:
        del auxiliary
        return mean_covariance.hn_2samp(
            _normal(data, 150, 300), _normal(data, 180, 300) + 0.1
        ).pvalue

    return (
        ("covariance.wl_1samp", "N=100,p=30; Sigma=1.5*I", wl1),
        (
            "covariance.wl_2samp",
            "N1=300,N2=360,p=30; Sigma1=I,Sigma2=1.3*I",
            wl2,
        ),
        (
            "covariance.lc_2samp",
            "n1=n2=30,p=50; Sigma1=I,Sigma2=2*I",
            lc,
        ),
        (
            "covariance.clx_2samp",
            "n1=n2=100,p=30; second-group first SD=1.8",
            clx,
        ),
        (
            "covariance.schott_2001_ksamp",
            "g=3,Ni=100,p=3; third-group SD=(1.6,1,1)",
            schott_2001,
        ),
        (
            "covariance.schott_2007_ksamp",
            "g=3,Ni=100,p=50; third-group Sigma=1.5*I",
            schott_2007,
        ),
        (
            "mean_covariance.llzs_1samp",
            "n=100,p=100; all mean coordinates shifted by 0.2",
            llzs,
        ),
        (
            "mean_covariance.lrt_1samp",
            "n=300,p=3; mean shift=(0.3,0,0)",
            lrt,
        ),
        (
            "mean_covariance.hn_2samp",
            "n1=150,n2=180,p=300; all second-group means shifted by 0.1",
            hn,
        ),
    )


def _run_frequentist() -> list[PowerResult]:
    results: list[PowerResult] = []
    for index, (method, design, evaluate) in enumerate(_evaluators()):
        seed_sequence = np.random.SeedSequence([SEED, index])
        data_seed, auxiliary_seed = seed_sequence.spawn(2)
        data_rng = np.random.default_rng(data_seed)
        auxiliary_rng = np.random.default_rng(auxiliary_seed)
        counts = np.zeros(len(LEVELS), dtype=np.int64)
        for _ in range(REPLICATIONS):
            pvalue = float(evaluate(data_rng, auxiliary_rng))
            counts += pvalue < np.asarray(LEVELS)
        raw_counts = tuple(int(value) for value in counts)
        results.append(
            PowerResult(
                method=method,
                design=design,
                rejection_counts=raw_counts,
                rejection_rates=tuple(value / REPLICATIONS for value in raw_counts),
            )
        )
    return results


def _run_lyl_direction() -> dict[str, object]:
    index = len(_evaluators())
    data_rng = np.random.default_rng(np.random.SeedSequence([SEED, index]))
    null_values = np.empty(REPLICATIONS)
    alternative_values = np.empty(REPLICATIONS)
    for replication in range(REPLICATIONS):
        first = _normal(data_rng, 50, 10)
        null_second = _normal(data_rng, 50, 10)
        latent = _normal(data_rng, 50, 10)
        alternative_second = latent.copy()
        alternative_second[:, 1] = (
            0.7 * latent[:, 0] + math.sqrt(1.0 - 0.7**2) * latent[:, 1]
        )
        null_values[replication] = covariance.lyl_2samp(first, null_second).statistic
        alternative_values[replication] = covariance.lyl_2samp(
            first, alternative_second
        ).statistic
    return {
        "method": "covariance.lyl_2samp",
        "design": (
            "n1=n2=50,p=10; alternative second group has correlation 0.7 "
            "between coordinates 1 and 2"
        ),
        "null_median_max_log_bf": float(np.median(null_values)),
        "alternative_median_max_log_bf": float(np.median(alternative_values)),
        "paired_alternative_exceeds_null_count": int(
            np.sum(alternative_values > null_values)
        ),
        "paired_alternative_exceeds_null_rate": float(
            np.mean(alternative_values > null_values)
        ),
    }


def main() -> int:
    payload = {
        "seed": SEED,
        "replications": REPLICATIONS,
        "levels": LEVELS,
        "stream_contract": (
            "Each method uses SeedSequence([seed, method_index]); frequentist "
            "methods spawn data then auxiliary streams, reset per method, and "
            "advance in replication order."
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pysht": version("pysht"),
        "frequentist": [asdict(result) for result in _run_frequentist()],
        "bayesian": _run_lyl_direction(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

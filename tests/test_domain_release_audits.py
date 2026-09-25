"""Opt-in 20,000-dataset release gates for the domain-expansion methods.

Set ``PYSHT_RUN_DOMAIN_RELEASE_AUDITS=1`` to run these expensive, named-seed
audits.  Independently seeded empirical null banks make the audit tractable;
focused tests separately replay each public call's exact RNG stream and
corrected Monte Carlo p-value.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from pysht.circular import (
    _TIE_RTOL as _CIRCULAR_TIE_RTOL,
)
from pysht.circular import (
    _hermans_rasson_statistic,
    _label_allocations,
    _mww_statistic,
    _rayleigh_statistic,
    _watson_statistic,
)
from pysht.normality import (
    _energy_normality_statistic,
    _henze_zirkler_statistic,
    _standardized_multivariate_sample,
)
from pysht.simplex import (
    _PERMUTATION_TIE_RTOL as _SIMPLEX_TIE_RTOL,
)
from pysht.simplex import (
    _alpha_transform,
    _ehy_simplex_log_statistic,
    _energy_statistic_invariant,
    _simplex_label_allocations,
)
from pysht.uniformity import _ehy_log_statistic

_RUN = os.environ.get("PYSHT_RUN_DOMAIN_RELEASE_AUDITS") == "1"
_REPLICATIONS = 20_000
_REFERENCE_DRAWS = 99_999
_LEVELS = np.array([0.01, 0.05, 0.10])


def _tolerances() -> NDArray[np.float64]:
    # The observed datasets are independent conditional on the empirical
    # reference bank, but their rejection indicators are correlated after the
    # same finite bank is integrated out.  Include both the binomial dataset
    # variance and the order-statistic variance of the reference quantile.
    return np.maximum(
        0.005,
        4.0
        * np.sqrt(
            _LEVELS * (1.0 - _LEVELS) * (1.0 / _REPLICATIONS + 1.0 / _REFERENCE_DRAWS)
        ),
    )


def _rates(
    reference: NDArray[np.float64],
    observed: NDArray[np.float64],
    *,
    lower_tail: bool,
) -> NDArray[np.float64]:
    ordered = np.sort(reference)
    if lower_tail:
        exceedances = np.searchsorted(ordered, observed, side="right")
    else:
        exceedances = ordered.size - np.searchsorted(ordered, observed, side="left")
    pvalues = (exceedances + 1.0) / (ordered.size + 1.0)
    return np.mean(pvalues[:, None] < _LEVELS, axis=0)


def _assert_gate(rates: NDArray[np.float64]) -> None:
    np.testing.assert_array_less(
        np.abs(rates - _LEVELS),
        np.nextafter(_tolerances(), math.inf),
    )


def _mvn_statistics(
    generator: np.random.Generator,
    draws: int,
    *,
    statistic: str,
    sample_size: int = 20,
    dimension: int = 2,
) -> NDArray[np.float64]:
    result = np.empty(draws)
    beta = (sample_size * (2.0 * dimension + 1.0) / 4.0) ** (
        1.0 / (dimension + 4.0)
    ) / math.sqrt(2.0)
    energy_scale = math.sqrt((sample_size - 1.0) / sample_size)
    for index in range(draws):
        standardized, _, _ = _standardized_multivariate_sample(
            generator.standard_normal((sample_size, dimension))
        )
        if statistic == "hz":
            result[index] = _henze_zirkler_statistic(standardized, beta=beta)
        else:
            result[index] = _energy_normality_statistic(energy_scale * standardized)
    return result


@pytest.mark.skipif(not _RUN, reason="opt-in 20,000-dataset release audit")
@pytest.mark.parametrize("statistic", ["hz", "energy"])
def test_multivariate_normality_null_gate(statistic: str) -> None:
    for reference_seed, dataset_seed in (
        (20260911, 20260912),
        (20260931, 20260913),
    ):
        reference = _mvn_statistics(
            np.random.default_rng(reference_seed),
            _REFERENCE_DRAWS,
            statistic=statistic,
        )
        observed = _mvn_statistics(
            np.random.default_rng(dataset_seed),
            _REPLICATIONS,
            statistic=statistic,
        )
        rates = _rates(reference, observed, lower_tail=False)
        print(statistic, reference_seed, dataset_seed, tuple(rates))
        _assert_gate(rates)


def _ehy_rectangular_statistics(
    generator: np.random.Generator,
    draws: int,
    *,
    alpha: float,
) -> NDArray[np.float64]:
    return np.fromiter(
        (
            _ehy_log_statistic(generator.random((25, 2)), alpha=alpha, n_neighbors=2)
            for _ in range(draws)
        ),
        dtype=np.float64,
        count=draws,
    )


def _ehy_simplex_statistics(
    generator: np.random.Generator,
    draws: int,
    *,
    alpha: float,
) -> NDArray[np.float64]:
    return np.fromiter(
        (
            _ehy_simplex_log_statistic(
                generator.dirichlet(np.ones(3), size=25),
                alpha=alpha,
                n_neighbors=2,
            )
            for _ in range(draws)
        ),
        dtype=np.float64,
        count=draws,
    )


@pytest.mark.skipif(not _RUN, reason="opt-in 20,000-dataset release audit")
@pytest.mark.parametrize("domain", ["rectangle", "simplex"])
@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_ehy_null_gate(domain: str, alpha: float) -> None:
    statistic: Callable[..., NDArray[np.float64]] = (
        _ehy_rectangular_statistics
        if domain == "rectangle"
        else _ehy_simplex_statistics
    )
    for reference_seed, dataset_seed in (
        (20260914, 20260915),
        (20260932, 20260916),
    ):
        reference = statistic(
            np.random.default_rng(reference_seed),
            _REFERENCE_DRAWS,
            alpha=alpha,
        )
        observed = statistic(
            np.random.default_rng(dataset_seed),
            _REPLICATIONS,
            alpha=alpha,
        )
        rates = _rates(reference, observed, lower_tail=alpha < 1.0)
        print(domain, alpha, reference_seed, dataset_seed, tuple(rates))
        _assert_gate(rates)


def _circular_statistics(
    generator: np.random.Generator,
    draws: int,
    statistic: Callable[[NDArray[np.float64]], float],
) -> NDArray[np.float64]:
    return np.fromiter(
        (
            statistic(generator.uniform(0.0, 2.0 * math.pi, size=20))
            for _ in range(draws)
        ),
        dtype=np.float64,
        count=draws,
    )


@pytest.mark.skipif(not _RUN, reason="opt-in 20,000-dataset release audit")
@pytest.mark.parametrize(
    "name,statistic",
    [
        ("rayleigh", _rayleigh_statistic),
        ("watson", _watson_statistic),
        ("hermans-rasson", _hermans_rasson_statistic),
    ],
)
def test_circular_uniformity_null_gate(
    name: str, statistic: Callable[[NDArray[np.float64]], float]
) -> None:
    for reference_seed, dataset_seed in (
        (20260917, 20260918),
        (20260933, 20260919),
    ):
        reference = _circular_statistics(
            np.random.default_rng(reference_seed), _REFERENCE_DRAWS, statistic
        )
        observed = _circular_statistics(
            np.random.default_rng(dataset_seed), _REPLICATIONS, statistic
        )
        rates = _rates(reference, observed, lower_tail=False)
        print(name, reference_seed, dataset_seed, tuple(rates))
        _assert_gate(rates)


@pytest.mark.skipif(not _RUN, reason="opt-in 20,000-dataset release audit")
def test_mardia_watson_wheeler_exact_conditional_gate() -> None:
    sizes = (8, 8)
    ranks = np.arange(1, 17, dtype=np.float64)
    cosine = np.cos(2.0 * math.pi * ranks / 16.0)
    sine = np.sin(2.0 * math.pi * ranks / 16.0)
    statistics = np.fromiter(
        (
            _mww_statistic(cosine, sine, labels, sizes)
            for labels in _label_allocations(sizes)
        ),
        dtype=np.float64,
        count=math.comb(16, 8),
    )
    rates = _exact_pvalue_rates(
        statistics,
        relative_tolerance=_CIRCULAR_TIE_RTOL,
        absolute_scale_floor=1.0,
    )
    print("mardia-watson-wheeler", tuple(rates))
    _assert_gate(rates)


def _exact_pvalue_rates(
    statistics: NDArray[np.float64],
    *,
    relative_tolerance: float,
    absolute_scale_floor: float = 0.0,
) -> NDArray[np.float64]:
    ordered = np.sort(statistics)
    thresholds = statistics - relative_tolerance * np.maximum(
        absolute_scale_floor, np.abs(statistics)
    )
    pvalues = (
        statistics.size - np.searchsorted(ordered, thresholds, side="left")
    ) / statistics.size
    return np.mean(pvalues[:, None] < _LEVELS, axis=0)


@pytest.mark.skipif(not _RUN, reason="opt-in 20,000-dataset release audit")
def test_alpha_energy_exact_conditional_gate() -> None:
    rng = np.random.default_rng(20260920)
    pooled = rng.dirichlet(np.ones(3), size=10)
    sizes = (5, 5)
    for alpha in (0.0, 0.4):
        transformed = _alpha_transform(pooled, alpha=alpha)
        distances = np.asarray(
            np.linalg.norm(transformed[:, None, :] - transformed[None, :, :], axis=2),
            dtype=np.float64,
        )
        statistics = np.fromiter(
            (
                _energy_statistic_invariant(distances, labels, sizes)
                for labels in _simplex_label_allocations(sizes)
            ),
            dtype=np.float64,
            count=math.comb(10, 5),
        )
        rates = _exact_pvalue_rates(
            statistics,
            relative_tolerance=_SIMPLEX_TIE_RTOL,
        )
        print("alpha-energy", alpha, tuple(rates))
        _assert_gate(rates)


@pytest.mark.skipif(not _RUN, reason="opt-in named-seed power audit")
def test_named_seed_domain_power_audit() -> None:
    """Check direction and sensitivity on prespecified strong alternatives.

    These are deliberately finite, named-seed power checks rather than claims
    about minimum power over an alternative class.  Shared reference banks are
    used only to make the audit tractable; fixed public calls independently
    verify each statistic and RNG stream in the focused module tests.
    """
    level = 0.05
    alternative_draws = 2_000
    reference_draws = 19_999

    normal_reference_rng = np.random.default_rng(20260921)
    hz_reference = _mvn_statistics(
        normal_reference_rng, reference_draws, statistic="hz", sample_size=40
    )
    normal_reference_rng = np.random.default_rng(20260921)
    energy_reference = _mvn_statistics(
        normal_reference_rng, reference_draws, statistic="energy", sample_size=40
    )
    beta = ((40 * 5.0 / 4.0) ** (1.0 / 6.0)) / math.sqrt(2.0)
    energy_scale = math.sqrt(39.0 / 40.0)
    alternative_rng = np.random.default_rng(20260922)
    hz_rejections = 0
    energy_rejections = 0
    hz_ordered = np.sort(hz_reference)
    energy_ordered = np.sort(energy_reference)
    for _ in range(alternative_draws):
        standardized, _, _ = _standardized_multivariate_sample(
            alternative_rng.standard_t(2.0, size=(40, 2))
        )
        hz_value = _henze_zirkler_statistic(standardized, beta=beta)
        energy_value = _energy_normality_statistic(energy_scale * standardized)
        hz_pvalue = (
            hz_ordered.size - np.searchsorted(hz_ordered, hz_value, side="left") + 1.0
        ) / (hz_ordered.size + 1.0)
        energy_pvalue = (
            energy_ordered.size
            - np.searchsorted(energy_ordered, energy_value, side="left")
            + 1.0
        ) / (energy_ordered.size + 1.0)
        hz_rejections += int(hz_pvalue < level)
        energy_rejections += int(energy_pvalue < level)

    rectangular_reference_rng = np.random.default_rng(20260923)
    rectangular_reference = np.fromiter(
        (
            _ehy_log_statistic(
                rectangular_reference_rng.random((40, 2)),
                alpha=0.5,
                n_neighbors=1,
            )
            for _ in range(reference_draws)
        ),
        dtype=np.float64,
        count=reference_draws,
    )
    rectangular_alternative_rng = np.random.default_rng(20260924)
    rectangular_alternative = np.fromiter(
        (
            _ehy_log_statistic(
                np.clip(
                    rectangular_alternative_rng.normal(0.2, 0.02, size=(40, 2)),
                    0.0,
                    1.0,
                ),
                alpha=0.5,
                n_neighbors=1,
            )
            for _ in range(alternative_draws)
        ),
        dtype=np.float64,
        count=alternative_draws,
    )
    rectangular_pvalues = (
        np.searchsorted(
            np.sort(rectangular_reference), rectangular_alternative, side="right"
        )
        + 1.0
    ) / (reference_draws + 1.0)
    rectangular_rejections = int(np.count_nonzero(rectangular_pvalues < level))

    simplex_reference_rng = np.random.default_rng(20260925)
    simplex_reference = np.fromiter(
        (
            _ehy_simplex_log_statistic(
                simplex_reference_rng.dirichlet(np.ones(3), size=40),
                alpha=0.5,
                n_neighbors=1,
            )
            for _ in range(reference_draws)
        ),
        dtype=np.float64,
        count=reference_draws,
    )
    simplex_alternative_rng = np.random.default_rng(20260926)
    simplex_alternative = np.fromiter(
        (
            _ehy_simplex_log_statistic(
                simplex_alternative_rng.dirichlet(np.full(3, 50.0), size=40),
                alpha=0.5,
                n_neighbors=1,
            )
            for _ in range(alternative_draws)
        ),
        dtype=np.float64,
        count=alternative_draws,
    )
    simplex_pvalues = (
        np.searchsorted(np.sort(simplex_reference), simplex_alternative, side="right")
        + 1.0
    ) / (reference_draws + 1.0)
    simplex_rejections = int(np.count_nonzero(simplex_pvalues < level))

    circular_rejections: dict[str, int] = {}
    for name, circular_statistic in (
        ("rayleigh", _rayleigh_statistic),
        ("watson", _watson_statistic),
        ("hermans-rasson", _hermans_rasson_statistic),
    ):
        reference_rng = np.random.default_rng(20260927)
        reference = _circular_statistics(
            reference_rng, reference_draws, circular_statistic
        )
        alternative_rng = np.random.default_rng(20260928)
        if name == "rayleigh":
            alternatives = (
                alternative_rng.vonmises(0.0, 2.5, 20) for _ in range(alternative_draws)
            )
        else:
            alternatives = (
                np.concatenate(
                    (
                        alternative_rng.vonmises(0.0, 8.0, 10),
                        alternative_rng.vonmises(math.pi, 8.0, 10),
                    )
                )
                for _ in range(alternative_draws)
            )
        ordered = np.sort(reference)
        rejected = 0
        for sample in alternatives:
            value = circular_statistic(sample)
            pvalue = (
                ordered.size - np.searchsorted(ordered, value, side="left") + 1.0
            ) / (ordered.size + 1.0)
            rejected += int(pvalue < level)
        circular_rejections[name] = rejected

    sizes = (5, 5)
    ranks = np.arange(1, 11, dtype=np.float64)
    cosine = np.cos(2.0 * math.pi * ranks / 10.0)
    sine = np.sin(2.0 * math.pi * ranks / 10.0)
    mww_null = np.sort(
        np.fromiter(
            (
                _mww_statistic(cosine, sine, labels, sizes)
                for labels in _label_allocations(sizes)
            ),
            dtype=np.float64,
            count=math.comb(10, 5),
        )
    )
    mww_rng = np.random.default_rng(20260929)
    mww_rejections = 0
    for _ in range(alternative_draws):
        first = np.remainder(mww_rng.vonmises(0.0, 8.0, 5), 2.0 * math.pi)
        second = np.remainder(mww_rng.vonmises(math.pi, 8.0, 5), 2.0 * math.pi)
        pooled = np.concatenate((first, second))
        labels = np.concatenate(
            (np.zeros(5, dtype=np.intp), np.ones(5, dtype=np.intp))
        )[np.argsort(pooled)]
        value = _mww_statistic(cosine, sine, labels, sizes)
        threshold = value - _CIRCULAR_TIE_RTOL * max(1.0, abs(value))
        pvalue = (
            mww_null.size - np.searchsorted(mww_null, threshold, side="left")
        ) / mww_null.size
        mww_rejections += int(pvalue < level)

    alpha_rng = np.random.default_rng(20260930)
    base_labels = np.concatenate(
        (np.zeros(5, dtype=np.intp), np.ones(5, dtype=np.intp))
    )
    alpha_energy_rejections = 0
    for _ in range(alternative_draws):
        first = alpha_rng.dirichlet((20.0, 1.0, 1.0), size=5)
        second = alpha_rng.dirichlet((1.0, 20.0, 1.0), size=5)
        transformed = _alpha_transform(np.vstack((first, second)), alpha=0.4)
        distances = np.linalg.norm(
            transformed[:, None, :] - transformed[None, :, :], axis=2
        )
        observed = _energy_statistic_invariant(distances, base_labels, sizes)
        exceedances = sum(
            _energy_statistic_invariant(distances, labels, sizes)
            >= observed - _SIMPLEX_TIE_RTOL * abs(observed)
            for labels in _simplex_label_allocations(sizes)
        )
        alpha_energy_rejections += int(exceedances / math.comb(10, 5) < level)

    print(
        "named-seed-power",
        hz_rejections,
        energy_rejections,
        rectangular_rejections,
        simplex_rejections,
        circular_rejections,
        mww_rejections,
        alpha_energy_rejections,
    )
    assert hz_rejections == 1_798
    assert energy_rejections == 1_845
    assert rectangular_rejections == 2_000
    assert simplex_rejections == 2_000
    assert circular_rejections == {
        "rayleigh": 2_000,
        "watson": 1_051,
        "hermans-rasson": 2_000,
    }
    assert mww_rejections == 2_000
    assert alpha_energy_rejections == 2_000

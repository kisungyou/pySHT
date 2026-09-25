"""Paper-formula and invariance checks for the pySHT-native mean tests."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from pysht import mean
from pysht.mean import cq_2samp, li_1samp, li_2samp, li_ksamp


def _pairwise_products(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(
        [
            values[i] @ values[j]
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]
    )


def _studentized(products: NDArray[np.float64]) -> tuple[float, float, float]:
    statistic = float(
        np.mean(products) / (np.std(products, ddof=1) / np.sqrt(len(products)))
    )
    degrees = float(len(products) - 1)
    return statistic, float(stats.t.sf(statistic, degrees)), degrees


def _li_vectors(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> NDArray[np.float64]:
    if len(first) > len(second):
        first, second = second, first
    n1, n2 = len(first), len(second)
    ratio = math.sqrt(n1 / n2)
    return (
        first
        - ratio * second[:n1]
        + ratio * np.mean(second[:n1], axis=0)
        - np.mean(second, axis=0)
    )


def _literal_trace_square(values: NDArray[np.float64]) -> float:
    n = len(values)
    first = second = third = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gij = float(values[i] @ values[j])
            first += gij * gij
            for k in range(n):
                if k in (i, j):
                    continue
                second += gij * float(values[j] @ values[k])
                for ell in range(n):
                    if ell in (i, j, k):
                        continue
                    third += gij * float(values[k] @ values[ell])
    return (
        first / (n * (n - 1))
        - 2 * second / (n * (n - 1) * (n - 2))
        + third / (n * (n - 1) * (n - 2) * (n - 3))
    )


def _literal_cross_trace(
    first: NDArray[np.float64], second: NDArray[np.float64]
) -> float:
    n1, n2 = len(first), len(second)
    terms = [0.0, 0.0, 0.0, 0.0]
    for i in range(n1):
        for j in range(n2):
            product = float(first[i] @ second[j])
            terms[0] += product * product
            for k in range(n1):
                if k == i:
                    continue
                terms[1] += product * float(first[k] @ second[j])
                for ell in range(n2):
                    if ell != j:
                        terms[3] += product * float(first[k] @ second[ell])
            for ell in range(n2):
                if ell != j:
                    terms[2] += product * float(first[i] @ second[ell])
    return (
        terms[0] / (n1 * n2)
        - terms[1] / (n1 * n2 * (n1 - 1))
        - terms[2] / (n1 * n2 * (n2 - 1))
        + terms[3] / (n1 * n2 * (n1 - 1) * (n2 - 1))
    )


@pytest.fixture
def samples() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(8317)
    return (
        rng.normal(size=(5, 7)),
        rng.normal(size=(6, 7)),
        rng.normal(size=(7, 7)),
    )


def test_li_one_sample_matches_literal_pair_product_t() -> None:
    rng = np.random.default_rng(72)
    x = rng.normal(size=(6, 11))
    popmean = np.linspace(-0.2, 0.3, x.shape[1])
    expected = _studentized(_pairwise_products(x - popmean))
    actual = li_1samp(x, popmean=popmean)

    np.testing.assert_allclose(actual.statistic, expected[0], rtol=2e-14)
    np.testing.assert_allclose(actual.pvalue, expected[1], rtol=2e-14)
    assert actual.df == expected[2]


def test_li_two_and_k_sample_match_final_paper_formulas(
    samples: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> None:
    x, y, z = samples
    expected_two = _studentized(_pairwise_products(_li_vectors(x, y)))
    actual_two = li_2samp(x, y)
    np.testing.assert_allclose(actual_two.statistic, expected_two[0], rtol=2e-14)
    np.testing.assert_allclose(actual_two.pvalue, expected_two[1], rtol=2e-14)

    first = x
    products = _pairwise_products(_li_vectors(first, y))
    products += _pairwise_products(_li_vectors(first, z))
    expected_k = _studentized(products)
    actual_k = li_ksamp(x, y, z)
    np.testing.assert_allclose(actual_k.statistic, expected_k[0], rtol=2e-14)
    np.testing.assert_allclose(actual_k.pvalue, expected_k[1], rtol=2e-14)


def test_li_invariances_and_boundaries(
    samples: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> None:
    x, y, z = samples
    shift = np.linspace(-3.0, 2.0, x.shape[1])
    baseline = li_2samp(x, y)
    transformed = li_2samp(1e120 * (x + shift), 1e120 * (y + shift))
    swapped = li_2samp(y, x)
    np.testing.assert_allclose(transformed.statistic, baseline.statistic, rtol=2e-13)
    np.testing.assert_allclose(swapped.statistic, baseline.statistic, rtol=2e-13)
    expected_k = li_ksamp(x, y, z)
    for permuted_groups in ((x, z, y), (y, x, z), (z, y, x)):
        assert li_ksamp(*permuted_groups) == expected_k
    with pytest.raises(ValueError, match="at least 3 observations"):
        li_1samp(x[:2])
    with pytest.raises(ValueError, match="variance"):
        li_1samp(np.zeros((4, 3)))


def test_li_k_sample_balanced_design_matches_theorem_five_construction() -> None:
    rng = np.random.default_rng(123)
    groups = tuple(rng.normal(size=(6, 30)) for _ in range(3))
    products = _pairwise_products(_li_vectors(groups[0], groups[1]))
    products += _pairwise_products(_li_vectors(groups[0], groups[2]))
    expected = _studentized(products)
    actual = li_ksamp(*groups)
    swapped_nonreferences = li_ksamp(groups[0], groups[2], groups[1])

    np.testing.assert_allclose(actual.statistic, expected[0], rtol=3e-14)
    np.testing.assert_allclose(actual.pvalue, expected[1], rtol=3e-14)
    assert actual.df == expected[2]
    assert swapped_nonreferences == actual


def test_li_k_sample_tied_reference_choice_is_explicitly_order_sensitive() -> None:
    rng = np.random.default_rng(124)
    groups = tuple(rng.normal(size=(6, 30)) for _ in range(3))
    first_reference = li_ksamp(*groups)
    second_reference = li_ksamp(groups[1], groups[0], groups[2])
    assert not math.isclose(
        first_reference.statistic,
        second_reference.statistic,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )


def test_cq_matches_literal_order_four_formula(
    samples: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> None:
    x, y, _ = samples
    n1, n2 = len(x), len(y)
    within1 = sum(
        float(x[i] @ x[j]) for i in range(n1) for j in range(n1) if i != j
    ) / (n1 * (n1 - 1))
    within2 = sum(
        float(y[i] @ y[j]) for i in range(n2) for j in range(n2) if i != j
    ) / (n2 * (n2 - 1))
    cross = 2 * sum(float(a @ b) for a in x for b in y) / (n1 * n2)
    xc = x - np.mean(x, axis=0)
    yc = y - np.mean(y, axis=0)
    variance = (
        2 * _literal_trace_square(xc) / (n1 * (n1 - 1))
        + 2 * _literal_trace_square(yc) / (n2 * (n2 - 1))
        + 4 * _literal_cross_trace(xc, yc) / (n1 * n2)
    )
    expected = (within1 + within2 - cross) / math.sqrt(variance)
    actual = cq_2samp(x, y)
    np.testing.assert_allclose(actual.statistic, expected, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(actual.pvalue, stats.norm.sf(expected), rtol=2e-13)


def test_cq_translation_scale_row_and_group_invariance(
    samples: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
) -> None:
    x, y, _ = samples
    baseline = cq_2samp(x, y)
    shift = np.arange(x.shape[1], dtype=float) * 10.0
    transformed = cq_2samp(1e100 * (x + shift), 1e100 * (y + shift))
    permuted = cq_2samp(x[::-1], y[[2, 0, 5, 3, 1, 4]])
    swapped = cq_2samp(y, x)
    for result in (transformed, permuted, swapped):
        np.testing.assert_allclose(result.statistic, baseline.statistic, rtol=3e-12)
        np.testing.assert_allclose(result.pvalue, baseline.pvalue, rtol=3e-12)


def test_xy_seeded_bootstrap_is_canonical_local_and_batched() -> None:
    rng = np.random.default_rng(991)
    x = rng.normal(size=(18, 35))
    y = rng.normal(size=(22, 35))
    state = np.random.get_state()
    started = time.perf_counter()
    baseline = mean._xy_2samp(x, y, n_resamples=513, rng=314)
    elapsed = time.perf_counter() - started
    permuted = mean._xy_2samp(x[::-1], y[::-1], n_resamples=513, rng=314)
    swapped = mean._xy_2samp(y, x, n_resamples=513, rng=314)
    assert baseline == permuted == swapped
    assert baseline.n_resamples == 513
    assert baseline.pvalue == (baseline.exceedances + 1) / 514
    assert elapsed < 5.0
    after = np.random.get_state()
    assert all(
        np.array_equal(left, right) for left, right in zip(state, after, strict=True)
    )


def test_xy_matches_fixed_multiplier_literal_oracle() -> None:
    rng = np.random.default_rng(812)
    x = rng.normal(size=(5, 6))
    y = rng.normal(size=(7, 6))
    seed = 44
    resamples = 19
    actual = mean._xy_2samp(x, y, n_resamples=resamples, rng=seed)

    # The public implementation canonicalizes the pair and removes a shared
    # anchor before applying one common scale. Reproduce that contract with a
    # deliberately literal one-draw loop.
    first, second = mean._canonical_randomization_pair(x, y)
    (first, second), scale = mean._globally_scaled_anchored_groups((first, second))
    n1, n2 = len(first), len(second)
    observed_scaled = math.sqrt(n1) * float(
        np.max(np.abs(np.mean(first, axis=0) - np.mean(second, axis=0)))
    )
    first -= np.mean(first, axis=0)
    second -= np.mean(second, axis=0)
    generator = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(resamples):
        multipliers = generator.standard_normal(n1 + n2)
        bootstrap = np.max(
            np.abs(
                multipliers[:n1] @ first / math.sqrt(n1)
                - math.sqrt(n1 / n2) * (multipliers[n1:] @ second / math.sqrt(n2))
            )
        )
        exceedances += int(bootstrap >= observed_scaled)

    np.testing.assert_allclose(actual.statistic, observed_scaled * scale)
    assert actual.exceedances == exceedances
    assert actual.pvalue == (exceedances + 1) / (resamples + 1)


def test_xy_rejects_invalid_controls() -> None:
    x = np.eye(4)
    with pytest.raises(TypeError, match="integer"):
        mean._xy_2samp(x, x, n_resamples=True)
    with pytest.raises(TypeError, match="rng"):
        mean._xy_2samp(x, x, rng=object())


def test_xy_remains_private_until_its_monte_carlo_gate_passes() -> None:
    assert "xy_2samp" not in mean.__all__
    assert not hasattr(mean, "xy_2samp")
    assert hasattr(mean, "_xy_2samp")
    assert not hasattr(mean, "lyl_2samp")

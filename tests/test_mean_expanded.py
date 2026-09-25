"""Independent formula and invariant checks for expanded mean procedures."""

from __future__ import annotations

import inspect
import math
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import optimize, stats

from pysht import mean
from pysht._results import BayesFactorTestResult, ResamplingTestResult


def _covariance(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = values - np.mean(values, axis=0)
    return np.asarray(centered.T @ centered / (values.shape[0] - 1), dtype=np.float64)


def _trace_square(matrix: NDArray[np.float64]) -> float:
    return float(np.trace(matrix @ matrix))


def _diagnostics(result: object) -> dict[str, bool | int | float | str]:
    return dict(result.diagnostics)  # type: ignore[attr-defined]


def _shared_feature_anchor(
    groups: tuple[NDArray[np.float64], ...],
) -> tuple[NDArray[np.float64], ...]:
    origin = np.minimum.reduce(tuple(np.min(group, axis=0) for group in groups))
    return tuple(group - origin for group in groups)


def _sorted_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Independent reconstruction of the public seeded-data ordering."""
    keys = tuple(values[:, column] for column in range(values.shape[1] - 1, -1, -1))
    return np.ascontiguousarray(values[np.lexsort(keys)])


def _canonical_randomization_pair(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    first = _sorted_rows(x)
    second = _sorted_rows(y)
    if first.shape[0] != second.shape[0]:
        return (first, second) if first.shape[0] < second.shape[0] else (second, first)
    unequal = np.flatnonzero(first.ravel() != second.ravel())
    if unequal.size and second.ravel()[unequal[0]] < first.ravel()[unequal[0]]:
        return second, first
    return first, second


def _bs_literal(
    difference: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    multiplier: float,
    within_df: int,
) -> float:
    trace = float(np.trace(covariance))
    trace_sigma2 = (
        within_df
        * (within_df + 1)
        / ((within_df - 1) * (within_df + 2))
        * (_trace_square(covariance) - trace**2 / within_df)
    )
    return (multiplier * float(difference @ difference) - trace) / math.sqrt(
        2.0 * trace_sigma2
    )


def _sd_literal(
    difference: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    multiplier: float,
    within_df: int,
) -> float:
    diagonal = np.diag(covariance)
    inverse_sd = 1.0 / np.sqrt(diagonal)
    correlation = covariance * np.outer(inverse_sd, inverse_sd)
    trace_r2 = _trace_square(correlation)
    p = covariance.shape[0]
    numerator = multiplier * float(np.sum(difference**2 / diagonal)) - within_df * p / (
        within_df - 2
    )
    denominator = math.sqrt(
        2.0 * (trace_r2 - p**2 / within_df) * (1.0 + trace_r2 / p**1.5)
    )
    return float(numerator / denominator)


def _projected_t2(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    projection: NDArray[np.float64],
) -> float:
    nx = x.shape[0]
    ny = y.shape[0]
    within_df = nx + ny - 2
    pooled = ((nx - 1) * _covariance(x) + (ny - 1) * _covariance(y)) / within_df
    difference = np.mean(x, axis=0) - np.mean(y, axis=0)
    projected_difference = projection.T @ difference
    projected_covariance = projection.T @ pooled @ projection
    return float(
        nx
        * ny
        / (nx + ny)
        * float(
            projected_difference
            @ np.linalg.solve(projected_covariance, projected_difference)
        )
    )


class TestTraceBasedMeanTests:
    @pytest.fixture
    def one_sample(self) -> NDArray[np.float64]:
        return np.random.default_rng(1001).normal(size=(14, 9))

    @pytest.fixture
    def two_samples(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rng = np.random.default_rng(1002)
        return rng.normal(size=(13, 8)), rng.normal(size=(17, 8))

    @pytest.mark.parametrize("rows,features", [(8, 20), (30, 3)])
    def test_adaptive_trace_moments_match_dense_literal(
        self, rows: int, features: int
    ) -> None:
        rng = np.random.default_rng(1049 + rows + features)
        first = rng.normal(size=(rows, features))
        second = rng.normal(size=(rows + 3, features))
        covariance = _covariance(first)
        trace, trace_squared = mean._covariance_trace_moments(first)
        assert trace == pytest.approx(float(np.trace(covariance)), rel=3e-14)
        assert trace_squared == pytest.approx(_trace_square(covariance), rel=3e-14)

        expected_cross = float(np.trace(covariance @ _covariance(second)))
        actual_cross = mean._trace_sample_covariance_product(first, second)
        assert actual_cross == pytest.approx(expected_cross, rel=5e-14)

    def test_dempster_one_sample_matches_trace_f_formula(
        self, one_sample: NDArray[np.float64]
    ) -> None:
        null = np.linspace(-0.2, 0.2, one_sample.shape[1])
        result = mean.dempster_1samp(one_sample, popmean=null)
        n = one_sample.shape[0]
        within_df = n - 1
        difference = np.mean(one_sample, axis=0) - null
        covariance = _covariance(one_sample)
        trace = float(np.trace(covariance))
        trace_sigma2 = (
            within_df**2
            / ((within_df - 1) * (within_df + 2))
            * (_trace_square(covariance) - trace**2 / within_df)
        )
        effective_rank = trace**2 / trace_sigma2
        expected_df = (
            float(math.floor(effective_rank)),
            float(math.floor(within_df * effective_rank)),
        )
        expected_statistic = n * float(difference @ difference) / trace

        assert result.statistic == pytest.approx(expected_statistic, rel=2e-14)
        assert result.df == expected_df
        assert result.pvalue == pytest.approx(
            stats.f.sf(expected_statistic, *expected_df), rel=2e-14
        )

    def test_dempster_two_sample_repairs_legacy_normal_tail(
        self,
        two_samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        x, y = two_samples
        result = mean.dempster_2samp(x, y)
        nx, ny = x.shape[0], y.shape[0]
        within_df = nx + ny - 2
        pooled = ((nx - 1) * _covariance(x) + (ny - 1) * _covariance(y)) / within_df
        difference = np.mean(x, axis=0) - np.mean(y, axis=0)
        trace = float(np.trace(pooled))
        expected_statistic = (
            nx * ny / (nx + ny) * float(difference @ difference) / trace
        )
        trace_sigma2 = (
            within_df**2
            / ((within_df - 1) * (within_df + 2))
            * (_trace_square(pooled) - trace**2 / within_df)
        )
        effective_rank = trace**2 / trace_sigma2
        expected_df = (
            float(math.floor(effective_rank)),
            float(math.floor(within_df * effective_rank)),
        )

        assert result.statistic == pytest.approx(expected_statistic, rel=2e-14)
        assert result.df == expected_df
        assert result.pvalue == pytest.approx(
            stats.f.sf(expected_statistic, *expected_df), rel=2e-14
        )
        assert not math.isclose(
            result.pvalue,
            float(stats.norm.sf(expected_statistic)),
            rel_tol=1e-5,
            abs_tol=1e-8,
        )

    def test_bai_saranadasa_formulas_and_corrected_one_sample_factor(
        self,
        one_sample: NDArray[np.float64],
        two_samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        one_covariance = _covariance(one_sample)
        one_difference = np.mean(one_sample, axis=0)
        one_expected = _bs_literal(
            one_difference,
            one_covariance,
            multiplier=float(one_sample.shape[0]),
            within_df=one_sample.shape[0] - 1,
        )
        one_result = mean.bs_1samp(one_sample)

        x, y = two_samples
        nx, ny = x.shape[0], y.shape[0]
        within_df = nx + ny - 2
        pooled = ((nx - 1) * _covariance(x) + (ny - 1) * _covariance(y)) / within_df
        two_expected = _bs_literal(
            np.mean(x, axis=0) - np.mean(y, axis=0),
            pooled,
            multiplier=nx * ny / (nx + ny),
            within_df=within_df,
        )
        two_result = mean.bs_2samp(x, y)

        assert one_result.statistic == pytest.approx(one_expected, rel=2e-14)
        assert one_result.pvalue == pytest.approx(stats.norm.sf(one_expected))
        assert two_result.statistic == pytest.approx(two_expected, rel=2e-14)
        assert two_result.pvalue == pytest.approx(stats.norm.sf(two_expected))

        n = one_sample.shape[0] - 1
        trace = float(np.trace(one_covariance))
        correction = _trace_square(one_covariance) - trace**2 / n
        legacy_denominator = math.sqrt(2.0 * n / (n - 1) * correction)
        legacy_statistic = (
            one_sample.shape[0] * float(one_difference @ one_difference) - trace
        ) / legacy_denominator
        assert not math.isclose(one_result.statistic, legacy_statistic, rel_tol=1e-10)

    def test_srivastava_du_formulas_and_feature_scale_invariance(
        self,
        one_sample: NDArray[np.float64],
        two_samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        one_expected = _sd_literal(
            np.mean(one_sample, axis=0),
            _covariance(one_sample),
            multiplier=float(one_sample.shape[0]),
            within_df=one_sample.shape[0] - 1,
        )
        one_result = mean.sd_1samp(one_sample)
        assert one_result.statistic == pytest.approx(one_expected, rel=3e-14)

        x, y = two_samples
        nx, ny = x.shape[0], y.shape[0]
        within_df = nx + ny - 2
        pooled = ((nx - 1) * _covariance(x) + (ny - 1) * _covariance(y)) / within_df
        two_expected = _sd_literal(
            np.mean(x, axis=0) - np.mean(y, axis=0),
            pooled,
            multiplier=nx * ny / (nx + ny),
            within_df=within_df,
        )
        two_result = mean.sd_2samp(x, y)
        feature_scales = np.geomspace(0.1, 10.0, x.shape[1])
        scaled = mean.sd_2samp(x * feature_scales, y * feature_scales)

        assert two_result.statistic == pytest.approx(two_expected, rel=3e-14)
        assert scaled.statistic == pytest.approx(two_result.statistic, rel=2e-13)
        assert scaled.pvalue == pytest.approx(two_result.pvalue, rel=2e-13)

    def test_srivastava_du_preserves_features_across_float64_exponents(
        self,
        one_sample: NDArray[np.float64],
        two_samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        """Coordinate-scale invariance must also hold computationally."""
        scales = np.geomspace(1.0e-300, 1.0e300, one_sample.shape[1])
        null = np.linspace(-0.2, 0.2, one_sample.shape[1])
        one_baseline = mean.sd_1samp(one_sample, popmean=null)
        one_scaled = mean.sd_1samp(
            one_sample * scales,
            popmean=null * scales,
        )
        assert one_scaled.statistic == pytest.approx(one_baseline.statistic, rel=3e-13)
        assert one_scaled.pvalue == pytest.approx(one_baseline.pvalue, rel=3e-13)

        x, y = two_samples
        two_scales = np.geomspace(1.0e-300, 1.0e300, x.shape[1])
        two_baseline = mean.sd_2samp(x, y)
        two_scaled = mean.sd_2samp(x * two_scales, y * two_scales)
        assert two_scaled.statistic == pytest.approx(two_baseline.statistic, rel=3e-13)
        assert two_scaled.pvalue == pytest.approx(two_baseline.pvalue, rel=3e-13)

    @pytest.mark.parametrize(
        "function",
        [
            mean.dempster_2samp,
            mean.bs_2samp,
            mean.sd_2samp,
        ],
    )
    def test_two_sample_trace_tests_are_group_exchange_invariant(
        self,
        function: object,
        two_samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        x, y = two_samples
        direct = function(x, y)  # type: ignore[operator]
        swapped = function(y, x)  # type: ignore[operator]
        assert swapped.statistic == pytest.approx(direct.statistic, rel=2e-13)
        assert swapped.pvalue == pytest.approx(direct.pvalue, rel=2e-13)

    def test_trace_tests_do_not_materialize_feature_covariance(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rng = np.random.default_rng(1050)
        x = rng.normal(size=(12, 5_000))
        y = rng.normal(size=(14, 5_000))

        def forbidden_covariance(_values: NDArray[np.float64]) -> NDArray[np.float64]:
            raise AssertionError("dense feature covariance allocation")

        monkeypatch.setattr(mean, "_sample_covariance", forbidden_covariance)
        for function in (
            mean.dempster_1samp,
            mean.bs_1samp,
            mean.sd_1samp,
        ):
            result = function(x)
            assert math.isfinite(result.statistic)
            assert math.isfinite(result.pvalue)
        for function in (
            mean.dempster_2samp,
            mean.bs_2samp,
            mean.sd_2samp,
        ):
            result = function(x, y)
            assert math.isfinite(result.statistic)
            assert math.isfinite(result.pvalue)

    @pytest.mark.parametrize("shift", [1.0e8, 1.0e14])
    @pytest.mark.parametrize(
        "function",
        [mean.dempster_1samp, mean.bs_1samp, mean.sd_1samp, mean.hotelling_1samp],
    )
    def test_one_sample_tests_center_before_scaling(
        self,
        function: object,
        shift: float,
    ) -> None:
        rng = np.random.default_rng(1099)
        values = rng.normal(size=(18, 5))
        translated = values + shift
        # Quantize the reference exactly as the translated float64 input was
        # quantized.  Any remaining discrepancy is algorithmic, not a claim
        # that information absent from the input can be recovered.
        centered_reference = translated - shift
        actual = function(  # type: ignore[operator]
            translated,
            popmean=np.full(values.shape[1], shift),
        )
        expected = function(centered_reference)  # type: ignore[operator]
        assert actual.statistic == pytest.approx(expected.statistic, rel=2e-13)
        assert actual.pvalue == pytest.approx(expected.pvalue, rel=2e-13)

    @pytest.mark.parametrize("shift", [1.0e8, 1.0e14])
    def test_one_sample_t_centers_before_scaling(self, shift: float) -> None:
        values = np.random.default_rng(1100).normal(size=18)
        translated = values + shift
        expected = mean.ttest_1samp(translated - shift)
        actual = mean.ttest_1samp(translated, popmean=shift)
        assert actual.statistic == pytest.approx(expected.statistic, rel=2e-13)
        assert actual.pvalue == pytest.approx(expected.pvalue, rel=2e-13)


class TestMultivariateBehrensFisher:
    @pytest.fixture
    def samples(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rng = np.random.default_rng(2001)
        transform_x = np.array([[1.0, 0.4, 0.1], [0.0, 1.3, -0.2], [0.0, 0.0, 0.8]])
        transform_y = np.array([[1.8, -0.3, 0.0], [0.0, 0.7, 0.2], [0.0, 0.0, 1.2]])
        x = rng.normal(size=(18, 3)) @ transform_x
        y = rng.normal(size=(23, 3)) @ transform_y + np.array([0.2, -0.1, 0.3])
        return x, y

    def test_all_four_methods_match_literal_matrix_formulas(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        nx, ny = x.shape[0], y.shape[0]
        p = x.shape[1]
        first = _covariance(x) / nx
        second = _covariance(y) / ny
        total = first + second
        difference = np.mean(x, axis=0) - np.mean(y, axis=0)
        solved = np.linalg.solve(total, difference)
        t2 = float(difference @ solved)

        first_fraction = float(solved @ first @ solved) / t2
        second_fraction = float(solved @ second @ solved) / t2
        yao_v = 1.0 / (first_fraction**2 / (nx - 1) + second_fraction**2 / (ny - 1))
        yao_df2 = yao_v - p + 1
        yao_f = t2 * yao_df2 / (yao_v * p)

        nvm_v = (_trace_square(total) + float(np.trace(total)) ** 2) / (
            (_trace_square(first) + float(np.trace(first)) ** 2) / (nx - 1)
            + (_trace_square(second) + float(np.trace(second)) ** 2) / (ny - 1)
        )
        nvm_df2 = nvm_v - p + 1

        total_inverse = np.linalg.inv(total)
        first_weight = first @ total_inverse
        second_weight = second @ total_inverse
        ky_denominator = (
            _trace_square(first_weight) + float(np.trace(first_weight)) ** 2
        ) / (nx - 1) + (
            _trace_square(second_weight) + float(np.trace(second_weight)) ** 2
        ) / (ny - 1)
        ky_v = p * (p + 1) / ky_denominator
        ky_df2 = ky_v - p + 1

        identity = np.eye(p)
        inverse_first = np.linalg.inv(first)
        inverse_second = np.linalg.inv(second)
        inverse_sum = inverse_first + inverse_second
        first_a = identity - np.linalg.solve(inverse_sum, inverse_first)
        second_a = identity - np.linalg.solve(inverse_sum, inverse_second)
        johansen_d = 0.5 * (
            (_trace_square(first_a) + float(np.trace(first_a)) ** 2) / (nx - 1)
            + (_trace_square(second_a) + float(np.trace(second_a)) ** 2) / (ny - 1)
        )
        johansen_v = p * (p + 2) / (3 * johansen_d)
        johansen_q = p + 2 * johansen_d - 6 * johansen_d / (p + 2)

        yao = mean.yao_2samp(x, y)
        nvm = mean.nvm_2samp(x, y)
        ky = mean.ky_2samp(x, y)
        johansen = mean.johansen_2samp(x, y)

        assert yao.statistic == pytest.approx(t2, rel=3e-14)
        assert yao.df == pytest.approx((float(p), yao_df2))
        assert yao.pvalue == pytest.approx(stats.f.sf(yao_f, p, yao_df2))
        assert nvm.df == pytest.approx((float(p), nvm_df2))
        assert nvm.pvalue == pytest.approx(
            stats.f.sf(t2 * nvm_df2 / (nvm_v * p), p, nvm_df2)
        )
        assert ky.df == pytest.approx((float(p), ky_df2))
        assert ky.pvalue == pytest.approx(
            stats.f.sf(t2 * ky_df2 / (ky_v * p), p, ky_df2)
        )
        assert johansen.df == pytest.approx((float(p), johansen_v))
        assert johansen.pvalue == pytest.approx(
            stats.f.sf(t2 / johansen_q, p, johansen_v)
        )

    @pytest.mark.parametrize(
        "function",
        [mean.yao_2samp, mean.nvm_2samp, mean.ky_2samp, mean.johansen_2samp],
    )
    def test_exchange_and_common_scale_invariance(
        self,
        function: object,
        samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        x, y = samples
        direct = function(x, y)  # type: ignore[operator]
        swapped = function(y, x)  # type: ignore[operator]
        scaled = function(1e120 * x, 1e120 * y)  # type: ignore[operator]
        assert swapped.statistic == pytest.approx(direct.statistic, rel=5e-13)
        assert swapped.pvalue == pytest.approx(direct.pvalue, rel=5e-13)
        assert scaled.statistic == pytest.approx(direct.statistic, rel=5e-13)
        assert scaled.pvalue == pytest.approx(direct.pvalue, rel=5e-13)

    @pytest.mark.parametrize(
        "function",
        [mean.yao_2samp, mean.ky_2samp, mean.johansen_2samp],
    )
    def test_affine_invariant_methods_preserve_extreme_column_units(
        self,
        function: object,
        samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        x, y = samples
        scales = np.geomspace(1.0e-150, 1.0e150, x.shape[1])
        baseline = function(x, y)  # type: ignore[operator]
        rescaled = function(x * scales, y * scales)  # type: ignore[operator]
        assert rescaled.statistic == pytest.approx(baseline.statistic, rel=8e-13)
        assert rescaled.pvalue == pytest.approx(baseline.pvalue, rel=8e-13)

    def test_nvm_keeps_affine_invariant_t2_at_extreme_column_units(
        self,
        samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        x, y = samples
        scales = np.geomspace(1.0e-150, 1.0e150, x.shape[1])
        baseline = mean.nvm_2samp(x, y)
        rescaled = mean.nvm_2samp(x * scales, y * scales)
        assert rescaled.statistic == pytest.approx(baseline.statistic, rel=8e-13)
        assert math.isfinite(rescaled.pvalue)
        assert rescaled.df is not None
        # The NVM trace-based degrees-of-freedom approximation itself is not
        # affine-invariant, so equality of p-values is deliberately not
        # asserted here.

    @pytest.mark.parametrize(
        "function",
        [mean.yao_2samp, mean.nvm_2samp, mean.ky_2samp],
    )
    def test_univariate_limit_matches_welch(
        self,
        function: object,
    ) -> None:
        x = np.array([0.1, 1.4, -0.2, 2.3, 0.8, 1.1])[:, None]
        y = np.array([-0.5, 0.7, 1.8, 0.3, -1.1, 0.2, 0.9])[:, None]
        actual = function(x, y)  # type: ignore[operator]
        expected = stats.ttest_ind(x[:, 0], y[:, 0], equal_var=False)
        assert actual.statistic == pytest.approx(expected.statistic**2)
        assert actual.pvalue == pytest.approx(expected.pvalue)

    @pytest.mark.parametrize(
        "function",
        [mean.yao_2samp, mean.nvm_2samp, mean.ky_2samp, mean.johansen_2samp],
    )
    def test_zero_difference_has_unit_tail_probability(
        self,
        function: object,
        samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        x, _ = samples
        result = function(x, x.copy())  # type: ignore[operator]
        assert result.statistic == pytest.approx(0.0, abs=1e-28)
        assert result.pvalue == 1.0


class TestLargeLocationAnchoring:
    @pytest.fixture
    def low_dimensional_pair(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rng = np.random.default_rng(2501)
        return rng.normal(size=(12, 3)), 1.2 * rng.normal(size=(15, 3))

    @pytest.mark.parametrize(
        "function",
        [
            mean.hotelling_2samp,
            mean.dempster_2samp,
            mean.bs_2samp,
            mean.sd_2samp,
            mean.yao_2samp,
            mean.johansen_2samp,
            mean.nvm_2samp,
            mean.ky_2samp,
        ],
    )
    def test_deterministic_two_sample_tests_use_shared_original_anchor(
        self,
        function: object,
        low_dimensional_pair: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        x, y = low_dimensional_pair
        translated = (x + 1.0e14, y + 1.0e14)
        anchored = _shared_feature_anchor(translated)
        actual = function(*translated)  # type: ignore[operator]
        expected = function(*anchored)  # type: ignore[operator]
        assert actual.statistic == pytest.approx(expected.statistic, rel=3e-12)
        assert actual.pvalue == pytest.approx(expected.pvalue, rel=3e-12)

    @pytest.mark.parametrize(
        "paired,equal_var",
        [(False, False), (False, True), (True, False)],
    )
    def test_two_sample_t_uses_shared_or_pairwise_anchor(
        self,
        paired: bool,
        equal_var: bool,
    ) -> None:
        rng = np.random.default_rng(2502)
        x = rng.normal(size=15)
        y = rng.normal(size=15 if paired else 18)
        translated = (x + 1.0e14, y + 1.0e14)
        anchored = _shared_feature_anchor(translated)
        kwargs = {"paired": paired, "equal_var": equal_var}
        actual = mean.ttest_2samp(*translated, **kwargs)
        expected = mean.ttest_2samp(*anchored, **kwargs)
        assert actual.statistic == pytest.approx(expected.statistic, rel=3e-13)
        assert actual.pvalue == pytest.approx(expected.pvalue, rel=3e-13)

    def test_anova_uses_shared_original_anchor(self) -> None:
        rng = np.random.default_rng(2503)
        groups = (
            rng.normal(size=12) + 1.0e14,
            rng.normal(size=15) + 1.0e14,
            rng.normal(size=18) + 1.0e14,
        )
        anchored = _shared_feature_anchor(groups)
        actual = mean.anova_oneway(*groups)
        expected = mean.anova_oneway(*anchored)
        assert actual.statistic == pytest.approx(expected.statistic, rel=3e-13)
        assert actual.pvalue == pytest.approx(expected.pvalue, rel=3e-13)

    @pytest.mark.parametrize(
        "function,kwargs",
        [
            (mean.ljw_2samp, {"calibration": "asymptotic", "rng": 2504}),
            (
                mean.thulin_2samp,
                {"n_subspaces": 5, "n_resamples": 19, "rng": 2505},
            ),
            (mean._clx_2samp, {"precision": np.eye(8)}),
            (mean.maximum_pairwise_bayes_factor_2samp, {}),
        ],
    )
    def test_sparse_randomized_and_bayesian_tests_share_anchor(
        self,
        function: object,
        kwargs: dict[str, object],
    ) -> None:
        rng = np.random.default_rng(2506)
        translated = (
            rng.normal(size=(5, 8)) + 1.0e14,
            rng.normal(size=(6, 8)) + 1.0e14,
        )
        anchored = _shared_feature_anchor(translated)
        actual = function(*translated, **kwargs)  # type: ignore[operator]
        expected = function(*anchored, **kwargs)  # type: ignore[operator]
        assert actual.statistic == pytest.approx(expected.statistic, rel=5e-12)
        if hasattr(actual, "pvalue"):
            assert actual.pvalue == pytest.approx(expected.pvalue, rel=5e-12)
        else:
            np.testing.assert_allclose(
                actual.component_log_bayes_factors,
                expected.component_log_bayes_factors,
                rtol=5e-12,
                atol=5e-12,
            )

    @pytest.mark.parametrize("function", [mean.schott_ksamp, mean.zx_ksamp])
    def test_high_dimensional_k_sample_tests_share_anchor(
        self,
        function: object,
    ) -> None:
        rng = np.random.default_rng(2507)
        translated = (
            rng.normal(size=(8, 5)) + 1.0e14,
            rng.normal(size=(10, 5)) + 1.0e14,
            rng.normal(size=(12, 5)) + 1.0e14,
        )
        anchored = _shared_feature_anchor(translated)
        actual = function(*translated)  # type: ignore[operator]
        expected = function(*anchored)  # type: ignore[operator]
        assert actual.statistic == pytest.approx(expected.statistic, rel=5e-12)
        assert actual.pvalue == pytest.approx(expected.pvalue, rel=5e-12)

    def test_opposite_float64_endpoints_do_not_leak_nan(self) -> None:
        spacing = np.spacing(1.0e308)
        x = -1.0e308 + spacing * np.arange(8, dtype=np.float64)
        y = 1.0e308 - spacing * np.arange(8, dtype=np.float64)
        result = mean.ttest_2samp(x, y)
        assert not math.isnan(result.statistic)
        assert not math.isnan(result.pvalue)
        assert all(not math.isnan(value) for _, value in result.estimates)

    def test_shared_anchor_is_row_and_group_order_deterministic(self) -> None:
        rng = np.random.default_rng(2508)
        x = rng.normal(size=(18, 6)) + 1.0e14
        y = rng.normal(size=(21, 6)) + 1.0e14
        direct = mean.bs_2samp(x, y)
        reordered = mean.bs_2samp(x[::-1], y[rng.permutation(y.shape[0])])
        exchanged = mean.bs_2samp(y, x)
        assert reordered.statistic == pytest.approx(direct.statistic, rel=3e-13)
        assert reordered.pvalue == pytest.approx(direct.pvalue, rel=3e-13)
        assert exchanged.statistic == pytest.approx(direct.statistic, rel=3e-13)
        assert exchanged.pvalue == pytest.approx(direct.pvalue, rel=3e-13)


class TestKSampleMeanTests:
    @pytest.fixture
    def groups(self) -> tuple[NDArray[np.float64], ...]:
        rng = np.random.default_rng(3001)
        return (
            rng.normal(size=(9, 7)),
            1.4 * rng.normal(size=(11, 7)) + 0.2,
            rng.normal(size=(13, 7)) @ np.diag(np.linspace(0.7, 1.4, 7)) - 0.1,
        )

    def test_schott_uses_manova_error_ssp_and_literal_calibration(
        self, groups: tuple[NDArray[np.float64], ...]
    ) -> None:
        result = mean.schott_ksamp(*groups)
        sizes = np.asarray([group.shape[0] for group in groups], dtype=float)
        total = int(np.sum(sizes))
        group_count = len(groups)
        e = total - group_count
        h = group_count - 1
        covariances = tuple(_covariance(group) for group in groups)
        error = sum(
            (group.shape[0] - 1) * covariance
            for group, covariance in zip(groups, covariances, strict=True)
        )
        means = tuple(np.mean(group, axis=0) for group in groups)
        grand = (
            sum(
                size * group_mean for size, group_mean in zip(sizes, means, strict=True)
            )
            / total
        )
        hypothesis = sum(
            size * np.outer(group_mean - grand, group_mean - grand)
            for size, group_mean in zip(sizes, means, strict=True)
        )
        expected_statistic = (
            float(np.trace(hypothesis)) / h - float(np.trace(error)) / e
        ) / math.sqrt(total - 1)
        a_value = (_trace_square(error) - float(np.trace(error)) ** 2 / e) / (
            (e + 2) * (e - 1)
        )
        expected_z = expected_statistic / math.sqrt(2 * a_value / (h * e))

        assert result.statistic == pytest.approx(expected_statistic, rel=3e-14)
        assert _diagnostics(result)["standardized statistic"] == pytest.approx(
            expected_z, rel=3e-14
        )
        assert result.pvalue == pytest.approx(stats.norm.sf(expected_z))

        legacy_error = sum(
            group.shape[0] * covariance
            for group, covariance in zip(groups, covariances, strict=True)
        )
        legacy_statistic = (
            float(np.trace(hypothesis)) / h - float(np.trace(legacy_error)) / e
        ) / math.sqrt(total - 1)
        assert not math.isclose(result.statistic, legacy_statistic, rel_tol=1e-10)

    @staticmethod
    def _cph_literal(
        groups: tuple[NDArray[np.float64], ...], estimator: str
    ) -> tuple[float, float]:
        sizes = np.asarray([group.shape[0] for group in groups], dtype=float)
        total = float(np.sum(sizes))
        means = tuple(np.mean(group, axis=0) for group in groups)
        within = 0.0
        for group, size in zip(groups, sizes, strict=True):
            gram = group @ group.T
            within += (
                (total - size)
                / (total * (size - 1))
                * float(np.sum(gram) - np.trace(gram))
            )
        between = 0.0
        for first_index in range(len(groups) - 1):
            for second_index in range(first_index + 1, len(groups)):
                between += (
                    2
                    * sizes[first_index]
                    * sizes[second_index]
                    / total
                    * float(means[first_index] @ means[second_index])
                )
        statistic = within - between

        covariances = tuple(_covariance(group) for group in groups)
        diagonal = 0.0
        if estimator == "original":
            for group, size in zip(groups, sizes, strict=True):
                first_size = group.shape[0] // 2 + 1
                first_covariance = _covariance(group[:first_size])
                second_covariance = _covariance(group[first_size:])
                diagonal += (
                    size
                    * (total - size) ** 2
                    / (size - 1)
                    * float(np.trace(first_covariance @ second_covariance))
                )
        else:
            for size, covariance in zip(sizes, covariances, strict=True):
                diagonal += (
                    size
                    * (total - size) ** 2
                    * (size - 1)
                    / ((size + 1) * (size - 2))
                    * (
                        _trace_square(covariance)
                        - float(np.trace(covariance)) ** 2 / (size - 1)
                    )
                )
        cross = 0.0
        for first_index in range(len(groups) - 1):
            for second_index in range(first_index + 1, len(groups)):
                cross += (
                    2
                    * sizes[first_index]
                    * sizes[second_index]
                    * float(
                        np.trace(covariances[first_index] @ covariances[second_index])
                    )
                )
        variance = 2 / total**2 * (diagonal + cross)
        return statistic, variance

    @pytest.mark.parametrize("estimator", ["original", "hu"])
    def test_cph_matches_paper_u_statistic_and_variance_estimators(
        self,
        groups: tuple[NDArray[np.float64], ...],
        estimator: Literal["original", "hu"],
    ) -> None:
        result = mean.cph_ksamp(*groups, variance_estimator=estimator)
        expected_statistic, expected_variance = self._cph_literal(groups, estimator)
        expected_z = expected_statistic / math.sqrt(expected_variance)
        assert result.statistic == pytest.approx(expected_statistic, rel=4e-14)
        assert _diagnostics(result)["standardized statistic"] == pytest.approx(
            expected_z, rel=5e-14
        )
        assert result.pvalue == pytest.approx(stats.norm.sf(expected_z))
        assert _diagnostics(result)["variance estimator"] == estimator

        for shift in (1.0e8, 1.0e10, 1.0e12):
            translated = mean.cph_ksamp(
                *(group + shift for group in groups),
                variance_estimator=estimator,
            )
            # The tolerance grows only with quantization already present in
            # the translated input (the spacing at 1e12 is about 1.2e-4).
            if shift < 1.0e10:
                tolerance = 2e-6
            elif shift < 1.0e12:
                tolerance = 1e-5
            else:
                tolerance = 8e-4
            assert translated.statistic == pytest.approx(
                result.statistic, rel=tolerance, abs=tolerance
            )
            assert translated.pvalue == pytest.approx(
                result.pvalue, rel=tolerance, abs=tolerance
            )

        sizes = np.asarray([group.shape[0] for group in groups], dtype=float)
        total = float(np.sum(sizes))
        within_ordered = sum(
            (total - size)
            / (total * (size - 1))
            * float(np.sum(group @ group.T) - np.trace(group @ group.T))
            for group, size in zip(groups, sizes, strict=True)
        )
        legacy_half_statistic = expected_statistic - 0.5 * within_ordered
        assert not math.isclose(result.statistic, legacy_half_statistic, rel_tol=1e-10)

    @pytest.mark.parametrize(
        "function,kwargs",
        [
            (mean.schott_ksamp, {}),
            (mean.cph_ksamp, {"variance_estimator": "hu"}),
        ],
    )
    def test_group_exchange_and_scale_behavior(
        self,
        function: object,
        kwargs: dict[str, str],
        groups: tuple[NDArray[np.float64], ...],
    ) -> None:
        direct = function(*groups, **kwargs)  # type: ignore[operator]
        reordered = function(groups[2], groups[0], groups[1], **kwargs)  # type: ignore[operator]
        scaled = function(*(3.5 * group for group in groups), **kwargs)  # type: ignore[operator]
        translated = function(*(group + 1.0e8 for group in groups), **kwargs)  # type: ignore[operator]
        assert reordered.statistic == pytest.approx(direct.statistic, rel=2e-13)
        assert reordered.pvalue == pytest.approx(direct.pvalue, rel=2e-13)
        assert scaled.statistic == pytest.approx(3.5**2 * direct.statistic, rel=3e-13)
        assert scaled.pvalue == pytest.approx(direct.pvalue, rel=3e-13)
        assert translated.statistic == pytest.approx(
            direct.statistic, rel=2e-6, abs=2e-6
        )
        assert translated.pvalue == pytest.approx(direct.pvalue, rel=2e-6)

    def test_k_sample_trace_tests_avoid_feature_covariance(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rng = np.random.default_rng(3050)
        groups = (
            rng.normal(size=(7, 5_000)),
            rng.normal(size=(8, 5_000)),
            rng.normal(size=(9, 5_000)),
        )

        def forbidden_covariance(_values: NDArray[np.float64]) -> NDArray[np.float64]:
            raise AssertionError("dense feature covariance allocation")

        monkeypatch.setattr(mean, "_sample_covariance", forbidden_covariance)
        results = (
            mean.schott_ksamp(*groups),
            mean.cph_ksamp(*groups, variance_estimator="original"),
            mean.cph_ksamp(*groups, variance_estimator="hu"),
            mean.zx_ksamp(*groups),
        )
        for result in results:
            assert math.isfinite(result.statistic)
            assert math.isfinite(result.pvalue)

    def test_zx_is_literal_scheffe_transformation_with_explicit_base_test(
        self,
    ) -> None:
        rng = np.random.default_rng(3002)
        groups = (
            rng.normal(size=(8, 3)),
            1.3 * rng.normal(size=(11, 3)),
            rng.normal(size=(13, 3)) - 0.2,
        )
        reference = groups[0]
        blocks = []
        for group in groups[1:]:
            partial = group[: reference.shape[0]]
            blocks.append(
                (reference - np.mean(group, axis=0))
                + math.sqrt(reference.shape[0] / group.shape[0])
                * (partial - np.mean(partial, axis=0))
            )
        transformed = np.concatenate(blocks, axis=1)
        expected = mean.bs_1samp(transformed)
        actual = mean.zx_ksamp(*groups, base_test="bai-saranadasa")
        assert actual.statistic == pytest.approx(expected.statistic, rel=3e-13)
        assert actual.pvalue == pytest.approx(expected.pvalue, rel=3e-13)

        small_groups = (
            rng.normal(size=(8, 1)),
            rng.normal(size=(10, 1)),
            rng.normal(size=(12, 1)),
        )
        reference = small_groups[0]
        transformed = np.concatenate(
            [
                (reference - np.mean(group, axis=0))
                + math.sqrt(reference.shape[0] / group.shape[0])
                * (
                    group[: reference.shape[0]]
                    - np.mean(group[: reference.shape[0]], axis=0)
                )
                for group in small_groups[1:]
            ],
            axis=1,
        )
        expected_hotelling = mean.hotelling_1samp(transformed)
        actual_hotelling = mean.zx_ksamp(*small_groups, base_test="hotelling")
        assert actual_hotelling.statistic == pytest.approx(
            expected_hotelling.statistic, rel=3e-13
        )
        assert actual_hotelling.pvalue == pytest.approx(
            expected_hotelling.pvalue, rel=3e-13
        )

    def test_zx_row_pairing_sensitivity_is_explicit_and_intrinsic(self) -> None:
        rng = np.random.default_rng(3003)
        groups = (
            rng.normal(size=(7, 3)),
            rng.normal(size=(12, 3)),
            rng.normal(size=(14, 3)),
        )
        baseline = mean.zx_ksamp(*groups)
        permutation = np.array([8, 2, 11, 0, 6, 10, 4, 9, 1, 7, 5, 3])
        reordered = mean.zx_ksamp(groups[0], groups[1][permutation], groups[2])
        assert not math.isclose(
            baseline.statistic,
            reordered.statistic,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )

        signature = inspect.signature(mean.zx_ksamp)
        assert signature.parameters["base_test"].default == "bai-saranadasa"
        assert "method" not in signature.parameters
        with pytest.raises(ValueError, match="base_test"):
            mean.zx_ksamp(*groups, base_test="l2")  # type: ignore[arg-type]

    def test_zx_tied_minimum_size_uses_input_order_as_reference(self) -> None:
        rng = np.random.default_rng(3004)
        groups = tuple(rng.normal(size=(10, 5)) for _ in range(3))
        reordered_groups = (groups[2], groups[0], groups[1])
        reference = reordered_groups[0]
        transformed = np.concatenate(
            [
                (reference - np.mean(group, axis=0)) + (group - np.mean(group, axis=0))
                for group in reordered_groups[1:]
            ],
            axis=1,
        )
        expected = mean.bs_1samp(transformed)
        actual = mean.zx_ksamp(*reordered_groups)
        original_order = mean.zx_ksamp(*groups)

        assert actual.statistic == pytest.approx(expected.statistic, rel=3e-13)
        assert actual.pvalue == pytest.approx(expected.pvalue, rel=3e-13)
        assert not math.isclose(
            actual.statistic,
            original_order.statistic,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )


class TestRandomizedMeanTests:
    @pytest.fixture
    def samples(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rng = np.random.default_rng(4001)
        return rng.normal(size=(5, 6)), rng.normal(size=(6, 6)) + 0.15

    def test_ljw_conditional_f_formula_uses_one_projection(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        seed = 7101
        actual = mean.ljw_2samp(x, y, calibration="asymptotic", rng=seed)
        first, second = _canonical_randomization_pair(x, y)
        within_df = first.shape[0] + second.shape[0] - 2
        dimension = within_df // 2
        generator = np.random.default_rng(seed)
        projection = generator.standard_normal((first.shape[1], dimension))
        expected_t2 = _projected_t2(first, second, projection)
        denominator_df = within_df - dimension + 1
        expected_f = denominator_df / (dimension * within_df) * expected_t2

        assert actual.statistic == pytest.approx(expected_t2, rel=4e-13)
        assert actual.df == (float(dimension), float(denominator_df))
        assert actual.pvalue == pytest.approx(
            stats.f.sf(expected_f, dimension, denominator_df), rel=4e-13
        )
        assert _diagnostics(actual)["projected dimension"] == dimension

    def test_ljw_monte_carlo_holds_projection_fixed_and_reports_uncertainty(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        seed = 7102
        resamples = 29
        actual = mean.ljw_2samp(
            x,
            y,
            calibration="monte-carlo",
            n_resamples=resamples,
            rng=seed,
        )
        assert isinstance(actual, ResamplingTestResult)
        first, second = _canonical_randomization_pair(x, y)
        generator = np.random.default_rng(seed)
        within_df = first.shape[0] + second.shape[0] - 2
        projection = generator.standard_normal((first.shape[1], within_df // 2))
        origin = np.minimum(np.min(first, axis=0), np.min(second, axis=0))
        x_shifted, y_shifted = first - origin, second - origin
        scale = max(
            float(np.max(np.abs(x_shifted))),
            float(np.max(np.abs(y_shifted))),
        )
        x_scaled, y_scaled = x_shifted / scale, y_shifted / scale
        observed = _projected_t2(x_scaled, y_scaled, projection)
        combined = np.vstack((x_scaled, y_scaled))
        exceedances = 0
        for _ in range(resamples):
            order = generator.permutation(combined.shape[0])
            permuted = _projected_t2(
                combined[order[: first.shape[0]]],
                combined[order[first.shape[0] :]],
                projection,
            )
            exceedances += int(
                permuted >= observed - 100 * np.finfo(float).eps * abs(observed)
            )

        assert actual.statistic == pytest.approx(observed, rel=5e-13)
        assert actual.exceedances == exceedances
        assert actual.pvalue == (exceedances + 1) / (resamples + 1)
        assert actual.monte_carlo_standard_error is not None
        assert actual.tail_probability_interval is not None
        assert actual.exact is False
        assert (
            mean.ljw_2samp(
                x,
                y,
                calibration="monte-carlo",
                n_resamples=resamples,
                rng=seed,
            )
            == actual
        )

    def test_ljw_projects_before_forming_covariances(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rng = np.random.default_rng(7103)
        first = rng.normal(size=(5, 5_000))
        second = rng.normal(size=(6, 5_000))
        original_covariance = mean._sample_covariance

        def guarded_covariance(values: NDArray[np.float64]) -> NDArray[np.float64]:
            if values.shape[1] > 10:
                raise AssertionError("ambient feature covariance allocation")
            return original_covariance(values)

        monkeypatch.setattr(mean, "_sample_covariance", guarded_covariance)
        for calibration in ("asymptotic", "monte-carlo"):
            result = mean.ljw_2samp(
                first,
                second,
                calibration=calibration,
                n_resamples=3,
                rng=7103,
            )
            assert math.isfinite(result.statistic)
            assert math.isfinite(result.pvalue)

    def test_thulin_fixes_subspace_plan_across_permutations(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        seed = 7201
        subspace_count = 5
        resamples = 23
        actual = mean.thulin_2samp(
            x,
            y,
            n_subspaces=subspace_count,
            n_resamples=resamples,
            rng=seed,
        )
        first, second = _canonical_randomization_pair(x, y)
        generator = np.random.default_rng(seed)
        dimension = (first.shape[0] + second.shape[0] - 2) // 2
        origin = np.minimum(np.min(first, axis=0), np.min(second, axis=0))
        x_shifted, y_shifted = first - origin, second - origin
        feature_scale = np.maximum(
            np.max(np.abs(x_shifted), axis=0),
            np.max(np.abs(y_shifted), axis=0),
        )
        feature_scale[feature_scale == 0] = 1
        x_scaled, y_scaled = x_shifted / feature_scale, y_shifted / feature_scale
        subspaces = tuple(
            np.asarray(
                generator.choice(x.shape[1], size=dimension, replace=False),
                dtype=np.intp,
            )
            for _ in range(subspace_count)
        )

        def aggregate(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
            identity = np.eye(first.shape[1])
            return float(
                np.mean(
                    [
                        _projected_t2(first, second, identity[:, columns])
                        for columns in subspaces
                    ]
                )
            )

        observed = aggregate(x_scaled, y_scaled)
        combined = np.vstack((x_scaled, y_scaled))
        exceedances = 0
        for _ in range(resamples):
            order = generator.permutation(combined.shape[0])
            permuted = aggregate(
                combined[order[: first.shape[0]]],
                combined[order[first.shape[0] :]],
            )
            exceedances += int(
                permuted >= observed - 100 * np.finfo(float).eps * abs(observed)
            )

        assert actual.statistic == pytest.approx(observed, rel=5e-13)
        assert actual.exceedances == exceedances
        assert actual.pvalue == (exceedances + 1) / (resamples + 1)
        assert _diagnostics(actual) == {
            "subspace dimension": dimension,
            "subspaces": subspace_count,
        }

        scales = np.geomspace(0.2, 5.0, x.shape[1])
        scaled = mean.thulin_2samp(
            x * scales,
            y * scales,
            n_subspaces=subspace_count,
            n_resamples=resamples,
            rng=seed,
        )
        assert scaled.statistic == pytest.approx(actual.statistic, rel=1e-12)
        assert scaled.pvalue == actual.pvalue

        signature = inspect.signature(mean.thulin_2samp)
        assert "subspace_dimension" not in signature.parameters

    def test_thulin_subspace_kernel_never_allocates_a_full_feature_identity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rng = np.random.default_rng(7202)
        first = rng.normal(size=(5, 5_000))
        second = rng.normal(size=(6, 5_000))
        subspaces = (
            np.array([3, 117, 4_999], dtype=np.intp),
            np.array([21, 900, 3_001], dtype=np.intp),
        )
        original_eye = np.eye

        def guarded_eye(
            size: int, *args: object, **kwargs: object
        ) -> NDArray[np.float64]:
            if size > 10:
                raise AssertionError("full-dimensional identity allocation")
            return np.asarray(original_eye(size, *args, **kwargs), dtype=np.float64)

        monkeypatch.setattr(mean.np, "eye", guarded_eye)
        statistic = mean._subspace_hotelling_statistic(first, second, subspaces)
        assert math.isfinite(statistic)

    @pytest.mark.parametrize(
        "function,kwargs",
        [
            (
                mean.ljw_2samp,
                {
                    "calibration": "monte-carlo",
                    "n_resamples": 41,
                    "rng": 7301,
                },
            ),
            (
                mean.thulin_2samp,
                {"n_subspaces": 7, "n_resamples": 41, "rng": 7302},
            ),
        ],
    )
    def test_seeded_randomization_is_row_and_group_order_invariant(
        self,
        function: object,
        kwargs: dict[str, object],
        samples: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> None:
        x, y = samples
        generator = np.random.default_rng(7303)
        equal_x = generator.normal(size=(6, 8))
        equal_y = generator.normal(size=(6, 8))
        for first, second in ((x, y), (equal_x, equal_y)):
            direct = function(first, second, **kwargs)  # type: ignore[operator]
            reordered = function(  # type: ignore[operator]
                second[generator.permutation(second.shape[0])],
                first[generator.permutation(first.shape[0])],
                **kwargs,
            )
            assert reordered == direct

    def test_random_controls_and_names_are_strict(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        with pytest.raises(ValueError, match="calibration"):
            mean.ljw_2samp(x, y, calibration="permutation")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="n_resamples"):
            mean.ljw_2samp(
                x,
                y,
                calibration="monte-carlo",
                n_resamples=True,
            )
        with pytest.raises(TypeError, match="rng"):
            mean.thulin_2samp(x, y, rng=True)


class TestSparseAndBayesianMeanTests:
    @pytest.fixture
    def samples(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        rng = np.random.default_rng(5001)
        x = rng.normal(size=(24, 5))
        y = rng.normal(size=(29, 5)) + np.array([0.15, 0.0, -0.1, 0.2, 0.0])
        return x, y

    def test_clx_known_precision_matches_oracle_equation_and_invariants(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        base = np.array(
            [
                [1.6, 0.2, 0.0, 0.0, 0.1],
                [0.2, 1.2, 0.1, 0.0, 0.0],
                [0.0, 0.1, 1.4, -0.2, 0.0],
                [0.0, 0.0, -0.2, 1.8, 0.1],
                [0.1, 0.0, 0.0, 0.1, 1.1],
            ]
        )
        precision = base.T @ base
        actual = mean._clx_2samp(x, y, precision=precision)
        difference = np.mean(x, axis=0) - np.mean(y, axis=0)
        transformed = precision @ difference
        effective_size = x.shape[0] * y.shape[0] / (x.shape[0] + y.shape[0])
        expected_statistic = effective_size * float(
            np.max(transformed**2 / np.diag(precision))
        )
        p = x.shape[1]
        centered = expected_statistic - 2 * math.log(p) + math.log(math.log(p))
        expected_p = -math.expm1(-math.exp(-0.5 * centered) / math.sqrt(math.pi))

        assert actual.statistic == pytest.approx(expected_statistic, rel=5e-13)
        assert actual.pvalue == pytest.approx(expected_p, rel=5e-13)
        assert mean._clx_2samp(y, x, precision=precision).statistic == pytest.approx(
            actual.statistic, rel=5e-13
        )

        scales = np.geomspace(0.4, 3.0, x.shape[1])
        transformed_precision = precision / np.outer(scales, scales)
        scaled = mean._clx_2samp(
            x * scales,
            y * scales,
            precision=transformed_precision,
        )
        assert scaled.statistic == pytest.approx(actual.statistic, rel=8e-13)
        assert scaled.pvalue == pytest.approx(actual.pvalue, rel=8e-13)

    def test_clx_estimators_are_deterministic_and_declared(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        adaptive = mean._clx_2samp(x, y, precision="adaptive-threshold")
        clime = mean._clx_2samp(x, y, precision="clime")
        assert math.isfinite(adaptive.statistic)
        assert math.isfinite(clime.statistic)
        assert _diagnostics(adaptive)["precision"] == "adaptive-threshold precision"
        assert _diagnostics(clime)["precision"] == "CLIME-estimated precision"
        assert mean._clx_2samp(x, y, precision="clime") == clime

        origin = np.minimum(np.min(x, axis=0), np.min(y, axis=0))
        x_shifted, y_shifted = x - origin, y - origin
        feature_scale = np.maximum(
            np.max(np.abs(x_shifted), axis=0),
            np.max(np.abs(y_shifted), axis=0),
        )
        feature_scale[feature_scale == 0] = 1
        first = x_shifted / feature_scale
        second = y_shifted / feature_scale
        nx, ny = x.shape[0], y.shape[0]
        pooled = ((nx - 1) * _covariance(first) + (ny - 1) * _covariance(second)) / (
            nx + ny
        )
        difference = np.mean(first, axis=0) - np.mean(second, axis=0)

        diagonal = np.diag(pooled)
        standard_deviation = np.sqrt(diagonal)
        correlation = pooled / np.outer(standard_deviation, standard_deviation)
        tuning = math.sqrt(math.log(x.shape[1]) / (nx + ny))
        identity = np.eye(x.shape[1])
        constraint = np.block(
            [[correlation, -correlation], [-correlation, correlation]]
        )
        unsymmetrized = np.empty_like(pooled)
        for column in range(x.shape[1]):
            target = identity[:, column]
            fitted = optimize.linprog(
                np.ones(2 * x.shape[1]),
                A_ub=constraint,
                b_ub=np.concatenate((target + tuning, tuning - target)),
                bounds=(0.0, None),
                method="highs",
            )
            assert fitted.success and fitted.x is not None
            unsymmetrized[:, column] = fitted.x[: x.shape[1]] - fitted.x[x.shape[1] :]
        symmetric = np.where(
            np.abs(unsymmetrized) <= np.abs(unsymmetrized.T),
            unsymmetrized,
            unsymmetrized.T,
        )
        clime_precision = symmetric / np.outer(standard_deviation, standard_deviation)

        thresholds = np.zeros_like(pooled)
        first_centered = first - np.mean(first, axis=0)
        second_centered = second - np.mean(second, axis=0)
        effective_size = nx * ny / (nx + ny)
        for first_column in range(x.shape[1] - 1):
            for second_column in range(first_column + 1, x.shape[1]):
                entry = pooled[first_column, second_column]
                first_products = (
                    first_centered[:, first_column] * first_centered[:, second_column]
                    - entry
                )
                second_products = (
                    second_centered[:, first_column] * second_centered[:, second_column]
                    - entry
                )
                theta = (
                    float(first_products @ first_products)
                    + float(second_products @ second_products)
                ) / (nx + ny)
                threshold = 2 * math.sqrt(
                    max(theta, 0) * math.log(x.shape[1]) / effective_size
                )
                thresholds[first_column, second_column] = threshold
                thresholds[second_column, first_column] = threshold
        thresholded = pooled * (np.abs(pooled) >= thresholds)
        eigenvalues = np.linalg.eigvalsh(thresholded)
        spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        floor = math.sqrt(np.finfo(np.float64).eps) * spectral_scale
        if eigenvalues[0] < floor:
            thresholded += (floor - eigenvalues[0]) * identity
        adaptive_precision = np.linalg.inv(thresholded)

        def estimated_statistic(precision: NDArray[np.float64]) -> float:
            transformed_difference = precision @ difference
            first_transformed = first @ precision
            second_transformed = second @ precision
            transformed_covariance = (
                (nx - 1) * _covariance(first_transformed)
                + (ny - 1) * _covariance(second_transformed)
            ) / (nx + ny)
            return float(
                effective_size
                * float(
                    np.max(transformed_difference**2 / np.diag(transformed_covariance))
                )
            )

        assert clime.statistic == pytest.approx(
            estimated_statistic(clime_precision), rel=2e-12, abs=2e-12
        )
        assert adaptive.statistic == pytest.approx(
            estimated_statistic(adaptive_precision), rel=2e-12, abs=2e-12
        )

    def test_clx_rejects_malformed_controls(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        with pytest.raises(ValueError, match="precision"):
            mean._clx_2samp(x, y, precision="CLX")
        with pytest.raises(ValueError, match="symmetric"):
            mean._clx_2samp(x, y, precision=np.triu(np.ones((5, 5))))
        with pytest.raises(ValueError, match="positive definite"):
            mean._clx_2samp(x, y, precision=np.zeros((5, 5)))
        with pytest.raises(ValueError, match="delta"):
            mean._clx_2samp(x, y, delta=0)

    @staticmethod
    def _lyl_components(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        a0: float,
        b0: float,
        gamma: float,
    ) -> NDArray[np.float64]:
        combined = np.vstack((x, y))

        def rss(values: NDArray[np.float64]) -> NDArray[np.float64]:
            centered = values - np.mean(values, axis=0)
            return np.asarray(np.sum(centered**2, axis=0), dtype=np.float64)

        numerator = 2 * b0 + rss(combined)
        denominator = 2 * b0 + rss(x) + rss(y)
        return np.asarray(
            0.5 * math.log(gamma / (1 + gamma))
            + (combined.shape[0] / 2 + a0) * (np.log(numerator) - np.log(denominator)),
            dtype=np.float64,
        )

    def test_lyl_matches_published_equation_four_at_default_prior_shape(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        gamma = 0.07
        expected = self._lyl_components(x, y, a0=0.0, b0=0.0, gamma=gamma)
        actual = mean.maximum_pairwise_bayes_factor_2samp(x, y, gamma=gamma)
        assert isinstance(actual, BayesFactorTestResult)
        np.testing.assert_allclose(
            actual.component_log_bayes_factors,
            expected,
            rtol=4e-13,
            atol=4e-13,
        )
        assert actual.statistic == pytest.approx(float(np.max(expected)), rel=4e-13)
        assert actual.statistic_name == "maximum log BF"
        assert actual.max_log_bayes_factor == actual.statistic
        assert not hasattr(actual, "pvalue")

        # With the scale-free default prior, a common positive change of
        # measurement units must not alter the evidence.  These magnitudes
        # would underflow/overflow if residual sums of squares were formed on
        # the unnormalized observations.
        for scale in (1.0e-200, 1.0e200):
            scaled = mean.maximum_pairwise_bayes_factor_2samp(
                scale * x, scale * y, gamma=gamma
            )
            np.testing.assert_allclose(
                scaled.component_log_bayes_factors,
                actual.component_log_bayes_factors,
                rtol=8e-13,
                atol=8e-13,
            )

    def test_lyl_default_uses_published_dimension_rate(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        alpha = 2.01
        gamma = max(x.shape[0] + y.shape[0], x.shape[1]) ** (-alpha)
        expected = self._lyl_components(x, y, a0=0.0, b0=0.0, gamma=gamma)
        actual = mean.maximum_pairwise_bayes_factor_2samp(x, y)
        np.testing.assert_allclose(
            actual.component_log_bayes_factors,
            expected,
            rtol=5e-13,
            atol=5e-13,
        )
        diagnostics = _diagnostics(actual)
        assert diagnostics["alpha"] == alpha
        assert diagnostics["gamma"] == pytest.approx(gamma)
        assert diagnostics["gamma source"] == "paper rate"
        assert diagnostics["prior specification"] == "paper Equation (4)"

        explicit = mean.maximum_pairwise_bayes_factor_2samp(x, y, alpha=7.0, gamma=0.04)
        assert _diagnostics(explicit)["gamma"] == 0.04
        assert _diagnostics(explicit)["gamma source"] == "explicit"
        assert "alpha" not in _diagnostics(explicit)

    def test_lyl_generalized_prior_reduces_correctly_and_is_invariant(
        self, samples: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> None:
        x, y = samples
        controls = {"a0": 0.4, "b0": 0.25, "gamma": 0.03}
        expected = self._lyl_components(x, y, **controls)
        actual = mean.maximum_pairwise_bayes_factor_2samp(x, y, **controls)
        np.testing.assert_allclose(
            actual.component_log_bayes_factors,
            expected,
            rtol=5e-13,
            atol=5e-13,
        )

        translated = mean.maximum_pairwise_bayes_factor_2samp(
            x + 1e8, y + 1e8, **controls
        )
        swapped = mean.maximum_pairwise_bayes_factor_2samp(y, x, **controls)
        scale = 1e100
        scaled = mean.maximum_pairwise_bayes_factor_2samp(
            scale * x,
            scale * y,
            a0=controls["a0"],
            b0=controls["b0"] * scale**2,
            gamma=controls["gamma"],
        )
        np.testing.assert_allclose(
            translated.component_log_bayes_factors,
            actual.component_log_bayes_factors,
            rtol=2e-7,
            atol=2e-7,
        )
        np.testing.assert_allclose(
            swapped.component_log_bayes_factors,
            actual.component_log_bayes_factors,
            rtol=5e-13,
            atol=5e-13,
        )
        np.testing.assert_allclose(
            scaled.component_log_bayes_factors,
            actual.component_log_bayes_factors,
            rtol=8e-13,
            atol=8e-13,
        )

    def test_lyl_constant_feature_boundaries(self) -> None:
        x = np.zeros((6, 2))
        y = np.ones((7, 2))
        assert mean.maximum_pairwise_bayes_factor_2samp(x, y).statistic == math.inf
        with pytest.raises(ValueError, match="constant feature"):
            mean.maximum_pairwise_bayes_factor_2samp(x, np.zeros((7, 2)))

    def test_lyl_prior_controls_are_strict(self) -> None:
        rng = np.random.default_rng(5099)
        x, y = rng.normal(size=(8, 3)), rng.normal(size=(9, 3))
        with pytest.raises(ValueError, match="alpha"):
            mean.maximum_pairwise_bayes_factor_2samp(x, y, alpha=0.0)
        with pytest.raises(ValueError, match="gamma"):
            mean.maximum_pairwise_bayes_factor_2samp(x, y, gamma=0.0)


def test_public_catalog_and_control_surface_are_exact() -> None:
    expected = {
        "anova_oneway",
        "bs_1samp",
        "bs_2samp",
        "cq_2samp",
        "cph_ksamp",
        "dempster_1samp",
        "dempster_2samp",
        "hotelling_1samp",
        "hotelling_2samp",
        "johansen_2samp",
        "ky_2samp",
        "li_1samp",
        "li_2samp",
        "li_ksamp",
        "ljw_2samp",
        "maximum_pairwise_bayes_factor_2samp",
        "nvm_2samp",
        "schott_ksamp",
        "sd_1samp",
        "sd_2samp",
        "thulin_2samp",
        "ttest_1samp",
        "ttest_2samp",
        "yao_2samp",
        "zx_ksamp",
    }
    assert set(mean.__all__) == expected
    assert all(name == name.lower() for name in mean.__all__)
    assert "clx_2samp" not in mean.__all__
    assert not hasattr(mean, "clx_2samp")

    ljw = inspect.signature(mean.ljw_2samp)
    assert ljw.parameters["calibration"].default == "asymptotic"
    assert ljw.parameters["n_resamples"].kind is inspect.Parameter.KEYWORD_ONLY
    thulin = inspect.signature(mean.thulin_2samp)
    assert list(thulin.parameters) == ["x", "y", "n_subspaces", "n_resamples", "rng"]
    lyl = inspect.signature(mean.maximum_pairwise_bayes_factor_2samp)
    assert lyl.parameters["alpha"].default == 2.01
    assert lyl.parameters["gamma"].default is None


def test_shared_multivariate_input_failures_are_explicit() -> None:
    x = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(40, dtype=float).reshape(10, 4)
    with pytest.raises(ValueError, match="same number of features"):
        mean.bs_2samp(x, y)
    with pytest.raises(ValueError, match="at least two samples"):
        mean.schott_ksamp(x)
    with pytest.raises(ValueError, match="at least 5 observations"):
        mean.cph_ksamp(x[:4], x[4:9], variance_estimator="original")
    with pytest.raises(ValueError, match="requires p"):
        mean.thulin_2samp(np.ones((8, 2)), np.eye(8, 2), rng=1)

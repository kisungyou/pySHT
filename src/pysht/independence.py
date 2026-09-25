"""Nonparametric tests of pairwise and mutual independence."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._distance_kernel import (
    as_sample,
    calibrate_blocks,
    canonical_block_keys,
    canonical_blocks,
    canonical_paired_row_order,
    distance_geometry,
    double_center,
    kernel_matrix,
)
from ._results import ResamplingTestResult
from ._validation import validate_real_scalar

type _IndexStatistic = Callable[[tuple[NDArray[np.intp], ...]], float]
type _KernelName = Literal["rbf", "laplacian"]
type _KernelControls = _KernelName | tuple[_KernelName, ...]
type _BandwidthControl = str | float
type _BandwidthControls = _BandwidthControl | tuple[_BandwidthControl, ...]

__all__ = ["dhsic", "distance_covariance", "distance_multivariance", "hsic"]


def _validate_blocks(
    samples: tuple[ArrayLike, ...],
    *,
    minimum_rows: int,
) -> tuple[NDArray[np.float64], ...]:
    """Validate row-paired random-vector samples."""
    if len(samples) < 2:
        raise ValueError("at least two random-vector samples are required")
    blocks = tuple(
        as_sample(sample, name=f"samples[{index}]", minimum_rows=minimum_rows)
        for index, sample in enumerate(samples)
    )
    sample_size = blocks[0].shape[0]
    if any(block.shape[0] != sample_size for block in blocks[1:]):
        raise ValueError("all random-vector samples must have the same number of rows")
    return blocks


def _canonical_kernel_bundles(
    blocks: tuple[NDArray[np.float64], ...],
    kernels: tuple[object, ...],
    bandwidths: tuple[object, ...],
) -> tuple[
    tuple[NDArray[np.float64], ...],
    tuple[object, ...],
    tuple[object, ...],
]:
    """Canonicalize paired blocks together with their kernel controls."""
    row_order = canonical_paired_row_order(blocks)
    ordered_blocks = tuple(np.ascontiguousarray(block[row_order]) for block in blocks)
    metric_keys = canonical_block_keys(ordered_blocks)
    bundles = sorted(
        (
            (block, kernel, bandwidth, metric_key)
            for block, kernel, bandwidth, metric_key in zip(
                ordered_blocks, kernels, bandwidths, metric_keys, strict=True
            )
        ),
        key=lambda bundle: (
            bundle[3],
            type(bundle[1]).__name__,
            repr(bundle[1]),
            type(bundle[2]).__name__,
            repr(bundle[2]),
        ),
    )
    return (
        tuple(bundle[0] for bundle in bundles),
        tuple(bundle[1] for bundle in bundles),
        tuple(bundle[2] for bundle in bundles),
    )


def _normalized_distance_matrices(
    blocks: tuple[NDArray[np.float64], ...],
    *,
    allow_constant: bool = False,
) -> tuple[tuple[NDArray[np.float64], ...], tuple[float, ...]]:
    """Return the normalized ``-C B C / mean(B)`` matrices of Eq. 4.25."""
    matrices: list[NDArray[np.float64]] = []
    log_scales: list[float] = []
    for block in blocks:
        geometry = distance_geometry(block)
        distances = geometry.distances
        mean_distance = float(np.mean(distances, dtype=np.float64))
        if mean_distance == 0.0:
            if not allow_constant:
                raise ValueError(
                    "normalized distance multivariance requires every marginal to vary"
                )
            matrices.append(np.zeros_like(distances))
            log_scales.append(-math.inf)
            continue
        matrices.append(-double_center(distances) / mean_distance)
        log_scales.append(geometry.log_scale + math.log(mean_distance))
    return tuple(matrices), tuple(log_scales)


def _restore_scaled(normalized: float, log_scale: float) -> float:
    """Restore a nonnegative homogeneous quantity from its log scale."""
    if normalized == 0.0:
        return 0.0
    log_value = math.log(normalized) + log_scale
    if log_value > math.log(np.finfo(np.float64).max):
        return math.inf
    if log_value < math.log(float(np.nextafter(0.0, 1.0))):
        return 0.0
    return math.exp(log_value)


def _aligned(
    matrix: NDArray[np.float64], indices: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Apply one marginal row permutation to both matrix axes."""
    return matrix[np.ix_(indices, indices)]


def _normalized_distance_covariance(
    matrices: tuple[NDArray[np.float64], ...],
    indices: tuple[NDArray[np.intp], ...],
) -> float:
    """Evaluate normalized sample distance covariance squared."""
    first = _aligned(matrices[0], indices[0])
    second = _aligned(matrices[1], indices[1])
    statistic = float(np.mean(first * second, dtype=np.float64))
    scale = float(np.mean(np.abs(first * second), dtype=np.float64))
    if statistic < 0.0 and abs(statistic) <= 500.0 * np.finfo(np.float64).eps * scale:
        return 0.0
    if statistic < 0.0:
        raise ValueError("normalized distance covariance became negative")
    return statistic


def _aligned_batch(
    matrix: NDArray[np.float64], indices: NDArray[np.intp]
) -> NDArray[np.float64]:
    """Apply a batch of row permutations to both matrix axes."""
    return matrix[indices[:, :, None], indices[:, None, :]]


def _normalized_distance_covariance_batch(
    matrices: tuple[NDArray[np.float64], ...],
    plans: NDArray[np.intp],
) -> NDArray[np.float64]:
    first = _aligned_batch(matrices[0], plans[:, 0])
    second = _aligned_batch(matrices[1], plans[:, 1])
    values = np.mean(first * second, axis=(1, 2), dtype=np.float64)
    scale = np.mean(np.abs(first * second), axis=(1, 2), dtype=np.float64)
    small_negative = (values < 0.0) & (
        np.abs(values) <= 500.0 * np.finfo(np.float64).eps * scale
    )
    values[small_negative] = 0.0
    if np.any(values < 0.0):
        raise ValueError("normalized distance covariance became negative")
    return np.asarray(values, dtype=np.float64)


def _resampling_result(
    *,
    observed: float,
    statistic: _IndexStatistic,
    batch_statistic: Callable[[NDArray[np.intp]], NDArray[np.float64]],
    blocks: tuple[NDArray[np.float64], ...],
    method: str,
    statistic_name: str,
    diagnostics: tuple[tuple[str, int | float | str], ...],
    calibration_observed: float | None = None,
    estimates: tuple[tuple[str, float], ...] = (),
    calibration: object,
    n_resamples: object,
    rng: int | np.integer | np.random.Generator | None,
) -> ResamplingTestResult:
    """Calibrate and construct a common independence-test result."""
    summary = calibrate_blocks(
        observed=observed if calibration_observed is None else calibration_observed,
        statistic=statistic,
        batch_statistic=batch_statistic,
        sample_size=blocks[0].shape[0],
        block_count=len(blocks),
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
    )
    return ResamplingTestResult(
        statistic=observed,
        pvalue=summary.pvalue,
        method=method,
        alternative="the random vectors are not jointly independent",
        data_name="samples",
        statistic_name=statistic_name,
        calibration=summary.label,
        diagnostics=diagnostics,
        estimates=estimates,
        n_resamples=summary.n_resamples,
        exceedances=summary.exceedances,
        exact=summary.exact,
        monte_carlo_standard_error=summary.standard_error,
        tail_probability_interval=summary.interval,
    )


def distance_covariance(
    x: ArrayLike,
    y: ArrayLike,
    *,
    calibration: Literal["permutation", "exact", "monte-carlo"] = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Test pairwise independence using distance covariance.

    Rows of ``x`` and ``y`` are paired observations; the feature dimensions
    may differ. The reported statistic is ``n V_n^2`` in the original distance
    units. Calibration uses the algebraically equivalent statistic obtained by
    dividing each doubly centered distance matrix by its mean interpoint
    distance. This positive normalization improves numerical stability without
    changing permutation ordering. Distance covariance, distance correlation,
    and both distance variances are also returned when representable.

    References
    ----------
    Székely, G. J., Rizzo, M. L. and Bakirov, N. K. (2007). Measuring and
    testing dependence by correlation of distances. *Annals of Statistics*,
    35, 2769--2794. https://doi.org/10.1214/009053607000000505
    """
    original_blocks = _validate_blocks((x, y), minimum_rows=2)
    original_matrices, original_log_scales = _normalized_distance_matrices(
        original_blocks, allow_constant=True
    )
    blocks = canonical_blocks(original_blocks)
    matrices, log_scales = _normalized_distance_matrices(blocks, allow_constant=True)
    identity = np.arange(blocks[0].shape[0], dtype=np.intp)
    observed_indices = (identity, identity)
    normalized = _normalized_distance_covariance(matrices, observed_indices)
    sample_size = blocks[0].shape[0]
    observed = _restore_scaled(
        sample_size * normalized,
        log_scales[0] + log_scales[1],
    )
    normalized_variance_x = float(np.mean(original_matrices[0] ** 2, dtype=np.float64))
    normalized_variance_y = float(np.mean(original_matrices[1] ** 2, dtype=np.float64))
    distance_covariance = _restore_scaled(
        math.sqrt(normalized),
        (log_scales[0] + log_scales[1]) / 2.0,
    )
    distance_variance_x = _restore_scaled(
        math.sqrt(normalized_variance_x), original_log_scales[0]
    )
    distance_variance_y = _restore_scaled(
        math.sqrt(normalized_variance_y), original_log_scales[1]
    )
    correlation_denominator = math.sqrt(normalized_variance_x * normalized_variance_y)
    squared_correlation = (
        0.0 if correlation_denominator == 0.0 else normalized / correlation_denominator
    )
    squared_correlation = min(1.0, max(0.0, squared_correlation))
    distance_correlation = math.sqrt(squared_correlation)
    raw_estimates = (
        ("distance covariance", distance_covariance),
        ("distance correlation", distance_correlation),
        ("distance variance x", distance_variance_x),
        ("distance variance y", distance_variance_y),
    )
    estimates = (
        raw_estimates if all(math.isfinite(value) for _, value in raw_estimates) else ()
    )
    diagnostics: tuple[tuple[str, int | float | str], ...] = (
        ("marginals", 2),
        ("metric", "Euclidean"),
        ("normalized V_n^2", normalized),
    )
    if not estimates:
        diagnostics += (("raw estimates", "outside float64"),)
    return _resampling_result(
        observed=observed,
        statistic=lambda indices: _normalized_distance_covariance(matrices, indices),
        batch_statistic=lambda plans: _normalized_distance_covariance_batch(
            matrices, plans
        ),
        blocks=blocks,
        method="distance covariance independence test (2007)",
        statistic_name="n V_n^2",
        diagnostics=diagnostics,
        calibration_observed=normalized,
        estimates=estimates,
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
    )


def _broadcast_controls(
    value: object,
    *,
    count: int,
    name: str,
) -> tuple[object, ...]:
    """Broadcast one kernel control or validate a length-d tuple."""
    if isinstance(value, tuple):
        if len(value) != count:
            raise ValueError(f"{name} must contain exactly {count} entries")
        return value
    return tuple(value for _ in range(count))


def _kernel_matrices(
    blocks: tuple[NDArray[np.float64], ...],
    *,
    kernel: _KernelControls,
    bandwidth: _BandwidthControls,
) -> tuple[
    tuple[NDArray[np.float64], ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[float | None, ...],
]:
    """Construct one fixed characteristic Gram matrix per marginal."""
    matrices: list[NDArray[np.float64]] = []
    selected_kernels: list[str] = []
    bandwidth_modes: list[str] = []
    numeric_bandwidths: list[float | None] = []
    raw_kernels = _broadcast_controls(kernel, count=len(blocks), name="kernel")
    raw_bandwidths = _broadcast_controls(bandwidth, count=len(blocks), name="bandwidth")
    blocks, kernels, bandwidths = _canonical_kernel_bundles(
        blocks, raw_kernels, raw_bandwidths
    )
    for block, block_kernel, block_bandwidth in zip(
        blocks, kernels, bandwidths, strict=True
    ):
        matrix, current_kernel, current_mode, current_bandwidth = kernel_matrix(
            distance_geometry(block),
            kernel=block_kernel,
            bandwidth=block_bandwidth,
        )
        matrices.append(matrix)
        selected_kernels.append(current_kernel)
        bandwidth_modes.append(current_mode)
        numeric_bandwidths.append(current_bandwidth)
    return (
        tuple(matrices),
        tuple(selected_kernels),
        tuple(bandwidth_modes),
        tuple(numeric_bandwidths),
    )


def _dhsic_statistic(
    grams: tuple[NDArray[np.float64], ...],
    indices: tuple[NDArray[np.intp], ...],
) -> float:
    """Evaluate ``dHSIC_hat_n`` from Definition 4 of Pfister et al."""
    aligned = tuple(
        _aligned(matrix, index) for matrix, index in zip(grams, indices, strict=True)
    )
    sample_size = aligned[0].shape[0]
    product_matrix = np.ones_like(aligned[0])
    for matrix in aligned:
        product_matrix *= matrix
    first_term = float(np.mean(product_matrix, dtype=np.float64))
    second_term = math.prod(
        float(np.mean(matrix, dtype=np.float64)) for matrix in aligned
    )
    row_product = np.ones(sample_size, dtype=np.float64)
    for matrix in aligned:
        row_product *= np.mean(matrix, axis=1, dtype=np.float64)
    third_term = 2.0 * float(np.mean(row_product, dtype=np.float64))
    estimate = first_term + second_term - third_term
    scale = abs(first_term) + abs(second_term) + abs(third_term)
    if estimate < 0.0 and abs(estimate) <= 500.0 * np.finfo(np.float64).eps * scale:
        estimate = 0.0
    if estimate < 0.0:
        raise ValueError("dHSIC became negative beyond floating-point tolerance")
    return float(estimate)


def _dhsic_statistic_batch(
    grams: tuple[NDArray[np.float64], ...],
    plans: NDArray[np.intp],
) -> NDArray[np.float64]:
    aligned = tuple(
        _aligned_batch(matrix, plans[:, block]) for block, matrix in enumerate(grams)
    )
    sample_size = aligned[0].shape[1]
    product_matrix = np.ones_like(aligned[0])
    for matrix in aligned:
        product_matrix *= matrix
    first_term = np.mean(product_matrix, axis=(1, 2), dtype=np.float64)
    second_term = np.ones(plans.shape[0], dtype=np.float64)
    row_product = np.ones((plans.shape[0], sample_size), dtype=np.float64)
    for matrix in aligned:
        second_term *= np.mean(matrix, axis=(1, 2), dtype=np.float64)
        row_product *= np.mean(matrix, axis=2, dtype=np.float64)
    third_term = 2.0 * np.mean(row_product, axis=1, dtype=np.float64)
    estimates = first_term + second_term - third_term
    scale = np.abs(first_term) + np.abs(second_term) + np.abs(third_term)
    small_negative = (estimates < 0.0) & (
        np.abs(estimates) <= 500.0 * np.finfo(np.float64).eps * scale
    )
    estimates[small_negative] = 0.0
    if np.any(estimates < 0.0):
        raise ValueError("dHSIC became negative beyond floating-point tolerance")
    return np.asarray(estimates, dtype=np.float64)


def _kernel_diagnostics(
    *,
    kernels: tuple[str, ...],
    bandwidth_modes: tuple[str, ...],
    numeric_bandwidths: tuple[float | None, ...],
) -> tuple[tuple[str, int | float | str], ...]:
    diagnostics: tuple[tuple[str, int | float | str], ...] = (
        ("marginals", len(kernels)),
    )
    for index, (kernel, mode, value) in enumerate(
        zip(kernels, bandwidth_modes, numeric_bandwidths, strict=True),
        start=1,
    ):
        diagnostics += (
            (f"kernel {index}", kernel),
            (f"bandwidth {index}", mode if value is None else value),
        )
    return diagnostics


def hsic(
    x: ArrayLike,
    y: ArrayLike,
    *,
    kernel_x: _KernelName = "rbf",
    kernel_y: _KernelName = "rbf",
    bandwidth_x: _BandwidthControl = "median",
    bandwidth_y: _BandwidthControl = "median",
    calibration: Literal["permutation", "exact", "monte-carlo"] = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Test pairwise independence with the biased empirical HSIC.

    ``kernel_x`` and ``kernel_y`` independently select a fixed RBF or Laplacian
    Gram matrix. A median bandwidth is estimated within the corresponding
    marginal before any permutation; alternatively, each marginal accepts its
    own explicit positive scalar bandwidth.

    References
    ----------
    Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B. and
    Smola, A. (2008). A kernel statistical test of independence. *Advances in
    Neural Information Processing Systems 20*, 585--592.
    """
    blocks = _validate_blocks((x, y), minimum_rows=2)
    grams, _, _, _ = _kernel_matrices(
        blocks,
        kernel=(kernel_x, kernel_y),
        bandwidth=(bandwidth_x, bandwidth_y),
    )
    identity = np.arange(blocks[0].shape[0], dtype=np.intp)
    observed = _dhsic_statistic(grams, (identity, identity))

    def diagnostic_bandwidth(value: _BandwidthControl, *, name: str) -> float | str:
        return (
            "median"
            if isinstance(value, str)
            else validate_real_scalar(value, name=name)
        )

    hsic_diagnostics: tuple[tuple[str, int | float | str], ...] = (
        ("kernel x", kernel_x),
        ("kernel y", kernel_y),
        (
            "bandwidth x",
            diagnostic_bandwidth(bandwidth_x, name="bandwidth_x"),
        ),
        (
            "bandwidth y",
            diagnostic_bandwidth(bandwidth_y, name="bandwidth_y"),
        ),
    )
    return _resampling_result(
        observed=observed,
        statistic=lambda indices: _dhsic_statistic(grams, indices),
        batch_statistic=lambda plans: _dhsic_statistic_batch(grams, plans),
        blocks=blocks,
        method="Hilbert-Schmidt independence criterion test (2008)",
        statistic_name="HSIC_b",
        diagnostics=hsic_diagnostics,
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
    )


def dhsic(
    *samples: ArrayLike,
    kernel: _KernelControls = "rbf",
    bandwidth: _BandwidthControls = "median",
    calibration: Literal["permutation", "exact", "monte-carlo"] = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Test mutual independence of two or more random vectors using dHSIC.

    Following the estimator's defining finite-sample regime, ``n >= 2*d`` is
    required for ``d`` marginals. For two marginals, this function reports
    ``n`` times the :func:`hsic` statistic; that positive scaling gives exactly
    the same permutation ordering, exceedance count, and p-value.

    References
    ----------
    Pfister, N., Bühlmann, P., Schölkopf, B. and Peters, J. (2018).
    Kernel-based tests for joint independence. *Journal of the Royal
    Statistical Society B*, 80, 5--31. https://doi.org/10.1111/rssb.12235
    """
    blocks = _validate_blocks(samples, minimum_rows=4)
    block_count = len(blocks)
    if blocks[0].shape[0] < 2 * block_count:
        raise ValueError("dhsic requires n >= 2*d for d random-vector samples")
    grams, selected_kernels, bandwidth_modes, numeric_bandwidths = _kernel_matrices(
        blocks, kernel=kernel, bandwidth=bandwidth
    )
    identity = np.arange(blocks[0].shape[0], dtype=np.intp)
    observed_indices = tuple(identity for _ in blocks)
    sample_size = blocks[0].shape[0]
    observed_unscaled = _dhsic_statistic(grams, observed_indices)
    observed = sample_size * observed_unscaled
    return _resampling_result(
        observed=observed,
        statistic=lambda indices: sample_size * _dhsic_statistic(grams, indices),
        batch_statistic=lambda plans: (
            sample_size * _dhsic_statistic_batch(grams, plans)
        ),
        blocks=blocks,
        method="d-variable Hilbert-Schmidt independence criterion test (2018)",
        statistic_name="n dHSIC_hat",
        diagnostics=_kernel_diagnostics(
            kernels=selected_kernels,
            bandwidth_modes=bandwidth_modes,
            numeric_bandwidths=numeric_bandwidths,
        ),
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
    )


def _total_distance_multivariance(
    matrices: tuple[NDArray[np.float64], ...],
    indices: tuple[NDArray[np.intp], ...],
) -> float:
    """Evaluate normalized total sample distance multivariance, Eq. 4.29."""
    if len(matrices) == 2:
        return _normalized_distance_covariance(matrices, indices)
    product_matrix = np.ones_like(matrices[0])
    for matrix, index in zip(matrices, indices, strict=True):
        product_matrix *= 1.0 + _aligned(matrix, index)
    mean_product = math.fsum(float(value) for value in product_matrix.flat) / (
        product_matrix.size
    )
    statistic = (mean_product - 1.0) / (2 ** len(matrices) - len(matrices) - 1.0)
    scale = max(abs(mean_product), 1.0)
    if statistic < 0.0 and abs(statistic) <= 500.0 * np.finfo(np.float64).eps * scale:
        return 0.0
    if not math.isfinite(statistic):
        raise ValueError("distance multivariance could not be represented")
    if statistic < 0.0:
        raise ValueError("distance multivariance became negative")
    return float(statistic)


def _total_distance_multivariance_batch(
    matrices: tuple[NDArray[np.float64], ...],
    plans: NDArray[np.intp],
) -> NDArray[np.float64]:
    if len(matrices) == 2:
        return _normalized_distance_covariance_batch(matrices, plans)
    product_matrix = np.ones(
        (plans.shape[0], matrices[0].shape[0], matrices[0].shape[0]),
        dtype=np.float64,
    )
    for block, matrix in enumerate(matrices):
        product_matrix *= 1.0 + _aligned_batch(matrix, plans[:, block])
    mean_product = np.mean(product_matrix, axis=(1, 2), dtype=np.float64)
    statistics = (mean_product - 1.0) / (2 ** len(matrices) - len(matrices) - 1.0)
    scale = np.maximum(np.abs(mean_product), 1.0)
    small_negative = (statistics < 0.0) & (
        np.abs(statistics) <= 500.0 * np.finfo(np.float64).eps * scale
    )
    statistics[small_negative] = 0.0
    if not np.all(np.isfinite(statistics)):
        raise ValueError("distance multivariance could not be represented")
    if np.any(statistics < 0.0):
        raise ValueError("distance multivariance became negative")
    return np.asarray(statistics, dtype=np.float64)


def distance_multivariance(
    *samples: ArrayLike,
    calibration: Literal["permutation", "exact", "monte-carlo"] = "permutation",
    n_resamples: int = 9_999,
    rng: int | np.integer | np.random.Generator | None = None,
) -> ResamplingTestResult:
    """Test mutual independence using normalized total distance multivariance.

    Unlike a collection of pairwise tests, total distance multivariance can
    detect higher-order dependence whose every pair is independent.  The
    statistic is the normalized total version in Eq. 4.29 of the primary
    paper, using Euclidean distances and a separate mean-distance
    normalization for every marginal.

    References
    ----------
    Böttcher, B., Keller-Ressel, M. and Schilling, R. L. (2019). Distance
    multivariance: New dependence measures for random vectors. *Annals of
    Statistics*, 47, 2757--2789. https://doi.org/10.1214/18-AOS1764
    """
    blocks = canonical_blocks(_validate_blocks(samples, minimum_rows=2))
    matrices, _ = _normalized_distance_matrices(blocks)
    identity = np.arange(blocks[0].shape[0], dtype=np.intp)
    observed_indices = tuple(identity for _ in blocks)
    sample_size = blocks[0].shape[0]
    observed = sample_size * _total_distance_multivariance(matrices, observed_indices)
    return _resampling_result(
        observed=observed,
        statistic=lambda indices: (
            sample_size * _total_distance_multivariance(matrices, indices)
        ),
        batch_statistic=lambda plans: (
            sample_size * _total_distance_multivariance_batch(matrices, plans)
        ),
        blocks=blocks,
        method="normalized total distance multivariance test (2019)",
        statistic_name="n M_bar_n^2",
        diagnostics=(("marginals", len(blocks)), ("metric", "Euclidean")),
        calibration=calibration,
        n_resamples=n_resamples,
        rng=rng,
    )

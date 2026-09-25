"""Private distance, kernel, and randomization utilities.

The helpers in this module deliberately expose no public API.  They provide
the common numerical and finite-randomization contracts used by distribution
equality and independence tests without imposing one statistic's units on
another statistic.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _core
from ._resampling import exact_pvalue, monte_carlo_calibration
from ._validation import (
    make_generator,
    validate_choice,
    validate_positive_integer,
    validate_real_scalar,
)

type Calibration = Literal["permutation", "exact", "monte-carlo"]
type Kernel = Literal["rbf", "laplacian"]
type Bandwidth = str | float
type GroupStatistic = Callable[[tuple[NDArray[np.intp], ...]], float]
type GroupBatchStatistic = Callable[[NDArray[np.intp]], NDArray[np.float64]]
type BlockStatistic = Callable[[tuple[NDArray[np.intp], ...]], float]
type BlockBatchStatistic = Callable[[NDArray[np.intp]], NDArray[np.float64]]

_TIE_RTOL = 100.0 * np.finfo(np.float64).eps
_BATCH_ELEMENT_BUDGET = 2_000_000
_CANONICAL_WORK_BUDGET = 1_000_000


class _CanonicalBudgetExceeded(Exception):
    """Stop optional graph canonicalization before its search becomes costly."""


def _quadratic_batch_size(
    sample_size: int,
    *,
    working_matrices: int,
) -> int:
    """Bound one resampling batch by an approximate matrix-element budget."""
    per_plan = max(1, sample_size * sample_size * working_matrices)
    return max(1, min(512, _BATCH_ELEMENT_BUDGET // per_plan))


@dataclass(frozen=True, slots=True)
class DistanceGeometry:
    """Dimensionless distances plus the logarithm of their common scale."""

    distances: NDArray[np.float64]
    log_scale: float


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Values needed to construct a :class:`ResamplingTestResult`."""

    pvalue: float
    n_resamples: int
    exceedances: int
    exact: bool
    standard_error: float | None
    interval: tuple[float, float] | None
    label: str


def as_sample(
    sample: ArrayLike,
    *,
    name: str,
    minimum_rows: int = 2,
) -> NDArray[np.float64]:
    """Validate a univariate or row-by-feature sample."""
    try:
        raw = np.asarray(sample)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real numeric array") from exc
    if raw.dtype == np.dtype(bool) or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real numeric values")
    if raw.ndim == 1:
        raw = raw[:, None]
    elif raw.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if raw.shape[0] < minimum_rows:
        raise ValueError(f"{name} must contain at least {minimum_rows} observations")
    if raw.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature")
    try:
        values = np.asarray(raw, dtype=np.float64, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain real numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values == 0.0):
        values = values.copy()
        values[values == 0.0] = 0.0
    return values


def sort_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return rows in deterministic lexicographic order."""
    keys = tuple(values[:, j] for j in range(values.shape[1] - 1, -1, -1))
    return np.ascontiguousarray(values[np.lexsort(keys)])


def canonical_groups(
    samples: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], ...]:
    """Canonicalize group sizes; equal-size group identities are immaterial."""
    return tuple(sorted(samples, key=lambda group: (group.shape[0], group.shape[1])))


def _distance_rank_layers(
    samples: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.int64], ...]:
    """Encode marginal distance layers by stable approximate equality ranks.

    Distance ranks, unlike decimal rounding, have no fixed grid boundary.  The
    small comparison radius accounts only for accumulated ``hypot`` and common-
    scale rounding.  It is used for canonical randomization plans, never to
    alter a reported statistic.  A metric lying at the radius boundary can
    receive a different valid Monte Carlo plan after rounded coordinate
    transformations; exact calibration is unaffected.
    """
    geometries = tuple(distance_geometry(sample).distances for sample in samples)
    size = geometries[0].shape[0]
    upper = np.triu_indices(size, k=1)
    entries: list[tuple[float, int, int]] = []
    for layer, (sample, distances) in enumerate(zip(samples, geometries, strict=True)):
        for offset, value in enumerate(distances[upper]):
            entries.append(
                (float(value), sample.shape[1], layer * upper[0].size + offset)
            )
    entries.sort(key=lambda entry: entry[0])
    flat_ranks = np.zeros(len(samples) * upper[0].size, dtype=np.int64)
    rank = 0
    anchor = -math.inf
    anchor_features = 1
    for value, feature_count, location in entries:
        tolerance = (
            8.0
            * np.finfo(np.float64).eps
            * max(1, feature_count, anchor_features)
            * max(1.0, abs(value), abs(anchor))
        )
        if rank == 0 or value - anchor > tolerance:
            rank += 1
            anchor = value
            anchor_features = feature_count
        flat_ranks[location] = rank
    layers: list[NDArray[np.int64]] = []
    for layer in range(len(samples)):
        matrix = np.zeros((size, size), dtype=np.int64)
        current = flat_ranks[layer * upper[0].size : (layer + 1) * upper[0].size]
        matrix[upper] = current
        matrix[(upper[1], upper[0])] = current
        layers.append(matrix)
    return tuple(layers)


def _canonical_metric_order(
    samples: Sequence[NDArray[np.float64]],
    *,
    initial_colors: NDArray[np.int64] | None = None,
    extra_edge_layers: Sequence[NDArray[np.int64]] = (),
) -> NDArray[np.intp]:
    """Canonical-label a complete edge-coloured metric graph.

    Colour refinement resolves ordinary data immediately.  For a symmetric
    geometry, individualization/refinement explores non-twin choices and picks
    the lexicographically least metric code. True twins are interchangeable
    and only one representative is explored. A fixed edge-work budget bounds
    refinement and twin comparisons; exhausted searches use coordinate order.
    That fallback may change a seed's realization after an isometry, but every
    permutation remains uniform on the same orbit.
    """
    if not samples:
        return np.empty(0, dtype=np.intp)
    sample_size = samples[0].shape[0]
    rank_layers = _distance_rank_layers(samples)
    edge_tokens = np.empty(
        (
            sample_size,
            sample_size,
            len(samples) + len(extra_edge_layers),
            2,
        ),
        dtype=np.int64,
    )
    for row in range(sample_size):
        for column in range(sample_size):
            components = sorted(
                (sample.shape[1], int(ranks[row, column]))
                for sample, ranks in zip(samples, rank_layers, strict=True)
            )
            components += [
                (-1 - layer, int(values[row, column]))
                for layer, values in enumerate(extra_edge_layers)
            ]
            edge_tokens[row, column] = components
    base = (
        np.zeros(sample_size, dtype=np.int64)
        if initial_colors is None
        else np.asarray(initial_colors, dtype=np.int64)
    )
    work_remaining = _CANONICAL_WORK_BUDGET
    work_per_pass = sample_size * sample_size * (len(samples) + len(extra_edge_layers))

    def charge_work() -> None:
        nonlocal work_remaining
        work_remaining -= work_per_pass
        if work_remaining < 0:
            raise _CanonicalBudgetExceeded

    def refine(colors: NDArray[np.int64]) -> NDArray[np.int64]:
        while True:
            charge_work()
            signatures = []
            for row in range(sample_size):
                neighbors = tuple(
                    sorted(
                        (
                            int(colors[other]),
                            tuple(int(value) for value in edge_tokens[row, other].flat),
                        )
                        for other in range(sample_size)
                    )
                )
                signatures.append((int(colors[row]), neighbors))
            levels = {
                signature: index
                for index, signature in enumerate(sorted(set(signatures)))
            }
            updated = np.asarray(
                [levels[signature] for signature in signatures], dtype=np.int64
            )
            if np.array_equal(
                colors[:, None] == colors[None, :],
                updated[:, None] == updated[None, :],
            ):
                return updated
            colors = updated

    def are_twins(first: int, second: int, colors: NDArray[np.int64]) -> bool:
        charge_work()
        if colors[first] != colors[second]:
            return False
        order = np.arange(sample_size, dtype=np.intp)
        order[first], order[second] = order[second], order[first]
        if any(
            not np.array_equal(values, values[np.ix_(order, order)])
            for values in extra_edge_layers
        ):
            return False
        original_layers = sorted(
            (sample.shape[1], ranks.tobytes(order="C"))
            for sample, ranks in zip(samples, rank_layers, strict=True)
        )
        swapped_layers = sorted(
            (
                sample.shape[1],
                ranks[np.ix_(order, order)].tobytes(order="C"),
            )
            for sample, ranks in zip(samples, rank_layers, strict=True)
        )
        return original_layers == swapped_layers

    def search(
        colors: NDArray[np.int64],
    ) -> tuple[tuple[int, ...], NDArray[np.intp]]:
        colors = refine(colors)
        levels, counts = np.unique(colors, return_counts=True)
        ambiguous = levels[counts > 1]
        if ambiguous.size == 0:
            order = np.asarray(np.argsort(colors, kind="stable"), dtype=np.intp)
            metric_layers = sorted(
                (
                    sample.shape[1],
                    tuple(int(value) for value in ranks[np.ix_(order, order)].flat),
                )
                for sample, ranks in zip(samples, rank_layers, strict=True)
            )
            metric_code = tuple(
                value
                for feature_count, values in metric_layers
                for value in (feature_count, *values)
            )
            extra_code = tuple(
                int(value)
                for values in extra_edge_layers
                for value in values[np.ix_(order, order)].flat
            )
            code = tuple(int(value) for value in base[order]) + extra_code + metric_code
            return code, order

        ambiguous_counts = counts[counts > 1]
        selected_level = ambiguous[int(np.argmin(ambiguous_counts))]
        cell = [int(value) for value in np.flatnonzero(colors == selected_level)]
        if all(are_twins(cell[0], candidate, colors) for candidate in cell[1:]):
            individualized = colors.copy()
            next_color = int(np.max(colors)) + 1
            for offset, candidate in enumerate(cell):
                individualized[candidate] = next_color + offset
            return search(individualized)
        representatives: list[int] = []
        for candidate in cell:
            if not any(
                are_twins(candidate, prior, colors) for prior in representatives
            ):
                representatives.append(candidate)
        best: tuple[tuple[int, ...], NDArray[np.intp]] | None = None
        for candidate in representatives:
            individualized = colors.copy()
            individualized[candidate] = int(np.max(colors)) + 1
            current = search(individualized)
            if best is None or current[0] < best[0]:
                best = current
        if (
            best is None
        ):  # pragma: no cover - a nonempty cell always has a representative
            raise RuntimeError("metric canonicalization failed")
        return best

    try:
        return search(base)[1]
    except _CanonicalBudgetExceeded:
        # Coordinate keys are inexpensive, deterministic, and independent of
        # the RNG. Sorting the block descriptors also preserves block order
        # invariance unless distinct paired rows have tied descriptors. No
        # inference depends on finding a unique geometric canonical form:
        # conditional on any one-to-one row order, uniform permutations have
        # exactly the required label/marginal-permutation distribution.
        def coordinate_key(
            row: int,
        ) -> tuple[int, tuple[tuple[int, tuple[float, ...]], ...]]:
            descriptors = tuple(
                sorted(
                    (sample.shape[1], tuple(float(value) for value in sample[row]))
                    for sample in samples
                )
            )
            return int(base[row]), descriptors

        return np.asarray(sorted(range(sample_size), key=coordinate_key), dtype=np.intp)


def pooled_groups(
    samples: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], tuple[NDArray[np.intp], ...]]:
    """Pool groups, sort rows, and recover their observed index allocation."""
    pooled = np.vstack(samples)
    labels = np.concatenate(
        [np.full(sample.shape[0], j, dtype=np.intp) for j, sample in enumerate(samples)]
    )
    # Equal-size groups receive the same initial colour: the statistics are
    # symmetric in group identity, and this makes swapping congruent groups a
    # no-op for the canonical plan.
    group_sizes = np.asarray([sample.shape[0] for sample in samples], dtype=np.int64)
    colors = np.concatenate(
        [
            np.full(sample.shape[0], group_sizes[j], dtype=np.int64)
            for j, sample in enumerate(samples)
        ]
    )
    membership = np.equal.outer(labels, labels).astype(np.int64)
    order = _canonical_metric_order(
        (pooled,),
        initial_colors=colors,
        extra_edge_layers=(membership,),
    )
    pooled = np.ascontiguousarray(pooled[order])
    labels = labels[order]
    indices = tuple(np.flatnonzero(labels == j) for j in range(len(samples)))
    return pooled, indices


def canonical_blocks(
    samples: Sequence[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], ...]:
    """Canonicalize marginal order and the common paired-row order."""
    row_order = canonical_paired_row_order(samples)
    ordered_samples = tuple(
        np.ascontiguousarray(sample[row_order]) for sample in samples
    )
    rank_layers = _distance_rank_layers(ordered_samples)
    ordered = tuple(
        sorted(
            zip(ordered_samples, rank_layers, strict=True),
            key=lambda bundle: (
                bundle[0].shape[1],
                bundle[1].tobytes(order="C"),
            ),
        )
    )
    return tuple(bundle[0] for bundle in ordered)


def canonical_block_keys(
    samples: Sequence[NDArray[np.float64]],
) -> tuple[tuple[int, bytes], ...]:
    """Return stable block-order keys after rows are already canonical."""
    ranks = _distance_rank_layers(samples)
    return tuple(
        (sample.shape[1], rank.tobytes(order="C"))
        for sample, rank in zip(samples, ranks, strict=True)
    )


def canonical_paired_row_order(
    samples: Sequence[NDArray[np.float64]],
) -> NDArray[np.intp]:
    """Return a metric canonical order for paired rows.

    The refinement operates on the complete edge-coloured graph whose edge
    colour is the sorted vector of normalized marginal distances.  Sorting
    within an edge removes marginal-order dependence.  Individualization is
    needed only for unresolved metric symmetries. When canonicalization fits
    its work budget, it preserves canonical distance tensors under isometries
    away from an approximate distance-rank boundary. At such a boundary, or
    after the bounded search falls back to coordinates, the plan can change
    but remains a uniform draw from the same randomization orbit.
    """
    return _canonical_metric_order(samples)


def distance_geometry(values: NDArray[np.float64]) -> DistanceGeometry:
    """Compute Euclidean distances after a safe common coordinate scaling."""
    # Centering before scaling preserves small, representable differences on a
    # huge common offset.  ``minimum / 2 + maximum / 2`` is an overflow-safe,
    # row-order-invariant midpoint even when the feature range itself exceeds
    # float64.  Subtracting a point inside [minimum, maximum] cannot overflow.
    minimum = np.min(values, axis=0)
    feature_maximum = np.max(values, axis=0)
    midpoint = minimum / 2.0 + feature_maximum / 2.0
    centered = values - midpoint
    coordinate_scale = float(np.max(np.abs(centered)))
    if coordinate_scale == 0.0:
        return DistanceGeometry(
            distances=np.zeros((values.shape[0], values.shape[0]), dtype=np.float64),
            log_scale=-math.inf,
        )
    scaled = centered / coordinate_scale
    try:
        distances = _core.pairwise_distances(scaled, scaled)
    except OverflowError as exc:
        raise ValueError("pairwise distances could not be represented") from exc
    distance_maximum = float(np.max(distances))
    if not math.isfinite(distance_maximum):
        raise ValueError("pairwise distances could not be represented")
    if distance_maximum == 0.0:
        return DistanceGeometry(
            distances=np.zeros_like(distances),
            log_scale=-math.inf,
        )
    distances = np.asarray(distances / distance_maximum, dtype=np.float64, order="C")
    return DistanceGeometry(
        distances=distances,
        log_scale=math.log(coordinate_scale) + math.log(distance_maximum),
    )


def double_center(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Double-center a square matrix without explicitly constructing ``H``."""
    row_means = np.mean(matrix, axis=1, keepdims=True, dtype=np.float64)
    column_means = np.mean(matrix, axis=0, keepdims=True, dtype=np.float64)
    grand_mean = float(np.mean(matrix, dtype=np.float64))
    return matrix - row_means - column_means + grand_mean


def kernel_matrix(
    geometry: DistanceGeometry,
    *,
    kernel: object,
    bandwidth: object,
) -> tuple[NDArray[np.float64], Kernel, str, float | None]:
    """Construct one supported characteristic kernel from fixed distances."""
    selected = validate_choice(
        kernel,
        name="kernel",
        choices=("rbf", "laplacian"),
    )
    distances = geometry.distances
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    positive = upper[upper > 0.0]

    numeric_bandwidth: float | None
    if isinstance(bandwidth, str):
        mode = validate_choice(
            bandwidth,
            name="bandwidth",
            choices=("median",),
        )
        if positive.size == 0:
            raise ValueError(
                "median bandwidth requires at least two distinct observations"
            )
        normalized_bandwidth = float(np.median(positive))
        log_ratio = np.full_like(distances, -math.inf)
        mask = distances > 0.0
        log_ratio[mask] = np.log(distances[mask] / normalized_bandwidth)
        bandwidth_label = mode
        numeric_bandwidth = None
    else:
        value = validate_real_scalar(bandwidth, name="bandwidth")
        if value <= 0.0:
            raise ValueError("bandwidth must be greater than 0")
        log_ratio = np.full_like(distances, -math.inf)
        mask = distances > 0.0
        log_ratio[mask] = np.log(distances[mask]) + geometry.log_scale - math.log(value)
        bandwidth_label = "explicit"
        numeric_bandwidth = value

    if selected == "laplacian":
        ratio = np.empty_like(log_ratio)
        too_large = log_ratio > math.log(np.finfo(np.float64).max)
        ratio[too_large] = math.inf
        ratio[~too_large] = np.exp(log_ratio[~too_large])
        gram = np.exp(-ratio)
    else:
        squared_ratio = np.empty_like(log_ratio)
        too_large = log_ratio > 0.5 * math.log(np.finfo(np.float64).max)
        squared_ratio[too_large] = math.inf
        squared_ratio[~too_large] = np.exp(2.0 * log_ratio[~too_large])
        gram = np.exp(-0.5 * squared_ratio)
    np.fill_diagonal(gram, 1.0)
    return (
        np.asarray(gram, dtype=np.float64),
        cast(Kernel, selected),
        bandwidth_label,
        numeric_bandwidth,
    )


def validate_calibration(value: object) -> Calibration:
    """Validate the common randomization selector."""
    return validate_choice(
        value,
        name="calibration",
        choices=("permutation", "exact", "monte-carlo"),
    )


def at_least_as_extreme(candidate: float, observed: float) -> bool:
    """Include upper-tail numerical ties without broad absolute tolerances."""
    if math.isnan(candidate) or math.isnan(observed):
        raise ValueError("resampling statistics must not be NaN")
    if observed == math.inf:
        return candidate == math.inf
    if observed == -math.inf:
        return True
    if math.isinf(candidate):
        return candidate > observed
    # Kernel V-statistics can subtract terms of order one to produce a value
    # close to zero.  Scaling by at least one therefore reflects the arithmetic
    # error of the terms being combined, while remaining near machine precision.
    tolerance = _TIE_RTOL * max(1.0, abs(candidate), abs(observed))
    return bool(candidate >= observed - tolerance)


def _group_allocations(
    population: tuple[int, ...], sizes: tuple[int, ...]
) -> Iterator[tuple[NDArray[np.intp], ...]]:
    """Enumerate every ordered fixed-size allocation exactly once."""
    if len(sizes) == 1:
        yield (np.asarray(population, dtype=np.intp),)
        return
    size = sizes[0]
    for chosen in combinations(population, size):
        chosen_set = set(chosen)
        remaining = tuple(index for index in population if index not in chosen_set)
        for tail in _group_allocations(remaining, sizes[1:]):
            yield (np.asarray(chosen, dtype=np.intp),) + tail


def group_orbit_size(sizes: Sequence[int]) -> int:
    """Return ``N! / prod(n_g!)`` using exact integer arithmetic."""
    total = sum(sizes)
    result = math.factorial(total)
    for size in sizes:
        result //= math.factorial(size)
    return result


def calibrate_groups(
    *,
    observed: float,
    statistic: GroupStatistic,
    batch_statistic: GroupBatchStatistic | None = None,
    sizes: tuple[int, ...],
    calibration: object,
    n_resamples: object,
    rng: int | np.integer | np.random.Generator | None,
) -> CalibrationSummary:
    """Calibrate a fixed-size group-label statistic."""
    selected = validate_calibration(calibration)
    budget = validate_positive_integer(n_resamples, name="n_resamples")
    if math.isnan(observed):
        raise ValueError("observed statistic must not be NaN")
    generator = make_generator(rng)
    total = group_orbit_size(sizes)
    use_exact = selected == "exact" or (selected == "permutation" and total <= budget)
    if selected == "exact" and total > budget:
        raise ValueError(
            f"exact calibration requires {total:,} allocations; increase "
            "n_resamples to at least that value"
        )
    exceedances = 0
    population = tuple(range(sum(sizes)))
    if use_exact and batch_statistic is None:
        for allocation in _group_allocations(population, sizes):
            exceedances += int(at_least_as_extreme(statistic(allocation), observed))
        return CalibrationSummary(
            pvalue=exact_pvalue(exceedances, total),
            n_resamples=total,
            exceedances=exceedances,
            exact=True,
            standard_error=None,
            interval=None,
            label="exact permutation",
        )

    total_size = sum(sizes)
    if use_exact:
        effective_resamples = total
        allocation_iterator = _group_allocations(population, sizes)
    else:
        effective_resamples = budget
        allocation_iterator = None

    batch_size = _quadratic_batch_size(
        total_size,
        working_matrices=max(4, len(sizes) + 2),
    )
    processed = 0
    while processed < effective_resamples:
        current_size = min(batch_size, effective_resamples - processed)
        labels = np.empty((current_size, total_size), dtype=np.intp)
        if use_exact:
            assert allocation_iterator is not None
            for row in range(current_size):
                allocation = next(allocation_iterator)
                for group, indices in enumerate(allocation):
                    labels[row, indices] = group
        else:
            boundaries = np.cumsum((0,) + sizes)
            base_labels = np.empty(total_size, dtype=np.intp)
            for group in range(len(sizes)):
                base_labels[boundaries[group] : boundaries[group + 1]] = group
            candidates = np.broadcast_to(base_labels, (current_size, total_size)).copy()
            labels = np.asarray(generator.permuted(candidates, axis=1), dtype=np.intp)
        if batch_statistic is None:  # pragma: no cover - retained private fallback
            values = np.asarray(
                [
                    statistic(
                        tuple(
                            np.flatnonzero(label_row == group)
                            for group in range(len(sizes))
                        )
                    )
                    for label_row in labels
                ]
            )
        else:
            values = np.asarray(batch_statistic(labels), dtype=np.float64)
        if values.shape != (current_size,):
            raise RuntimeError("batched group statistic returned an invalid shape")
        if np.any(np.isnan(values)):
            raise ValueError("resampled statistics must not contain NaN")
        if observed == math.inf:
            exceedances += int(np.count_nonzero(values == math.inf))
        elif observed == -math.inf:
            exceedances += current_size
        else:
            finite = np.isfinite(values)
            exceedances += int(np.count_nonzero(values == math.inf))
            finite_values = values[finite]
            tolerance = _TIE_RTOL * np.maximum.reduce(
                (
                    np.ones_like(finite_values),
                    np.abs(finite_values),
                    np.full_like(finite_values, abs(observed)),
                )
            )
            exceedances += int(np.count_nonzero(finite_values >= observed - tolerance))
        processed += current_size

    if use_exact:
        return CalibrationSummary(
            pvalue=exact_pvalue(exceedances, total),
            n_resamples=total,
            exceedances=exceedances,
            exact=True,
            standard_error=None,
            interval=None,
            label="exact permutation",
        )
    pvalue, standard_error, interval = monte_carlo_calibration(exceedances, budget)
    return CalibrationSummary(
        pvalue=pvalue,
        n_resamples=budget,
        exceedances=exceedances,
        exact=False,
        standard_error=standard_error,
        interval=interval,
        label="Monte Carlo permutation",
    )


def independence_orbit_size(sample_size: int, block_count: int) -> int:
    """Return the nonredundant joint-independence permutation orbit size."""
    return int(math.factorial(sample_size) ** (block_count - 1))


def calibrate_blocks(
    *,
    observed: float,
    statistic: BlockStatistic,
    batch_statistic: BlockBatchStatistic | None = None,
    sample_size: int,
    block_count: int,
    calibration: object,
    n_resamples: object,
    rng: int | np.integer | np.random.Generator | None,
) -> CalibrationSummary:
    """Calibrate independence by independently permuting marginal rows.

    A common row permutation is redundant, so the first block is fixed and the
    remaining ``block_count - 1`` permutations are enumerated or sampled.
    """
    selected = validate_calibration(calibration)
    budget = validate_positive_integer(n_resamples, name="n_resamples")
    if math.isnan(observed):
        raise ValueError("observed statistic must not be NaN")
    generator = make_generator(rng)
    total = independence_orbit_size(sample_size, block_count)
    use_exact = selected == "exact" or (selected == "permutation" and total <= budget)
    if selected == "exact" and total > budget:
        raise ValueError(
            f"exact calibration requires {total:,} marginal alignments; increase "
            "n_resamples to at least that value"
        )
    identity = np.arange(sample_size, dtype=np.intp)
    exceedances = 0
    if use_exact and batch_statistic is None:
        all_permutations = permutations(range(sample_size))
        for tails in product(all_permutations, repeat=block_count - 1):
            indices = (identity,) + tuple(
                np.fromiter(tail, dtype=np.intp, count=sample_size) for tail in tails
            )
            exceedances += int(at_least_as_extreme(statistic(indices), observed))
        return CalibrationSummary(
            pvalue=exact_pvalue(exceedances, total),
            n_resamples=total,
            exceedances=exceedances,
            exact=True,
            standard_error=None,
            interval=None,
            label="exact marginal permutation",
        )

    if use_exact:
        effective_resamples = total
        tail_iterator = product(
            permutations(range(sample_size)), repeat=block_count - 1
        )
    else:
        effective_resamples = budget
        tail_iterator = None

    batch_size = _quadratic_batch_size(
        sample_size,
        working_matrices=max(4, 2 * block_count + 1),
    )
    processed = 0
    while processed < effective_resamples:
        current_size = min(batch_size, effective_resamples - processed)
        plans = np.empty((current_size, block_count, sample_size), dtype=np.intp)
        plans[:, 0, :] = identity
        if use_exact:
            assert tail_iterator is not None
            filled = 0
            for filled, tails in enumerate(
                (next(tail_iterator) for _ in range(current_size)), start=1
            ):
                for block, tail in enumerate(tails, start=1):
                    plans[filled - 1, block] = tail
            if filled != current_size:  # pragma: no cover - orbit size is exact
                raise RuntimeError("exact marginal orbit ended unexpectedly")
        else:
            candidates = np.broadcast_to(
                identity,
                (current_size, block_count - 1, sample_size),
            ).copy()
            plans[:, 1:, :] = generator.permuted(candidates, axis=2)
        if batch_statistic is None:  # pragma: no cover - retained private fallback
            values = np.asarray(
                [
                    statistic(tuple(plan[block] for block in range(block_count)))
                    for plan in plans
                ]
            )
        else:
            values = np.asarray(batch_statistic(plans), dtype=np.float64)
        if values.shape != (current_size,):
            raise RuntimeError("batched statistic returned an invalid shape")
        if np.any(np.isnan(values)):
            raise ValueError("resampled statistics must not contain NaN")
        if observed == math.inf:
            exceedances += int(np.count_nonzero(values == math.inf))
        elif observed == -math.inf:
            exceedances += current_size
        else:
            finite = np.isfinite(values)
            exceedances += int(np.count_nonzero(values == math.inf))
            finite_values = values[finite]
            tolerance = _TIE_RTOL * np.maximum.reduce(
                (
                    np.ones_like(finite_values),
                    np.abs(finite_values),
                    np.full_like(finite_values, abs(observed)),
                )
            )
            exceedances += int(np.count_nonzero(finite_values >= observed - tolerance))
        processed += current_size

    if use_exact:
        return CalibrationSummary(
            pvalue=exact_pvalue(exceedances, total),
            n_resamples=total,
            exceedances=exceedances,
            exact=True,
            standard_error=None,
            interval=None,
            label="exact marginal permutation",
        )
    pvalue, standard_error, interval = monte_carlo_calibration(exceedances, budget)
    return CalibrationSummary(
        pvalue=pvalue,
        n_resamples=budget,
        exceedances=exceedances,
        exact=False,
        standard_error=standard_error,
        interval=interval,
        label="Monte Carlo marginal permutation",
    )


__all__: list[str] = []

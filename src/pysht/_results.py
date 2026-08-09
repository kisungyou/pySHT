"""Result objects returned by statistical tests.

The classes in this module deliberately separate a test's computed values from
their presentation.  They are immutable value objects, while their string
representation follows the compact, human-readable style of R's ``htest``
objects.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import SupportsFloat, SupportsIndex, cast

import numpy as np

type DegreesOfFreedom = float | tuple[float, ...]
type ConfidenceInterval = tuple[float, float]
type NamedEstimate = tuple[str, float]


def _coerce_float(value: object, *, field_name: str) -> float:
    """Return *value* as a float, rejecting booleans and non-numeric values."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be a real number, not bool")
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a real number")
    try:
        return float(cast(str | SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a real number") from exc


def _format_number(value: float) -> str:
    """Format a scalar compactly without hiding small values."""
    if value == math.inf:
        return "Inf"
    if value == -math.inf:
        return "-Inf"
    if value == 0:
        return "0"
    return format(value, ".6g")


def _format_df(df: DegreesOfFreedom) -> str:
    if isinstance(df, tuple):
        values = ", ".join(_format_number(value) for value in df)
        return f"({values})"
    return _format_number(df)


def _validate_optional_label(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_required_label(value: object, *, field_name: str) -> str:
    result = _validate_optional_label(value, field_name=field_name)
    if result is None:
        raise TypeError(f"{field_name} must be a string")
    return result


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class HypothesisTestResult:
    """Immutable result of a frequentist hypothesis test.

    Parameters are keyword-only so that adding optional presentation metadata
    does not make positional construction ambiguous.
    """

    statistic: float
    pvalue: float
    method: str
    alternative: str
    data_name: str | None = None
    statistic_name: str = "statistic"
    calibration: str | None = None
    df: DegreesOfFreedom | None = None
    confidence_interval: ConfidenceInterval | None = None
    confidence_level: float | None = None
    estimates: tuple[NamedEstimate, ...] = ()

    def __post_init__(self) -> None:
        statistic = _coerce_float(self.statistic, field_name="statistic")
        if math.isnan(statistic):
            raise ValueError("statistic must not be NaN")

        pvalue = _coerce_float(self.pvalue, field_name="pvalue")
        if not math.isfinite(pvalue) or not 0.0 <= pvalue <= 1.0:
            raise ValueError("pvalue must be finite and between 0 and 1")

        method = _validate_required_label(self.method, field_name="method")
        alternative = _validate_required_label(
            self.alternative, field_name="alternative"
        )
        data_name = _validate_optional_label(self.data_name, field_name="data_name")
        statistic_name = _validate_required_label(
            self.statistic_name, field_name="statistic_name"
        )
        calibration = _validate_optional_label(
            self.calibration, field_name="calibration"
        )

        df = self.df
        if df is not None:
            raw_values = df if isinstance(df, tuple) else (df,)
            if not raw_values:
                raise ValueError("df must not be an empty tuple")
            values = tuple(
                _coerce_float(value, field_name="df") for value in raw_values
            )
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError("df values must be finite and greater than 0")
            df = values if isinstance(df, tuple) else values[0]

        confidence_interval = self.confidence_interval
        confidence_level = self.confidence_level
        if confidence_interval is None:
            if confidence_level is not None:
                raise ValueError("confidence_level requires a confidence_interval")
        else:
            if (
                not isinstance(confidence_interval, tuple)
                or len(confidence_interval) != 2
            ):
                raise TypeError("confidence_interval must be a (lower, upper) tuple")
            lower = _coerce_float(
                confidence_interval[0], field_name="confidence_interval"
            )
            upper = _coerce_float(
                confidence_interval[1], field_name="confidence_interval"
            )
            if math.isnan(lower) or math.isnan(upper):
                raise ValueError("confidence_interval bounds must not be NaN")
            if lower > upper:
                raise ValueError(
                    "confidence_interval lower bound must not exceed upper bound"
                )
            confidence_interval = (lower, upper)
            if confidence_level is None:
                raise ValueError("confidence_interval requires a confidence_level")
            confidence_level = _coerce_float(
                confidence_level, field_name="confidence_level"
            )
            if not math.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
                raise ValueError("confidence_level must be finite and between 0 and 1")

        if not isinstance(self.estimates, tuple):
            raise TypeError("estimates must be a tuple of (name, value) pairs")
        estimates: list[NamedEstimate] = []
        seen_estimate_names: set[str] = set()
        for item in self.estimates:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("each estimate must be a (name, value) tuple")
            name = _validate_required_label(item[0], field_name="estimate name")
            if name in seen_estimate_names:
                raise ValueError(f"duplicate estimate name: {name!r}")
            value = _coerce_float(item[1], field_name=f"estimate {name!r}")
            if not math.isfinite(value):
                raise ValueError("estimate values must be finite")
            seen_estimate_names.add(name)
            estimates.append((name, value))

        object.__setattr__(self, "statistic", statistic)
        object.__setattr__(self, "pvalue", pvalue)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "alternative", alternative)
        object.__setattr__(self, "data_name", data_name)
        object.__setattr__(self, "statistic_name", statistic_name)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "confidence_interval", confidence_interval)
        object.__setattr__(self, "confidence_level", confidence_level)
        object.__setattr__(self, "estimates", tuple(estimates))

    @property
    def p_value(self) -> float:
        """Alias for :attr:`pvalue` using the spelling common in prose."""
        return self.pvalue

    def _statistic_line(self) -> str:
        fields = [f"{self.statistic_name} = {_format_number(self.statistic)}"]
        if self.df is not None:
            fields.append(f"df = {_format_df(self.df)}")
        fields.append(f"p-value = {_format_number(self.pvalue)}")
        return ", ".join(fields)

    def _extra_lines(self) -> tuple[str, ...]:
        lines: tuple[str, ...] = ()
        if self.confidence_interval is not None:
            assert self.confidence_level is not None
            percentage = _format_number(100.0 * self.confidence_level)
            lower, upper = self.confidence_interval
            lines += (
                f"{percentage} percent confidence interval:",
                f" {_format_number(lower)} {_format_number(upper)}",
            )
        if self.estimates:
            lines += ("sample estimates:",)
            lines += tuple(
                f"{name} = {_format_number(value)}" for name, value in self.estimates
            )
        return lines

    def _format(self) -> str:
        body: list[str] = []
        if self.data_name is not None:
            body.append(f"data: {self.data_name}")
        body.append(self._statistic_line())
        body.append(f"alternative hypothesis: {self.alternative}")
        if self.calibration is not None:
            body.append(f"calibration: {self.calibration}")
        body.extend(self._extra_lines())
        return f"{self.method}\n\n" + "\n".join(body)

    def __str__(self) -> str:
        return self._format()

    def __repr__(self) -> str:
        return self._format()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ResamplingTestResult(HypothesisTestResult):
    """Immutable test result with Monte Carlo or permutation diagnostics."""

    n_resamples: int
    exceedances: int
    monte_carlo_standard_error: float | None = None

    def __post_init__(self) -> None:
        HypothesisTestResult.__post_init__(self)

        if isinstance(self.n_resamples, bool):
            raise TypeError("n_resamples must be an integer, not bool")
        if isinstance(self.exceedances, bool):
            raise TypeError("exceedances must be an integer, not bool")
        try:
            n_resamples = operator.index(self.n_resamples)
        except TypeError as exc:
            raise TypeError("n_resamples must be an integer") from exc
        try:
            exceedances = operator.index(self.exceedances)
        except TypeError as exc:
            raise TypeError("exceedances must be an integer") from exc

        if n_resamples <= 0:
            raise ValueError("n_resamples must be greater than 0")
        if not 0 <= exceedances <= n_resamples:
            raise ValueError("exceedances must be between 0 and n_resamples")

        standard_error = self.monte_carlo_standard_error
        if standard_error is not None:
            standard_error = _coerce_float(
                standard_error, field_name="monte_carlo_standard_error"
            )
            if not math.isfinite(standard_error) or standard_error < 0:
                raise ValueError(
                    "monte_carlo_standard_error must be finite and non-negative"
                )

        object.__setattr__(self, "n_resamples", n_resamples)
        object.__setattr__(self, "exceedances", exceedances)
        object.__setattr__(self, "monte_carlo_standard_error", standard_error)

    def _extra_lines(self) -> tuple[str, ...]:
        exceedance_label = "exceedance" if self.exceedances == 1 else "exceedances"
        lines: tuple[str, ...] = HypothesisTestResult._extra_lines(self) + (
            (
                "resampling: "
                f"{self.n_resamples:,} resamples, "
                f"{self.exceedances:,} {exceedance_label}"
            ),
        )
        if self.monte_carlo_standard_error is not None:
            lines += (
                (
                    "Monte Carlo standard error: "
                    f"{_format_number(self.monte_carlo_standard_error)}"
                ),
            )
        return lines


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DistanceTestResult(ResamplingTestResult):
    """Resampling result retaining numerical distance normalization metadata."""

    normalized_statistic: float
    distance_scale: float
    exact: bool = False

    def __post_init__(self) -> None:
        ResamplingTestResult.__post_init__(self)
        normalized_statistic = _coerce_float(
            self.normalized_statistic, field_name="normalized_statistic"
        )
        if not math.isfinite(normalized_statistic) or normalized_statistic < 0.0:
            raise ValueError("normalized_statistic must be finite and non-negative")
        distance_scale = _coerce_float(self.distance_scale, field_name="distance_scale")
        if math.isnan(distance_scale) or distance_scale < 0.0:
            raise ValueError("distance_scale must be non-negative and not NaN")
        if not isinstance(self.exact, (bool, np.bool_)):
            raise TypeError("exact must be a boolean")
        exact = bool(self.exact)
        if exact:
            expected_pvalue = self.exceedances / self.n_resamples
            if self.monte_carlo_standard_error is not None:
                raise ValueError(
                    "an exact result must not have a Monte Carlo standard error"
                )
        else:
            expected_pvalue = (self.exceedances + 1.0) / (self.n_resamples + 1.0)
        if not math.isclose(self.pvalue, expected_pvalue, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                "pvalue is inconsistent with the permutation exceedance count"
            )
        object.__setattr__(self, "normalized_statistic", normalized_statistic)
        object.__setattr__(self, "distance_scale", distance_scale)
        object.__setattr__(self, "exact", exact)

    def _extra_lines(self) -> tuple[str, ...]:
        if self.exact:
            lines = HypothesisTestResult._extra_lines(self) + (
                (
                    "enumeration: "
                    f"{self.n_resamples:,} labelings, "
                    f"{self.exceedances:,} at least as extreme"
                ),
            )
        else:
            lines = ResamplingTestResult._extra_lines(self)
        return lines + (
            (
                "numerical normalization: "
                f"T_mn / d_max^2 = {_format_number(self.normalized_statistic)}, "
                f"d_max = {_format_number(self.distance_scale)}"
            ),
        )


__all__ = ["DistanceTestResult", "HypothesisTestResult", "ResamplingTestResult"]

"""Statistical conventions shared by historical risk estimators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

DEFAULT_OBSERVATIONS_PER_YEAR = 252
DEFAULT_ROLLING_WINDOW = 21
SAMPLE_DDOF = 1
MINIMUM_SAMPLE_OBSERVATIONS = SAMPLE_DDOF + 1
MINIMUM_RETURN_SUMMARY_OBSERVATIONS = 4
DEFAULT_QUANTILES = (0.05, 0.25, 0.75, 0.95)
DEFAULT_QUANTILE_METHOD = "linear"
SUPPORTED_QUANTILE_METHODS = frozenset(
    {"linear", "lower", "higher", "midpoint", "nearest"}
)
DEFAULT_DOWNSIDE_TARGET = 0.0
QuantileMethod = Literal["linear", "lower", "higher", "midpoint", "nearest"]


def validate_positive_integer(value: object, name: str) -> int:
    """Return a validated positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_quantiles(quantiles: object) -> tuple[float, ...]:
    """Return sorted, unique empirical probabilities strictly between zero and one."""
    if isinstance(quantiles, (str, bytes)) or not isinstance(quantiles, Sequence):
        raise ValueError("QUANTILES must be a non-empty sequence of numbers.")
    validated: list[float] = []
    for value in quantiles:
        if isinstance(value, bool):
            raise ValueError("Each QUANTILES value must be strictly between 0 and 1.")
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Each QUANTILES value must be strictly between 0 and 1."
            ) from exc
        if not 0.0 < probability < 1.0:
            raise ValueError("Each QUANTILES value must be strictly between 0 and 1.")
        validated.append(probability)
    if not validated:
        raise ValueError("QUANTILES must contain at least one probability.")
    return tuple(sorted(set(validated)))


def validate_quantile_method(method: object) -> QuantileMethod:
    """Return a supported empirical-quantile interpolation method."""
    normalized = str(method).strip().lower()
    if normalized not in SUPPORTED_QUANTILE_METHODS:
        choices = ", ".join(sorted(SUPPORTED_QUANTILE_METHODS))
        raise ValueError(f"QUANTILE_METHOD must be one of: {choices}.")
    return cast(QuantileMethod, normalized)


def validate_finite_number(value: Any, name: str) -> float:
    """Return a finite floating-point configuration value."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be a finite number.")
    return result


def resolve_rolling_min_observations(
    rolling_window: int,
    rolling_min_observations: int | None,
) -> int:
    """Validate a trailing window and resolve its minimum sample size."""
    window = validate_positive_integer(rolling_window, "ROLLING_WINDOW")
    minimum = window if rolling_min_observations is None else rolling_min_observations
    validate_positive_integer(minimum, "ROLLING_MIN_OBSERVATIONS")
    if minimum < MINIMUM_SAMPLE_OBSERVATIONS:
        raise ValueError(
            "ROLLING_MIN_OBSERVATIONS must be at least 2 for sample estimators."
        )
    if minimum > window:
        raise ValueError("ROLLING_MIN_OBSERVATIONS must not exceed ROLLING_WINDOW.")
    return minimum


def validate_rolling_sample(
    sample_size: int,
    rolling_window: int,
    rolling_min_observations: int | None,
) -> int:
    """Validate rolling parameters and ensure the sample can meet the minimum."""
    minimum = resolve_rolling_min_observations(rolling_window, rolling_min_observations)
    if sample_size < minimum:
        raise ValueError(
            "Insufficient return observations for rolling estimation: "
            f"need at least {minimum}, but received {sample_size}."
        )
    return minimum


def estimation_conventions(
    observations_per_year: int,
    rolling_window: int,
    rolling_min_observations: int,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    quantile_method: str = DEFAULT_QUANTILE_METHOD,
    downside_target: float = DEFAULT_DOWNSIDE_TARGET,
) -> dict[str, Any]:
    """Return machine-readable conventions for run manifests."""
    return {
        "return_input": {
            "simple_returns": "reported as a separate output only",
            "log_returns": [
                "return_summary",
                "volatility",
                "covariance",
                "pearson_correlation",
            ],
        },
        "sample_estimators": {
            "standard_deviation_ddof": SAMPLE_DDOF,
            "covariance_ddof": SAMPLE_DDOF,
            "pearson_correlation": "sample-centered; undefined for constant series",
            "skewness": (
                "bias-corrected Fisher-Pearson sample coefficient; requires at "
                "least 3 observations"
            ),
            "kurtosis": (
                "bias-corrected Fisher excess kurtosis; normal distribution equals "
                "0; requires at least 4 observations"
            ),
        },
        "return_summary": {
            "minimum_non_missing_observations_per_instrument": (
                MINIMUM_RETURN_SUMMARY_OBSERVATIONS
            ),
            "reason": (
                "the combined summary includes bias-corrected skewness and "
                "excess kurtosis"
            ),
        },
        "return_distribution": {
            "return_type": "daily_log_return",
            "quantiles": list(quantiles),
            "quantile_method": quantile_method,
            "downside_deviation": {
                "target": downside_target,
                "target_return_type": "daily_log_return",
                "denominator": "all_non_missing_observations",
                "formula": "sqrt(sum(min(return - target, 0)^2) / n)",
                "annualized": False,
            },
            "interpretation": "historical_description_not_forecast",
        },
        "annualization": {
            "observations_per_year": observations_per_year,
            "volatility_scaling": "square_root_of_time",
            "covariance_output": "daily_not_annualized",
        },
        "rolling": {
            "orientation": "trailing_including_current_observation",
            "window": rolling_window,
            "minimum_observations": rolling_min_observations,
            "initial_values": (
                "NaN until minimum_observations; partial trailing windows are "
                "estimated thereafter when minimum_observations < window"
            ),
            "future_observations_used": False,
        },
        "missing_data": {
            "alignment": "complete_case_adjusted_prices",
            "forward_fill": False,
            "gap_spanning_return_policy": (
                "fail_when_retained_complete_case_dates_are_nonconsecutive_within_"
                "the_provider_observation_union"
            ),
        },
    }


__all__ = [
    "DEFAULT_DOWNSIDE_TARGET",
    "DEFAULT_OBSERVATIONS_PER_YEAR",
    "DEFAULT_QUANTILES",
    "DEFAULT_QUANTILE_METHOD",
    "DEFAULT_ROLLING_WINDOW",
    "MINIMUM_RETURN_SUMMARY_OBSERVATIONS",
    "MINIMUM_SAMPLE_OBSERVATIONS",
    "SAMPLE_DDOF",
    "SUPPORTED_QUANTILE_METHODS",
    "QuantileMethod",
    "estimation_conventions",
    "resolve_rolling_min_observations",
    "validate_finite_number",
    "validate_positive_integer",
    "validate_quantile_method",
    "validate_quantiles",
    "validate_rolling_sample",
]

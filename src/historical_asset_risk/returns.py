"""Vectorized simple-return and logarithmic-return calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from historical_asset_risk.estimation import (
    DEFAULT_DOWNSIDE_TARGET,
    DEFAULT_QUANTILE_METHOD,
    DEFAULT_QUANTILES,
    MINIMUM_RETURN_SUMMARY_OBSERVATIONS,
    validate_finite_number,
    validate_quantile_method,
    validate_quantiles,
)


def _validate_prices(prices: pd.DataFrame) -> None:
    if prices.empty:
        raise ValueError("The price matrix is empty.")
    if (prices <= 0).any(axis=None):
        raise ValueError("All adjusted prices must be strictly positive.")


def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage changes and remove invalid observations."""
    _validate_prices(prices)
    returns = prices.pct_change(fill_method=None)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if returns.empty:
        raise ValueError("The simple-return matrix is empty after cleaning.")
    return returns


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-additive log returns and remove invalid observations."""
    _validate_prices(prices)
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if returns.empty:
        raise ValueError("The log-return matrix is empty after cleaning.")
    return returns


def _quantile_column(probability: float) -> str:
    return f"quantile_{format(probability, '.15g')}"


def summarize_returns(
    log_returns: pd.DataFrame,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    quantile_method: str = DEFAULT_QUANTILE_METHOD,
    downside_target: float = DEFAULT_DOWNSIDE_TARGET,
) -> pd.DataFrame:
    """Describe daily log returns using explicit empirical estimators."""
    if log_returns.empty:
        raise ValueError("Log returns are required for distribution statistics.")
    observation_counts = log_returns.count()
    insufficient = observation_counts.loc[
        observation_counts < MINIMUM_RETURN_SUMMARY_OBSERVATIONS
    ]
    if not insufficient.empty:
        details = ", ".join(
            f"{instrument} ({int(count)})" for instrument, count in insufficient.items()
        )
        raise ValueError(
            f"At least {MINIMUM_RETURN_SUMMARY_OBSERVATIONS} non-missing log-return "
            "observations per instrument are required for bias-corrected skewness "
            "and excess kurtosis; received "
            f"{details}."
        )
    probabilities = validate_quantiles(quantiles)
    method = validate_quantile_method(quantile_method)
    target = validate_finite_number(downside_target, "DOWNSIDE_TARGET")

    summary = log_returns.agg(["count", "mean", "std", "min"]).T.rename(
        columns={
            "count": "observation_count",
            "mean": "mean_daily_return",
            "std": "standard_deviation",
            "min": "minimum_return",
        }
    )
    empirical_quantiles = log_returns.quantile(
        probabilities, axis=0, interpolation=method
    )
    for probability in probabilities:
        summary[_quantile_column(probability)] = empirical_quantiles.loc[probability]

    shortfalls = (log_returns - target).clip(upper=0.0)
    non_missing_count = log_returns.count()
    summary["median_return"] = log_returns.median()
    summary["maximum_return"] = log_returns.max()
    summary["sample_skewness"] = log_returns.skew()
    summary["sample_excess_kurtosis"] = log_returns.kurt()
    summary["downside_deviation"] = np.sqrt(shortfalls.pow(2).sum() / non_missing_count)
    return summary

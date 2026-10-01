"""Daily, annualized, and rolling volatility calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from historical_asset_risk.estimation import (
    DEFAULT_OBSERVATIONS_PER_YEAR,
    DEFAULT_ROLLING_WINDOW,
    SAMPLE_DDOF,
    validate_positive_integer,
    validate_rolling_sample,
)


def volatility_summary(
    log_returns: pd.DataFrame,
    observations_per_year: int = DEFAULT_OBSERVATIONS_PER_YEAR,
) -> pd.DataFrame:
    """Calculate sample daily volatility and square-root-of-time annualization."""
    if log_returns.empty:
        raise ValueError("Log returns are required for volatility calculations.")
    validate_positive_integer(observations_per_year, "OBSERVATIONS_PER_YEAR")
    daily = log_returns.std(ddof=SAMPLE_DDOF)
    annualized = daily * np.sqrt(observations_per_year)
    return pd.DataFrame(
        {"daily_volatility": daily, "annualized_volatility": annualized}
    ).rename_axis("ticker")


def rolling_volatility(
    log_returns: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    observations_per_year: int = DEFAULT_OBSERVATIONS_PER_YEAR,
    rolling_min_observations: int | None = None,
) -> pd.DataFrame:
    """Calculate trailing annualized sample volatility without look-ahead."""
    validate_positive_integer(observations_per_year, "OBSERVATIONS_PER_YEAR")
    minimum = validate_rolling_sample(
        len(log_returns), rolling_window, rolling_min_observations
    )
    rolling = log_returns.rolling(window=rolling_window, min_periods=minimum).std(
        ddof=SAMPLE_DDOF
    )
    return rolling * np.sqrt(observations_per_year)


__all__ = ["rolling_volatility", "volatility_summary"]

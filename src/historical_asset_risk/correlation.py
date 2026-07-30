"""Complete-case covariance and Pearson correlation estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from historical_asset_risk.estimation import (
    DEFAULT_ROLLING_WINDOW,
    MINIMUM_SAMPLE_OBSERVATIONS,
    SAMPLE_DDOF,
    validate_rolling_sample,
)


def align_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Keep dates that contain a valid log return for every selected asset."""
    if log_returns.shape[1] < 2:
        raise ValueError("At least two assets are required for correlation analysis.")
    aligned = log_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if aligned.empty:
        raise ValueError("The aligned return matrix is empty after cleaning.")
    return aligned


def correlation_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlation from complete-case daily log returns."""
    aligned = _dependence_sample(log_returns)
    return aligned.corr(method="pearson")


def covariance_matrix(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate the sample covariance matrix (ddof=1) of daily log returns."""
    aligned = _dependence_sample(log_returns)
    return aligned.cov(ddof=SAMPLE_DDOF)


def _dependence_sample(log_returns: pd.DataFrame) -> pd.DataFrame:
    aligned = align_returns(log_returns)
    if len(aligned) < MINIMUM_SAMPLE_OBSERVATIONS:
        raise ValueError(
            "At least two complete return observations are required for "
            "dependence estimation."
        )
    return aligned


def rolling_covariance(
    log_returns: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    rolling_min_observations: int | None = None,
) -> pd.DataFrame:
    """Calculate trailing sample covariance without using future observations."""
    aligned = align_returns(log_returns)
    minimum = validate_rolling_sample(
        len(aligned), rolling_window, rolling_min_observations
    )
    result = aligned.rolling(window=rolling_window, min_periods=minimum).cov(
        pairwise=True, ddof=SAMPLE_DDOF
    )
    result.index.names = [aligned.index.name or "date", "ticker"]
    return result


def rolling_correlation(
    log_returns: pd.DataFrame,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    rolling_min_observations: int | None = None,
) -> pd.DataFrame:
    """Calculate trailing Pearson correlation without future observations."""
    aligned = align_returns(log_returns)
    minimum = validate_rolling_sample(
        len(aligned), rolling_window, rolling_min_observations
    )
    result = aligned.rolling(window=rolling_window, min_periods=minimum).corr(
        pairwise=True
    )
    result.index.names = [aligned.index.name or "date", "ticker"]
    return result


def extreme_correlation_pairs(
    correlations: pd.DataFrame,
) -> tuple[tuple[str, str, float], tuple[str, str, float]]:
    """Return the highest and lowest unique off-diagonal asset pairs."""
    if correlations.shape[0] < 2 or correlations.shape[0] != correlations.shape[1]:
        raise ValueError(
            "A square correlation matrix with at least two assets is required."
        )
    mask = np.triu(np.ones(correlations.shape, dtype=bool), k=1)
    pairs = correlations.where(mask).stack()
    if pairs.empty:
        raise ValueError("No off-diagonal correlation pairs are available.")
    highest_pair = pairs.idxmax()
    lowest_pair = pairs.idxmin()
    highest = (
        str(highest_pair[0]),
        str(highest_pair[1]),
        float(pairs.loc[highest_pair]),
    )
    lowest = (str(lowest_pair[0]), str(lowest_pair[1]), float(pairs.loc[lowest_pair]))
    return highest, lowest

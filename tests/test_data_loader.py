"""Offline tests for strict adjusted-price ingestion and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from historical_asset_risk.data_loader import (
    MarketDataError,
    _extract_adjusted_close,
    clean_adjusted_prices,
    validate_configuration,
)


def test_configuration_normalizes_symbols_and_rejects_bad_inputs() -> None:
    assert validate_configuration(
        [" spy ", "QQQ", "spy"], "2024-01-01", "2024-12-31", 21
    ) == ["SPY", "QQQ"]
    with pytest.raises(ValueError, match="At least two"):
        validate_configuration(["SPY"], "2024-01-01", "2024-12-31", 21)
    with pytest.raises(ValueError, match="earlier"):
        validate_configuration(["SPY", "QQQ"], "2025-01-01", "2024-01-01", 21)
    with pytest.raises(ValueError, match="positive integer"):
        validate_configuration(["SPY", "QQQ"], "2024-01-01", "2025-01-01", 0)


def test_extract_adjusted_close_never_falls_back_to_raw_close() -> None:
    raw = pd.DataFrame({"Close": [100.0, 101.0]})
    with pytest.raises(MarketDataError, match="Raw Close will not be used"):
        _extract_adjusted_close(raw, ["SPY"])


def test_clean_prices_sorts_deduplicates_and_converts_values() -> None:
    dates = pd.to_datetime(
        [
            "2024-01-05",
            "2024-01-03",
            "2024-01-03",
            "2024-01-04",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    raw = pd.DataFrame(
        {
            "AAA": ["103", "100", "101", "102", "104", "105"],
            "BBB": ["53", "50", "51", "52", "54", "55"],
        },
        index=dates,
    )
    cleaned = clean_adjusted_prices(raw, ["AAA", "BBB"], rolling_window=2)
    assert cleaned.index.is_monotonic_increasing
    assert cleaned.index.is_unique
    assert not cleaned.isna().any(axis=None)
    assert cleaned.loc[pd.Timestamp("2024-01-03"), "AAA"] == 101


def test_clean_prices_rejects_gap_spanning_daily_returns() -> None:
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    prices = pd.DataFrame(
        {
            "AAA": [100, 101, 102, 103, 104, 105],
            "BBB": [50, 51, None, 53, 54, 55],
        },
        index=dates,
    )

    with pytest.raises(MarketDataError, match="gap-spanning returns"):
        clean_adjusted_prices(prices, ["AAA", "BBB"], rolling_window=2)


def test_clean_prices_reports_invalid_ticker_and_insufficient_history() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    missing = pd.DataFrame({"AAA": range(5), "BBB": [None] * 5}, index=dates)
    with pytest.raises(MarketDataError, match="BBB"):
        clean_adjusted_prices(missing, ["AAA", "BBB"], rolling_window=2)
    short = pd.DataFrame({"AAA": [1, 2], "BBB": [2, 3]}, index=dates[:2])
    with pytest.raises(MarketDataError, match="Insufficient common price history"):
        clean_adjusted_prices(short, ["AAA", "BBB"], rolling_window=2)

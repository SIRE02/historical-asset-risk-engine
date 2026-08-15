"""Provider-independent adjusted-price normalization and validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from historical_asset_risk.config import AnalysisConfig
from historical_asset_risk.providers import (
    CANONICAL_COLUMNS,
    MarketDataError,
    MarketDataProvider,
    ProviderPayload,
    YahooFinanceProvider,
    extract_yahoo_adjusted_close,
)


@dataclass(frozen=True)
class MarketDataResult:
    """Validated prices and the lineage/quality evidence used to obtain them."""

    prices: pd.DataFrame
    canonical_records: pd.DataFrame
    payload: ProviderPayload
    quality_report: dict[str, Any]


def validate_configuration(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    rolling_window: int,
) -> list[str]:
    """Validate legacy loader arguments and return normalized symbols."""
    if isinstance(tickers, (str, bytes)):
        raise ValueError("TICKERS must be a sequence containing at least two symbols.")
    normalized = list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers))
    normalized = [ticker for ticker in normalized if ticker]
    if len(normalized) < 2:
        raise ValueError("At least two unique ticker symbols are required.")
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except (TypeError, ValueError) as exc:
        raise ValueError("START_DATE and END_DATE must use YYYY-MM-DD format.") from exc
    if start >= end:
        raise ValueError("START_DATE must be earlier than END_DATE.")
    if (
        isinstance(rolling_window, bool)
        or not isinstance(rolling_window, int)
        or rolling_window <= 0
    ):
        raise ValueError("ROLLING_WINDOW must be a positive integer.")
    return normalized


def _canonical_from_wide(adjusted_prices: pd.DataFrame) -> pd.DataFrame:
    frame = adjusted_prices.copy()
    frame.index.name = "date"
    return frame.reset_index().melt(
        id_vars="date", var_name="ticker", value_name="adjusted_close"
    )


def normalize_and_validate(
    records: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Normalize canonical records, validate prices, and produce quality evidence."""
    missing_columns = [column for column in CANONICAL_COLUMNS if column not in records]
    if missing_columns:
        raise MarketDataError(
            "Provider data is missing canonical column(s): "
            + ", ".join(missing_columns)
        )

    data = records.loc[:, list(CANONICAL_COLUMNS)].copy()
    source_rows = len(data)
    source_missing_prices = int(data["adjusted_close"].isna().sum())
    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    data["date"] = parsed_dates
    data["ticker"] = data["ticker"].astype("string").str.strip().str.upper()
    requested = list(config.tickers)
    data = data.loc[data["ticker"].isin(requested)].copy()

    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    data = data.loc[
        data["date"].notna() & (data["date"] >= start) & (data["date"] < end)
    ]
    duplicate_rows = int(data.duplicated(subset=["date", "ticker"], keep="last").sum())
    data = data.drop_duplicates(subset=["date", "ticker"], keep="last")

    source_prices = data["adjusted_close"]
    scoped_missing_prices = int(source_prices.isna().sum())
    numeric_prices = pd.to_numeric(source_prices, errors="coerce")
    nonnumeric_prices = source_prices.notna() & numeric_prices.isna()
    nonpositive_prices = numeric_prices.notna() & (numeric_prices <= 0)
    invalid_prices = int((nonnumeric_prices | nonpositive_prices).sum())
    data["adjusted_close"] = numeric_prices.mask(nonpositive_prices)
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

    returned = sorted(
        data.loc[data["adjusted_close"].notna(), "ticker"].unique().tolist()
    )
    unusable = [ticker for ticker in requested if ticker not in returned]
    if unusable:
        raise MarketDataError(
            "No usable adjusted-price data was returned for: "
            + ", ".join(unusable)
            + "."
        )

    wide = data.pivot(index="date", columns="ticker", values="adjusted_close")
    wide = wide.reindex(columns=requested).sort_index()
    aligned = wide.dropna(how="any")
    union_index = wide.index
    union_position = {
        timestamp: position for position, timestamp in enumerate(union_index)
    }
    gap_spanning_intervals = [
        (previous, current)
        for previous, current in zip(aligned.index, aligned.index[1:], strict=False)
        if union_position[current] - union_position[previous] != 1
    ]
    if gap_spanning_intervals:
        examples = ", ".join(
            f"{start.date().isoformat()} to {end.date().isoformat()}"
            for start, end in gap_spanning_intervals[:3]
        )
        raise MarketDataError(
            "Complete-case alignment would create gap-spanning returns between "
            "nonconsecutive provider observation dates: "
            f"{examples}. Missing interior observations must not be treated as "
            "daily returns."
        )
    rolling_minimum = config.rolling_min_observations
    # AnalysisConfig resolves this during validation.
    assert rolling_minimum is not None
    minimum_prices = rolling_minimum + 1
    if len(aligned) < minimum_prices:
        raise MarketDataError(
            "Insufficient common price history: "
            f"need at least {minimum_prices} aligned prices for "
            f"ROLLING_MIN_OBSERVATIONS={rolling_minimum}, "
            f"but received {len(aligned)}."
        )
    aligned.columns.name = None
    aligned.index.name = "date"

    union_dates = int(len(union_index))
    instruments: list[dict[str, Any]] = []
    for ticker in requested:
        ticker_rows = data.loc[data["ticker"] == ticker]
        valid_before = int(wide[ticker].notna().sum())
        instruments.append(
            {
                "ticker": ticker,
                "rows_after_date_filter": int(len(ticker_rows)),
                "valid_observations_before_alignment": valid_before,
                "missing_values_before_alignment": int(wide[ticker].isna().sum()),
                "valid_observations_after_alignment": int(len(aligned)),
                "common_history_reduction": valid_before - int(len(aligned)),
            }
        )

    report: dict[str, Any] = {
        "requested_instruments": requested,
        "returned_instruments": returned,
        "source_row_count": source_rows,
        "invalid_date_count": invalid_dates,
        "duplicate_date_instrument_rows_removed": duplicate_rows,
        "source_missing_adjusted_close_values": source_missing_prices,
        "missing_adjusted_close_values": scoped_missing_prices,
        "invalid_price_values_removed": invalid_prices,
        "union_date_count_before_alignment": union_dates,
        "common_date_count_after_alignment": int(len(aligned)),
        "common_history_rows_removed": union_dates - int(len(aligned)),
        "first_common_date": aligned.index.min().date().isoformat(),
        "last_common_date": aligned.index.max().date().isoformat(),
        "instruments": instruments,
    }
    return aligned, data, report


def load_market_data(
    config: AnalysisConfig,
    provider: MarketDataProvider,
) -> MarketDataResult:
    """Acquire, provider-normalize, common-normalize, and validate market data."""
    payload = provider.acquire(config)
    canonical = provider.normalize(payload, config.tickers)
    prices, normalized_records, quality = normalize_and_validate(canonical, config)
    return MarketDataResult(prices, normalized_records, payload, quality)


def persist_acquisition(records: pd.DataFrame, path: Path) -> None:
    """Save normalized pre-alignment records for explicit later CSV reuse."""
    if all(column in records.columns for column in CANONICAL_COLUMNS):
        canonical = records.loc[:, list(CANONICAL_COLUMNS)].copy()
    else:
        canonical = _canonical_from_wide(records).loc[:, list(CANONICAL_COLUMNS)]
    canonical = canonical.sort_values(["date", "ticker"], kind="stable")
    canonical.to_csv(path, index=False, date_format="%Y-%m-%d")


def persist_quality_report(report: dict[str, Any], path: Path) -> None:
    """Persist a machine-readable data-quality report."""
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def clean_adjusted_prices(
    adjusted_prices: pd.DataFrame,
    tickers: Sequence[str],
    rolling_window: int,
) -> pd.DataFrame:
    """Compatibility wrapper for provider-independent wide price validation."""
    symbols = tuple(str(ticker).strip().upper() for ticker in tickers)
    valid_dates = pd.DatetimeIndex(
        pd.to_datetime(adjusted_prices.index, errors="coerce")
    ).dropna()
    if len(valid_dates) == 0:
        raise MarketDataError("The adjusted-price data contains no valid dates.")
    start = valid_dates.min().date().isoformat()
    end = (valid_dates.max().date() + timedelta(days=1)).isoformat()
    config = AnalysisConfig(
        tickers=symbols,
        start_date=start,
        end_date=end,
        rolling_window=rolling_window,
    )
    prices, _records, _quality = normalize_and_validate(
        _canonical_from_wide(adjusted_prices), config
    )
    return prices


def _extract_adjusted_close(raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Compatibility alias for Yahoo adjusted-close extraction."""
    return extract_yahoo_adjusted_close(raw, list(tickers))


def download_adjusted_prices(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    rolling_window: int,
) -> pd.DataFrame:
    """Compatibility wrapper for an explicit Yahoo provider run."""
    symbols = validate_configuration(tickers, start_date, end_date, rolling_window)
    config = AnalysisConfig(
        provider="yahoo",
        tickers=tuple(symbols),
        start_date=start_date,
        end_date=end_date,
        rolling_window=rolling_window,
    )
    return load_market_data(config, YahooFinanceProvider()).prices

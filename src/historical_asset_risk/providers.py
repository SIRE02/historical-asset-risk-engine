"""Market-data provider interfaces and concrete adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import yfinance as yf

from historical_asset_risk.config import AnalysisConfig

CANONICAL_COLUMNS = ("date", "ticker", "adjusted_close")


class MarketDataError(RuntimeError):
    """Raised when usable adjusted-price data cannot be obtained."""


@dataclass(frozen=True)
class ProviderPayload:
    """Unprocessed provider response plus acquisition lineage."""

    data: pd.DataFrame
    provider: str
    source: str
    acquired_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    """Common interface implemented by every adjusted-price provider."""

    name: str

    def acquire(self, config: AnalysisConfig) -> ProviderPayload:
        """Read or download the provider-specific payload."""
        ...

    def normalize(
        self, payload: ProviderPayload, requested_tickers: tuple[str, ...]
    ) -> pd.DataFrame:
        """Convert the payload to the canonical long-form record schema."""
        ...


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def extract_yahoo_adjusted_close(
    raw: pd.DataFrame, tickers: tuple[str, ...] | list[str]
) -> pd.DataFrame:
    """Extract adjusted closes without substituting raw closing prices."""
    if raw.empty:
        raise MarketDataError("The market-data provider returned an empty response.")

    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            if "Adj Close" in raw.columns.get_level_values(level):
                adjusted = raw.xs("Adj Close", axis=1, level=level, drop_level=True)
                break
        else:
            raise MarketDataError(
                "Adjusted Close is unavailable. Raw Close will not be used as a "
                "fallback."
            )
    else:
        if "Adj Close" not in raw.columns:
            raise MarketDataError(
                "Adjusted Close is unavailable. Raw Close will not be used as a "
                "fallback."
            )
        if len(tickers) != 1:
            raise MarketDataError(
                "The provider response did not identify every requested ticker."
            )
        adjusted = raw.loc[:, ["Adj Close"]].copy()
        adjusted.columns = [tickers[0]]

    if isinstance(adjusted, pd.Series):
        adjusted = adjusted.to_frame(name=tickers[0])
    adjusted.columns = [str(column).strip().upper() for column in adjusted.columns]
    adjusted.index.name = "date"
    return adjusted


def _wide_to_canonical(wide: pd.DataFrame) -> pd.DataFrame:
    frame = wide.copy()
    frame.index.name = "date"
    return (
        frame.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="adjusted_close")
        .loc[:, list(CANONICAL_COLUMNS)]
    )


class YahooFinanceProvider:
    """Acquire unadjusted Yahoo fields and select Yahoo's adjusted close."""

    name = "yahoo"

    def acquire(self, config: AnalysisConfig) -> ProviderPayload:
        try:
            raw = yf.download(
                tickers=list(config.tickers),
                start=config.start_date,
                end=config.end_date,
                auto_adjust=False,
                actions=False,
                progress=False,
                group_by="column",
                # Sequential requests avoid yfinance's shared SQLite cache races.
                # Acquisition is infrequent and reliability outweighs the small
                # speed benefit for the repository's modest ticker sets.
                threads=False,
            )
        except Exception as exc:  # yfinance exposes backend-specific exceptions
            raise MarketDataError(f"Market-data download failed: {exc}") from exc
        return ProviderPayload(
            data=raw,
            provider=self.name,
            source="Yahoo Finance via yfinance",
            acquired_at=_utc_now(),
            metadata={"adjustment_field": "Adj Close", "end_date_exclusive": True},
        )

    def normalize(
        self, payload: ProviderPayload, requested_tickers: tuple[str, ...]
    ) -> pd.DataFrame:
        return _wide_to_canonical(
            extract_yahoo_adjusted_close(payload.data, list(requested_tickers))
        )


class CSVProvider:
    """Read canonical adjusted-price records from a local CSV file."""

    name = "csv"

    def acquire(self, config: AnalysisConfig) -> ProviderPayload:
        if config.csv_path is None:  # guarded by AnalysisConfig; defensive for callers
            raise MarketDataError("CSV input requires csv_path.")
        path = Path(config.csv_path)
        try:
            raw = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise MarketDataError(
                f"Could not read CSV market data from {path}: {exc}"
            ) from exc
        return ProviderPayload(
            data=raw,
            provider=self.name,
            source=str(path.resolve()),
            acquired_at=_utc_now(),
            metadata={
                "schema": list(CANONICAL_COLUMNS),
                "file_modified_at": datetime.fromtimestamp(
                    path.stat().st_mtime, UTC
                ).isoformat(),
            },
        )

    def normalize(
        self, payload: ProviderPayload, requested_tickers: tuple[str, ...]
    ) -> pd.DataFrame:
        missing = [
            column for column in CANONICAL_COLUMNS if column not in payload.data.columns
        ]
        if missing:
            raise MarketDataError(
                "CSV input is missing canonical column(s): " + ", ".join(missing)
            )
        return payload.data.loc[:, list(CANONICAL_COLUMNS)].copy()


def provider_for(config: AnalysisConfig) -> MarketDataProvider:
    """Construct the explicitly configured provider; no fallback is attempted."""
    if config.provider == "yahoo":
        return YahooFinanceProvider()
    if config.provider == "csv":
        return CSVProvider()
    raise ValueError(f"Unsupported provider: {config.provider}")

"""Deterministic tests for provider adapters and the common data pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from historical_asset_risk.config import AnalysisConfig
from historical_asset_risk.data_loader import (
    load_market_data,
    normalize_and_validate,
    persist_acquisition,
)
from historical_asset_risk.providers import (
    CSVProvider,
    MarketDataError,
    ProviderPayload,
    YahooFinanceProvider,
)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 104.0, 103.0],
            "BBB": [50.0, 51.0, 52.0, 53.0, 54.0],
        },
        index=pd.date_range("2024-01-02", periods=5, freq="B"),
    )


def _canonical(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame.index.name = "date"
    return frame.reset_index().melt(
        id_vars="date", var_name="ticker", value_name="adjusted_close"
    )


class SavedYahooProvider(YahooFinanceProvider):
    def __init__(self, raw: pd.DataFrame) -> None:
        self.raw = raw

    def acquire(self, config: AnalysisConfig) -> ProviderPayload:
        return ProviderPayload(
            data=self.raw,
            provider="yahoo",
            source="saved yfinance response",
            acquired_at="2024-02-01T00:00:00+00:00",
        )


def test_yahoo_and_csv_produce_identical_validated_prices(tmp_path: Path) -> None:
    prices = _prices()
    raw = pd.concat({"Adj Close": prices, "Close": prices + 0.5}, axis=1)
    csv_path = tmp_path / "prices.csv"
    _canonical(prices).to_csv(csv_path, index=False)
    common = {
        "tickers": ("AAA", "BBB"),
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "rolling_window": 2,
    }

    yahoo = load_market_data(
        AnalysisConfig(provider="yahoo", **common), SavedYahooProvider(raw)
    )
    csv = load_market_data(
        AnalysisConfig(provider="csv", csv_path=csv_path, **common), CSVProvider()
    )

    pd.testing.assert_frame_equal(yahoo.prices, csv.prices)
    assert yahoo.payload.provider == "yahoo"
    assert csv.payload.provider == "csv"


def test_yahoo_acquisition_disables_internal_download_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _prices()
    raw = pd.concat({"Adj Close": prices, "Close": prices + 0.5}, axis=1)
    captured: dict[str, object] = {}

    def saved_download(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return raw

    monkeypatch.setattr(
        "historical_asset_risk.providers.yf.download",
        saved_download,
    )
    config = AnalysisConfig(
        provider="yahoo",
        tickers=("AAA", "BBB"),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=2,
    )

    payload = YahooFinanceProvider().acquire(config)

    assert captured == {
        "tickers": ["AAA", "BBB"],
        "start": config.start_date,
        "end": config.end_date,
        "auto_adjust": False,
        "actions": False,
        "progress": False,
        "group_by": "column",
        "threads": False,
    }
    assert payload.data is raw
    assert payload.provider == "yahoo"
    assert payload.metadata == {
        "adjustment_field": "Adj Close",
        "end_date_exclusive": True,
    }


def test_quality_report_counts_duplicates_invalid_prices_and_alignment() -> None:
    records = _canonical(_prices())
    records = pd.concat(
        [
            records,
            pd.DataFrame(
                [
                    {"date": "2024-01-02", "ticker": "AAA", "adjusted_close": 100.5},
                    {"date": "2024-01-08", "ticker": "BBB", "adjusted_close": -1},
                ]
            ),
        ],
        ignore_index=True,
    )
    config = AnalysisConfig(
        tickers=("AAA", "BBB"),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=2,
    )
    prices, _normalized, quality = normalize_and_validate(records, config)

    assert quality["duplicate_date_instrument_rows_removed"] == 2
    assert quality["invalid_price_values_removed"] == 1
    assert quality["common_history_rows_removed"] == 1
    assert len(prices) == 4


def test_quality_missing_count_is_scoped_to_requested_in_range_rows() -> None:
    records = pd.concat(
        [
            _canonical(_prices()),
            pd.DataFrame(
                [
                    {
                        "date": "2024-01-03",
                        "ticker": "ZZZ",
                        "adjusted_close": 1.0,
                    },
                    {
                        "date": "2023-12-01",
                        "ticker": "AAA",
                        "adjusted_close": 1.0,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    records.loc[records.index[-2:], "adjusted_close"] = None
    config = AnalysisConfig(
        tickers=("AAA", "BBB"),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=2,
    )

    _prices_result, _normalized, quality = normalize_and_validate(records, config)

    assert quality["source_missing_adjusted_close_values"] == 2
    assert quality["missing_adjusted_close_values"] == 0


def test_csv_schema_is_strict_and_yahoo_errors_do_not_fallback(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2024-01-01"], "AAA": [100]}).to_csv(csv_path, index=False)
    config = AnalysisConfig(
        provider="csv",
        csv_path=csv_path,
        tickers=("AAA", "BBB"),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=2,
    )
    payload = CSVProvider().acquire(config)
    with pytest.raises(MarketDataError, match="canonical column"):
        CSVProvider().normalize(payload, config.tickers)

    yahoo_raw = pd.DataFrame({"Close": [100.0, 101.0]})
    with pytest.raises(MarketDataError, match="Raw Close will not be used"):
        YahooFinanceProvider().normalize(
            ProviderPayload(yahoo_raw, "yahoo", "saved", "now"), ("AAA",)
        )


def test_persisted_yahoo_acquisition_is_reusable_as_csv(tmp_path: Path) -> None:
    prices = _prices()
    csv_path = tmp_path / "acquired_adjusted_prices.csv"
    persist_acquisition(prices, csv_path)
    config = AnalysisConfig(
        provider="csv",
        csv_path=csv_path,
        tickers=("AAA", "BBB"),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=2,
    )

    reloaded = load_market_data(config, CSVProvider()).prices

    pd.testing.assert_frame_equal(
        reloaded, prices.rename_axis("date"), check_freq=False
    )


def test_persisted_acquisition_retains_pre_alignment_missing_rows(
    tmp_path: Path,
) -> None:
    records = _canonical(_prices())
    missing_date = pd.Timestamp("2024-01-08")
    records.loc[
        (records["date"] == missing_date) & (records["ticker"] == "BBB"),
        "adjusted_close",
    ] = None
    csv_path = tmp_path / "acquired_adjusted_prices.csv"

    persist_acquisition(records, csv_path)

    persisted = pd.read_csv(csv_path, parse_dates=["date"])
    retained = persisted.loc[
        (persisted["date"] == missing_date) & (persisted["ticker"] == "BBB"),
        "adjusted_close",
    ]
    assert len(retained) == 1
    assert retained.isna().all()


def test_configured_rolling_minimum_controls_the_history_gate() -> None:
    records = _canonical(_prices().iloc[:3])
    config = AnalysisConfig(
        tickers=("AAA", "BBB"),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=5,
        rolling_min_observations=2,
    )

    prices, _normalized, _quality = normalize_and_validate(records, config)

    assert len(prices) == 3

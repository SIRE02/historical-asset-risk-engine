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
    price_content_hash,
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
    last_bbb = (records["ticker"] == "BBB") & (records["date"] == "2024-01-08")
    records.loc[last_bbb, "adjusted_close"] = -1
    records = pd.concat(
        [
            records,
            pd.DataFrame(
                [{"date": "2024-01-02", "ticker": "AAA", "adjusted_close": 100.5}]
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

    assert quality["duplicate_date_instrument_rows_removed"] == 1
    assert quality["invalid_price_values_removed"] == 1
    assert quality["common_history_rows_removed"] == 1
    assert len(prices) == 4


def test_invalid_duplicate_never_replaces_a_valid_price() -> None:
    def duplicate(day: str, ticker: str, value: object) -> dict[str, object]:
        return {"date": day, "ticker": ticker, "adjusted_close": value}

    records = pd.concat(
        [
            _canonical(_prices()),
            pd.DataFrame(
                [
                    # Valid then invalid: the valid original survives.
                    duplicate("2024-01-03", "AAA", -1),
                    # Valid then nonnumeric: the valid original survives.
                    duplicate("2024-01-04", "BBB", "n/a"),
                    # Valid then valid: the later row is a correction and wins.
                    duplicate("2024-01-05", "AAA", 104.5),
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

    assert quality["duplicate_date_instrument_rows_removed"] == 3
    assert quality["invalid_price_values_removed"] == 0
    assert len(prices) == 5
    assert prices.loc["2024-01-03", "AAA"] == 101.0
    assert prices.loc["2024-01-04", "BBB"] == 52.0
    assert prices.loc["2024-01-05", "AAA"] == 104.5

    # Invalid first, valid last also keeps the valid row.
    reversed_order = pd.concat(
        [
            pd.DataFrame([duplicate("2024-01-03", "AAA", -1)]),
            _canonical(_prices()),
        ],
        ignore_index=True,
    )
    prices, _normalized, quality = normalize_and_validate(reversed_order, config)
    assert quality["invalid_price_values_removed"] == 0
    assert prices.loc["2024-01-03", "AAA"] == 101.0


def test_infinite_prices_are_invalid_not_valid_observations() -> None:
    records = _canonical(_prices()).astype({"adjusted_close": object})
    last_aaa = (records["ticker"] == "AAA") & (records["date"] == "2024-01-08")
    records.loc[last_aaa, "adjusted_close"] = "inf"
    config = AnalysisConfig(
        tickers=("AAA", "BBB"),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=2,
    )
    prices, _normalized, quality = normalize_and_validate(records, config)

    assert quality["invalid_price_values_removed"] == 1
    assert len(prices) == 4
    assert prices.index.max() == pd.Timestamp("2024-01-05")

    interior = _canonical(_prices())
    middle_aaa = (interior["ticker"] == "AAA") & (interior["date"] == "2024-01-04")
    interior.loc[middle_aaa, "adjusted_close"] = float("inf")
    with pytest.raises(MarketDataError, match="gap-spanning"):
        normalize_and_validate(interior, config)


def test_intraday_observations_are_rejected_not_treated_as_sessions() -> None:
    config = AnalysisConfig(
        tickers=("AAA", "BBB"),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=2,
    )
    # Morning and afternoon prices on every session: distinct timestamps, so
    # deduplication alone would keep both and halve the return interval.
    morning = _canonical(_prices())
    morning["date"] = morning["date"].dt.strftime("%Y-%m-%d 10:00")
    afternoon = _canonical(_prices())
    afternoon["date"] = afternoon["date"].dt.strftime("%Y-%m-%d 15:00")
    intraday = pd.concat([morning, afternoon], ignore_index=True)
    with pytest.raises(MarketDataError, match="carry a time of day"):
        normalize_and_validate(intraday, config)

    # A single observation per session still fails if it carries a time.
    at_close = _canonical(_prices())
    at_close["date"] = at_close["date"].dt.strftime("%Y-%m-%d 16:00")
    with pytest.raises(MarketDataError, match="AAA 2024-01-02T16:00:00"):
        normalize_and_validate(at_close, config)

    # Midnight timestamps are session dates written with a time, and pass.
    midnight = _canonical(_prices())
    midnight["date"] = midnight["date"].dt.strftime("%Y-%m-%d 00:00:00")
    prices, _normalized, _quality = normalize_and_validate(midnight, config)
    assert len(prices) == 5


def test_price_content_hash_tracks_every_price_and_nothing_else() -> None:
    prices = _prices()
    revised = prices.copy()
    revised.iloc[2, 0] = 102.5

    assert price_content_hash(prices) == price_content_hash(prices.copy())
    assert price_content_hash(prices).startswith("sha256:")
    assert price_content_hash(prices) != price_content_hash(revised)
    assert price_content_hash(prices) != price_content_hash(prices[["BBB", "AAA"]])


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

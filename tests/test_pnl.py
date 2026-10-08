"""Known-value and invariant tests for return alignment and hypothetical P&L."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from historical_asset_risk.contracts import (
    Instrument,
    InstrumentType,
    PortfolioReturnAlignmentError,
)
from historical_asset_risk.market_calendar import resolve_market_calendar
from historical_asset_risk.pnl import (
    ProxyExposureSnapshot,
    align_portfolio_simple_returns,
    compute_data_snapshot_id,
    hypothetical_pnl,
    map_provider_tickers_to_instrument_ids,
    proxy_realized_pnl,
)


def _instrument(instrument_id: str, ticker: str) -> Instrument:
    return Instrument(
        instrument_id=instrument_id,
        provider_ticker=ticker,
        listing_venue="ARCX",
        instrument_type=InstrumentType.STANDARD_ETF,
        price_currency="USD",
        market_calendar_id="XNYS",
        market_timezone="America/New_York",
        adjusted_return_series_id=f"fixture:{ticker}",
        return_adjustment_basis="total-return adjusted close",
    )


INSTRUMENTS = (_instrument("US_SPY", "SPY"), _instrument("US_QQQ", "QQQ"))
PRICE_INDEX = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
SIMPLE_RETURNS = pd.DataFrame(
    {"SPY": [0.1, 0.1], "QQQ": [-0.1, 0.0], "TLT": [0.5, 0.5]},
    index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
)


def test_mapping_ignores_extra_columns_and_reports_missing_held_columns() -> None:
    mapping = map_provider_tickers_to_instrument_ids(
        INSTRUMENTS,
        ["SPY", "QQQ", "TLT"],
        required_instrument_ids=["US_SPY", "US_QQQ"],
    )
    assert mapping == {"SPY": "US_SPY", "QQQ": "US_QQQ"}

    with pytest.raises(PortfolioReturnAlignmentError, match="no simple-return column"):
        map_provider_tickers_to_instrument_ids(
            INSTRUMENTS, ["SPY"], required_instrument_ids=["US_SPY", "US_QQQ"]
        )
    with pytest.raises(PortfolioReturnAlignmentError, match="absent from the registry"):
        map_provider_tickers_to_instrument_ids(
            INSTRUMENTS, ["SPY", "QQQ"], required_instrument_ids=["US_ZZZ"]
        )


def test_align_recovers_interval_starts_and_long_form_rows() -> None:
    aligned = align_portfolio_simple_returns(
        SIMPLE_RETURNS,
        INSTRUMENTS,
        held_instrument_ids=["US_SPY", "US_QQQ"],
        price_index=PRICE_INDEX,
        market_calendar_id="XNYS",
        data_snapshot_id="data_test",
    )
    assert list(aligned.matrix.columns) == ["US_SPY", "US_QQQ"]
    assert aligned.period_start.iloc[0] == pd.Timestamp("2024-01-02")
    assert aligned.period_start.iloc[1] == pd.Timestamp("2024-01-03")

    long_frame = aligned.long_frame()
    assert set(long_frame["instrument_id"]) == {"US_SPY", "US_QQQ"}
    spy_first = long_frame.loc[long_frame["instrument_id"] == "US_SPY"].iloc[0]
    assert spy_first["period_start"] == "2024-01-02"
    assert spy_first["period_end"] == "2024-01-03"
    assert spy_first["simple_return"] == pytest.approx(0.1)
    assert (long_frame["market_calendar_id"] == "XNYS").all()


def test_align_fails_when_a_held_return_column_is_missing() -> None:
    with pytest.raises(PortfolioReturnAlignmentError, match="no simple-return column"):
        align_portfolio_simple_returns(
            SIMPLE_RETURNS.drop(columns=["QQQ"]),
            INSTRUMENTS,
            held_instrument_ids=["US_SPY", "US_QQQ"],
            price_index=PRICE_INDEX,
            market_calendar_id="XNYS",
            data_snapshot_id="data_test",
        )


def test_hypothetical_pnl_known_values_long_short() -> None:
    aligned = align_portfolio_simple_returns(
        SIMPLE_RETURNS,
        INSTRUMENTS,
        held_instrument_ids=["US_SPY", "US_QQQ"],
        price_index=PRICE_INDEX,
        market_calendar_id="XNYS",
        data_snapshot_id="data_test",
    )
    result = hypothetical_pnl({"US_SPY": 1000.0, "US_QQQ": -250.0}, aligned)
    assert result["hypothetical_pnl"].tolist() == pytest.approx([125.0, 100.0])
    assert result["hypothetical_loss"].tolist() == pytest.approx([-125.0, -100.0])
    assert result["period_start"].tolist() == ["2024-01-02", "2024-01-03"]


def test_hypothetical_pnl_scales_linearly_and_reorders_invariantly() -> None:
    aligned = align_portfolio_simple_returns(
        SIMPLE_RETURNS,
        INSTRUMENTS,
        held_instrument_ids=["US_SPY", "US_QQQ"],
        price_index=PRICE_INDEX,
        market_calendar_id="XNYS",
        data_snapshot_id="data_test",
    )
    base = hypothetical_pnl({"US_SPY": 1000.0, "US_QQQ": -250.0}, aligned)
    scaled = hypothetical_pnl({"US_SPY": 3000.0, "US_QQQ": -750.0}, aligned)
    assert scaled["hypothetical_pnl"].tolist() == pytest.approx(
        (base["hypothetical_pnl"] * 3.0).tolist()
    )
    reordered = hypothetical_pnl({"US_QQQ": -250.0, "US_SPY": 1000.0}, aligned)
    assert reordered["hypothetical_pnl"].tolist() == pytest.approx(
        base["hypothetical_pnl"].tolist()
    )


def test_hypothetical_pnl_missing_or_extra_exposure_fails() -> None:
    aligned = align_portfolio_simple_returns(
        SIMPLE_RETURNS,
        INSTRUMENTS,
        held_instrument_ids=["US_SPY", "US_QQQ"],
        price_index=PRICE_INDEX,
        market_calendar_id="XNYS",
        data_snapshot_id="data_test",
    )
    with pytest.raises(
        PortfolioReturnAlignmentError, match="missing a currency exposure"
    ):
        hypothetical_pnl({"US_SPY": 1000.0}, aligned)
    with pytest.raises(PortfolioReturnAlignmentError, match="absent from the aligned"):
        hypothetical_pnl({"US_SPY": 1.0, "US_QQQ": 1.0, "US_TLT": 1.0}, aligned)


def test_hypothetical_pnl_rejects_duplicate_return_columns() -> None:
    # One exposure applied to both "US_SPY" columns would double the P&L.
    duplicated = pd.DataFrame(
        [[0.01, 0.01]],
        columns=["US_SPY", "US_SPY"],
        index=pd.to_datetime(["2024-01-03"]),
    )
    with pytest.raises(PortfolioReturnAlignmentError, match="duplicate instrument"):
        hypothetical_pnl({"US_SPY": 100.0}, duplicated)


def test_cash_only_book_produces_zero_market_pnl() -> None:
    empty_matrix = pd.DataFrame(index=pd.to_datetime(["2024-01-03", "2024-01-04"]))
    result = hypothetical_pnl(
        {},
        empty_matrix,
        period_start={
            pd.Timestamp("2024-01-03"): pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-04"): pd.Timestamp("2024-01-03"),
        },
    )
    assert result["hypothetical_pnl"].tolist() == [0.0, 0.0]
    assert result["hypothetical_loss"].tolist() == [0.0, 0.0]


def _proxy_snapshot(as_of: str, exposures: dict[str, float]) -> ProxyExposureSnapshot:
    return ProxyExposureSnapshot(
        portfolio_snapshot_id=f"snap-{as_of}",
        portfolio_id="portfolio-1",
        exposure_snapshot_id=f"exp-{as_of}",
        as_of_date=date.fromisoformat(as_of),
        market_calendar_id="XNYS",
        currency_exposures=exposures,
    )


def test_proxy_realized_pnl_forward_aligns_to_next_session() -> None:
    aligned = align_portfolio_simple_returns(
        SIMPLE_RETURNS,
        INSTRUMENTS,
        held_instrument_ids=["US_SPY", "US_QQQ"],
        price_index=PRICE_INDEX,
        market_calendar_id="XNYS",
        data_snapshot_id="data_test",
    )
    calendar = resolve_market_calendar("XNYS")
    result = proxy_realized_pnl(
        [
            _proxy_snapshot("2024-01-02", {"US_SPY": 1000.0, "US_QQQ": -250.0}),
            _proxy_snapshot("2024-01-03", {"US_SPY": 2000.0, "US_QQQ": 0.0}),
            _proxy_snapshot("2024-01-04", {"US_SPY": 1.0, "US_QQQ": 1.0}),
        ],
        aligned,
        next_session=calendar.next_session,
    )
    by_date = result.set_index("as_of_date")
    assert by_date.loc["2024-01-02", "target_period_end"] == "2024-01-03"
    assert by_date.loc["2024-01-02", "proxy_realized_pnl"] == pytest.approx(125.0)
    assert by_date.loc["2024-01-02", "proxy_realized_loss"] == pytest.approx(-125.0)
    assert by_date.loc["2024-01-03", "proxy_realized_pnl"] == pytest.approx(200.0)
    # 2024-01-04 -> next session 2024-01-05 is outside the return sample.
    assert by_date.loc["2024-01-04", "outcome_status"] == "missing_target_return"
    assert pd.isna(by_date.loc["2024-01-04", "proxy_realized_pnl"])


def test_next_session_skips_the_weekend_not_civil_date_plus_one() -> None:
    calendar = resolve_market_calendar("XNYS")
    # 2024-01-05 is a Friday; the next session is Monday 2024-01-08.
    assert calendar.next_session(date(2024, 1, 5)) == date(2024, 1, 8)
    # 2024-11-28 is Thanksgiving; the Wednesday before rolls to the Friday after.
    assert calendar.next_session(date(2024, 11, 27)) == date(2024, 11, 29)


def test_proxy_realized_pnl_reports_missing_instrument_returns() -> None:
    aligned = align_portfolio_simple_returns(
        SIMPLE_RETURNS,
        INSTRUMENTS,
        held_instrument_ids=["US_SPY"],
        price_index=PRICE_INDEX,
        market_calendar_id="XNYS",
        data_snapshot_id="data_test",
    )
    calendar = resolve_market_calendar("XNYS")
    result = proxy_realized_pnl(
        [_proxy_snapshot("2024-01-02", {"US_SPY": 1000.0, "US_QQQ": -250.0})],
        aligned,
        next_session=calendar.next_session,
    )
    row = result.iloc[0]
    assert row["outcome_status"] == "missing_instrument_return"
    assert row["missing_instruments"] == "US_QQQ"
    assert pd.isna(row["proxy_realized_pnl"])


def test_proxy_on_the_last_calendar_session_is_recorded_not_raised() -> None:
    aligned = align_portfolio_simple_returns(
        SIMPLE_RETURNS,
        INSTRUMENTS,
        held_instrument_ids=["US_SPY"],
        price_index=PRICE_INDEX,
        market_calendar_id="XNYS",
        data_snapshot_id="data_test",
    )
    calendar = resolve_market_calendar("XNYS")
    assert calendar.is_session(date(2035, 12, 31))
    result = proxy_realized_pnl(
        [_proxy_snapshot("2035-12-31", {"US_SPY": 1000.0})],
        aligned,
        next_session=calendar.next_session,
    )
    row = result.iloc[0]
    assert row["outcome_status"] == "missing_target_return"
    assert row["target_period_end"] == ""
    assert pd.isna(row["proxy_realized_pnl"])


def test_data_snapshot_id_is_deterministic_and_content_sensitive() -> None:
    base = {
        "provider": "csv",
        "source": "fixture",
        "actual_start_date": "2024-01-02",
        "actual_end_date": "2024-01-08",
        "observation_count": 5,
        "instruments": ["SPY", "QQQ"],
        "price_content_hash": "sha256:" + "0" * 64,
    }
    assert compute_data_snapshot_id(base) == compute_data_snapshot_id(dict(base))
    changed = dict(base, observation_count=6)
    assert compute_data_snapshot_id(base) != compute_data_snapshot_id(changed)
    # Same source, dates, count and tickers, but a revised price.
    revised = dict(base, price_content_hash="sha256:" + "1" * 64)
    assert compute_data_snapshot_id(base) != compute_data_snapshot_id(revised)

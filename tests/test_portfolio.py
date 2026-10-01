"""Known-value, invariant, and validation tests for portfolio books."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime

import pytest

from historical_asset_risk.contracts import (
    Cash,
    Instrument,
    InstrumentType,
    NonPositivePortfolioValueError,
    PortfolioCalendarError,
    PortfolioCurrencyError,
    PortfolioDateError,
    PortfolioDuplicateError,
    PortfolioIdentifierError,
    PortfolioPriceError,
    PortfolioQuantityError,
    PortfolioTimestampError,
    PortfolioTimezoneError,
    Position,
)
from historical_asset_risk.market_calendar import resolve_market_calendar
from historical_asset_risk.portfolio import (
    calculate_currency_valuation,
    validate_portfolio_snapshot,
    value_portfolio,
)


@pytest.fixture
def instruments() -> tuple[Instrument, ...]:
    return (
        Instrument(
            instrument_id="US_SPY",
            provider_ticker="SPY",
            listing_venue="ARCX",
            instrument_type=InstrumentType.STANDARD_ETF,
            price_currency="USD",
            market_calendar_id="XNYS",
            market_timezone="America/New_York",
            adjusted_return_series_id="yahoo:SPY:adj-close",
            return_adjustment_basis="vendor split and distribution adjusted",
        ),
        Instrument(
            instrument_id="US_QQQ",
            provider_ticker="QQQ",
            listing_venue="XNAS",
            instrument_type=InstrumentType.STANDARD_ETF,
            price_currency="USD",
            market_calendar_id="XNYS",
            market_timezone="America/New_York",
            adjusted_return_series_id="yahoo:QQQ:adj-close",
            return_adjustment_basis="vendor split and distribution adjusted",
        ),
    )


@pytest.fixture
def positions() -> tuple[Position, ...]:
    timestamp = datetime.fromisoformat("2024-01-02T16:00:00-05:00")
    return (
        Position(
            portfolio_snapshot_id="snapshot-1",
            portfolio_id="portfolio-1",
            as_of_date=date(2024, 1, 2),
            instrument_id="US_SPY",
            quantity=10.0,
            valuation_price=100.0,
            valuation_price_timestamp=timestamp,
            valuation_price_source="official close",
            source_row_id="row-1",
        ),
        Position(
            portfolio_snapshot_id="snapshot-1",
            portfolio_id="portfolio-1",
            as_of_date=date(2024, 1, 2),
            instrument_id="US_QQQ",
            quantity=-5.0,
            valuation_price=50.0,
            valuation_price_timestamp=timestamp,
            valuation_price_source="official close",
            source_row_id="row-2",
        ),
    )


@pytest.fixture
def cash() -> Cash:
    return Cash(
        portfolio_snapshot_id="snapshot-1",
        portfolio_id="portfolio-1",
        as_of_date=date(2024, 1, 2),
        as_of_timestamp=datetime.fromisoformat("2024-01-02T16:05:00-05:00"),
        base_currency="USD",
        market_calendar_id="XNYS",
        market_timezone="America/New_York",
        cash_amount=500.0,
        cash_source="custodian",
    )


def test_long_short_known_values_and_reconciliation(
    instruments: tuple[Instrument, ...],
    positions: tuple[Position, ...],
    cash: Cash,
) -> None:
    snapshot = validate_portfolio_snapshot(
        instruments, positions, (cash,), ("SPY", "QQQ", "TLT")
    )
    result = value_portfolio(snapshot)

    assert [position.position_value for position in result.positions] == [
        1000.0,
        -250.0,
    ]
    assert result.long_exposure == 1000.0
    assert result.short_exposure == -250.0
    assert result.net_instrument_exposure == 750.0
    assert result.gross_exposure == 1250.0
    assert result.portfolio_value == 1250.0
    assert [position.weight for position in result.positions] == [0.8, -0.2]
    assert result.cash_weight == 0.4
    assert result.instrument_weight_sum + result.cash_weight == pytest.approx(1.0)
    assert result.gross_exposure_ratio == 1.0
    assert result.net_exposure_ratio == 0.6
    assert snapshot.reconciliation.extra_market_data_instruments == ("TLT",)


def test_one_asset_long_only_zero_and_negative_cash_cases(
    instruments: tuple[Instrument, ...],
    positions: tuple[Position, ...],
    cash: Cash,
) -> None:
    one_asset_position = replace(
        positions[0],
        quantity=2.0,
        valuation_price=100.0,
    )
    with_zero_cash = value_portfolio(
        validate_portfolio_snapshot(
            instruments[:1],
            (one_asset_position,),
            (replace(cash, cash_amount=0.0),),
            ("SPY",),
        )
    )
    assert with_zero_cash.portfolio_value == 200.0
    assert with_zero_cash.cash_weight == 0.0
    assert with_zero_cash.positions[0].weight == 1.0

    with_negative_cash = value_portfolio(
        validate_portfolio_snapshot(
            instruments[:1],
            (one_asset_position,),
            (replace(cash, cash_amount=-50.0),),
            ("SPY",),
        )
    )
    assert with_negative_cash.portfolio_value == 150.0
    assert with_negative_cash.cash_weight == pytest.approx(-1 / 3)
    assert with_negative_cash.positions[0].weight == pytest.approx(4 / 3)


def test_snapshot_identity_is_order_independent_and_materially_sensitive(
    instruments: tuple[Instrument, ...],
    positions: tuple[Position, ...],
    cash: Cash,
) -> None:
    first = value_portfolio(
        validate_portfolio_snapshot(instruments, positions, (cash,), ("SPY", "QQQ"))
    )
    reordered = value_portfolio(
        validate_portfolio_snapshot(
            tuple(reversed(instruments)),
            tuple(reversed(positions)),
            (cash,),
            ("QQQ", "SPY"),
        )
    )
    changed_positions = (replace(positions[0], quantity=11.0), positions[1])
    changed = value_portfolio(
        validate_portfolio_snapshot(
            instruments, changed_positions, (cash,), ("SPY", "QQQ")
        )
    )

    assert first.exposure_snapshot_id == reordered.exposure_snapshot_id
    assert first.exposure_snapshot_id != changed.exposure_snapshot_id
    assert first.portfolio_value == reordered.portfolio_value


def test_positive_scaling_preserves_weights_and_ratios(
    instruments: tuple[Instrument, ...],
    positions: tuple[Position, ...],
    cash: Cash,
) -> None:
    original = value_portfolio(
        validate_portfolio_snapshot(instruments, positions, (cash,), ("SPY", "QQQ"))
    )
    scaled_positions = tuple(
        replace(position, quantity=position.quantity * 3.0) for position in positions
    )
    scaled = value_portfolio(
        validate_portfolio_snapshot(
            instruments,
            scaled_positions,
            (replace(cash, cash_amount=cash.cash_amount * 3.0),),
            ("SPY", "QQQ"),
        )
    )

    assert scaled.portfolio_value == original.portfolio_value * 3.0
    assert scaled.gross_exposure == original.gross_exposure * 3.0
    assert [item.weight for item in scaled.positions] == pytest.approx(
        [item.weight for item in original.positions]
    )
    assert scaled.cash_weight == pytest.approx(original.cash_weight)
    assert scaled.gross_exposure_ratio == pytest.approx(original.gross_exposure_ratio)


def test_non_positive_nav_fails_weight_based_calculation(
    instruments: tuple[Instrument, ...],
    positions: tuple[Position, ...],
    cash: Cash,
) -> None:
    snapshot = validate_portfolio_snapshot(
        instruments,
        positions,
        (replace(cash, cash_amount=-750.0),),
        ("SPY", "QQQ"),
    )
    currency_valuation = calculate_currency_valuation(snapshot)
    assert currency_valuation.net_instrument_exposure == 750.0
    assert currency_valuation.portfolio_value == 0.0
    with pytest.raises(
        NonPositivePortfolioValueError, match="strictly positive portfolio value"
    ):
        value_portfolio(snapshot)


def test_duplicate_and_missing_instrument_failures_are_specific(
    instruments: tuple[Instrument, ...],
    positions: tuple[Position, ...],
    cash: Cash,
) -> None:
    with pytest.raises(PortfolioDuplicateError, match="snapshot/instrument"):
        validate_portfolio_snapshot(
            instruments, (positions[0], positions[0]), (cash,), ("SPY",)
        )
    with pytest.raises(PortfolioIdentifierError, match="missing"):
        validate_portfolio_snapshot(instruments, positions, (cash,), ("SPY",))


def test_currency_session_and_timestamp_failures(
    instruments: tuple[Instrument, ...],
    positions: tuple[Position, ...],
    cash: Cash,
) -> None:
    with pytest.raises(PortfolioCurrencyError, match="does not match"):
        validate_portfolio_snapshot(
            (replace(instruments[0], price_currency="EUR"), instruments[1]),
            positions,
            (cash,),
            ("SPY", "QQQ"),
        )

    holiday_cash = replace(
        cash,
        as_of_date=date(2024, 1, 1),
        as_of_timestamp=datetime.fromisoformat("2024-01-01T16:05:00-05:00"),
    )
    holiday_positions = tuple(
        replace(
            position,
            as_of_date=date(2024, 1, 1),
            valuation_price_timestamp=datetime.fromisoformat(
                "2024-01-01T16:00:00-05:00"
            ),
        )
        for position in positions
    )
    with pytest.raises(PortfolioDateError, match="not a valid"):
        validate_portfolio_snapshot(
            instruments, holiday_positions, (holiday_cash,), ("SPY", "QQQ")
        )

    late_position = replace(
        positions[0],
        valuation_price_timestamp=datetime.fromisoformat("2024-01-02T16:10:00-05:00"),
    )
    with pytest.raises(PortfolioTimestampError, match="later"):
        validate_portfolio_snapshot(
            instruments,
            (late_position, positions[1]),
            (cash,),
            ("SPY", "QQQ"),
        )

    wrong_offset = replace(
        positions[0],
        valuation_price_timestamp=datetime.fromisoformat("2024-01-02T16:00:00+01:00"),
    )
    with pytest.raises(PortfolioTimezoneError, match="offset inconsistent"):
        validate_portfolio_snapshot(
            instruments,
            (wrong_offset, positions[1]),
            (cash,),
            ("SPY", "QQQ"),
        )

    wrong_market_timezone = replace(cash, market_timezone="Asia/Kolkata")
    with pytest.raises(PortfolioTimezoneError, match="requires market_timezone"):
        validate_portfolio_snapshot(
            instruments,
            positions,
            (wrong_market_timezone,),
            ("SPY", "QQQ"),
        )


def test_xnys_does_not_observe_saturday_new_year_on_preceding_friday() -> None:
    calendar = resolve_market_calendar("XNYS")

    assert calendar.is_session(date(2021, 12, 31))
    assert calendar.is_session(date(2027, 12, 31))


def test_consecutive_sessions_allow_weekends_and_holidays_but_not_gaps() -> None:
    calendar = resolve_market_calendar("XNYS")
    # Fri 2024-01-12 -> Tue 2024-01-16 crosses a weekend and MLK Day: consecutive.
    calendar.require_consecutive_sessions(
        [date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 16)]
    )

    with pytest.raises(PortfolioCalendarError, match=r"missing 2024-01-08"):
        calendar.require_consecutive_sessions(
            [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 9)]
        )
    with pytest.raises(PortfolioCalendarError, match="non-XNYS sessions: 2024-01-15"):
        calendar.require_consecutive_sessions(
            [date(2024, 1, 12), date(2024, 1, 15), date(2024, 1, 16)]
        )


@pytest.mark.parametrize(
    ("field", "value", "error_type"),
    [
        ("quantity", 0.0, PortfolioQuantityError),
        ("quantity", float("inf"), PortfolioQuantityError),
        ("valuation_price", 0.0, PortfolioPriceError),
        ("valuation_price", -1.0, PortfolioPriceError),
        ("valuation_price", float("nan"), PortfolioPriceError),
    ],
)
def test_invalid_position_numbers_fail_explicitly(
    instruments: tuple[Instrument, ...],
    positions: tuple[Position, ...],
    cash: Cash,
    field: str,
    value: float,
    error_type: type[ValueError],
) -> None:
    invalid_position = replace(positions[0], **{field: value})
    with pytest.raises(error_type):
        validate_portfolio_snapshot(
            instruments,
            (invalid_position, positions[1]),
            (cash,),
            ("SPY", "QQQ"),
        )

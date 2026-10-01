"""Provider- and file-system-independent portfolio validation and valuation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from historical_asset_risk.contracts import (
    EXPOSURE_SNAPSHOT_SCHEMA_VERSION,
    Cash,
    CurrencyValuedPosition,
    Instrument,
    InstrumentEligibilityError,
    InstrumentType,
    NonPositivePortfolioValueError,
    PortfolioCalendarError,
    PortfolioCashError,
    PortfolioCurrencyError,
    PortfolioCurrencyValuation,
    PortfolioDateError,
    PortfolioDuplicateError,
    PortfolioIdentifierError,
    PortfolioPriceError,
    PortfolioQuantityError,
    PortfolioReconciliation,
    PortfolioTimestampError,
    PortfolioTimezoneError,
    PortfolioValuation,
    Position,
    ValidatedPortfolioSnapshot,
    ValuedPosition,
)
from historical_asset_risk.market_calendar import resolve_market_calendar


def _unique_by(
    records: Iterable[Instrument | Position],
    key_name: str,
    label: str,
) -> None:
    seen: set[str | tuple[str, str]] = set()
    for record in records:
        if key_name == "snapshot_instrument":
            assert isinstance(record, Position)
            key: str | tuple[str, str] = (
                record.portfolio_snapshot_id,
                record.instrument_id,
            )
        else:
            value = getattr(record, key_name)
            key = str(value)
        if key in seen:
            raise PortfolioDuplicateError(f"Duplicate {label}: {key!r}.")
        seen.add(key)


def _validate_timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise PortfolioTimezoneError(
            f"Unknown IANA market_timezone {value!r}."
        ) from exc


def _require_aware(timestamp: datetime, field_name: str) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise PortfolioTimestampError(f"{field_name} must be timezone-aware.")


def _required_text(value: str, field_name: str, *, upper: bool = False) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise PortfolioIdentifierError(f"{field_name} must be non-empty.")
    return normalized.upper() if upper else normalized


def _normalize_contracts(
    instruments: Sequence[Instrument],
    positions: Sequence[Position],
    cash_records: Sequence[Cash],
) -> tuple[tuple[Instrument, ...], tuple[Position, ...], tuple[Cash, ...]]:
    normalized_instruments = tuple(
        replace(
            instrument,
            instrument_id=_required_text(instrument.instrument_id, "instrument_id"),
            provider_ticker=_required_text(
                instrument.provider_ticker, "provider_ticker", upper=True
            ),
            listing_venue=_required_text(
                instrument.listing_venue, "listing_venue", upper=True
            ),
            price_currency=_required_text(
                instrument.price_currency, "price_currency", upper=True
            ),
            market_calendar_id=_required_text(
                instrument.market_calendar_id, "market_calendar_id", upper=True
            ),
            market_timezone=_required_text(
                instrument.market_timezone, "market_timezone"
            ),
            adjusted_return_series_id=_required_text(
                instrument.adjusted_return_series_id,
                "adjusted_return_series_id",
            ),
            return_adjustment_basis=_required_text(
                instrument.return_adjustment_basis, "return_adjustment_basis"
            ),
        )
        for instrument in instruments
    )
    normalized_positions = tuple(
        replace(
            position,
            portfolio_snapshot_id=_required_text(
                position.portfolio_snapshot_id, "portfolio_snapshot_id"
            ),
            portfolio_id=_required_text(position.portfolio_id, "portfolio_id"),
            instrument_id=_required_text(position.instrument_id, "instrument_id"),
            valuation_price_source=_required_text(
                position.valuation_price_source, "valuation_price_source"
            ),
            source_row_id=(
                str(position.source_row_id).strip()
                if position.source_row_id is not None
                and str(position.source_row_id).strip()
                else None
            ),
        )
        for position in positions
    )
    normalized_cash = tuple(
        replace(
            cash,
            portfolio_snapshot_id=_required_text(
                cash.portfolio_snapshot_id, "portfolio_snapshot_id"
            ),
            portfolio_id=_required_text(cash.portfolio_id, "portfolio_id"),
            base_currency=_required_text(
                cash.base_currency, "base_currency", upper=True
            ),
            market_calendar_id=_required_text(
                cash.market_calendar_id, "market_calendar_id", upper=True
            ),
            market_timezone=_required_text(cash.market_timezone, "market_timezone"),
            cash_amount=0.0 if cash.cash_amount == 0 else cash.cash_amount,
            cash_source=_required_text(cash.cash_source, "cash_source"),
        )
        for cash in cash_records
    )
    return normalized_instruments, normalized_positions, normalized_cash


def _normalized_market_universe(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized = tuple(str(value).strip().upper() for value in values)
    if any(not value for value in normalized):
        raise PortfolioIdentifierError(
            "Market-data instrument identifiers must be non-empty."
        )
    if len(set(normalized)) != len(normalized):
        raise PortfolioDuplicateError(
            "The selected market-data universe contains duplicate identifiers."
        )
    return normalized


def _validate_held_instrument(instrument: Instrument, cash: Cash) -> None:
    if instrument.instrument_type not in {
        InstrumentType.EQUITY,
        InstrumentType.STANDARD_ETF,
    }:
        raise InstrumentEligibilityError(
            f"Instrument {instrument.instrument_id!r} has unsupported "
            f"type {instrument.instrument_type!s}."
        )
    if instrument.price_currency != cash.base_currency:
        raise PortfolioCurrencyError(
            f"Instrument {instrument.instrument_id!r} currency "
            f"{instrument.price_currency!r} does not match portfolio "
            f"base currency {cash.base_currency!r}."
        )
    if instrument.market_calendar_id != cash.market_calendar_id:
        raise PortfolioCalendarError(
            f"Instrument {instrument.instrument_id!r} calendar "
            f"{instrument.market_calendar_id!r} does not match "
            f"portfolio calendar {cash.market_calendar_id!r}."
        )
    if instrument.market_timezone != cash.market_timezone:
        raise PortfolioTimezoneError(
            f"Instrument {instrument.instrument_id!r} timezone "
            f"{instrument.market_timezone!r} does not match portfolio "
            f"timezone {cash.market_timezone!r}."
        )
    _validate_timezone(instrument.market_timezone)


def validate_portfolio_snapshot(
    instruments: Sequence[Instrument],
    positions: Sequence[Position],
    cash_records: Sequence[Cash],
    market_data_instruments: Sequence[str] | None = None,
) -> ValidatedPortfolioSnapshot:
    """Validate and reconcile exactly one immutable portfolio snapshot."""
    if not instruments:
        raise PortfolioIdentifierError("The instrument registry is empty.")
    if not positions:
        raise PortfolioIdentifierError("At least one position is required.")
    if len(cash_records) != 1:
        raise PortfolioCashError(
            "Exactly one explicit cash record is required per portfolio snapshot; "
            f"received {len(cash_records)}."
        )
    instruments, positions, cash_records = _normalize_contracts(
        instruments, positions, cash_records
    )

    _unique_by(instruments, "instrument_id", "instrument_id")
    _unique_by(instruments, "provider_ticker", "provider_ticker mapping")
    _unique_by(instruments, "adjusted_return_series_id", "return-series identity")
    _unique_by(positions, "snapshot_instrument", "snapshot/instrument row")
    source_rows = [
        position.source_row_id
        for position in positions
        if position.source_row_id is not None
    ]
    if len(source_rows) != len(set(source_rows)):
        raise PortfolioDuplicateError("Duplicate source_row_id in positions.")

    instrument_by_id = {item.instrument_id: item for item in instruments}
    cash = cash_records[0]
    identities = {
        (
            position.portfolio_snapshot_id,
            position.portfolio_id,
            position.as_of_date,
        )
        for position in positions
    }
    if len(identities) != 1:
        raise PortfolioIdentifierError(
            "Positions must belong to exactly one portfolio_snapshot_id, "
            "portfolio_id, and as_of_date."
        )
    position_identity = next(iter(identities))
    cash_identity = (cash.portfolio_snapshot_id, cash.portfolio_id, cash.as_of_date)
    if position_identity != cash_identity:
        raise PortfolioCashError(
            "The cash portfolio_snapshot_id, portfolio_id, and as_of_date must "
            "match every position."
        )

    calendar = resolve_market_calendar(cash.market_calendar_id)
    if not calendar.is_session(cash.as_of_date):
        raise PortfolioDateError(
            f"as_of_date {cash.as_of_date} is not a valid "
            f"{calendar.calendar_id} market session."
        )
    if cash.market_timezone != calendar.market_timezone:
        raise PortfolioTimezoneError(
            f"{calendar.calendar_id} requires market_timezone "
            f"{calendar.market_timezone!r}; received {cash.market_timezone!r}."
        )
    portfolio_zone = _validate_timezone(cash.market_timezone)
    _require_aware(cash.as_of_timestamp, "as_of_timestamp")
    if (
        cash.as_of_timestamp.utcoffset()
        != cash.as_of_timestamp.astimezone(portfolio_zone).utcoffset()
    ):
        raise PortfolioTimezoneError(
            "as_of_timestamp UTC offset does not match market_timezone."
        )
    if cash.as_of_timestamp.astimezone(portfolio_zone).date() != cash.as_of_date:
        raise PortfolioTimestampError(
            "as_of_timestamp must belong to the declared as_of_date in market_timezone."
        )
    if not math.isfinite(cash.cash_amount):
        raise PortfolioCashError("cash_amount must be finite.")

    held_ids: list[str] = []
    held_tickers: list[str] = []
    held_instruments: list[Instrument] = []
    for position in positions:
        instrument = instrument_by_id.get(position.instrument_id)
        if instrument is None:
            raise PortfolioIdentifierError(
                f"Position instrument_id {position.instrument_id!r} is absent "
                "from the instrument registry."
            )
        # Eligibility applies to what the book holds. A shared registry may
        # list other instruments (another currency, a leveraged ETF) that this
        # snapshot never values; only their identifiers must stay unique.
        _validate_held_instrument(instrument, cash)
        if not math.isfinite(position.quantity) or position.quantity == 0:
            raise PortfolioQuantityError(
                f"Position quantity for {position.instrument_id!r} must be finite "
                "and non-zero."
            )
        if not math.isfinite(position.valuation_price) or position.valuation_price <= 0:
            raise PortfolioPriceError(
                f"valuation_price for {position.instrument_id!r} must be finite "
                "and strictly positive."
            )
        _require_aware(position.valuation_price_timestamp, "valuation_price_timestamp")
        if (
            position.valuation_price_timestamp.utcoffset()
            != position.valuation_price_timestamp.astimezone(portfolio_zone).utcoffset()
        ):
            raise PortfolioTimezoneError(
                f"valuation_price_timestamp for {position.instrument_id!r} has "
                "an offset inconsistent with market_timezone."
            )
        local_price_timestamp = position.valuation_price_timestamp.astimezone(
            portfolio_zone
        )
        if local_price_timestamp.date() != cash.as_of_date:
            raise PortfolioTimestampError(
                f"valuation_price_timestamp for {position.instrument_id!r} must "
                "belong to the snapshot session."
            )
        if position.valuation_price_timestamp > cash.as_of_timestamp:
            raise PortfolioTimestampError(
                f"valuation_price_timestamp for {position.instrument_id!r} is "
                "later than as_of_timestamp."
            )
        held_ids.append(instrument.instrument_id)
        held_tickers.append(instrument.provider_ticker)
        held_instruments.append(instrument)

    universe = _normalized_market_universe(market_data_instruments)
    missing = tuple(
        ticker for ticker in held_tickers if universe and ticker not in universe
    )
    if missing:
        details = ", ".join(missing)
        raise PortfolioIdentifierError(
            "Held instruments are missing from the selected market-data universe: "
            f"{details}."
        )
    extras = tuple(ticker for ticker in universe if ticker not in set(held_tickers))
    reconciliation = PortfolioReconciliation(
        held_instrument_ids=tuple(held_ids),
        held_provider_tickers=tuple(held_tickers),
        missing_market_data_instruments=(),
        extra_market_data_instruments=extras,
    )
    return ValidatedPortfolioSnapshot(
        positions=tuple(positions),
        cash=cash,
        instruments=tuple(held_instruments),
        reconciliation=reconciliation,
        calendar_source=calendar.source,
        calendar_version=calendar.version,
    )


def _canonical_snapshot(snapshot: ValidatedPortfolioSnapshot) -> dict[str, object]:
    cash = snapshot.cash
    portfolio_zone = ZoneInfo(cash.market_timezone)
    instrument_by_id = {
        instrument.instrument_id: instrument for instrument in snapshot.instruments
    }
    positions: list[dict[str, object]] = []
    for position in sorted(snapshot.positions, key=lambda item: item.instrument_id):
        instrument = instrument_by_id[position.instrument_id]
        positions.append(
            {
                "instrument_id": position.instrument_id,
                "provider_ticker": instrument.provider_ticker,
                "adjusted_return_series_id": instrument.adjusted_return_series_id,
                "return_adjustment_basis": instrument.return_adjustment_basis,
                "quantity": position.quantity,
                "valuation_price": position.valuation_price,
                "valuation_price_timestamp": (
                    position.valuation_price_timestamp.astimezone(
                        portfolio_zone
                    ).isoformat()
                ),
                "valuation_price_source": position.valuation_price_source,
            }
        )
    return {
        "exposure_snapshot_schema_version": EXPOSURE_SNAPSHOT_SCHEMA_VERSION,
        "instrument_registry_schema_version": (
            snapshot.instrument_registry_schema_version
        ),
        "positions_schema_version": snapshot.positions_schema_version,
        "cash_schema_version": snapshot.cash_schema_version,
        "portfolio_snapshot_id": cash.portfolio_snapshot_id,
        "portfolio_id": cash.portfolio_id,
        "as_of_date": cash.as_of_date.isoformat(),
        "as_of_timestamp": cash.as_of_timestamp.astimezone(portfolio_zone).isoformat(),
        "base_currency": cash.base_currency,
        "market_calendar_id": cash.market_calendar_id,
        "market_timezone": cash.market_timezone,
        "calendar_version": snapshot.calendar_version,
        "cash_amount": cash.cash_amount,
        "cash_source": cash.cash_source,
        "positions": positions,
    }


def exposure_snapshot_id(snapshot: ValidatedPortfolioSnapshot) -> str:
    """Derive an order-independent identity from normalized material content."""
    payload = json.dumps(
        _canonical_snapshot(snapshot),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "exp_" + hashlib.sha256(payload).hexdigest()


def value_portfolio(snapshot: ValidatedPortfolioSnapshot) -> PortfolioValuation:
    """Calculate values, weights, and exposure ratios for a valid snapshot."""
    currency_valuation = calculate_currency_valuation(snapshot)
    portfolio_value = currency_valuation.portfolio_value
    if portfolio_value <= 0:
        raise NonPositivePortfolioValueError(
            "Weights and exposure ratios require strictly positive portfolio "
            f"value; received {portfolio_value:.15g}."
        )

    valued_positions = tuple(
        ValuedPosition(
            position=item.position,
            instrument=item.instrument,
            position_value=item.position_value,
            weight=item.position_value / portfolio_value,
        )
        for item in currency_valuation.positions
    )
    cash_weight = snapshot.cash.cash_amount / portfolio_value
    return PortfolioValuation(
        exposure_snapshot_id=currency_valuation.exposure_snapshot_id,
        snapshot=snapshot,
        positions=valued_positions,
        long_exposure=currency_valuation.long_exposure,
        short_exposure=currency_valuation.short_exposure,
        net_instrument_exposure=currency_valuation.net_instrument_exposure,
        gross_exposure=currency_valuation.gross_exposure,
        portfolio_value=portfolio_value,
        cash_weight=cash_weight,
        instrument_weight_sum=math.fsum(
            position.weight for position in valued_positions
        ),
        gross_exposure_ratio=currency_valuation.gross_exposure / portfolio_value,
        net_exposure_ratio=(
            currency_valuation.net_instrument_exposure / portfolio_value
        ),
    )


def calculate_currency_valuation(
    snapshot: ValidatedPortfolioSnapshot,
) -> PortfolioCurrencyValuation:
    """Calculate signed currency exposures without requiring positive value."""
    raw_values = tuple(
        position.quantity * position.valuation_price for position in snapshot.positions
    )
    long_exposure = math.fsum(max(value, 0.0) for value in raw_values)
    short_exposure = math.fsum(min(value, 0.0) for value in raw_values)
    net_exposure = math.fsum(raw_values)
    gross_exposure = math.fsum(abs(value) for value in raw_values)
    portfolio_value = snapshot.cash.cash_amount + net_exposure

    instrument_by_id = {
        instrument.instrument_id: instrument for instrument in snapshot.instruments
    }
    valued_positions = tuple(
        CurrencyValuedPosition(
            position=position,
            instrument=instrument_by_id[position.instrument_id],
            position_value=value,
        )
        for position, value in zip(snapshot.positions, raw_values, strict=True)
    )
    return PortfolioCurrencyValuation(
        exposure_snapshot_id=exposure_snapshot_id(snapshot),
        snapshot=snapshot,
        positions=valued_positions,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        net_instrument_exposure=net_exposure,
        gross_exposure=gross_exposure,
        portfolio_value=portfolio_value,
    )


__all__ = [
    "calculate_currency_valuation",
    "exposure_snapshot_id",
    "validate_portfolio_snapshot",
    "value_portfolio",
]

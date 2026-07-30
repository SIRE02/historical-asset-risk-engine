"""Versioned, immutable domain contracts for portfolio valuation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

INSTRUMENT_REGISTRY_SCHEMA_ID = "historical-asset-risk/instrument-registry"
INSTRUMENT_REGISTRY_SCHEMA_VERSION = "1.experimental"
POSITIONS_SCHEMA_ID = "historical-asset-risk/positions"
POSITIONS_SCHEMA_VERSION = "1.experimental"
CASH_SCHEMA_ID = "historical-asset-risk/cash"
CASH_SCHEMA_VERSION = "1.experimental"
VALIDATED_POSITIONS_SCHEMA_ID = "historical-asset-risk/validated-positions"
VALIDATED_POSITIONS_SCHEMA_VERSION = "1.experimental"
PORTFOLIO_VALUATION_SCHEMA_ID = "historical-asset-risk/portfolio-valuation"
PORTFOLIO_VALUATION_SCHEMA_VERSION = "1.experimental"
PORTFOLIO_EXPOSURE_SCHEMA_ID = "historical-asset-risk/portfolio-exposure-summary"
PORTFOLIO_EXPOSURE_SCHEMA_VERSION = "1.experimental"
EXPOSURE_SNAPSHOT_SCHEMA_VERSION = "1.experimental"


class PortfolioError(ValueError):
    """Base class for readable portfolio-domain failures."""


class PortfolioSchemaError(PortfolioError):
    """Raised when an input or artifact does not satisfy its declared schema."""


class PortfolioDateError(PortfolioError):
    """Raised when snapshot dates are malformed or inconsistent."""


class PortfolioDuplicateError(PortfolioError):
    """Raised when a contract contains a prohibited duplicate identity."""


class PortfolioCurrencyError(PortfolioError):
    """Raised when instrument and portfolio currencies do not reconcile."""


class PortfolioIdentifierError(PortfolioError):
    """Raised when canonical and provider identifiers do not reconcile."""


class InstrumentEligibilityError(PortfolioError):
    """Raised when an instrument is outside the initial eligible universe."""


class PortfolioCalendarError(PortfolioError):
    """Raised when a calendar is unsupported or a date is not a session."""


class PortfolioTimezoneError(PortfolioError):
    """Raised when timezone identifiers or timestamp zones do not reconcile."""


class PortfolioTimestampError(PortfolioError):
    """Raised when a snapshot or valuation timestamp violates session policy."""


class PortfolioPriceError(PortfolioError):
    """Raised when a valuation price is missing, non-finite, or non-positive."""


class PortfolioQuantityError(PortfolioError):
    """Raised when a position quantity is missing, non-finite, or zero."""


class PortfolioCashError(PortfolioError):
    """Raised when the explicit cash contract is missing or inconsistent."""


class NonPositivePortfolioValueError(PortfolioError):
    """Raised when weight-based results are requested for non-positive value."""


class ArtifactSchemaError(PortfolioSchemaError):
    """Raised when a persisted artifact has no supported declared schema."""


class InstrumentType(StrEnum):
    """Initially supported linear instrument classifications."""

    EQUITY = "equity"
    STANDARD_ETF = "standard_etf"


@dataclass(frozen=True)
class Instrument:
    """One canonical instrument-registry record."""

    instrument_id: str
    provider_ticker: str
    listing_venue: str
    instrument_type: InstrumentType
    price_currency: str
    market_calendar_id: str
    market_timezone: str
    adjusted_return_series_id: str
    return_adjustment_basis: str


@dataclass(frozen=True)
class Position:
    """One normalized position in an immutable source snapshot."""

    portfolio_snapshot_id: str
    portfolio_id: str
    as_of_date: date
    instrument_id: str
    quantity: float
    valuation_price: float
    valuation_price_timestamp: datetime
    valuation_price_source: str
    source_row_id: str | None = None


@dataclass(frozen=True)
class Cash:
    """The one explicit cash record and portfolio-level snapshot metadata."""

    portfolio_snapshot_id: str
    portfolio_id: str
    as_of_date: date
    as_of_timestamp: datetime
    base_currency: str
    market_calendar_id: str
    market_timezone: str
    cash_amount: float
    cash_source: str


@dataclass(frozen=True)
class PortfolioReconciliation:
    """Identifier-based reconciliation against the selected market universe."""

    held_instrument_ids: tuple[str, ...]
    held_provider_tickers: tuple[str, ...]
    missing_market_data_instruments: tuple[str, ...]
    extra_market_data_instruments: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedPortfolioSnapshot:
    """One completely reconciled portfolio snapshot ready for calculation."""

    positions: tuple[Position, ...]
    cash: Cash
    instruments: tuple[Instrument, ...]
    reconciliation: PortfolioReconciliation
    instrument_registry_schema_version: str = INSTRUMENT_REGISTRY_SCHEMA_VERSION
    positions_schema_version: str = POSITIONS_SCHEMA_VERSION
    cash_schema_version: str = CASH_SCHEMA_VERSION
    calendar_source: str = ""
    calendar_version: str = ""


@dataclass(frozen=True)
class ValuedPosition:
    """A validated position enriched with registry lineage and valuation."""

    position: Position
    instrument: Instrument
    position_value: float
    weight: float


@dataclass(frozen=True)
class CurrencyValuedPosition:
    """A validated position with its signed base-currency value."""

    position: Position
    instrument: Instrument
    position_value: float


@dataclass(frozen=True)
class PortfolioCurrencyValuation:
    """Currency values available for any finite portfolio value."""

    exposure_snapshot_id: str
    snapshot: ValidatedPortfolioSnapshot
    positions: tuple[CurrencyValuedPosition, ...]
    long_exposure: float
    short_exposure: float
    net_instrument_exposure: float
    gross_exposure: float
    portfolio_value: float


@dataclass(frozen=True)
class PortfolioValuation:
    """Reconciled currency values and positive-value exposure measures."""

    exposure_snapshot_id: str
    snapshot: ValidatedPortfolioSnapshot
    positions: tuple[ValuedPosition, ...]
    long_exposure: float
    short_exposure: float
    net_instrument_exposure: float
    gross_exposure: float
    portfolio_value: float
    cash_weight: float
    instrument_weight_sum: float
    gross_exposure_ratio: float
    net_exposure_ratio: float


__all__ = [
    "ArtifactSchemaError",
    "CASH_SCHEMA_ID",
    "CASH_SCHEMA_VERSION",
    "Cash",
    "CurrencyValuedPosition",
    "EXPOSURE_SNAPSHOT_SCHEMA_VERSION",
    "INSTRUMENT_REGISTRY_SCHEMA_ID",
    "INSTRUMENT_REGISTRY_SCHEMA_VERSION",
    "Instrument",
    "InstrumentEligibilityError",
    "InstrumentType",
    "NonPositivePortfolioValueError",
    "PortfolioCalendarError",
    "PortfolioCashError",
    "PortfolioCurrencyError",
    "PortfolioCurrencyValuation",
    "PortfolioDateError",
    "PortfolioDuplicateError",
    "PortfolioError",
    "PortfolioIdentifierError",
    "PortfolioPriceError",
    "PortfolioQuantityError",
    "PortfolioReconciliation",
    "PortfolioSchemaError",
    "PortfolioTimestampError",
    "PortfolioTimezoneError",
    "PortfolioValuation",
    "PORTFOLIO_EXPOSURE_SCHEMA_ID",
    "PORTFOLIO_EXPOSURE_SCHEMA_VERSION",
    "PORTFOLIO_VALUATION_SCHEMA_ID",
    "PORTFOLIO_VALUATION_SCHEMA_VERSION",
    "POSITIONS_SCHEMA_ID",
    "POSITIONS_SCHEMA_VERSION",
    "Position",
    "VALIDATED_POSITIONS_SCHEMA_ID",
    "VALIDATED_POSITIONS_SCHEMA_VERSION",
    "ValidatedPortfolioSnapshot",
    "ValuedPosition",
]

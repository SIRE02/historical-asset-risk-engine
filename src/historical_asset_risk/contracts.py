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

# --- Portfolio analytics: P&L, aggregation, and realizations -------------------
# Simple-return artifacts are deliberately distinct in schema id and units from
# the descriptive log-return covariance/correlation artifacts.
PORTFOLIO_ANALYTICS_SCHEMA_VERSION = "1.experimental"
PORTFOLIO_ALIGNED_SIMPLE_RETURNS_SCHEMA_ID = (
    "historical-asset-risk/portfolio-aligned-simple-returns"
)
HYPOTHETICAL_PNL_SCHEMA_ID = "historical-asset-risk/hypothetical-portfolio-pnl"
PROXY_REALIZED_PNL_SCHEMA_ID = "historical-asset-risk/proxy-realized-portfolio-pnl"
RISK_REALIZATIONS_SCHEMA_ID = "historical-asset-risk/risk-realizations"
# Independent of the artifact schema version: identifies the realization
# calculation so a downstream engine can pin the identity it joins forecasts to.
RISK_REALIZATION_CALCULATION_VERSION = "historical-asset-risk/proxy-realization@1"
SIMPLE_RETURN_COVARIANCE_SCHEMA_ID = (
    "historical-asset-risk/portfolio-simple-return-covariance"
)
SIMPLE_RETURN_CORRELATION_SCHEMA_ID = (
    "historical-asset-risk/portfolio-simple-return-correlation"
)
SIMPLE_RETURN_SUMMARY_SCHEMA_ID = "historical-asset-risk/simple-return-summary"
PORTFOLIO_RISK_SUMMARY_SCHEMA_ID = "historical-asset-risk/portfolio-risk-summary"
PORTFOLIO_RISK_CONTRIBUTIONS_SCHEMA_ID = (
    "historical-asset-risk/portfolio-risk-contributions"
)
PORTFOLIO_CONCENTRATION_SUMMARY_SCHEMA_ID = (
    "historical-asset-risk/portfolio-concentration-summary"
)

# Frozen column contracts for the portfolio-loss and realization artifacts that
# the tail-risk and stress layer builds on. The schema version stays
# ``experimental`` pending promotion, but the column set, order, units, and
# loss sign of these three files must not change.
HYPOTHETICAL_PNL_COLUMNS: tuple[str, ...] = (
    "portfolio_snapshot_id",
    "exposure_snapshot_id",
    "portfolio_id",
    "as_of_date",
    "as_of_timestamp",
    "base_currency",
    "market_calendar_id",
    "data_snapshot_id",
    "period_start",
    "period_end",
    "return_type",
    "hypothetical_pnl",
    "hypothetical_loss",
    "units",
    "loss_sign",
    "schema_version",
)
PROXY_REALIZED_PNL_COLUMNS: tuple[str, ...] = (
    "portfolio_snapshot_id",
    "portfolio_id",
    "exposure_snapshot_id",
    "as_of_date",
    "target_period_start",
    "target_period_end",
    "market_calendar_id",
    "held_instrument_count",
    "missing_instruments",
    "proxy_realized_pnl",
    "proxy_realized_loss",
    "outcome_status",
    "data_snapshot_id",
    "return_type",
    "units",
    "loss_sign",
    "calculation_version",
    "schema_version",
)
# --- Tail analytics: tail risk and stress -------------------------------------
TAIL_ANALYTICS_SCHEMA_VERSION = "1.experimental"
PORTFOLIO_VALUE_AT_RISK_SCHEMA_ID = "historical-asset-risk/portfolio-value-at-risk"
PORTFOLIO_EXPECTED_SHORTFALL_SCHEMA_ID = (
    "historical-asset-risk/portfolio-expected-shortfall"
)
EXPECTED_SHORTFALL_TAIL_WEIGHTS_SCHEMA_ID = (
    "historical-asset-risk/expected-shortfall-tail-weights"
)
TAIL_RISK_COMPARISON_SCHEMA_ID = "historical-asset-risk/tail-risk-comparison"
TRAILING_TAIL_RISK_SCHEMA_ID = "historical-asset-risk/trailing-portfolio-tail-risk"
STRESS_TEST_RESULTS_SCHEMA_ID = "historical-asset-risk/stress-test-results"
STRESS_CONTRIBUTIONS_SCHEMA_ID = "historical-asset-risk/stress-contributions"

RISK_REALIZATIONS_COLUMNS: tuple[str, ...] = (
    "portfolio_id",
    "exposure_snapshot_id",
    "market_calendar_id",
    "target_period_start",
    "target_period_end",
    "units",
    "loss_sign",
    "realized_loss",
    "outcome_status",
    "data_snapshot_id",
    "calculation_version",
    "schema_version",
)

# Frozen column contracts for the tail-risk and stress artifacts. The schema
# version stays experimental pending promotion, but these column sets, orders,
# and loss sign must not drift.
_TAIL_IDENTITY = (
    "portfolio_snapshot_id",
    "exposure_snapshot_id",
    "portfolio_id",
    "as_of_date",
    "base_currency",
    "data_snapshot_id",
)
PORTFOLIO_VALUE_AT_RISK_COLUMNS: tuple[str, ...] = (
    *_TAIL_IDENTITY,
    "dimension",
    "units",
    "confidence_level",
    "observation_count",
    "value_at_risk",
    "order_statistic_rank",
    "quantile_method",
    "tail_sample_warning",
    "schema_version",
)
PORTFOLIO_EXPECTED_SHORTFALL_COLUMNS: tuple[str, ...] = (
    *_TAIL_IDENTITY,
    "dimension",
    "units",
    "confidence_level",
    "observation_count",
    "expected_shortfall",
    "value_at_risk",
    "nominal_tail_observations",
    "full_tail_count",
    "boundary_weight",
    "contributing_observation_count",
    "es_ge_var",
    "tail_sample_warning",
    "schema_version",
)
EXPECTED_SHORTFALL_TAIL_WEIGHTS_COLUMNS: tuple[str, ...] = (
    "exposure_snapshot_id",
    "dimension",
    "units",
    "confidence_level",
    "order_statistic_rank",
    "loss",
    "weight",
    "schema_version",
)
TAIL_RISK_COMPARISON_COLUMNS: tuple[str, ...] = (
    *_TAIL_IDENTITY,
    "dimension",
    "units",
    "confidence_level",
    "observation_count",
    "historical_var",
    "historical_es",
    "normal_var_mean_included",
    "normal_es_mean_included",
    "normal_var_zero_mean",
    "normal_es_zero_mean",
    "mean_loss",
    "standard_deviation",
    "z_alpha",
    "schema_version",
)
TRAILING_TAIL_RISK_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "exposure_snapshot_id",
    "window",
    "window_observation_count",
    "confidence_level",
    "value_at_risk",
    "expected_shortfall",
    "es_ge_var",
    "outcome_status",
    "units",
    "schema_version",
)
STRESS_TEST_RESULTS_COLUMNS: tuple[str, ...] = (
    *_TAIL_IDENTITY,
    "scenario_id",
    "scenario_version",
    "name",
    "kind",
    "content_hash",
    "scenario_pnl",
    "scenario_loss",
    "units",
    "loss_sign",
    "schema_version",
)
STRESS_CONTRIBUTIONS_COLUMNS: tuple[str, ...] = (
    "exposure_snapshot_id",
    "scenario_id",
    "scenario_version",
    "instrument_id",
    "currency_exposure",
    "simple_return_shock",
    "scenario_pnl",
    "scenario_loss",
    "schema_version",
)


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


class PortfolioReturnAlignmentError(PortfolioError):
    """Raised when held instruments cannot be aligned to a simple-return matrix."""


class PortfolioCovarianceError(PortfolioError):
    """Raised when a simple-return covariance sample is too small or degenerate."""


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
    "HYPOTHETICAL_PNL_COLUMNS",
    "HYPOTHETICAL_PNL_SCHEMA_ID",
    "PORTFOLIO_ANALYTICS_SCHEMA_VERSION",
    "EXPECTED_SHORTFALL_TAIL_WEIGHTS_COLUMNS",
    "EXPECTED_SHORTFALL_TAIL_WEIGHTS_SCHEMA_ID",
    "TAIL_ANALYTICS_SCHEMA_VERSION",
    "PORTFOLIO_EXPECTED_SHORTFALL_COLUMNS",
    "PORTFOLIO_EXPECTED_SHORTFALL_SCHEMA_ID",
    "PORTFOLIO_VALUE_AT_RISK_COLUMNS",
    "PORTFOLIO_VALUE_AT_RISK_SCHEMA_ID",
    "PROXY_REALIZED_PNL_COLUMNS",
    "RISK_REALIZATIONS_COLUMNS",
    "STRESS_CONTRIBUTIONS_COLUMNS",
    "STRESS_CONTRIBUTIONS_SCHEMA_ID",
    "STRESS_TEST_RESULTS_COLUMNS",
    "STRESS_TEST_RESULTS_SCHEMA_ID",
    "TAIL_RISK_COMPARISON_COLUMNS",
    "TAIL_RISK_COMPARISON_SCHEMA_ID",
    "TRAILING_TAIL_RISK_COLUMNS",
    "TRAILING_TAIL_RISK_SCHEMA_ID",
    "PORTFOLIO_ALIGNED_SIMPLE_RETURNS_SCHEMA_ID",
    "PORTFOLIO_CONCENTRATION_SUMMARY_SCHEMA_ID",
    "PORTFOLIO_RISK_CONTRIBUTIONS_SCHEMA_ID",
    "PORTFOLIO_RISK_SUMMARY_SCHEMA_ID",
    "PROXY_REALIZED_PNL_SCHEMA_ID",
    "RISK_REALIZATION_CALCULATION_VERSION",
    "PortfolioCovarianceError",
    "PortfolioReturnAlignmentError",
    "RISK_REALIZATIONS_SCHEMA_ID",
    "SIMPLE_RETURN_CORRELATION_SCHEMA_ID",
    "SIMPLE_RETURN_COVARIANCE_SCHEMA_ID",
    "SIMPLE_RETURN_SUMMARY_SCHEMA_ID",
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

"""Strict book CSV parsing and version-aware artifact loading."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from historical_asset_risk.contracts import (
    EXPECTED_SHORTFALL_TAIL_WEIGHTS_COLUMNS,
    EXPECTED_SHORTFALL_TAIL_WEIGHTS_SCHEMA_ID,
    HYPOTHETICAL_PNL_COLUMNS,
    HYPOTHETICAL_PNL_SCHEMA_ID,
    INSTRUMENT_REGISTRY_SCHEMA_ID,
    INSTRUMENT_REGISTRY_SCHEMA_VERSION,
    PORTFOLIO_ALIGNED_SIMPLE_RETURNS_SCHEMA_ID,
    PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    PORTFOLIO_CONCENTRATION_SUMMARY_SCHEMA_ID,
    PORTFOLIO_EXPECTED_SHORTFALL_COLUMNS,
    PORTFOLIO_EXPECTED_SHORTFALL_SCHEMA_ID,
    PORTFOLIO_EXPOSURE_SCHEMA_ID,
    PORTFOLIO_EXPOSURE_SCHEMA_VERSION,
    PORTFOLIO_RISK_CONTRIBUTIONS_SCHEMA_ID,
    PORTFOLIO_RISK_SUMMARY_SCHEMA_ID,
    PORTFOLIO_VALUATION_SCHEMA_ID,
    PORTFOLIO_VALUATION_SCHEMA_VERSION,
    PORTFOLIO_VALUE_AT_RISK_COLUMNS,
    PORTFOLIO_VALUE_AT_RISK_SCHEMA_ID,
    PROXY_REALIZED_PNL_COLUMNS,
    PROXY_REALIZED_PNL_SCHEMA_ID,
    RISK_REALIZATIONS_COLUMNS,
    RISK_REALIZATIONS_SCHEMA_ID,
    SIMPLE_RETURN_CORRELATION_SCHEMA_ID,
    SIMPLE_RETURN_COVARIANCE_SCHEMA_ID,
    SIMPLE_RETURN_SUMMARY_SCHEMA_ID,
    STRESS_CATALOG_SCHEMA_ID,
    STRESS_CONTRIBUTIONS_COLUMNS,
    STRESS_CONTRIBUTIONS_SCHEMA_ID,
    STRESS_SCENARIO_SCHEMA_VERSION,
    STRESS_TEST_RESULTS_COLUMNS,
    STRESS_TEST_RESULTS_SCHEMA_ID,
    TAIL_ANALYTICS_SCHEMA_VERSION,
    TAIL_RISK_COMPARISON_COLUMNS,
    TAIL_RISK_COMPARISON_SCHEMA_ID,
    TRAILING_TAIL_RISK_COLUMNS,
    TRAILING_TAIL_RISK_SCHEMA_ID,
    VALIDATED_POSITIONS_SCHEMA_ID,
    VALIDATED_POSITIONS_SCHEMA_VERSION,
    ArtifactSchemaError,
    Cash,
    Instrument,
    InstrumentType,
    PortfolioCashError,
    PortfolioDateError,
    PortfolioIdentifierError,
    PortfolioPriceError,
    PortfolioQuantityError,
    PortfolioSchemaError,
    PortfolioTimestampError,
    PortfolioValuation,
    Position,
)

ADJUSTED_PRICES_SCHEMA_ID = "historical-asset-risk/adjusted-prices"
SIMPLE_RETURNS_SCHEMA_ID = "historical-asset-risk/simple-returns"
LOG_RETURNS_SCHEMA_ID = "historical-asset-risk/log-returns"
TABULAR_ARTIFACT_SCHEMA_VERSION = "1.experimental"

INSTRUMENT_COLUMNS = (
    "instrument_id",
    "provider_ticker",
    "listing_venue",
    "instrument_type",
    "price_currency",
    "market_calendar_id",
    "market_timezone",
    "adjusted_return_series_id",
    "return_adjustment_basis",
)
POSITION_COLUMNS = (
    "portfolio_snapshot_id",
    "portfolio_id",
    "as_of_date",
    "instrument_id",
    "quantity",
    "valuation_price",
    "valuation_price_timestamp",
    "valuation_price_source",
)
POSITION_OPTIONAL_COLUMNS = ("source_row_id",)
CASH_COLUMNS = (
    "portfolio_snapshot_id",
    "portfolio_id",
    "as_of_date",
    "as_of_timestamp",
    "base_currency",
    "market_calendar_id",
    "market_timezone",
    "cash_amount",
    "cash_source",
)

ARTIFACT_SCHEMAS: dict[str, tuple[str, str]] = {
    "acquired_adjusted_prices.csv": (
        "historical-asset-risk/acquired-adjusted-price-records",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "adjusted_prices.csv": (
        ADJUSTED_PRICES_SCHEMA_ID,
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "simple_returns.csv": (
        SIMPLE_RETURNS_SCHEMA_ID,
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "log_returns.csv": (LOG_RETURNS_SCHEMA_ID, TABULAR_ARTIFACT_SCHEMA_VERSION),
    "return_summary.csv": (
        "historical-asset-risk/return-summary",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "volatility_summary.csv": (
        "historical-asset-risk/volatility-summary",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "rolling_volatility.csv": (
        "historical-asset-risk/rolling-volatility",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "covariance_matrix.csv": (
        "historical-asset-risk/covariance-matrix",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "correlation_matrix.csv": (
        "historical-asset-risk/correlation-matrix",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "rolling_covariance.csv": (
        "historical-asset-risk/rolling-covariance",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "rolling_correlation.csv": (
        "historical-asset-risk/rolling-correlation",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "validated_instruments.csv": (
        INSTRUMENT_REGISTRY_SCHEMA_ID,
        INSTRUMENT_REGISTRY_SCHEMA_VERSION,
    ),
    "validated_positions.csv": (
        VALIDATED_POSITIONS_SCHEMA_ID,
        VALIDATED_POSITIONS_SCHEMA_VERSION,
    ),
    "portfolio_valuation.csv": (
        PORTFOLIO_VALUATION_SCHEMA_ID,
        PORTFOLIO_VALUATION_SCHEMA_VERSION,
    ),
    "portfolio_exposure_summary.csv": (
        PORTFOLIO_EXPOSURE_SCHEMA_ID,
        PORTFOLIO_EXPOSURE_SCHEMA_VERSION,
    ),
    "data_quality_report.json": (
        "historical-asset-risk/data-quality-report",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "run_manifest.json": (
        "historical-asset-risk/run-manifest",
        TABULAR_ARTIFACT_SCHEMA_VERSION,
    ),
    "portfolio_aligned_simple_returns.csv": (
        PORTFOLIO_ALIGNED_SIMPLE_RETURNS_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "hypothetical_portfolio_pnl.csv": (
        HYPOTHETICAL_PNL_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "proxy_realized_portfolio_pnl.csv": (
        PROXY_REALIZED_PNL_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "risk_realizations.csv": (
        RISK_REALIZATIONS_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "portfolio_simple_return_covariance.csv": (
        SIMPLE_RETURN_COVARIANCE_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "portfolio_simple_return_correlation.csv": (
        SIMPLE_RETURN_CORRELATION_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "simple_return_summary.csv": (
        SIMPLE_RETURN_SUMMARY_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "portfolio_risk_summary.csv": (
        PORTFOLIO_RISK_SUMMARY_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "portfolio_risk_contributions.csv": (
        PORTFOLIO_RISK_CONTRIBUTIONS_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "portfolio_concentration_summary.csv": (
        PORTFOLIO_CONCENTRATION_SUMMARY_SCHEMA_ID,
        PORTFOLIO_ANALYTICS_SCHEMA_VERSION,
    ),
    "portfolio_value_at_risk.csv": (
        PORTFOLIO_VALUE_AT_RISK_SCHEMA_ID,
        TAIL_ANALYTICS_SCHEMA_VERSION,
    ),
    "portfolio_expected_shortfall.csv": (
        PORTFOLIO_EXPECTED_SHORTFALL_SCHEMA_ID,
        TAIL_ANALYTICS_SCHEMA_VERSION,
    ),
    "portfolio_expected_shortfall_tail_weights.csv": (
        EXPECTED_SHORTFALL_TAIL_WEIGHTS_SCHEMA_ID,
        TAIL_ANALYTICS_SCHEMA_VERSION,
    ),
    "tail_risk_comparison.csv": (
        TAIL_RISK_COMPARISON_SCHEMA_ID,
        TAIL_ANALYTICS_SCHEMA_VERSION,
    ),
    "trailing_portfolio_tail_risk.csv": (
        TRAILING_TAIL_RISK_SCHEMA_ID,
        TAIL_ANALYTICS_SCHEMA_VERSION,
    ),
    "stress_test_results.csv": (
        STRESS_TEST_RESULTS_SCHEMA_ID,
        TAIL_ANALYTICS_SCHEMA_VERSION,
    ),
    "stress_contributions.csv": (
        STRESS_CONTRIBUTIONS_SCHEMA_ID,
        TAIL_ANALYTICS_SCHEMA_VERSION,
    ),
    "stress_scenario_catalog.json": (
        STRESS_CATALOG_SCHEMA_ID,
        STRESS_SCENARIO_SCHEMA_VERSION,
    ),
}
ARTIFACT_UNITS: dict[str, str] = {
    "acquired_adjusted_prices.csv": "provider_adjusted_price",
    "adjusted_prices.csv": "provider_adjusted_price",
    "simple_returns.csv": "decimal_return_per_observation",
    "log_returns.csv": "log_return_per_observation",
    "return_summary.csv": "mixed_units_by_named_column",
    "volatility_summary.csv": "decimal_log_return_volatility",
    "rolling_volatility.csv": "annualized_decimal_log_return_volatility",
    "covariance_matrix.csv": "daily_log_return_squared",
    "correlation_matrix.csv": "dimensionless",
    "rolling_covariance.csv": "daily_log_return_squared",
    "rolling_correlation.csv": "dimensionless",
    "validated_instruments.csv": "contract_metadata",
    "validated_positions.csv": "base_currency_value_and_decimal_weight",
    "portfolio_valuation.csv": "base_currency",
    "portfolio_exposure_summary.csv": "decimal_ratio",
    "data_quality_report.json": "structured_counts_and_lineage",
    "run_manifest.json": "structured_lineage",
    "portfolio_aligned_simple_returns.csv": "decimal_simple_return_per_interval",
    "hypothetical_portfolio_pnl.csv": "base_currency_pnl_loss_is_positive",
    "proxy_realized_portfolio_pnl.csv": "base_currency_pnl_loss_is_positive",
    "risk_realizations.csv": "base_currency_loss_identity",
    "portfolio_simple_return_covariance.csv": "daily_simple_return_squared",
    "portfolio_simple_return_correlation.csv": "dimensionless",
    "simple_return_summary.csv": "mixed_units_by_named_column",
    "portfolio_risk_summary.csv": "mixed_units_by_named_column",
    "portfolio_risk_contributions.csv": "mixed_units_by_named_column",
    "portfolio_concentration_summary.csv": "mixed_units_by_named_column",
    "portfolio_value_at_risk.csv": "loss_units_by_named_dimension",
    "portfolio_expected_shortfall.csv": "loss_units_by_named_dimension",
    "portfolio_expected_shortfall_tail_weights.csv": "loss_and_dimensionless_weight",
    "tail_risk_comparison.csv": "loss_units_by_named_dimension",
    "trailing_portfolio_tail_risk.csv": "base_currency_loss",
    "stress_test_results.csv": "base_currency_pnl_loss_is_positive",
    "stress_contributions.csv": "base_currency_pnl_loss_is_positive",
    "stress_scenario_catalog.json": "decimal_simple_return_shocks",
}

_VALIDATED_INSTRUMENT_OUTPUT_COLUMNS = INSTRUMENT_COLUMNS + ("schema_version",)
_VALIDATED_POSITION_OUTPUT_COLUMNS = (
    "portfolio_snapshot_id",
    "exposure_snapshot_id",
    "portfolio_id",
    "as_of_date",
    "instrument_id",
    "provider_ticker",
    "quantity",
    "valuation_price",
    "valuation_price_timestamp",
    "valuation_price_source",
    "price_currency",
    "position_value",
    "weight",
    "source_row_id",
    "schema_version",
)
_PORTFOLIO_VALUATION_OUTPUT_COLUMNS = (
    "portfolio_snapshot_id",
    "exposure_snapshot_id",
    "portfolio_id",
    "as_of_date",
    "as_of_timestamp",
    "base_currency",
    "market_calendar_id",
    "market_timezone",
    "calendar_source",
    "calendar_version",
    "cash_amount",
    "long_exposure",
    "short_exposure",
    "net_instrument_exposure",
    "gross_exposure",
    "portfolio_value",
    "schema_version",
)
_PORTFOLIO_EXPOSURE_OUTPUT_COLUMNS = (
    "portfolio_snapshot_id",
    "exposure_snapshot_id",
    "portfolio_id",
    "as_of_date",
    "base_currency",
    "cash_weight",
    "instrument_weight_sum",
    "gross_exposure_ratio",
    "net_exposure_ratio",
    "schema_version",
)
_STRICT_OUTPUT_COLUMNS = {
    "validated_instruments.csv": _VALIDATED_INSTRUMENT_OUTPUT_COLUMNS,
    "validated_positions.csv": _VALIDATED_POSITION_OUTPUT_COLUMNS,
    "portfolio_valuation.csv": _PORTFOLIO_VALUATION_OUTPUT_COLUMNS,
    "portfolio_exposure_summary.csv": _PORTFOLIO_EXPOSURE_OUTPUT_COLUMNS,
    # Frozen portfolio-loss and realization contract (the tail-risk layer
    # depends on these; the column set and order must not drift).
    "hypothetical_portfolio_pnl.csv": HYPOTHETICAL_PNL_COLUMNS,
    "proxy_realized_portfolio_pnl.csv": PROXY_REALIZED_PNL_COLUMNS,
    "risk_realizations.csv": RISK_REALIZATIONS_COLUMNS,
    # Frozen tail-risk and stress contract.
    "portfolio_value_at_risk.csv": PORTFOLIO_VALUE_AT_RISK_COLUMNS,
    "portfolio_expected_shortfall.csv": PORTFOLIO_EXPECTED_SHORTFALL_COLUMNS,
    "portfolio_expected_shortfall_tail_weights.csv": (
        EXPECTED_SHORTFALL_TAIL_WEIGHTS_COLUMNS
    ),
    "tail_risk_comparison.csv": TAIL_RISK_COMPARISON_COLUMNS,
    "trailing_portfolio_tail_risk.csv": TRAILING_TAIL_RISK_COLUMNS,
    "stress_test_results.csv": STRESS_TEST_RESULTS_COLUMNS,
    "stress_contributions.csv": STRESS_CONTRIBUTIONS_COLUMNS,
}


def _read_rows(
    path: Path,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise PortfolioSchemaError(f"{path} has no CSV header.")
            duplicates = sorted(
                {name for name in fieldnames if fieldnames.count(name) > 1}
            )
            if duplicates:
                raise PortfolioSchemaError(
                    f"{path} contains duplicate column(s): {', '.join(duplicates)}."
                )
            missing = [name for name in required if name not in fieldnames]
            unknown = [
                name for name in fieldnames if name not in set(required) | set(optional)
            ]
            if missing:
                raise PortfolioSchemaError(
                    f"{path} is missing required column(s): {', '.join(missing)}."
                )
            if unknown:
                raise PortfolioSchemaError(
                    f"{path} contains unknown column(s): {', '.join(unknown)}."
                )
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise PortfolioSchemaError(f"Could not read {path}: {exc}") from exc
    if not rows:
        raise PortfolioSchemaError(f"{path} contains no data rows.")
    return rows


def _text(
    row: Mapping[str, str], name: str, row_number: int, *, upper: bool = False
) -> str:
    value = str(row.get(name, "")).strip()
    if not value:
        raise PortfolioIdentifierError(f"Row {row_number}: {name} must be non-empty.")
    return value.upper() if upper else value


def _date(row: Mapping[str, str], name: str, row_number: int) -> date:
    raw = str(row.get(name, "")).strip()
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise PortfolioDateError(
            f"Row {row_number}: {name} must use YYYY-MM-DD format."
        ) from exc
    return parsed


def _timestamp(row: Mapping[str, str], name: str, row_number: int) -> datetime:
    raw = str(row.get(name, "")).strip()
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise PortfolioTimestampError(
            f"Row {row_number}: {name} must be an ISO 8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PortfolioTimestampError(
            f"Row {row_number}: {name} must be timezone-aware."
        )
    return parsed


def _number(
    row: Mapping[str, str],
    name: str,
    row_number: int,
    error_type: type[PortfolioPriceError]
    | type[PortfolioQuantityError]
    | type[PortfolioCashError],
) -> float:
    raw = str(row.get(name, "")).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise error_type(f"Row {row_number}: {name} must be numeric.") from exc
    if not math.isfinite(value):
        raise error_type(f"Row {row_number}: {name} must be finite.")
    return value


def read_instrument_registry(path: Path) -> tuple[Instrument, ...]:
    """Parse and normalize the initial strict instrument-registry CSV."""
    rows = _read_rows(path, INSTRUMENT_COLUMNS)
    result: list[Instrument] = []
    for row_number, row in enumerate(rows, start=2):
        raw_type = _text(row, "instrument_type", row_number).lower()
        try:
            instrument_type = InstrumentType(raw_type)
        except ValueError as exc:
            raise PortfolioSchemaError(
                f"Row {row_number}: instrument_type must be 'equity' or 'standard_etf'."
            ) from exc
        result.append(
            Instrument(
                instrument_id=_text(row, "instrument_id", row_number),
                provider_ticker=_text(row, "provider_ticker", row_number, upper=True),
                listing_venue=_text(row, "listing_venue", row_number, upper=True),
                instrument_type=instrument_type,
                price_currency=_text(row, "price_currency", row_number, upper=True),
                market_calendar_id=_text(
                    row, "market_calendar_id", row_number, upper=True
                ),
                market_timezone=_text(row, "market_timezone", row_number),
                adjusted_return_series_id=_text(
                    row, "adjusted_return_series_id", row_number
                ),
                return_adjustment_basis=_text(
                    row, "return_adjustment_basis", row_number
                ),
            )
        )
    return tuple(result)


def read_positions(path: Path) -> tuple[Position, ...]:
    """Parse and normalize one strict positions CSV."""
    rows = _read_rows(path, POSITION_COLUMNS, POSITION_OPTIONAL_COLUMNS)
    result: list[Position] = []
    for row_number, row in enumerate(rows, start=2):
        source_row_id = str(row.get("source_row_id", "")).strip() or None
        result.append(
            Position(
                portfolio_snapshot_id=_text(row, "portfolio_snapshot_id", row_number),
                portfolio_id=_text(row, "portfolio_id", row_number),
                as_of_date=_date(row, "as_of_date", row_number),
                instrument_id=_text(row, "instrument_id", row_number),
                quantity=_number(row, "quantity", row_number, PortfolioQuantityError),
                valuation_price=_number(
                    row, "valuation_price", row_number, PortfolioPriceError
                ),
                valuation_price_timestamp=_timestamp(
                    row, "valuation_price_timestamp", row_number
                ),
                valuation_price_source=_text(row, "valuation_price_source", row_number),
                source_row_id=source_row_id,
            )
        )
    return tuple(result)


def read_cash(path: Path) -> tuple[Cash, ...]:
    """Parse and normalize the explicit cash CSV."""
    rows = _read_rows(path, CASH_COLUMNS)
    result: list[Cash] = []
    for row_number, row in enumerate(rows, start=2):
        result.append(
            Cash(
                portfolio_snapshot_id=_text(row, "portfolio_snapshot_id", row_number),
                portfolio_id=_text(row, "portfolio_id", row_number),
                as_of_date=_date(row, "as_of_date", row_number),
                as_of_timestamp=_timestamp(row, "as_of_timestamp", row_number),
                base_currency=_text(row, "base_currency", row_number, upper=True),
                market_calendar_id=_text(
                    row, "market_calendar_id", row_number, upper=True
                ),
                market_timezone=_text(row, "market_timezone", row_number),
                cash_amount=_number(row, "cash_amount", row_number, PortfolioCashError),
                cash_source=_text(row, "cash_source", row_number),
            )
        )
    return tuple(result)


def portfolio_artifact_frames(
    valuation: PortfolioValuation,
) -> dict[str, pd.DataFrame]:
    """Create the four required deterministic book CSV tables."""
    snapshot = valuation.snapshot
    cash = snapshot.cash
    instrument_rows = [
        {
            "instrument_id": instrument.instrument_id,
            "provider_ticker": instrument.provider_ticker,
            "listing_venue": instrument.listing_venue,
            "instrument_type": instrument.instrument_type.value,
            "price_currency": instrument.price_currency,
            "market_calendar_id": instrument.market_calendar_id,
            "market_timezone": instrument.market_timezone,
            "adjusted_return_series_id": instrument.adjusted_return_series_id,
            "return_adjustment_basis": instrument.return_adjustment_basis,
            "schema_version": snapshot.instrument_registry_schema_version,
        }
        for instrument in snapshot.instruments
    ]
    position_rows = [
        {
            "portfolio_snapshot_id": item.position.portfolio_snapshot_id,
            "exposure_snapshot_id": valuation.exposure_snapshot_id,
            "portfolio_id": item.position.portfolio_id,
            "as_of_date": item.position.as_of_date.isoformat(),
            "instrument_id": item.position.instrument_id,
            "provider_ticker": item.instrument.provider_ticker,
            "quantity": item.position.quantity,
            "valuation_price": item.position.valuation_price,
            "valuation_price_timestamp": (
                item.position.valuation_price_timestamp.isoformat()
            ),
            "valuation_price_source": item.position.valuation_price_source,
            "price_currency": item.instrument.price_currency,
            "position_value": item.position_value,
            "weight": item.weight,
            "source_row_id": item.position.source_row_id,
            "schema_version": VALIDATED_POSITIONS_SCHEMA_VERSION,
        }
        for item in valuation.positions
    ]
    valuation_row = {
        "portfolio_snapshot_id": cash.portfolio_snapshot_id,
        "exposure_snapshot_id": valuation.exposure_snapshot_id,
        "portfolio_id": cash.portfolio_id,
        "as_of_date": cash.as_of_date.isoformat(),
        "as_of_timestamp": cash.as_of_timestamp.isoformat(),
        "base_currency": cash.base_currency,
        "market_calendar_id": cash.market_calendar_id,
        "market_timezone": cash.market_timezone,
        "calendar_source": snapshot.calendar_source,
        "calendar_version": snapshot.calendar_version,
        "cash_amount": cash.cash_amount,
        "long_exposure": valuation.long_exposure,
        "short_exposure": valuation.short_exposure,
        "net_instrument_exposure": valuation.net_instrument_exposure,
        "gross_exposure": valuation.gross_exposure,
        "portfolio_value": valuation.portfolio_value,
        "schema_version": PORTFOLIO_VALUATION_SCHEMA_VERSION,
    }
    exposure_row = {
        "portfolio_snapshot_id": cash.portfolio_snapshot_id,
        "exposure_snapshot_id": valuation.exposure_snapshot_id,
        "portfolio_id": cash.portfolio_id,
        "as_of_date": cash.as_of_date.isoformat(),
        "base_currency": cash.base_currency,
        "cash_weight": valuation.cash_weight,
        "instrument_weight_sum": valuation.instrument_weight_sum,
        "gross_exposure_ratio": valuation.gross_exposure_ratio,
        "net_exposure_ratio": valuation.net_exposure_ratio,
        "schema_version": PORTFOLIO_EXPOSURE_SCHEMA_VERSION,
    }
    return {
        "validated_instruments.csv": pd.DataFrame(instrument_rows),
        "validated_positions.csv": pd.DataFrame(position_rows),
        "portfolio_valuation.csv": pd.DataFrame([valuation_row]),
        "portfolio_exposure_summary.csv": pd.DataFrame([exposure_row]),
    }


def artifact_schema_inventory(
    artifacts: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Return manifest schema declarations for supported emitted artifacts."""
    return {
        name: {
            "schema_id": ARTIFACT_SCHEMAS[name][0],
            "schema_version": ARTIFACT_SCHEMAS[name][1],
            "units": ARTIFACT_UNITS[name],
        }
        for name in sorted(artifacts)
        if name in ARTIFACT_SCHEMAS
    }


def _manifest_artifact_declaration(
    manifest_path: Path, artifact_name: str
) -> tuple[str, str, str, dict[str, Any]]:
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactSchemaError(
            f"Could not read artifact manifest {manifest_path}: {exc}"
        ) from exc
    declarations = manifest.get("artifact_schemas")
    if not isinstance(declarations, dict):
        raise ArtifactSchemaError("Manifest is missing artifact_schemas.")
    declaration = declarations.get(artifact_name)
    if not isinstance(declaration, dict):
        raise ArtifactSchemaError(
            f"Manifest has no schema identity for {artifact_name!r}."
        )
    schema_id = declaration.get("schema_id")
    schema_version = declaration.get("schema_version")
    units = declaration.get("units")
    if (
        not isinstance(schema_id, str)
        or not isinstance(schema_version, str)
        or not isinstance(units, str)
    ):
        raise ArtifactSchemaError(
            f"Manifest schema declaration for {artifact_name!r} is malformed."
        )
    return schema_id, schema_version, units, manifest


def load_artifact(path: Path, manifest_path: Path) -> pd.DataFrame:
    """Load a supported CSV only after validating its manifest declaration."""
    artifact_path = Path(path)
    if artifact_path.suffix.lower() != ".csv":
        raise ArtifactSchemaError(
            f"The public tabular loader accepts CSV artifacts, not "
            f"{artifact_path.suffix or 'extensionless'} files."
        )
    expected = ARTIFACT_SCHEMAS.get(artifact_path.name)
    if expected is None:
        raise ArtifactSchemaError(
            f"No public artifact loader is registered for {artifact_path.name!r}."
        )
    schema_id, schema_version, units, manifest = _manifest_artifact_declaration(
        manifest_path, artifact_path.name
    )
    expected_units = ARTIFACT_UNITS[artifact_path.name]
    if (schema_id, schema_version) != expected or units != expected_units:
        raise ArtifactSchemaError(
            f"Unsupported schema for {artifact_path.name!r}: "
            f"{schema_id!r} version {schema_version!r} with units {units!r}; "
            f"expected {expected[0]!r} version {expected[1]!r} with units "
            f"{expected_units!r}."
        )
    try:
        frame = pd.read_csv(artifact_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise ArtifactSchemaError(f"Could not load {artifact_path}: {exc}") from exc
    if frame.empty:
        raise ArtifactSchemaError(f"{artifact_path.name} contains no rows.")

    required_columns = _STRICT_OUTPUT_COLUMNS.get(artifact_path.name)
    if required_columns is not None and tuple(frame.columns) != required_columns:
        raise ArtifactSchemaError(
            f"{artifact_path.name} columns do not match its declared schema."
        )
    if (
        "schema_version" in frame.columns
        and not (frame["schema_version"].astype("string") == schema_version).all()
    ):
        raise ArtifactSchemaError(
            f"{artifact_path.name} row schema versions do not match its manifest."
        )

    if artifact_path.name in {
        "adjusted_prices.csv",
        "simple_returns.csv",
        "log_returns.csv",
    }:
        if "date" not in frame.columns:
            raise ArtifactSchemaError(
                f"{artifact_path.name} is missing required 'date' column."
            )
        expected_instruments = manifest.get("data_source", {}).get("instruments")
        actual_instruments = [column for column in frame.columns if column != "date"]
        if expected_instruments != actual_instruments:
            raise ArtifactSchemaError(
                f"{artifact_path.name} instrument identities or ordering do not "
                "match the manifest."
            )
        parsed_dates = pd.to_datetime(frame.pop("date"), errors="coerce")
        if parsed_dates.isna().any():
            raise ArtifactSchemaError(
                f"{artifact_path.name} contains invalid observation dates."
            )
        frame.index = parsed_dates
        frame.index.name = "date"
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any(axis=None):
            raise ArtifactSchemaError(
                f"{artifact_path.name} contains missing or nonnumeric values."
            )
        if not numeric.map(math.isfinite).all(axis=None):
            raise ArtifactSchemaError(
                f"{artifact_path.name} contains non-finite values."
            )
        frame = numeric
        frame.index = parsed_dates
        frame.index.name = "date"

    portfolio_manifest = manifest.get("portfolio")
    if isinstance(portfolio_manifest, dict):
        reconciliation = portfolio_manifest.get("reconciliation", {})
        expected_ids = reconciliation.get("held_instrument_ids")
        if (
            artifact_path.name == "validated_instruments.csv"
            and frame["instrument_id"].tolist() != expected_ids
        ):
            raise ArtifactSchemaError(
                "validated_instruments.csv identities do not match the manifest."
            )
        if (
            artifact_path.name == "validated_positions.csv"
            and frame["instrument_id"].tolist() != expected_ids
        ):
            raise ArtifactSchemaError(
                "validated_positions.csv identities do not match the manifest."
            )
    return frame


__all__ = [
    "ADJUSTED_PRICES_SCHEMA_ID",
    "ARTIFACT_SCHEMAS",
    "ARTIFACT_UNITS",
    "LOG_RETURNS_SCHEMA_ID",
    "SIMPLE_RETURNS_SCHEMA_ID",
    "TABULAR_ARTIFACT_SCHEMA_VERSION",
    "artifact_schema_inventory",
    "load_artifact",
    "portfolio_artifact_frames",
    "read_cash",
    "read_instrument_registry",
    "read_positions",
]

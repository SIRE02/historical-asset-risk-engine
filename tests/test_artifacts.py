"""Strict parsing and version-aware consumer artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from historical_asset_risk.artifacts import (
    ADJUSTED_PRICES_SCHEMA_ID,
    ARTIFACT_UNITS,
    TABULAR_ARTIFACT_SCHEMA_VERSION,
    load_artifact,
    read_cash,
    read_instrument_registry,
    read_positions,
)
from historical_asset_risk.contracts import (
    ArtifactSchemaError,
    PortfolioSchemaError,
)


def test_book_csv_parsers_preserve_lineage_and_reject_unknown_columns(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "registry.csv"
    registry_path.write_text(
        "instrument_id,provider_ticker,listing_venue,instrument_type,"
        "price_currency,market_calendar_id,market_timezone,"
        "adjusted_return_series_id,return_adjustment_basis\n"
        "US_SPY, spy ,ARCX,standard_etf,usd,xnys,America/New_York,"
        "yahoo:SPY:adj-close,vendor total return\n",
        encoding="utf-8",
    )
    positions_path = tmp_path / "positions.csv"
    positions_path.write_text(
        "portfolio_snapshot_id,portfolio_id,as_of_date,instrument_id,quantity,"
        "valuation_price,valuation_price_timestamp,valuation_price_source,"
        "source_row_id\n"
        "snapshot-1,portfolio-1,2024-01-02,US_SPY,10,100,"
        "2024-01-02T16:00:00-05:00,official close,row-1\n",
        encoding="utf-8",
    )
    cash_path = tmp_path / "cash.csv"
    cash_path.write_text(
        "portfolio_snapshot_id,portfolio_id,as_of_date,as_of_timestamp,"
        "base_currency,market_calendar_id,market_timezone,cash_amount,cash_source\n"
        "snapshot-1,portfolio-1,2024-01-02,2024-01-02T16:05:00-05:00,"
        "usd,xnys,America/New_York,0,custodian\n",
        encoding="utf-8",
    )

    instrument = read_instrument_registry(registry_path)[0]
    position = read_positions(positions_path)[0]
    cash = read_cash(cash_path)[0]
    assert instrument.provider_ticker == "SPY"
    assert instrument.price_currency == "USD"
    assert instrument.market_calendar_id == "XNYS"
    assert position.source_row_id == "row-1"
    assert cash.cash_amount == 0.0

    positions_path.write_text(
        positions_path.read_text(encoding="utf-8").replace(
            "source_row_id\n", "source_row_id,mystery\n"
        ),
        encoding="utf-8",
    )
    with pytest.raises(PortfolioSchemaError, match="unknown column"):
        read_positions(positions_path)


def test_artifact_loader_requires_exact_manifest_schema_and_identity_order(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "adjusted_prices.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "SPY": [100.0, 101.0],
            "QQQ": [50.0, 51.0],
        }
    ).to_csv(artifact_path, index=False)
    manifest_path = tmp_path / "run_manifest.json"
    manifest = {
        "data_source": {"instruments": ["SPY", "QQQ"]},
        "artifact_schemas": {
            "adjusted_prices.csv": {
                "schema_id": ADJUSTED_PRICES_SCHEMA_ID,
                "schema_version": TABULAR_ARTIFACT_SCHEMA_VERSION,
                "units": ARTIFACT_UNITS["adjusted_prices.csv"],
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_artifact(artifact_path, manifest_path)
    assert list(loaded.columns) == ["SPY", "QQQ"]
    assert isinstance(loaded.index, pd.DatetimeIndex)

    manifest["artifact_schemas"]["adjusted_prices.csv"]["schema_version"] = "999"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactSchemaError, match="Unsupported schema"):
        load_artifact(artifact_path, manifest_path)

    manifest.pop("artifact_schemas")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactSchemaError, match="missing artifact_schemas"):
        load_artifact(artifact_path, manifest_path)

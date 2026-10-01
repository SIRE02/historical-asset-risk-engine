"""Freeze the portfolio-loss and realization contract the tail layer reads.

The schema version stays ``1.experimental`` pending promotion, but the
schema id, units, loss sign, and exact column set of the loss and realization
artifacts must not drift while tail-risk and stress work reads them.
These assertions use literal expected values so a change to a source constant
is still caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from historical_asset_risk import cli
from historical_asset_risk.artifacts import (
    ARTIFACT_SCHEMAS,
    ARTIFACT_UNITS,
    load_artifact,
)
from historical_asset_risk.contracts import ArtifactSchemaError

_FROZEN_SCHEMAS = {
    "hypothetical_portfolio_pnl.csv": (
        "historical-asset-risk/hypothetical-portfolio-pnl",
        "1.experimental",
        "base_currency_pnl_loss_is_positive",
    ),
    "proxy_realized_portfolio_pnl.csv": (
        "historical-asset-risk/proxy-realized-portfolio-pnl",
        "1.experimental",
        "base_currency_pnl_loss_is_positive",
    ),
    "risk_realizations.csv": (
        "historical-asset-risk/risk-realizations",
        "1.experimental",
        "base_currency_loss_identity",
    ),
    "portfolio_simple_return_covariance.csv": (
        "historical-asset-risk/portfolio-simple-return-covariance",
        "1.experimental",
        "daily_simple_return_squared",
    ),
}

_FROZEN_COLUMNS = {
    "hypothetical_portfolio_pnl.csv": (
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
    ),
    "proxy_realized_portfolio_pnl.csv": (
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
    ),
    "risk_realizations.csv": (
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
    ),
}

_REGISTRY = (
    "instrument_id,provider_ticker,listing_venue,instrument_type,"
    "price_currency,market_calendar_id,market_timezone,"
    "adjusted_return_series_id,return_adjustment_basis\n"
    "US_SPY,SPY,ARCX,standard_etf,USD,XNYS,America/New_York,"
    "fixture:SPY,total-return adjusted close\n"
    "US_QQQ,QQQ,XNAS,standard_etf,USD,XNYS,America/New_York,"
    "fixture:QQQ,total-return adjusted close\n"
)
_POSITIONS = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,instrument_id,quantity,"
    "valuation_price,valuation_price_timestamp,valuation_price_source,source_row_id\n"
    "snap-1,portfolio-1,2024-01-02,US_SPY,10,100,"
    "2024-01-02T16:00:00-05:00,fixture,r1\n"
    "snap-1,portfolio-1,2024-01-02,US_QQQ,-5,50,"
    "2024-01-02T16:00:00-05:00,fixture,r2\n"
)
_CASH = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,as_of_timestamp,"
    "base_currency,market_calendar_id,market_timezone,cash_amount,cash_source\n"
    "snap-1,portfolio-1,2024-01-02,2024-01-02T16:05:00-05:00,"
    "USD,XNYS,America/New_York,500,fixture\n"
)
_POSITIONS_HISTORY = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,instrument_id,quantity,"
    "valuation_price,valuation_price_timestamp,valuation_price_source,source_row_id\n"
    "h1,portfolio-1,2024-01-02,US_SPY,10,100,"
    "2024-01-02T16:00:00-05:00,fixture,hr1\n"
    "h2,portfolio-1,2024-01-03,US_SPY,10,101,"
    "2024-01-03T16:00:00-05:00,fixture,hr2\n"
)
_CASH_HISTORY = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,as_of_timestamp,"
    "base_currency,market_calendar_id,market_timezone,cash_amount,cash_source\n"
    "h1,portfolio-1,2024-01-02,2024-01-02T16:05:00-05:00,"
    "USD,XNYS,America/New_York,500,fixture\n"
    "h2,portfolio-1,2024-01-03,2024-01-03T16:05:00-05:00,"
    "USD,XNYS,America/New_York,500,fixture\n"
)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["SPY"] * len(dates) + ["QQQ"] * len(dates),
            "adjusted_close": [
                100,
                101,
                102,
                101,
                103,
                104,
                105,
                106,
                50,
                51,
                50,
                52,
                53,
                52,
                54,
                55,
            ],
        }
    ).to_csv(tmp_path / "prices.csv", index=False)
    for name, body in {
        "registry.csv": _REGISTRY,
        "positions.csv": _POSITIONS,
        "cash.csv": _CASH,
        "ph.csv": _POSITIONS_HISTORY,
        "ch.csv": _CASH_HISTORY,
    }.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "plot_rolling_volatility",
        lambda _d, path, _w: path.write_bytes(b"chart"),
    )
    monkeypatch.setattr(
        cli,
        "plot_correlation_heatmap",
        lambda _d, path: path.write_bytes(b"chart"),
    )
    output_dir = tmp_path / "out"
    from historical_asset_risk.config import AnalysisConfig

    cli.run_analysis(
        AnalysisConfig(
            provider="csv",
            csv_path=tmp_path / "prices.csv",
            output_dir=output_dir,
            tickers=("SPY", "QQQ"),
            start_date="2024-01-01",
            end_date="2024-02-01",
            rolling_window=3,
            instrument_registry_path=tmp_path / "registry.csv",
            positions_path=tmp_path / "positions.csv",
            cash_path=tmp_path / "cash.csv",
            positions_history_path=tmp_path / "ph.csv",
            cash_history_path=tmp_path / "ch.csv",
        )
    )
    return output_dir


def test_frozen_schema_identities_and_units() -> None:
    for name, (schema_id, version, units) in _FROZEN_SCHEMAS.items():
        assert ARTIFACT_SCHEMAS[name] == (schema_id, version)
        assert ARTIFACT_UNITS[name] == units


def test_frozen_loss_and_realization_columns_are_emitted_and_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = _run(tmp_path, monkeypatch)
    manifest = output_dir / "run_manifest.json"

    for name, expected_columns in _FROZEN_COLUMNS.items():
        frame = pd.read_csv(output_dir / name)
        assert tuple(frame.columns) == expected_columns, name
        loaded = load_artifact(output_dir / name, manifest)
        assert tuple(loaded.columns) == expected_columns, name
        assert (loaded["loss_sign"] == "loss_is_positive").all()

    manifest_data = json.loads(manifest.read_text())
    for name in _FROZEN_COLUMNS:
        assert name in manifest_data["artifact_schemas"]


def test_load_artifact_rejects_a_drifted_loss_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = _run(tmp_path, monkeypatch)
    target = output_dir / "hypothetical_portfolio_pnl.csv"
    frame = pd.read_csv(target)
    frame.rename(columns={"hypothetical_loss": "loss"}).to_csv(target, index=False)
    with pytest.raises(ArtifactSchemaError, match="do not match its declared schema"):
        load_artifact(target, output_dir / "run_manifest.json")

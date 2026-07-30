"""Offline end-to-end Phase 3 portfolio workflow test."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from historical_asset_risk import cli
from historical_asset_risk.artifacts import load_artifact
from historical_asset_risk.config import AnalysisConfig


def test_complete_portfolio_run_emits_reconciled_versioned_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    records = pd.DataFrame(
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
    )
    prices_path = tmp_path / "prices.csv"
    records.to_csv(prices_path, index=False)
    registry_path = tmp_path / "registry.csv"
    registry_path.write_text(
        "instrument_id,provider_ticker,listing_venue,instrument_type,"
        "price_currency,market_calendar_id,market_timezone,"
        "adjusted_return_series_id,return_adjustment_basis\n"
        "US_SPY,SPY,ARCX,standard_etf,USD,XNYS,America/New_York,"
        "fixture:SPY,total-return adjusted close\n"
        "US_QQQ,QQQ,XNAS,standard_etf,USD,XNYS,America/New_York,"
        "fixture:QQQ,total-return adjusted close\n",
        encoding="utf-8",
    )
    positions_path = tmp_path / "positions.csv"
    positions_path.write_text(
        "portfolio_snapshot_id,portfolio_id,as_of_date,instrument_id,quantity,"
        "valuation_price,valuation_price_timestamp,valuation_price_source,"
        "source_row_id\n"
        "snapshot-1,portfolio-1,2024-01-02,US_SPY,10,100,"
        "2024-01-02T16:00:00-05:00,fixture,row-1\n"
        "snapshot-1,portfolio-1,2024-01-02,US_QQQ,-5,50,"
        "2024-01-02T16:00:00-05:00,fixture,row-2\n",
        encoding="utf-8",
    )
    cash_path = tmp_path / "cash.csv"
    cash_path.write_text(
        "portfolio_snapshot_id,portfolio_id,as_of_date,as_of_timestamp,"
        "base_currency,market_calendar_id,market_timezone,cash_amount,cash_source\n"
        "snapshot-1,portfolio-1,2024-01-02,2024-01-02T16:05:00-05:00,"
        "USD,XNYS,America/New_York,500,fixture\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "outputs"
    config = AnalysisConfig(
        provider="csv",
        csv_path=prices_path,
        instrument_registry_path=registry_path,
        positions_path=positions_path,
        cash_path=cash_path,
        output_dir=output_dir,
        tickers=("SPY", "QQQ"),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=3,
    )
    monkeypatch.setattr(
        cli,
        "plot_rolling_volatility",
        lambda _data, path, _window: path.write_bytes(b"chart"),
    )
    monkeypatch.setattr(
        cli,
        "plot_correlation_heatmap",
        lambda _data, path: path.write_bytes(b"chart"),
    )

    cli.run_analysis(config)

    required = {
        "validated_instruments.csv",
        "validated_positions.csv",
        "portfolio_valuation.csv",
        "portfolio_exposure_summary.csv",
    }
    assert required.issubset(path.name for path in output_dir.iterdir())
    valuation = pd.read_csv(output_dir / "portfolio_valuation.csv").iloc[0]
    assert valuation["long_exposure"] == 1000.0
    assert valuation["short_exposure"] == -250.0
    assert valuation["gross_exposure"] == 1250.0
    assert valuation["portfolio_value"] == 1250.0
    exposure = pd.read_csv(output_dir / "portfolio_exposure_summary.csv").iloc[0]
    assert exposure["cash_weight"] == pytest.approx(0.4)
    assert exposure["instrument_weight_sum"] == pytest.approx(0.6)
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["portfolio"]["portfolio_id"] == "portfolio-1"
    assert manifest["portfolio"]["calendar"]["version"]
    assert set(required).issubset(manifest["artifact_schemas"])
    loaded_positions = load_artifact(
        output_dir / "validated_positions.csv",
        output_dir / "run_manifest.json",
    )
    assert loaded_positions["instrument_id"].tolist() == ["US_SPY", "US_QQQ"]
    quality = json.loads((output_dir / "data_quality_report.json").read_text())
    assert quality["portfolio"]["position_count"] == 2
    assert quality["portfolio"]["validation_exceptions"] == []

    changed_cash_path = tmp_path / "changed_cash.csv"
    changed_cash_path.write_text(
        cash_path.read_text(encoding="utf-8").replace(",500,fixture", ",600,fixture"),
        encoding="utf-8",
    )
    changed_config = replace(config, cash_path=changed_cash_path)
    with pytest.raises(ValueError, match="materially different portfolio run"):
        cli.run_analysis(changed_config)

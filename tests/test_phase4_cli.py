"""Offline end-to-end Phase 4 portfolio P&L and risk workflow test."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from historical_asset_risk import cli
from historical_asset_risk.artifacts import load_artifact
from historical_asset_risk.config import AnalysisConfig

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
    "snapshot-1,portfolio-1,2024-01-02,US_SPY,10,100,"
    "2024-01-02T16:00:00-05:00,fixture,row-1\n"
    "snapshot-1,portfolio-1,2024-01-02,US_QQQ,-5,50,"
    "2024-01-02T16:00:00-05:00,fixture,row-2\n"
)
_CASH = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,as_of_timestamp,"
    "base_currency,market_calendar_id,market_timezone,cash_amount,cash_source\n"
    "snapshot-1,portfolio-1,2024-01-02,2024-01-02T16:05:00-05:00,"
    "USD,XNYS,America/New_York,500,fixture\n"
)


def _prices(tmp_path: Path) -> Path:
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
    path = tmp_path / "prices.csv"
    records.to_csv(path, index=False)
    return path


_POSITIONS_HISTORY = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,instrument_id,quantity,"
    "valuation_price,valuation_price_timestamp,valuation_price_source,source_row_id\n"
    "snap-h1,portfolio-1,2024-01-02,US_SPY,10,100,"
    "2024-01-02T16:00:00-05:00,fixture,h-row-1\n"
    "snap-h2,portfolio-1,2024-01-03,US_SPY,10,101,"
    "2024-01-03T16:00:00-05:00,fixture,h-row-2\n"
    "snap-h2,portfolio-1,2024-01-03,US_QQQ,-5,51,"
    "2024-01-03T16:00:00-05:00,fixture,h-row-3\n"
)
_CASH_HISTORY = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,as_of_timestamp,"
    "base_currency,market_calendar_id,market_timezone,cash_amount,cash_source\n"
    "snap-h1,portfolio-1,2024-01-02,2024-01-02T16:05:00-05:00,"
    "USD,XNYS,America/New_York,500,fixture\n"
    "snap-h2,portfolio-1,2024-01-03,2024-01-03T16:05:00-05:00,"
    "USD,XNYS,America/New_York,500,fixture\n"
)


def _base_config(
    tmp_path: Path, output_dir: Path, *, portfolio: bool, history: bool = False
) -> AnalysisConfig:
    kwargs: dict[str, object] = {
        "provider": "csv",
        "csv_path": _prices(tmp_path),
        "output_dir": output_dir,
        "tickers": ("SPY", "QQQ"),
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "rolling_window": 3,
    }
    if portfolio:
        registry = tmp_path / "registry.csv"
        registry.write_text(_REGISTRY, encoding="utf-8")
        positions = tmp_path / "positions.csv"
        positions.write_text(_POSITIONS, encoding="utf-8")
        cash = tmp_path / "cash.csv"
        cash.write_text(_CASH, encoding="utf-8")
        kwargs.update(
            instrument_registry_path=registry,
            positions_path=positions,
            cash_path=cash,
        )
    if history:
        positions_history = tmp_path / "positions_history.csv"
        positions_history.write_text(_POSITIONS_HISTORY, encoding="utf-8")
        cash_history = tmp_path / "cash_history.csv"
        cash_history.write_text(_CASH_HISTORY, encoding="utf-8")
        kwargs.update(
            positions_history_path=positions_history,
            cash_history_path=cash_history,
        )
    return AnalysisConfig(**kwargs)


@pytest.fixture(autouse=True)
def _stub_charts(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_phase4_book_run_emits_reconciled_pnl_and_risk(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    cli.run_analysis(_base_config(tmp_path, output_dir, portfolio=True))

    names = {path.name for path in output_dir.iterdir()}
    assert {
        "portfolio_aligned_simple_returns.csv",
        "hypothetical_portfolio_pnl.csv",
        "portfolio_simple_return_covariance.csv",
        "portfolio_simple_return_correlation.csv",
        "simple_return_summary.csv",
        "portfolio_risk_summary.csv",
        "portfolio_risk_contributions.csv",
        "portfolio_concentration_summary.csv",
    }.issubset(names)

    pnl = pd.read_csv(output_dir / "hypothetical_portfolio_pnl.csv")
    # Interval 2024-01-02 -> 2024-01-03: SPY +1%, QQQ +2%; 1000*0.01 - 250*0.02.
    first = pnl.iloc[0]
    assert first["period_start"] == "2024-01-02"
    assert first["hypothetical_pnl"] == pytest.approx(5.0)
    assert first["hypothetical_loss"] == pytest.approx(-5.0)
    assert (pnl["hypothetical_loss"] == -pnl["hypothetical_pnl"]).all()

    contributions = pd.read_csv(output_dir / "portfolio_risk_contributions.csv")
    summary = pd.read_csv(output_dir / "portfolio_risk_summary.csv").iloc[0]
    assert contributions["component_volatility"].sum() == pytest.approx(
        summary["return_volatility"]
    )
    assert contributions["percentage_component"].sum() == pytest.approx(1.0)
    assert summary["return_variance"] >= 0.0

    covariance = pd.read_csv(output_dir / "portfolio_simple_return_covariance.csv")
    assert covariance["instrument_id"].tolist() == ["US_SPY", "US_QQQ"]

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    phase4 = manifest["portfolio_phase4"]
    assert phase4["covariance"]["ddof"] == 1
    assert phase4["covariance"]["shrinkage"] == "none"
    assert phase4["distinct_from"]["log_return_covariance_artifact"] == (
        "covariance_matrix.csv"
    )
    assert "portfolio_risk_summary.csv" in manifest["artifact_schemas"]

    quality = json.loads((output_dir / "data_quality_report.json").read_text())
    assert quality["portfolio"]["phase4"]["component_volatility_reconciles"] is True
    assert quality["portfolio"]["phase4"]["proxy_realized_pnl_available"] is False

    loaded = load_artifact(
        output_dir / "hypothetical_portfolio_pnl.csv",
        output_dir / "run_manifest.json",
    )
    assert "hypothetical_pnl" in loaded.columns

    # The simple-return covariance stays a distinct artifact from the Phase 2
    # log-return covariance: different schema id, different values.
    schemas = manifest["artifact_schemas"]
    assert (
        schemas["portfolio_simple_return_covariance.csv"]["schema_id"]
        != schemas["covariance_matrix.csv"]["schema_id"]
    )
    log_cov = pd.read_csv(output_dir / "covariance_matrix.csv", index_col=0)
    simple_cov = pd.read_csv(
        output_dir / "portfolio_simple_return_covariance.csv", index_col=0
    )
    assert log_cov.to_numpy() != pytest.approx(simple_cov.to_numpy())


def test_proxy_realized_history_emits_realization_identity(tmp_path: Path) -> None:
    output_dir = tmp_path / "with_history"
    cli.run_analysis(_base_config(tmp_path, output_dir, portfolio=True, history=True))

    names = {path.name for path in output_dir.iterdir()}
    assert {"proxy_realized_portfolio_pnl.csv", "risk_realizations.csv"}.issubset(names)

    proxy = pd.read_csv(output_dir / "proxy_realized_portfolio_pnl.csv")
    assert proxy["as_of_date"].tolist() == ["2024-01-02", "2024-01-03"]
    assert (proxy["outcome_status"] == "realized").all()
    assert proxy.loc[0, "target_period_end"] == "2024-01-03"
    # snap-h1: 1000 USD SPY exposure over SPY +1% = 10 USD.
    assert proxy.loc[0, "proxy_realized_pnl"] == pytest.approx(10.0)
    assert (proxy["proxy_realized_loss"] == -proxy["proxy_realized_pnl"]).all()

    realizations = pd.read_csv(output_dir / "risk_realizations.csv")
    assert (realizations["loss_sign"] == "loss_is_positive").all()
    assert realizations["calculation_version"].nunique() == 1
    assert set(realizations["outcome_status"]) == {"realized"}

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    realization_meta = manifest["portfolio_phase4"]["realizations"]
    assert realization_meta["status"] == "computed"
    assert realization_meta["realized_count"] == 2
    assert "predicted_loss" in realization_meta["downstream_forecast_join_fields"]
    quality = json.loads((output_dir / "data_quality_report.json").read_text())
    assert quality["portfolio"]["phase4"]["proxy_realized_pnl_available"] is True


def test_returns_only_run_writes_no_phase4_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "returns_only"
    cli.run_analysis(_base_config(tmp_path, output_dir, portfolio=False))

    names = {path.name for path in output_dir.iterdir()}
    assert "simple_returns.csv" in names
    assert not any(
        name.startswith(("portfolio_simple_return", "hypothetical_", "portfolio_risk"))
        for name in names
    )
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert "portfolio_phase4" not in manifest


def test_frozen_consumer_return_artifacts_are_unchanged_by_phase4(
    tmp_path: Path,
) -> None:
    with_book = tmp_path / "with_book"
    without_book = tmp_path / "without_book"
    cli.run_analysis(_base_config(tmp_path, without_book, portfolio=False))
    cli.run_analysis(_base_config(tmp_path, with_book, portfolio=True))

    for name in ("adjusted_prices.csv", "simple_returns.csv", "log_returns.csv"):
        assert (with_book / name).read_text() == (without_book / name).read_text()

    for report in (with_book, without_book):
        schemas = json.loads((report / "run_manifest.json").read_text())[
            "artifact_schemas"
        ]
        assert schemas["simple_returns.csv"]["units"] == (
            "decimal_return_per_observation"
        )

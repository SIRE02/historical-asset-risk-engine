"""Offline end-to-end Phase 5 tail-risk and stress workflow test."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from historical_asset_risk import cli
from historical_asset_risk.config import AnalysisConfig
from historical_asset_risk.stress import (
    STRESS_CATALOG_SCHEMA_ID,
    STRESS_SCENARIO_SCHEMA_VERSION,
    InstrumentShock,
    canonical_scenario_payload,
    scenario_content_hash,
)

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


def _catalog(tmp_path: Path) -> Path:
    payload = canonical_scenario_payload(
        scenario_id="risk-off",
        scenario_version="1",
        name="Risk-off",
        kind="hypothetical",
        description="fixture",
        shocks=[InstrumentShock("US_SPY", -0.12), InstrumentShock("US_QQQ", -0.18)],
    )
    entry = {**payload, "content_hash": scenario_content_hash(payload)}
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_id": STRESS_CATALOG_SCHEMA_ID,
                "schema_version": STRESS_SCENARIO_SCHEMA_VERSION,
                "scenarios": [entry],
            }
        ),
        encoding="utf-8",
    )
    return path


_POSITIONS_HISTORY = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,instrument_id,quantity,"
    "valuation_price,valuation_price_timestamp,valuation_price_source,source_row_id\n"
    "h1,portfolio-1,2024-01-05,US_SPY,10,101,"
    "2024-01-05T16:00:00-05:00,fixture,hr1\n"
    "h2,portfolio-1,2024-01-09,US_SPY,10,104,"
    "2024-01-09T16:00:00-05:00,fixture,hr2\n"
)
_CASH_HISTORY = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,as_of_timestamp,"
    "base_currency,market_calendar_id,market_timezone,cash_amount,cash_source\n"
    "h1,portfolio-1,2024-01-05,2024-01-05T16:05:00-05:00,"
    "USD,XNYS,America/New_York,500,fixture\n"
    "h2,portfolio-1,2024-01-09,2024-01-09T16:05:00-05:00,"
    "USD,XNYS,America/New_York,500,fixture\n"
)


def _config(
    tmp_path: Path,
    output_dir: Path,
    *,
    portfolio: bool,
    stress: bool = False,
    history: bool = False,
    quantile_method: str = "linear",
) -> AnalysisConfig:
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
    kwargs: dict[str, object] = {
        "provider": "csv",
        "csv_path": tmp_path / "prices.csv",
        "output_dir": output_dir,
        "tickers": ("SPY", "QQQ"),
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "rolling_window": 3,
        "quantile_method": quantile_method,
        "tail_risk_confidence_level": 0.9,
    }
    if portfolio:
        (tmp_path / "registry.csv").write_text(_REGISTRY, encoding="utf-8")
        (tmp_path / "positions.csv").write_text(_POSITIONS, encoding="utf-8")
        (tmp_path / "cash.csv").write_text(_CASH, encoding="utf-8")
        kwargs.update(
            instrument_registry_path=tmp_path / "registry.csv",
            positions_path=tmp_path / "positions.csv",
            cash_path=tmp_path / "cash.csv",
        )
    if stress:
        kwargs["stress_catalog_path"] = _catalog(tmp_path)
    if history:
        (tmp_path / "ph.csv").write_text(_POSITIONS_HISTORY, encoding="utf-8")
        (tmp_path / "ch.csv").write_text(_CASH_HISTORY, encoding="utf-8")
        kwargs.update(
            positions_history_path=tmp_path / "ph.csv",
            cash_history_path=tmp_path / "ch.csv",
        )
    return AnalysisConfig(**kwargs)


@pytest.fixture(autouse=True)
def _stub_charts(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_phase5_emits_reconciled_var_es_and_stress(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    cli.run_analysis(_config(tmp_path, output_dir, portfolio=True, stress=True))

    names = {p.name for p in output_dir.iterdir()}
    assert {
        "portfolio_value_at_risk.csv",
        "portfolio_expected_shortfall.csv",
        "portfolio_expected_shortfall_tail_weights.csv",
        "tail_risk_comparison.csv",
        "stress_test_results.csv",
        "stress_contributions.csv",
        "stress_scenario_catalog.json",
    }.issubset(names)

    var = pd.read_csv(output_dir / "portfolio_value_at_risk.csv")
    es = pd.read_csv(output_dir / "portfolio_expected_shortfall.csv")
    assert set(var["dimension"]) == {"return", "currency"}
    merged = var.merge(es, on="dimension", suffixes=("_var", "_es"))
    assert (merged["expected_shortfall"] >= merged["value_at_risk_var"]).all()
    assert (es["es_ge_var"]).all()

    weights = pd.read_csv(output_dir / "portfolio_expected_shortfall_tail_weights.csv")
    for _dimension, group in weights.groupby("dimension"):
        assert group["weight"].sum() == pytest.approx(1.0)

    results = pd.read_csv(output_dir / "stress_test_results.csv").iloc[0]
    # SPY 1000 * -0.12 + QQQ -250 * -0.18 = -120 + 45 = -75
    assert results["scenario_pnl"] == pytest.approx(-75.0)
    assert results["scenario_loss"] == pytest.approx(75.0)
    contributions = pd.read_csv(output_dir / "stress_contributions.csv")
    assert contributions["scenario_pnl"].sum() == pytest.approx(-75.0)

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    phase5 = manifest["portfolio_phase5"]
    assert phase5["confidence_level"] == 0.9
    assert phase5["quantile_method"] == "generalized_inverse_empirical_cdf"
    assert phase5["stress"]["probability_fields"] == "none"
    quality = json.loads((output_dir / "data_quality_report.json").read_text())
    assert quality["portfolio"]["phase5"]["es_ge_var"] is True


def test_descriptive_quantile_method_does_not_change_canonical_var(
    tmp_path: Path,
) -> None:
    lower = tmp_path / "lower"
    higher = tmp_path / "higher"
    cli.run_analysis(_config(tmp_path, lower, portfolio=True, quantile_method="lower"))
    cli.run_analysis(
        _config(tmp_path, higher, portfolio=True, quantile_method="higher")
    )
    var_lower = pd.read_csv(lower / "portfolio_value_at_risk.csv")["value_at_risk"]
    var_higher = pd.read_csv(higher / "portfolio_value_at_risk.csv")["value_at_risk"]
    assert var_lower.tolist() == pytest.approx(var_higher.tolist())


def test_trailing_tail_risk_series_is_keyed_by_as_of_date(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    cli.run_analysis(_config(tmp_path, output_dir, portfolio=True, history=True))

    trailing = pd.read_csv(output_dir / "trailing_portfolio_tail_risk.csv")
    assert trailing["as_of_date"].tolist() == ["2024-01-05", "2024-01-09"]
    assert "exposure_snapshot_id" in trailing.columns
    # every trailing window ends no later than its as_of_date, so the later
    # snapshot sees at least as many observations as the earlier one
    counts = trailing.set_index("as_of_date")["window_observation_count"]
    assert counts["2024-01-09"] >= counts["2024-01-05"]
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["portfolio_phase5"]["trailing"]["not_a_forecast_store"] is True


def test_returns_only_run_writes_no_phase5_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "returns_only"
    cli.run_analysis(_config(tmp_path, output_dir, portfolio=False))
    names = {p.name for p in output_dir.iterdir()}
    assert not any(
        n.startswith(
            (
                "portfolio_value_at_risk",
                "portfolio_expected_shortfall",
                "tail_risk_comparison",
                "stress_",
            )
        )
        for n in names
    )
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert "portfolio_phase5" not in manifest

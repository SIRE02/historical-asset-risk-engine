"""Offline end-to-end tail-risk and stress workflow test."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from historical_asset_risk import cli
from historical_asset_risk.artifacts import (
    _AS_OF_BOOK_ARTIFACTS,
    ARTIFACT_SCHEMAS,
    load_artifact,
)
from historical_asset_risk.config import AnalysisConfig
from historical_asset_risk.contracts import ArtifactSchemaError
from historical_asset_risk.data_loader import price_content_hash
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


def test_tail_analytics_emits_reconciled_var_es_and_stress(tmp_path: Path) -> None:
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
    tail_analytics = manifest["tail_analytics"]
    assert tail_analytics["confidence_level"] == 0.9
    assert tail_analytics["quantile_method"] == "generalized_inverse_empirical_cdf"
    assert tail_analytics["stress"]["probability_fields"] == "none"
    quality = json.loads((output_dir / "data_quality_report.json").read_text())
    assert quality["portfolio"]["tail"]["es_ge_var"] is True


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
    assert manifest["tail_analytics"]["trailing"]["not_a_forecast_store"] is True


def test_returns_only_run_writes_no_tail_analytics_files(tmp_path: Path) -> None:
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
    assert "tail_analytics" not in manifest


def test_headline_window_is_the_full_sample_and_warnings_are_not_repeated(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    config = replace(
        _config(tmp_path, output_dir, portfolio=True, history=True),
        tail_risk_window=3,
    )
    cli.run_analysis(config)

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    tail_analytics = manifest["tail_analytics"]
    assert tail_analytics["window"] == "full_aligned_sample"
    assert tail_analytics["trailing"]["window"] == 3
    var = pd.read_csv(output_dir / "portfolio_value_at_risk.csv")
    assert (var["observation_count"] == tail_analytics["observation_count"]).all()

    # 7 intervals at 0.9 leave a tail of 0.7: VaR and ES warn identically, and
    # each dimension's warning is stored once.
    quality = json.loads((output_dir / "data_quality_report.json").read_text())
    warnings = quality["portfolio"]["tail"]["tail_sample_warnings"]
    assert len(warnings) == 2
    assert {warning.split(":")[0] for warning in warnings} == {"return", "currency"}


def test_rerun_removes_stale_engine_artifacts_and_keeps_other_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    cli.run_analysis(_config(tmp_path, output_dir, portfolio=True, stress=True))
    assert (output_dir / "stress_test_results.csv").is_file()
    (output_dir / "notes.txt").write_text("kept", encoding="utf-8")

    cli.run_analysis(_config(tmp_path, output_dir, portfolio=True, stress=False))

    names = {p.name for p in output_dir.iterdir()}
    assert (
        not {
            "stress_test_results.csv",
            "stress_contributions.csv",
            "stress_scenario_catalog.json",
        }
        & names
    )
    assert "notes.txt" in names
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert names - {"notes.txt"} == set(manifest["generated_artifacts"])


def test_load_artifact_refuses_a_file_from_another_book(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    cli.run_analysis(
        _config(tmp_path, output_dir, portfolio=True, stress=True, history=True)
    )
    manifest_path = output_dir / "run_manifest.json"

    # Every CSV of a genuine run loads, including the history files whose rows
    # carry the earlier snapshots' identities rather than the as-of book's.
    written = {path.name for path in output_dir.glob("*.csv")}
    assert _AS_OF_BOOK_ARTIFACTS <= written
    for name in written & set(ARTIFACT_SCHEMAS):
        load_artifact(output_dir / name, manifest_path)
    trailing = pd.read_csv(output_dir / "trailing_portfolio_tail_risk.csv")
    assert trailing["exposure_snapshot_id"].nunique() > 1

    valuation_path = output_dir / "portfolio_valuation.csv"
    genuine = pd.read_csv(valuation_path)
    # A blank identity must not pass as a match: comparing a nullable column
    # and calling ``.all()`` would skip the missing value.
    for tampered_id in ("exp_from_another_run", None):
        valuation = genuine.assign(
            exposure_snapshot_id=tampered_id, portfolio_value=999999
        )
        valuation.to_csv(valuation_path, index=False)
        with pytest.raises(ArtifactSchemaError, match="exposure_snapshot_id"):
            load_artifact(valuation_path, manifest_path)

    genuine.assign(schema_version=None).to_csv(valuation_path, index=False)
    with pytest.raises(ArtifactSchemaError, match="row schema versions"):
        load_artifact(valuation_path, manifest_path)


def test_returns_only_run_refuses_a_portfolio_output_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    cli.run_analysis(_config(tmp_path, output_dir, portfolio=True))
    with pytest.raises(ValueError, match="materially different portfolio run"):
        cli.run_analysis(_config(tmp_path, output_dir, portfolio=False))
    assert (output_dir / "portfolio_value_at_risk.csv").is_file()


def test_manifest_price_hash_matches_the_written_prices(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    cli.run_analysis(_config(tmp_path, output_dir, portfolio=True))

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    written = pd.read_csv(
        output_dir / "adjusted_prices.csv", index_col="date", parse_dates=["date"]
    )
    assert manifest["data_source"]["price_content_hash"] == price_content_hash(written)


def test_load_artifact_refuses_prices_that_no_longer_match_the_hash(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "out"
    cli.run_analysis(_config(tmp_path, output_dir, portfolio=True))
    prices_path = output_dir / "adjusted_prices.csv"
    manifest_path = output_dir / "run_manifest.json"
    load_artifact(prices_path, manifest_path)

    edited = pd.read_csv(prices_path)
    edited.loc[1, "SPY"] = 80.0
    edited.to_csv(prices_path, index=False)
    with pytest.raises(ArtifactSchemaError, match="price_content_hash"):
        load_artifact(prices_path, manifest_path)

    # A v0.1.1 manifest has no hash; its prices still load.
    manifest = json.loads(manifest_path.read_text())
    del manifest["data_source"]["price_content_hash"]
    manifest_path.write_text(json.dumps(manifest))
    assert load_artifact(prices_path, manifest_path).loc["2024-01-03", "SPY"] == 80.0


def test_portfolio_run_rejects_a_session_missing_from_every_ticker(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, tmp_path / "out", portfolio=True)
    prices_path = tmp_path / "prices.csv"
    prices = pd.read_csv(prices_path)
    prices.loc[prices["date"] != "2024-01-08"].to_csv(prices_path, index=False)

    with pytest.raises(ValueError, match=r"skip XNYS session.*missing 2024-01-08"):
        cli.run_analysis(config)
    assert not (tmp_path / "out").exists()

    # A returns-only run has no calendar, so the provider-union rule is the
    # only gap check it gets; it still runs on the same prices.
    returns_only = replace(
        config,
        instrument_registry_path=None,
        positions_path=None,
        cash_path=None,
        output_dir=tmp_path / "r",
    )
    cli.run_analysis(returns_only)
    assert (tmp_path / "r" / "simple_returns.csv").is_file()


def test_history_holding_an_unpriced_instrument_fails_naming_it(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, tmp_path / "out", portfolio=True, history=True)
    with (tmp_path / "registry.csv").open("a", encoding="utf-8") as registry:
        registry.write(
            "US_TLT,TLT,XNAS,standard_etf,USD,XNYS,America/New_York,"
            "fixture:TLT,total-return adjusted close\n"
        )
    with (tmp_path / "ph.csv").open("a", encoding="utf-8") as history:
        history.write(
            "h2,portfolio-1,2024-01-09,US_TLT,5,90,"
            "2024-01-09T16:00:00-05:00,fixture,hr3\n"
        )

    # The documented reason `missing_instrument_return` never appears in a CLI
    # run: the history is checked against the priced tickers before any P&L.
    with pytest.raises(ValueError, match="missing from the selected.*TLT"):
        cli.run_analysis(config)

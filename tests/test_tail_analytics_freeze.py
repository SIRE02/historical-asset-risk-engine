"""Freeze the tail-risk and stress column contract.

Schema versions stay ``1.experimental`` pending promotion, but the schema id,
units, and exact column set of each tail-analytics artifact must not drift
behind the frozen public API. Expected values are literal so a
change to a source constant is still caught.
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
from historical_asset_risk.config import AnalysisConfig
from historical_asset_risk.contracts import ArtifactSchemaError
from historical_asset_risk.stress import (
    STRESS_CATALOG_SCHEMA_ID,
    STRESS_SCENARIO_SCHEMA_VERSION,
    InstrumentShock,
    canonical_scenario_payload,
    scenario_content_hash,
)

_FROZEN_SCHEMAS = {
    "portfolio_value_at_risk.csv": (
        "historical-asset-risk/portfolio-value-at-risk",
        "1.experimental",
        "loss_units_by_named_dimension",
    ),
    "portfolio_expected_shortfall.csv": (
        "historical-asset-risk/portfolio-expected-shortfall",
        "1.experimental",
        "loss_units_by_named_dimension",
    ),
    "portfolio_expected_shortfall_tail_weights.csv": (
        "historical-asset-risk/expected-shortfall-tail-weights",
        "1.experimental",
        "loss_and_dimensionless_weight",
    ),
    "tail_risk_comparison.csv": (
        "historical-asset-risk/tail-risk-comparison",
        "1.experimental",
        "loss_units_by_named_dimension",
    ),
    "trailing_portfolio_tail_risk.csv": (
        "historical-asset-risk/trailing-portfolio-tail-risk",
        "1.experimental",
        "base_currency_loss",
    ),
    "stress_test_results.csv": (
        "historical-asset-risk/stress-test-results",
        "1.experimental",
        "base_currency_pnl_loss_is_positive",
    ),
    "stress_contributions.csv": (
        "historical-asset-risk/stress-contributions",
        "1.experimental",
        "base_currency_pnl_loss_is_positive",
    ),
    "stress_scenario_catalog.json": (
        "historical-asset-risk/stress-scenario-catalog",
        "1.experimental",
        "decimal_simple_return_shocks",
    ),
}

_FROZEN_COLUMNS = {
    "portfolio_value_at_risk.csv": (
        "portfolio_snapshot_id",
        "exposure_snapshot_id",
        "portfolio_id",
        "as_of_date",
        "base_currency",
        "data_snapshot_id",
        "dimension",
        "units",
        "confidence_level",
        "observation_count",
        "value_at_risk",
        "order_statistic_rank",
        "quantile_method",
        "tail_sample_warning",
        "schema_version",
    ),
    "portfolio_expected_shortfall.csv": (
        "portfolio_snapshot_id",
        "exposure_snapshot_id",
        "portfolio_id",
        "as_of_date",
        "base_currency",
        "data_snapshot_id",
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
    ),
    "portfolio_expected_shortfall_tail_weights.csv": (
        "exposure_snapshot_id",
        "dimension",
        "units",
        "confidence_level",
        "order_statistic_rank",
        "loss",
        "weight",
        "schema_version",
    ),
    "tail_risk_comparison.csv": (
        "portfolio_snapshot_id",
        "exposure_snapshot_id",
        "portfolio_id",
        "as_of_date",
        "base_currency",
        "data_snapshot_id",
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
    ),
    "trailing_portfolio_tail_risk.csv": (
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
    ),
    "stress_test_results.csv": (
        "portfolio_snapshot_id",
        "exposure_snapshot_id",
        "portfolio_id",
        "as_of_date",
        "base_currency",
        "data_snapshot_id",
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
    ),
    "stress_contributions.csv": (
        "exposure_snapshot_id",
        "scenario_id",
        "scenario_version",
        "instrument_id",
        "currency_exposure",
        "simple_return_shock",
        "scenario_pnl",
        "scenario_loss",
        "schema_version",
    ),
}

_REGISTRY = (
    "instrument_id,provider_ticker,listing_venue,instrument_type,"
    "price_currency,market_calendar_id,market_timezone,"
    "adjusted_return_series_id,return_adjustment_basis\n"
    "US_SPY,SPY,ARCX,standard_etf,USD,XNYS,America/New_York,fixture:SPY,adj\n"
    "US_QQQ,QQQ,XNAS,standard_etf,USD,XNYS,America/New_York,fixture:QQQ,adj\n"
)
_POSITIONS = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,instrument_id,quantity,"
    "valuation_price,valuation_price_timestamp,valuation_price_source,source_row_id\n"
    "snap-1,portfolio-1,2024-01-02,US_SPY,10,100,2024-01-02T16:00:00-05:00,fx,r1\n"
    "snap-1,portfolio-1,2024-01-02,US_QQQ,-5,50,2024-01-02T16:00:00-05:00,fx,r2\n"
)
_CASH = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,as_of_timestamp,"
    "base_currency,market_calendar_id,market_timezone,cash_amount,cash_source\n"
    "snap-1,portfolio-1,2024-01-02,2024-01-02T16:05:00-05:00,"
    "USD,XNYS,America/New_York,500,fx\n"
)
_POSITIONS_HISTORY = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,instrument_id,quantity,"
    "valuation_price,valuation_price_timestamp,valuation_price_source,source_row_id\n"
    "h1,portfolio-1,2024-01-05,US_SPY,10,101,2024-01-05T16:00:00-05:00,fx,hr1\n"
    "h2,portfolio-1,2024-01-09,US_SPY,10,104,2024-01-09T16:00:00-05:00,fx,hr2\n"
)
_CASH_HISTORY = (
    "portfolio_snapshot_id,portfolio_id,as_of_date,as_of_timestamp,"
    "base_currency,market_calendar_id,market_timezone,cash_amount,cash_source\n"
    "h1,portfolio-1,2024-01-05,2024-01-05T16:05:00-05:00,USD,XNYS,America/New_York,500,fx\n"
    "h2,portfolio-1,2024-01-09,2024-01-09T16:05:00-05:00,USD,XNYS,America/New_York,500,fx\n"
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
    payload = canonical_scenario_payload(
        scenario_id="risk-off",
        scenario_version="1",
        name="Risk-off",
        kind="hypothetical",
        description="fixture",
        shocks=[InstrumentShock("US_SPY", -0.12), InstrumentShock("US_QQQ", -0.18)],
    )
    for name, body in {
        "registry.csv": _REGISTRY,
        "positions.csv": _POSITIONS,
        "cash.csv": _CASH,
        "ph.csv": _POSITIONS_HISTORY,
        "ch.csv": _CASH_HISTORY,
    }.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    (tmp_path / "catalog.json").write_text(
        json.dumps(
            {
                "schema_id": STRESS_CATALOG_SCHEMA_ID,
                "schema_version": STRESS_SCENARIO_SCHEMA_VERSION,
                "scenarios": [
                    {**payload, "content_hash": scenario_content_hash(payload)}
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli, "plot_rolling_volatility", lambda _d, p, _w: p.write_bytes(b"c")
    )
    monkeypatch.setattr(
        cli, "plot_correlation_heatmap", lambda _d, p: p.write_bytes(b"c")
    )
    output_dir = tmp_path / "out"
    cli.run_analysis(
        AnalysisConfig(
            provider="csv",
            csv_path=tmp_path / "prices.csv",
            output_dir=output_dir,
            tickers=("SPY", "QQQ"),
            start_date="2024-01-01",
            end_date="2024-02-01",
            rolling_window=3,
            tail_risk_confidence_level=0.9,
            instrument_registry_path=tmp_path / "registry.csv",
            positions_path=tmp_path / "positions.csv",
            cash_path=tmp_path / "cash.csv",
            positions_history_path=tmp_path / "ph.csv",
            cash_history_path=tmp_path / "ch.csv",
            stress_catalog_path=tmp_path / "catalog.json",
        )
    )
    return output_dir


def test_frozen_tail_analytics_schema_identities_and_units() -> None:
    for name, (schema_id, version, units) in _FROZEN_SCHEMAS.items():
        assert ARTIFACT_SCHEMAS[name] == (schema_id, version), name
        assert ARTIFACT_UNITS[name] == units, name


def test_frozen_tail_analytics_columns_are_emitted_and_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = _run(tmp_path, monkeypatch)
    manifest = output_dir / "run_manifest.json"
    for name, expected in _FROZEN_COLUMNS.items():
        frame = pd.read_csv(output_dir / name)
        assert tuple(frame.columns) == expected, name
        loaded = load_artifact(output_dir / name, manifest)
        assert tuple(loaded.columns) == expected, name


def test_stress_catalog_declares_its_schema_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog is discoverable from the manifest, like every other artifact."""
    output_dir = _run(tmp_path, monkeypatch)
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    declared = manifest["artifact_schemas"]["stress_scenario_catalog.json"]
    assert declared == {
        "schema_id": STRESS_CATALOG_SCHEMA_ID,
        "schema_version": STRESS_SCENARIO_SCHEMA_VERSION,
        "units": "decimal_simple_return_shocks",
    }
    catalog = json.loads(
        (output_dir / "stress_scenario_catalog.json").read_text(encoding="utf-8")
    )
    # The file describes itself with the same identity the manifest declares.
    assert catalog["schema_id"] == declared["schema_id"]
    assert catalog["schema_version"] == declared["schema_version"]


def test_load_artifact_still_refuses_the_non_tabular_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registering a JSON schema must not make it loadable as a table."""
    output_dir = _run(tmp_path, monkeypatch)
    with pytest.raises(ArtifactSchemaError, match="accepts CSV artifacts"):
        load_artifact(
            output_dir / "stress_scenario_catalog.json",
            output_dir / "run_manifest.json",
        )


def test_load_artifact_rejects_a_drifted_tail_analytics_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = _run(tmp_path, monkeypatch)
    target = output_dir / "portfolio_value_at_risk.csv"
    frame = pd.read_csv(target)
    frame.rename(columns={"value_at_risk": "var"}).to_csv(target, index=False)
    with pytest.raises(ArtifactSchemaError, match="do not match its declared schema"):
        load_artifact(target, output_dir / "run_manifest.json")

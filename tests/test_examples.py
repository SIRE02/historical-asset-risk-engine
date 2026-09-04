"""Checks for committed, runnable documentation examples."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from historical_asset_risk import cli
from historical_asset_risk.config import load_configuration

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_example_configuration_resolves_full_market_settings() -> None:
    config = load_configuration(_REPO_ROOT / "config.example.toml")

    assert config.provider == "yahoo"
    assert config.tickers == ("SPY", "QQQ", "TLT", "GLD")
    assert config.start_date == "2001-01-01"
    assert config.end_date == "2025-01-01"
    assert config.rolling_window == 63
    assert config.rolling_min_observations == 63
    assert config.portfolio_enabled is True
    assert config.csv_path is None
    assert config.instrument_registry_path == Path(
        "examples/data/instrument_registry.csv"
    )
    assert config.positions_path == Path("examples/data/positions_market.csv")
    assert config.cash_path == Path("examples/data/cash_market.csv")
    assert config.tail_risk_confidence_level == 0.99
    assert config.tail_risk_window == 252
    assert config.stress_catalog_path == Path("examples/stress_catalog.example.json")


@pytest.mark.parametrize(
    ("config_name", "expected_dir"),
    [
        ("config.long_short.toml", "outputs/example-long-short"),
        ("config.long_only.toml", "outputs/example-long-only"),
    ],
)
def test_offline_portfolio_examples_run_from_repo_root(
    config_name: str,
    expected_dir: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_configuration(_REPO_ROOT / "examples" / config_name)
    assert config.provider == "csv"
    assert str(config.output_dir) == str(Path(expected_dir))
    assert config.stress_catalog_path == Path("examples/stress_catalog.example.json")

    monkeypatch.chdir(_REPO_ROOT)
    monkeypatch.setattr(
        cli, "plot_rolling_volatility", lambda _d, p, _w: p.write_bytes(b"c")
    )
    monkeypatch.setattr(
        cli, "plot_correlation_heatmap", lambda _d, p: p.write_bytes(b"c")
    )
    config = replace(config, output_dir=tmp_path / "out")
    cli.run_analysis(config)

    names = {p.name for p in (tmp_path / "out").iterdir()}
    assert {
        "simple_returns.csv",
        "hypothetical_portfolio_pnl.csv",
        "portfolio_value_at_risk.csv",
        "stress_test_results.csv",
        "run_manifest.json",
    }.issubset(names)
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["portfolio_phase5"]["stress"]["scenario_count"] == 3

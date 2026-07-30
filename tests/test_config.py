"""Configuration loading and validation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from historical_asset_risk.config import AnalysisConfig, load_configuration


def test_json_configuration_is_normalized_and_overridden(tmp_path: Path) -> None:
    config_path = tmp_path / "analysis.json"
    config_path.write_text(
        json.dumps(
            {
                "provider": "yahoo",
                "tickers": [" spy ", "QQQ", "spy"],
                "start_date": "2023-01-01",
                "end_date": "2024-01-01",
                "rolling_window": 20,
                "rolling_min_observations": 4,
                "observations_per_year": 260,
                "quantiles": [0.9, 0.1, 0.1],
                "quantile_method": "nearest",
                "downside_target": -0.001,
            }
        ),
        encoding="utf-8",
    )
    config = load_configuration(config_path, {"rolling_window": 5})
    assert config.tickers == ("SPY", "QQQ")
    assert config.rolling_window == 5
    assert config.rolling_min_observations == 4
    assert config.observations_per_year == 260
    assert config.quantiles == (0.1, 0.9)
    assert config.quantile_method == "nearest"
    assert config.downside_target == pytest.approx(-0.001)


def test_csv_configuration_requires_an_existing_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CSV_PATH is required"):
        AnalysisConfig(provider="csv", tickers=("AAA", "BBB"))
    with pytest.raises(ValueError, match="readable file"):
        AnalysisConfig(
            provider="csv",
            csv_path=tmp_path / "missing.csv",
            tickers=("AAA", "BBB"),
        )


def test_unknown_options_and_invalid_core_values_fail_early(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.json"
    config_path.write_text('{"mystery": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_configuration(config_path)
    with pytest.raises(ValueError, match="earlier"):
        AnalysisConfig(
            tickers=("AAA", "BBB"),
            start_date="2025-01-01",
            end_date="2024-01-01",
        )
    with pytest.raises(ValueError, match="OBSERVATIONS_PER_YEAR"):
        AnalysisConfig(tickers=("AAA", "BBB"), observations_per_year=0)
    with pytest.raises(ValueError, match="must not exceed"):
        AnalysisConfig(
            tickers=("AAA", "BBB"),
            rolling_window=5,
            rolling_min_observations=6,
        )
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        AnalysisConfig(tickers=("AAA", "BBB"), quantiles=(0.0, 0.95))
    with pytest.raises(ValueError, match="QUANTILE_METHOD"):
        AnalysisConfig(tickers=("AAA", "BBB"), quantile_method="unknown")
    with pytest.raises(ValueError, match="DOWNSIDE_TARGET"):
        AnalysisConfig(tickers=("AAA", "BBB"), downside_target=float("nan"))


def test_rolling_minimum_defaults_to_full_window() -> None:
    config = AnalysisConfig(tickers=("AAA", "BBB"), rolling_window=7)
    assert config.rolling_min_observations == 7
    assert config.observations_per_year == 252
    assert config.quantiles == (0.05, 0.25, 0.75, 0.95)
    assert config.quantile_method == "linear"
    assert config.downside_target == 0.0


def test_portfolio_paths_are_all_or_nothing_and_serialized(tmp_path: Path) -> None:
    registry = tmp_path / "registry.csv"
    positions = tmp_path / "positions.csv"
    cash = tmp_path / "cash.csv"
    for path in (registry, positions, cash):
        path.write_text("fixture\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires all three paths"):
        AnalysisConfig(
            tickers=("AAA", "BBB"),
            instrument_registry_path=registry,
        )

    config = AnalysisConfig(
        tickers=("AAA", "BBB"),
        instrument_registry_path=registry,
        positions_path=positions,
        cash_path=cash,
    )
    assert config.portfolio_enabled is True
    assert config.to_dict()["instrument_registry_path"] == str(registry)

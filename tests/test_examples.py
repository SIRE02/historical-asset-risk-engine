"""Checks for committed, runnable documentation examples."""

from pathlib import Path

from historical_asset_risk.config import load_configuration


def test_committed_example_configuration_resolves_full_market_settings() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    config = load_configuration(repository_root / "config.example.toml")

    assert config.provider == "yahoo"
    assert config.tickers == ("SPY", "QQQ", "TLT", "GLD")
    assert config.start_date == "2021-01-01"
    assert config.end_date == "2025-01-01"
    assert config.rolling_window == 21
    assert config.rolling_min_observations == 21
    assert config.portfolio_enabled is True
    assert config.csv_path is None
    assert config.instrument_registry_path == Path(
        "examples/data/instrument_registry.csv"
    )
    assert config.positions_path == Path("examples/data/positions.csv")
    assert config.cash_path == Path("examples/data/cash.csv")

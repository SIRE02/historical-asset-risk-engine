"""Offline integration tests for the complete configured analysis workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from historical_asset_risk import __version__, cli
from historical_asset_risk.artifacts import load_artifact
from historical_asset_risk.config import AnalysisConfig
from historical_asset_risk.contracts import ArtifactSchemaError
from historical_asset_risk.data_loader import MarketDataError
from historical_asset_risk.providers import ProviderPayload

EXPECTED_CSV_ARTIFACTS = {
    "adjusted_prices.csv",
    "simple_returns.csv",
    "log_returns.csv",
    "return_summary.csv",
    "volatility_summary.csv",
    "rolling_volatility.csv",
    "covariance_matrix.csv",
    "correlation_matrix.csv",
    "rolling_covariance.csv",
    "rolling_correlation.csv",
    "rolling_volatility.png",
    "correlation_heatmap.png",
    "data_quality_report.json",
    "run_manifest.json",
}


@pytest.fixture
def prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=8, freq="B")
    return pd.DataFrame(
        {
            "SPY": [100, 102, 101, 104, 106, 105, 108, 110],
            "QQQ": [50, 49, 51, 52, 51, 53, 54, 55],
            "TLT": [80, 81, 82, 81, 83, 84, 86, 85],
            "GLD": [120, 121, 119, 122, 123, 125, 124, 127],
        },
        index=index,
    )


def _write_canonical_csv(prices: pd.DataFrame, path: Path) -> None:
    frame = prices.copy()
    frame.index.name = "date"
    frame.reset_index().melt(
        id_vars="date", var_name="ticker", value_name="adjusted_close"
    ).to_csv(path, index=False)


def test_complete_csv_analysis_writes_reports_and_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame
) -> None:
    csv_path = tmp_path / "prices.csv"
    output_dir = tmp_path / "outputs"
    _write_canonical_csv(prices, csv_path)
    config = AnalysisConfig(
        provider="csv",
        csv_path=csv_path,
        output_dir=output_dir,
        tickers=tuple(prices.columns),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=3,
    )
    monkeypatch.setattr(
        cli,
        "plot_rolling_volatility",
        lambda _data, path, _window: path.write_bytes(b"test chart"),
    )
    monkeypatch.setattr(
        cli,
        "plot_correlation_heatmap",
        lambda _data, path: path.write_bytes(b"test chart"),
    )

    cli.run_analysis(config)

    assert {path.name for path in output_dir.iterdir()} == EXPECTED_CSV_ARTIFACTS
    correlations = pd.read_csv(output_dir / "correlation_matrix.csv", index_col=0)
    assert list(correlations.columns) == list(prices.columns)
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["project"] == "historical-asset-risk-engine"
    assert manifest["project_version"] == __version__
    assert manifest["configuration"] == config.to_dict()
    assert manifest["data_source"]["provider"] == "csv"
    assert manifest["data_source"]["source"] == str(csv_path.resolve())
    assert manifest["data_source"]["observation_count"] == len(prices)
    assert (
        manifest["estimation_conventions"]["sample_estimators"]["covariance_ddof"] == 1
    )
    assert manifest["estimation_conventions"]["return_summary"] == {
        "minimum_non_missing_observations_per_instrument": 4,
        "reason": (
            "the combined summary includes bias-corrected skewness and excess kurtosis"
        ),
    }
    assert (
        manifest["estimation_conventions"]["rolling"]["future_observations_used"]
        is False
    )
    assert manifest["estimation_conventions"]["missing_data"] == {
        "alignment": "complete_case_adjusted_prices",
        "forward_fill": False,
        "gap_spanning_return_policy": (
            "fail_when_retained_complete_case_dates_are_nonconsecutive_within_"
            "the_provider_observation_union"
        ),
    }
    assert manifest["data_source"]["canonical_record_stage"] == (
        "normalized_requested_in_range_pre_complete_case_alignment"
    )
    distribution = manifest["estimation_conventions"]["return_distribution"]
    assert distribution["quantiles"] == [0.05, 0.25, 0.75, 0.95]
    assert distribution["quantile_method"] == "linear"
    assert distribution["downside_deviation"] == {
        "target": 0.0,
        "target_return_type": "daily_log_return",
        "denominator": "all_non_missing_observations",
        "formula": "sqrt(sum(min(return - target, 0)^2) / n)",
        "annualized": False,
    }
    summary = pd.read_csv(output_dir / "return_summary.csv", index_col=0)
    assert {
        "median_return",
        "quantile_0.05",
        "quantile_0.25",
        "quantile_0.75",
        "quantile_0.95",
        "sample_skewness",
        "sample_excess_kurtosis",
        "downside_deviation",
    }.issubset(summary.columns)
    assert "data_quality_report.json" in manifest["generated_artifacts"]


def test_yahoo_analysis_persists_normalized_records_before_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame
) -> None:
    records = (
        prices.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="adjusted_close")
    )
    missing_date = prices.index[-1]
    records.loc[
        (records["date"] == missing_date) & (records["ticker"] == "QQQ"),
        "adjusted_close",
    ] = None

    class SavedYahooProvider:
        name = "yahoo"

        def acquire(self, _config: AnalysisConfig) -> ProviderPayload:
            return ProviderPayload(
                records,
                provider="yahoo",
                source="saved provider response",
                acquired_at="2024-02-01T00:00:00+00:00",
            )

        def normalize(
            self,
            payload: ProviderPayload,
            _requested_tickers: tuple[str, ...],
        ) -> pd.DataFrame:
            return payload.data.copy()

    output_dir = tmp_path / "outputs"
    config = AnalysisConfig(
        provider="yahoo",
        output_dir=output_dir,
        tickers=tuple(prices.columns),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=3,
    )
    monkeypatch.setattr(
        cli,
        "plot_rolling_volatility",
        lambda _data, path, _window: path.write_bytes(b"test chart"),
    )
    monkeypatch.setattr(
        cli,
        "plot_correlation_heatmap",
        lambda _data, path: path.write_bytes(b"test chart"),
    )

    cli.run_analysis(config, SavedYahooProvider())

    acquisition = pd.read_csv(
        output_dir / "acquired_adjusted_prices.csv", parse_dates=["date"]
    )
    retained = acquisition.loc[
        (acquisition["date"] == missing_date) & (acquisition["ticker"] == "QQQ"),
        "adjusted_close",
    ]
    assert len(retained) == 1
    assert retained.isna().all()
    aligned_prices = pd.read_csv(output_dir / "adjusted_prices.csv", index_col="date")
    assert len(aligned_prices) == len(prices) - 1


def _csv_config(prices: pd.DataFrame, tmp_path: Path, name: str) -> AnalysisConfig:
    csv_path = tmp_path / f"{name}.csv"
    _write_canonical_csv(prices, csv_path)
    return AnalysisConfig(
        provider="csv",
        csv_path=csv_path,
        output_dir=tmp_path / "outputs",
        tickers=tuple(prices.columns),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=3,
    )


def test_failed_rerun_leaves_the_previous_run_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame
) -> None:
    monkeypatch.setattr(
        cli,
        "plot_rolling_volatility",
        lambda _data, path, _window: path.write_bytes(b"test chart"),
    )
    monkeypatch.setattr(
        cli,
        "plot_correlation_heatmap",
        lambda _data, path: path.write_bytes(b"test chart"),
    )
    first = _csv_config(prices, tmp_path, "first")
    cli.run_analysis(first)
    output_dir = first.output_dir
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    def failing_heatmap(_data: object, _path: Path) -> None:
        raise RuntimeError("chart backend failed")

    # The rerun writes its price and return CSVs before the chart fails.
    monkeypatch.setattr(cli, "plot_correlation_heatmap", failing_heatmap)
    with pytest.raises(RuntimeError, match="chart backend failed"):
        cli.run_analysis(_csv_config(prices * 2, tmp_path, "second"))

    after = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    assert after == before


class _InMemoryYahooProvider:
    """Supplies prices as provider floats, as Yahoo does, with no CSV parse."""

    name = "yahoo"

    def __init__(self, prices: pd.DataFrame) -> None:
        self.records = (
            prices.rename_axis("date")
            .reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="adjusted_close")
        )

    def acquire(self, _config: AnalysisConfig) -> ProviderPayload:
        return ProviderPayload(
            self.records,
            provider="yahoo",
            source="in-memory provider floats",
            acquired_at="2024-02-01T00:00:00+00:00",
        )

    def normalize(
        self, payload: ProviderPayload, _requested_tickers: tuple[str, ...]
    ) -> pd.DataFrame:
        return payload.data.copy()


def test_untouched_unrounded_prices_pass_the_price_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame
) -> None:
    monkeypatch.setattr(
        cli,
        "plot_rolling_volatility",
        lambda _data, path, _window: path.write_bytes(b"test chart"),
    )
    monkeypatch.setattr(
        cli,
        "plot_correlation_heatmap",
        lambda _data, path: path.write_bytes(b"test chart"),
    )
    rng = np.random.default_rng(7)
    unrounded = prices * np.exp(rng.normal(0.0, 0.01, prices.shape))
    output_dir = tmp_path / "outputs"
    config = AnalysisConfig(
        provider="yahoo",
        output_dir=output_dir,
        tickers=tuple(prices.columns),
        start_date="2024-01-01",
        end_date="2024-02-01",
        rolling_window=3,
    )
    cli.run_analysis(config, _InMemoryYahooProvider(unrounded))
    prices_path = output_dir / "adjusted_prices.csv"

    # The fixture must exercise the hazard: pandas' default parser lands one
    # ulp away from some written provider floats, which would change the hash.
    default = pd.read_csv(prices_path, index_col="date")
    exact = pd.read_csv(prices_path, index_col="date", float_precision="round_trip")
    assert (default != exact).any(axis=None)

    loaded = load_artifact(prices_path, output_dir / "run_manifest.json")
    assert loaded.to_numpy().tolist() == unrounded.to_numpy().tolist()


def test_interrupted_publish_leaves_no_manifest_to_validate_against(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prices: pd.DataFrame
) -> None:
    monkeypatch.setattr(
        cli,
        "plot_rolling_volatility",
        lambda _data, path, _window: path.write_bytes(b"test chart"),
    )
    monkeypatch.setattr(
        cli,
        "plot_correlation_heatmap",
        lambda _data, path: path.write_bytes(b"test chart"),
    )
    first = _csv_config(prices, tmp_path, "first")
    cli.run_analysis(first)
    output_dir = first.output_dir

    real_replace = Path.replace

    def failing_replace(source: Path, target: Path) -> Path:
        if source.name == "log_returns.csv":
            raise OSError("file is locked")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError, match="file is locked"):
        cli.run_analysis(_csv_config(prices * 2, tmp_path, "second"))

    # New prices were published but the old manifest was not left beside them.
    assert not (output_dir / "run_manifest.json").exists()
    with pytest.raises(ArtifactSchemaError, match="Could not read artifact manifest"):
        load_artifact(
            output_dir / "adjusted_prices.csv", output_dir / "run_manifest.json"
        )
    assert not list(output_dir.glob(".staging-*"))


def test_cli_arguments_override_configuration_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "analysis.toml"
    config_path.write_text(
        """[analysis]
provider = "yahoo"
tickers = ["SPY", "QQQ"]
start_date = "2023-01-01"
end_date = "2024-01-01"
rolling_window = 10
""",
        encoding="utf-8",
    )
    captured: list[AnalysisConfig] = []
    monkeypatch.setattr(cli, "run_analysis", captured.append)

    cli.main(
        [
            "--config",
            str(config_path),
            "--tickers",
            "GLD,TLT",
            "--rolling-window",
            "5",
            "--rolling-min-observations",
            "3",
            "--observations-per-year",
            "260",
            "--quantiles",
            "0.1,0.5",
            "0.9",
            "--quantile-method",
            "nearest",
            "--downside-target",
            "-0.001",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert captured[0].tickers == ("GLD", "TLT")
    assert captured[0].rolling_window == 5
    assert captured[0].rolling_min_observations == 3
    assert captured[0].observations_per_year == 260
    assert captured[0].quantiles == (0.1, 0.5, 0.9)
    assert captured[0].quantile_method == "nearest"
    assert captured[0].downside_target == pytest.approx(-0.001)
    assert captured[0].start_date == "2023-01-01"


def test_cli_converts_expected_failures_to_exit_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_config: AnalysisConfig) -> None:
        raise MarketDataError("provider unavailable")

    monkeypatch.setattr(cli, "run_analysis", fail)

    with pytest.raises(SystemExit, match="Analysis failed: provider unavailable"):
        cli.main([])

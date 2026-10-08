"""Command-line entry point for the Historical Asset Risk Engine."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from historical_asset_risk.artifacts import (
    ARTIFACT_SCHEMAS,
    portfolio_artifact_frames,
    read_cash,
    read_instrument_registry,
    read_positions,
)
from historical_asset_risk.config import AnalysisConfig, load_configuration
from historical_asset_risk.contracts import (
    Cash,
    Instrument,
    PortfolioValuation,
    Position,
)
from historical_asset_risk.correlation import (
    correlation_matrix,
    covariance_matrix,
    extreme_correlation_pairs,
    rolling_correlation,
    rolling_covariance,
)
from historical_asset_risk.data_loader import (
    MarketDataError,
    load_market_data,
    persist_acquisition,
    persist_quality_report,
)
from historical_asset_risk.market_calendar import resolve_market_calendar
from historical_asset_risk.pnl import ProxyExposureSnapshot, compute_data_snapshot_id
from historical_asset_risk.portfolio import (
    calculate_currency_valuation,
    validate_portfolio_snapshot,
    value_portfolio,
)
from historical_asset_risk.portfolio_analytics import (
    PortfolioAnalyticsResult,
    compute_portfolio_analytics,
)
from historical_asset_risk.providers import MarketDataProvider, provider_for
from historical_asset_risk.reporting import build_run_manifest, persist_run_manifest
from historical_asset_risk.returns import (
    calculate_log_returns,
    calculate_simple_returns,
    summarize_returns,
)
from historical_asset_risk.risk_metrics import rolling_volatility, volatility_summary
from historical_asset_risk.stress import load_stress_catalog
from historical_asset_risk.tail_analytics import (
    TailAnalyticsResult,
    compute_tail_analytics,
)
from historical_asset_risk.visualizations import (
    plot_correlation_heatmap,
    plot_rolling_volatility,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="historical-asset-risk",
        description=(
            "Analyze validated adjusted daily prices from Yahoo or a local CSV."
        ),
    )
    parser.add_argument("--config", type=Path, help="TOML or JSON configuration file")
    parser.add_argument("--provider", choices=("yahoo", "csv"))
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Ticker symbols separated by spaces (commas are also accepted)",
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--rolling-window", type=int)
    parser.add_argument("--rolling-min-observations", type=int)
    parser.add_argument("--observations-per-year", type=int)
    parser.add_argument(
        "--quantiles",
        nargs="+",
        help="Empirical probabilities separated by spaces (commas are also accepted)",
    )
    parser.add_argument(
        "--quantile-method",
        choices=("linear", "lower", "higher", "midpoint", "nearest"),
    )
    parser.add_argument(
        "--downside-target",
        type=float,
        help="Daily log-return target used by downside deviation",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--instrument-registry-path", type=Path)
    parser.add_argument("--positions-path", type=Path)
    parser.add_argument("--cash-path", type=Path)
    parser.add_argument("--positions-history-path", type=Path)
    parser.add_argument("--cash-history-path", type=Path)
    parser.add_argument("--tail-risk-confidence-level", type=float)
    parser.add_argument("--tail-risk-window", type=int)
    parser.add_argument("--stress-catalog-path", type=Path)
    return parser


def _configuration_from_args(arguments: argparse.Namespace) -> AnalysisConfig:
    tickers: list[str] | None = None
    if arguments.tickers is not None:
        tickers = [
            ticker
            for item in arguments.tickers
            for ticker in item.split(",")
            if ticker.strip()
        ]
    quantiles: list[str] | None = None
    if arguments.quantiles is not None:
        quantiles = [
            probability
            for item in arguments.quantiles
            for probability in item.split(",")
            if probability.strip()
        ]
    overrides: dict[str, Any] = {
        "provider": arguments.provider,
        "tickers": tickers,
        "start_date": arguments.start_date,
        "end_date": arguments.end_date,
        "rolling_window": arguments.rolling_window,
        "rolling_min_observations": arguments.rolling_min_observations,
        "observations_per_year": arguments.observations_per_year,
        "quantiles": quantiles,
        "quantile_method": arguments.quantile_method,
        "downside_target": arguments.downside_target,
        "output_dir": arguments.output_dir,
        "csv_path": arguments.csv_path,
        "instrument_registry_path": arguments.instrument_registry_path,
        "positions_path": arguments.positions_path,
        "cash_path": arguments.cash_path,
        "positions_history_path": arguments.positions_history_path,
        "cash_history_path": arguments.cash_history_path,
        "tail_risk_confidence_level": arguments.tail_risk_confidence_level,
        "tail_risk_window": arguments.tail_risk_window,
        "stress_catalog_path": arguments.stress_catalog_path,
    }
    return load_configuration(arguments.config, overrides)


def _load_and_value_portfolio(
    config: AnalysisConfig,
) -> tuple[PortfolioValuation, tuple[Instrument, ...]] | None:
    if not config.portfolio_enabled:
        return None
    assert config.instrument_registry_path is not None
    assert config.positions_path is not None
    assert config.cash_path is not None
    instruments = read_instrument_registry(config.instrument_registry_path)
    positions = read_positions(config.positions_path)
    cash = read_cash(config.cash_path)
    snapshot = validate_portfolio_snapshot(
        instruments,
        positions,
        cash,
        market_data_instruments=config.tickers,
    )
    return value_portfolio(snapshot), instruments


def _load_exposure_history(
    config: AnalysisConfig, instruments: tuple[Instrument, ...]
) -> list[ProxyExposureSnapshot]:
    if config.positions_history_path is None:
        return []
    assert config.cash_history_path is not None
    positions = read_positions(config.positions_history_path)
    cash_records = read_cash(config.cash_history_path)
    cash_by_snapshot: dict[str, list[Cash]] = defaultdict(list)
    for record in cash_records:
        cash_by_snapshot[record.portfolio_snapshot_id].append(record)
    positions_by_snapshot: dict[str, list[Position]] = defaultdict(list)
    for position in positions:
        positions_by_snapshot[position.portfolio_snapshot_id].append(position)

    history: list[ProxyExposureSnapshot] = []
    for snapshot_id, snapshot_positions in positions_by_snapshot.items():
        snapshot_cash = cash_by_snapshot.get(snapshot_id, [])
        if len(snapshot_cash) != 1:
            raise ValueError(
                f"Exposure-history snapshot {snapshot_id!r} needs exactly one cash "
                f"row; found {len(snapshot_cash)}."
            )
        validated = validate_portfolio_snapshot(
            instruments,
            snapshot_positions,
            (snapshot_cash[0],),
            market_data_instruments=config.tickers,
        )
        currency = calculate_currency_valuation(validated)
        history.append(
            ProxyExposureSnapshot(
                portfolio_snapshot_id=snapshot_cash[0].portfolio_snapshot_id,
                portfolio_id=snapshot_cash[0].portfolio_id,
                exposure_snapshot_id=currency.exposure_snapshot_id,
                as_of_date=snapshot_cash[0].as_of_date,
                market_calendar_id=snapshot_cash[0].market_calendar_id,
                currency_exposures={
                    item.position.instrument_id: item.position_value
                    for item in currency.positions
                },
            )
        )
    extra_cash = sorted(set(cash_by_snapshot) - set(positions_by_snapshot))
    if extra_cash:
        raise ValueError(
            "Exposure-history cash rows without positions: " + ", ".join(extra_cash)
        )
    history.sort(key=lambda snapshot: snapshot.as_of_date)
    return history


def _prevent_snapshot_overwrite(
    output_dir: Path, portfolio: PortfolioValuation | None
) -> None:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot verify the existing portfolio run manifest {manifest_path}: {exc}"
        ) from exc
    existing_id = existing.get("portfolio", {}).get("exposure_snapshot_id")
    new_id = portfolio.exposure_snapshot_id if portfolio is not None else None
    # A run without a book counts as different: it would remove the earlier
    # run's portfolio artifacts as stale.
    if existing_id is not None and existing_id != new_id:
        raise ValueError(
            "Refusing to overwrite a materially different portfolio run in "
            f"{output_dir}; choose a new OUTPUT_DIR."
        )


def _remove_stale_artifacts(output_dir: Path, artifacts: Sequence[str]) -> None:
    """Delete engine-owned files an earlier run left that this run does not write.

    Only names in the artifact schema registry are touched, so anything else a
    person keeps in the directory is left alone. Without this, a rerun that
    drops an optional layer (stress, proxy history) leaves the previous run's
    files beside a manifest that no longer lists them.
    """
    for name in sorted(set(ARTIFACT_SCHEMAS) - set(artifacts)):
        stale = output_dir / name
        if stale.is_file():
            stale.unlink()


def _publish_run(staging_dir: Path, output_dir: Path, artifacts: Sequence[str]) -> None:
    """Move a completed run from ``staging_dir`` into ``output_dir``, manifest last.

    The old manifest goes first: if a move fails part-way, the directory has no
    manifest, so ``load_artifact`` refuses its files instead of validating new
    files against the previous run's declarations.
    """
    manifest_name = "run_manifest.json"
    (output_dir / manifest_name).unlink(missing_ok=True)
    _remove_stale_artifacts(output_dir, artifacts)
    for name in artifacts:
        if name != manifest_name:
            (staging_dir / name).replace(output_dir / name)
    (staging_dir / manifest_name).replace(output_dir / manifest_name)


def run_analysis(
    config: AnalysisConfig | None = None,
    provider: MarketDataProvider | None = None,
) -> None:
    """Run the complete analysis from one validated configuration."""
    config = config or load_configuration()
    loaded_portfolio = _load_and_value_portfolio(config)
    portfolio = loaded_portfolio[0] if loaded_portfolio is not None else None
    registry_instruments = loaded_portfolio[1] if loaded_portfolio is not None else ()
    exposure_history = (
        _load_exposure_history(config, registry_instruments)
        if loaded_portfolio is not None
        else []
    )
    selected_provider = provider or provider_for(config)
    print(
        f"Loading adjusted prices from {config.provider} for "
        f"{', '.join(config.tickers)}..."
    )
    market_data = load_market_data(config, selected_provider)
    prices = market_data.prices
    if portfolio is not None:
        resolve_market_calendar(
            portfolio.snapshot.cash.market_calendar_id
        ).require_consecutive_sessions(day.date() for day in prices.index)
    simple_returns = calculate_simple_returns(prices)
    log_returns = calculate_log_returns(prices)
    return_summary = summarize_returns(
        log_returns,
        config.quantiles,
        config.quantile_method,
        config.downside_target,
    )
    vol_summary = volatility_summary(log_returns, config.observations_per_year)
    rolling = rolling_volatility(
        log_returns,
        config.rolling_window,
        config.observations_per_year,
        config.rolling_min_observations,
    )
    covariances = covariance_matrix(log_returns)
    correlations = correlation_matrix(log_returns)
    rolling_covariances = rolling_covariance(
        log_returns, config.rolling_window, config.rolling_min_observations
    )
    rolling_correlations = rolling_correlation(
        log_returns, config.rolling_window, config.rolling_min_observations
    )
    highest, lowest = extreme_correlation_pairs(correlations)

    portfolio_analytics_result: PortfolioAnalyticsResult | None = None
    tail_analytics_result: TailAnalyticsResult | None = None
    if portfolio is not None:
        data_source = {
            "provider": market_data.payload.provider,
            "source": market_data.payload.source,
            "actual_start_date": market_data.quality_report["first_common_date"],
            "actual_end_date": market_data.quality_report["last_common_date"],
            "observation_count": market_data.quality_report[
                "common_date_count_after_alignment"
            ],
            "instruments": list(prices.columns),
            "price_content_hash": market_data.price_content_hash,
        }
        portfolio_analytics_result = compute_portfolio_analytics(
            portfolio,
            simple_returns,
            prices.index,
            data_source=data_source,
            observations_per_year=config.observations_per_year,
            registry_instruments=registry_instruments,
            exposure_history=exposure_history,
        )

        stress_scenarios = (
            load_stress_catalog(config.stress_catalog_path)
            if config.stress_catalog_path is not None
            else ()
        )
        tail_analytics_result = compute_tail_analytics(
            portfolio,
            simple_returns,
            prices.index,
            data_snapshot_id=compute_data_snapshot_id(data_source),
            confidence_level=config.tail_risk_confidence_level,
            window=config.tail_risk_window,
            registry_instruments=registry_instruments,
            exposure_history=exposure_history,
            stress_scenarios=stress_scenarios,
        )

    output_dir = config.output_dir
    _prevent_snapshot_overwrite(output_dir, portfolio)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
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
    ]
    if portfolio is not None:
        artifacts.extend(
            [
                "validated_instruments.csv",
                "validated_positions.csv",
                "portfolio_valuation.csv",
                "portfolio_exposure_summary.csv",
            ]
        )
    if portfolio_analytics_result is not None:
        artifacts.extend(sorted(portfolio_analytics_result.frames))
    if tail_analytics_result is not None:
        artifacts.extend(sorted(tail_analytics_result.frames))
        if tail_analytics_result.catalog_json is not None:
            artifacts.append("stress_scenario_catalog.json")
    if market_data.payload.provider == "yahoo":
        artifacts.append("acquired_adjusted_prices.csv")
    # Every file is written to a staging folder first and published only once
    # the whole run has succeeded, so a failure part-way leaves the previous
    # run's files and manifest untouched rather than mixed with this run's.
    with tempfile.TemporaryDirectory(
        prefix=".staging-", dir=output_dir, ignore_cleanup_errors=True
    ) as staging:
        staging_dir = Path(staging)
        prices.to_csv(staging_dir / "adjusted_prices.csv", index_label="date")
        simple_returns.to_csv(staging_dir / "simple_returns.csv", index_label="date")
        log_returns.to_csv(staging_dir / "log_returns.csv", index_label="date")
        return_summary.to_csv(staging_dir / "return_summary.csv")
        vol_summary.to_csv(staging_dir / "volatility_summary.csv")
        rolling.to_csv(staging_dir / "rolling_volatility.csv", index_label="date")
        covariances.to_csv(staging_dir / "covariance_matrix.csv")
        correlations.to_csv(staging_dir / "correlation_matrix.csv")
        rolling_covariances.to_csv(
            staging_dir / "rolling_covariance.csv", index_label=["date", "ticker"]
        )
        rolling_correlations.to_csv(
            staging_dir / "rolling_correlation.csv", index_label=["date", "ticker"]
        )
        plot_rolling_volatility(
            rolling, staging_dir / "rolling_volatility.png", config.rolling_window
        )
        plot_correlation_heatmap(correlations, staging_dir / "correlation_heatmap.png")
        quality_report = dict(market_data.quality_report)
        if portfolio is not None:
            for name, frame in portfolio_artifact_frames(portfolio).items():
                frame.to_csv(staging_dir / name, index=False)
            reconciliation = portfolio.snapshot.reconciliation
            quality_report["portfolio"] = {
                "portfolio_snapshot_id": portfolio.snapshot.cash.portfolio_snapshot_id,
                "exposure_snapshot_id": portfolio.exposure_snapshot_id,
                "position_count": len(portfolio.positions),
                "held_instrument_count": len(reconciliation.held_instrument_ids),
                "missing_market_data_instruments": list(
                    reconciliation.missing_market_data_instruments
                ),
                "extra_market_data_instruments_ignored": list(
                    reconciliation.extra_market_data_instruments
                ),
                "validation_exceptions": [],
            }
            if portfolio_analytics_result is not None:
                for name, frame in portfolio_analytics_result.frames.items():
                    frame.to_csv(staging_dir / name, index=False)
                quality_report["portfolio"]["analytics"] = (
                    portfolio_analytics_result.quality_section
                )
            if tail_analytics_result is not None:
                for name, frame in tail_analytics_result.frames.items():
                    frame.to_csv(staging_dir / name, index=False)
                if tail_analytics_result.catalog_json is not None:
                    (staging_dir / "stress_scenario_catalog.json").write_text(
                        json.dumps(tail_analytics_result.catalog_json, indent=2) + "\n",
                        encoding="utf-8",
                    )
                quality_report["portfolio"]["tail"] = (
                    tail_analytics_result.quality_section
                )
        persist_quality_report(quality_report, staging_dir / "data_quality_report.json")
        if market_data.payload.provider == "yahoo":
            persist_acquisition(
                market_data.canonical_records,
                staging_dir / "acquired_adjusted_prices.csv",
            )
        manifest = build_run_manifest(config, market_data, artifacts, portfolio)
        if portfolio_analytics_result is not None:
            manifest["portfolio_analytics"] = (
                portfolio_analytics_result.manifest_section
            )
        if tail_analytics_result is not None:
            manifest["tail_analytics"] = tail_analytics_result.manifest_section
        persist_run_manifest(manifest, staging_dir / "run_manifest.json")
        _publish_run(staging_dir, output_dir, artifacts)

    print("\nDaily and annualized volatility:")
    print(vol_summary.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nPearson correlation matrix:")
    print(correlations.to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"\nHighest pair: {highest[0]} / {highest[1]} ({highest[2]:.3f})")
    print(f"Lowest pair:  {lowest[0]} / {lowest[1]} ({lowest[2]:.3f})")
    if portfolio is not None:
        print(
            "\nPortfolio value and exposure: "
            f"NAV {portfolio.portfolio_value:.2f} "
            f"{portfolio.snapshot.cash.base_currency}, "
            f"gross {portfolio.gross_exposure:.2f}, "
            f"net {portfolio.net_instrument_exposure:.2f}"
        )
    if portfolio_analytics_result is not None:
        for line in portfolio_analytics_result.summary_lines:
            print(line)
    if tail_analytics_result is not None:
        for line in tail_analytics_result.summary_lines:
            print(line)
    print(
        "\nSaved tables, charts, quality report, and manifest to: "
        f"{output_dir.resolve()}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and convert expected failures into a concise exit message."""
    try:
        arguments = _parser().parse_args(argv)
        run_analysis(_configuration_from_args(arguments))
    except (MarketDataError, OSError, ValueError) as exc:
        raise SystemExit(f"Analysis failed: {exc}") from exc


__all__ = ["main", "run_analysis"]


if __name__ == "__main__":
    main()

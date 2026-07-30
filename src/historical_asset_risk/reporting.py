"""Run-lineage manifest generation."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from historical_asset_risk import __version__
from historical_asset_risk.artifacts import artifact_schema_inventory
from historical_asset_risk.config import AnalysisConfig
from historical_asset_risk.contracts import (
    CASH_SCHEMA_ID,
    CASH_SCHEMA_VERSION,
    INSTRUMENT_REGISTRY_SCHEMA_ID,
    INSTRUMENT_REGISTRY_SCHEMA_VERSION,
    POSITIONS_SCHEMA_ID,
    POSITIONS_SCHEMA_VERSION,
    PortfolioValuation,
)
from historical_asset_risk.data_loader import MarketDataResult
from historical_asset_risk.estimation import estimation_conventions


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for dependency in ("numpy", "pandas", "matplotlib", "seaborn", "yfinance"):
        try:
            result[dependency] = version(dependency)
        except PackageNotFoundError:
            result[dependency] = "not-installed"
    return result


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build_run_manifest(
    config: AnalysisConfig,
    market_data: MarketDataResult,
    artifacts: list[str],
    portfolio: PortfolioValuation | None = None,
) -> dict[str, Any]:
    """Build the reproducibility and source-lineage record for a completed run."""
    quality = market_data.quality_report
    rolling_minimum = config.rolling_min_observations
    # AnalysisConfig resolves this during validation.
    assert rolling_minimum is not None
    manifest: dict[str, Any] = {
        "project": "historical-asset-risk-engine",
        "project_version": __version__,
        "git_commit": _git_commit(),
        "execution_timestamp": datetime.now(UTC).isoformat(),
        "configuration": config.to_dict(),
        "estimation_conventions": estimation_conventions(
            config.observations_per_year,
            config.rolling_window,
            rolling_minimum,
            config.quantiles,
            config.quantile_method,
            config.downside_target,
        ),
        "data_source": {
            "provider": market_data.payload.provider,
            "source": market_data.payload.source,
            "acquired_at": market_data.payload.acquired_at,
            "provider_metadata": market_data.payload.metadata,
            "actual_start_date": quality["first_common_date"],
            "actual_end_date": quality["last_common_date"],
            "observation_count": quality["common_date_count_after_alignment"],
            "instruments": list(market_data.prices.columns),
            "canonical_record_stage": (
                "normalized_requested_in_range_pre_complete_case_alignment"
            ),
        },
        "dependency_versions": _dependency_versions(),
        "generated_artifacts": sorted(artifacts),
        "artifact_schemas": artifact_schema_inventory(artifacts),
        "consumer_compatibility": {
            "package_version": __version__,
            "artifact_schemas": artifact_schema_inventory(artifacts),
            "fixture_identity": "phase3-downstream-consumer-v1",
            "contract_status": "experimental",
        },
    }
    if portfolio is not None:
        snapshot = portfolio.snapshot
        cash = snapshot.cash
        assert config.instrument_registry_path is not None
        assert config.positions_path is not None
        assert config.cash_path is not None
        input_paths = {
            "instrument_registry": config.instrument_registry_path,
            "positions": config.positions_path,
            "cash": config.cash_path,
        }
        manifest["portfolio"] = {
            "portfolio_snapshot_id": cash.portfolio_snapshot_id,
            "exposure_snapshot_id": portfolio.exposure_snapshot_id,
            "portfolio_id": cash.portfolio_id,
            "as_of_date": cash.as_of_date.isoformat(),
            "as_of_timestamp": cash.as_of_timestamp.isoformat(),
            "base_currency": cash.base_currency,
            "calendar": {
                "calendar_id": cash.market_calendar_id,
                "timezone": cash.market_timezone,
                "source": snapshot.calendar_source,
                "version": snapshot.calendar_version,
            },
            "input_sources": {
                name: {
                    "path": str(path.resolve()),
                    "file_modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime, UTC
                    ).isoformat(),
                }
                for name, path in input_paths.items()
            },
            "input_schemas": {
                "instrument_registry": {
                    "schema_id": INSTRUMENT_REGISTRY_SCHEMA_ID,
                    "schema_version": INSTRUMENT_REGISTRY_SCHEMA_VERSION,
                },
                "positions": {
                    "schema_id": POSITIONS_SCHEMA_ID,
                    "schema_version": POSITIONS_SCHEMA_VERSION,
                },
                "cash": {
                    "schema_id": CASH_SCHEMA_ID,
                    "schema_version": CASH_SCHEMA_VERSION,
                },
            },
            "reconciliation": {
                "held_instrument_ids": list(
                    snapshot.reconciliation.held_instrument_ids
                ),
                "held_provider_tickers": list(
                    snapshot.reconciliation.held_provider_tickers
                ),
                "missing_market_data_instruments": list(
                    snapshot.reconciliation.missing_market_data_instruments
                ),
                "extra_market_data_instruments": list(
                    snapshot.reconciliation.extra_market_data_instruments
                ),
            },
        }
    return manifest


def persist_run_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write a deterministic, human-readable JSON manifest."""
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

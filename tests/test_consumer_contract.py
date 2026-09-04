"""Phase 6 frozen-consumer contract: the v0.1.1 handoff must not change.

MRFE, NMARE, and WFEA consume these files and loader rules. Phases 4-6 may add
artifacts and APIs; they may not change the schema id, version, units, or
required columns of the frozen set, and a returns-only run must still produce
it without a portfolio book.
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

_FROZEN_CONSUMER_FILES = {
    "adjusted_prices.csv": (
        "historical-asset-risk/adjusted-prices",
        "1.experimental",
        "provider_adjusted_price",
    ),
    "simple_returns.csv": (
        "historical-asset-risk/simple-returns",
        "1.experimental",
        "decimal_return_per_observation",
    ),
    "log_returns.csv": (
        "historical-asset-risk/log-returns",
        "1.experimental",
        "log_return_per_observation",
    ),
}
_FROZEN_JSON_FILES = {
    "data_quality_report.json": "historical-asset-risk/data-quality-report",
    "run_manifest.json": "historical-asset-risk/run-manifest",
}


def _returns_only_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
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
            tickers=("AAA", "BBB"),
            start_date="2024-01-01",
            end_date="2024-02-01",
            rolling_window=3,
        )
    )
    return output_dir


def test_frozen_consumer_schema_identities_are_unchanged() -> None:
    for name, (schema_id, version, units) in _FROZEN_CONSUMER_FILES.items():
        assert ARTIFACT_SCHEMAS[name] == (schema_id, version), name
        assert ARTIFACT_UNITS[name] == units, name
    for name, schema_id in _FROZEN_JSON_FILES.items():
        assert ARTIFACT_SCHEMAS[name][0] == schema_id, name


def test_every_registered_artifact_has_a_version_and_units() -> None:
    for name, (schema_id, version) in ARTIFACT_SCHEMAS.items():
        assert schema_id and version, name
        assert name in ARTIFACT_UNITS, name


def test_returns_only_run_writes_the_frozen_set_without_a_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = _returns_only_run(tmp_path, monkeypatch)
    names = {p.name for p in output_dir.iterdir()}

    assert _FROZEN_CONSUMER_FILES.keys() <= names
    assert _FROZEN_JSON_FILES.keys() <= names
    assert not any(
        n.startswith(
            (
                "portfolio_",
                "hypothetical_",
                "validated_",
                "proxy_",
                "risk_realizations",
                "stress_",
                "tail_risk",
                "trailing_",
            )
        )
        for n in names
    )

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    for key in (
        "data_source",
        "estimation_conventions",
        "generated_artifacts",
        "artifact_schemas",
        "consumer_compatibility",
    ):
        assert key in manifest, key
    assert "portfolio" not in manifest
    assert "portfolio_phase4" not in manifest
    assert "portfolio_phase5" not in manifest

    quality = json.loads((output_dir / "data_quality_report.json").read_text())
    assert "portfolio" not in quality

    for name in _FROZEN_CONSUMER_FILES:
        loaded = load_artifact(output_dir / name, output_dir / "run_manifest.json")
        assert list(loaded.columns) == ["AAA", "BBB"]
        assert isinstance(loaded.index, pd.DatetimeIndex)

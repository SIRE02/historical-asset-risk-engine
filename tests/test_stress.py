"""Known-value and invariant tests for Phase 5 stress scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from historical_asset_risk.stress import (
    STRESS_CATALOG_SCHEMA_ID,
    STRESS_SCENARIO_SCHEMA_VERSION,
    InstrumentShock,
    StressScenarioError,
    apply_stress,
    canonical_scenario_payload,
    historical_shock_vector,
    load_stress_catalog,
    scenario_content_hash,
)


def _scenario_dict(
    scenario_id: str,
    shocks: dict[str, float],
    *,
    kind: str = "hypothetical",
    version: str = "1",
) -> dict[str, object]:
    shock_objects = [
        InstrumentShock(instrument_id, value) for instrument_id, value in shocks.items()
    ]
    payload = canonical_scenario_payload(
        scenario_id=scenario_id,
        scenario_version=version,
        name=scenario_id,
        kind=kind,
        description="fixture scenario",
        shocks=shock_objects,
    )
    return {**payload, "content_hash": scenario_content_hash(payload)}


def _catalog(tmp_path: Path, scenarios: list[dict[str, object]]) -> Path:
    path = tmp_path / "stress_catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_id": STRESS_CATALOG_SCHEMA_ID,
                "schema_version": STRESS_SCENARIO_SCHEMA_VERSION,
                "scenarios": scenarios,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_catalog_loads_and_verifies_hashes(tmp_path: Path) -> None:
    catalog = _catalog(
        tmp_path,
        [_scenario_dict("risk-off", {"US_SPY": -0.10, "US_QQQ": -0.05})],
    )
    scenarios = load_stress_catalog(catalog)
    assert len(scenarios) == 1
    assert scenarios[0].scenario_id == "risk-off"
    assert scenarios[0].content_hash.startswith("sha256:")


def test_catalog_rejects_mutated_content_under_same_identity(tmp_path: Path) -> None:
    scenario = _scenario_dict("risk-off", {"US_SPY": -0.10})
    # keep the original hash but change the shock magnitude
    scenario["shocks"] = [{"instrument_id": "US_SPY", "simple_return_shock": -0.30}]
    catalog = _catalog(tmp_path, [scenario])
    with pytest.raises(StressScenarioError, match="content hash does not match"):
        load_stress_catalog(catalog)


def test_catalog_rejects_duplicate_identity(tmp_path: Path) -> None:
    catalog = _catalog(
        tmp_path,
        [
            _scenario_dict("dup", {"US_SPY": -0.1}),
            _scenario_dict("dup", {"US_SPY": -0.2}),
        ],
    )
    with pytest.raises(StressScenarioError, match="Duplicate scenario identity"):
        load_stress_catalog(catalog)


def test_apply_stress_known_values_and_contribution_reconciliation(
    tmp_path: Path,
) -> None:
    catalog = load_stress_catalog(
        _catalog(
            tmp_path,
            [_scenario_dict("risk-off", {"US_SPY": -0.10, "US_QQQ": -0.05})],
        )
    )
    result = apply_stress(catalog[0], {"US_SPY": 1000.0, "US_QQQ": -500.0})
    # 1000*-0.10 = -100 ; -500*-0.05 = +25
    assert result.scenario_pnl == pytest.approx(-75.0)
    assert result.scenario_loss == pytest.approx(75.0)
    assert sum(c.scenario_pnl for c in result.contributions) == pytest.approx(-75.0)
    qqq = next(c for c in result.contributions if c.instrument_id == "US_QQQ")
    assert qqq.scenario_pnl == pytest.approx(25.0)  # short exposure reverses the shock


def test_zero_shock_produces_zero_pnl(tmp_path: Path) -> None:
    catalog = load_stress_catalog(
        _catalog(tmp_path, [_scenario_dict("calm", {"US_SPY": 0.0, "US_QQQ": 0.0})])
    )
    result = apply_stress(catalog[0], {"US_SPY": 1000.0, "US_QQQ": -500.0})
    assert result.scenario_pnl == 0.0
    assert result.scenario_loss == 0.0


def test_apply_stress_fails_on_unshocked_held_instrument(tmp_path: Path) -> None:
    catalog = load_stress_catalog(
        _catalog(tmp_path, [_scenario_dict("partial", {"US_SPY": -0.1})])
    )
    with pytest.raises(StressScenarioError, match="does not shock held instrument"):
        apply_stress(catalog[0], {"US_SPY": 1000.0, "US_QQQ": -500.0})


def test_historical_shock_vector_compounds_simple_returns() -> None:
    simple_returns = pd.DataFrame(
        {"US_SPY": [0.1, -0.1], "US_QQQ": [0.0, 0.2]},
        index=pd.to_datetime(["2024-01-03", "2024-01-04"]),
    )
    shock = historical_shock_vector(
        simple_returns, session_start="2024-01-03", session_end="2024-01-04"
    )
    assert shock["US_SPY"] == pytest.approx(1.1 * 0.9 - 1.0)
    assert shock["US_QQQ"] == pytest.approx(1.0 * 1.2 - 1.0)

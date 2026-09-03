"""Phase 5 orchestration: tail-risk and stress artifacts for a released book.

Composes :mod:`historical_asset_risk.tail_risk` and
:mod:`historical_asset_risk.stress` over the Phase 4 loss samples. Performs no
file writing and no market-data acquisition; the CLI owns those.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from historical_asset_risk.contracts import (
    EXPECTED_SHORTFALL_TAIL_WEIGHTS_COLUMNS,
    PHASE5_SCHEMA_VERSION,
    PORTFOLIO_EXPECTED_SHORTFALL_COLUMNS,
    PORTFOLIO_VALUE_AT_RISK_COLUMNS,
    STRESS_CONTRIBUTIONS_COLUMNS,
    STRESS_TEST_RESULTS_COLUMNS,
    TAIL_RISK_COMPARISON_COLUMNS,
    TRAILING_TAIL_RISK_COLUMNS,
    Instrument,
    PortfolioValuation,
)
from historical_asset_risk.pnl import (
    ProxyExposureSnapshot,
    align_portfolio_simple_returns,
)
from historical_asset_risk.stress import (
    STRESS_CATALOG_SCHEMA_ID,
    STRESS_SCENARIO_SCHEMA_VERSION,
    StressScenario,
    apply_stress,
)
from historical_asset_risk.tail_risk import (
    CANONICAL_QUANTILE_METHOD,
    historical_es,
    historical_var,
    normal_es,
    normal_var,
)

PHASE5_CORE_ARTIFACT_NAMES: tuple[str, ...] = (
    "portfolio_value_at_risk.csv",
    "portfolio_expected_shortfall.csv",
    "portfolio_expected_shortfall_tail_weights.csv",
    "tail_risk_comparison.csv",
)

_DIMENSIONS = (
    ("return", "return_decimal"),
    ("currency", "base_currency"),
)


@dataclass(frozen=True)
class Phase5Result:
    """Everything the CLI needs to persist the Phase 5 tail-risk layer."""

    frames: dict[str, pd.DataFrame]
    catalog_json: dict[str, Any] | None
    quality_section: dict[str, Any]
    manifest_section: dict[str, Any]
    summary_lines: tuple[str, ...]


def _loss_sample(matrix: np.ndarray, weights: list[float]) -> np.ndarray:
    return -(matrix @ np.asarray(weights, dtype=float))


def compute_phase5(
    valuation: PortfolioValuation,
    simple_returns: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    *,
    data_snapshot_id: str,
    confidence_level: float,
    window: int | None,
    registry_instruments: Sequence[Instrument] = (),
    exposure_history: Sequence[ProxyExposureSnapshot] = (),
    stress_scenarios: Sequence[StressScenario] = (),
) -> Phase5Result:
    """Compute every Phase 5 artifact frame for one released book."""
    snapshot = valuation.snapshot
    cash = snapshot.cash
    held_ids = list(snapshot.reconciliation.held_instrument_ids)
    alpha = float(confidence_level)

    aligned = align_portfolio_simple_returns(
        simple_returns,
        snapshot.instruments,
        held_instrument_ids=held_ids,
        price_index=price_index,
        market_calendar_id=cash.market_calendar_id,
        data_snapshot_id=data_snapshot_id,
    )
    matrix = aligned.matrix.to_numpy(dtype=float)
    return_weights = [
        next(
            item.weight
            for item in valuation.positions
            if item.position.instrument_id == instrument_id
        )
        for instrument_id in aligned.matrix.columns
    ]
    currency_exposures_by_id = {
        item.position.instrument_id: item.position_value
        for item in valuation.positions
    }
    currency_vector = [
        currency_exposures_by_id[instrument_id]
        for instrument_id in aligned.matrix.columns
    ]

    identity = {
        "portfolio_snapshot_id": cash.portfolio_snapshot_id,
        "exposure_snapshot_id": valuation.exposure_snapshot_id,
        "portfolio_id": cash.portfolio_id,
        "as_of_date": cash.as_of_date.isoformat(),
        "base_currency": cash.base_currency,
        "data_snapshot_id": data_snapshot_id,
    }
    samples = {
        "return": _loss_sample(matrix, return_weights),
        "currency": _loss_sample(matrix, currency_vector),
    }

    var_rows: list[dict[str, Any]] = []
    es_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    es_ge_var_all = True

    for dimension, units in _DIMENSIONS:
        losses = samples[dimension]
        hv = historical_var(losses, alpha, units=units)
        he = historical_es(losses, alpha, units=units)
        nv = normal_var(losses, alpha, units=units)
        ne = normal_es(losses, alpha, units=units)
        es_ge_var_all = es_ge_var_all and he.es_ge_var
        for warning in (hv.tail_sample_warning, he.tail_sample_warning):
            if warning and warning not in warnings:
                warnings.append(f"{dimension}: {warning}")

        var_rows.append(
            {
                **identity,
                "dimension": dimension,
                "units": units,
                "confidence_level": alpha,
                "observation_count": hv.observation_count,
                "value_at_risk": hv.value_at_risk,
                "order_statistic_rank": hv.order_statistic_rank,
                "quantile_method": hv.quantile_method,
                "tail_sample_warning": hv.tail_sample_warning or "",
                "schema_version": PHASE5_SCHEMA_VERSION,
            }
        )
        es_rows.append(
            {
                **identity,
                "dimension": dimension,
                "units": units,
                "confidence_level": alpha,
                "observation_count": he.observation_count,
                "expected_shortfall": he.expected_shortfall,
                "value_at_risk": he.value_at_risk,
                "nominal_tail_observations": he.nominal_tail_observations,
                "full_tail_count": he.full_tail_count,
                "boundary_weight": he.boundary_weight,
                "contributing_observation_count": he.contributing_observation_count,
                "es_ge_var": he.es_ge_var,
                "tail_sample_warning": he.tail_sample_warning or "",
                "schema_version": PHASE5_SCHEMA_VERSION,
            }
        )
        for item in he.tail_contributions:
            weight_rows.append(
                {
                    "exposure_snapshot_id": valuation.exposure_snapshot_id,
                    "dimension": dimension,
                    "units": units,
                    "confidence_level": alpha,
                    "order_statistic_rank": item.order_statistic_rank,
                    "loss": item.loss,
                    "weight": item.weight,
                    "schema_version": PHASE5_SCHEMA_VERSION,
                }
            )
        comparison_rows.append(
            {
                **identity,
                "dimension": dimension,
                "units": units,
                "confidence_level": alpha,
                "observation_count": hv.observation_count,
                "historical_var": hv.value_at_risk,
                "historical_es": he.expected_shortfall,
                "normal_var_mean_included": nv.mean_included,
                "normal_es_mean_included": ne.mean_included,
                "normal_var_zero_mean": nv.zero_mean,
                "normal_es_zero_mean": ne.zero_mean,
                "mean_loss": nv.mean_loss,
                "standard_deviation": nv.standard_deviation,
                "z_alpha": nv.z_alpha,
                "schema_version": PHASE5_SCHEMA_VERSION,
            }
        )

    frames: dict[str, pd.DataFrame] = {
        "portfolio_value_at_risk.csv": pd.DataFrame(
            var_rows, columns=list(PORTFOLIO_VALUE_AT_RISK_COLUMNS)
        ),
        "portfolio_expected_shortfall.csv": pd.DataFrame(
            es_rows, columns=list(PORTFOLIO_EXPECTED_SHORTFALL_COLUMNS)
        ),
        "portfolio_expected_shortfall_tail_weights.csv": pd.DataFrame(
            weight_rows, columns=list(EXPECTED_SHORTFALL_TAIL_WEIGHTS_COLUMNS)
        ),
        "tail_risk_comparison.csv": pd.DataFrame(
            comparison_rows, columns=list(TAIL_RISK_COMPARISON_COLUMNS)
        ),
    }

    trailing = _trailing_tail_risk(
        exposure_history,
        registry_instruments or snapshot.instruments,
        simple_returns,
        price_index,
        market_calendar_id=cash.market_calendar_id,
        data_snapshot_id=data_snapshot_id,
        confidence_level=alpha,
        window=window,
    )
    if trailing is not None:
        frames["trailing_portfolio_tail_risk.csv"] = trailing["frame"]

    stress = _stress_artifacts(
        stress_scenarios, currency_exposures_by_id, identity, valuation
    )
    frames.update(stress["frames"])

    manifest_section = {
        "confidence_level": alpha,
        "window": window if window is not None else "full_aligned_sample",
        "quantile_method": CANONICAL_QUANTILE_METHOD,
        "es_algorithm": "exact_finite_sample_equal_mass_with_fractional_boundary",
        "dimensions": [name for name, _ in _DIMENSIONS],
        "observation_count": int(matrix.shape[0]),
        "normal_model": "gaussian_comparison_benchmark_not_a_normality_claim",
        "annualized": False,
        "not_a_forecast": (
            "static bounded-sample measurement; no coverage tests, research "
            "splits, or forecast records"
        ),
        "trailing": trailing["manifest"] if trailing is not None else {
            "status": "not_supplied"
        },
        "stress": stress["manifest"],
    }
    quality_section = {
        "loss_observation_count": int(matrix.shape[0]),
        "confidence_level": alpha,
        "es_ge_var": es_ge_var_all,
        "tail_sample_warnings": warnings,
        "trailing_record_count": (
            trailing["record_count"] if trailing is not None else 0
        ),
        "stress_scenario_count": stress["scenario_count"],
    }
    var_return = frames["portfolio_value_at_risk.csv"].iloc[0]["value_at_risk"]
    var_currency = frames["portfolio_value_at_risk.csv"].iloc[1]["value_at_risk"]
    summary = (
        f"Phase 5 tail risk (alpha {alpha:.4g}): return VaR {var_return:.6f}, "
        f"currency VaR {var_currency:.2f} {cash.base_currency}"
    )
    if stress["scenario_count"]:
        summary += f"; {stress['scenario_count']} stress scenario(s)"
    return Phase5Result(
        frames=frames,
        catalog_json=stress["catalog_json"],
        quality_section=quality_section,
        manifest_section=manifest_section,
        summary_lines=(summary,),
    )


def _trailing_tail_risk(
    exposure_history: Sequence[ProxyExposureSnapshot],
    instruments: Sequence[Instrument],
    simple_returns: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    *,
    market_calendar_id: str,
    data_snapshot_id: str,
    confidence_level: float,
    window: int | None,
) -> dict[str, Any] | None:
    if not exposure_history:
        return None
    history_ids = sorted(
        {
            instrument_id
            for snapshot in exposure_history
            for instrument_id in snapshot.currency_exposures
        }
    )
    aligned = align_portfolio_simple_returns(
        simple_returns,
        instruments,
        held_instrument_ids=history_ids,
        price_index=price_index,
        market_calendar_id=market_calendar_id,
        data_snapshot_id=data_snapshot_id,
    )
    end_dates = pd.DatetimeIndex(aligned.matrix.index)
    rows: list[dict[str, Any]] = []
    for snapshot in sorted(exposure_history, key=lambda item: item.as_of_date):
        as_of = pd.Timestamp(snapshot.as_of_date)
        mask = end_dates <= as_of
        sub = aligned.matrix.loc[mask]
        if window is not None:
            sub = sub.tail(window)
        if sub.empty:
            rows.append(
                {
                    "as_of_date": snapshot.as_of_date.isoformat(),
                    "exposure_snapshot_id": snapshot.exposure_snapshot_id,
                    "window": window if window is not None else "trailing_to_date",
                    "window_observation_count": 0,
                    "confidence_level": confidence_level,
                    "value_at_risk": "",
                    "expected_shortfall": "",
                    "es_ge_var": "",
                    "outcome_status": "no_trailing_observations",
                    "units": "base_currency",
                    "schema_version": PHASE5_SCHEMA_VERSION,
                }
            )
            continue
        exposure_vector = [
            float(snapshot.currency_exposures.get(instrument_id, 0.0))
            for instrument_id in sub.columns
        ]
        losses = -(sub.to_numpy(dtype=float) @ np.asarray(exposure_vector))
        hv = historical_var(losses, confidence_level, units="base_currency")
        he = historical_es(losses, confidence_level, units="base_currency")
        rows.append(
            {
                "as_of_date": snapshot.as_of_date.isoformat(),
                "exposure_snapshot_id": snapshot.exposure_snapshot_id,
                "window": window if window is not None else "trailing_to_date",
                "window_observation_count": int(len(sub)),
                "confidence_level": confidence_level,
                "value_at_risk": hv.value_at_risk,
                "expected_shortfall": he.expected_shortfall,
                "es_ge_var": he.es_ge_var,
                "outcome_status": "measured",
                "units": "base_currency",
                "schema_version": PHASE5_SCHEMA_VERSION,
            }
        )
    return {
        "frame": pd.DataFrame(rows, columns=list(TRAILING_TAIL_RISK_COLUMNS)),
        "record_count": len(rows),
        "manifest": {
            "status": "computed",
            "keyed_by": ["as_of_date", "exposure_snapshot_id", "window", "confidence"],
            "record_count": len(rows),
            "not_a_forecast_store": True,
        },
    }


def _stress_artifacts(
    scenarios: Sequence[StressScenario],
    currency_exposures: dict[str, float],
    identity: dict[str, Any],
    valuation: PortfolioValuation,
) -> dict[str, Any]:
    if not scenarios:
        return {
            "frames": {},
            "catalog_json": None,
            "scenario_count": 0,
            "manifest": {"status": "not_supplied", "probability_fields": "none"},
        }
    result_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    catalog_entries: list[dict[str, Any]] = []
    for scenario in scenarios:
        outcome = apply_stress(scenario, currency_exposures)
        result_rows.append(
            {
                **identity,
                "scenario_id": scenario.scenario_id,
                "scenario_version": scenario.scenario_version,
                "name": scenario.name,
                "kind": scenario.kind,
                "content_hash": scenario.content_hash,
                "scenario_pnl": outcome.scenario_pnl,
                "scenario_loss": outcome.scenario_loss,
                "units": "base_currency",
                "loss_sign": "loss_is_positive",
                "schema_version": PHASE5_SCHEMA_VERSION,
            }
        )
        for item in outcome.contributions:
            contribution_rows.append(
                {
                    "exposure_snapshot_id": valuation.exposure_snapshot_id,
                    "scenario_id": scenario.scenario_id,
                    "scenario_version": scenario.scenario_version,
                    "instrument_id": item.instrument_id,
                    "currency_exposure": item.currency_exposure,
                    "simple_return_shock": item.simple_return_shock,
                    "scenario_pnl": item.scenario_pnl,
                    "scenario_loss": item.scenario_loss,
                    "schema_version": PHASE5_SCHEMA_VERSION,
                }
            )
        catalog_entries.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_version": scenario.scenario_version,
                "name": scenario.name,
                "kind": scenario.kind,
                "description": scenario.description,
                "content_hash": scenario.content_hash,
                "session_start": scenario.session_start,
                "session_end": scenario.session_end,
                "shock_count": len(scenario.shocks),
            }
        )
    return {
        "frames": {
            "stress_test_results.csv": pd.DataFrame(
                result_rows, columns=list(STRESS_TEST_RESULTS_COLUMNS)
            ),
            "stress_contributions.csv": pd.DataFrame(
                contribution_rows, columns=list(STRESS_CONTRIBUTIONS_COLUMNS)
            ),
        },
        "catalog_json": {
            "schema_id": STRESS_CATALOG_SCHEMA_ID,
            "schema_version": STRESS_SCENARIO_SCHEMA_VERSION,
            "resolved_at_exposure_snapshot_id": valuation.exposure_snapshot_id,
            "scenarios": catalog_entries,
        },
        "scenario_count": len(scenarios),
        "manifest": {
            "status": "computed",
            "scenario_count": len(scenarios),
            "identities": [
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_version": scenario.scenario_version,
                    "content_hash": scenario.content_hash,
                }
                for scenario in scenarios
            ],
            "probability_fields": "none",
            "note": "deterministic P&L under a named shock; not a forecast",
        },
    }


__all__ = ["PHASE5_CORE_ARTIFACT_NAMES", "Phase5Result", "compute_phase5"]

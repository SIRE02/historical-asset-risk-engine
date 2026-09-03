"""Phase 4 orchestration: turn a Phase 3 book plus simple returns into artifacts.

This module composes the pure functions in :mod:`historical_asset_risk.pnl` and
:mod:`historical_asset_risk.portfolio_risk` into the deterministic CSV frames,
data-quality section, and run-manifest section emitted when a portfolio book is
present. It performs no file writing and no market-data acquisition; the CLI
owns those.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from historical_asset_risk.contracts import (
    HYPOTHETICAL_PNL_COLUMNS,
    HYPOTHETICAL_PNL_SCHEMA_ID,
    PHASE4_SCHEMA_VERSION,
    PORTFOLIO_ALIGNED_SIMPLE_RETURNS_SCHEMA_ID,
    PORTFOLIO_CONCENTRATION_SUMMARY_SCHEMA_ID,
    PORTFOLIO_RISK_CONTRIBUTIONS_SCHEMA_ID,
    PORTFOLIO_RISK_SUMMARY_SCHEMA_ID,
    PROXY_REALIZED_PNL_COLUMNS,
    PROXY_REALIZED_PNL_SCHEMA_ID,
    RISK_REALIZATION_CALCULATION_VERSION,
    RISK_REALIZATIONS_COLUMNS,
    RISK_REALIZATIONS_SCHEMA_ID,
    SIMPLE_RETURN_CORRELATION_SCHEMA_ID,
    SIMPLE_RETURN_COVARIANCE_SCHEMA_ID,
    SIMPLE_RETURN_SUMMARY_SCHEMA_ID,
    Instrument,
    PortfolioValuation,
)
from historical_asset_risk.estimation import SAMPLE_DDOF
from historical_asset_risk.market_calendar import resolve_market_calendar
from historical_asset_risk.pnl import (
    ALIGNED_SIMPLE_RETURN_TYPE,
    ProxyExposureSnapshot,
    align_portfolio_simple_returns,
    compute_data_snapshot_id,
    hypothetical_pnl,
    proxy_realized_pnl,
)
from historical_asset_risk.portfolio_risk import (
    compound_simple_returns,
    portfolio_risk_from_covariance,
    sample_simple_return_covariance,
    simple_return_correlation,
    simple_return_summary,
)

PHASE4_ARTIFACT_NAMES: tuple[str, ...] = (
    "portfolio_aligned_simple_returns.csv",
    "hypothetical_portfolio_pnl.csv",
    "portfolio_simple_return_covariance.csv",
    "portfolio_simple_return_correlation.csv",
    "simple_return_summary.csv",
    "portfolio_risk_summary.csv",
    "portfolio_risk_contributions.csv",
    "portfolio_concentration_summary.csv",
)


@dataclass(frozen=True)
class Phase4Result:
    """Everything the CLI needs to persist the Phase 4 book layer."""

    frames: dict[str, pd.DataFrame]
    """Artifact name -> deterministic frame. Written with ``index=False``."""
    matrix_frames: frozenset[str]
    """Frames in ``frames`` that carry a meaningful index column."""
    quality_section: dict[str, Any]
    manifest_section: dict[str, Any]
    summary_lines: tuple[str, ...]


def _index_frame(frame: pd.DataFrame, index_label: str) -> pd.DataFrame:
    reset = frame.copy()
    reset.index = reset.index.astype(str)
    reset.index.name = index_label
    return reset.reset_index()


# Fields a downstream forecasting engine would populate when it joins its own
# predictions to a HARE realization record. HARE never fills these itself.
_FORECAST_JOIN_FIELDS = (
    "forecast_id",
    "forecast_generated_at",
    "forecast_horizon_start",
    "forecast_horizon_end",
    "predicted_loss",
    "predicted_quantile_level",
    "model_id",
)
_REALIZATION_NOTE = (
    "HARE emits realization identity only. It does not generate competing "
    "forecasts or run Kupiec, Christoffersen, or pinball coverage tests."
)


def _proxy_realized_artifacts(
    exposure_history: Sequence[ProxyExposureSnapshot],
    instruments: Sequence[Instrument],
    simple_returns: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    *,
    market_calendar_id: str,
    data_snapshot_id: str,
) -> dict[str, Any]:
    """Build proxy realized P&L and realization-identity frames, if a history."""
    if not exposure_history:
        return {
            "frames": {},
            "available": False,
            "quality": None,
            "manifest": {
                "status": "not_supplied",
                "calculation_version": RISK_REALIZATION_CALCULATION_VERSION,
                "downstream_forecast_join_fields": list(_FORECAST_JOIN_FIELDS),
                "notes": _REALIZATION_NOTE,
            },
        }

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
    calendar = resolve_market_calendar(market_calendar_id)
    proxy = proxy_realized_pnl(
        exposure_history, aligned, next_session=calendar.next_session
    )

    proxy_frame = proxy.assign(
        data_snapshot_id=data_snapshot_id,
        return_type=ALIGNED_SIMPLE_RETURN_TYPE,
        units="base_currency",
        loss_sign="loss_is_positive",
        calculation_version=RISK_REALIZATION_CALCULATION_VERSION,
        schema_version=PHASE4_SCHEMA_VERSION,
    ).loc[:, list(PROXY_REALIZED_PNL_COLUMNS)]
    realizations_frame = pd.DataFrame(
        {
            "portfolio_id": proxy["portfolio_id"],
            "exposure_snapshot_id": proxy["exposure_snapshot_id"],
            "market_calendar_id": proxy["market_calendar_id"],
            "target_period_start": proxy["target_period_start"],
            "target_period_end": proxy["target_period_end"],
            "units": "base_currency",
            "loss_sign": "loss_is_positive",
            "realized_loss": proxy["proxy_realized_loss"],
            "outcome_status": proxy["outcome_status"],
            "data_snapshot_id": data_snapshot_id,
            "calculation_version": RISK_REALIZATION_CALCULATION_VERSION,
            "schema_version": PHASE4_SCHEMA_VERSION,
        }
    ).loc[:, list(RISK_REALIZATIONS_COLUMNS)]

    record_count = int(len(proxy))
    realized_count = int((proxy["outcome_status"] == "realized").sum())
    status_counts = {
        str(key): int(value)
        for key, value in proxy["outcome_status"].value_counts().items()
    }
    return {
        "frames": {
            "proxy_realized_portfolio_pnl.csv": proxy_frame,
            "risk_realizations.csv": realizations_frame,
        },
        "available": True,
        "quality": {
            "record_count": record_count,
            "realized_count": realized_count,
            "outcome_status_counts": status_counts,
            "history_instrument_ids": history_ids,
        },
        "manifest": {
            "status": "computed",
            "calculation_version": RISK_REALIZATION_CALCULATION_VERSION,
            "target_horizon": "next_valid_session",
            "record_count": record_count,
            "realized_count": realized_count,
            "units": "base_currency",
            "loss_sign": "loss_is_positive",
            "downstream_forecast_join_fields": list(_FORECAST_JOIN_FIELDS),
            "notes": _REALIZATION_NOTE,
        },
    }


def compute_phase4(
    valuation: PortfolioValuation,
    simple_returns: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    *,
    data_source: Mapping[str, Any],
    observations_per_year: int,
    registry_instruments: Sequence[Instrument] = (),
    exposure_history: Sequence[ProxyExposureSnapshot] = (),
) -> Phase4Result:
    """Compute every Phase 4 artifact frame for one valued snapshot.

    ``exposure_history`` is an optional ordered collection of dated Phase 3
    exposure snapshots. When supplied it adds proxy realized P&L and the
    versioned realization identity artifact.
    """
    snapshot = valuation.snapshot
    cash = snapshot.cash
    held_ids = list(snapshot.reconciliation.held_instrument_ids)
    data_snapshot_id = compute_data_snapshot_id(data_source)

    aligned = align_portfolio_simple_returns(
        simple_returns,
        snapshot.instruments,
        held_instrument_ids=held_ids,
        price_index=price_index,
        market_calendar_id=cash.market_calendar_id,
        data_snapshot_id=data_snapshot_id,
    )

    weights = {
        item.position.instrument_id: item.weight for item in valuation.positions
    }
    currency_exposures = {
        item.position.instrument_id: item.position_value
        for item in valuation.positions
    }

    pnl = hypothetical_pnl(currency_exposures, aligned)
    covariance = sample_simple_return_covariance(aligned.matrix)
    correlation = simple_return_correlation(aligned.matrix)
    summary = simple_return_summary(aligned.matrix)
    cumulative = compound_simple_returns(aligned.matrix)
    risk = portfolio_risk_from_covariance(
        weights,
        currency_exposures,
        covariance,
        observations_per_year=observations_per_year,
    )
    observation_count = int(aligned.matrix.shape[0])

    identity = {
        "portfolio_snapshot_id": cash.portfolio_snapshot_id,
        "exposure_snapshot_id": valuation.exposure_snapshot_id,
        "portfolio_id": cash.portfolio_id,
        "as_of_date": cash.as_of_date.isoformat(),
        "as_of_timestamp": cash.as_of_timestamp.isoformat(),
        "base_currency": cash.base_currency,
        "market_calendar_id": cash.market_calendar_id,
        "data_snapshot_id": data_snapshot_id,
    }

    aligned_frame = aligned.long_frame()

    pnl_frame = pnl.assign(
        return_type=ALIGNED_SIMPLE_RETURN_TYPE,
        units="base_currency",
        loss_sign="loss_is_positive",
        schema_version=PHASE4_SCHEMA_VERSION,
        **identity,
    )
    pnl_frame = pnl_frame.loc[:, list(HYPOTHETICAL_PNL_COLUMNS)]

    summary_frame = _index_frame(summary, "instrument_id").assign(
        cumulative_simple_return=lambda frame: frame["instrument_id"].map(
            cumulative.to_dict()
        ),
        return_type=ALIGNED_SIMPLE_RETURN_TYPE,
        ddof=SAMPLE_DDOF,
        schema_version=PHASE4_SCHEMA_VERSION,
    )

    contributions_frame = risk.contributions.assign(
        exposure_snapshot_id=valuation.exposure_snapshot_id,
        currency_exposure=lambda frame: frame["instrument_id"].map(currency_exposures),
        zero_volatility_policy_applied=risk.zero_volatility,
        schema_version=PHASE4_SCHEMA_VERSION,
    )
    contributions_frame = contributions_frame.loc[
        :,
        [
            "exposure_snapshot_id",
            "instrument_id",
            "weight",
            "currency_exposure",
            "portfolio_covariance_vector",
            "marginal_volatility",
            "component_volatility",
            "percentage_component",
            "zero_volatility_policy_applied",
            "schema_version",
        ],
    ]

    risk_summary_frame = pd.DataFrame(
        [
            {
                **identity,
                "observation_count": observation_count,
                "ddof": SAMPLE_DDOF,
                "observations_per_year": risk.observations_per_year,
                "return_variance": risk.return_variance,
                "return_volatility": risk.return_volatility,
                "currency_variance": risk.currency_variance,
                "currency_volatility": risk.currency_volatility,
                "annualized_return_volatility": risk.annualized_return_volatility,
                "annualized_currency_volatility": risk.annualized_currency_volatility,
                "annualization_method": "square_root_of_time_approximation",
                "zero_volatility_policy_applied": risk.zero_volatility,
                "covariance_condition_number": risk.condition_number,
                "covariance_warning": risk.warning or "",
                "schema_version": PHASE4_SCHEMA_VERSION,
            }
        ]
    )

    concentration_frame = pd.DataFrame(
        [
            {
                "exposure_snapshot_id": valuation.exposure_snapshot_id,
                "portfolio_id": cash.portfolio_id,
                "as_of_date": cash.as_of_date.isoformat(),
                **{key: float(value) for key, value in risk.concentration.items()},
                "schema_version": PHASE4_SCHEMA_VERSION,
            }
        ]
    )

    frames: dict[str, pd.DataFrame] = {
        "portfolio_aligned_simple_returns.csv": aligned_frame,
        "hypothetical_portfolio_pnl.csv": pnl_frame,
        "portfolio_simple_return_covariance.csv": _index_frame(
            covariance, "instrument_id"
        ),
        "portfolio_simple_return_correlation.csv": _index_frame(
            correlation, "instrument_id"
        ),
        "simple_return_summary.csv": summary_frame,
        "portfolio_risk_summary.csv": risk_summary_frame,
        "portfolio_risk_contributions.csv": contributions_frame,
        "portfolio_concentration_summary.csv": concentration_frame,
    }

    proxy_section = _proxy_realized_artifacts(
        exposure_history,
        registry_instruments or snapshot.instruments,
        simple_returns,
        price_index,
        market_calendar_id=cash.market_calendar_id,
        data_snapshot_id=data_snapshot_id,
    )
    frames.update(proxy_section["frames"])

    manifest_section = {
        "data_snapshot_id": data_snapshot_id,
        "exposure_snapshot_id": valuation.exposure_snapshot_id,
        "aligned_return_type": ALIGNED_SIMPLE_RETURN_TYPE,
        "covariance": {
            "estimator": "sample",
            "ddof": SAMPLE_DDOF,
            "observation_count": observation_count,
            "window": "full_aligned_sample",
            "market_calendar_id": cash.market_calendar_id,
            "condition_number": risk.condition_number,
            "warning": risk.warning,
            "shrinkage": "none",
            "annualized": False,
        },
        "aggregation": {
            "return_variance": risk.return_variance,
            "return_volatility": risk.return_volatility,
            "currency_variance": risk.currency_variance,
            "currency_volatility": risk.currency_volatility,
            "annualization": "square_root_of_time_approximation",
            "zero_volatility_policy_applied": risk.zero_volatility,
        },
        "loss_sign": "loss_is_positive",
        "distinct_from": {
            "log_return_covariance_artifact": "covariance_matrix.csv",
            "reason": "simple-return covariance for exact linear P&L aggregation",
        },
        "realizations": proxy_section["manifest"],
    }

    quality_section = {
        "aligned_observation_count": observation_count,
        "aligned_interval_start": aligned_frame["period_start"].min(),
        "aligned_interval_end": aligned_frame["period_end"].max(),
        "held_instrument_ids": held_ids,
        "hypothetical_pnl_rows": int(len(pnl_frame)),
        "component_volatility_reconciles": bool(
            abs(
                float(contributions_frame["component_volatility"].sum())
                - risk.return_volatility
            )
            <= 1e-9 + 1e-6 * risk.return_volatility
        ),
        "covariance_warning": risk.warning,
        "proxy_realized_pnl_available": proxy_section["available"],
        "proxy_realized_pnl": proxy_section["quality"],
    }

    proxy_summary = ""
    if proxy_section["available"]:
        proxy_summary = (
            f"; proxy realized {proxy_section['quality']['realized_count']}"
            f"/{proxy_section['quality']['record_count']} snapshots"
        )
    summary_lines = (
        f"Phase 4 book P&L: {observation_count} aligned intervals, "
        f"return vol {risk.return_volatility:.6f}, "
        f"currency vol {risk.currency_volatility:.2f} {cash.base_currency}"
        + (f" [{risk.warning}]" if risk.warning else "")
        + proxy_summary,
    )

    return Phase4Result(
        frames=frames,
        matrix_frames=frozenset(
            {
                "portfolio_simple_return_covariance.csv",
                "portfolio_simple_return_correlation.csv",
            }
        ),
        quality_section=quality_section,
        manifest_section=manifest_section,
        summary_lines=summary_lines,
    )


PHASE4_SCHEMA_IDS: dict[str, str] = {
    "portfolio_aligned_simple_returns.csv": (
        PORTFOLIO_ALIGNED_SIMPLE_RETURNS_SCHEMA_ID
    ),
    "hypothetical_portfolio_pnl.csv": HYPOTHETICAL_PNL_SCHEMA_ID,
    "portfolio_simple_return_covariance.csv": SIMPLE_RETURN_COVARIANCE_SCHEMA_ID,
    "portfolio_simple_return_correlation.csv": SIMPLE_RETURN_CORRELATION_SCHEMA_ID,
    "simple_return_summary.csv": SIMPLE_RETURN_SUMMARY_SCHEMA_ID,
    "portfolio_risk_summary.csv": PORTFOLIO_RISK_SUMMARY_SCHEMA_ID,
    "portfolio_risk_contributions.csv": PORTFOLIO_RISK_CONTRIBUTIONS_SCHEMA_ID,
    "portfolio_concentration_summary.csv": PORTFOLIO_CONCENTRATION_SUMMARY_SCHEMA_ID,
    "proxy_realized_portfolio_pnl.csv": PROXY_REALIZED_PNL_SCHEMA_ID,
    "risk_realizations.csv": RISK_REALIZATIONS_SCHEMA_ID,
}


__all__ = [
    "PHASE4_ARTIFACT_NAMES",
    "PHASE4_SCHEMA_IDS",
    "Phase4Result",
    "compute_phase4",
]

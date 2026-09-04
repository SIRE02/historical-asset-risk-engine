"""Phase 4 portfolio profit-and-loss from a Phase 3 book and simple returns.

This module is deliberately independent of providers, configuration, plotting,
and file writing. It maps Phase 1 provider return columns to canonical
``instrument_id`` values, aligns simple returns to a held book, and produces
hypothetical historical P&L and (with a dated exposure history) proxy realized
P&L. It never rewrites ``simple_returns.csv`` or any other frozen consumer file.

Conventions (see ``docs/historical-asset-risk-engine.md`` section 5.3):

* Portfolio P&L uses daily **simple** returns because currency P&L aggregates
  linearly as ``exposure * simple_return``.
* Loss is positive: ``loss = -pnl``.
* Cash has zero market return and is not part of the P&L vector.
* Returns, exposures, and weights align by ``instrument_id``, never by column
  position.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from historical_asset_risk.contracts import (
    PHASE4_SCHEMA_VERSION,
    Instrument,
    PortfolioReturnAlignmentError,
)

ALIGNED_SIMPLE_RETURN_TYPE = "simple_return_pct_change"


def compute_data_snapshot_id(data_source: Mapping[str, Any]) -> str:
    """Derive a deterministic identity for the aligned market-data sample.

    The identity is a hash of the provider, source, realized date bounds,
    observation count, and instrument identities recorded in the run manifest's
    ``data_source`` block. The same market sample always yields the same id.
    """
    material = {
        "provider": data_source.get("provider"),
        "source": data_source.get("source"),
        "actual_start_date": data_source.get("actual_start_date"),
        "actual_end_date": data_source.get("actual_end_date"),
        "observation_count": data_source.get("observation_count"),
        "instruments": list(data_source.get("instruments", []) or []),
    }
    payload = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "data_" + hashlib.sha256(payload).hexdigest()


def map_provider_tickers_to_instrument_ids(
    instruments: Sequence[Instrument],
    available_tickers: Sequence[str],
    *,
    required_instrument_ids: Sequence[str],
) -> dict[str, str]:
    """Map provider return columns to canonical ``instrument_id`` values.

    ``available_tickers`` are the provider columns of ``simple_returns.csv``.
    ``required_instrument_ids`` are the instruments actually held in the book.

    Fails on duplicate provider mappings, held instruments whose provider
    column is absent from the return matrix, and required identities that are
    not in the registry. Extra provider columns that no held instrument uses
    are ignored rather than mapped.
    """
    normalized_available: list[str] = []
    for ticker in available_tickers:
        text = str(ticker).strip().upper()
        if not text:
            raise PortfolioReturnAlignmentError(
                "Simple-return column identifiers must be non-empty."
            )
        normalized_available.append(text)
    available_set = set(normalized_available)
    if len(available_set) != len(normalized_available):
        raise PortfolioReturnAlignmentError(
            "The simple-return matrix has duplicate provider columns."
        )

    instrument_by_id: dict[str, Instrument] = {}
    ticker_to_id: dict[str, str] = {}
    for instrument in instruments:
        instrument_by_id[instrument.instrument_id] = instrument
        ticker = str(instrument.provider_ticker).strip().upper()
        existing = ticker_to_id.get(ticker)
        if existing is not None and existing != instrument.instrument_id:
            raise PortfolioReturnAlignmentError(
                f"Provider ticker {ticker!r} maps to multiple instrument ids: "
                f"{existing!r} and {instrument.instrument_id!r}."
            )
        ticker_to_id[ticker] = instrument.instrument_id

    unknown_required = [
        instrument_id
        for instrument_id in required_instrument_ids
        if instrument_id not in instrument_by_id
    ]
    if unknown_required:
        raise PortfolioReturnAlignmentError(
            "Required instrument id(s) absent from the registry: "
            + ", ".join(sorted(unknown_required))
            + "."
        )

    mapping: dict[str, str] = {}
    missing: list[str] = []
    for instrument_id in required_instrument_ids:
        ticker = str(instrument_by_id[instrument_id].provider_ticker).strip().upper()
        if ticker not in available_set:
            missing.append(f"{instrument_id} ({ticker})")
            continue
        mapping[ticker] = instrument_id
    if missing:
        raise PortfolioReturnAlignmentError(
            "Held instruments have no simple-return column: "
            + ", ".join(sorted(missing))
            + "."
        )
    return mapping


@dataclass(frozen=True)
class AlignedPortfolioReturns:
    """Held-book simple returns aligned by ``instrument_id`` with intervals."""

    matrix: pd.DataFrame
    """Wide returns: ``period_end`` DatetimeIndex, held ``instrument_id`` columns."""
    period_start: pd.Series
    """Interval start session per ``period_end``, aligned to ``matrix.index``."""
    market_calendar_id: str
    return_type: str
    data_snapshot_id: str

    def long_frame(self) -> pd.DataFrame:
        """Return the deterministic long-form ``instrument_id`` x interval table."""
        rows: list[dict[str, object]] = []
        for instrument_id in self.matrix.columns:
            series = self.matrix[instrument_id]
            for period_end, value in series.items():
                rows.append(
                    {
                        "instrument_id": instrument_id,
                        "period_start": pd.Timestamp(self.period_start.loc[period_end])
                        .date()
                        .isoformat(),
                        "period_end": pd.Timestamp(period_end).date().isoformat(),
                        "market_calendar_id": self.market_calendar_id,
                        "return_type": self.return_type,
                        "data_snapshot_id": self.data_snapshot_id,
                        "simple_return": float(value),
                        "schema_version": PHASE4_SCHEMA_VERSION,
                    }
                )
        return pd.DataFrame(rows)


def align_portfolio_simple_returns(
    simple_returns: pd.DataFrame,
    instruments: Sequence[Instrument],
    *,
    held_instrument_ids: Sequence[str],
    price_index: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    market_calendar_id: str,
    data_snapshot_id: str,
) -> AlignedPortfolioReturns:
    """Restrict and rename Phase 1 simple returns to the held book.

    ``simple_returns`` is the wide Phase 1 matrix (``date`` index, provider
    columns). ``price_index`` is the full adjusted-price session index used to
    recover each return's interval start (the preceding session). The Phase 2
    gap-spanning policy guarantees retained sessions are consecutive, so the
    interval start is simply the previous entry in ``price_index``.
    """
    if simple_returns.empty:
        raise PortfolioReturnAlignmentError("The simple-return matrix is empty.")
    ordered_ids = list(dict.fromkeys(str(item) for item in held_instrument_ids))
    if not ordered_ids:
        raise PortfolioReturnAlignmentError("No held instrument ids were supplied.")

    mapping = map_provider_tickers_to_instrument_ids(
        instruments,
        [str(column) for column in simple_returns.columns],
        required_instrument_ids=ordered_ids,
    )
    id_to_ticker = {instrument_id: ticker for ticker, instrument_id in mapping.items()}

    end_index = pd.DatetimeIndex(pd.to_datetime(simple_returns.index))
    if end_index.has_duplicates or not end_index.is_monotonic_increasing:
        raise PortfolioReturnAlignmentError(
            "Simple-return observation dates must be unique and ascending."
        )
    full_index = pd.DatetimeIndex(pd.to_datetime(list(price_index)))
    position = {timestamp: order for order, timestamp in enumerate(full_index)}
    starts: list[pd.Timestamp] = []
    for period_end in end_index:
        order = position.get(period_end)
        if order is None or order == 0:
            raise PortfolioReturnAlignmentError(
                f"Return dated {period_end.date().isoformat()} has no preceding "
                "session in the adjusted-price index."
            )
        starts.append(full_index[order - 1])

    columns = {
        instrument_id: pd.to_numeric(
            simple_returns[id_to_ticker[instrument_id]], errors="coerce"
        ).to_numpy(dtype=float)
        for instrument_id in ordered_ids
    }
    matrix = pd.DataFrame(columns, index=end_index, columns=ordered_ids)
    matrix.index.name = "period_end"
    if (
        matrix.size == 0
        or not matrix.apply(lambda column: column.map(math.isfinite)).to_numpy().all()
    ):
        raise PortfolioReturnAlignmentError(
            "Aligned simple returns contain missing or non-finite values."
        )
    period_start = pd.Series(starts, index=end_index, name="period_start")
    return AlignedPortfolioReturns(
        matrix=matrix,
        period_start=period_start,
        market_calendar_id=str(market_calendar_id),
        return_type=ALIGNED_SIMPLE_RETURN_TYPE,
        data_snapshot_id=str(data_snapshot_id),
    )


def _aligned_exposure_vector(
    currency_exposures: Mapping[str, float],
    instrument_ids: Sequence[str],
) -> list[float]:
    vector: list[float] = []
    missing: list[str] = []
    for instrument_id in instrument_ids:
        if instrument_id not in currency_exposures:
            missing.append(instrument_id)
            continue
        value = float(currency_exposures[instrument_id])
        if not math.isfinite(value):
            raise PortfolioReturnAlignmentError(
                f"Currency exposure for {instrument_id!r} is not finite."
            )
        vector.append(value)
    if missing:
        raise PortfolioReturnAlignmentError(
            "Aligned returns are missing a currency exposure for: "
            + ", ".join(sorted(missing))
            + "."
        )
    extra = sorted(set(currency_exposures) - set(instrument_ids))
    if extra:
        raise PortfolioReturnAlignmentError(
            "Currency exposures include instruments absent from the aligned "
            "return matrix: " + ", ".join(extra) + "."
        )
    return vector


def hypothetical_pnl(
    currency_exposures: Mapping[str, float],
    aligned_returns: AlignedPortfolioReturns | pd.DataFrame,
    *,
    period_start: Mapping[pd.Timestamp, pd.Timestamp] | pd.Series | None = None,
) -> pd.DataFrame:
    """Current-book historical simulation P&L over each return interval.

    ``hypothetical_pnl[s | t] = transpose(exposure[t]) * simple_return[s]`` with
    exposures frozen at the as-of book and returns spanning historical interval
    ``s``. ``hypothetical_loss = -hypothetical_pnl``. Cash is excluded by the
    caller. A cash-only book (no instrument exposures) yields all-zero P&L.
    """
    starts: Mapping[object, object] | pd.Series | None
    if isinstance(aligned_returns, AlignedPortfolioReturns):
        matrix = aligned_returns.matrix
        starts = aligned_returns.period_start
    else:
        matrix = aligned_returns
        starts = period_start

    instrument_ids = [str(column) for column in matrix.columns]
    exposure_vector = _aligned_exposure_vector(currency_exposures, instrument_ids)
    values = matrix.to_numpy(dtype=float)
    if instrument_ids:
        pnl = [float(value) for value in values @ np.asarray(exposure_vector)]
    else:
        pnl = [0.0] * len(matrix)

    def _start(period_end: object) -> str | None:
        if starts is None:
            return None
        raw = starts[period_end]
        return pd.Timestamp(raw).date().isoformat()

    return pd.DataFrame(
        {
            "period_start": [_start(item) for item in matrix.index],
            "period_end": [
                pd.Timestamp(item).date().isoformat() for item in matrix.index
            ],
            "hypothetical_pnl": pnl,
            "hypothetical_loss": [-value for value in pnl],
        }
    )


@dataclass(frozen=True)
class ProxyExposureSnapshot:
    """One dated, immutable Phase 3 exposure snapshot in an ordered history.

    ``currency_exposures`` are beginning-of-period signed base-currency
    exposures (``instrument_id`` -> ``quantity * valuation_price``).
    """

    portfolio_snapshot_id: str
    portfolio_id: str
    exposure_snapshot_id: str
    as_of_date: date
    market_calendar_id: str
    currency_exposures: Mapping[str, float]


def proxy_realized_pnl(
    exposure_history: Sequence[ProxyExposureSnapshot],
    aligned_returns: AlignedPortfolioReturns,
    *,
    next_session: Callable[[date], date],
) -> pd.DataFrame:
    """Proxy realized P&L from dated exposures and next-session simple returns.

    ``proxy_realized_pnl[t + 1] = transpose(x_t) * simple_return[t + 1]`` where
    ``t + 1`` is the next valid session on the declared calendar. Adjusted total
    returns omit intraday position changes, fees, taxes, financing, and
    separately reconciled corporate-action cash flows; this is an educational
    outcome, not accounting P&L.

    Snapshots whose target return is unavailable (outside the sample, or
    reached across a session gap) or whose held instruments lack a return
    column are reported with a non-``realized`` ``outcome_status`` and a null
    P&L rather than being dropped.
    """
    if not exposure_history:
        raise PortfolioReturnAlignmentError("The exposure history is empty.")
    ordered = sorted(exposure_history, key=lambda snap: snap.as_of_date)
    seen_dates = [snap.as_of_date for snap in ordered]
    if len(set(seen_dates)) != len(seen_dates):
        raise PortfolioReturnAlignmentError(
            "The exposure history has duplicate as_of_date snapshots."
        )

    matrix = aligned_returns.matrix
    end_positions = {pd.Timestamp(item): item for item in matrix.index}
    rows: list[dict[str, object]] = []
    for snap in ordered:
        target_end = next_session(snap.as_of_date)
        target_end_ts = pd.Timestamp(target_end)
        exposures = dict(snap.currency_exposures)
        missing_instruments = sorted(
            instrument_id
            for instrument_id in exposures
            if instrument_id not in matrix.columns
        )
        row: dict[str, object] = {
            "portfolio_snapshot_id": snap.portfolio_snapshot_id,
            "portfolio_id": snap.portfolio_id,
            "exposure_snapshot_id": snap.exposure_snapshot_id,
            "as_of_date": snap.as_of_date.isoformat(),
            "target_period_start": snap.as_of_date.isoformat(),
            "target_period_end": target_end.isoformat(),
            "market_calendar_id": snap.market_calendar_id,
            "held_instrument_count": len(exposures),
            "missing_instruments": ";".join(missing_instruments),
            "proxy_realized_pnl": math.nan,
            "proxy_realized_loss": math.nan,
            "outcome_status": "realized",
        }
        if target_end_ts not in end_positions:
            row["outcome_status"] = "missing_target_return"
            rows.append(row)
            continue
        actual_start = pd.Timestamp(
            aligned_returns.period_start.loc[end_positions[target_end_ts]]
        )
        if actual_start.date() != snap.as_of_date:
            row["outcome_status"] = "target_return_spans_session_gap"
            rows.append(row)
            continue
        if missing_instruments:
            row["outcome_status"] = "missing_instrument_return"
            rows.append(row)
            continue
        target_row = matrix.loc[end_positions[target_end_ts]]
        realized = math.fsum(
            float(value) * float(target_row[instrument_id])
            for instrument_id, value in exposures.items()
        )
        row["proxy_realized_pnl"] = realized
        row["proxy_realized_loss"] = -realized
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "ALIGNED_SIMPLE_RETURN_TYPE",
    "AlignedPortfolioReturns",
    "ProxyExposureSnapshot",
    "align_portfolio_simple_returns",
    "compute_data_snapshot_id",
    "hypothetical_pnl",
    "map_provider_tickers_to_instrument_ids",
    "proxy_realized_pnl",
]

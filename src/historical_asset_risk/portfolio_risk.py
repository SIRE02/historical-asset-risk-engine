"""Phase 4 simple-return covariance risk and Euler contributions.

Provider-, configuration-, and file-system-independent. These estimators use
**aligned simple returns** (see :mod:`historical_asset_risk.pnl`) because exact
linear P&L aggregation requires simple-return covariance. The Phase 2
log-return ``covariance_matrix.csv`` remains a separate asset-level descriptive
artifact and is never reused here.

Conventions:

* Sample covariance only (``ddof=1``). No shrinkage, EWMA, GARCH, or DCC.
* ``portfolio_return_variance = w' Sigma w`` and
  ``portfolio_currency_variance = x' Sigma x``.
* Annualized volatility is a labeled square-root-of-time approximation.
* Negative component contributions are valid for hedges and are never clipped.
* Zero-volatility portfolios use an explicit policy and never divide by zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from historical_asset_risk.contracts import (
    PortfolioCovarianceError,
    PortfolioReturnAlignmentError,
)
from historical_asset_risk.estimation import SAMPLE_DDOF

# Portfolio variance below this magnitude is treated as zero for the
# division-by-volatility policy and for the non-negativity invariant.
VARIANCE_ZERO_TOLERANCE = 1e-18
# Reciprocal-condition-number threshold below which Sigma is flagged as
# ill-conditioned / effectively singular.
CONDITION_WARNING_THRESHOLD = 1e12


def _ordered_matrix(matrix: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    instrument_ids = [str(column) for column in matrix.columns]
    if len(set(instrument_ids)) != len(instrument_ids):
        raise PortfolioReturnAlignmentError(
            "The simple-return matrix has duplicate instrument columns."
        )
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PortfolioCovarianceError(
            "The simple-return sample contains missing or non-finite values."
        )
    return instrument_ids, values


def sample_simple_return_covariance(
    matrix: pd.DataFrame, *, ddof: int = SAMPLE_DDOF
) -> pd.DataFrame:
    """Named sample covariance of aligned simple returns (``instrument_id``)."""
    instrument_ids, values = _ordered_matrix(matrix)
    observations = values.shape[0]
    if observations < ddof + 1:
        raise PortfolioCovarianceError(
            f"Sample covariance needs at least {ddof + 1} aligned observations; "
            f"received {observations}."
        )
    covariance = np.cov(values, rowvar=False, ddof=ddof)
    covariance = np.atleast_2d(covariance)
    return pd.DataFrame(covariance, index=instrument_ids, columns=instrument_ids)


def simple_return_correlation(matrix: pd.DataFrame) -> pd.DataFrame:
    """Sibling Pearson correlation on the same aligned simple-return sample."""
    instrument_ids, values = _ordered_matrix(matrix)
    if values.shape[0] < 2:
        raise PortfolioCovarianceError(
            "Sample correlation needs at least 2 aligned observations."
        )
    standard_deviation = values.std(axis=0, ddof=SAMPLE_DDOF)
    # np.corrcoef divides by a zero standard deviation for a constant series;
    # surface that as a domain error rather than emitting NaN-laden output.
    constant = [
        instrument_ids[position]
        for position, value in enumerate(standard_deviation)
        if not value > 0.0
    ]
    if constant:
        raise PortfolioCovarianceError(
            "Correlation is undefined for constant simple-return series: "
            + ", ".join(constant)
            + "."
        )
    correlation = np.atleast_2d(np.corrcoef(values, rowvar=False))
    return pd.DataFrame(correlation, index=instrument_ids, columns=instrument_ids)


def simple_return_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument count, mean, std, min, and max of aligned simple returns."""
    instrument_ids, values = _ordered_matrix(matrix)
    frame = pd.DataFrame(values, columns=instrument_ids)
    summary = pd.DataFrame(
        {
            "observation_count": frame.count().astype(int),
            "mean_simple_return": frame.mean(),
            "standard_deviation": frame.std(ddof=SAMPLE_DDOF),
            "minimum_simple_return": frame.min(),
            "maximum_simple_return": frame.max(),
        }
    )
    summary.index.name = "instrument_id"
    return summary


def compound_simple_returns(
    matrix: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Cumulative simple return ``prod(1 + r) - 1`` over an explicit session range.

    ``start`` and ``end`` are matched against the matrix session index and are
    inclusive. This helper feeds Phase 5 historical shocks; it is not a weekly
    research panel.
    """
    instrument_ids = [str(column) for column in matrix.columns]
    index = pd.DatetimeIndex(pd.to_datetime(matrix.index))
    selected = matrix.copy()
    selected.index = index
    if start is not None:
        selected = selected.loc[selected.index >= pd.Timestamp(start)]
    if end is not None:
        selected = selected.loc[selected.index <= pd.Timestamp(end)]
    if selected.empty:
        raise PortfolioCovarianceError(
            "No aligned simple-return observations fall in the requested session "
            "range."
        )
    values = selected.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PortfolioCovarianceError(
            "The requested session range contains non-finite simple returns."
        )
    compounded = np.prod(1.0 + values, axis=0) - 1.0
    return pd.Series(compounded, index=instrument_ids, name="cumulative_simple_return")


@dataclass(frozen=True)
class CovarianceConditionReport:
    """Numerical conditioning of a simple-return covariance matrix."""

    condition_number: float
    warning: str | None


def covariance_condition_report(
    covariance: pd.DataFrame,
) -> CovarianceConditionReport:
    """Report the condition number and a singular / ill-conditioned warning."""
    matrix = covariance.to_numpy(dtype=float)
    warning: str | None = None
    try:
        condition_number = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        condition_number = math.inf
    if not math.isfinite(condition_number):
        warning = "covariance matrix is singular"
    elif condition_number > CONDITION_WARNING_THRESHOLD:
        warning = (
            f"covariance matrix is ill-conditioned (condition number "
            f"{condition_number:.3e})"
        )
    return CovarianceConditionReport(
        condition_number=condition_number, warning=warning
    )


@dataclass(frozen=True)
class PortfolioRiskFromCovariance:
    """Return- and currency-space variance, volatility, and Euler contributions."""

    instrument_ids: tuple[str, ...]
    return_variance: float
    return_volatility: float
    currency_variance: float
    currency_volatility: float
    observations_per_year: int
    annualized_return_volatility: float
    annualized_currency_volatility: float
    zero_volatility: bool
    contributions: pd.DataFrame
    concentration: dict[str, float]
    condition_number: float
    warning: str | None


def _aligned_vector(
    values: Mapping[str, float], instrument_ids: Sequence[str], label: str
) -> np.ndarray:
    missing = [key for key in instrument_ids if key not in values]
    if missing:
        raise PortfolioReturnAlignmentError(
            f"{label} is missing entries for: " + ", ".join(sorted(missing)) + "."
        )
    extra = sorted(set(values) - set(instrument_ids))
    if extra:
        raise PortfolioReturnAlignmentError(
            f"{label} includes instruments absent from the covariance matrix: "
            + ", ".join(extra)
            + "."
        )
    vector = np.array([float(values[key]) for key in instrument_ids], dtype=float)
    if not np.isfinite(vector).all():
        raise PortfolioReturnAlignmentError(f"{label} contains non-finite values.")
    return vector


def portfolio_risk_from_covariance(
    weights: Mapping[str, float],
    currency_exposures: Mapping[str, float],
    covariance: pd.DataFrame,
    *,
    observations_per_year: int,
) -> PortfolioRiskFromCovariance:
    """Aggregate simple-return covariance into portfolio risk and contributions.

    All inputs align by ``instrument_id`` using the covariance column order as
    canonical. ``weights`` are return weights ``w_i`` and ``currency_exposures``
    are signed base-currency exposures ``x_i``.
    """
    instrument_ids = [str(column) for column in covariance.columns]
    if list(covariance.index.astype(str)) != instrument_ids:
        raise PortfolioReturnAlignmentError(
            "The covariance matrix must be square and identically ordered on both "
            "axes."
        )
    sigma = covariance.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise PortfolioCovarianceError("The covariance matrix has non-finite entries.")

    weight_vector = _aligned_vector(weights, instrument_ids, "weights")
    exposure_vector = _aligned_vector(
        currency_exposures, instrument_ids, "currency_exposures"
    )

    return_variance = float(weight_vector @ sigma @ weight_vector)
    currency_variance = float(exposure_vector @ sigma @ exposure_vector)
    if return_variance < -VARIANCE_ZERO_TOLERANCE:
        raise PortfolioCovarianceError(
            f"Portfolio return variance is negative ({return_variance:.3e}); the "
            "covariance matrix is not positive semidefinite."
        )
    if currency_variance < -VARIANCE_ZERO_TOLERANCE:
        raise PortfolioCovarianceError(
            f"Portfolio currency variance is negative ({currency_variance:.3e})."
        )
    return_variance = max(return_variance, 0.0)
    currency_variance = max(currency_variance, 0.0)
    return_volatility = math.sqrt(return_variance)
    currency_volatility = math.sqrt(currency_variance)
    annualization = math.sqrt(float(observations_per_year))

    zero_volatility = return_variance <= VARIANCE_ZERO_TOLERANCE
    covariance_vector = sigma @ weight_vector
    if zero_volatility:
        marginal = np.zeros_like(weight_vector)
        component = np.zeros_like(weight_vector)
        percentage = np.zeros_like(weight_vector)
    else:
        marginal = covariance_vector / return_volatility
        component = weight_vector * marginal
        percentage = component / return_volatility

    contributions = pd.DataFrame(
        {
            "instrument_id": instrument_ids,
            "weight": weight_vector,
            "portfolio_covariance_vector": covariance_vector,
            "marginal_volatility": marginal,
            "component_volatility": component,
            "percentage_component": percentage,
        }
    )

    gross_weight = float(np.sum(np.abs(weight_vector)))
    herfindahl = (
        float(np.sum((np.abs(weight_vector) / gross_weight) ** 2))
        if gross_weight > 0.0
        else 0.0
    )
    concentration = {
        "instrument_count": float(len(instrument_ids)),
        "net_weight": float(np.sum(weight_vector)),
        "gross_weight": gross_weight,
        "max_absolute_weight": (
            float(np.max(np.abs(weight_vector))) if len(weight_vector) else 0.0
        ),
        "gross_weight_herfindahl": herfindahl,
        "effective_names": (1.0 / herfindahl) if herfindahl > 0.0 else 0.0,
    }

    condition = covariance_condition_report(covariance)
    return PortfolioRiskFromCovariance(
        instrument_ids=tuple(instrument_ids),
        return_variance=return_variance,
        return_volatility=return_volatility,
        currency_variance=currency_variance,
        currency_volatility=currency_volatility,
        observations_per_year=int(observations_per_year),
        annualized_return_volatility=return_volatility * annualization,
        annualized_currency_volatility=currency_volatility * annualization,
        zero_volatility=zero_volatility,
        contributions=contributions,
        concentration=concentration,
        condition_number=condition.condition_number,
        warning=condition.warning,
    )


__all__ = [
    "CONDITION_WARNING_THRESHOLD",
    "CovarianceConditionReport",
    "PortfolioRiskFromCovariance",
    "VARIANCE_ZERO_TOLERANCE",
    "compound_simple_returns",
    "covariance_condition_report",
    "portfolio_risk_from_covariance",
    "sample_simple_return_covariance",
    "simple_return_correlation",
    "simple_return_summary",
]

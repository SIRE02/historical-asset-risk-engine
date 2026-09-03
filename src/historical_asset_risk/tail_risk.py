"""Phase 5 historical tail-risk measurement on a portfolio-loss sample.

Provider-, configuration-, and file-system-independent. These estimators
measure the tail of a *released* Phase 4 loss sample (loss-positive:
``loss = -pnl``). They are measurements, not forecasts: no coverage tests, no
research splits, no next-session forecast records.

Conventions (roadmap sections 11.2-11.3):

* Confidence level ``alpha`` satisfies ``0.5 < alpha < 1``.
* Canonical historical VaR is the generalized-inverse empirical quantile
  ``VaR_alpha(L) = L_(ceil(n * alpha))`` on ascending losses. The Phase 2
  descriptive ``quantile_method`` never redefines this.
* Historical ES uses the same equal-mass empirical distribution with an exact
  fractional boundary weight; its upper-tail probability is exactly
  ``1 - alpha``.
* VaR is never silently floored at zero; a negative VaR stays negative.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from historical_asset_risk.estimation import MINIMUM_SAMPLE_OBSERVATIONS, SAMPLE_DDOF

_STANDARD_NORMAL = statistics.NormalDist(0.0, 1.0)

CONFIDENCE_LEVEL_MIN = 0.5
CONFIDENCE_LEVEL_MAX = 1.0
CANONICAL_QUANTILE_METHOD = "generalized_inverse_empirical_cdf"
ES_VAR_TOLERANCE = 1e-9


class TailRiskError(ValueError):
    """Raised when a tail-risk input violates a documented precondition."""


def _validate_confidence_level(confidence_level: float) -> float:
    try:
        alpha = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise TailRiskError("confidence_level must be a number.") from exc
    if not math.isfinite(alpha) or not (
        CONFIDENCE_LEVEL_MIN < alpha < CONFIDENCE_LEVEL_MAX
    ):
        raise TailRiskError(
            "confidence_level must satisfy 0.5 < alpha < 1; "
            f"received {confidence_level!r}."
        )
    return alpha


def _ascending_losses(losses: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(losses), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise TailRiskError("The loss sample must be a non-empty 1-D sequence.")
    if not np.isfinite(array).all():
        raise TailRiskError("The loss sample contains missing or non-finite values.")
    return np.sort(array, kind="stable")


def _tail_sample_warning(nominal_tail_observations: float, alpha: float) -> str | None:
    if nominal_tail_observations < 1.0:
        return (
            f"the {alpha:.4g} tail holds only "
            f"{nominal_tail_observations:.3g} expected observations; the estimate "
            "is dominated by the single most extreme loss"
        )
    if nominal_tail_observations < 10.0:
        return (
            f"the {alpha:.4g} tail holds about "
            f"{nominal_tail_observations:.3g} expected observations; treat the "
            "estimate as indicative"
        )
    return None


@dataclass(frozen=True)
class HistoricalValueAtRisk:
    """Canonical generalized-inverse empirical VaR of a loss sample."""

    confidence_level: float
    observation_count: int
    value_at_risk: float
    order_statistic_rank: int
    """1-indexed rank ``ceil(n * alpha)`` of the reported order statistic."""
    quantile_method: str
    units: str
    tail_sample_warning: str | None


def historical_var(
    losses: Iterable[float],
    confidence_level: float,
    *,
    units: str = "unspecified",
) -> HistoricalValueAtRisk:
    """Return ``VaR_alpha(L) = L_(ceil(n * alpha))`` on ascending losses."""
    alpha = _validate_confidence_level(confidence_level)
    ordered = _ascending_losses(losses)
    n = int(ordered.size)
    rank = min(max(math.ceil(n * alpha), 1), n)
    return HistoricalValueAtRisk(
        confidence_level=alpha,
        observation_count=n,
        value_at_risk=float(ordered[rank - 1]),
        order_statistic_rank=rank,
        quantile_method=CANONICAL_QUANTILE_METHOD,
        units=str(units),
        tail_sample_warning=_tail_sample_warning(n * (1.0 - alpha), alpha),
    )


@dataclass(frozen=True)
class TailContribution:
    """One order statistic's deterministic weight in the ES average."""

    order_statistic_rank: int
    loss: float
    weight: float


@dataclass(frozen=True)
class HistoricalExpectedShortfall:
    """Exact finite-sample ES from the same equal-mass empirical distribution."""

    confidence_level: float
    observation_count: int
    expected_shortfall: float
    value_at_risk: float
    nominal_tail_observations: float
    """``m = n * (1 - alpha)`` - the exact upper-tail mass in observations."""
    full_tail_count: int
    """``k = floor(m)`` - order statistics carrying full weight ``1 / m``."""
    boundary_weight: float
    """``delta = m - k`` - the fractional weight on ``L_(n - k)``."""
    contributing_observation_count: int
    tail_contributions: tuple[TailContribution, ...]
    es_ge_var: bool
    units: str
    tail_sample_warning: str | None


def historical_es(
    losses: Iterable[float],
    confidence_level: float,
    *,
    units: str = "unspecified",
) -> HistoricalExpectedShortfall:
    """Return the exact finite-sample historical ES with audit weights.

    ``m = n * (1 - alpha)``, ``k = floor(m)``, ``delta = m - k`` and
    ``ES_alpha = (sum of the largest k losses + delta * L_(n - k)) / m``.
    The reported ``tail_contributions`` weights always sum to one.
    """
    alpha = _validate_confidence_level(confidence_level)
    ordered = _ascending_losses(losses)
    n = int(ordered.size)
    m = n * (1.0 - alpha)
    k = math.floor(m)
    delta = m - k

    contributions: list[TailContribution] = []
    for rank in range(n - k + 1, n + 1):
        contributions.append(
            TailContribution(rank, float(ordered[rank - 1]), 1.0 / m)
        )
    if delta > 0.0:
        boundary_rank = n - k
        contributions.append(
            TailContribution(
                boundary_rank, float(ordered[boundary_rank - 1]), delta / m
            )
        )

    es_value = math.fsum(item.loss * item.weight for item in contributions)
    var = historical_var(ordered, alpha, units=units)
    es_ge_var = es_value + ES_VAR_TOLERANCE + abs(var.value_at_risk) * 1e-9 >= (
        var.value_at_risk
    )
    return HistoricalExpectedShortfall(
        confidence_level=alpha,
        observation_count=n,
        expected_shortfall=es_value,
        value_at_risk=var.value_at_risk,
        nominal_tail_observations=m,
        full_tail_count=k,
        boundary_weight=delta,
        contributing_observation_count=len(contributions),
        tail_contributions=tuple(contributions),
        es_ge_var=es_ge_var,
        units=str(units),
        tail_sample_warning=_tail_sample_warning(m, alpha),
    )


def _loss_moments(losses: Iterable[float]) -> tuple[np.ndarray, float, float]:
    ordered = _ascending_losses(losses)
    if ordered.size < MINIMUM_SAMPLE_OBSERVATIONS:
        raise TailRiskError(
            "The normal comparison needs at least "
            f"{MINIMUM_SAMPLE_OBSERVATIONS} loss observations for a sample "
            "standard deviation."
        )
    mean_loss = float(np.mean(ordered))
    sd_loss = float(np.std(ordered, ddof=SAMPLE_DDOF))
    return ordered, mean_loss, sd_loss


@dataclass(frozen=True)
class NormalValueAtRisk:
    """Gaussian benchmark VaR - a comparison, not a normality claim."""

    confidence_level: float
    observation_count: int
    mean_loss: float
    standard_deviation: float
    z_alpha: float
    mean_included: float
    zero_mean: float
    units: str


@dataclass(frozen=True)
class NormalExpectedShortfall:
    """Gaussian benchmark ES using ``phi(z_alpha) / (1 - alpha)``."""

    confidence_level: float
    observation_count: int
    mean_loss: float
    standard_deviation: float
    z_alpha: float
    density_at_z_alpha: float
    mean_included: float
    zero_mean: float
    units: str


def normal_var(
    losses: Iterable[float],
    confidence_level: float,
    *,
    units: str = "unspecified",
) -> NormalValueAtRisk:
    """Return the mean-included and zero-mean Gaussian VaR of the loss sample."""
    alpha = _validate_confidence_level(confidence_level)
    ordered, mean_loss, sd_loss = _loss_moments(losses)
    z_alpha = _STANDARD_NORMAL.inv_cdf(alpha)
    return NormalValueAtRisk(
        confidence_level=alpha,
        observation_count=int(ordered.size),
        mean_loss=mean_loss,
        standard_deviation=sd_loss,
        z_alpha=z_alpha,
        mean_included=mean_loss + sd_loss * z_alpha,
        zero_mean=sd_loss * z_alpha,
        units=str(units),
    )


def normal_es(
    losses: Iterable[float],
    confidence_level: float,
    *,
    units: str = "unspecified",
) -> NormalExpectedShortfall:
    """Return the mean-included and zero-mean Gaussian ES of the loss sample."""
    alpha = _validate_confidence_level(confidence_level)
    ordered, mean_loss, sd_loss = _loss_moments(losses)
    z_alpha = _STANDARD_NORMAL.inv_cdf(alpha)
    density = _STANDARD_NORMAL.pdf(z_alpha)
    factor = density / (1.0 - alpha)
    return NormalExpectedShortfall(
        confidence_level=alpha,
        observation_count=int(ordered.size),
        mean_loss=mean_loss,
        standard_deviation=sd_loss,
        z_alpha=z_alpha,
        density_at_z_alpha=density,
        mean_included=mean_loss + sd_loss * factor,
        zero_mean=sd_loss * factor,
        units=str(units),
    )


__all__ = [
    "CANONICAL_QUANTILE_METHOD",
    "CONFIDENCE_LEVEL_MAX",
    "CONFIDENCE_LEVEL_MIN",
    "ES_VAR_TOLERANCE",
    "HistoricalExpectedShortfall",
    "HistoricalValueAtRisk",
    "NormalExpectedShortfall",
    "NormalValueAtRisk",
    "TailContribution",
    "TailRiskError",
    "historical_es",
    "historical_var",
    "normal_es",
    "normal_var",
]

"""Known-value and invariant tests for Phase 5 historical VaR and ES."""

from __future__ import annotations

import statistics

import pytest

from historical_asset_risk.tail_risk import (
    TailRiskError,
    historical_es,
    historical_var,
    normal_es,
    normal_var,
)

# Ascending losses, n = 10: L_(1) = -2 ... L_(9) = 8, L_(10) = 20.
LOSSES = [3, -1, 8, 0, 5, -2, 20, 1, 4, 2]


@pytest.mark.parametrize(
    ("alpha", "expected_var", "expected_rank"),
    [
        (0.80, 5.0, 8),
        (0.85, 8.0, 9),
        (0.90, 8.0, 9),
        (0.95, 20.0, 10),
    ],
)
def test_historical_var_known_values(
    alpha: float, expected_var: float, expected_rank: int
) -> None:
    result = historical_var(LOSSES, alpha, units="base_currency")
    assert result.value_at_risk == pytest.approx(expected_var)
    assert result.order_statistic_rank == expected_rank
    assert result.observation_count == 10
    assert result.quantile_method == "generalized_inverse_empirical_cdf"
    assert result.units == "base_currency"


def test_var_is_confidence_level_monotone() -> None:
    values = [historical_var(LOSSES, a).value_at_risk for a in (0.6, 0.75, 0.9, 0.99)]
    assert values == sorted(values)


@pytest.mark.parametrize(
    ("alpha", "expected_es"),
    [
        (0.80, 14.0),  # m=2, k=2: mean(8, 20)
        (0.85, 16.0),  # m=1.5, k=1, delta=0.5: (20 + 0.5*8) / 1.5
        (0.90, 20.0),  # m=1, k=1: the single largest loss
    ],
)
def test_historical_es_known_values_with_fractional_boundary(
    alpha: float, expected_es: float
) -> None:
    result = historical_es(LOSSES, alpha)
    assert result.expected_shortfall == pytest.approx(expected_es)
    assert result.es_ge_var is True
    assert result.expected_shortfall >= historical_var(LOSSES, alpha).value_at_risk


def test_es_tail_contribution_weights_sum_to_one() -> None:
    result = historical_es(LOSSES, 0.85)
    total = sum(item.weight for item in result.tail_contributions)
    assert total == pytest.approx(1.0)
    assert result.full_tail_count == 1
    assert result.boundary_weight == pytest.approx(0.5)
    assert result.contributing_observation_count == 2
    ranks = {item.order_statistic_rank for item in result.tail_contributions}
    assert ranks == {9, 10}


def test_constant_losses_give_equal_var_and_es() -> None:
    for alpha in (0.6, 0.9, 0.99):
        assert historical_var([4.0] * 6, alpha).value_at_risk == 4.0
        assert historical_es([4.0] * 6, alpha).expected_shortfall == pytest.approx(4.0)


def test_all_gain_sample_keeps_negative_var_and_es() -> None:
    gains = [-5.0, -4.0, -3.0, -2.0, -1.0]
    var = historical_var(gains, 0.9)
    es = historical_es(gains, 0.9)
    assert var.value_at_risk == -1.0
    assert es.expected_shortfall == pytest.approx(-1.0)
    assert es.es_ge_var is True


def test_single_observation_and_small_tail_warns() -> None:
    var = historical_var([7.0], 0.95)
    es = historical_es([7.0], 0.95)
    assert var.value_at_risk == 7.0
    assert es.expected_shortfall == pytest.approx(7.0)
    assert var.tail_sample_warning is not None
    assert es.tail_sample_warning is not None


@pytest.mark.parametrize("alpha", [0.5, 1.0, 0.3, 1.5, float("nan")])
def test_confidence_level_bounds_are_enforced(alpha: float) -> None:
    with pytest.raises(TailRiskError, match="0.5 < alpha < 1"):
        historical_var(LOSSES, alpha)
    with pytest.raises(TailRiskError, match="0.5 < alpha < 1"):
        historical_es(LOSSES, alpha)


def test_normal_variants_reconcile_to_an_independent_derivation() -> None:
    losses = [-1.0, 1.0]  # mean 0, sd (ddof=1) = sqrt(2)
    alpha = 0.975
    sd = statistics.stdev(losses)
    z = statistics.NormalDist().inv_cdf(alpha)
    phi = statistics.NormalDist().pdf(z)

    var = normal_var(losses, alpha, units="return")
    es = normal_es(losses, alpha, units="return")

    assert var.z_alpha == pytest.approx(z)
    assert var.zero_mean == pytest.approx(sd * z)
    assert var.mean_included == pytest.approx(0.0 + sd * z)
    assert es.zero_mean == pytest.approx(sd * phi / (1.0 - alpha))
    assert es.mean_included == pytest.approx(sd * phi / (1.0 - alpha))
    assert es.zero_mean > var.zero_mean  # normal ES exceeds normal VaR
    assert var.units == "return"


def test_normal_variants_use_mean_loss_offset() -> None:
    losses = [2.0, 4.0, 6.0]  # mean 4, sd 2
    var = normal_var(losses, 0.95)
    assert var.mean_loss == pytest.approx(4.0)
    assert var.mean_included == pytest.approx(4.0 + var.zero_mean)


def test_normal_variants_require_two_observations() -> None:
    with pytest.raises(TailRiskError, match="at least 2 loss observations"):
        normal_var([5.0], 0.95)
    with pytest.raises(TailRiskError, match="at least 2 loss observations"):
        normal_es([5.0], 0.95)


def test_empty_and_non_finite_samples_fail() -> None:
    with pytest.raises(TailRiskError, match="non-empty"):
        historical_var([], 0.95)
    with pytest.raises(TailRiskError, match="non-finite"):
        historical_es([1.0, float("inf"), 2.0], 0.95)

"""Known-value and invariant tests for simple-return covariance risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from historical_asset_risk.contracts import (
    PortfolioCovarianceError,
    PortfolioReturnAlignmentError,
)
from historical_asset_risk.portfolio_risk import (
    compound_simple_returns,
    portfolio_risk_from_covariance,
    sample_simple_return_covariance,
    simple_return_correlation,
    simple_return_summary,
)

RETURNS = pd.DataFrame(
    {"US_A": [0.1, -0.1, 0.2], "US_B": [0.1, -0.1, 0.2]},
    index=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
)


def _cov(a: float, b: float, c: float) -> pd.DataFrame:
    return pd.DataFrame(
        [[a, b], [b, c]], index=["US_A", "US_B"], columns=["US_A", "US_B"]
    )


def test_sample_covariance_and_correlation_known_values() -> None:
    covariance = sample_simple_return_covariance(RETURNS)
    assert covariance.loc["US_A", "US_A"] == pytest.approx(0.0233333333, rel=1e-6)
    assert covariance.loc["US_A", "US_B"] == pytest.approx(0.0233333333, rel=1e-6)

    correlation = simple_return_correlation(RETURNS)
    assert correlation.loc["US_A", "US_B"] == pytest.approx(1.0)

    anti = RETURNS.assign(US_B=-RETURNS["US_A"])
    assert simple_return_correlation(anti).loc["US_A", "US_B"] == pytest.approx(-1.0)


def test_covariance_rejects_small_samples_and_constant_correlation() -> None:
    with pytest.raises(PortfolioCovarianceError, match="at least 2 aligned"):
        sample_simple_return_covariance(RETURNS.iloc[:1])
    constant = RETURNS.assign(US_B=[0.0, 0.0, 0.0])
    with pytest.raises(PortfolioCovarianceError, match="constant simple-return"):
        simple_return_correlation(constant)


def test_simple_return_summary_and_compounding() -> None:
    summary = simple_return_summary(RETURNS)
    assert summary.loc["US_A", "observation_count"] == 3
    assert summary.loc["US_A", "maximum_simple_return"] == pytest.approx(0.2)

    compounded = compound_simple_returns(RETURNS)
    assert compounded.loc["US_A"] == pytest.approx(1.1 * 0.9 * 1.2 - 1.0)
    windowed = compound_simple_returns(RETURNS, start="2024-01-04", end="2024-01-05")
    assert windowed.loc["US_A"] == pytest.approx(0.9 * 1.2 - 1.0)


def test_portfolio_risk_known_values_and_euler_reconciliation() -> None:
    result = portfolio_risk_from_covariance(
        {"US_A": 0.5, "US_B": 0.5},
        {"US_A": 1000.0, "US_B": -500.0},
        _cov(0.04, 0.0, 0.09),
        observations_per_year=252,
    )
    assert result.return_variance == pytest.approx(0.0325)
    assert result.return_volatility == pytest.approx(0.18027756377)
    assert result.currency_variance == pytest.approx(62500.0)
    assert result.currency_volatility == pytest.approx(250.0)
    assert result.annualized_return_volatility == pytest.approx(
        0.18027756377 * np.sqrt(252)
    )

    component = result.contributions["component_volatility"]
    percentage = result.contributions["percentage_component"]
    assert component.sum() == pytest.approx(result.return_volatility)
    assert percentage.sum() == pytest.approx(1.0)


def test_negative_hedge_contribution_is_not_clipped() -> None:
    result = portfolio_risk_from_covariance(
        {"US_A": 1.2, "US_B": -0.2},
        {"US_A": 1.2, "US_B": -0.2},
        _cov(0.04, 0.02, 0.09),
        observations_per_year=252,
    )
    contributions = result.contributions.set_index("instrument_id")
    assert contributions.loc["US_B", "component_volatility"] < 0.0
    assert result.contributions["component_volatility"].sum() == pytest.approx(
        result.return_volatility
    )


def test_zero_volatility_uses_policy_without_dividing() -> None:
    result = portfolio_risk_from_covariance(
        {"US_A": 0.5, "US_B": 0.5},
        {"US_A": 1.0, "US_B": 1.0},
        _cov(0.0, 0.0, 0.0),
        observations_per_year=252,
    )
    assert result.zero_volatility is True
    assert result.return_volatility == 0.0
    assert (result.contributions["marginal_volatility"] == 0.0).all()
    assert (result.contributions["component_volatility"] == 0.0).all()


def test_hedge_on_singular_covariance_is_zero_volatility_not_an_error() -> None:
    # C = A + B, so long A + long B + short C has zero true variance. Rounding
    # leaves the computed variance a few ulps either side of zero.
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.normal(0.0, 0.01, 250)
        b = rng.normal(0.0, 0.012, 250)
        covariance = sample_simple_return_covariance(
            pd.DataFrame({"US_A": a, "US_B": b, "US_C": a + b})
        )
        size = float(rng.uniform(1e5, 1e7))
        exposures = {"US_A": size, "US_B": size, "US_C": -size}
        weights = {key: value / 1e6 for key, value in exposures.items()}
        result = portfolio_risk_from_covariance(
            weights, exposures, covariance, observations_per_year=252
        )
        assert result.zero_volatility is True
        assert result.return_volatility == 0.0
        assert result.currency_volatility == 0.0
        assert (result.contributions["component_volatility"] == 0.0).all()


def test_materially_negative_variance_is_still_rejected() -> None:
    with pytest.raises(PortfolioCovarianceError, match="not positive semidefinite"):
        portfolio_risk_from_covariance(
            {"US_A": 1.0, "US_B": -1.0},
            {"US_A": 1.0, "US_B": -1.0},
            _cov(1.0, 2.0, 1.0),
            observations_per_year=252,
        )


def test_matrix_negative_only_off_the_book_direction_is_rejected() -> None:
    # Eigenvalues 3 and -1: the long-long book's variance (6) is positive, so a
    # check on that book alone would accept a matrix that is not a covariance.
    with pytest.raises(PortfolioCovarianceError, match="not positive semidefinite"):
        portfolio_risk_from_covariance(
            {"US_A": 1.0, "US_B": 1.0},
            {"US_A": 1.0, "US_B": 1.0},
            _cov(1.0, 2.0, 1.0),
            observations_per_year=252,
        )


def test_nonsymmetric_covariance_is_rejected() -> None:
    # Total variance would still reconcile, but Sigma w would not be the
    # volatility gradient, so the marginal contributions would be wrong.
    nonsymmetric = pd.DataFrame(
        [[0.04, 0.03], [-0.01, 0.09]],
        index=["US_A", "US_B"],
        columns=["US_A", "US_B"],
    )
    with pytest.raises(PortfolioCovarianceError, match="not symmetric"):
        portfolio_risk_from_covariance(
            {"US_A": 0.5, "US_B": 0.5},
            {"US_A": 50.0, "US_B": 50.0},
            nonsymmetric,
            observations_per_year=252,
        )


@pytest.mark.parametrize("observations_per_year", [0, -252, 252.0, True])
def test_annualization_factor_must_be_a_positive_integer(
    observations_per_year: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        portfolio_risk_from_covariance(
            {"US_A": 0.5, "US_B": 0.5},
            {"US_A": 50.0, "US_B": 50.0},
            _cov(0.04, 0.01, 0.09),
            observations_per_year=observations_per_year,
        )


def test_singular_covariance_is_flagged() -> None:
    result = portfolio_risk_from_covariance(
        {"US_A": 0.5, "US_B": 0.5},
        {"US_A": 1.0, "US_B": 1.0},
        _cov(0.04, 0.04, 0.04),
        observations_per_year=252,
    )
    assert result.warning is not None


def test_reordering_instruments_leaves_risk_unchanged() -> None:
    forward = portfolio_risk_from_covariance(
        {"US_A": 0.5, "US_B": 0.5},
        {"US_A": 1000.0, "US_B": -500.0},
        _cov(0.04, 0.01, 0.09),
        observations_per_year=252,
    )
    reversed_cov = _cov(0.04, 0.01, 0.09).loc[["US_B", "US_A"], ["US_B", "US_A"]]
    backward = portfolio_risk_from_covariance(
        {"US_B": 0.5, "US_A": 0.5},
        {"US_B": -500.0, "US_A": 1000.0},
        reversed_cov,
        observations_per_year=252,
    )
    assert backward.return_variance == pytest.approx(forward.return_variance)
    assert backward.currency_variance == pytest.approx(forward.currency_variance)


def test_duplicate_covariance_labels_are_rejected_not_double_counted() -> None:
    # One supplied weight would fill both "US_A" slots: 40% volatility and a
    # gross weight of 2 for a 20%-volatility, fully invested book.
    duplicated = pd.DataFrame(
        [[0.04, 0.04], [0.04, 0.04]], index=["US_A", "US_A"], columns=["US_A", "US_A"]
    )
    with pytest.raises(PortfolioReturnAlignmentError, match="duplicate instrument"):
        portfolio_risk_from_covariance(
            {"US_A": 1.0}, {"US_A": 100.0}, duplicated, observations_per_year=252
        )


def test_risk_alignment_failures_are_specific() -> None:
    with pytest.raises(PortfolioReturnAlignmentError, match="missing entries"):
        portfolio_risk_from_covariance(
            {"US_A": 1.0},
            {"US_A": 1.0, "US_B": 1.0},
            _cov(0.04, 0.0, 0.09),
            observations_per_year=252,
        )
    with pytest.raises(
        PortfolioReturnAlignmentError, match="absent from the covariance"
    ):
        portfolio_risk_from_covariance(
            {"US_A": 1.0, "US_B": 1.0, "US_C": 1.0},
            {"US_A": 1.0, "US_B": 1.0, "US_C": 1.0},
            _cov(0.04, 0.0, 0.09),
            observations_per_year=252,
        )

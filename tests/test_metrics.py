"""Sanity checks and known-value tests for quantitative calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from historical_asset_risk.correlation import (
    align_returns,
    correlation_matrix,
    extreme_correlation_pairs,
)
from historical_asset_risk.returns import (
    calculate_log_returns,
    calculate_simple_returns,
    summarize_returns,
)
from historical_asset_risk.risk_metrics import rolling_volatility, volatility_summary


@pytest.fixture
def prices() -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=8, freq="B")
    return pd.DataFrame(
        {
            "AAA": [100, 102, 101, 104, 106, 105, 108, 110],
            "BBB": [50, 49, 51, 52, 51, 53, 54, 55],
            "CCC": [80, 81, 82, 81, 83, 84, 86, 85],
        },
        index=index,
    )


def test_return_calculations_are_vectorized_and_finite(prices: pd.DataFrame) -> None:
    simple = calculate_simple_returns(prices)
    log = calculate_log_returns(prices)
    assert len(simple) < len(prices)
    assert len(log) < len(prices)
    assert simple.iloc[0, 0] == pytest.approx(0.02)
    assert log.iloc[0, 0] == pytest.approx(np.log(1.02))
    assert np.isfinite(simple.to_numpy()).all()
    assert np.isfinite(log.to_numpy()).all()


def test_return_distribution_statistics_match_hand_calculated_fixture() -> None:
    returns = pd.DataFrame({"AAA": [-0.01, 0.0, 0.01, 0.02, 0.03]})

    summary = summarize_returns(
        returns,
        quantiles=(0.25, 0.75),
        quantile_method="linear",
        downside_target=0.01,
    ).loc["AAA"]

    assert summary["median_return"] == pytest.approx(0.01)
    assert summary["quantile_0.25"] == pytest.approx(0.0)
    assert summary["quantile_0.75"] == pytest.approx(0.02)
    assert summary["sample_skewness"] == pytest.approx(0.0, abs=1e-15)
    assert summary["sample_excess_kurtosis"] == pytest.approx(-1.2)
    # sqrt(((-0.02)^2 + (-0.01)^2) / 5) = 0.01
    assert summary["downside_deviation"] == pytest.approx(0.01)

    asymmetric = summarize_returns(
        pd.DataFrame({"AAA": [0.0, 0.0, 0.0, 1.0]}), quantiles=(0.5,)
    ).loc["AAA"]
    assert asymmetric["sample_skewness"] == pytest.approx(2.0)
    assert asymmetric["sample_excess_kurtosis"] == pytest.approx(4.0)


def test_quantile_interpolation_method_is_applied_and_validated() -> None:
    returns = pd.DataFrame({"AAA": [0.0, 1.0, 2.0, 3.0]})

    linear = summarize_returns(returns, quantiles=(0.25,), quantile_method="linear")
    lower = summarize_returns(returns, quantiles=(0.25,), quantile_method="lower")

    assert linear.loc["AAA", "quantile_0.25"] == pytest.approx(0.75)
    assert lower.loc["AAA", "quantile_0.25"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="QUANTILE_METHOD"):
        summarize_returns(returns, quantile_method="unsupported")
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        summarize_returns(returns, quantiles=(0.0, 0.5))


def test_distribution_summary_rejects_insufficient_shape_sample() -> None:
    returns = pd.DataFrame({"AAA": [-0.01, 0.0, 0.01]})

    with pytest.raises(ValueError, match=r"At least 4.*AAA \(3\)"):
        summarize_returns(returns)


def test_volatility_is_sample_based_and_non_negative(prices: pd.DataFrame) -> None:
    log = calculate_log_returns(prices)
    summary = volatility_summary(log, observations_per_year=252)
    expected_daily = log.std(ddof=1)
    pd.testing.assert_series_equal(
        summary["daily_volatility"], expected_daily, check_names=False
    )
    np.testing.assert_allclose(
        summary["annualized_volatility"], expected_daily * np.sqrt(252)
    )
    assert (summary >= 0).all(axis=None)


def test_rolling_window_validation_and_values(prices: pd.DataFrame) -> None:
    log = calculate_log_returns(prices)
    rolling = rolling_volatility(log, rolling_window=3, observations_per_year=252)
    assert rolling.iloc[:2].isna().all(axis=None)
    expected = log.iloc[:3].std(ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(rolling.iloc[2], expected)
    with pytest.raises(ValueError, match="Insufficient return observations"):
        rolling_volatility(
            log.iloc[:2],
            rolling_window=3,
            observations_per_year=252,
        )


def test_rolling_volatility_supports_incomplete_initial_windows(
    prices: pd.DataFrame,
) -> None:
    log = calculate_log_returns(prices)
    rolling = rolling_volatility(
        log,
        rolling_window=4,
        observations_per_year=252,
        rolling_min_observations=2,
    )
    assert rolling.iloc[0].isna().all()
    expected = log.iloc[:2].std(ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(rolling.iloc[1], expected)


def test_correlations_satisfy_mathematical_expectations(prices: pd.DataFrame) -> None:
    log = calculate_log_returns(prices)
    aligned = align_returns(log)
    correlations = correlation_matrix(log)
    assert not aligned.isna().any(axis=None)
    assert ((correlations >= -1) & (correlations <= 1)).all(axis=None)
    np.testing.assert_allclose(np.diag(correlations), 1.0, atol=1e-12)
    highest, lowest = extreme_correlation_pairs(correlations)
    assert highest[2] >= lowest[2]
    assert highest[0] != highest[1]
    assert lowest[0] != lowest[1]


def test_invalid_prices_and_empty_alignment_raise_clear_errors() -> None:
    invalid = pd.DataFrame({"AAA": [100.0, 0.0], "BBB": [90.0, 91.0]})
    with pytest.raises(ValueError, match="positive"):
        calculate_log_returns(invalid)
    empty_alignment = pd.DataFrame({"AAA": [0.1, np.nan], "BBB": [np.nan, 0.2]})
    with pytest.raises(ValueError, match="empty"):
        align_returns(empty_alignment)

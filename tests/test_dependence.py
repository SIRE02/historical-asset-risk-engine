"""Known-value and invariant tests for covariance and rolling dependence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from historical_asset_risk.correlation import (
    covariance_matrix,
    rolling_correlation,
    rolling_covariance,
)


def test_sample_covariance_matches_hand_calculated_values() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [1.0, 2.0, 3.0],
            "BBB": [2.0, 4.0, 6.0],
            "CCC": [2.0, 0.0, 1.0],
        }
    )
    expected = pd.DataFrame(
        [
            [1.0, 2.0, -0.5],
            [2.0, 4.0, -1.0],
            [-0.5, -1.0, 1.0],
        ],
        index=returns.columns,
        columns=returns.columns,
    )

    actual = covariance_matrix(returns)

    pd.testing.assert_frame_equal(actual, expected)


def test_covariance_is_symmetric_and_positive_semidefinite() -> None:
    returns = pd.DataFrame(
        {
            "AAA": [0.01, -0.02, 0.015, 0.005, -0.01],
            "BBB": [0.02, -0.01, 0.012, -0.003, 0.004],
            "CCC": [-0.004, 0.008, -0.002, 0.01, -0.006],
        }
    )

    covariance = covariance_matrix(returns)

    np.testing.assert_allclose(covariance, covariance.T, atol=1e-15)
    assert np.linalg.eigvalsh(covariance.to_numpy()).min() >= -1e-15


def test_rolling_dependence_is_trailing_and_respects_minimum_observations() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.DataFrame(
        {"AAA": [1.0, 2.0, 4.0, 8.0, 16.0], "BBB": [2.0, 1.0, 3.0, 7.0, 15.0]},
        index=dates,
    )

    covariance = rolling_covariance(
        returns, rolling_window=3, rolling_min_observations=2
    )
    correlation = rolling_correlation(
        returns, rolling_window=3, rolling_min_observations=2
    )

    assert covariance.loc[dates[0]].isna().all(axis=None)
    assert correlation.loc[dates[0]].isna().all(axis=None)
    pd.testing.assert_frame_equal(
        covariance.loc[dates[1]], returns.iloc[:2].cov(ddof=1), check_names=False
    )
    pd.testing.assert_frame_equal(
        covariance.loc[dates[3]], returns.iloc[1:4].cov(ddof=1), check_names=False
    )
    pd.testing.assert_frame_equal(
        correlation.loc[dates[3]], returns.iloc[1:4].corr(), check_names=False
    )

    prefix = rolling_covariance(
        returns.iloc[:4], rolling_window=3, rolling_min_observations=2
    )
    pd.testing.assert_frame_equal(covariance.loc[dates[3]], prefix.loc[dates[3]])


def test_default_rolling_minimum_requires_a_complete_initial_window() -> None:
    returns = pd.DataFrame(
        {"AAA": [1.0, 2.0, 3.0], "BBB": [3.0, 2.0, 1.0]},
        index=pd.date_range("2024-01-01", periods=3),
    )

    covariance = rolling_covariance(returns, rolling_window=3)

    assert covariance.loc[returns.index[:2]].isna().all(axis=None)
    pd.testing.assert_frame_equal(
        covariance.loc[returns.index[2]], returns.cov(ddof=1), check_names=False
    )


def test_constant_series_have_zero_covariance_and_undefined_correlation() -> None:
    returns = pd.DataFrame(
        {"CONSTANT": [1.0, 1.0, 1.0, 1.0], "VARIABLE": [1.0, 2.0, 4.0, 8.0]},
        index=pd.date_range("2024-01-01", periods=4),
    )

    covariance = rolling_covariance(returns, 3, 2)
    correlation = rolling_correlation(returns, 3, 2)
    last_covariance = covariance.loc[returns.index[-1]]
    last_correlation = correlation.loc[returns.index[-1]]

    assert (last_covariance.loc["CONSTANT"] == 0).all()
    assert (last_covariance["CONSTANT"] == 0).all()
    assert last_correlation.loc["CONSTANT"].isna().all()
    assert last_correlation["CONSTANT"].isna().all()
    assert last_correlation.loc["VARIABLE", "VARIABLE"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("window", "minimum", "message"),
    [(5, 3, "Insufficient"), (3, 1, "at least 2"), (3, 4, "must not exceed")],
)
def test_rolling_parameter_and_sample_failures_are_explicit(
    window: int, minimum: int, message: str
) -> None:
    returns = pd.DataFrame({"AAA": [1.0, 2.0], "BBB": [2.0, 3.0]})
    with pytest.raises(ValueError, match=message):
        rolling_covariance(returns, window, minimum)


def test_dependence_requires_two_complete_observations() -> None:
    returns = pd.DataFrame({"AAA": [1.0], "BBB": [2.0]})
    with pytest.raises(ValueError, match="At least two complete"):
        covariance_matrix(returns)

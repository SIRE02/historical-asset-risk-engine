"""Regression tests for chart date handling."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
import pytest

from historical_asset_risk import visualizations


def test_rolling_volatility_uses_calendar_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = pd.date_range("2024-01-03", periods=7, freq="B")
    rolling = pd.DataFrame(
        {
            "SPY": [float("nan"), float("nan"), 0.18, 0.24, 0.24, 0.09, 0.01],
            "QQQ": [float("nan"), float("nan"), 0.48, 0.48, 0.47, 0.46, 0.46],
        },
        index=index,
    )
    captured: dict[str, object] = {}
    original_subplots = visualizations.plt.subplots

    def capture_subplots(*args: object, **kwargs: object) -> tuple[object, object]:
        figure, axis = original_subplots(*args, **kwargs)
        captured["axis"] = axis
        return figure, axis

    monkeypatch.setattr(visualizations.plt, "subplots", capture_subplots)

    visualizations.plot_rolling_volatility(
        rolling, tmp_path / "rolling_volatility.png", 3
    )

    axis = captured["axis"]
    plotted_dates = axis.lines[0].get_xdata(orig=False)
    assert mdates.num2date(plotted_dates[0]).date() == date(2024, 1, 3)
    assert mdates.num2date(plotted_dates[-1]).date() == date(2024, 1, 11)

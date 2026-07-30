"""Charts for historical asset volatility and correlation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_rolling_volatility(
    rolling: pd.DataFrame, output_path: Path, window: int
) -> None:
    """Save a labeled annualized rolling-volatility line chart."""
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axis = plt.subplots(figsize=(12, 7))
    dates = pd.to_datetime(rolling.index).to_pydatetime()
    for ticker in rolling.columns:
        axis.plot(dates, rolling[ticker], linewidth=1.35, label=str(ticker))
    axis.set_title(f"{window}-Day Rolling Annualized Volatility", pad=14, weight="bold")
    axis.set_xlabel("Date")
    axis.set_ylabel("Annualized volatility")
    axis.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(axis.xaxis.get_major_locator())
    )
    axis.legend(title="Ticker", frameon=True, ncols=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_correlation_heatmap(correlations: pd.DataFrame, output_path: Path) -> None:
    """Save a labeled Pearson-correlation heatmap."""
    sns.set_theme(style="white", context="notebook")
    size = max(7, 1.2 * len(correlations))
    figure, axis = plt.subplots(figsize=(size, size - 0.5))
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.7,
        cbar_kws={"label": "Pearson correlation", "shrink": 0.82},
        ax=axis,
    )
    axis.set_title("Correlation of Daily Log Returns", pad=14, weight="bold")
    axis.set_xlabel("")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

"""Historical Asset Risk Engine package."""

# Keep this source version synchronized with ``project.version`` in pyproject.toml.
# The clean-wheel packaging test enforces equality with installed distribution
# metadata. A source constant avoids stale editable-install metadata in manifests.
__version__ = "0.1.1"

from historical_asset_risk.artifacts import load_artifact
from historical_asset_risk.pnl import (
    align_portfolio_simple_returns,
    hypothetical_pnl,
    map_provider_tickers_to_instrument_ids,
    proxy_realized_pnl,
)
from historical_asset_risk.portfolio_risk import (
    compound_simple_returns,
    portfolio_risk_from_covariance,
    sample_simple_return_covariance,
    simple_return_correlation,
    simple_return_summary,
)
from historical_asset_risk.stress import apply_stress, load_stress_catalog
from historical_asset_risk.tail_risk import (
    historical_es,
    historical_var,
    normal_es,
    normal_var,
)

__all__ = [
    "__version__",
    "align_portfolio_simple_returns",
    "apply_stress",
    "compound_simple_returns",
    "historical_es",
    "historical_var",
    "hypothetical_pnl",
    "load_artifact",
    "load_stress_catalog",
    "map_provider_tickers_to_instrument_ids",
    "normal_es",
    "normal_var",
    "portfolio_risk_from_covariance",
    "proxy_realized_pnl",
    "sample_simple_return_covariance",
    "simple_return_correlation",
    "simple_return_summary",
]

"""Historical Asset Risk Engine - stable public package interface.

Everything re-exported here is a supported, versioned contract: the pure
calculation functions, their result and input dataclasses, the artifact schema
registry, and ``load_artifact``. Providers, plotting, run orchestration
(``compute_portfolio_analytics`` / ``compute_tail_analytics`` /
``cli.run_analysis``), and the command-line entry point are deliberately not
part of this surface; import them from their submodules if you need them.
"""

from __future__ import annotations

# Keep this source version synchronized with ``project.version`` in pyproject.toml.
# The clean-wheel packaging test enforces equality with installed distribution
# metadata. A source constant avoids stale editable-install metadata in manifests.
__version__ = "0.2.1"

from historical_asset_risk.artifacts import (
    ARTIFACT_SCHEMAS,
    ARTIFACT_UNITS,
    artifact_schema_inventory,
    load_artifact,
    read_cash,
    read_instrument_registry,
    read_positions,
)
from historical_asset_risk.config import AnalysisConfig, load_configuration
from historical_asset_risk.contracts import (
    ArtifactSchemaError,
    Cash,
    CurrencyValuedPosition,
    Instrument,
    InstrumentEligibilityError,
    InstrumentType,
    NonPositivePortfolioValueError,
    PortfolioCalendarError,
    PortfolioCashError,
    PortfolioCovarianceError,
    PortfolioCurrencyError,
    PortfolioCurrencyValuation,
    PortfolioDateError,
    PortfolioDuplicateError,
    PortfolioError,
    PortfolioIdentifierError,
    PortfolioPriceError,
    PortfolioQuantityError,
    PortfolioReconciliation,
    PortfolioReturnAlignmentError,
    PortfolioSchemaError,
    PortfolioTimestampError,
    PortfolioTimezoneError,
    PortfolioValuation,
    Position,
    ValidatedPortfolioSnapshot,
    ValuedPosition,
)
from historical_asset_risk.correlation import (
    correlation_matrix,
    covariance_matrix,
    extreme_correlation_pairs,
    rolling_correlation,
    rolling_covariance,
)
from historical_asset_risk.market_calendar import (
    MarketCalendar,
    resolve_market_calendar,
)
from historical_asset_risk.pnl import (
    AlignedPortfolioReturns,
    ProxyExposureSnapshot,
    align_portfolio_simple_returns,
    compute_data_snapshot_id,
    hypothetical_pnl,
    map_provider_tickers_to_instrument_ids,
    proxy_realized_pnl,
)
from historical_asset_risk.portfolio import (
    calculate_currency_valuation,
    exposure_snapshot_id,
    validate_portfolio_snapshot,
    value_portfolio,
)
from historical_asset_risk.portfolio_risk import (
    CovarianceConditionReport,
    PortfolioRiskFromCovariance,
    compound_simple_returns,
    covariance_condition_report,
    portfolio_risk_from_covariance,
    sample_simple_return_covariance,
    simple_return_correlation,
    simple_return_summary,
)
from historical_asset_risk.returns import (
    calculate_log_returns,
    calculate_simple_returns,
    summarize_returns,
)
from historical_asset_risk.risk_metrics import rolling_volatility, volatility_summary
from historical_asset_risk.stress import (
    InstrumentShock,
    StressContribution,
    StressResult,
    StressScenario,
    StressScenarioError,
    apply_stress,
    canonical_scenario_payload,
    historical_shock_vector,
    load_stress_catalog,
    scenario_content_hash,
)
from historical_asset_risk.tail_risk import (
    HistoricalExpectedShortfall,
    HistoricalValueAtRisk,
    NormalExpectedShortfall,
    NormalValueAtRisk,
    TailContribution,
    TailRiskError,
    historical_es,
    historical_var,
    normal_es,
    normal_var,
)

__all__ = [
    "ARTIFACT_SCHEMAS",
    "ARTIFACT_UNITS",
    "AlignedPortfolioReturns",
    "AnalysisConfig",
    "ArtifactSchemaError",
    "Cash",
    "CovarianceConditionReport",
    "CurrencyValuedPosition",
    "HistoricalExpectedShortfall",
    "HistoricalValueAtRisk",
    "Instrument",
    "InstrumentEligibilityError",
    "InstrumentShock",
    "InstrumentType",
    "MarketCalendar",
    "NonPositivePortfolioValueError",
    "NormalExpectedShortfall",
    "NormalValueAtRisk",
    "PortfolioCalendarError",
    "PortfolioCashError",
    "PortfolioCovarianceError",
    "PortfolioCurrencyError",
    "PortfolioCurrencyValuation",
    "PortfolioDateError",
    "PortfolioDuplicateError",
    "PortfolioError",
    "PortfolioIdentifierError",
    "PortfolioPriceError",
    "PortfolioQuantityError",
    "PortfolioReconciliation",
    "PortfolioReturnAlignmentError",
    "PortfolioRiskFromCovariance",
    "PortfolioSchemaError",
    "PortfolioTimestampError",
    "PortfolioTimezoneError",
    "PortfolioValuation",
    "Position",
    "ProxyExposureSnapshot",
    "StressContribution",
    "StressResult",
    "StressScenario",
    "StressScenarioError",
    "TailContribution",
    "TailRiskError",
    "ValidatedPortfolioSnapshot",
    "ValuedPosition",
    "__version__",
    "align_portfolio_simple_returns",
    "apply_stress",
    "artifact_schema_inventory",
    "calculate_currency_valuation",
    "calculate_log_returns",
    "calculate_simple_returns",
    "canonical_scenario_payload",
    "compound_simple_returns",
    "compute_data_snapshot_id",
    "correlation_matrix",
    "covariance_condition_report",
    "covariance_matrix",
    "exposure_snapshot_id",
    "extreme_correlation_pairs",
    "historical_es",
    "historical_shock_vector",
    "historical_var",
    "hypothetical_pnl",
    "load_artifact",
    "load_configuration",
    "load_stress_catalog",
    "map_provider_tickers_to_instrument_ids",
    "normal_es",
    "normal_var",
    "portfolio_risk_from_covariance",
    "proxy_realized_pnl",
    "read_cash",
    "read_instrument_registry",
    "read_positions",
    "resolve_market_calendar",
    "rolling_correlation",
    "rolling_covariance",
    "rolling_volatility",
    "sample_simple_return_covariance",
    "scenario_content_hash",
    "simple_return_correlation",
    "simple_return_summary",
    "summarize_returns",
    "validate_portfolio_snapshot",
    "value_portfolio",
    "volatility_summary",
]

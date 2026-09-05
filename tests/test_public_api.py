"""Freeze the intentional public package surface (Phase 6)."""

from __future__ import annotations

import historical_asset_risk as hare

_EXPECTED_PUBLIC_API = {
    # version
    "__version__",
    # artifact loading + schema registry
    "load_artifact",
    "ARTIFACT_SCHEMAS",
    "ARTIFACT_UNITS",
    "artifact_schema_inventory",
    "read_instrument_registry",
    "read_positions",
    "read_cash",
    # configuration + calendar
    "AnalysisConfig",
    "load_configuration",
    "MarketCalendar",
    "resolve_market_calendar",
    # Phase 1-2 descriptive calculations
    "calculate_simple_returns",
    "calculate_log_returns",
    "summarize_returns",
    "volatility_summary",
    "rolling_volatility",
    "correlation_matrix",
    "covariance_matrix",
    "rolling_correlation",
    "rolling_covariance",
    "extreme_correlation_pairs",
    # Phase 3 portfolio contract
    "validate_portfolio_snapshot",
    "value_portfolio",
    "calculate_currency_valuation",
    "exposure_snapshot_id",
    # Phase 4 P&L and covariance risk
    "map_provider_tickers_to_instrument_ids",
    "align_portfolio_simple_returns",
    "hypothetical_pnl",
    "proxy_realized_pnl",
    "compute_data_snapshot_id",
    "sample_simple_return_covariance",
    "simple_return_correlation",
    "simple_return_summary",
    "compound_simple_returns",
    "portfolio_risk_from_covariance",
    "covariance_condition_report",
    # Phase 5 tail risk and stress
    "historical_var",
    "historical_es",
    "normal_var",
    "normal_es",
    "apply_stress",
    "load_stress_catalog",
    "scenario_content_hash",
    "canonical_scenario_payload",
    "historical_shock_vector",
    # contract / result types
    "Instrument",
    "InstrumentType",
    "Position",
    "Cash",
    "ValidatedPortfolioSnapshot",
    "PortfolioValuation",
    "PortfolioCurrencyValuation",
    "ValuedPosition",
    "CurrencyValuedPosition",
    "PortfolioReconciliation",
    "AlignedPortfolioReturns",
    "ProxyExposureSnapshot",
    "PortfolioRiskFromCovariance",
    "CovarianceConditionReport",
    "HistoricalValueAtRisk",
    "HistoricalExpectedShortfall",
    "NormalValueAtRisk",
    "NormalExpectedShortfall",
    "TailContribution",
    "InstrumentShock",
    "StressScenario",
    "StressResult",
    "StressContribution",
    # error types
    "PortfolioError",
    "PortfolioSchemaError",
    "PortfolioDateError",
    "PortfolioDuplicateError",
    "PortfolioCurrencyError",
    "PortfolioIdentifierError",
    "PortfolioCalendarError",
    "PortfolioTimezoneError",
    "PortfolioTimestampError",
    "PortfolioPriceError",
    "PortfolioQuantityError",
    "PortfolioCashError",
    "PortfolioReturnAlignmentError",
    "PortfolioCovarianceError",
    "NonPositivePortfolioValueError",
    "InstrumentEligibilityError",
    "ArtifactSchemaError",
    "TailRiskError",
    "StressScenarioError",
}

# Infrastructure that must NOT leak into the frozen mathematical surface.
_EXPLICITLY_PRIVATE = {
    "run_analysis",
    "main",
    "compute_portfolio_analytics",
    "compute_tail_analytics",
    "PortfolioAnalyticsResult",
    "TailAnalyticsResult",
    "provider_for",
    "YahooFinanceProvider",
    "CSVProvider",
    "load_market_data",
    "build_run_manifest",
    "plot_rolling_volatility",
    "plot_correlation_heatmap",
    "portfolio_artifact_frames",
}


def test_public_api_matches_the_frozen_inventory() -> None:
    assert set(hare.__all__) == _EXPECTED_PUBLIC_API


def test_every_public_name_resolves() -> None:
    for name in hare.__all__:
        assert hasattr(hare, name), name
    # no accidental duplicates
    assert len(hare.__all__) == len(set(hare.__all__))


def test_infrastructure_is_not_in_the_public_surface() -> None:
    assert _EXPLICITLY_PRIVATE.isdisjoint(hare.__all__)


def test_version_is_reported_consistently() -> None:
    from importlib.metadata import version

    assert hare.__version__ == "0.2.0"
    assert version("historical-asset-risk-engine") == hare.__version__

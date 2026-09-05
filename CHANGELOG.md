# Changelog

All notable changes to `historical-asset-risk-engine`. The project uses
semantic versioning; while the version is `0.x`, minor bumps may add contracts
but do not change the meaning of a frozen consumer artifact.

## 0.2.0

Public package freeze. The portfolio-analytics and tail-analytics layers are
complete and their contracts are released as `experimental`.

### Added

- **Portfolio P&L, aggregation, and realizations.**
  `align_portfolio_simple_returns`, `hypothetical_pnl`, `proxy_realized_pnl`,
  `sample_simple_return_covariance`, `simple_return_correlation`,
  `simple_return_summary`, `compound_simple_returns`,
  `portfolio_risk_from_covariance` (Euler contributions, concentration),
  `map_provider_tickers_to_instrument_ids`. New CLI inputs
  `--positions-history-path` / `--cash-history-path`. New artifacts:
  `portfolio_aligned_simple_returns.csv`, `hypothetical_portfolio_pnl.csv`,
  `proxy_realized_portfolio_pnl.csv`, `risk_realizations.csv`,
  `portfolio_simple_return_covariance.csv`,
  `portfolio_simple_return_correlation.csv`, `simple_return_summary.csv`,
  `portfolio_risk_summary.csv`, `portfolio_risk_contributions.csv`,
  `portfolio_concentration_summary.csv`.
- **Tail risk and stress.** `historical_var`, `historical_es`,
  `normal_var`, `normal_es`, `apply_stress`, `load_stress_catalog`. New CLI
  inputs `--tail-risk-confidence-level` / `--tail-risk-window` /
  `--stress-catalog-path`. New artifacts: `portfolio_value_at_risk.csv`,
  `portfolio_expected_shortfall.csv`,
  `portfolio_expected_shortfall_tail_weights.csv`, `tail_risk_comparison.csv`,
  `trailing_portfolio_tail_risk.csv`, `stress_scenario_catalog.json`,
  `stress_test_results.csv`, `stress_contributions.csv`.
- Curated public API: `historical_asset_risk.__all__` now exposes the
  intentional stable surface (calculations, contract and result dataclasses,
  the artifact schema registry, `load_artifact`, `AnalysisConfig`, CSV
  readers, market calendar).
- `MarketCalendar.next_session` for one-day forward alignment.
- Per-estimator methodology pages under `docs/methodology/`.
- Offline long/short and long-only example configurations and a stress-catalog
  example under `examples/`.
- Orchestration modules `portfolio_analytics` and `tail_analytics`
  (`compute_portfolio_analytics`, `compute_tail_analytics`), deliberately
  outside the frozen public surface; import them from their submodules.
- Every emitted data artifact now carries a `run_manifest.json` schema
  declaration. `stress_scenario_catalog.json` was the last one without: it is
  registered under `historical-asset-risk/stress-scenario-catalog`, so a
  consumer can discover its identity from the manifest instead of only from
  the file's own header. `load_artifact` still accepts CSV only.

### Changed

- Frozen `v0.1.1` consumer files (`adjusted_prices.csv`, `simple_returns.csv`,
  `log_returns.csv`, `data_quality_report.json`, `run_manifest.json`) are
  unchanged. The new `run_manifest.json` sections (`portfolio_analytics`,
  `tail_analytics`) and `data_quality_report.json` sections
  (`portfolio.analytics`, `portfolio.tail`) are additive, and are
  `experimental` alongside the schemas they describe.

### Removed

- The pre-provider compatibility wrappers `data_loader.download_adjusted_prices`
  and `data_loader.validate_configuration`. Neither was reachable from the CLI
  or the public package surface, and `download_adjusted_prices` had no caller
  and no test. Acquisition goes through `providers.provider_for` and
  `load_market_data`; `AnalysisConfig` performs the validation. The remaining
  wrappers `clean_adjusted_prices` and `normalize_and_validate` are unchanged.

## 0.1.1

- Acquisition reliability and reproducibility improvements: installed VCS
  commit recorded in the run manifest, pinned `requirements.lock`, provider
  hardening.
- Risk charts and standardized README ticker examples.

## 0.1.0

- First stable foundation release: descriptive asset-risk statistics, the
  installable package and configuration system, and the portfolio contract
  (valuation, exposure, snapshot identity).

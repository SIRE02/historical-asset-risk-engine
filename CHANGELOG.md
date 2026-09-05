# Changelog

All notable changes to `historical-asset-risk-engine`. The project uses
semantic versioning; while the version is `0.x`, minor bumps may add contracts
but do not change the meaning of a frozen consumer artifact.

## Unreleased

### Changed

- **Renamed the two orchestration modules after what they compute rather than
  when they were built.** `phase4` -> `portfolio_analytics` and `phase5` ->
  `tail_analytics`, with `compute_phase4` / `Phase4Result` ->
  `compute_portfolio_analytics` / `PortfolioAnalyticsResult` and
  `compute_phase5` / `Phase5Result` -> `compute_tail_analytics` /
  `TailAnalyticsResult`. These were never part of the frozen public surface
  (`historical_asset_risk.__all__` is unchanged), so no released contract moves.
- `contracts.PHASE4_SCHEMA_VERSION` -> `PORTFOLIO_ANALYTICS_SCHEMA_VERSION` and
  `PHASE5_SCHEMA_VERSION` -> `TAIL_ANALYTICS_SCHEMA_VERSION`. The values stay
  `1.experimental`; no emitted `schema_version` value changes.
- **Experimental report keys renamed.** `run_manifest.json` sections
  `portfolio_phase4` / `portfolio_phase5` are now `portfolio_analytics` /
  `tail_analytics`, and `data_quality_report.json` sections
  `portfolio.phase4` / `portfolio.phase5` are now `portfolio.analytics` /
  `portfolio.tail`. These sections were added as `experimental` in 0.2.0; the
  frozen `v0.1.1` manifest and quality-report keys are untouched. Consumers
  reading the 0.2.0 names must update.
- Artifact filenames, schema ids, column sets, units, and loss signs are
  unchanged; every artifact is byte-identical apart from the two report files.
- Methodology pages and the CLI run summary no longer label layers by build
  phase.

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

### Changed

- Frozen `v0.1.1` consumer files (`adjusted_prices.csv`, `simple_returns.csv`,
  `log_returns.csv`, `data_quality_report.json`, `run_manifest.json`) are
  unchanged. Quality-report and manifest additions
  (`portfolio_phase4`, `portfolio_phase5`, `portfolio.phase4`,
  `portfolio.phase5`) are additive.

## 0.1.1

- Acquisition reliability and reproducibility improvements: installed VCS
  commit recorded in the run manifest, pinned `requirements.lock`, provider
  hardening.
- Risk charts and standardized README ticker examples.

## 0.1.0

- First stable foundation release: descriptive asset-risk statistics, the
  installable package and configuration system, and the portfolio contract
  (valuation, exposure, snapshot identity).

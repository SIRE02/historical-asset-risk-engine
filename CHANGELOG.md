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
  `log_returns.csv`, `data_quality_report.json`, `run_manifest.json`) keep
  their columns, units, schema ids and versions. The new `run_manifest.json`
  sections (`portfolio_analytics`, `tail_analytics`), the new
  `data_source.price_content_hash` and `git_worktree_dirty` keys, and the
  `data_quality_report.json`
  sections (`portfolio.analytics`, `portfolio.tail`) are additive, and are
  `experimental` alongside the schemas they describe.
- `data_snapshot_id` now also hashes `price_content_hash`, a SHA-256 of the
  aligned adjusted prices. Before, it covered only provider, source, date
  bounds, observation count and tickers, so a revised price under the same
  dates and source (a Yahoo dividend re-adjustment, an edited CSV) kept the id
  that realization rows join on.
- A rerun into an existing output directory removes engine-owned artifacts it
  no longer writes, and a run without a portfolio refuses a directory that
  holds a portfolio run.
- **Breaking:** relative paths written in a configuration file now resolve
  against that file's folder instead of the working directory, so a
  configuration means the same thing wherever the command runs. Command-line
  path flags still resolve against the working directory. A configuration
  whose paths were written relative to the folder it was run from must be
  updated; the bundled `examples/config.*.toml` now use `data/...` and
  `../outputs/...`, and `config.example.toml` (at the repo root) is unchanged.
- Eligibility, currency, calendar and timezone checks apply to the
  instruments a snapshot holds. Other registry rows need only unique
  identifiers, so one registry can list instruments a given book cannot hold.
- `run_manifest.json` records `git_worktree_dirty`: whether tracked files
  differed from `git_commit` (`null` when it cannot be checked).

### Fixed

- An infinite adjusted close (`inf` parses as a number) was accepted as a valid
  price, written to `adjusted_prices.csv`, and turned the next simple return
  into exactly `-1`. It is now an invalid price, counted in
  `invalid_price_values_removed`, and the alignment policy rejects the hole it
  leaves.
- Historical ES tail counts, the VaR rank and the small-tail warning are
  computed in exact rational arithmetic. Float error made `full_tail_count`,
  `boundary_weight` and `contributing_observation_count` wrong by a whole
  observation at common inputs (n = 10 at 0.80, n = 100 at 0.99), and made the
  VaR and ES warnings disagree at an exact tail count of 1 or 10.
- A perfectly hedged book on a singular covariance was rejected as not
  positive semidefinite about half the time; the variance tolerance now scales
  with `|w|' |Sigma| |w|`.
- `tail_analytics.window` in `run_manifest.json` recorded the trailing window
  beside the full-sample headline count. It is now always
  `full_aligned_sample`; the trailing window is recorded under
  `tail_analytics.trailing.window`.
- Identical tail-sample warnings were stored twice in
  `data_quality_report.json`.
- A session missing from every ticker passed alignment, because the gap check
  compares tickers against each other and none had that date. The return
  across it covered two sessions but entered P&L, covariance and VaR/ES as one
  day. Portfolio runs now require the aligned price dates to be consecutive
  sessions on the portfolio calendar (`portfolio.calendar.price_date_policy`
  in `run_manifest.json`). Returns-only runs have no calendar and keep the
  provider-date check alone.
- A duplicate date/ticker row replaced the earlier row before prices were
  validated, so a later invalid duplicate discarded a valid price and could
  cost the date its place in the aligned sample. Validity is now decided
  first, and the last valid row wins; `duplicate_date_instrument_rows_removed`
  still counts every dropped row.
- Stress shocks below `-1` are rejected. Scenario `session_start` and
  `session_end` are validated as exact `YYYY-MM-DD` dates with start no later
  than end, given both or neither, and required for historical scenarios.
- A proxy snapshot on 2035-12-31, the calendar's last session, raised instead
  of being recorded as `missing_target_return`.
- The methodology said estimation windows never use observations after
  `as_of_date`. The headline book measures replay the book over the whole
  configured sample, as the P&L and tail pages and their tests already
  required; the cross-cutting rule now says so.

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

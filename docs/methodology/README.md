# Methodology

These pages document every estimator the Historical Asset Risk Engine
publishes: its use case, formula, units, sign convention, alignment and
missing-data rules, assumptions, known limitations, and the tests that pin it.

This is a measurement package. It does not forecast returns, volatility, or
tail risk; it does not optimize or trade; it does not run risk-model coverage
diagnostics. Those belong to consuming research projects.

| Page | Scope |
| --- | --- |
| [Descriptive statistics](descriptive-statistics.md) | Returns, volatility, correlation, distribution statistics |
| [Portfolio contract](portfolio-contract.md) | Valuation, exposure, weights, snapshot identity |
| [P&L and covariance risk](pnl-and-covariance-risk.md) | Hypothetical and proxy realized P&L, simple-return covariance, Euler contributions, realization identity |
| [Tail risk and stress](tail-risk-and-stress.md) | Historical and normal VaR/ES, trailing book series, named stress |

## Cross-cutting conventions

- **Return construction.** Asset-level descriptive statistics use daily **log**
  returns. Portfolio P&L, covariance, and tail risk use daily **simple**
  returns, because currency P&L aggregates linearly as
  `exposure * simple_return`. The two return artifacts are always distinct.
- **Loss sign.** Loss is positive: `loss = -pnl`. Every output declares its
  units (return, currency, variance, volatility, correlation, probability, or
  loss).
- **Time.** `as_of_date` is a session on the portfolio's declared calendar. A
  one-day proxy or trailing step uses the next / prior valid session on that
  calendar, never civil `date ± 1`. Estimation windows never use observations
  after `as_of_date`.
- **No silent repair.** Missing, stale, or invalid data fails against an
  explicit policy. Nothing is forward-filled, proxied, or substituted quietly.
- **Schema versions.** Every schema introduced after the `v0.1.1` handoff
  carries an `experimental` version. Frozen `v0.1.1` consumer files
  (`adjusted_prices.csv`, `simple_returns.csv`, `log_returns.csv`,
  `data_quality_report.json`, `run_manifest.json`) keep their meaning, columns,
  units, schema id, and version.

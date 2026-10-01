# Portfolio P&L and simple-return covariance risk

All estimators here consume a validated book and the `simple_returns` matrix. They are pure functions (`historical_asset_risk`
public API); the CSV artifacts are optional and written only when a book is
present. `simple_returns.csv` is never rewritten.

## Aligned simple returns

`align_portfolio_simple_returns` maps the provider return columns to canonical
`instrument_id` through the registry, restricts to the held book, and records
each return's explicit interval (`period_start`, `period_end`) recovered from
the adjusted-price session index.

- **Fails on** a held instrument with no return column, a duplicate provider
  mapping, or a required id absent from the registry.
- Extra market tickers not held by the book are ignored.
- **Artifact:** `portfolio_aligned_simple_returns.csv` (long form).

## Hypothetical historical P&L

For as-of currency exposures `x_t` and a historical simple-return vector `r_s`
aligned by `instrument_id`:

```
hypothetical_pnl[s | t]  = transpose(x_t) * r_s
hypothetical_loss[s | t] = -hypothetical_pnl[s | t]
```

- **Units:** base currency. **Sign:** loss positive.
- Exposures are frozen at `t`; the return scenario spans historical interval
  `s`. This is current-portfolio historical simulation, never labeled realized
  performance. Cash has zero market return and is not in the P&L vector.
- A cash-only book produces all-zero market P&L.
- **Artifact:** `hypothetical_portfolio_pnl.csv` (frozen columns).

## Proxy realized P&L and realization identity

Given an ordered history of dated snapshots
(`--positions-history-path` / `--cash-history-path`):

```
proxy_realized_pnl[t + 1] = transpose(x_t) * simple_return[t + 1]
```

where `t + 1` is the next valid session on the declared calendar. Snapshots
whose target return is outside the sample, reached across a session gap, or
missing a held instrument's return are reported with a non-`realized`
`outcome_status` and a null P&L rather than dropped. A snapshot on the
calendar's last supported session has no resolvable `t + 1`; it is reported as
`missing_target_return` with an empty `target_period_end`.

Two statuses arise only when `proxy_realized_pnl` is called directly, because
a CLI run rejects their cause up front. `target_return_spans_session_gap`: the
CLI fails price dates that skip a session. `missing_instrument_return`: the CLI
validates every history snapshot against the selected tickers, so an
instrument the history holds but the run does not price fails the run with the
instrument named.

- Adjusted total returns omit intraday position changes, fees, taxes,
  financing, and separately reconciled corporate-action cash flows. This is an
  educational outcome, not accounting P&L.
- Each row is also a **realization record**: portfolio, exposure snapshot,
  calendar, exact target interval, units, loss sign, data snapshot, and
  `calculation_version`. A downstream forecasting engine joins its own
  predictions to these rows; HARE does not generate forecasts or run
  Kupiec / Christoffersen / pinball tests.
- **Artifacts:** `proxy_realized_portfolio_pnl.csv`, `risk_realizations.csv`
  (frozen columns).

## Simple-return covariance and companions

`sample_simple_return_covariance` is the sample covariance (`ddof = 1`) of the
aligned simple-return matrix. It is a **distinct artifact** from the
descriptive log-return `covariance_matrix.csv`: different schema id, different units
(`daily_simple_return_squared`).

- Sibling `simple_return_correlation` on the same sample; undefined for a
  constant series (fails rather than emitting NaN).
- `simple_return_summary`: count, mean, std, min, max per instrument.
- `compound_simple_returns`: `product(1 + r) - 1` over an explicit session
  range - the multi-session shock helper the stress layer uses.
- No shrinkage, EWMA, GARCH, or DCC. Sigma is never annualized or projected
  silently. A singular / ill-conditioned matrix is flagged, not repaired.
- **Artifacts:** `portfolio_simple_return_covariance.csv`,
  `portfolio_simple_return_correlation.csv`, `simple_return_summary.csv`.

## Portfolio risk aggregation (Euler contributions)

For return weights `w_t`, currency exposures `x_t`, and aligned simple-return
covariance `Sigma_t`:

```
portfolio_return_variance    = transpose(w_t) * Sigma_t * w_t
portfolio_return_volatility  = sqrt(portfolio_return_variance)
portfolio_currency_variance  = transpose(x_t) * Sigma_t * x_t
portfolio_currency_volatility = sqrt(portfolio_currency_variance)
```

For positive portfolio volatility:

```
portfolio_covariance_vector = Sigma_t * w_t
marginal_volatility[i]      = portfolio_covariance_vector[i] / portfolio_volatility
component_volatility[i]     = w[i] * marginal_volatility[i]
percentage_component[i]     = component_volatility[i] / portfolio_volatility
```

- `Sigma_t * w_t` is the covariance of each instrument return with the
  portfolio return, not "marginal variance"; the gradient of portfolio
  variance is `2 * Sigma_t * w_t`.
- Component contributions sum to portfolio volatility; percentage
  contributions sum to one when volatility is positive.
- **Negative component contributions are valid for hedges and are never
  clipped.**
- A zero-volatility portfolio uses an explicit policy: marginal, component,
  and percentage contributions are set to zero and never divide. Variance
  within `1e-12 * |w|' |Sigma| |w|` of zero (floor `1e-18`) counts as zero:
  rounding leaves a perfect hedge on a singular Sigma a few ulps either side
  of zero, not exactly there. Only variance more negative than that tolerance
  is rejected as not positive semidefinite.
- Annualized volatility is a labeled square-root-of-time approximation.
- Gross concentration summary: gross weight, Herfindahl, effective names. No
  optimizer.
- **Artifacts:** `portfolio_risk_summary.csv`,
  `portfolio_risk_contributions.csv`, `portfolio_concentration_summary.csv`.

## Tests

`tests/test_pnl.py`, `tests/test_portfolio_risk.py`,
`tests/test_portfolio_analytics_cli.py`,
`tests/test_portfolio_analytics_freeze.py`: hand-calculated
one- and two-asset P&L / Sigma / volatility / Euler; long/short with a negative
hedge; cash-only; perfect +/-1 correlation and singular Sigma; zero-volatility
no-division; missing held instrument fails while extra tickers are ignored;
instrument reordering leaves totals unchanged; next-session proxy alignment;
frozen consumer and portfolio-analytics column contracts.

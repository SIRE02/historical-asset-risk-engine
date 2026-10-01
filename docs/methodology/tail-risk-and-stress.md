# Tail risk and stress

These estimators measure the tail of the released portfolio-loss
sample and apply named shocks to the same book. They are static
bounded-sample measurements: no coverage tests, no research splits, no
next-session forecast records. One-day horizon in the initial release.

The loss sample is `-portfolio_return` (return dimension) and
`hypothetical_loss` (currency dimension) over every aligned interval in the
configured sample, including intervals that end after `as_of_date`. Both
dimensions are reported separately. `--tail-risk-window` does not shorten this
sample; it applies only to the trailing series below.

## Historical Value at Risk

For ascending ordered losses `L_(1) <= ... <= L_(n)` and `0.5 < alpha < 1`,
the canonical historical VaR is the generalized-inverse empirical quantile:

```
VaR_alpha(L) = L_(ceil(n * alpha))
```

- **Units:** loss units of the dimension. **Sign:** loss positive; a negative
  VaR stays negative and is never floored at zero.
- The descriptive `quantile_method` is an asset-level statistic and
  does **not** redefine risk VaR.
- Reports `observation_count`, the order-statistic rank, and a small-tail
  warning when the expected tail count `m = n * (1 - alpha)` is below 10. Below
  1, the warning says the estimate is dominated by the single most extreme
  loss. VaR and ES carry the same warning because they read the same `m`.
- **Artifact:** `portfolio_value_at_risk.csv` (frozen columns).

## Historical Expected Shortfall

Same equal-mass empirical distribution; upper-tail probability exactly
`1 - alpha`. Let `m = n * (1 - alpha)`, `k = floor(m)`, `delta = m - k`:

```
ES_alpha(L) = (sum of the largest k losses + delta * L_(n - k)) / m
```

- The sum is empty when `k = 0`.
- `m`, `k`, `delta`, the VaR rank `ceil(n * alpha)` and the small-tail warning
  are computed in exact rational arithmetic on `alpha` as written (`0.99` is
  `99/100`). In binary floating point `10 * (1 - 0.80)` is
  `1.9999999999999996`, so `floor` would drop a full-weight observation, and
  `100 * (1 - 0.99)` is `1.0000000000000009`, which would add a spurious
  boundary observation with weight about `9e-16`.
- The reported `tail_contributions` weights (`1 / m` on each of the `k` largest
  losses, `delta / m` on `L_(n - k)`) always sum to one and are written to
  `portfolio_expected_shortfall_tail_weights.csv` so ES is auditable.
- `ES_alpha >= VaR_alpha` is verified within tolerance under the loss-positive
  convention.
- **Artifact:** `portfolio_expected_shortfall.csv` (frozen columns).

## Normal-parametric comparison

A Gaussian benchmark, not a claim that losses are normal. With sample mean
`mean_loss`, sample standard deviation `sd_loss` (`ddof = 1`), standard-normal
quantile `z_alpha`, and density `phi`:

```
normal_var_mean_included = mean_loss + sd_loss * z_alpha
normal_es_mean_included  = mean_loss + sd_loss * phi(z_alpha) / (1 - alpha)
normal_var_zero_mean     = sd_loss * z_alpha
normal_es_zero_mean      = sd_loss * phi(z_alpha) / (1 - alpha)
```

Both the mean-included and zero-mean variants are reported. `z_alpha` and
`phi` come from `statistics.NormalDist` (no extra dependency). Student-t and
conditional distributions stay in the forecasting project.

- **Artifact:** `tail_risk_comparison.csv` - historical VaR/ES beside the four
  normal values (frozen columns).

## Trailing book VaR and ES

With an exposure history, the loss sample is rebuilt at each `as_of_date` from
the snapshot known that day:

```
historical_loss[s | t] = -transpose(x_t) * simple_return[s]
```

for intervals `s` ending no later than `t`, over the configured
`--tail-risk-window` (or all prior intervals). The series is keyed by
`as_of_date`, `exposure_snapshot_id`, window, and confidence. It carries no
`forecast_id`, research split, or `generation_mode` - it is a risk report of
dated books, not a forecast store.

- **Artifact:** `trailing_portfolio_tail_risk.csv` (frozen columns).

## Named historical and hypothetical stress

A scenario has no probability. Scenarios are authored in a catalog JSON
(`--stress-catalog-path`); each `(scenario_id, scenario_version)` pair is
immutable and its `content_hash` is re-verified on load.

- **Historical** scenarios are authored from observed simple returns over an
  explicit session range with `historical_shock_vector`:
  `cumulative_shock[i] = product(1 + simple_return[i, d]) - 1`. A run applies
  the stored, hash-verified shocks and never re-derives them, so a published
  scenario does not change when the price file does.
- **Hypothetical** scenarios are explicit `instrument_id` simple-return shocks
  `q[i]`.
- Every shock is finite and at least `-1`: a simple return cannot fall below
  a total loss.

Applied to the current as-of currency exposures:

```
scenario_pnl[i]  = x[i] * q[i]
scenario_loss[i] = -scenario_pnl[i]
```

- Every held instrument must be shocked by the scenario; a held instrument
  with no shock fails. Instrument contributions sum to the portfolio P&L.
  Zero shock gives zero P&L; a short exposure reverses a linear shock.
- Factor shocks and silent proxies are out of scope.
- **Artifacts:** `stress_scenario_catalog.json` (resolved view),
  `stress_test_results.csv`, `stress_contributions.csv` (frozen columns).

## Tests

`tests/test_tail_risk.py`, `tests/test_stress.py`,
`tests/test_tail_analytics_cli.py`, `tests/test_tail_analytics_freeze.py`:
hand-calculated
VaR / fractionally weighted ES on small ordered samples; constant, all-gain,
and single-outlier fixtures; confidence-level monotonicity; `ES >= VaR`;
normal formulas against an independent derivation; next-session trailing
alignment; zero-shock and short-exposure stress; scenario content-hash
rejection of mutated content; frozen tail-analytics column contracts.

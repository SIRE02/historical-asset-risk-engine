# Portfolio contract, valuation, and exposure

Released in `v0.1.0`. Defines exactly what a portfolio is before any P&L or
risk calculation. One immutable snapshot is processed at a time; a collection
of snapshots (P&L and tail-risk histories) reuses this same row schema.

## Input contracts

A versioned instrument registry, position rows, and exactly one explicit cash
record per snapshot. Every held instrument uses the portfolio calendar,
timezone, and base currency (no FX conversion in the initial release); registry
rows the snapshot does not hold need only unique identifiers. Valuation prices
are a separately named as-of market price - never a silently substituted
adjusted close. Cash is an explicit signed base-currency amount; zero cash is
a valid explicit record.

The engine rejects missing / non-finite quantities and prices, zero
quantities, non-positive prices, mixed currencies / calendars / timezones,
timestamps inconsistent with the session, unknown instrument ids, duplicate
identifiers, reused snapshot ids for different content, and unsupported
instrument types (including leveraged or inverse ETFs).

A portfolio run also requires the aligned adjusted-price dates to be
consecutive sessions on the portfolio calendar. A price on a non-session, or a
session missing from every instrument, fails the run: the return across that
hole would cover more than one session but be treated as one day.

## Valuation and exposure

```
position_value[i]        = quantity[i] * valuation_price[i]
net_instrument_exposure  = sum(position_value[i])
gross_exposure           = sum(abs(position_value[i]))
portfolio_value          = cash + net_instrument_exposure
```

For strictly positive portfolio value:

```
weight[i]            = position_value[i] / portfolio_value
gross_exposure_ratio = gross_exposure / portfolio_value
net_exposure_ratio   = net_instrument_exposure / portfolio_value
```

Currency-exposure summaries may be reported for any finite portfolio value;
weight-, return-, and volatility-based calculations fail explicitly for
non-positive value rather than dividing by an unstable denominator.

## Snapshot identity

After validation the engine derives a deterministic `exposure_snapshot_id`
from the schema version and canonical normalized content. Reordering
instruments does not change it; changing material content does. Scaling
quantities and cash by the same positive factor scales currency values by that
factor and leaves weights and ratios unchanged.

## Outputs

`validated_instruments.csv`, `validated_positions.csv`,
`portfolio_valuation.csv`, `portfolio_exposure_summary.csv`, plus portfolio
sections in the data-quality report and run manifest.

## Tests

`tests/test_portfolio.py`, `tests/test_portfolio_cli.py`,
`tests/test_artifacts.py`: hand-calculated long-only and long/short books,
reconciliation invariants, identity stability, scaling invariance,
non-positive-NAV failure, and every rejection path with a specific message.

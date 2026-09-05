# Descriptive asset-risk statistics

Released in `v0.1.0`. These operate on the validated adjusted-price matrix and
use daily **log** returns unless noted. All are pure functions on the public
API and are also written as CSV artifacts.

## Returns

```
simple_return(t) = P(t) / P(t - 1) - 1
log_return(t)    = log(P(t) / P(t - 1))
```

Simple and log returns are kept as separate, visibly named artifacts. Invalid
observations (non-positive prices, gaps) fail; nothing is forward-filled.
Complete-case alignment across instruments; a retained return interval that
would span a non-consecutive session pair fails rather than being treated as a
daily return.

- **Artifacts:** `simple_returns.csv`, `log_returns.csv` (frozen consumer
  files), `return_summary.csv`.

## Volatility

Sample standard deviation of daily log returns (`ddof = 1`), and its
square-root-of-time annualization:

```
annualized_volatility = daily_volatility * sqrt(observations_per_year)
```

Rolling volatility is trailing, no-look-ahead, `NaN` until
`rolling_min_observations`.

- **Artifacts:** `volatility_summary.csv`, `rolling_volatility.csv` (+ chart).

## Dependence

Pearson correlation and sample covariance of daily log returns; rolling
variants on a trailing window. Correlation is undefined for a constant series.
The highest and lowest unique correlation pairs are reported.

- **Artifacts:** `correlation_matrix.csv`, `covariance_matrix.csv`,
  `rolling_correlation.csv`, `rolling_covariance.csv` (+ heatmap). The
  log-return covariance is descriptive and is **not** reused as the
  simple-return P&L covariance.

## Distribution statistics

On daily log returns: arithmetic mean, median, min, max; configurable
empirical quantiles with an explicit interpolation method; bias-corrected
sample skewness (>= 3 observations) and excess kurtosis (>= 4 observations);
downside deviation with an explicit target and `all_non_missing_observations`
denominator. Fewer than the minimum observations fails visibly.

- **Artifact:** `return_summary.csv`.

## Tests

`tests/test_metrics.py`, `tests/test_dependence.py`, `tests/test_data_loader.py`:
known-value returns and volatility reconciliation, rolling-window boundaries,
correlation invariants, constant-series behavior, insufficient-sample failures,
gap-spanning return rejection.

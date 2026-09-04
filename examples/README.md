# Runnable examples

Every file here is synthetic. The two portfolio configurations run offline
(`provider = "csv"`) against the tiny bundled price matrix, so they reproduce
identically from the built wheel with no network access.

## Install

```
python -m pip install historical-asset-risk-engine
```

or, from a checkout:

```
python -m build
python -m pip install dist/historical_asset_risk_engine-*.whl
```

## Run

```
# Long/short book (long SPY, short QQQ) + Phase 5 tail risk and stress
historical-asset-risk --config examples/config.long_short.toml

# Long-only book
historical-asset-risk --config examples/config.long_only.toml
```

Each run writes to its own `output_dir`: the frozen return artifacts, the
Phase 3 valuation and exposure tables, the Phase 4 P&L / covariance / Euler
artifacts, and the Phase 5 VaR / ES / comparison / stress artifacts, plus the
`run_manifest.json` and `data_quality_report.json` lineage records.

A returns-only run needs no portfolio files and writes only the descriptive
and frozen-consumer set:

```
historical-asset-risk --provider csv \
  --csv-path examples/data/adjusted_prices.csv \
  --tickers SPY QQQ TLT GLD \
  --start-date 2024-01-01 --end-date 2024-02-01 \
  --rolling-window 3 --output-dir outputs/example-returns-only
```

## Files

| File | Purpose |
| --- | --- |
| `data/adjusted_prices.csv` | Synthetic adjusted-close matrix (SPY, QQQ, TLT, GLD; 8 sessions) |
| `data/instrument_registry.csv` | Canonical instrument metadata |
| `data/positions.csv` / `data/cash.csv` | Long/short snapshot |
| `data/positions_long_only.csv` / `data/cash_long_only.csv` | Long-only snapshot |
| `stress_catalog.example.json` | Two hash-verified hypothetical scenarios |
| `config.long_short.toml` / `config.long_only.toml` | Offline end-to-end runs |

See [`docs/methodology/`](../docs/methodology/README.md) for what each artifact
means.

# Runnable examples

Three configurations:

- **`../config.example.toml`** (repo root) - a four-instrument long/short book
  measured over a multi-year **live Yahoo** window. Needs network access; also
  writes `acquired_adjusted_prices.csv` so the window replays offline afterward.
- **`config.long_short.toml`** and **`config.long_only.toml`** - offline
  (`provider = "csv"`) against the tiny bundled synthetic price matrix, so they
  reproduce identically from the built wheel with no network access.

All bundled data is synthetic; the Yahoo run fetches real prices.

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
# Offline long/short book + tail risk and stress
historical-asset-risk --config examples/config.long_short.toml

# Offline long-only book
historical-asset-risk --config examples/config.long_only.toml

# Realistic multi-year run (needs network)
historical-asset-risk --config config.example.toml
```

Each run writes to its own `output_dir`: the frozen return artifacts, the
valuation and exposure tables, the P&L / covariance / Euler artifacts, and
the VaR / ES / comparison / stress artifacts, plus the
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
| `data/adjusted_prices.csv` | Synthetic adjusted-close matrix (SPY, QQQ, TLT, GLD; 8 sessions) for the offline runs |
| `data/instrument_registry.csv` | Canonical metadata for the four instruments |
| `data/positions.csv` / `data/cash.csv` | Offline long/short snapshot (SPY, QQQ) |
| `data/positions_long_only.csv` / `data/cash_long_only.csv` | Offline long-only snapshot |
| `data/positions_market.csv` / `data/cash_market.csv` | Four-instrument long/short book (long SPY + QQQ, short TLT, long GLD) for the Yahoo run |
| `stress_catalog.example.json` | Three hash-verified scenarios (two hypothetical, one historical) covering all four instruments |
| `config.long_short.toml` / `config.long_only.toml` | Offline end-to-end runs |

See [`docs/methodology/`](../docs/methodology/README.md) for what each artifact
means.

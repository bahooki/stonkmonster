# StonkModel

Pattern-specific stacked ML scanner for equities.

This app builds separate models per candlestick pattern (for example `bearish_engulfing`) and predicts whether the next bar is likely to move up or down. It supports:

- Historical OHLCV ingestion for S&P 500 symbols (or custom universe) via FMP, Polygon, or yfinance
- Pattern detection (engulfing, doji, hammer, harami, star patterns, and more)
- Large technical indicator feature set (price action + `ta` backend, plus optional `pandas-ta` enrichment)
- Optional external features:
  - Fundamental ratios (`yfinance`)
  - Politician trade flow (via normalized CSV you provide)
- Stacked ensemble training (tree models + optional LightGBM/XGBoost/CatBoost)
- Automatic feature pruning (missingness/constant/correlation + permutation-importance selection)
- Automated permutation-based feature importance per pattern model
- Per-pattern backtesting and live scanning
- FastAPI endpoints + Streamlit control center

## Architecture

- `src/stonkmodel/data`: universe + historical market data + external feature loaders
- `src/stonkmodel/features`: candlestick patterns, indicators, labels, dataset builder
- `src/stonkmodel/models`: stacked classifier + model artifact persistence + trainer
- `src/stonkmodel/backtest`: pattern-level walk-forward style evaluation
- `src/stonkmodel/scanner`: latest-bar signal generation from trained models
- `src/stonkmodel/api`: FastAPI web app
- `src/stonkmodel/ui`: Streamlit dashboard
- `scripts/`: CLI entry points

## Quick start

```bash
cd /Users/bahooki/Documents/StonkModel
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional boosted model dependencies:

```bash
pip install -e '.[boosters]'
```

Optional extra indicator pack (`pandas-ta`):

```bash
pip install -e '.[indicators]'
```

## Run the web apps

FastAPI:

```bash
uvicorn stonkmodel.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs)

Streamlit dashboard:

```bash
streamlit run src/stonkmodel/ui/dashboard.py
```

The Streamlit app includes end-to-end operations:
- Dataset build/update and registry
- Model training and result tables
- Backtesting with model/pattern filters and threshold controls
- Scanner runs with confidence and model filters
- Historical saved model registry/details/feature importance
- Pattern coverage, interval sweeps, and runtime config view

## End-to-end CLI flow

1) Build dataset

```bash
python scripts/build_dataset.py --interval 1d --years 15 --refresh-prices
```

Optional historical window slice (example keeps only 10 to 5 years ago):

```bash
python scripts/build_dataset.py --interval 1d --years 10 --years-ago-start 5 --years-ago-end 10 --refresh-prices
```

2) Train pattern models

```bash
python scripts/train_models.py --dataset-name model_dataset --interval 1d
```

3) Backtest

```bash
python scripts/run_backtest.py --dataset-name model_dataset --interval 1d
```

4) Scan latest signals

```bash
python scripts/scan_signals.py --interval 1d --top-n 50
```

5) Sweep intervals and pick the strongest one

```bash
python scripts/sweep_intervals.py --intervals 1d 1h 30m 15m --years 10
```

6) Review feature importance reports

```bash
python scripts/report_feature_importance.py --interval 1d --top-n 25
```

## Environment variables

Create `.env` (optional):

```bash
APP_ENV=dev
DEFAULT_INTERVAL=1d
HISTORY_YEARS=5
MAX_SYMBOLS=1000
MARKET_DATA_PROVIDER=auto
FUNDAMENTALS_PROVIDER=auto

# Recommended for premium feeds
FMP_API_KEY=your_fmp_key_here
FMP_BASE_URL=https://financialmodelingprep.com/stable

FORWARD_HORIZON_BARS=1
RETURN_THRESHOLD=0.0
MIN_PATTERN_COUNT=100
TOP_N_SIGNALS=50

# Optional alternative higher-resolution feed
POLYGON_API_KEY=your_key_here

# Optional custom universe
UNIVERSE_SOURCE=sp500  # or sp100
# CUSTOM_UNIVERSE_CSV=/abs/path/to/universe.csv
```

Quick FMP-first setup:

```bash
export FMP_API_KEY=your_fmp_key_here
export MARKET_DATA_PROVIDER=fmp
export FUNDAMENTALS_PROVIDER=fmp
```

## Politician trades input format

If you have a politician trade feed, normalize it to CSV:

- `symbol`
- `trade_date` (ISO date/time)
- `amount_usd`
- `side` (`buy` or `sell`)
- `politician` (optional)

Example row:

```csv
symbol,trade_date,amount_usd,side,politician
NVDA,2025-04-20,15000,buy,Jane Doe
```

Then pass it in:

```bash
python scripts/build_dataset.py --politician-trades-csv /abs/path/politician_trades.csv
```

Template files are included:

- `data/examples/politician_trades_template.csv`
- `data/examples/custom_universe_template.csv`

## Notes on "as far back / fastest interval"

- `yfinance` is used by default and has historical limits for sub-daily intervals.
- With `MARKET_DATA_PROVIDER=auto`, FMP is used first when `FMP_API_KEY` is set.
- If `POLYGON_API_KEY` is set, intraday ingestion (`1m`/`5m`/etc.) will route through Polygon.
- For full-market, deep intraday history, a paid institutional feed is usually required.
- On Python 3.14, `pandas-ta` may be unavailable depending on upstream `numba` support. The app still runs with the built-in `ta` fallback backend and engineered base indicators.

## Important

This project is for research and model development. It does not provide financial advice, and backtest performance does not guarantee future results.

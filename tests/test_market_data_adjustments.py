from __future__ import annotations

from pathlib import Path

import pandas as pd

from stonkmodel.data.market_data import DownloadSpec, MarketDataClient, ParquetMarketStore


def _client(tmp_path: Path) -> MarketDataClient:
    store = ParquetMarketStore(tmp_path / "raw")
    return MarketDataClient(store=store, market_data_provider="yfinance")


def test_normalize_ohlcv_applies_adj_close_factor(tmp_path: Path) -> None:
    client = _client(tmp_path)
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Open": [100.0, 52.0],
            "High": [101.0, 55.0],
            "Low": [99.0, 50.0],
            "Close": [100.0, 50.0],
            "Adj Close": [100.0, 25.0],
            "Volume": [1000.0, 1200.0],
        }
    )

    out = client._normalize_ohlcv(frame, "AAA")
    assert not out.empty
    assert "adj_close" in out.columns

    # Day 2 has a 0.5 adjustment factor. OHLC should be scaled into adjusted space.
    day2 = out.iloc[1]
    assert float(day2["open"]) == 26.0
    assert float(day2["high"]) == 27.5
    assert float(day2["low"]) == 25.0
    assert float(day2["close"]) == 25.0


def test_load_or_fetch_respects_year_lookback_on_cache(tmp_path: Path) -> None:
    client = _client(tmp_path)
    symbol = "AAA"
    interval = "1d"
    now_utc = pd.Timestamp.now(tz="UTC")
    dt = pd.date_range(end=now_utc.floor("D"), periods=1200, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "datetime": dt,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
            "symbol": symbol,
        }
    )
    client.store.save(symbol, interval, frame)

    out = client.load_or_fetch(DownloadSpec(symbols=[symbol], interval=interval, years=1))
    assert symbol in out
    clipped = out[symbol]
    assert not clipped.empty
    assert len(clipped) < len(frame)

    min_dt = pd.to_datetime(clipped["datetime"], utc=True, errors="coerce").min()
    expected_start = now_utc - pd.DateOffset(years=1)
    assert min_dt >= (expected_start - pd.Timedelta(days=2))

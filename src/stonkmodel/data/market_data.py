from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import errno
import gc
from pathlib import Path
import time
from typing import Callable, Iterable, Literal

import httpx
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from stonkmodel.data.fmp import FMPClient


_YF_INTERVAL_MAP = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
    "90m": "90m",
    "1h": "60m",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}

_FMP_INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "1hour",
    "1h": "1hour",
}

Provider = Literal["fmp", "polygon", "yfinance"]
ProgressCallback = Callable[[float, str], None]


@dataclass
class DownloadSpec:
    symbols: list[str]
    interval: str = "1d"
    years: int = 15


class ParquetMarketStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> Path:
        safe_interval = interval.replace("/", "_")
        return self.root / f"{symbol}_{safe_interval}.parquet"

    def save(self, symbol: str, interval: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        path = self._path(symbol, interval)
        # Explicit file-handle management avoids lingering descriptors under heavy write loops.
        try:
            with path.open("wb") as handle:
                frame.to_parquet(handle, index=False)
        except OSError as exc:
            if exc.errno != errno.EMFILE:
                raise
            # Descriptor pressure is usually transient (sockets still closing); retry once.
            gc.collect()
            time.sleep(0.2)
            with path.open("wb") as handle:
                frame.to_parquet(handle, index=False)

    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        path = self._path(symbol, interval)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)


class MarketDataClient:
    def __init__(
        self,
        store: ParquetMarketStore,
        market_data_provider: Literal["auto", "fmp", "polygon", "yfinance"] = "auto",
        fmp_api_key: str | None = None,
        fmp_base_url: str = "https://financialmodelingprep.com/stable",
        polygon_api_key: str | None = None,
        workers: int = 8,
    ) -> None:
        self.store = store
        self.market_data_provider = market_data_provider
        self.fmp_api_key = fmp_api_key
        self.fmp_base_url = fmp_base_url
        self.polygon_api_key = polygon_api_key
        self.workers = workers

    def fetch(
        self,
        spec: DownloadSpec,
        refresh: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, pd.DataFrame]:
        interval = spec.interval
        if not self._supports_any_provider(interval):
            raise ValueError(f"Unsupported interval `{interval}`")

        provider_order = self._provider_order(interval)
        if not provider_order:
            raise ValueError("No market data provider available. Configure FMP_API_KEY, POLYGON_API_KEY, or use yfinance")

        remaining = list(spec.symbols)
        out: dict[str, pd.DataFrame] = {}
        saved = 0
        total = max(1, len(spec.symbols))

        def emit(done: int, message: str) -> None:
            if progress_callback is None:
                return
            pct = max(0.0, min(100.0, (float(done) / float(total)) * 100.0))
            progress_callback(pct, message)

        emit(0, f"Fetching market data for {len(spec.symbols)} symbols")

        for provider in provider_order:
            if not remaining:
                break
            emit(len(out), f"Fetching with {provider} ({len(remaining)} symbols remaining)")
            if provider == "fmp":
                frames = self._fetch_fmp(remaining, interval, spec.years)
            elif provider == "polygon":
                frames = self._fetch_polygon(remaining, interval, spec.years)
            else:
                frames = self._fetch_yfinance(remaining, interval, spec.years)

            for symbol, frame in frames.items():
                cleaned = self._normalize_ohlcv(frame, symbol)
                if cleaned.empty:
                    continue
                out[symbol] = cleaned
                emit(len(out), f"{provider}: loaded {symbol} ({len(out)}/{total})")
                if refresh:
                    self.store.save(symbol, interval, cleaned)
                    saved += 1
                    if saved % 100 == 0:
                        gc.collect()

            remaining = [s for s in remaining if s not in out]

        # Fallback to cache for symbols not fetched this round.
        for symbol in remaining:
            cached = self.store.load(symbol, interval)
            if not cached.empty:
                out[symbol] = cached
                emit(len(out), f"Cache fallback: loaded {symbol} ({len(out)}/{total})")

        emit(len(out), f"History fetch complete ({len(out)}/{total} symbols)")

        return out

    def load_or_fetch(self, spec: DownloadSpec, progress_callback: ProgressCallback | None = None) -> dict[str, pd.DataFrame]:
        cached: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        total = max(1, len(spec.symbols))

        def emit(done: int, message: str) -> None:
            if progress_callback is None:
                return
            pct = max(0.0, min(100.0, (float(done) / float(total)) * 100.0))
            progress_callback(pct, message)

        emit(0, f"Checking cache for {len(spec.symbols)} symbols")
        for symbol in spec.symbols:
            data = self.store.load(symbol, spec.interval)
            if data.empty:
                missing.append(symbol)
            else:
                cached[symbol] = data
            emit(len(cached), f"Cache scan {len(cached)}/{total} hit(s), {len(missing)} missing")

        if not missing:
            emit(len(cached), "All symbols loaded from cache")
            return cached

        fetched = self.fetch(
            DownloadSpec(symbols=missing, interval=spec.interval, years=spec.years),
            refresh=True,
            progress_callback=(
                None
                if progress_callback is None
                else (lambda p, msg: progress_callback((len(cached) / total) * 100.0 + ((100.0 - ((len(cached) / total) * 100.0)) * (p / 100.0)), msg))
            ),
        )
        emit(len(cached) + len(fetched), f"Cache + fetch complete ({len(cached) + len(fetched)}/{total})")
        return {**cached, **fetched}

    def _provider_order(self, interval: str) -> list[Provider]:
        if self.market_data_provider == "fmp":
            return ["fmp"] if self.fmp_api_key and self._supports_fmp_interval(interval) else []
        if self.market_data_provider == "polygon":
            return ["polygon"] if self.polygon_api_key and self._supports_polygon_interval(interval) else []
        if self.market_data_provider == "yfinance":
            return ["yfinance"] if interval in _YF_INTERVAL_MAP else []

        # auto
        order: list[Provider] = []
        if self.fmp_api_key and self._supports_fmp_interval(interval):
            order.append("fmp")
        if self.polygon_api_key and self._supports_polygon_interval(interval):
            order.append("polygon")
        if interval in _YF_INTERVAL_MAP:
            order.append("yfinance")
        return order

    def _supports_any_provider(self, interval: str) -> bool:
        return interval in _YF_INTERVAL_MAP or self._supports_fmp_interval(interval) or self._supports_polygon_interval(interval)

    def _supports_fmp_interval(self, interval: str) -> bool:
        return interval in set(_FMP_INTERVAL_MAP.keys()) | {"1d"}

    def _supports_polygon_interval(self, interval: str) -> bool:
        return interval in {"1m", "5m", "15m", "30m", "60m", "1h", "1d"}

    def _fetch_fmp(self, symbols: list[str], interval: str, years: int) -> dict[str, pd.DataFrame]:
        if not self.fmp_api_key:
            return {}

        client = FMPClient(api_key=self.fmp_api_key, base_url=self.fmp_base_url)
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=365 * years)

        out: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=self._effective_workers()) as pool:
            futures = {
                pool.submit(self._fetch_fmp_symbol, client, symbol, interval, start, end): symbol for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    frame = future.result()
                except Exception:
                    continue
                if frame is not None and not frame.empty:
                    out[symbol] = frame

        return out

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
    def _fetch_fmp_symbol(
        self,
        client: FMPClient,
        symbol: str,
        interval: str,
        start,
        end,
    ) -> pd.DataFrame:
        if interval == "1d":
            rows = client.get_historical_price_eod(symbol=symbol, start=start, end=end)
        else:
            token = _FMP_INTERVAL_MAP.get(interval)
            if token is None:
                return pd.DataFrame()
            rows = client.get_historical_chart(symbol=symbol, interval_token=token, start=start, end=end)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def _fetch_yfinance(self, symbols: list[str], interval: str, years: int) -> dict[str, pd.DataFrame]:
        yf_interval = _YF_INTERVAL_MAP.get(interval, interval)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365 * years)
        out: dict[str, pd.DataFrame] = {}

        # Keep batches modest and disable internal yfinance threading to avoid fd exhaustion.
        batch_size = max(10, min(40, self.workers * 2))
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            raw = yf.download(
                tickers=" ".join(batch),
                interval=yf_interval,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=False,
                threads=False,
                progress=False,
            )
            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                for symbol in batch:
                    if symbol not in raw.columns.get_level_values(0):
                        continue
                    frame = raw[symbol].reset_index()
                    out[symbol] = frame
            else:
                # Single ticker batch collapses to flat columns.
                frame = raw.reset_index()
                if batch:
                    out[batch[0]] = frame

        return out

    def _fetch_polygon(self, symbols: list[str], interval: str, years: int) -> dict[str, pd.DataFrame]:
        if not self.polygon_api_key:
            return {}

        multiplier, timespan = self._polygon_interval(interval)
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=365 * years)

        out: dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=self._effective_workers()) as pool:
            futures = {
                pool.submit(
                    self._fetch_polygon_symbol,
                    symbol,
                    multiplier,
                    timespan,
                    start.isoformat(),
                    end.isoformat(),
                ): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    frame = future.result()
                except Exception:
                    continue
                if frame is not None and not frame.empty:
                    out[symbol] = frame

        return out

    @retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(4))
    def _fetch_polygon_symbol(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        assert self.polygon_api_key is not None
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/"
            f"{multiplier}/{timespan}/{start}/{end}"
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.polygon_api_key}
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        results = payload.get("results", [])
        if not results:
            return pd.DataFrame()

        frame = pd.DataFrame(results)
        frame = frame.rename(
            columns={
                "t": "datetime",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "n": "trade_count",
                "vw": "vwap",
            }
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"], unit="ms", utc=True)
        return frame

    def _polygon_interval(self, interval: str) -> tuple[int, str]:
        mapping = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "60m": (1, "hour"),
            "1h": (1, "hour"),
            "1d": (1, "day"),
        }
        if interval not in mapping:
            raise ValueError(f"Interval {interval} not supported by Polygon adapter")
        return mapping[interval]

    def _effective_workers(self) -> int:
        # Bound workers to avoid exhausting OS file descriptors with concurrent sockets.
        return max(1, min(self.workers, 8))

    def _normalize_ohlcv(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        if frame.empty:
            return frame

        # Align common column names across providers.
        columns = {str(c).lower(): c for c in frame.columns}
        dt_col = columns["datetime"] if "datetime" in columns else columns["date"] if "date" in columns else None
        if dt_col is None:
            for col in frame.columns:
                if str(col).lower() in {"date", "datetime"}:
                    dt_col = col
                    break
        if dt_col is None:
            return pd.DataFrame()

        rename_map = {}
        for col in frame.columns:
            lower = str(col).lower()
            if lower in {"open", "high", "low", "close", "volume", "adj close", "adj_close", "adjclose"}:
                rename_map[col] = lower.replace(" ", "_").replace("adjclose", "adj_close")
            elif col == dt_col:
                rename_map[col] = "datetime"

        out = frame.rename(columns=rename_map)
        required = ["datetime", "open", "high", "low", "close", "volume"]
        if not set(required).issubset(out.columns):
            return pd.DataFrame()

        ordered = required + (["adj_close"] if "adj_close" in out.columns else [])
        out = out[ordered]
        for col in ["open", "high", "low", "close", "volume", "adj_close"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
        out = out.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
        out["symbol"] = symbol
        out = out.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
        return out


def merge_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames.values(), ignore_index=True)


def filter_minimum_history(frames: dict[str, pd.DataFrame], min_rows: int) -> dict[str, pd.DataFrame]:
    return {symbol: frame for symbol, frame in frames.items() if len(frame) >= min_rows}


def symbols_with_data(frames: dict[str, pd.DataFrame]) -> Iterable[str]:
    for symbol, frame in frames.items():
        if not frame.empty:
            yield symbol

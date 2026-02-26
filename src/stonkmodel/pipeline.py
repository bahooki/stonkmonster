from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from stonkmodel.backtest.walk_forward import run_pattern_backtests, summarize_pattern_coverage
from stonkmodel.config import Settings
from stonkmodel.data.external_features import build_fundamental_table, engineer_politician_features, load_politician_trades
from stonkmodel.data.market_data import DownloadSpec, MarketDataClient, ParquetMarketStore, filter_minimum_history
from stonkmodel.data.universe import get_custom_symbols, get_sp100_symbols_resilient, get_sp500_symbols_resilient
from stonkmodel.features.dataset import DatasetBuilder, DatasetOptions
from stonkmodel.models.stacking import PatternModelIO
from stonkmodel.models.trainer import PatternTrainer, TrainConfig
from stonkmodel.scanner.scanner import ScanConfig, SignalScanner


@dataclass
class BuildResult:
    universe: str
    years_ago_start: int | None
    years_ago_end: int | None
    symbols_requested: int
    symbols_loaded: int
    rows: int
    dataset_path: str


class StonkService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = ParquetMarketStore(settings.raw_data_dir)
        self.market = MarketDataClient(
            store=self.store,
            market_data_provider=settings.market_data_provider,
            fmp_api_key=settings.fmp_api_key,
            fmp_base_url=settings.fmp_base_url,
            polygon_api_key=settings.polygon_api_key,
            workers=settings.request_workers,
        )
        self.dataset_builder = DatasetBuilder(settings.processed_data_dir)
        self.model_io = PatternModelIO(settings.models_dir)
        self.trainer = PatternTrainer(self.model_io)
        self.scanner = SignalScanner(self.model_io)

    def resolve_universe(self, universe: Literal["sp500", "sp100", "custom"] | None = None) -> list[str]:
        chosen = universe or self.settings.universe_source
        if chosen == "custom":
            if self.settings.custom_universe_csv is None:
                raise ValueError("universe_source=custom requires CUSTOM_UNIVERSE_CSV")
            symbols = get_custom_symbols(self.settings.custom_universe_csv)
        elif chosen == "sp100":
            symbols = get_sp100_symbols_resilient(
                limit=self.settings.max_symbols,
                fmp_api_key=self.settings.fmp_api_key,
                fmp_base_url=self.settings.fmp_base_url,
                cache_path=self.settings.processed_data_dir / "sp100_symbols_cache.csv",
            )
        else:
            symbols = get_sp500_symbols_resilient(
                limit=self.settings.max_symbols,
                fmp_api_key=self.settings.fmp_api_key,
                fmp_base_url=self.settings.fmp_base_url,
                cache_path=self.settings.processed_data_dir / "sp500_symbols_cache.csv",
            )

        return symbols[: self.settings.max_symbols]

    def load_history(
        self,
        interval: str | None = None,
        years: int | None = None,
        refresh: bool = False,
        universe: Literal["sp500", "sp100", "custom"] | None = None,
        years_ago_start: int | None = None,
        years_ago_end: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        symbols = self.resolve_universe(universe=universe)
        lookback_years = int(years) if years is not None else int(self.settings.history_years)
        if lookback_years < 1:
            lookback_years = 1
        if years_ago_end is not None:
            lookback_years = max(lookback_years, int(years_ago_end))
        spec = DownloadSpec(
            symbols=symbols,
            interval=interval or self.settings.default_interval,
            years=lookback_years,
        )
        if refresh:
            frames = self.market.fetch(spec, refresh=True)
        else:
            frames = self.market.load_or_fetch(spec)
        frames = self._filter_frames_by_years_ago_window(
            frames=frames,
            years_ago_start=years_ago_start,
            years_ago_end=years_ago_end,
        )
        frames = filter_minimum_history(frames, min_rows=120)
        return frames

    def build_dataset(
        self,
        interval: str | None = None,
        years: int | None = None,
        refresh_prices: bool = False,
        dataset_name: str = "model_dataset",
        politician_trades_csv: Path | None = None,
        universe: Literal["sp500", "sp100", "custom"] | None = None,
        years_ago_start: int | None = None,
        years_ago_end: int | None = None,
    ) -> BuildResult:
        resolved_universe = universe or self.settings.universe_source
        frames = self.load_history(
            interval=interval,
            years=years,
            refresh=refresh_prices,
            universe=resolved_universe,
            years_ago_start=years_ago_start,
            years_ago_end=years_ago_end,
        )
        symbols_requested = len(self.resolve_universe(universe=resolved_universe))

        options = DatasetOptions(
            horizon_bars=self.settings.forward_horizon_bars,
            return_threshold=self.settings.return_threshold,
            split_date=self.settings.train_test_split_date,
            politician_trades_csv=politician_trades_csv,
            include_fundamentals=True,
            include_politician_trades=politician_trades_csv is not None,
            fundamentals_provider=self.settings.fundamentals_provider,
            fmp_api_key=self.settings.fmp_api_key,
            fmp_base_url=self.settings.fmp_base_url,
            request_workers=self.settings.request_workers,
        )
        dataset = self.dataset_builder.build(frames, options)
        path = self.dataset_builder.save_dataset(dataset, name=dataset_name)

        return BuildResult(
            universe=resolved_universe,
            years_ago_start=years_ago_start,
            years_ago_end=years_ago_end,
            symbols_requested=symbols_requested,
            symbols_loaded=len(frames),
            rows=len(dataset),
            dataset_path=str(path),
        )

    def train(
        self,
        dataset_name: str = "model_dataset",
        interval: str | None = None,
        min_pattern_rows: int | None = None,
    ) -> pd.DataFrame:
        dataset = self.dataset_builder.load_dataset(dataset_name)
        if dataset.empty:
            raise RuntimeError("Dataset is empty. Run build_dataset first.")

        config = TrainConfig(
            interval=interval or self.settings.default_interval,
            horizon_bars=self.settings.forward_horizon_bars,
            min_pattern_rows=min_pattern_rows or self.settings.min_pattern_count,
        )
        return self.trainer.train_all(dataset, config)

    def backtest(
        self,
        dataset_name: str = "model_dataset",
        interval: str | None = None,
        long_threshold: float = 0.55,
        short_threshold: float = 0.45,
        fee_bps: float = 1.0,
        include_patterns: set[str] | None = None,
        include_model_files: set[str] | None = None,
    ) -> pd.DataFrame:
        dataset = self.dataset_builder.load_dataset(dataset_name)
        if dataset.empty:
            raise RuntimeError("Dataset is empty. Run build_dataset first.")

        return run_pattern_backtests(
            dataset,
            model_io=self.model_io,
            interval=interval or self.settings.default_interval,
            horizon_bars=self.settings.forward_horizon_bars,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            fee_bps=fee_bps,
            include_patterns=include_patterns,
            include_model_files=include_model_files,
        )

    def scan(
        self,
        interval: str | None = None,
        years: int = 2,
        top_n: int | None = None,
        refresh_prices: bool = True,
        politician_trades_csv: Path | None = None,
        include_patterns: set[str] | None = None,
        include_model_files: set[str] | None = None,
        min_confidence: float = 0.5,
        universe: Literal["sp500", "sp100", "custom"] | None = None,
    ) -> pd.DataFrame:
        used_interval = interval or self.settings.default_interval
        frames = self.load_history(interval=used_interval, years=years, refresh=refresh_prices, universe=universe)
        if not frames:
            return pd.DataFrame()

        fundamentals = build_fundamental_table(
            symbols=sorted(frames.keys()),
            cache_path=self.settings.processed_data_dir / "fundamentals.parquet",
            refresh=False,
            max_symbols=300,
            provider=self.settings.fundamentals_provider,
            fmp_api_key=self.settings.fmp_api_key,
            fmp_base_url=self.settings.fmp_base_url,
            request_workers=self.settings.request_workers,
        )

        politician = None
        if politician_trades_csv:
            trades = load_politician_trades(politician_trades_csv)
            politician = engineer_politician_features(trades) if not trades.empty else None

        return self.scanner.scan(
            latest_frames=frames,
            config=ScanConfig(
                interval=used_interval,
                horizon_bars=self.settings.forward_horizon_bars,
                top_n=top_n or self.settings.top_n_signals,
                min_confidence=min_confidence,
            ),
            fundamentals=fundamentals,
            politician_features=politician,
            include_patterns=include_patterns,
            include_model_files=include_model_files,
        )

    def coverage(self, dataset_name: str = "model_dataset") -> pd.DataFrame:
        dataset = self.dataset_builder.load_dataset(dataset_name)
        return summarize_pattern_coverage(dataset)

    def feature_importance(
        self,
        interval: str | None = None,
        horizon_bars: int | None = None,
        top_n_per_pattern: int = 20,
    ) -> pd.DataFrame:
        return self.model_io.get_feature_importance_summary(
            interval=interval,
            horizon_bars=horizon_bars,
            top_n_per_pattern=top_n_per_pattern,
        )

    def list_datasets(self) -> pd.DataFrame:
        return self.dataset_builder.list_datasets()

    def dataset_summary(self, dataset_name: str) -> dict[str, object]:
        return self.dataset_builder.dataset_summary(dataset_name)

    def model_registry(self, interval: str | None = None, pattern: str | None = None) -> pd.DataFrame:
        return self.model_io.get_model_registry(interval=interval, pattern=pattern)

    def model_details(self, model_file: str, top_n_importance: int = 30) -> dict[str, object]:
        details = self.model_io.get_model_details(model_file, top_n_importance=top_n_importance)
        importance = details.get("importance")
        if isinstance(importance, pd.DataFrame):
            details["importance"] = importance.to_dict(orient="records")
        return details

    def sweep_intervals(
        self,
        intervals: list[str],
        years: int | None = None,
        refresh_prices: bool = False,
        base_dataset_name: str = "model_dataset",
        politician_trades_csv: Path | None = None,
        universe: Literal["sp500", "sp100", "custom"] | None = None,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for interval in intervals:
            dataset_name = f"{base_dataset_name}_{interval.replace('/', '_')}"
            build = self.build_dataset(
                interval=interval,
                years=years,
                refresh_prices=refresh_prices,
                dataset_name=dataset_name,
                politician_trades_csv=politician_trades_csv,
                universe=universe,
            )
            train = self.train(dataset_name=dataset_name, interval=interval)
            backtest = self.backtest(dataset_name=dataset_name, interval=interval)

            rows.append(
                {
                    "interval": interval,
                    "dataset_rows": build.rows,
                    "symbols_loaded": build.symbols_loaded,
                    "models_trained": int(len(train)),
                    "best_model_roc_auc": float(train["roc_auc"].max()) if not train.empty else float("nan"),
                    "mean_model_roc_auc": float(train["roc_auc"].mean()) if not train.empty else float("nan"),
                    "best_pattern_return": float(backtest["cumulative_return"].max()) if not backtest.empty else float("nan"),
                    "mean_pattern_return": float(backtest["cumulative_return"].mean()) if not backtest.empty else float("nan"),
                    "best_pattern_sharpe": float(backtest["sharpe"].max()) if not backtest.empty else float("nan"),
                    "dataset_name": dataset_name,
                }
            )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["best_pattern_return", "best_model_roc_auc"], ascending=False)

    def settings_dict(self) -> dict[str, object]:
        return self.settings.model_dump()

    def _filter_frames_by_years_ago_window(
        self,
        frames: dict[str, pd.DataFrame],
        years_ago_start: int | None,
        years_ago_end: int | None,
    ) -> dict[str, pd.DataFrame]:
        if years_ago_start is None and years_ago_end is None:
            return frames

        start = int(years_ago_start or 0)
        end = int(years_ago_end if years_ago_end is not None else start)
        if start < 0:
            start = 0
        if end < start:
            start, end = end, start

        today = pd.Timestamp.now(tz="UTC").date()
        older_bound = (pd.Timestamp(today) - pd.DateOffset(years=end)).date()
        newer_bound = (pd.Timestamp(today) - pd.DateOffset(years=start)).date()

        out: dict[str, pd.DataFrame] = {}
        for symbol, frame in frames.items():
            if frame.empty or "datetime" not in frame.columns:
                continue
            work = frame.copy()
            dt = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
            mask = dt.dt.date.between(older_bound, newer_bound, inclusive="both")
            filtered = work.loc[mask].copy()
            if filtered.empty:
                continue
            out[symbol] = filtered
        return out

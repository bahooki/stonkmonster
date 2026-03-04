from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
from typing import Callable, Literal
import warnings

import numpy as np
import pandas as pd

from stonkmodel.backtest.walk_forward import (
    optimize_saved_model_thresholds,
    run_pattern_backtests,
    run_walk_forward_retraining_backtests,
    summarize_pattern_coverage,
)
from stonkmodel.config import Settings
from stonkmodel.data.external_features import (
    build_fundamental_table,
    engineer_politician_features,
    fetch_politician_trades_fmp,
    load_politician_trades,
)
from stonkmodel.data.macro import build_macro_feature_table
from stonkmodel.data.market_data import DownloadSpec, MarketDataClient, ParquetMarketStore, filter_minimum_history
from stonkmodel.data.universe import (
    build_static_membership_intervals,
    filter_frames_by_membership_intervals,
    get_custom_symbols,
    get_sp100_symbols_resilient,
    get_sp500_membership_intervals_resilient,
    get_sp500_symbols_resilient,
)
from stonkmodel.features.dataset import DatasetBuilder, DatasetOptions
from stonkmodel.models.stacking import PatternModelIO
from stonkmodel.models.trainer import PatternTrainer, TrainConfig
from stonkmodel.scanner.scanner import ScanConfig, SignalScanner


ProgressCallback = Callable[[float, str], None]


def _emit_progress(callback: ProgressCallback | None, pct: float, message: str) -> None:
    if callback is None:
        return
    callback(max(0.0, min(100.0, float(pct))), str(message))


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
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, pd.DataFrame]:
        _emit_progress(progress_callback, 3.0, "Resolving universe")
        resolved_universe = universe or self.settings.universe_source
        symbols = self.resolve_universe(universe=resolved_universe)
        _emit_progress(progress_callback, 8.0, f"Universe resolved ({len(symbols)} symbols)")
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
            _emit_progress(progress_callback, 10.0, "Downloading market history")
            frames = self.market.fetch(
                spec,
                refresh=True,
                progress_callback=lambda p, msg: _emit_progress(progress_callback, 10.0 + (0.55 * p), msg),
            )
        else:
            _emit_progress(progress_callback, 10.0, "Loading cached history and fetching gaps")
            frames = self.market.load_or_fetch(
                spec,
                progress_callback=lambda p, msg: _emit_progress(progress_callback, 10.0 + (0.55 * p), msg),
            )
        _emit_progress(progress_callback, 68.0, f"Loaded history for {len(frames)} symbols")
        frames = self._filter_frames_by_years_ago_window(
            frames=frames,
            years_ago_start=years_ago_start,
            years_ago_end=years_ago_end,
        )
        _emit_progress(progress_callback, 78.0, "Applied date window filter")
        frames = self._filter_frames_by_universe_membership(
            frames=frames,
            universe=resolved_universe,
            symbols=symbols,
        )
        _emit_progress(progress_callback, 90.0, "Applied universe membership filter")
        frames = filter_minimum_history(frames, min_rows=120)
        _emit_progress(progress_callback, 100.0, f"History ready ({len(frames)} symbols)")
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
        progress_callback: ProgressCallback | None = None,
    ) -> BuildResult:
        _emit_progress(progress_callback, 1.0, "Starting dataset build")
        resolved_universe = universe or self.settings.universe_source
        frames = self.load_history(
            interval=interval,
            years=years,
            refresh=refresh_prices,
            universe=resolved_universe,
            years_ago_start=years_ago_start,
            years_ago_end=years_ago_end,
            progress_callback=lambda p, msg: _emit_progress(progress_callback, 5.0 + (0.65 * p), msg),
        )
        _emit_progress(progress_callback, 72.0, "History loaded; building features and labels")
        symbols_requested = len(self.resolve_universe(universe=resolved_universe))

        options = DatasetOptions(
            horizon_bars=self.settings.forward_horizon_bars,
            return_threshold=self.settings.return_threshold,
            label_mode=self.settings.label_mode,
            split_date=self.settings.train_test_split_date,
            politician_trades_csv=politician_trades_csv,
            include_fundamentals=True,
            include_politician_trades=bool(politician_trades_csv is not None or self.settings.fmp_api_key),
            include_macro_features=True,
            fundamentals_provider=self.settings.fundamentals_provider,
            fmp_api_key=self.settings.fmp_api_key,
            fmp_base_url=self.settings.fmp_base_url,
            fred_api_key=self.settings.fred_api_key,
            request_workers=self.settings.request_workers,
        )
        dataset = self.dataset_builder.build(frames, options)
        _emit_progress(progress_callback, 95.0, "Saving dataset artifact")
        path = self.dataset_builder.save_dataset(dataset, name=dataset_name)
        _emit_progress(progress_callback, 100.0, f"Dataset build completed ({len(dataset)} rows)")

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
        model_name: str | None = None,
        progress_callback: ProgressCallback | None = None,
        parallel_patterns: int | None = None,
        fast_mode: bool = False,
        max_rows_per_pattern: int | None = None,
        include_patterns: set[str] | None = None,
        candidate_models_per_pattern: int = 1,
    ) -> pd.DataFrame:
        _emit_progress(progress_callback, 1.0, "Loading dataset")
        dataset = self.dataset_builder.load_dataset(dataset_name)
        if dataset.empty:
            raise RuntimeError("Dataset is empty. Run build_dataset first.")
        _emit_progress(progress_callback, 5.0, "Training pattern models")

        cpu_count = max(1, int(os.cpu_count() or 1))
        if parallel_patterns is None:
            auto_parallel = min(4, cpu_count)
            parallel_patterns = auto_parallel if fast_mode else 1
        parallel_patterns = max(1, int(parallel_patterns))
        model_n_jobs = 1 if parallel_patterns > 1 else -1
        if max_rows_per_pattern is None and fast_mode:
            max_rows_per_pattern = 120000
        _emit_progress(
            progress_callback,
            6.0,
            (
                "Training config: "
                f"parallel_patterns={parallel_patterns}, "
                f"model_n_jobs={model_n_jobs}, "
                f"fast_mode={fast_mode}, "
                f"max_rows_per_pattern={max_rows_per_pattern}, "
                f"candidate_models_per_pattern={candidate_models_per_pattern}"
            ),
        )

        config = TrainConfig(
            interval=interval or self.settings.default_interval,
            horizon_bars=self.settings.forward_horizon_bars,
            min_pattern_rows=min_pattern_rows or self.settings.min_pattern_count,
            model_name=model_name,
            parallel_patterns=parallel_patterns,
            model_n_jobs=model_n_jobs,
            fast_mode=fast_mode,
            max_rows_per_pattern=max_rows_per_pattern,
            enable_permutation_importance=not fast_mode,
            enable_timeseries_cv=not fast_mode,
            include_patterns=include_patterns,
            candidate_random_states=tuple([42, 1337, 7, 2024, 99, 314][: max(1, int(candidate_models_per_pattern))]),
            transaction_cost_bps=2.0,
            purged_cv_embargo_bars=1,
            meta_filter_enabled=True,
            meta_filter_threshold=0.55,
        )
        table = self.trainer.train_all(
            dataset,
            config,
            progress_callback=lambda p, msg: _emit_progress(progress_callback, 5.0 + (0.95 * p), msg),
        )
        _emit_progress(progress_callback, 100.0, f"Training complete ({len(table)} models)")
        return table

    def optimize_model_thresholds_from_backtest(
        self,
        dataset_name: str = "model_dataset",
        interval: str | None = None,
        fee_bps: float = 1.0,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
        short_borrow_bps_per_day: float = 0.0,
        latency_bars: int = 1,
        embargo_bars: int = 1,
        include_patterns: set[str] | None = None,
        include_model_files: set[str] | None = None,
        min_trades: int = 40,
        max_eval_rows_per_pattern: int | None = None,
        parallel_models: int = 4,
        persist: bool = True,
    ) -> pd.DataFrame:
        dataset = self.dataset_builder.load_dataset(dataset_name)
        if dataset.empty:
            raise RuntimeError("Dataset is empty. Run build_dataset first.")

        used_interval = interval or self.settings.default_interval
        return optimize_saved_model_thresholds(
            dataset=dataset,
            model_io=self.model_io,
            interval=used_interval,
            horizon_bars=self.settings.forward_horizon_bars,
            fee_bps=float(fee_bps),
            spread_bps=float(spread_bps),
            slippage_bps=float(slippage_bps),
            short_borrow_bps_per_day=float(short_borrow_bps_per_day),
            latency_bars=int(latency_bars),
            embargo_bars=int(embargo_bars),
            include_patterns=include_patterns,
            include_model_files=include_model_files,
            min_trades=int(min_trades),
            max_eval_rows_per_pattern=max_eval_rows_per_pattern,
            parallel_models=int(parallel_models),
            persist=bool(persist),
        )

    def recursive_train_from_backtest(
        self,
        dataset_name: str = "model_dataset",
        interval: str | None = None,
        min_pattern_rows: int | None = None,
        base_model_name: str | None = None,
        rounds: int = 2,
        keep_top_patterns: int = 6,
        min_trades_to_keep: int = 50,
        parallel_patterns: int | None = None,
        fast_mode: bool = False,
        max_rows_per_pattern: int | None = None,
        candidate_models_per_pattern: int = 1,
        fee_bps: float = 1.0,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
        short_borrow_bps_per_day: float = 0.0,
        latency_bars: int = 1,
        min_threshold_opt_trades: int = 40,
        max_eval_rows_per_pattern: int | None = 250000,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        used_interval = interval or self.settings.default_interval
        total_rounds = max(1, int(rounds))
        selected_patterns: set[str] | None = None
        history_rows: list[dict[str, object]] = []
        best_objective = float("-inf")
        non_improve_rounds = 0

        for round_idx in range(1, total_rounds + 1):
            round_pct_start = ((round_idx - 1) / total_rounds) * 100.0
            _emit_progress(progress_callback, round_pct_start + 1.0, f"Round {round_idx}/{total_rounds}: training models")
            round_model_name = f"{base_model_name}_r{round_idx}" if base_model_name else f"recursive_r{round_idx}"
            train_table = self.train(
                dataset_name=dataset_name,
                interval=used_interval,
                min_pattern_rows=min_pattern_rows,
                model_name=round_model_name,
                progress_callback=lambda p, msg: _emit_progress(
                    progress_callback,
                    round_pct_start + 1.0 + (35.0 * (p / 100.0)),
                    f"Round {round_idx}: {msg}",
                ),
                parallel_patterns=parallel_patterns,
                fast_mode=fast_mode,
                max_rows_per_pattern=max_rows_per_pattern,
                include_patterns=selected_patterns,
                candidate_models_per_pattern=candidate_models_per_pattern,
            )
            if train_table.empty:
                history_rows.append(
                    {
                        "round": int(round_idx),
                        "models_trained": 0,
                        "best_cumulative_return": np.nan,
                        "mean_cumulative_return": np.nan,
                        "objective_score": np.nan,
                        "patterns_kept_next": 0,
                    }
                )
                break

            round_model_files = set(train_table["model_file"].dropna().astype(str).tolist())
            _emit_progress(progress_callback, round_pct_start + 38.0, f"Round {round_idx}: optimizing thresholds")
            threshold_table = self.optimize_model_thresholds_from_backtest(
                dataset_name=dataset_name,
                interval=used_interval,
                fee_bps=float(fee_bps),
                spread_bps=float(spread_bps),
                slippage_bps=float(slippage_bps),
                short_borrow_bps_per_day=float(short_borrow_bps_per_day),
                latency_bars=int(latency_bars),
                embargo_bars=1,
                include_model_files=round_model_files,
                include_patterns=selected_patterns,
                min_trades=int(min_threshold_opt_trades),
                max_eval_rows_per_pattern=max_eval_rows_per_pattern,
                parallel_models=max(1, int(parallel_patterns or 1)),
                persist=True,
            )

            _emit_progress(progress_callback, round_pct_start + 62.0, f"Round {round_idx}: scoring backtest")
            bt = self.backtest(
                dataset_name=dataset_name,
                interval=used_interval,
                mode="saved_models",
                fee_bps=float(fee_bps),
                use_model_thresholds=True,
                spread_bps=float(spread_bps),
                slippage_bps=float(slippage_bps),
                short_borrow_bps_per_day=float(short_borrow_bps_per_day),
                latency_bars=int(latency_bars),
                include_model_files=round_model_files,
                include_patterns=selected_patterns,
                parallel_models=max(1, int(parallel_patterns or 1)),
                max_eval_rows_per_pattern=max_eval_rows_per_pattern,
                include_portfolio=False,
            )
            bt_work = bt.copy()
            if not bt_work.empty and "pattern" in bt_work.columns:
                bt_work = bt_work.loc[bt_work["pattern"].astype(str) != "portfolio_combined"].copy()
            if not bt_work.empty:
                bt_work["cumulative_return"] = pd.to_numeric(bt_work["cumulative_return"], errors="coerce")
                bt_work["sharpe"] = pd.to_numeric(bt_work.get("sharpe"), errors="coerce")
                bt_work["max_drawdown"] = pd.to_numeric(bt_work.get("max_drawdown"), errors="coerce")
                bt_work["trades"] = pd.to_numeric(bt_work.get("trades"), errors="coerce")
                best_cum = float(bt_work["cumulative_return"].max())
                mean_cum = float(bt_work["cumulative_return"].mean())
                objective = float(
                    bt_work["cumulative_return"].mean()
                    + (0.20 * bt_work["sharpe"].replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())
                    - (0.15 * bt_work["max_drawdown"].abs().replace([np.inf, -np.inf], np.nan).fillna(0.0).mean())
                )

                ranked = bt_work.sort_values(["cumulative_return", "sharpe"], ascending=[False, False])
                ranked = ranked.loc[ranked["trades"] >= int(min_trades_to_keep)]
                keep_n = max(1, int(keep_top_patterns))
                kept = ranked.head(keep_n)["pattern"].dropna().astype(str).tolist()
                selected_patterns = set(kept) if kept else selected_patterns
            else:
                best_cum = float("nan")
                mean_cum = float("nan")
                objective = float("-inf")

            history_rows.append(
                {
                    "round": int(round_idx),
                    "models_trained": int(len(train_table)),
                    "thresholds_updated": int(threshold_table["threshold_updated"].sum()) if not threshold_table.empty else 0,
                    "best_cumulative_return": best_cum,
                    "mean_cumulative_return": mean_cum,
                    "objective_score": objective if np.isfinite(objective) else np.nan,
                    "patterns_kept_next": int(len(selected_patterns or [])),
                    "kept_patterns": ",".join(sorted(selected_patterns)) if selected_patterns else "",
                }
            )

            if np.isfinite(objective) and objective > best_objective:
                best_objective = objective
                non_improve_rounds = 0
            else:
                non_improve_rounds += 1
            if non_improve_rounds >= 2:
                _emit_progress(progress_callback, round_pct_start + 95.0, f"Round {round_idx}: early stop (no objective improvement)")
                break

            _emit_progress(progress_callback, ((round_idx) / total_rounds) * 100.0, f"Round {round_idx} complete")

        _emit_progress(progress_callback, 100.0, "Recursive optimization complete")
        return pd.DataFrame(history_rows)

    @staticmethod
    def _objective_row_from_backtest(
        backtest_table: pd.DataFrame,
        min_trades: int | None = None,
    ) -> dict[str, object] | None:
        if backtest_table.empty:
            return None
        work = backtest_table.copy()
        if "model_file" not in work.columns and "pattern" not in work.columns:
            return None

        for col in ("cumulative_return", "sharpe", "max_drawdown", "win_rate_trade", "win_rate", "trades"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
            else:
                work[col] = np.nan
        if "model_file" not in work.columns:
            work["model_file"] = ""
        if "pattern" not in work.columns:
            work["pattern"] = ""
        work["model_file"] = work["model_file"].astype(str)
        work["pattern"] = work["pattern"].astype(str)

        # Prefer combined portfolio rows, then fall back to best individual row.
        preferred = work.loc[
            work["model_file"].str.startswith("portfolio_")
            | work["pattern"].str.startswith("portfolio_")
        ].copy()
        candidate = preferred if not preferred.empty else work.copy()
        if min_trades is not None and int(min_trades) > 0:
            candidate = candidate.loc[candidate["trades"] >= int(min_trades)].copy()
        if candidate.empty:
            return None

        candidate["objective_score"] = (
            candidate["cumulative_return"].replace([np.inf, -np.inf], np.nan).fillna(-1e9)
            + (0.30 * candidate["sharpe"].replace([np.inf, -np.inf], np.nan).fillna(0.0))
            + (0.20 * candidate["win_rate_trade"].replace([np.inf, -np.inf], np.nan).fillna(candidate["win_rate"].fillna(0.0)))
            - (0.20 * candidate["max_drawdown"].abs().replace([np.inf, -np.inf], np.nan).fillna(0.0))
            - (0.0002 * candidate["trades"].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        )
        picked = candidate.sort_values(["objective_score", "cumulative_return"], ascending=[False, False]).iloc[0].to_dict()
        return picked

    @staticmethod
    def _top_patterns_from_backtest(
        backtest_table: pd.DataFrame,
        top_n: int = 8,
        min_trades: int | None = None,
    ) -> set[str]:
        if backtest_table.empty or "pattern" not in backtest_table.columns:
            return set()
        work = backtest_table.copy()
        work["pattern"] = work["pattern"].astype(str)
        work = work.loc[
            ~work["pattern"].str.startswith("portfolio_")
            & ~work["pattern"].str.startswith("spread_")
            & (work["pattern"] != "")
        ].copy()
        if work.empty:
            return set()
        for col in ("cumulative_return", "sharpe", "trades"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
            else:
                work[col] = np.nan
        if min_trades is not None and int(min_trades) > 0:
            work = work.loc[work["trades"] >= int(min_trades)].copy()
        if work.empty:
            return set()
        work["rank_score"] = (
            work["cumulative_return"].replace([np.inf, -np.inf], np.nan).fillna(-1e9)
            + (0.20 * work["sharpe"].replace([np.inf, -np.inf], np.nan).fillna(0.0))
            - (0.0001 * work["trades"].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        )
        return set(work.sort_values("rank_score", ascending=False).head(max(1, int(top_n)))["pattern"].dropna().astype(str).tolist())

    def auto_improve(
        self,
        dataset_name: str = "model_dataset",
        interval: str | None = None,
        iterations: int = 8,
        max_minutes: int = 180,
        patience: int = 3,
        min_significant_improvement: float = 0.10,
        min_iteration_trades: int = 40,
        fee_bps: float = 1.0,
        spread_bps: float = 0.5,
        slippage_bps: float = 0.5,
        short_borrow_bps_per_day: float = 0.0,
        latency_bars: int = 1,
        parallel_patterns: int = 4,
        include_spread_strategies: bool = True,
        random_seed: int = 42,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        dataset = self.dataset_builder.load_dataset(dataset_name)
        if dataset.empty:
            raise RuntimeError("Dataset is empty. Run build_dataset first.")

        used_interval = interval or self.settings.default_interval
        max_iters = max(1, int(iterations))
        max_wall_seconds = max(60, int(max_minutes) * 60)
        patience_limit = max(1, int(patience))
        rng = np.random.default_rng(int(random_seed))

        # Realism floor.
        realistic_fee = max(0.0, float(fee_bps))
        realistic_spread = max(0.0, float(spread_bps))
        realistic_slippage = max(0.0, float(slippage_bps))
        realistic_latency = max(1, int(latency_bars))
        min_trades_gate = max(0, int(min_iteration_trades))
        dataset_rows = int(len(dataset))
        memory_safe_mode = dataset_rows >= 750_000
        very_large_mode = dataset_rows >= 1_500_000
        effective_parallel_patterns = max(1, int(parallel_patterns))
        if memory_safe_mode:
            cap = 2 if very_large_mode else 3
            effective_parallel_patterns = min(effective_parallel_patterns, cap)
            if effective_parallel_patterns < max(1, int(parallel_patterns)):
                _emit_progress(
                    progress_callback,
                    1.0,
                    (
                        "Auto-improve memory-safe mode: "
                        f"dataset_rows={dataset_rows}, parallel_patterns downshifted to {effective_parallel_patterns}"
                    ),
                )
        effective_include_spreads = bool(include_spread_strategies) and not very_large_mode
        if bool(include_spread_strategies) and not effective_include_spreads:
            _emit_progress(
                progress_callback,
                1.0,
                "Auto-improve memory-safe mode: spread overlays disabled for very large dataset",
            )

        started = datetime.now(timezone.utc)
        history: list[dict[str, object]] = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        _emit_progress(progress_callback, 2.0, "Auto-improve: baseline strict-OOS backtest")
        try:
            baseline_bt = self.backtest(
                dataset_name=dataset_name,
                interval=used_interval,
                mode="saved_models",
                fee_bps=realistic_fee,
                spread_bps=realistic_spread,
                slippage_bps=realistic_slippage,
                short_borrow_bps_per_day=float(short_borrow_bps_per_day),
                latency_bars=realistic_latency,
                use_model_thresholds=True,
                include_portfolio=True,
                include_spread_strategies=effective_include_spreads,
                parallel_models=effective_parallel_patterns,
                max_eval_rows_per_pattern=250000,
            )
        except Exception as exc:
            baseline_bt = pd.DataFrame()
            history = [
                {
                    "stage": "baseline_error",
                    "iteration": 0,
                    "status": "error",
                    "error": str(exc),
                }
            ]
            out = pd.DataFrame(history)
            out_path = self.settings.models_dir / f"autoloop_history_{stamp}.parquet"
            if not out.empty:
                out.to_parquet(out_path, index=False)
            _emit_progress(progress_callback, 100.0, f"Auto-improve failed during baseline backtest: {exc}")
            return out
        baseline_pick = self._objective_row_from_backtest(
            baseline_bt,
            min_trades=min_trades_gate if min_trades_gate > 0 else None,
        )
        baseline_obj = float(baseline_pick.get("objective_score", float("-inf"))) if baseline_pick else float("-inf")
        best_obj = baseline_obj
        best_patterns: set[str] = self._top_patterns_from_backtest(
            baseline_bt,
            top_n=8,
            min_trades=min_trades_gate if min_trades_gate > 0 else None,
        )
        best_iteration = 0
        no_improve = 0

        def is_significant_improvement(candidate_obj: float, incumbent_obj: float) -> bool:
            if not np.isfinite(candidate_obj):
                return False
            if not np.isfinite(incumbent_obj):
                return True
            rel = max(0.0, float(min_significant_improvement))
            min_required_delta = max(1e-9, abs(float(incumbent_obj)) * rel)
            return (float(candidate_obj) - float(incumbent_obj)) > min_required_delta

        history.append(
            {
                "stage": "baseline",
                "iteration": 0,
                "objective_score": baseline_obj if np.isfinite(baseline_obj) else np.nan,
                "cumulative_return": float(baseline_pick.get("cumulative_return", np.nan)) if baseline_pick else np.nan,
                "sharpe": float(baseline_pick.get("sharpe", np.nan)) if baseline_pick else np.nan,
                "max_drawdown": float(baseline_pick.get("max_drawdown", np.nan)) if baseline_pick else np.nan,
                "trades": float(baseline_pick.get("trades", np.nan)) if baseline_pick else np.nan,
                "selected_patterns": ",".join(sorted(best_patterns)),
                "min_iteration_trades": int(min_trades_gate),
            }
        )

        for i in range(1, max_iters + 1):
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            if elapsed > max_wall_seconds:
                history.append(
                    {
                        "stage": "stop",
                        "iteration": i,
                        "reason": "max_minutes_exceeded",
                    }
                )
                break

            if no_improve >= patience_limit:
                history.append(
                    {
                        "stage": "stop",
                        "iteration": i,
                        "reason": "patience_exceeded",
                    }
                )
                break

            pct = 5.0 + (80.0 * (i - 1) / max(1, max_iters))
            _emit_progress(progress_callback, pct, f"Auto-improve iteration {i}/{max_iters}: sampling train/backtest config")

            if very_large_mode:
                cfg_fast = True
                cfg_candidates = int(rng.choice([1, 2], p=[0.75, 0.25]))
                cfg_min_rows = int(rng.choice([80, 120, 180], p=[0.35, 0.45, 0.20]))
                cfg_max_rows = int(rng.choice([80000, 120000, 180000, 250000], p=[0.25, 0.35, 0.25, 0.15]))
            elif memory_safe_mode:
                cfg_fast = bool(rng.choice([True, False], p=[0.9, 0.1]))
                cfg_candidates = int(rng.choice([1, 2], p=[0.60, 0.40]))
                cfg_min_rows = int(rng.choice([80, 120, 180, 240], p=[0.20, 0.45, 0.25, 0.10]))
                cfg_max_rows = int(rng.choice([120000, 200000, 300000, 500000], p=[0.35, 0.35, 0.20, 0.10]))
            else:
                cfg_fast = bool(rng.choice([True, False], p=[0.7, 0.3]))
                cfg_candidates = int(rng.choice([1, 2, 3], p=[0.25, 0.50, 0.25]))
                cfg_min_rows = int(rng.choice([80, 120, 180, 240], p=[0.20, 0.45, 0.25, 0.10]))
                cfg_max_rows = rng.choice([120000, 250000, 500000, None], p=[0.40, 0.35, 0.15, 0.10])
            cfg_top_k = int(rng.choice([2, 3, 4, 5], p=[0.25, 0.45, 0.20, 0.10]))
            cfg_min_abs_score = float(rng.choice([0.10, 0.12, 0.15, 0.18, 0.22], p=[0.20, 0.20, 0.30, 0.20, 0.10]))
            cfg_rebalance = int(rng.choice([2, 3, 5, 8], p=[0.20, 0.50, 0.20, 0.10]))
            cfg_cooldown = int(rng.choice([2, 3, 5, 8, 12], p=[0.20, 0.30, 0.25, 0.15, 0.10]))

            include_patterns: set[str] | None = None
            if best_patterns and bool(rng.choice([True, False], p=[0.65, 0.35])):
                pattern_list = sorted(best_patterns)
                max_k = len(pattern_list)
                if max_k <= 1:
                    k = 1
                elif max_k == 2:
                    k = int(rng.integers(1, 3))
                else:
                    high = min(10, max_k) + 1
                    k = int(rng.integers(2, high))
                k = max(1, min(max_k, int(k)))
                sampled = rng.choice(pattern_list, size=k, replace=False).tolist()
                include_patterns = set(str(x) for x in sampled)

            model_name = f"autoloop_{stamp}_i{i:02d}"
            iteration_parallel = int(effective_parallel_patterns)
            train_table = pd.DataFrame()
            threshold_table = pd.DataFrame()
            bt = pd.DataFrame()
            iteration_error: Exception | None = None
            for attempt_idx in range(2):
                try:
                    train_table = self.train(
                        dataset_name=dataset_name,
                        interval=used_interval,
                        min_pattern_rows=int(cfg_min_rows),
                        model_name=model_name,
                        parallel_patterns=int(iteration_parallel),
                        fast_mode=cfg_fast,
                        max_rows_per_pattern=cfg_max_rows if cfg_max_rows is None else int(cfg_max_rows),
                        include_patterns=include_patterns,
                        candidate_models_per_pattern=int(cfg_candidates),
                    )
                    if train_table.empty:
                        break
                    model_files = set(train_table["model_file"].dropna().astype(str).tolist())

                    threshold_table = self.optimize_model_thresholds_from_backtest(
                        dataset_name=dataset_name,
                        interval=used_interval,
                        fee_bps=realistic_fee,
                        spread_bps=realistic_spread,
                        slippage_bps=realistic_slippage,
                        short_borrow_bps_per_day=float(short_borrow_bps_per_day),
                        latency_bars=realistic_latency,
                        embargo_bars=1,
                        include_patterns=include_patterns,
                        include_model_files=model_files,
                        min_trades=40,
                        max_eval_rows_per_pattern=250000,
                        parallel_models=int(iteration_parallel),
                        persist=True,
                    )

                    bt = self.backtest(
                        dataset_name=dataset_name,
                        interval=used_interval,
                        mode="saved_models",
                        fee_bps=realistic_fee,
                        spread_bps=realistic_spread,
                        slippage_bps=realistic_slippage,
                        short_borrow_bps_per_day=float(short_borrow_bps_per_day),
                        latency_bars=realistic_latency,
                        use_model_thresholds=True,
                        include_patterns=include_patterns,
                        include_model_files=model_files,
                        include_portfolio=True,
                        include_spread_strategies=effective_include_spreads,
                        portfolio_top_k_per_side=int(cfg_top_k),
                        portfolio_min_abs_score=float(cfg_min_abs_score),
                        portfolio_rebalance_every_n_bars=int(cfg_rebalance),
                        portfolio_symbol_cooldown_bars=int(cfg_cooldown),
                        parallel_models=int(iteration_parallel),
                        max_eval_rows_per_pattern=250000,
                    )
                    iteration_error = None
                    break
                except OSError as exc:
                    iteration_error = exc
                    # Resource fallback path: retry once with single-thread workers.
                    if int(getattr(exc, "errno", -1)) == 24 and attempt_idx == 0 and int(iteration_parallel) > 1:
                        iteration_parallel = 1
                        _emit_progress(
                            progress_callback,
                            pct,
                            f"Iteration {i}: hit open-file limit, retrying with parallel=1",
                        )
                        continue
                    break
                except Exception as exc:
                    iteration_error = exc
                    break

            if iteration_error is not None:
                no_improve += 1
                history.append(
                    {
                        "stage": "search",
                        "iteration": i,
                        "model_name": model_name,
                        "status": "error",
                        "error": str(iteration_error),
                        "parallel_patterns_used": int(iteration_parallel),
                    }
                )
                gc.collect()
                continue

            if train_table.empty:
                no_improve += 1
                history.append(
                    {
                        "stage": "search",
                        "iteration": i,
                        "model_name": model_name,
                        "status": "no_models_trained",
                    }
                )
                continue
            picked_raw = self._objective_row_from_backtest(bt)
            picked = self._objective_row_from_backtest(
                bt,
                min_trades=min_trades_gate if min_trades_gate > 0 else None,
            )
            if picked is None and picked_raw is not None and min_trades_gate > 0:
                no_improve += 1
                history.append(
                    {
                        "stage": "search",
                        "iteration": i,
                        "model_name": model_name,
                        "status": "below_min_trades_gate",
                        "trades": float(picked_raw.get("trades", np.nan)),
                        "min_iteration_trades": int(min_trades_gate),
                        "parallel_patterns_used": int(iteration_parallel),
                    }
                )
                gc.collect()
                continue
            obj = float(picked.get("objective_score", float("-inf"))) if picked else float("-inf")
            improved = is_significant_improvement(obj, best_obj)
            if improved:
                best_obj = obj
                best_patterns = (
                    self._top_patterns_from_backtest(
                        bt,
                        top_n=10,
                        min_trades=min_trades_gate if min_trades_gate > 0 else None,
                    )
                    or best_patterns
                )
                best_iteration = int(i)
                no_improve = 0
            else:
                no_improve += 1

            history.append(
                {
                    "stage": "search",
                    "iteration": i,
                    "model_name": model_name,
                    "fast_mode": bool(cfg_fast),
                    "candidate_models_per_pattern": int(cfg_candidates),
                    "min_pattern_rows": int(cfg_min_rows),
                    "max_rows_per_pattern": int(cfg_max_rows) if cfg_max_rows is not None else 0,
                    "portfolio_top_k_per_side": int(cfg_top_k),
                    "portfolio_min_abs_score": float(cfg_min_abs_score),
                    "portfolio_rebalance_every_n_bars": int(cfg_rebalance),
                    "portfolio_symbol_cooldown_bars": int(cfg_cooldown),
                    "models_trained": int(len(train_table)),
                    "thresholds_updated": int(threshold_table["threshold_updated"].sum()) if not threshold_table.empty else 0,
                    "objective_score": obj if np.isfinite(obj) else np.nan,
                    "cumulative_return": float(picked.get("cumulative_return", np.nan)) if picked else np.nan,
                    "sharpe": float(picked.get("sharpe", np.nan)) if picked else np.nan,
                    "max_drawdown": float(picked.get("max_drawdown", np.nan)) if picked else np.nan,
                    "trades": float(picked.get("trades", np.nan)) if picked else np.nan,
                    "improved": int(improved),
                    "selected_patterns": ",".join(sorted(include_patterns or set())),
                    "best_patterns_so_far": ",".join(sorted(best_patterns)),
                    "parallel_patterns_used": int(iteration_parallel),
                    "min_iteration_trades": int(min_trades_gate),
                }
            )
            gc.collect()

        # Final strict walk-forward verification of the best discovered setup.
        if best_patterns:
            _emit_progress(progress_callback, 92.0, "Auto-improve final strict walk-forward verification")
            try:
                verify_windows = 60 if very_large_mode else 120
                verify_train_rows = 180000 if very_large_mode else 250000
                verify = self.backtest(
                    dataset_name=dataset_name,
                    interval=used_interval,
                    mode="walk_forward_retrain",
                    fee_bps=realistic_fee,
                    spread_bps=realistic_spread,
                    slippage_bps=realistic_slippage,
                    short_borrow_bps_per_day=float(short_borrow_bps_per_day),
                    latency_bars=realistic_latency,
                    include_patterns=best_patterns,
                    min_pattern_rows=max(80, int(self.settings.min_pattern_count)),
                    fast_mode=False,
                    parallel_models=effective_parallel_patterns,
                    max_windows_per_pattern=verify_windows,
                    max_train_rows_per_window=verify_train_rows,
                    include_portfolio=True,
                    include_spread_strategies=effective_include_spreads,
                )
                verify_pick = self._objective_row_from_backtest(
                    verify,
                    min_trades=min_trades_gate if min_trades_gate > 0 else None,
                )
                history.append(
                    {
                        "stage": "final_verify",
                        "iteration": best_iteration,
                        "model_name": f"autoloop_{stamp}_best",
                        "objective_score": float(verify_pick.get("objective_score", np.nan)) if verify_pick else np.nan,
                        "cumulative_return": float(verify_pick.get("cumulative_return", np.nan)) if verify_pick else np.nan,
                        "sharpe": float(verify_pick.get("sharpe", np.nan)) if verify_pick else np.nan,
                        "max_drawdown": float(verify_pick.get("max_drawdown", np.nan)) if verify_pick else np.nan,
                        "trades": float(verify_pick.get("trades", np.nan)) if verify_pick else np.nan,
                        "selected_patterns": ",".join(sorted(best_patterns)),
                        "min_iteration_trades": int(min_trades_gate),
                    }
                )
            except Exception as exc:
                history.append(
                    {
                        "stage": "final_verify",
                        "iteration": best_iteration,
                        "model_name": f"autoloop_{stamp}_best",
                        "status": "error",
                        "error": str(exc),
                        "selected_patterns": ",".join(sorted(best_patterns)),
                        "min_iteration_trades": int(min_trades_gate),
                    }
                )

        out = pd.DataFrame(history)
        out_path = self.settings.models_dir / f"autoloop_history_{stamp}.parquet"
        if not out.empty:
            out.to_parquet(out_path, index=False)
        _emit_progress(
            progress_callback,
            100.0,
            (
                "Auto-improve complete. "
                f"Iterations attempted: {max_iters}, history: {out_path.name}"
            ),
        )
        return out

    def _run_backtest(
        self,
        dataset_name: str = "model_dataset",
        interval: str | None = None,
        mode: Literal["saved_models", "walk_forward_retrain"] = "saved_models",
        long_threshold: float | None = 0.65,
        short_threshold: float | None = 0.35,
        fee_bps: float = 1.0,
        include_patterns: set[str] | None = None,
        include_model_files: set[str] | None = None,
        use_model_thresholds: bool = False,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
        short_borrow_bps_per_day: float = 0.0,
        latency_bars: int = 1,
        embargo_bars: int = 1,
        train_window_days: int = 504,
        test_window_days: int = 63,
        step_days: int = 21,
        min_pattern_rows: int | None = None,
        fast_mode: bool = False,
        parallel_models: int = 1,
        max_eval_rows_per_pattern: int | None = None,
        max_windows_per_pattern: int | None = None,
        max_train_rows_per_window: int | None = None,
        include_portfolio: bool = True,
        portfolio_top_k_per_side: int = 5,
        portfolio_max_gross_exposure: float = 1.0,
        portfolio_pattern_selection: str = "all",
        portfolio_best_patterns_top_n: int = 6,
        portfolio_min_pattern_trades: int = 40,
        portfolio_min_pattern_win_rate_trade: float = 0.55,
        portfolio_min_abs_score: float = 0.15,
        portfolio_rebalance_every_n_bars: int = 3,
        portfolio_symbol_cooldown_bars: int = 5,
        portfolio_volatility_scaling: bool = True,
        portfolio_max_symbol_weight: float = 0.35,
        include_spread_strategies: bool = False,
        spread_lookback_bars: int = 63,
        spread_top_components: int = 3,
        spread_min_edge: float = 0.02,
        spread_switch_cost_bps: float = 0.0,
        spread_include_neutral_overlay: bool = True,
        spread_include_regime_switch: bool = True,
        spread_target_vol_annual: float = 0.0,
        return_curves: bool = False,
        initial_investment: float = 10000.0,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        dataset = self.dataset_builder.load_dataset(dataset_name)
        if dataset.empty:
            raise RuntimeError("Dataset is empty. Run build_dataset first.")

        used_interval = interval or self.settings.default_interval
        if mode == "walk_forward_retrain":
            return run_walk_forward_retraining_backtests(
                dataset=dataset,
                interval=used_interval,
                horizon_bars=self.settings.forward_horizon_bars,
                train_window_days=int(train_window_days),
                test_window_days=int(test_window_days),
                step_days=int(step_days),
                min_pattern_rows=int(min_pattern_rows or self.settings.min_pattern_count),
                include_patterns=include_patterns,
                fee_bps=float(fee_bps),
                spread_bps=float(spread_bps),
                slippage_bps=float(slippage_bps),
                short_borrow_bps_per_day=float(short_borrow_bps_per_day),
                latency_bars=int(latency_bars),
                embargo_bars=int(embargo_bars),
                fast_mode=bool(fast_mode),
                parallel_patterns=int(parallel_models),
                max_windows_per_pattern=max_windows_per_pattern,
                max_train_rows_per_window=max_train_rows_per_window,
                include_portfolio=bool(include_portfolio),
                portfolio_top_k_per_side=int(portfolio_top_k_per_side),
                portfolio_max_gross_exposure=float(portfolio_max_gross_exposure),
                portfolio_pattern_selection=str(portfolio_pattern_selection),
                portfolio_best_patterns_top_n=int(portfolio_best_patterns_top_n),
                portfolio_min_pattern_trades=int(portfolio_min_pattern_trades),
                portfolio_min_pattern_win_rate_trade=float(portfolio_min_pattern_win_rate_trade),
                portfolio_min_abs_score=float(portfolio_min_abs_score),
                portfolio_rebalance_every_n_bars=int(portfolio_rebalance_every_n_bars),
                portfolio_symbol_cooldown_bars=int(portfolio_symbol_cooldown_bars),
                portfolio_volatility_scaling=bool(portfolio_volatility_scaling),
                portfolio_max_symbol_weight=float(portfolio_max_symbol_weight),
                include_spread_strategies=bool(include_spread_strategies),
                spread_lookback_bars=int(spread_lookback_bars),
                spread_top_components=int(spread_top_components),
                spread_min_edge=float(spread_min_edge),
                spread_switch_cost_bps=float(spread_switch_cost_bps),
                spread_include_neutral_overlay=bool(spread_include_neutral_overlay),
                spread_include_regime_switch=bool(spread_include_regime_switch),
                spread_target_vol_annual=float(spread_target_vol_annual),
                return_curves=bool(return_curves),
                initial_investment=float(initial_investment),
            )

        return run_pattern_backtests(
            dataset=dataset,
            model_io=self.model_io,
            interval=used_interval,
            horizon_bars=self.settings.forward_horizon_bars,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            fee_bps=fee_bps,
            include_patterns=include_patterns,
            include_model_files=include_model_files,
            use_model_thresholds=use_model_thresholds,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            short_borrow_bps_per_day=short_borrow_bps_per_day,
            latency_bars=latency_bars,
            embargo_bars=embargo_bars,
            parallel_models=int(parallel_models),
            max_eval_rows_per_pattern=max_eval_rows_per_pattern,
            include_portfolio=bool(include_portfolio),
            portfolio_top_k_per_side=int(portfolio_top_k_per_side),
            portfolio_max_gross_exposure=float(portfolio_max_gross_exposure),
            portfolio_pattern_selection=str(portfolio_pattern_selection),
            portfolio_best_patterns_top_n=int(portfolio_best_patterns_top_n),
            portfolio_min_pattern_trades=int(portfolio_min_pattern_trades),
            portfolio_min_pattern_win_rate_trade=float(portfolio_min_pattern_win_rate_trade),
            portfolio_min_abs_score=float(portfolio_min_abs_score),
            portfolio_rebalance_every_n_bars=int(portfolio_rebalance_every_n_bars),
            portfolio_symbol_cooldown_bars=int(portfolio_symbol_cooldown_bars),
            portfolio_volatility_scaling=bool(portfolio_volatility_scaling),
            portfolio_max_symbol_weight=float(portfolio_max_symbol_weight),
            include_spread_strategies=bool(include_spread_strategies),
            spread_lookback_bars=int(spread_lookback_bars),
            spread_top_components=int(spread_top_components),
            spread_min_edge=float(spread_min_edge),
            spread_switch_cost_bps=float(spread_switch_cost_bps),
            spread_include_neutral_overlay=bool(spread_include_neutral_overlay),
            spread_include_regime_switch=bool(spread_include_regime_switch),
            spread_target_vol_annual=float(spread_target_vol_annual),
            return_curves=bool(return_curves),
            initial_investment=float(initial_investment),
        )

    def backtest(
        self,
        dataset_name: str = "model_dataset",
        interval: str | None = None,
        mode: Literal["saved_models", "walk_forward_retrain"] = "saved_models",
        long_threshold: float | None = 0.65,
        short_threshold: float | None = 0.35,
        fee_bps: float = 1.0,
        include_patterns: set[str] | None = None,
        include_model_files: set[str] | None = None,
        use_model_thresholds: bool = False,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
        short_borrow_bps_per_day: float = 0.0,
        latency_bars: int = 1,
        embargo_bars: int = 1,
        train_window_days: int = 504,
        test_window_days: int = 63,
        step_days: int = 21,
        min_pattern_rows: int | None = None,
        fast_mode: bool = False,
        parallel_models: int = 1,
        max_eval_rows_per_pattern: int | None = None,
        max_windows_per_pattern: int | None = None,
        max_train_rows_per_window: int | None = None,
        include_portfolio: bool = True,
        portfolio_top_k_per_side: int = 5,
        portfolio_max_gross_exposure: float = 1.0,
        portfolio_pattern_selection: str = "all",
        portfolio_best_patterns_top_n: int = 6,
        portfolio_min_pattern_trades: int = 40,
        portfolio_min_pattern_win_rate_trade: float = 0.55,
        portfolio_min_abs_score: float = 0.15,
        portfolio_rebalance_every_n_bars: int = 3,
        portfolio_symbol_cooldown_bars: int = 5,
        portfolio_volatility_scaling: bool = True,
        portfolio_max_symbol_weight: float = 0.35,
        include_spread_strategies: bool = False,
        spread_lookback_bars: int = 63,
        spread_top_components: int = 3,
        spread_min_edge: float = 0.02,
        spread_switch_cost_bps: float = 0.0,
        spread_include_neutral_overlay: bool = True,
        spread_include_regime_switch: bool = True,
        spread_target_vol_annual: float = 0.0,
    ) -> pd.DataFrame:
        result = self._run_backtest(
            dataset_name=dataset_name,
            interval=interval,
            mode=mode,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            fee_bps=fee_bps,
            include_patterns=include_patterns,
            include_model_files=include_model_files,
            use_model_thresholds=use_model_thresholds,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            short_borrow_bps_per_day=short_borrow_bps_per_day,
            latency_bars=latency_bars,
            embargo_bars=embargo_bars,
            train_window_days=train_window_days,
            test_window_days=test_window_days,
            step_days=step_days,
            min_pattern_rows=min_pattern_rows,
            fast_mode=fast_mode,
            parallel_models=parallel_models,
            max_eval_rows_per_pattern=max_eval_rows_per_pattern,
            max_windows_per_pattern=max_windows_per_pattern,
            max_train_rows_per_window=max_train_rows_per_window,
            include_portfolio=include_portfolio,
            portfolio_top_k_per_side=portfolio_top_k_per_side,
            portfolio_max_gross_exposure=portfolio_max_gross_exposure,
            portfolio_pattern_selection=portfolio_pattern_selection,
            portfolio_best_patterns_top_n=portfolio_best_patterns_top_n,
            portfolio_min_pattern_trades=portfolio_min_pattern_trades,
            portfolio_min_pattern_win_rate_trade=portfolio_min_pattern_win_rate_trade,
            portfolio_min_abs_score=portfolio_min_abs_score,
            portfolio_rebalance_every_n_bars=portfolio_rebalance_every_n_bars,
            portfolio_symbol_cooldown_bars=portfolio_symbol_cooldown_bars,
            portfolio_volatility_scaling=portfolio_volatility_scaling,
            portfolio_max_symbol_weight=portfolio_max_symbol_weight,
            include_spread_strategies=include_spread_strategies,
            spread_lookback_bars=spread_lookback_bars,
            spread_top_components=spread_top_components,
            spread_min_edge=spread_min_edge,
            spread_switch_cost_bps=spread_switch_cost_bps,
            spread_include_neutral_overlay=spread_include_neutral_overlay,
            spread_include_regime_switch=spread_include_regime_switch,
            spread_target_vol_annual=spread_target_vol_annual,
            return_curves=False,
            initial_investment=10000.0,
        )
        if isinstance(result, tuple):
            return result[0]
        return result

    def backtest_with_details(
        self,
        dataset_name: str = "model_dataset",
        interval: str | None = None,
        mode: Literal["saved_models", "walk_forward_retrain"] = "saved_models",
        long_threshold: float | None = 0.65,
        short_threshold: float | None = 0.35,
        fee_bps: float = 1.0,
        include_patterns: set[str] | None = None,
        include_model_files: set[str] | None = None,
        use_model_thresholds: bool = False,
        spread_bps: float = 0.0,
        slippage_bps: float = 0.0,
        short_borrow_bps_per_day: float = 0.0,
        latency_bars: int = 1,
        embargo_bars: int = 1,
        train_window_days: int = 504,
        test_window_days: int = 63,
        step_days: int = 21,
        min_pattern_rows: int | None = None,
        fast_mode: bool = False,
        parallel_models: int = 1,
        max_eval_rows_per_pattern: int | None = None,
        max_windows_per_pattern: int | None = None,
        max_train_rows_per_window: int | None = None,
        include_portfolio: bool = True,
        portfolio_top_k_per_side: int = 5,
        portfolio_max_gross_exposure: float = 1.0,
        portfolio_pattern_selection: str = "all",
        portfolio_best_patterns_top_n: int = 6,
        portfolio_min_pattern_trades: int = 40,
        portfolio_min_pattern_win_rate_trade: float = 0.55,
        portfolio_min_abs_score: float = 0.15,
        portfolio_rebalance_every_n_bars: int = 3,
        portfolio_symbol_cooldown_bars: int = 5,
        portfolio_volatility_scaling: bool = True,
        portfolio_max_symbol_weight: float = 0.35,
        include_spread_strategies: bool = False,
        spread_lookback_bars: int = 63,
        spread_top_components: int = 3,
        spread_min_edge: float = 0.02,
        spread_switch_cost_bps: float = 0.0,
        spread_include_neutral_overlay: bool = True,
        spread_include_regime_switch: bool = True,
        spread_target_vol_annual: float = 0.0,
        initial_investment: float = 10000.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        result = self._run_backtest(
            dataset_name=dataset_name,
            interval=interval,
            mode=mode,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            fee_bps=fee_bps,
            include_patterns=include_patterns,
            include_model_files=include_model_files,
            use_model_thresholds=use_model_thresholds,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            short_borrow_bps_per_day=short_borrow_bps_per_day,
            latency_bars=latency_bars,
            embargo_bars=embargo_bars,
            train_window_days=train_window_days,
            test_window_days=test_window_days,
            step_days=step_days,
            min_pattern_rows=min_pattern_rows,
            fast_mode=fast_mode,
            parallel_models=parallel_models,
            max_eval_rows_per_pattern=max_eval_rows_per_pattern,
            max_windows_per_pattern=max_windows_per_pattern,
            max_train_rows_per_window=max_train_rows_per_window,
            include_portfolio=include_portfolio,
            portfolio_top_k_per_side=portfolio_top_k_per_side,
            portfolio_max_gross_exposure=portfolio_max_gross_exposure,
            portfolio_pattern_selection=portfolio_pattern_selection,
            portfolio_best_patterns_top_n=portfolio_best_patterns_top_n,
            portfolio_min_pattern_trades=portfolio_min_pattern_trades,
            portfolio_min_pattern_win_rate_trade=portfolio_min_pattern_win_rate_trade,
            portfolio_min_abs_score=portfolio_min_abs_score,
            portfolio_rebalance_every_n_bars=portfolio_rebalance_every_n_bars,
            portfolio_symbol_cooldown_bars=portfolio_symbol_cooldown_bars,
            portfolio_volatility_scaling=portfolio_volatility_scaling,
            portfolio_max_symbol_weight=portfolio_max_symbol_weight,
            include_spread_strategies=include_spread_strategies,
            spread_lookback_bars=spread_lookback_bars,
            spread_top_components=spread_top_components,
            spread_min_edge=spread_min_edge,
            spread_switch_cost_bps=spread_switch_cost_bps,
            spread_include_neutral_overlay=spread_include_neutral_overlay,
            spread_include_regime_switch=spread_include_regime_switch,
            spread_target_vol_annual=spread_target_vol_annual,
            return_curves=True,
            initial_investment=initial_investment,
        )
        if isinstance(result, tuple):
            return result
        return result, pd.DataFrame(
            columns=[
                "datetime",
                "pattern",
                "model_file",
                "period_return",
                "cumulative_return",
                "equity_value",
                "initial_investment",
                "curve_variant",
            ]
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
        use_model_thresholds: bool = True,
        long_threshold: float | None = None,
        short_threshold: float | None = None,
    ) -> pd.DataFrame:
        used_interval = interval or self.settings.default_interval
        frames = self.load_history(interval=used_interval, years=years, refresh=refresh_prices, universe=universe)
        if not frames:
            return pd.DataFrame()

        fundamentals = build_fundamental_table(
            symbols=sorted(frames.keys()),
            cache_path=self.settings.processed_data_dir / "fundamentals.parquet",
            refresh=False,
            max_symbols=len(frames),
            provider=self.settings.fundamentals_provider,
            fmp_api_key=self.settings.fmp_api_key,
            fmp_base_url=self.settings.fmp_base_url,
            request_workers=self.settings.request_workers,
        )

        politician = None
        if politician_trades_csv:
            trades = load_politician_trades(politician_trades_csv)
            politician = engineer_politician_features(trades) if not trades.empty else None
        elif self.settings.fmp_api_key:
            merged = pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()
            if not merged.empty and {"symbol", "datetime"}.issubset(merged.columns):
                dt = pd.to_datetime(merged["datetime"], utc=True, errors="coerce")
                trades = fetch_politician_trades_fmp(
                    symbols=sorted(frames.keys()),
                    fmp_api_key=self.settings.fmp_api_key,
                    fmp_base_url=self.settings.fmp_base_url,
                    start_datetime=dt.min(),
                    end_datetime=dt.max(),
                    request_workers=self.settings.request_workers,
                )
                politician = engineer_politician_features(trades) if not trades.empty else None

        macro = None
        merged = pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()
        if not merged.empty and "datetime" in merged.columns:
            dt = pd.to_datetime(merged["datetime"], utc=True, errors="coerce")
            dt_start = dt.min()
            dt_end = dt.max()
            if pd.notna(dt_start) and pd.notna(dt_end):
                macro = build_macro_feature_table(
                    cache_path=self.settings.processed_data_dir / "macro_features.parquet",
                    start_datetime=dt_start,
                    end_datetime=dt_end,
                    refresh=False,
                    fmp_api_key=self.settings.fmp_api_key,
                    fmp_base_url=self.settings.fmp_base_url,
                    fred_api_key=self.settings.fred_api_key,
                    request_workers=self.settings.request_workers,
                )

        return self.scanner.scan(
            latest_frames=frames,
            config=ScanConfig(
                interval=used_interval,
                horizon_bars=self.settings.forward_horizon_bars,
                top_n=top_n or self.settings.top_n_signals,
                min_confidence=min_confidence,
                use_model_thresholds=use_model_thresholds,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
            ),
            fundamentals=fundamentals,
            politician_features=politician,
            macro_features=macro,
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

    def recommend_thresholds(
        self,
        interval: str | None = None,
        include_patterns: set[str] | None = None,
        include_model_files: set[str] | None = None,
    ) -> dict[str, object]:
        registry = self.model_io.get_model_registry(interval=interval)
        if registry.empty:
            return {
                "recommended_long_threshold": 0.65,
                "recommended_short_threshold": 0.35,
                "models_considered": 0,
                "method": "fallback_default",
                "explanation": "No trained models available for recommendation; using default thresholds.",
            }

        work = registry.copy()
        if include_patterns:
            work = work.loc[work["pattern"].astype(str).isin(include_patterns)]
        if include_model_files:
            work = work.loc[work["model_file"].astype(str).isin(include_model_files)]
        if work.empty:
            return {
                "recommended_long_threshold": 0.65,
                "recommended_short_threshold": 0.35,
                "models_considered": 0,
                "method": "fallback_default",
                "explanation": "No models matched the current filters; using default thresholds.",
            }

        for col in ("tuned_long_threshold", "tuned_short_threshold", "roc_auc", "train_rows"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        work = work.loc[
            work["tuned_long_threshold"].between(0.0, 1.0, inclusive="both")
            & work["tuned_short_threshold"].between(0.0, 1.0, inclusive="both")
        ].copy()
        if work.empty:
            return {
                "recommended_long_threshold": 0.65,
                "recommended_short_threshold": 0.35,
                "models_considered": 0,
                "method": "fallback_default",
                "explanation": "No models had tuned thresholds saved; using default thresholds.",
            }

        roc = pd.to_numeric(work.get("roc_auc"), errors="coerce").fillna(0.5)
        train_rows = pd.to_numeric(work.get("train_rows"), errors="coerce").fillna(0.0)
        weights = (roc - 0.5).clip(lower=0.01) * np.log1p(train_rows.clip(lower=1.0))
        if float(weights.sum()) <= 0:
            weights = pd.Series(np.ones(len(work), dtype=float), index=work.index)

        rec_long = float(np.average(work["tuned_long_threshold"], weights=weights))
        rec_short = float(np.average(work["tuned_short_threshold"], weights=weights))
        rec_long = float(np.clip(rec_long, 0.60, 0.95))
        rec_short = float(np.clip(rec_short, 0.05, 0.40))
        if rec_short >= rec_long:
            mid = (rec_long + rec_short) / 2.0
            rec_long = float(min(0.95, max(0.5, mid + 0.05)))
            rec_short = float(max(0.05, min(0.5, mid - 0.05)))

        return {
            "recommended_long_threshold": rec_long,
            "recommended_short_threshold": rec_short,
            "models_considered": int(len(work)),
            "weighted_mean_roc_auc": float(np.average(roc, weights=weights)),
            "median_tuned_long": float(work["tuned_long_threshold"].median()),
            "median_tuned_short": float(work["tuned_short_threshold"].median()),
            "method": "roc_auc_train_rows_weighted_average",
            "explanation": (
                "Weighted by each model's ROC-AUC edge over 0.5 and train sample size. "
                "Use as a starting point; then refine per backtest mode/cost assumptions."
            ),
        }

    def _groundup_runs_path(self) -> Path:
        return self.settings.models_dir / "groundup_runs.parquet"

    def _groundup_deployment_path(self) -> Path:
        return self.settings.models_dir / "groundup_deployment.json"

    @staticmethod
    def _normalize_run_id(run_id: str) -> str:
        txt = str(run_id or "").strip()
        if not txt:
            raise ValueError("run_id cannot be empty")
        return txt

    def groundup_runs(self) -> pd.DataFrame:
        path = self._groundup_runs_path()
        if not path.exists():
            return pd.DataFrame(
                columns=[
                    "run_id",
                    "run_name",
                    "dataset_name",
                    "interval",
                    "created_utc",
                    "status",
                    "model_count",
                    "model_files_json",
                    "objective_score",
                    "cumulative_return",
                    "sharpe",
                    "max_drawdown",
                    "trades",
                    "win_rate_trade",
                    "models_trained",
                    "thresholds_updated",
                    "avg_model_roc_auc",
                    "avg_model_train_rows",
                    "notes",
                ]
            )
        try:
            out = pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
        if out.empty:
            return out
        if "created_utc" in out.columns:
            out["created_utc"] = pd.to_datetime(out["created_utc"], utc=True, errors="coerce")
            out = out.sort_values("created_utc", ascending=False, na_position="last")
        return out.reset_index(drop=True)

    def _save_groundup_runs(self, frame: pd.DataFrame) -> None:
        out = frame.copy()
        if "created_utc" in out.columns:
            out["created_utc"] = pd.to_datetime(out["created_utc"], utc=True, errors="coerce")
        if "model_files_json" in out.columns:
            out["model_files_json"] = out["model_files_json"].astype(str)
        if "notes" in out.columns:
            out["notes"] = out["notes"].fillna("").astype(str)
        path = self._groundup_runs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)

    def groundup_models_for_run(self, run_id: str) -> set[str]:
        run_key = self._normalize_run_id(run_id)
        runs = self.groundup_runs()
        if runs.empty or "run_id" not in runs.columns:
            return set()
        row = runs.loc[runs["run_id"].astype(str) == run_key]
        if row.empty:
            return set()
        payload = row.iloc[0].get("model_files_json", "[]")
        try:
            data = json.loads(str(payload))
        except Exception:
            return set()
        if not isinstance(data, list):
            return set()
        out: set[str] = set()
        for item in data:
            txt = str(item or "").strip()
            if not txt:
                continue
            safe = Path(txt).name
            if not safe:
                continue
            out.add(safe)
        return out

    def groundup_register_run(
        self,
        run_id: str,
        run_name: str,
        dataset_name: str,
        interval: str,
        model_files: set[str],
        train_table: pd.DataFrame,
        backtest_table: pd.DataFrame,
        status: Literal["champion", "challenger", "archived"] = "challenger",
        notes: str | None = None,
    ) -> dict[str, object]:
        run_key = self._normalize_run_id(run_id)
        cleaned_name = str(run_name or "").strip() or run_key
        used_interval = str(interval or self.settings.default_interval)
        model_list = sorted(
            set(
                Path(str(x)).name
                for x in model_files
                if str(x).strip() and Path(str(x)).name
            )
        )
        picked = self._objective_row_from_backtest(backtest_table)
        train_work = train_table.copy() if isinstance(train_table, pd.DataFrame) else pd.DataFrame()

        avg_roc = float(pd.to_numeric(train_work.get("roc_auc"), errors="coerce").mean()) if not train_work.empty else np.nan
        avg_train_rows = float(pd.to_numeric(train_work.get("train_rows"), errors="coerce").mean()) if not train_work.empty else np.nan
        thresholds_updated = (
            int(pd.to_numeric(train_work.get("thresholds_updated"), errors="coerce").fillna(0).sum()) if "thresholds_updated" in train_work.columns else np.nan
        )

        row = {
            "run_id": run_key,
            "run_name": cleaned_name,
            "dataset_name": str(dataset_name),
            "interval": used_interval,
            "created_utc": pd.Timestamp.now(tz="UTC"),
            "status": str(status),
            "model_count": int(len(model_list)),
            "model_files_json": json.dumps(model_list),
            "objective_score": float(picked.get("objective_score", np.nan)) if picked else np.nan,
            "cumulative_return": float(picked.get("cumulative_return", np.nan)) if picked else np.nan,
            "sharpe": float(picked.get("sharpe", np.nan)) if picked else np.nan,
            "max_drawdown": float(picked.get("max_drawdown", np.nan)) if picked else np.nan,
            "trades": float(picked.get("trades", np.nan)) if picked else np.nan,
            "win_rate_trade": float(picked.get("win_rate_trade", np.nan)) if picked else np.nan,
            "models_trained": int(len(train_work)),
            "thresholds_updated": int(thresholds_updated) if np.isfinite(thresholds_updated) else np.nan,
            "avg_model_roc_auc": float(avg_roc) if np.isfinite(avg_roc) else np.nan,
            "avg_model_train_rows": float(avg_train_rows) if np.isfinite(avg_train_rows) else np.nan,
            "notes": str(notes or ""),
        }

        runs = self.groundup_runs()
        if runs.empty:
            out = pd.DataFrame([row])
        else:
            base = runs.loc[runs["run_id"].astype(str) != run_key].copy()
            out = pd.concat([base, pd.DataFrame([row])], ignore_index=True)
        self._save_groundup_runs(out)
        return row

    def groundup_set_run_status(
        self,
        run_id: str,
        status: Literal["champion", "challenger", "archived"],
        notes: str | None = None,
    ) -> dict[str, object]:
        run_key = self._normalize_run_id(run_id)
        runs = self.groundup_runs()
        if runs.empty or "run_id" not in runs.columns:
            return {"updated": False, "run_id": run_key}
        mask = runs["run_id"].astype(str) == run_key
        if not bool(mask.any()):
            return {"updated": False, "run_id": run_key}
        runs.loc[mask, "status"] = str(status)
        if notes is not None:
            runs.loc[mask, "notes"] = str(notes)
        self._save_groundup_runs(runs)
        return {"updated": True, "run_id": run_key, "status": str(status)}

    def groundup_get_deployment(self) -> dict[str, object]:
        path = self._groundup_deployment_path()
        default = {
            "champion_run_id": None,
            "challenger_run_id": None,
            "policy": {
                "min_relative_improvement": 0.05,
                "min_trade_count": 80,
                "max_champion_age_days": 45,
            },
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        if not path.exists():
            return default
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
        if not isinstance(payload, dict):
            return default
        out = default.copy()
        out.update(payload)
        policy = out.get("policy", {})
        if not isinstance(policy, dict):
            policy = {}
        out["policy"] = {
            "min_relative_improvement": float(policy.get("min_relative_improvement", 0.05)),
            "min_trade_count": int(policy.get("min_trade_count", 80)),
            "max_champion_age_days": int(policy.get("max_champion_age_days", 45)),
        }
        return out

    def groundup_set_deployment(
        self,
        champion_run_id: str | None = None,
        challenger_run_id: str | None = None,
        min_relative_improvement: float | None = None,
        min_trade_count: int | None = None,
        max_champion_age_days: int | None = None,
    ) -> dict[str, object]:
        state = self.groundup_get_deployment()
        if champion_run_id is not None:
            state["champion_run_id"] = str(champion_run_id).strip() or None
        if challenger_run_id is not None:
            state["challenger_run_id"] = str(challenger_run_id).strip() or None
        policy = dict(state.get("policy", {}))
        if min_relative_improvement is not None:
            policy["min_relative_improvement"] = max(0.0, float(min_relative_improvement))
        if min_trade_count is not None:
            policy["min_trade_count"] = max(0, int(min_trade_count))
        if max_champion_age_days is not None:
            policy["max_champion_age_days"] = max(1, int(max_champion_age_days))
        state["policy"] = policy
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        path = self._groundup_deployment_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        return state

    def groundup_promotion_decision(self) -> dict[str, object]:
        state = self.groundup_get_deployment()
        champion_id = state.get("champion_run_id")
        challenger_id = state.get("challenger_run_id")
        policy = dict(state.get("policy", {}))
        min_rel = float(policy.get("min_relative_improvement", 0.05))
        min_trades = int(policy.get("min_trade_count", 80))
        max_age_days = int(policy.get("max_champion_age_days", 45))

        runs = self.groundup_runs()
        if runs.empty:
            return {"promote": False, "reason": "no_runs"}
        if not champion_id or not challenger_id:
            return {"promote": False, "reason": "champion_or_challenger_missing"}

        c_row = runs.loc[runs["run_id"].astype(str) == str(champion_id)]
        h_row = runs.loc[runs["run_id"].astype(str) == str(challenger_id)]
        if c_row.empty or h_row.empty:
            return {"promote": False, "reason": "champion_or_challenger_not_found"}

        champion = c_row.iloc[0]
        challenger = h_row.iloc[0]
        c_obj = float(pd.to_numeric(champion.get("objective_score"), errors="coerce"))
        h_obj = float(pd.to_numeric(challenger.get("objective_score"), errors="coerce"))
        h_trades = float(pd.to_numeric(challenger.get("trades"), errors="coerce"))

        c_created = pd.to_datetime(champion.get("created_utc"), utc=True, errors="coerce")
        age_days = float("nan")
        stale = False
        if pd.notna(c_created):
            age_days = float((datetime.now(timezone.utc) - c_created.to_pydatetime()).total_seconds() / (24.0 * 3600.0))
            stale = age_days >= float(max_age_days)

        rel_thresh = abs(c_obj) * max(0.0, min_rel) if np.isfinite(c_obj) else 0.0
        beats_objective = bool(np.isfinite(h_obj) and (not np.isfinite(c_obj) or (h_obj - c_obj) > max(1e-9, rel_thresh)))
        trades_ok = bool(np.isfinite(h_trades) and h_trades >= float(min_trades))

        promote = bool(trades_ok and (beats_objective or stale))
        reason = "objective_and_trade_gate"
        if not trades_ok:
            reason = "challenger_trade_count_too_low"
        elif stale and not beats_objective:
            reason = "champion_stale_challenger_trade_ok"
        elif not beats_objective:
            reason = "challenger_not_significantly_better"

        return {
            "promote": promote,
            "reason": reason,
            "champion_run_id": str(champion_id),
            "challenger_run_id": str(challenger_id),
            "champion_objective": c_obj if np.isfinite(c_obj) else np.nan,
            "challenger_objective": h_obj if np.isfinite(h_obj) else np.nan,
            "challenger_trades": h_trades if np.isfinite(h_trades) else np.nan,
            "champion_age_days": age_days if np.isfinite(age_days) else np.nan,
            "policy": {
                "min_relative_improvement": float(min_rel),
                "min_trade_count": int(min_trades),
                "max_champion_age_days": int(max_age_days),
            },
        }

    def groundup_promote_challenger(self) -> dict[str, object]:
        decision = self.groundup_promotion_decision()
        if not bool(decision.get("promote")):
            return {"promoted": False, **decision}
        champion_id = str(decision.get("champion_run_id", "")).strip() or None
        challenger_id = str(decision.get("challenger_run_id", "")).strip() or None
        if not challenger_id:
            return {"promoted": False, "reason": "challenger_missing"}

        if champion_id:
            self.groundup_set_run_status(champion_id, status="archived")
        self.groundup_set_run_status(challenger_id, status="champion")
        self.groundup_set_deployment(champion_run_id=challenger_id, challenger_run_id=None)
        out = dict(decision)
        out["promoted"] = True
        out["new_champion_run_id"] = challenger_id
        return out

    def groundup_regime_snapshot(self, dataset_name: str = "model_dataset") -> pd.DataFrame:
        dataset = self.dataset_builder.load_dataset(dataset_name)
        if dataset.empty:
            return pd.DataFrame(columns=["regime", "rows", "share", "avg_future_return"])
        if "future_return" in dataset.columns:
            dataset["future_return"] = pd.to_numeric(dataset["future_return"], errors="coerce")
        regime_cols = [
            c
            for c in (
                "macro_regime_high_stress",
                "macro_regime_risk_off",
                "regime_high_vol_downtrend",
                "regime_low_vol_uptrend",
            )
            if c in dataset.columns
        ]
        if not regime_cols:
            return pd.DataFrame(columns=["regime", "rows", "share", "avg_future_return"])

        rows: list[dict[str, object]] = []
        total = float(len(dataset))
        for col in regime_cols:
            mask = pd.to_numeric(dataset[col], errors="coerce").fillna(0).astype(float) > 0
            count = int(mask.sum())
            if count <= 0:
                continue
            avg_ret = float(pd.to_numeric(dataset.loc[mask, "future_return"], errors="coerce").mean()) if "future_return" in dataset.columns else np.nan
            rows.append(
                {
                    "regime": col,
                    "rows": count,
                    "share": float(count / total) if total > 0 else np.nan,
                    "avg_future_return": avg_ret if np.isfinite(avg_ret) else np.nan,
                }
            )
        if not rows:
            return pd.DataFrame(columns=["regime", "rows", "share", "avg_future_return"])
        return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)

    def list_datasets(self) -> pd.DataFrame:
        return self.dataset_builder.list_datasets()

    def dataset_summary(self, dataset_name: str) -> dict[str, object]:
        return self.dataset_builder.dataset_summary(dataset_name)

    def delete_dataset(self, dataset_name: str) -> bool:
        return self.dataset_builder.delete_dataset(dataset_name)

    def model_registry(self, interval: str | None = None, pattern: str | None = None) -> pd.DataFrame:
        return self.model_io.get_model_registry(interval=interval, pattern=pattern)

    def model_details(self, model_file: str, top_n_importance: int = 30) -> dict[str, object]:
        details = self.model_io.get_model_details(model_file, top_n_importance=top_n_importance)
        importance = details.get("importance")
        if isinstance(importance, pd.DataFrame):
            details["importance"] = importance.to_dict(orient="records")
        return details

    def delete_model(self, model_file: str) -> dict[str, object]:
        return self.model_io.delete_model(model_file)

    def delete_models(self, model_files: list[str]) -> dict[str, object]:
        return self.model_io.delete_models(model_files)

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

    def _filter_frames_by_universe_membership(
        self,
        frames: dict[str, pd.DataFrame],
        universe: Literal["sp500", "sp100", "custom"],
        symbols: list[str],
    ) -> dict[str, pd.DataFrame]:
        if not frames:
            return {}

        if universe == "sp500":
            try:
                intervals = get_sp500_membership_intervals_resilient(
                    cache_path=self.settings.processed_data_dir / "sp500_membership_intervals.csv",
                    symbols=symbols,
                )
            except Exception as exc:
                # Upstream membership metadata can occasionally fail; keep dataset build available
                # while surfacing a warning to avoid silent survivorship-bias drift.
                warnings.warn(
                    f"S&P 500 membership history unavailable; using unfiltered frames. Reason: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return frames
            return filter_frames_by_membership_intervals(frames, intervals)

        if universe == "sp100":
            intervals = build_static_membership_intervals(symbols=symbols)
            return filter_frames_by_membership_intervals(frames, intervals)

        return frames

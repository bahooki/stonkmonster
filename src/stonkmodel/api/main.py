from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from stonkmodel.api.schemas import (
    AutoImproveRequest,
    BacktestRequest,
    BuildDatasetRequest,
    RecursiveTrainRequest,
    ScanRequest,
    SweepIntervalsRequest,
    ThresholdOptimizeRequest,
    TrainRequest,
)
from stonkmodel.config import get_settings
from stonkmodel.pipeline import StonkService

settings = get_settings()
service = StonkService(settings)
app = FastAPI(title=settings.app_name)


def _table_records(table: pd.DataFrame) -> list[dict[str, object]]:
    if table.empty:
        return []
    work = table.copy()
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = work[col].astype("string")
    work = work.where(pd.notna(work), None)
    return work.to_dict(orient="records")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <html>
      <head>
        <title>StonkModel API</title>
        <style>
          body { font-family: ui-sans-serif, system-ui; margin: 3rem; max-width: 960px; }
          code { background: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 4px; }
        </style>
      </head>
      <body>
        <h1>StonkModel</h1>
        <p>Pattern-specific stacked ML scanner with FMP/yfinance/polygon providers.</p>
        <p>Use <code>/docs</code> for interactive API calls.</p>
      </body>
    </html>
    """


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config")
def config() -> dict[str, object]:
    payload = service.settings_dict()
    payload["fmp_api_key_set"] = bool(settings.fmp_api_key)
    payload["fred_api_key_set"] = bool(settings.fred_api_key)
    payload["polygon_api_key_set"] = bool(settings.polygon_api_key)
    payload.pop("fmp_api_key", None)
    payload.pop("fred_api_key", None)
    payload.pop("polygon_api_key", None)
    return payload


@app.get("/datasets")
def datasets() -> dict[str, object]:
    table = service.list_datasets()
    return {
        "rows": len(table),
        "datasets": _table_records(table),
    }


@app.get("/dataset/summary")
def dataset_summary(dataset_name: str = "model_dataset") -> dict[str, object]:
    return service.dataset_summary(dataset_name)


@app.post("/dataset/build")
def build_dataset(payload: BuildDatasetRequest) -> dict[str, object]:
    try:
        result = service.build_dataset(
            interval=payload.interval,
            years=payload.years,
            refresh_prices=payload.refresh_prices,
            dataset_name=payload.dataset_name,
            politician_trades_csv=payload.politician_trades_csv,
            universe=payload.universe,
            years_ago_start=payload.years_ago_start,
            years_ago_end=payload.years_ago_end,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "universe": result.universe,
        "years_ago_start": result.years_ago_start,
        "years_ago_end": result.years_ago_end,
        "symbols_requested": result.symbols_requested,
        "symbols_loaded": result.symbols_loaded,
        "rows": result.rows,
        "dataset_path": result.dataset_path,
    }


@app.post("/train")
def train(payload: TrainRequest) -> dict[str, object]:
    try:
        summary = service.train(
            dataset_name=payload.dataset_name,
            interval=payload.interval,
            min_pattern_rows=payload.min_pattern_rows,
            model_name=payload.model_name,
            parallel_patterns=payload.parallel_patterns,
            fast_mode=payload.fast_mode,
            max_rows_per_pattern=payload.max_rows_per_pattern,
            include_patterns=set(payload.include_patterns or []) or None,
            candidate_models_per_pattern=payload.candidate_models_per_pattern,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "rows": len(summary),
        "results": _table_records(summary),
    }


@app.post("/train/recursive")
def train_recursive(payload: RecursiveTrainRequest) -> dict[str, object]:
    try:
        summary = service.recursive_train_from_backtest(
            dataset_name=payload.dataset_name,
            interval=payload.interval,
            min_pattern_rows=payload.min_pattern_rows,
            base_model_name=payload.base_model_name,
            rounds=payload.rounds,
            keep_top_patterns=payload.keep_top_patterns,
            min_trades_to_keep=payload.min_trades_to_keep,
            parallel_patterns=payload.parallel_patterns,
            fast_mode=payload.fast_mode,
            max_rows_per_pattern=payload.max_rows_per_pattern,
            candidate_models_per_pattern=payload.candidate_models_per_pattern,
            fee_bps=payload.fee_bps,
            spread_bps=payload.spread_bps,
            slippage_bps=payload.slippage_bps,
            short_borrow_bps_per_day=payload.short_borrow_bps_per_day,
            latency_bars=payload.latency_bars,
            min_threshold_opt_trades=payload.min_threshold_opt_trades,
            max_eval_rows_per_pattern=payload.max_eval_rows_per_pattern,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "rows": len(summary),
        "results": _table_records(summary),
    }


@app.post("/train/autopilot")
def train_autopilot(payload: AutoImproveRequest) -> dict[str, object]:
    try:
        summary = service.auto_improve(
            dataset_name=payload.dataset_name,
            interval=payload.interval,
            iterations=payload.iterations,
            max_minutes=payload.max_minutes,
            patience=payload.patience,
            min_significant_improvement=payload.min_significant_improvement,
            min_iteration_trades=payload.min_iteration_trades,
            fee_bps=payload.fee_bps,
            spread_bps=payload.spread_bps,
            slippage_bps=payload.slippage_bps,
            short_borrow_bps_per_day=payload.short_borrow_bps_per_day,
            latency_bars=payload.latency_bars,
            parallel_patterns=payload.parallel_patterns,
            include_spread_strategies=payload.include_spread_strategies,
            random_seed=payload.random_seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "rows": len(summary),
        "results": _table_records(summary),
    }


@app.post("/thresholds/optimize")
def optimize_thresholds(payload: ThresholdOptimizeRequest) -> dict[str, object]:
    try:
        summary = service.optimize_model_thresholds_from_backtest(
            dataset_name=payload.dataset_name,
            interval=payload.interval,
            fee_bps=payload.fee_bps,
            spread_bps=payload.spread_bps,
            slippage_bps=payload.slippage_bps,
            short_borrow_bps_per_day=payload.short_borrow_bps_per_day,
            latency_bars=payload.latency_bars,
            embargo_bars=payload.embargo_bars,
            include_patterns=set(payload.include_patterns or []) or None,
            include_model_files=set(payload.include_model_files or []) or None,
            min_trades=payload.min_trades,
            max_eval_rows_per_pattern=payload.max_eval_rows_per_pattern,
            parallel_models=payload.parallel_models,
            persist=payload.persist,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "rows": len(summary),
        "results": _table_records(summary),
    }


@app.post("/backtest")
def backtest(payload: BacktestRequest) -> dict[str, object]:
    try:
        results = service.backtest(
            dataset_name=payload.dataset_name,
            interval=payload.interval,
            mode=payload.mode,
            long_threshold=payload.long_threshold,
            short_threshold=payload.short_threshold,
            fee_bps=payload.fee_bps,
            use_model_thresholds=payload.use_model_thresholds,
            spread_bps=payload.spread_bps,
            slippage_bps=payload.slippage_bps,
            short_borrow_bps_per_day=payload.short_borrow_bps_per_day,
            latency_bars=payload.latency_bars,
            embargo_bars=payload.embargo_bars,
            train_window_days=payload.train_window_days,
            test_window_days=payload.test_window_days,
            step_days=payload.step_days,
            min_pattern_rows=payload.min_pattern_rows,
            fast_mode=payload.fast_mode,
            parallel_models=payload.parallel_models,
            max_eval_rows_per_pattern=payload.max_eval_rows_per_pattern,
            max_windows_per_pattern=payload.max_windows_per_pattern,
            max_train_rows_per_window=payload.max_train_rows_per_window,
            include_portfolio=payload.include_portfolio,
            portfolio_top_k_per_side=payload.portfolio_top_k_per_side,
            portfolio_max_gross_exposure=payload.portfolio_max_gross_exposure,
            portfolio_pattern_selection=payload.portfolio_pattern_selection,
            portfolio_best_patterns_top_n=payload.portfolio_best_patterns_top_n,
            portfolio_min_pattern_trades=payload.portfolio_min_pattern_trades,
            portfolio_min_pattern_win_rate_trade=payload.portfolio_min_pattern_win_rate_trade,
            portfolio_min_abs_score=payload.portfolio_min_abs_score,
            portfolio_rebalance_every_n_bars=payload.portfolio_rebalance_every_n_bars,
            portfolio_symbol_cooldown_bars=payload.portfolio_symbol_cooldown_bars,
            portfolio_volatility_scaling=payload.portfolio_volatility_scaling,
            portfolio_max_symbol_weight=payload.portfolio_max_symbol_weight,
            include_spread_strategies=payload.include_spread_strategies,
            spread_lookback_bars=payload.spread_lookback_bars,
            spread_top_components=payload.spread_top_components,
            spread_min_edge=payload.spread_min_edge,
            spread_switch_cost_bps=payload.spread_switch_cost_bps,
            spread_include_neutral_overlay=payload.spread_include_neutral_overlay,
            spread_include_regime_switch=payload.spread_include_regime_switch,
            spread_target_vol_annual=payload.spread_target_vol_annual,
            include_patterns=set(payload.include_patterns or []) or None,
            include_model_files=set(payload.include_model_files or []) or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "rows": len(results),
        "results": _table_records(results),
    }


@app.post("/scan")
def scan(payload: ScanRequest) -> dict[str, object]:
    try:
        signals = service.scan(
            interval=payload.interval,
            years=payload.years,
            top_n=payload.top_n,
            refresh_prices=payload.refresh_prices,
            politician_trades_csv=payload.politician_trades_csv,
            include_patterns=set(payload.include_patterns or []) or None,
            include_model_files=set(payload.include_model_files or []) or None,
            min_confidence=payload.min_confidence,
            use_model_thresholds=payload.use_model_thresholds,
            long_threshold=payload.long_threshold,
            short_threshold=payload.short_threshold,
            universe=payload.universe,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "rows": len(signals),
        "signals": _table_records(signals),
    }


@app.get("/coverage")
def coverage(dataset_name: str = "model_dataset") -> dict[str, object]:
    table = service.coverage(dataset_name)
    return {
        "rows": len(table),
        "coverage": _table_records(table),
    }


@app.get("/models")
def models() -> dict[str, object]:
    files = service.model_io.list_models()
    return {
        "count": len(files),
        "models": [str(Path(f).name) for f in files],
    }


@app.get("/models/registry")
def model_registry(interval: str | None = None, pattern: str | None = None) -> dict[str, object]:
    table = service.model_registry(interval=interval, pattern=pattern)
    return {
        "rows": len(table),
        "models": _table_records(table),
    }


@app.get("/models/detail")
def model_detail(model_file: str, top_n_importance: int = 30) -> dict[str, object]:
    try:
        return service.model_details(model_file=model_file, top_n_importance=top_n_importance)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/feature-importance")
def feature_importance(
    interval: str | None = None,
    horizon_bars: int | None = None,
    top_n_per_pattern: int = 20,
) -> dict[str, object]:
    table = service.feature_importance(
        interval=interval,
        horizon_bars=horizon_bars,
        top_n_per_pattern=top_n_per_pattern,
    )
    return {
        "rows": len(table),
        "results": _table_records(table),
    }


@app.get("/thresholds/recommend")
def recommend_thresholds(
    interval: str | None = None,
    patterns: str | None = None,
    model_files: str | None = None,
) -> dict[str, object]:
    pattern_set = {p.strip() for p in str(patterns or "").split(",") if p.strip()} or None
    model_file_set = {m.strip() for m in str(model_files or "").split(",") if m.strip()} or None
    return service.recommend_thresholds(
        interval=interval,
        include_patterns=pattern_set,
        include_model_files=model_file_set,
    )


@app.post("/interval/sweep")
def interval_sweep(payload: SweepIntervalsRequest) -> dict[str, object]:
    try:
        table = service.sweep_intervals(
            intervals=payload.intervals,
            years=payload.years,
            refresh_prices=payload.refresh_prices,
            base_dataset_name=payload.base_dataset_name,
            politician_trades_csv=payload.politician_trades_csv,
            universe=payload.universe,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "rows": len(table),
        "results": _table_records(table),
    }

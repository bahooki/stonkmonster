from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from stonkmodel.api.schemas import BacktestRequest, BuildDatasetRequest, ScanRequest, SweepIntervalsRequest, TrainRequest
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
    payload["polygon_api_key_set"] = bool(settings.polygon_api_key)
    payload.pop("fmp_api_key", None)
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
            long_threshold=payload.long_threshold,
            short_threshold=payload.short_threshold,
            fee_bps=payload.fee_bps,
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


@app.post("/interval/sweep")
def interval_sweep(payload: SweepIntervalsRequest) -> dict[str, object]:
    try:
        table = service.sweep_intervals(
            intervals=payload.intervals,
            years=payload.years,
            refresh_prices=payload.refresh_prices,
            base_dataset_name=payload.base_dataset_name,
            politician_trades_csv=payload.politician_trades_csv,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "rows": len(table),
        "results": _table_records(table),
    }

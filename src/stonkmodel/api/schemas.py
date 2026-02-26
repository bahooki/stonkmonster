from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class BuildDatasetRequest(BaseModel):
    interval: str | None = None
    years: int | None = Field(default=None, ge=1, le=40)
    years_ago_start: int | None = Field(default=None, ge=0, le=40)
    years_ago_end: int | None = Field(default=None, ge=0, le=40)
    refresh_prices: bool = False
    dataset_name: str = "model_dataset"
    politician_trades_csv: Path | None = None
    universe: Literal["sp500", "sp100", "custom"] | None = None


class TrainRequest(BaseModel):
    dataset_name: str = "model_dataset"
    interval: str | None = None
    min_pattern_rows: int | None = Field(default=None, ge=10, le=100000)
    model_name: str | None = None
    parallel_patterns: int | None = Field(default=None, ge=1, le=32)
    fast_mode: bool = False
    max_rows_per_pattern: int | None = Field(default=None, ge=1000, le=2000000)


class BacktestRequest(BaseModel):
    dataset_name: str = "model_dataset"
    interval: str | None = None
    mode: Literal["saved_models", "walk_forward_retrain"] = "saved_models"
    long_threshold: float | None = Field(default=0.55, ge=0.0, le=1.0)
    short_threshold: float | None = Field(default=0.45, ge=0.0, le=1.0)
    fee_bps: float = Field(default=1.0, ge=0.0, le=1000.0)
    use_model_thresholds: bool = False
    spread_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    slippage_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    short_borrow_bps_per_day: float = Field(default=0.0, ge=0.0, le=1000.0)
    latency_bars: int = Field(default=0, ge=0, le=100)
    train_window_days: int = Field(default=504, ge=30, le=3650)
    test_window_days: int = Field(default=63, ge=5, le=730)
    step_days: int = Field(default=21, ge=1, le=365)
    min_pattern_rows: int | None = Field(default=None, ge=10, le=100000)
    fast_mode: bool = False
    parallel_models: int = Field(default=1, ge=1, le=64)
    max_eval_rows_per_pattern: int | None = Field(default=None, ge=1000, le=20000000)
    max_windows_per_pattern: int | None = Field(default=None, ge=1, le=20000)
    max_train_rows_per_window: int | None = Field(default=None, ge=1000, le=20000000)
    include_patterns: list[str] | None = None
    include_model_files: list[str] | None = None


class ScanRequest(BaseModel):
    interval: str | None = None
    years: int = Field(default=2, ge=1, le=20)
    top_n: int = Field(default=50, ge=1, le=500)
    refresh_prices: bool = True
    politician_trades_csv: Path | None = None
    include_patterns: list[str] | None = None
    include_model_files: list[str] | None = None
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    use_model_thresholds: bool = True
    long_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    short_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    universe: Literal["sp500", "sp100", "custom"] | None = None


class SweepIntervalsRequest(BaseModel):
    intervals: list[str] = Field(default_factory=lambda: ["1d", "1h", "30m", "15m"])
    years: int | None = Field(default=None, ge=1, le=40)
    refresh_prices: bool = False
    base_dataset_name: str = "model_dataset"
    politician_trades_csv: Path | None = None
    universe: Literal["sp500", "sp100", "custom"] | None = None

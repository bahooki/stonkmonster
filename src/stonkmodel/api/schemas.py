from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class BuildDatasetRequest(BaseModel):
    interval: str | None = None
    years: int | None = Field(default=None, ge=1, le=40)
    years_ago_start: int | None = Field(default=None, ge=0, le=40)
    years_ago_end: int | None = Field(default=None, ge=0, le=40)
    refresh_prices: bool = False
    dataset_name: str = "model_dataset"
    politician_trades_csv: Path | None = None


class TrainRequest(BaseModel):
    dataset_name: str = "model_dataset"
    interval: str | None = None
    min_pattern_rows: int | None = Field(default=None, ge=10, le=100000)


class BacktestRequest(BaseModel):
    dataset_name: str = "model_dataset"
    interval: str | None = None
    long_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    short_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    fee_bps: float = Field(default=1.0, ge=0.0, le=1000.0)
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


class SweepIntervalsRequest(BaseModel):
    intervals: list[str] = Field(default_factory=lambda: ["1d", "1h", "30m", "15m"])
    years: int | None = Field(default=None, ge=1, le=40)
    refresh_prices: bool = False
    base_dataset_name: str = "model_dataset"
    politician_trades_csv: Path | None = None

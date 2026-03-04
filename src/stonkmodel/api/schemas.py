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
    include_patterns: list[str] | None = None
    candidate_models_per_pattern: int = Field(default=1, ge=1, le=6)


class RecursiveTrainRequest(BaseModel):
    dataset_name: str = "model_dataset"
    interval: str | None = None
    min_pattern_rows: int | None = Field(default=None, ge=10, le=100000)
    base_model_name: str | None = None
    rounds: int = Field(default=2, ge=1, le=10)
    keep_top_patterns: int = Field(default=6, ge=1, le=50)
    min_trades_to_keep: int = Field(default=50, ge=1, le=200000)
    parallel_patterns: int | None = Field(default=None, ge=1, le=32)
    fast_mode: bool = True
    max_rows_per_pattern: int | None = Field(default=None, ge=1000, le=2000000)
    candidate_models_per_pattern: int = Field(default=2, ge=1, le=6)
    fee_bps: float = Field(default=1.0, ge=0.0, le=1000.0)
    spread_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    slippage_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    short_borrow_bps_per_day: float = Field(default=0.0, ge=0.0, le=1000.0)
    latency_bars: int = Field(default=1, ge=0, le=100)
    min_threshold_opt_trades: int = Field(default=40, ge=1, le=200000)
    max_eval_rows_per_pattern: int | None = Field(default=250000, ge=1000, le=20000000)


class AutoImproveRequest(BaseModel):
    dataset_name: str = "model_dataset"
    interval: str | None = None
    iterations: int = Field(default=8, ge=1, le=200)
    max_minutes: int = Field(default=180, ge=5, le=24 * 60)
    patience: int = Field(default=3, ge=1, le=50)
    min_significant_improvement: float = Field(default=0.10, ge=0.0, le=5.0)
    min_iteration_trades: int = Field(default=40, ge=0, le=500000)
    fee_bps: float = Field(default=1.0, ge=0.0, le=1000.0)
    spread_bps: float = Field(default=0.5, ge=0.0, le=1000.0)
    slippage_bps: float = Field(default=0.5, ge=0.0, le=1000.0)
    short_borrow_bps_per_day: float = Field(default=0.0, ge=0.0, le=1000.0)
    latency_bars: int = Field(default=1, ge=0, le=100)
    parallel_patterns: int = Field(default=4, ge=1, le=64)
    include_spread_strategies: bool = True
    random_seed: int = Field(default=42, ge=0, le=1_000_000_000)


class ThresholdOptimizeRequest(BaseModel):
    dataset_name: str = "model_dataset"
    interval: str | None = None
    fee_bps: float = Field(default=1.0, ge=0.0, le=1000.0)
    spread_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    slippage_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    short_borrow_bps_per_day: float = Field(default=0.0, ge=0.0, le=1000.0)
    latency_bars: int = Field(default=1, ge=0, le=100)
    embargo_bars: int = Field(default=1, ge=0, le=100)
    include_patterns: list[str] | None = None
    include_model_files: list[str] | None = None
    min_trades: int = Field(default=40, ge=1, le=200000)
    max_eval_rows_per_pattern: int | None = Field(default=250000, ge=1000, le=20000000)
    parallel_models: int = Field(default=4, ge=1, le=64)
    persist: bool = True


class BacktestRequest(BaseModel):
    dataset_name: str = "model_dataset"
    interval: str | None = None
    mode: Literal["saved_models", "walk_forward_retrain"] = "saved_models"
    long_threshold: float | None = Field(default=0.65, ge=0.0, le=1.0)
    short_threshold: float | None = Field(default=0.35, ge=0.0, le=1.0)
    fee_bps: float = Field(default=1.0, ge=0.0, le=1000.0)
    use_model_thresholds: bool = False
    spread_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    slippage_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    short_borrow_bps_per_day: float = Field(default=0.0, ge=0.0, le=1000.0)
    latency_bars: int = Field(default=1, ge=0, le=100)
    embargo_bars: int = Field(default=1, ge=0, le=100)
    train_window_days: int = Field(default=504, ge=30, le=3650)
    test_window_days: int = Field(default=63, ge=5, le=730)
    step_days: int = Field(default=21, ge=1, le=365)
    min_pattern_rows: int | None = Field(default=None, ge=10, le=100000)
    fast_mode: bool = False
    parallel_models: int = Field(default=1, ge=1, le=64)
    max_eval_rows_per_pattern: int | None = Field(default=None, ge=1000, le=20000000)
    max_windows_per_pattern: int | None = Field(default=None, ge=1, le=20000)
    max_train_rows_per_window: int | None = Field(default=None, ge=1000, le=20000000)
    include_portfolio: bool = True
    portfolio_top_k_per_side: int = Field(default=5, ge=1, le=500)
    portfolio_max_gross_exposure: float = Field(default=1.0, ge=0.0, le=10.0)
    portfolio_pattern_selection: Literal["all", "best", "both"] = "all"
    portfolio_best_patterns_top_n: int = Field(default=6, ge=1, le=50)
    portfolio_min_pattern_trades: int = Field(default=40, ge=0, le=200000)
    portfolio_min_pattern_win_rate_trade: float = Field(default=0.55, ge=0.0, le=1.0)
    portfolio_min_abs_score: float = Field(default=0.15, ge=0.0, le=1.0)
    portfolio_rebalance_every_n_bars: int = Field(default=3, ge=1, le=100)
    portfolio_symbol_cooldown_bars: int = Field(default=5, ge=0, le=1000)
    portfolio_volatility_scaling: bool = True
    portfolio_max_symbol_weight: float = Field(default=0.35, ge=0.01, le=1.0)
    include_spread_strategies: bool = False
    spread_lookback_bars: int = Field(default=63, ge=10, le=5000)
    spread_top_components: int = Field(default=3, ge=1, le=100)
    spread_min_edge: float = Field(default=0.02, ge=0.0, le=5.0)
    spread_switch_cost_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    spread_include_neutral_overlay: bool = True
    spread_include_regime_switch: bool = True
    spread_target_vol_annual: float = Field(default=0.0, ge=0.0, le=5.0)
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

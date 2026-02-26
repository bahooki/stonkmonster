#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stonkmodel.config import get_settings
from stonkmodel.pipeline import StonkService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy backtests on trained pattern models")
    parser.add_argument("--dataset-name", default="model_dataset", help="Existing dataset name")
    parser.add_argument("--interval", default=None, help="Model interval")
    parser.add_argument("--mode", choices=["saved_models", "walk_forward_retrain"], default="saved_models")
    parser.add_argument("--long-threshold", type=float, default=0.55, help="Probability threshold for long entries")
    parser.add_argument("--short-threshold", type=float, default=0.45, help="Probability threshold for short entries")
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Per-trade fee in basis points")
    parser.add_argument("--use-model-thresholds", action="store_true", help="Use tuned thresholds saved with each model")
    parser.add_argument("--spread-bps", type=float, default=0.0, help="Bid/ask spread cost in basis points")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage cost in basis points")
    parser.add_argument("--short-borrow-bps-per-day", type=float, default=0.0, help="Short borrow cost per day in basis points")
    parser.add_argument("--latency-bars", type=int, default=0, help="Execution latency in bars")
    parser.add_argument("--train-window-days", type=int, default=504, help="Walk-forward train window")
    parser.add_argument("--test-window-days", type=int, default=63, help="Walk-forward test window")
    parser.add_argument("--step-days", type=int, default=21, help="Walk-forward step size")
    parser.add_argument("--min-pattern-rows", type=int, default=None, help="Minimum rows per pattern for walk-forward retraining")
    parser.add_argument("--fast-mode", action="store_true", help="Faster backtest settings")
    parser.add_argument("--parallel-models", type=int, default=1, help="Saved-model backtest parallel workers")
    parser.add_argument("--max-eval-rows-per-pattern", type=int, default=None, help="Cap eval rows per pattern in saved-model mode")
    parser.add_argument("--max-windows-per-pattern", type=int, default=None, help="Cap walk-forward windows per pattern")
    parser.add_argument("--max-train-rows-per-window", type=int, default=None, help="Cap walk-forward train rows per window")
    parser.add_argument("--pattern", action="append", default=None, help="Optional pattern filter, repeatable")
    parser.add_argument("--model-file", action="append", default=None, help="Optional model file filter, repeatable")
    args = parser.parse_args()

    service = StonkService(get_settings())
    table = service.backtest(
        dataset_name=args.dataset_name,
        interval=args.interval,
        mode=args.mode,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        fee_bps=args.fee_bps,
        use_model_thresholds=args.use_model_thresholds,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        short_borrow_bps_per_day=args.short_borrow_bps_per_day,
        latency_bars=args.latency_bars,
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days,
        step_days=args.step_days,
        min_pattern_rows=args.min_pattern_rows,
        fast_mode=args.fast_mode,
        parallel_models=args.parallel_models,
        max_eval_rows_per_pattern=args.max_eval_rows_per_pattern,
        max_windows_per_pattern=args.max_windows_per_pattern,
        max_train_rows_per_window=args.max_train_rows_per_window,
        include_patterns=set(args.pattern or []) or None,
        include_model_files=set(args.model_file or []) or None,
    )
    if table.empty:
        print("No backtest results. Train models first.")
        return
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

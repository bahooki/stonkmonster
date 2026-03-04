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
    parser = argparse.ArgumentParser(description="Run automated train/optimize/backtest improvement loop")
    parser.add_argument("--dataset-name", default="model_dataset", help="Existing dataset name")
    parser.add_argument("--interval", default=None, help="Interval tied to model/backtest runs")
    parser.add_argument("--iterations", type=int, default=8, help="Max loop iterations")
    parser.add_argument("--max-minutes", type=int, default=180, help="Wall-clock time limit")
    parser.add_argument("--patience", type=int, default=3, help="Early-stop after this many non-improving iterations")
    parser.add_argument("--min-significant-improvement", type=float, default=0.10, help="Required relative improvement to count")
    parser.add_argument("--min-iteration-trades", type=int, default=40, help="Minimum trades required for an iteration to qualify as best")
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Fee assumption in bps")
    parser.add_argument("--spread-bps", type=float, default=0.5, help="Spread assumption in bps")
    parser.add_argument("--slippage-bps", type=float, default=0.5, help="Slippage assumption in bps")
    parser.add_argument("--short-borrow-bps-per-day", type=float, default=0.0, help="Daily short borrow in bps")
    parser.add_argument("--latency-bars", type=int, default=1, help="Execution latency bars")
    parser.add_argument("--parallel-patterns", type=int, default=4, help="Parallel pattern workers")
    parser.add_argument("--no-spread-overlays", action="store_true", help="Disable spread overlays during evaluation")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for search reproducibility")
    args = parser.parse_args()

    service = StonkService(get_settings())
    table = service.auto_improve(
        dataset_name=args.dataset_name,
        interval=args.interval,
        iterations=args.iterations,
        max_minutes=args.max_minutes,
        patience=args.patience,
        min_significant_improvement=args.min_significant_improvement,
        min_iteration_trades=args.min_iteration_trades,
        fee_bps=args.fee_bps,
        spread_bps=args.spread_bps,
        slippage_bps=args.slippage_bps,
        short_borrow_bps_per_day=args.short_borrow_bps_per_day,
        latency_bars=args.latency_bars,
        parallel_patterns=args.parallel_patterns,
        include_spread_strategies=(not args.no_spread_overlays),
        random_seed=args.random_seed,
    )
    if table.empty:
        print("No autopilot history rows generated.")
        return
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

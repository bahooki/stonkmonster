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
    parser.add_argument("--long-threshold", type=float, default=0.55, help="Probability threshold for long entries")
    parser.add_argument("--short-threshold", type=float, default=0.45, help="Probability threshold for short entries")
    parser.add_argument("--fee-bps", type=float, default=1.0, help="Per-trade fee in basis points")
    parser.add_argument("--pattern", action="append", default=None, help="Optional pattern filter, repeatable")
    parser.add_argument("--model-file", action="append", default=None, help="Optional model file filter, repeatable")
    args = parser.parse_args()

    service = StonkService(get_settings())
    table = service.backtest(
        dataset_name=args.dataset_name,
        interval=args.interval,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        fee_bps=args.fee_bps,
        include_patterns=set(args.pattern or []) or None,
        include_model_files=set(args.model_file or []) or None,
    )
    if table.empty:
        print("No backtest results. Train models first.")
        return
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

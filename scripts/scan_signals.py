#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stonkmodel.config import get_settings
from stonkmodel.pipeline import StonkService


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan latest bars for top pattern-model signals")
    parser.add_argument("--interval", default=None, help="Bar interval")
    parser.add_argument("--years", type=int, default=2, help="History window used for feature context")
    parser.add_argument("--top-n", type=int, default=50, help="Top N signals")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum signal confidence")
    parser.add_argument("--no-refresh", action="store_true", help="Use cached prices only")
    parser.add_argument("--politician-trades-csv", type=Path, default=None, help="Optional politician trades CSV")
    parser.add_argument("--use-model-thresholds", action="store_true", help="Use tuned thresholds from model metadata")
    parser.add_argument("--long-threshold", type=float, default=None, help="Override long threshold")
    parser.add_argument("--short-threshold", type=float, default=None, help="Override short threshold")
    parser.add_argument("--universe", choices=["sp500", "sp100", "custom"], default=None, help="Universe override")
    parser.add_argument("--pattern", action="append", default=None, help="Optional pattern filter, repeatable")
    parser.add_argument("--model-file", action="append", default=None, help="Optional model file filter, repeatable")
    args = parser.parse_args()

    service = StonkService(get_settings())
    signals = service.scan(
        interval=args.interval,
        years=args.years,
        top_n=args.top_n,
        refresh_prices=not args.no_refresh,
        politician_trades_csv=args.politician_trades_csv,
        include_patterns=set(args.pattern or []) or None,
        include_model_files=set(args.model_file or []) or None,
        min_confidence=args.min_confidence,
        use_model_thresholds=args.use_model_thresholds,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        universe=args.universe,
    )
    if signals.empty:
        print("No signals. Ensure models are trained for this interval/horizon.")
        return

    print(signals.to_string(index=False))


if __name__ == "__main__":
    main()

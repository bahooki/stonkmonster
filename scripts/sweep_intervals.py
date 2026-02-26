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
    parser = argparse.ArgumentParser(description="Find the strongest interval by training + backtesting each interval")
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["1d", "1h", "30m", "15m"],
        help="Intervals to evaluate",
    )
    parser.add_argument("--years", type=int, default=None, help="History years")
    parser.add_argument("--refresh-prices", action="store_true", help="Force refresh from providers")
    parser.add_argument("--base-dataset-name", default="model_dataset", help="Dataset name prefix")
    parser.add_argument("--politician-trades-csv", type=Path, default=None, help="Optional politician trades CSV")
    parser.add_argument("--universe", choices=["sp500", "sp100", "custom"], default=None, help="Universe override")
    args = parser.parse_args()

    service = StonkService(get_settings())
    table = service.sweep_intervals(
        intervals=args.intervals,
        years=args.years,
        refresh_prices=args.refresh_prices,
        base_dataset_name=args.base_dataset_name,
        politician_trades_csv=args.politician_trades_csv,
        universe=args.universe,
    )

    if table.empty:
        print("No interval results produced.")
        return

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

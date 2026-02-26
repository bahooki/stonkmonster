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
    parser = argparse.ArgumentParser(description="Build historical feature dataset for pattern models")
    parser.add_argument("--interval", default=None, help="Bar interval (e.g. 1d, 1h, 15m)")
    parser.add_argument("--years", type=int, default=None, help="History years")
    parser.add_argument("--years-ago-start", type=int, default=None, help="Keep data newer than this many years ago")
    parser.add_argument("--years-ago-end", type=int, default=None, help="Keep data older than this many years ago")
    parser.add_argument("--refresh-prices", action="store_true", help="Force fresh market data pull")
    parser.add_argument("--dataset-name", default="model_dataset", help="Output dataset name")
    parser.add_argument("--politician-trades-csv", type=Path, default=None, help="Optional normalized politician trades CSV")
    parser.add_argument("--universe", choices=["sp500", "sp100", "custom"], default=None, help="Universe override")
    args = parser.parse_args()

    service = StonkService(get_settings())
    result = service.build_dataset(
        interval=args.interval,
        years=args.years,
        refresh_prices=args.refresh_prices,
        dataset_name=args.dataset_name,
        politician_trades_csv=args.politician_trades_csv,
        universe=args.universe,
        years_ago_start=args.years_ago_start,
        years_ago_end=args.years_ago_end,
    )

    print(
        " ".join(
            [
                f"dataset={result.dataset_path}",
                f"universe={result.universe}",
                f"years_ago_start={result.years_ago_start}",
                f"years_ago_end={result.years_ago_end}",
                f"rows={result.rows}",
                f"symbols_loaded={result.symbols_loaded}/{result.symbols_requested}",
            ]
        )
    )


if __name__ == "__main__":
    main()

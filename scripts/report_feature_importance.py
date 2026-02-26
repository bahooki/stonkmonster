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
    parser = argparse.ArgumentParser(description="Show automated feature-importance reports from trained pattern models")
    parser.add_argument("--interval", default=None, help="Filter by interval")
    parser.add_argument("--horizon-bars", type=int, default=None, help="Filter by horizon bars")
    parser.add_argument("--top-n", type=int, default=20, help="Top N features per pattern")
    parser.add_argument("--pattern", default=None, help="Optional single pattern filter")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional output CSV path")
    args = parser.parse_args()

    service = StonkService(get_settings())
    table = service.feature_importance(
        interval=args.interval,
        horizon_bars=args.horizon_bars,
        top_n_per_pattern=args.top_n,
    )

    if args.pattern:
        table = table.loc[table["pattern"] == args.pattern]

    if table.empty:
        print("No feature-importance reports found. Train models first.")
        return

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out_csv, index=False)
        print(f"saved={args.out_csv}")

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

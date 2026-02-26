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
    parser = argparse.ArgumentParser(description="Train per-pattern stacked models")
    parser.add_argument("--dataset-name", default="model_dataset", help="Existing dataset name")
    parser.add_argument("--interval", default=None, help="Interval tied to model names")
    parser.add_argument("--min-pattern-rows", type=int, default=None, help="Override minimum pattern rows required")
    args = parser.parse_args()

    service = StonkService(get_settings())
    summary = service.train(
        dataset_name=args.dataset_name,
        interval=args.interval,
        min_pattern_rows=args.min_pattern_rows,
    )
    if summary.empty:
        print("No pattern models trained. Increase history or lower min pattern count.")
        return

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

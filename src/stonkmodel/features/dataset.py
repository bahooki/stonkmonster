from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from stonkmodel.data.external_features import (
    build_fundamental_table,
    engineer_politician_features,
    load_politician_trades,
    merge_external_features,
)
from stonkmodel.data.market_data import merge_frames
from stonkmodel.features.indicators import add_indicators, infer_feature_columns
from stonkmodel.features.labels import add_forward_labels, add_train_test_split
from stonkmodel.features.patterns import PATTERN_COLUMNS, add_candlestick_patterns


@dataclass
class DatasetOptions:
    horizon_bars: int = 1
    return_threshold: float = 0.0
    split_date: str | None = None
    politician_trades_csv: Path | None = None
    include_fundamentals: bool = True
    include_politician_trades: bool = True
    fundamentals_provider: Literal["auto", "fmp", "yfinance"] = "auto"
    fmp_api_key: str | None = None
    fmp_base_url: str = "https://financialmodelingprep.com/stable"
    request_workers: int = 8


class DatasetBuilder:
    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def build(self, frames: dict[str, pd.DataFrame], options: DatasetOptions) -> pd.DataFrame:
        base = merge_frames(frames)
        if base.empty:
            return base

        work = base.sort_values(["symbol", "datetime"]).reset_index(drop=True)

        work = add_candlestick_patterns(work)
        work = add_indicators(work)

        fundamentals = None
        if options.include_fundamentals:
            fundamentals = build_fundamental_table(
                symbols=sorted(work["symbol"].unique().tolist()),
                cache_path=self.processed_dir / "fundamentals.parquet",
                refresh=False,
                provider=options.fundamentals_provider,
                fmp_api_key=options.fmp_api_key,
                fmp_base_url=options.fmp_base_url,
                request_workers=options.request_workers,
            )

        politician = None
        if options.include_politician_trades and options.politician_trades_csv:
            raw = load_politician_trades(options.politician_trades_csv)
            if not raw.empty:
                politician = engineer_politician_features(raw)

        work = merge_external_features(work, fundamental_table=fundamentals, politician_features=politician)

        work = add_forward_labels(work, horizon_bars=options.horizon_bars, threshold=options.return_threshold)
        work = add_train_test_split(work, split_date=options.split_date)

        work = work.replace([np.inf, -np.inf], np.nan)
        work = work.dropna(subset=["future_return", "future_direction"]).reset_index(drop=True)
        return work

    def save_dataset(self, dataset: pd.DataFrame, name: str = "model_dataset") -> Path:
        path = self.processed_dir / f"{name}.parquet"
        dataset.to_parquet(path, index=False)
        return path

    def load_dataset(self, name: str = "model_dataset") -> pd.DataFrame:
        path = self.processed_dir / f"{name}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def list_datasets(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for path in sorted(self.processed_dir.glob("*.parquet")):
            name = path.stem
            stat = path.stat()
            row: dict[str, object] = {
                "dataset_name": name,
                "file": path.name,
                "path": str(path),
                "size_mb": round(stat.st_size / (1024 * 1024), 3),
                "modified_utc": pd.to_datetime(stat.st_mtime, unit="s", utc=True).isoformat(),
                "rows": np.nan,
                "symbols": np.nan,
                "start_datetime": None,
                "end_datetime": None,
            }
            try:
                frame = pd.read_parquet(path, columns=["symbol", "datetime"])
                row["rows"] = int(len(frame))
                if "symbol" in frame.columns:
                    row["symbols"] = int(frame["symbol"].nunique())
                if "datetime" in frame.columns and not frame.empty:
                    start_dt = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").min()
                    end_dt = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").max()
                    row["start_datetime"] = start_dt.isoformat() if pd.notna(start_dt) else None
                    row["end_datetime"] = end_dt.isoformat() if pd.notna(end_dt) else None
            except Exception:
                pass
            rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "dataset_name",
                    "file",
                    "path",
                    "size_mb",
                    "modified_utc",
                    "rows",
                    "symbols",
                    "start_datetime",
                    "end_datetime",
                ]
            )

        return pd.DataFrame(rows).sort_values("modified_utc", ascending=False).reset_index(drop=True)

    def dataset_summary(self, name: str) -> dict[str, object]:
        frame = self.load_dataset(name)
        if frame.empty:
            return {
                "dataset_name": name,
                "rows": 0,
                "columns": 0,
                "symbols": 0,
                "start_datetime": None,
                "end_datetime": None,
            }

        return {
            "dataset_name": name,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
            "start_datetime": pd.to_datetime(frame["datetime"], utc=True, errors="coerce").min().isoformat()
            if "datetime" in frame.columns
            else None,
            "end_datetime": pd.to_datetime(frame["datetime"], utc=True, errors="coerce").max().isoformat()
            if "datetime" in frame.columns
            else None,
            "has_labels": bool({"future_return", "future_direction"}.issubset(frame.columns)),
            "columns_list": list(frame.columns),
        }


def get_pattern_datasets(dataset: pd.DataFrame, min_rows: int = 100) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if dataset.empty:
        return out

    for col in PATTERN_COLUMNS:
        pattern_name = col.replace("pattern_", "")
        subset = dataset.loc[dataset[col] == 1].copy()
        if len(subset) >= min_rows:
            out[pattern_name] = subset

    none_subset = dataset.loc[dataset["pattern"].isna()].copy()
    if len(none_subset) >= min_rows:
        out["none"] = none_subset

    return out


def split_xy(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = infer_feature_columns(dataset, exclude=PATTERN_COLUMNS)
    x = dataset[feature_cols].copy()
    y = dataset["future_direction"].astype(int)

    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0.0)
    return x, y, feature_cols

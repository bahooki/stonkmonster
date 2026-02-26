from __future__ import annotations

import pandas as pd


def add_forward_labels(frame: pd.DataFrame, horizon_bars: int = 1, threshold: float = 0.0) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
    groups: list[pd.DataFrame] = []
    for _, g in out.groupby("symbol", sort=False):
        x = g.copy()
        x["future_return"] = x["close"].shift(-horizon_bars) / x["close"] - 1.0
        x["future_direction"] = (x["future_return"] > threshold).astype("Int64")
        groups.append(x)

    labeled = pd.concat(groups, ignore_index=True)
    return labeled


def add_train_test_split(frame: pd.DataFrame, split_date: str | None) -> pd.DataFrame:
    out = frame.copy()
    if split_date is None:
        # Default split is chronological by datetime, not row position,
        # to keep train/test days strictly separated.
        dt = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
        unique_dt = pd.Series(dt.dropna().unique()).sort_values().reset_index(drop=True)

        if len(unique_dt) >= 2:
            cutoff_pos = max(1, int(len(unique_dt) * 0.8))
            cutoff_pos = min(cutoff_pos, len(unique_dt) - 1)
            cutoff_dt = unique_dt.iloc[cutoff_pos]
            out["split"] = "train"
            out.loc[dt >= cutoff_dt, "split"] = "test"
            return out

        out["split"] = "train"
        cutoff_idx = int(len(out) * 0.8)
        out.loc[out.index >= cutoff_idx, "split"] = "test"
        return out

    dt = pd.to_datetime(split_date, utc=True)
    out["split"] = "train"
    out.loc[out["datetime"] >= dt, "split"] = "test"
    return out

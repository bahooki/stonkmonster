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
        out["split"] = "train"
        cutoff_idx = int(len(out) * 0.8)
        out.loc[out.index >= cutoff_idx, "split"] = "test"
        return out

    dt = pd.to_datetime(split_date, utc=True)
    out["split"] = "train"
    out.loc[out["datetime"] >= dt, "split"] = "test"
    return out

from __future__ import annotations

import pandas as pd


def add_forward_labels(
    frame: pd.DataFrame,
    horizon_bars: int = 1,
    threshold: float = 0.0,
    label_mode: str = "excess",
) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
    groups: list[pd.DataFrame] = []
    for _, g in out.groupby("symbol", sort=False):
        x = g.copy()
        price_col = "adj_close" if "adj_close" in x.columns else "close"
        x["future_return"] = x[price_col].shift(-horizon_bars) / x[price_col] - 1.0
        groups.append(x)

    labeled = pd.concat(groups, ignore_index=True)
    dt = pd.to_datetime(labeled["datetime"], utc=True, errors="coerce")
    labeled["future_market_return"] = labeled.groupby(dt)["future_return"].transform("mean")
    labeled["future_excess_return"] = labeled["future_return"] - labeled["future_market_return"]
    labeled["future_rank_pct"] = labeled.groupby(dt)["future_return"].rank(method="average", pct=True)

    mode = str(label_mode or "excess").strip().lower()
    if mode == "absolute":
        labeled["future_direction"] = (labeled["future_return"] > float(threshold)).astype("Int64")
    elif mode == "cross_sectional":
        rank_threshold = float(threshold)
        if not (0.0 <= rank_threshold <= 1.0):
            rank_threshold = 0.5
        labeled["future_direction"] = (labeled["future_rank_pct"] > rank_threshold).astype("Int64")
    else:
        labeled["future_direction"] = (labeled["future_excess_return"] > float(threshold)).astype("Int64")
    return labeled


def add_train_test_split(frame: pd.DataFrame, split_date: str | None) -> pd.DataFrame:
    out = frame.copy()
    if "datetime" not in out.columns:
        raise ValueError("Input frame must include a `datetime` column")
    dt = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out["datetime"] = dt
    out = out.dropna(subset=["datetime"]).copy()
    if out.empty:
        out["split"] = []
        return out

    if split_date is None:
        # Default split is chronological by datetime, not row position,
        # to keep train/test days strictly separated.
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

    split_dt = pd.to_datetime(split_date, utc=True, errors="coerce")
    if pd.isna(split_dt):
        raise ValueError(f"Invalid split_date: {split_date}")
    out["split"] = "train"
    out.loc[dt >= split_dt, "split"] = "test"
    return out

from __future__ import annotations

import numpy as np
import pandas as pd

PATTERN_COLUMNS = [
    "pattern_bullish_engulfing",
    "pattern_bearish_engulfing",
    "pattern_hammer",
    "pattern_shooting_star",
    "pattern_doji",
    "pattern_morning_star",
    "pattern_evening_star",
    "pattern_bullish_harami",
    "pattern_bearish_harami",
    "pattern_piercing_line",
    "pattern_dark_cloud_cover",
]


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _upper_shadow(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["close", "open"]].max(axis=1)


def _lower_shadow(df: pd.DataFrame) -> pd.Series:
    return df[["close", "open"]].min(axis=1) - df["low"]


def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary candlestick pattern columns for each row."""
    if df.empty:
        return df

    frame = df.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
    output: list[pd.DataFrame] = []

    for _, group in frame.groupby("symbol", sort=False):
        g = group.copy()

        prev_open = g["open"].shift(1)
        prev_close = g["close"].shift(1)
        prev_high = g["high"].shift(1)
        prev_low = g["low"].shift(1)

        body = _body(g)
        prev_body = body.shift(1)
        full_range = (g["high"] - g["low"]).replace(0, np.nan)

        bullish = g["close"] > g["open"]
        bearish = g["close"] < g["open"]
        prev_bullish = prev_close > prev_open
        prev_bearish = prev_close < prev_open

        g["pattern_bullish_engulfing"] = (
            bullish
            & prev_bearish
            & (g["open"] <= prev_close)
            & (g["close"] >= prev_open)
            & (body > prev_body)
        ).astype(int)

        g["pattern_bearish_engulfing"] = (
            bearish
            & prev_bullish
            & (g["open"] >= prev_close)
            & (g["close"] <= prev_open)
            & (body > prev_body)
        ).astype(int)

        lower = _lower_shadow(g)
        upper = _upper_shadow(g)

        g["pattern_hammer"] = (
            (lower > (2.2 * body))
            & (upper < (0.6 * body + 1e-9))
            & (body / full_range < 0.45)
        ).astype(int)

        g["pattern_shooting_star"] = (
            (upper > (2.2 * body))
            & (lower < (0.6 * body + 1e-9))
            & (body / full_range < 0.45)
        ).astype(int)

        g["pattern_doji"] = ((body / full_range) <= 0.1).fillna(0).astype(int)

        prev2_open = g["open"].shift(2)
        prev2_close = g["close"].shift(2)
        prev2_body = body.shift(2)

        # Morning star: bearish long candle, then indecision, then bullish recovery.
        g["pattern_morning_star"] = (
            (prev2_close < prev2_open)
            & (prev2_body > prev2_body.rolling(20, min_periods=1).median())
            & (prev_body < prev2_body * 0.6)
            & bullish
            & (g["close"] > (prev2_open + prev2_close) / 2)
        ).astype(int)

        # Evening star: bullish long candle, then indecision, then bearish reversal.
        g["pattern_evening_star"] = (
            (prev2_close > prev2_open)
            & (prev2_body > prev2_body.rolling(20, min_periods=1).median())
            & (prev_body < prev2_body * 0.6)
            & bearish
            & (g["close"] < (prev2_open + prev2_close) / 2)
        ).astype(int)

        g["pattern_bullish_harami"] = (
            prev_bearish
            & bullish
            & (g["open"] > prev_close)
            & (g["close"] < prev_open)
            & (body < prev_body)
        ).astype(int)

        g["pattern_bearish_harami"] = (
            prev_bullish
            & bearish
            & (g["open"] < prev_close)
            & (g["close"] > prev_open)
            & (body < prev_body)
        ).astype(int)

        g["pattern_piercing_line"] = (
            prev_bearish
            & bullish
            & (g["open"] < prev_low)
            & (g["close"] > (prev_open + prev_close) / 2)
            & (g["close"] < prev_open)
        ).astype(int)

        g["pattern_dark_cloud_cover"] = (
            prev_bullish
            & bearish
            & (g["open"] > prev_high)
            & (g["close"] < (prev_open + prev_close) / 2)
            & (g["close"] > prev_open)
        ).astype(int)

        output.append(g)

    out = pd.concat(output, ignore_index=True)

    def _assign_pattern(row: pd.Series) -> str | None:
        for col in PATTERN_COLUMNS:
            if row.get(col, 0) == 1:
                return col.replace("pattern_", "")
        return None

    out["pattern"] = out[PATTERN_COLUMNS].apply(_assign_pattern, axis=1)
    return out


def pattern_hits(frame: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
    col = f"pattern_{pattern_name}"
    if col not in frame.columns:
        raise ValueError(f"Pattern `{pattern_name}` not found")
    return frame.loc[frame[col] == 1].copy()

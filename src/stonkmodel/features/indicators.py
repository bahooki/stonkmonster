from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pandas_ta as pandas_ta_lib
except Exception:  # pragma: no cover - optional dependency handling
    pandas_ta_lib = None

try:
    import ta as ta_lib
except Exception:  # pragma: no cover - optional dependency handling
    ta_lib = None


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window=window, min_periods=max(5, window // 3)).std().replace(0, np.nan)
    return (series - mean) / std


def _add_base_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    close = g["close"]
    high = g["high"]
    low = g["low"]
    volume = g["volume"].replace(0, np.nan)

    g["ret_1"] = close.pct_change(1)
    g["ret_2"] = close.pct_change(2)
    g["ret_3"] = close.pct_change(3)
    g["ret_5"] = close.pct_change(5)
    g["ret_10"] = close.pct_change(10)
    g["log_ret_1"] = np.log(close / close.shift(1))

    g["hl_spread"] = (high - low) / close
    g["co_spread"] = (g["close"] - g["open"]) / g["open"].replace(0, np.nan)
    g["gap"] = (g["open"] - g["close"].shift(1)) / g["close"].shift(1)

    for window in (5, 10, 20, 50, 100, 200):
        g[f"volatility_{window}"] = g["ret_1"].rolling(window=window, min_periods=max(5, window // 2)).std()
        g[f"z_close_{window}"] = _rolling_zscore(close, window)
        g[f"z_volume_{window}"] = _rolling_zscore(volume, window)

    g["range_to_atr20_proxy"] = (high - low) / (
        (high - low).rolling(20, min_periods=5).mean().replace(0, np.nan)
    )
    g["up_streak"] = (g["ret_1"] > 0).astype(int).groupby((g["ret_1"] <= 0).cumsum()).cumsum()
    g["down_streak"] = (g["ret_1"] < 0).astype(int).groupby((g["ret_1"] >= 0).cumsum()).cumsum()
    return g


def _safe_add_indicator(frame: pd.DataFrame, builder) -> None:
    try:
        new_cols = builder(frame)
        if isinstance(new_cols, pd.DataFrame):
            for col in new_cols.columns:
                frame[col] = new_cols[col]
        elif isinstance(new_cols, pd.Series):
            frame[new_cols.name] = new_cols
    except Exception:
        # Indicator failures are tolerated; many TA functions fail on short histories.
        return


def _apply_pandas_ta(frame: pd.DataFrame) -> pd.DataFrame:
    if pandas_ta_lib is None:
        return frame

    ta = pandas_ta_lib
    f = frame.copy()

    for length in (5, 8, 10, 14, 20, 21, 34, 50, 89, 100, 144, 200):
        _safe_add_indicator(f, lambda x, length=length: ta.sma(x["close"], length=length).rename(f"sma_{length}"))
        _safe_add_indicator(f, lambda x, length=length: ta.ema(x["close"], length=length).rename(f"ema_{length}"))
        _safe_add_indicator(f, lambda x, length=length: ta.wma(x["close"], length=length).rename(f"wma_{length}"))

    _safe_add_indicator(f, lambda x: ta.hma(x["close"], length=21).rename("hma_21"))
    _safe_add_indicator(f, lambda x: ta.hma(x["close"], length=55).rename("hma_55"))

    for length in (7, 14, 21):
        _safe_add_indicator(f, lambda x, length=length: ta.rsi(x["close"], length=length).rename(f"rsi_{length}"))
        _safe_add_indicator(f, lambda x, length=length: ta.cci(x["high"], x["low"], x["close"], length=length).rename(f"cci_{length}"))
        _safe_add_indicator(f, lambda x, length=length: ta.roc(x["close"], length=length).rename(f"roc_{length}"))
        _safe_add_indicator(
            f,
            lambda x, length=length: ta.willr(x["high"], x["low"], x["close"], length=length).rename(f"willr_{length}"),
        )

    _safe_add_indicator(f, lambda x: ta.stoch(x["high"], x["low"], x["close"]))
    _safe_add_indicator(f, lambda x: ta.stochrsi(x["close"]))
    _safe_add_indicator(f, lambda x: ta.macd(x["close"]))
    _safe_add_indicator(f, lambda x: ta.ppo(x["close"]))
    _safe_add_indicator(f, lambda x: ta.trix(x["close"]))
    _safe_add_indicator(f, lambda x: ta.kst(x["close"]))

    _safe_add_indicator(f, lambda x: ta.atr(x["high"], x["low"], x["close"], length=14).rename("atr_14"))
    _safe_add_indicator(f, lambda x: ta.natr(x["high"], x["low"], x["close"], length=14).rename("natr_14"))
    _safe_add_indicator(f, lambda x: ta.adx(x["high"], x["low"], x["close"], length=14))
    _safe_add_indicator(f, lambda x: ta.psar(x["high"], x["low"], x["close"]))
    _safe_add_indicator(f, lambda x: ta.supertrend(x["high"], x["low"], x["close"], length=10, multiplier=3.0))

    _safe_add_indicator(f, lambda x: ta.bbands(x["close"], length=20, std=2.0))
    _safe_add_indicator(f, lambda x: ta.kc(x["high"], x["low"], x["close"], length=20))
    _safe_add_indicator(f, lambda x: ta.donchian(x["high"], x["low"], lower_length=20, upper_length=20))

    _safe_add_indicator(f, lambda x: ta.obv(x["close"], x["volume"]).rename("obv"))
    _safe_add_indicator(f, lambda x: ta.mfi(x["high"], x["low"], x["close"], x["volume"], length=14).rename("mfi_14"))
    _safe_add_indicator(f, lambda x: ta.cmf(x["high"], x["low"], x["close"], x["volume"], length=20).rename("cmf_20"))
    _safe_add_indicator(f, lambda x: ta.pvt(x["close"], x["volume"]).rename("pvt"))
    _safe_add_indicator(f, lambda x: ta.eom(x["high"], x["low"], x["close"], x["volume"], length=14).rename("eom_14"))
    _safe_add_indicator(f, lambda x: ta.vortex(x["high"], x["low"], x["close"], length=14))

    _safe_add_indicator(f, lambda x: ta.er(x["close"], length=10).rename("efficiency_ratio_10"))
    _safe_add_indicator(f, lambda x: ta.ui(x["close"], length=14).rename("ulcer_index_14"))
    _safe_add_indicator(f, lambda x: ta.entropy(x["close"], length=10).rename("entropy_10"))

    return f


def _apply_ta_fallback(frame: pd.DataFrame) -> pd.DataFrame:
    if ta_lib is None:
        return frame

    f = frame.copy()

    for length in (5, 8, 10, 14, 20, 21, 34, 50, 89, 100, 144, 200):
        _safe_add_indicator(
            f,
            lambda x, length=length: ta_lib.trend.sma_indicator(close=x["close"], window=length).rename(f"sma_{length}"),
        )
        _safe_add_indicator(
            f,
            lambda x, length=length: ta_lib.trend.ema_indicator(close=x["close"], window=length).rename(f"ema_{length}"),
        )
        _safe_add_indicator(
            f,
            lambda x, length=length: ta_lib.trend.wma_indicator(close=x["close"], window=length).rename(f"wma_{length}"),
        )

    for length in (7, 14, 21):
        _safe_add_indicator(
            f,
            lambda x, length=length: ta_lib.momentum.rsi(close=x["close"], window=length).rename(f"rsi_{length}"),
        )
        _safe_add_indicator(
            f,
            lambda x, length=length: ta_lib.trend.cci(
                high=x["high"],
                low=x["low"],
                close=x["close"],
                window=length,
            ).rename(f"cci_{length}"),
        )
        _safe_add_indicator(
            f,
            lambda x, length=length: ta_lib.momentum.roc(close=x["close"], window=length).rename(f"roc_{length}"),
        )
        _safe_add_indicator(
            f,
            lambda x, length=length: ta_lib.momentum.williams_r(
                high=x["high"],
                low=x["low"],
                close=x["close"],
                lbp=length,
            ).rename(f"willr_{length}"),
        )

    _safe_add_indicator(
        f,
        lambda x: ta_lib.momentum.stoch(high=x["high"], low=x["low"], close=x["close"], window=14, smooth_window=3).rename(
            "stoch_k_14"
        ),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.momentum.stoch_signal(
            high=x["high"],
            low=x["low"],
            close=x["close"],
            window=14,
            smooth_window=3,
        ).rename("stoch_d_14"),
    )

    _safe_add_indicator(f, lambda x: ta_lib.momentum.stochrsi(close=x["close"], window=14).rename("stochrsi_14"))
    _safe_add_indicator(f, lambda x: ta_lib.momentum.stochrsi_k(close=x["close"], window=14).rename("stochrsi_k_14"))
    _safe_add_indicator(f, lambda x: ta_lib.momentum.stochrsi_d(close=x["close"], window=14).rename("stochrsi_d_14"))

    _safe_add_indicator(f, lambda x: ta_lib.trend.macd(close=x["close"], window_fast=12, window_slow=26).rename("macd_12_26"))
    _safe_add_indicator(f, lambda x: ta_lib.trend.macd_signal(close=x["close"], window_fast=12, window_slow=26, window_sign=9).rename("macd_signal_9"))
    _safe_add_indicator(f, lambda x: ta_lib.trend.macd_diff(close=x["close"], window_fast=12, window_slow=26, window_sign=9).rename("macd_diff_9"))

    _safe_add_indicator(f, lambda x: ta_lib.momentum.ppo(close=x["close"], window_slow=26, window_fast=12).rename("ppo_12_26"))
    _safe_add_indicator(f, lambda x: ta_lib.momentum.ppo_signal(close=x["close"], window_slow=26, window_fast=12, window_sign=9).rename("ppo_signal_9"))
    _safe_add_indicator(f, lambda x: ta_lib.momentum.ppo_hist(close=x["close"], window_slow=26, window_fast=12, window_sign=9).rename("ppo_hist_9"))

    _safe_add_indicator(f, lambda x: ta_lib.trend.trix(close=x["close"], window=15).rename("trix_15"))
    _safe_add_indicator(f, lambda x: ta_lib.trend.kst(close=x["close"]).rename("kst"))
    _safe_add_indicator(f, lambda x: ta_lib.trend.kst_sig(close=x["close"]).rename("kst_signal"))

    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.average_true_range(
            high=x["high"],
            low=x["low"],
            close=x["close"],
            window=14,
        ).rename("atr_14"),
    )
    _safe_add_indicator(
        f,
        lambda x: (ta_lib.volatility.average_true_range(high=x["high"], low=x["low"], close=x["close"], window=14)
        / x["close"].replace(0, np.nan)).rename("natr_14"),
    )

    _safe_add_indicator(
        f,
        lambda x: ta_lib.trend.adx(high=x["high"], low=x["low"], close=x["close"], window=14).rename("adx_14"),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.trend.adx_pos(high=x["high"], low=x["low"], close=x["close"], window=14).rename("adx_pos_14"),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.trend.adx_neg(high=x["high"], low=x["low"], close=x["close"], window=14).rename("adx_neg_14"),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.trend.psar_up(high=x["high"], low=x["low"], close=x["close"], step=0.02, max_step=0.2).rename(
            "psar_up"
        ),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.trend.psar_down(high=x["high"], low=x["low"], close=x["close"], step=0.02, max_step=0.2).rename(
            "psar_down"
        ),
    )

    _safe_add_indicator(f, lambda x: ta_lib.volatility.bollinger_mavg(close=x["close"], window=20).rename("bb_mavg_20_2"))
    _safe_add_indicator(f, lambda x: ta_lib.volatility.bollinger_hband(close=x["close"], window=20, window_dev=2).rename("bb_hband_20_2"))
    _safe_add_indicator(f, lambda x: ta_lib.volatility.bollinger_lband(close=x["close"], window=20, window_dev=2).rename("bb_lband_20_2"))
    _safe_add_indicator(f, lambda x: ta_lib.volatility.bollinger_pband(close=x["close"], window=20, window_dev=2).rename("bb_pband_20_2"))
    _safe_add_indicator(f, lambda x: ta_lib.volatility.bollinger_wband(close=x["close"], window=20, window_dev=2).rename("bb_wband_20_2"))

    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.keltner_channel_mband(high=x["high"], low=x["low"], close=x["close"], window=20).rename(
            "kc_mband_20"
        ),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.keltner_channel_hband(high=x["high"], low=x["low"], close=x["close"], window=20).rename(
            "kc_hband_20"
        ),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.keltner_channel_lband(high=x["high"], low=x["low"], close=x["close"], window=20).rename(
            "kc_lband_20"
        ),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.keltner_channel_wband(high=x["high"], low=x["low"], close=x["close"], window=20).rename(
            "kc_wband_20"
        ),
    )

    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.donchian_channel_hband(high=x["high"], low=x["low"], close=x["close"], window=20).rename(
            "donchian_hband_20"
        ),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.donchian_channel_lband(high=x["high"], low=x["low"], close=x["close"], window=20).rename(
            "donchian_lband_20"
        ),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.donchian_channel_mband(high=x["high"], low=x["low"], close=x["close"], window=20).rename(
            "donchian_mband_20"
        ),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.donchian_channel_wband(high=x["high"], low=x["low"], close=x["close"], window=20).rename(
            "donchian_wband_20"
        ),
    )

    _safe_add_indicator(f, lambda x: ta_lib.volume.on_balance_volume(close=x["close"], volume=x["volume"]).rename("obv"))
    _safe_add_indicator(
        f,
        lambda x: ta_lib.volume.money_flow_index(
            high=x["high"],
            low=x["low"],
            close=x["close"],
            volume=x["volume"],
            window=14,
        ).rename("mfi_14"),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.volume.chaikin_money_flow(
            high=x["high"],
            low=x["low"],
            close=x["close"],
            volume=x["volume"],
            window=20,
        ).rename("cmf_20"),
    )
    _safe_add_indicator(f, lambda x: ta_lib.volume.volume_price_trend(close=x["close"], volume=x["volume"]).rename("pvt"))
    _safe_add_indicator(
        f,
        lambda x: ta_lib.volume.ease_of_movement(
            high=x["high"],
            low=x["low"],
            volume=x["volume"],
            window=14,
        ).rename("eom_14"),
    )

    _safe_add_indicator(
        f,
        lambda x: ta_lib.trend.vortex_indicator_pos(high=x["high"], low=x["low"], close=x["close"], window=14).rename(
            "vortex_pos_14"
        ),
    )
    _safe_add_indicator(
        f,
        lambda x: ta_lib.trend.vortex_indicator_neg(high=x["high"], low=x["low"], close=x["close"], window=14).rename(
            "vortex_neg_14"
        ),
    )

    _safe_add_indicator(
        f,
        lambda x: ta_lib.volatility.ulcer_index(close=x["close"], window=14).rename("ulcer_index_14"),
    )

    return f


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Feature factory for technical indicators and engineered price action features."""
    if frame.empty:
        return frame

    work = frame.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
    enriched: list[pd.DataFrame] = []
    for _, group in work.groupby("symbol", sort=False):
        g = _add_base_features(group)
        g = _apply_pandas_ta(g)
        g = _apply_ta_fallback(g)

        # Relative-value features frequently useful for tabular models.
        derived: dict[str, pd.Series] = {}
        for col in [c for c in g.columns if c.startswith(("sma_", "ema_", "wma_", "hma_"))]:
            derived[f"dist_{col}"] = (g["close"] - g[col]) / g[col].replace(0, np.nan)

        for col in [c for c in g.columns if c.startswith("volume") or c.startswith("obv")]:
            if col in g.columns:
                derived[f"delta_{col}"] = g[col].pct_change(1)

        if derived:
            g = pd.concat([g, pd.DataFrame(derived, index=g.index)], axis=1)

        enriched.append(g)

    out = pd.concat(enriched, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def infer_feature_columns(frame: pd.DataFrame, exclude: Iterable[str] | None = None) -> list[str]:
    reserved = {
        "symbol",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "future_return",
        "future_direction",
        "pattern",
        "split",
    }
    if exclude:
        reserved.update(exclude)

    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    cols = [c for c in numeric if c not in reserved and not c.startswith("pattern_")]
    return sorted(cols)

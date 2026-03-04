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
    g["dollar_volume"] = close * volume
    g["log_dollar_volume"] = np.log1p(g["dollar_volume"].clip(lower=0))

    for window in (5, 10, 20, 50, 100, 200):
        g[f"volatility_{window}"] = g["ret_1"].rolling(window=window, min_periods=max(5, window // 2)).std()
        g[f"z_close_{window}"] = _rolling_zscore(close, window)
        g[f"z_volume_{window}"] = _rolling_zscore(volume, window)
        g[f"dollar_volume_mean_{window}"] = g["dollar_volume"].rolling(window=window, min_periods=max(5, window // 3)).mean()

    g["range_to_atr20_proxy"] = (high - low) / (
        (high - low).rolling(20, min_periods=5).mean().replace(0, np.nan)
    )
    g["amihud_illiquidity_20"] = (
        g["ret_1"].abs() / g["dollar_volume"].replace(0, np.nan)
    ).rolling(20, min_periods=5).mean()
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
    out = _add_market_context_features(out)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _add_market_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "datetime" not in frame.columns:
        return frame

    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime"]).copy()
    if out.empty:
        return frame

    market = (
        out.groupby("datetime", as_index=True)
        .agg(
            market_close=("close", "mean"),
            breadth_up_share_1=("ret_1", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            breadth_down_share_1=("ret_1", lambda s: float((pd.to_numeric(s, errors="coerce") < 0).mean())),
            cross_section_dispersion_1=("ret_1", lambda s: float(pd.to_numeric(s, errors="coerce").std())),
        )
        .sort_index()
    )

    market["market_return_1"] = market["market_close"].pct_change(1)
    for window in (5, 10, 20, 63):
        market[f"market_return_{window}"] = market["market_close"].pct_change(window)

    market["market_volatility_20"] = market["market_return_1"].rolling(20, min_periods=5).std()
    market["market_volatility_63"] = market["market_return_1"].rolling(63, min_periods=15).std()
    market["market_trend_20"] = (
        market["market_close"] / market["market_close"].rolling(20, min_periods=5).mean().replace(0, np.nan)
    ) - 1.0
    market["market_drawdown_63"] = (
        market["market_close"] / market["market_close"].rolling(63, min_periods=10).max().replace(0, np.nan)
    ) - 1.0
    market["breadth_momentum_20"] = market["breadth_up_share_1"].rolling(20, min_periods=5).mean()

    rolling_vol_median = market["market_volatility_20"].rolling(252, min_periods=30).median()
    market["regime_low_vol_uptrend"] = (
        (market["market_trend_20"] > 0) & (market["market_volatility_20"] <= rolling_vol_median)
    ).astype(float)
    market["regime_high_vol_downtrend"] = (
        (market["market_trend_20"] < 0) & (market["market_volatility_20"] > rolling_vol_median)
    ).astype(float)

    out = out.merge(market.reset_index(), on="datetime", how="left")

    for window in (5, 10, 20):
        ret_col = f"ret_{window}"
        market_col = f"market_return_{window}"
        if ret_col in out.columns and market_col in out.columns:
            out[f"alpha_ret_{window}"] = out[ret_col] - out[market_col]

    if "ret_1" in out.columns and "breadth_up_share_1" in out.columns:
        out["breadth_edge_1"] = (pd.to_numeric(out["ret_1"], errors="coerce") > 0).astype(float) - out["breadth_up_share_1"]

    if {"symbol", "ret_1", "market_return_1"}.issubset(out.columns):
        per_symbol: list[pd.DataFrame] = []
        for _, g in out.groupby("symbol", sort=False):
            x = g.sort_values("datetime").copy()
            stock_ret = pd.to_numeric(x["ret_1"], errors="coerce")
            market_ret = pd.to_numeric(x["market_return_1"], errors="coerce")
            market_var_63 = market_ret.rolling(63, min_periods=20).var()
            market_var_20 = market_ret.rolling(20, min_periods=10).var()
            x["beta_63"] = stock_ret.rolling(63, min_periods=20).cov(market_ret) / market_var_63.replace(0, np.nan)
            x["beta_20"] = stock_ret.rolling(20, min_periods=10).cov(market_ret) / market_var_20.replace(0, np.nan)
            x["idiosyncratic_return_1"] = stock_ret - (x["beta_63"] * market_ret)
            x["idiosyncratic_vol_20"] = x["idiosyncratic_return_1"].rolling(20, min_periods=5).std()
            x["idiosyncratic_momentum_20"] = x["idiosyncratic_return_1"].rolling(20, min_periods=5).sum()
            if "dollar_volume" in x.columns:
                x["flow_shock_20"] = (
                    (x["dollar_volume"] - x["dollar_volume"].rolling(20, min_periods=5).mean())
                    / x["dollar_volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
                )
            per_symbol.append(x)
        out = pd.concat(per_symbol, ignore_index=True)

    if "datetime" in out.columns:
        for col in ("ret_5", "ret_20", "volatility_20"):
            if col in out.columns:
                out[f"cs_rank_{col}"] = out.groupby("datetime")[col].rank(pct=True)

    # Calendar/seasonality covariates can capture earnings cadence,
    # month-end flows, and options-expiration effects.
    dt = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    weekday = dt.dt.weekday
    month = dt.dt.month
    day_of_year = dt.dt.dayofyear
    out["cal_weekday"] = weekday.astype(float)
    out["cal_month"] = month.astype(float)
    out["cal_weekday_sin"] = np.sin((2.0 * np.pi * weekday) / 7.0)
    out["cal_weekday_cos"] = np.cos((2.0 * np.pi * weekday) / 7.0)
    out["cal_month_sin"] = np.sin((2.0 * np.pi * (month - 1.0)) / 12.0)
    out["cal_month_cos"] = np.cos((2.0 * np.pi * (month - 1.0)) / 12.0)
    out["cal_doy_sin"] = np.sin((2.0 * np.pi * day_of_year) / 365.25)
    out["cal_doy_cos"] = np.cos((2.0 * np.pi * day_of_year) / 365.25)
    out["is_month_start"] = dt.dt.is_month_start.astype(float)
    out["is_month_end"] = dt.dt.is_month_end.astype(float)
    out["is_quarter_end"] = dt.dt.is_quarter_end.astype(float)
    out["is_opex_friday"] = ((dt.dt.weekday == 4) & dt.dt.day.between(15, 21)).astype(float)

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
        "future_market_return",
        "future_excess_return",
        "future_rank_pct",
        "future_direction",
        "pattern",
        "split",
    }
    if exclude:
        reserved.update(exclude)

    numeric = frame.select_dtypes(include=["number", "bool"]).columns
    cols = [c for c in numeric if c not in reserved and not c.startswith("pattern_")]
    return sorted(cols)

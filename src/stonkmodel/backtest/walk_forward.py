from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stonkmodel.features.dataset import split_xy
from stonkmodel.features.patterns import PATTERN_COLUMNS
from stonkmodel.models.calibration import (
    apply_probability_calibration,
    fit_probability_calibration,
    optimize_thresholds_from_validation,
    resolve_thresholds,
)
from stonkmodel.models.feature_selection import FeatureSelectionConfig, select_features
from stonkmodel.models.stacking import PatternModelIO, build_stacking_classifier


@dataclass
class BacktestResult:
    pattern: str
    model_file: str
    trades: int
    win_rate: float
    avg_trade_return: float
    cumulative_return: float
    sharpe: float
    sortino: float | None = None
    max_drawdown: float | None = None
    profit_factor: float | None = None
    annualized_return: float | None = None
    annualized_volatility: float | None = None
    win_rate_trade: float | None = None
    win_rate_period: float | None = None
    backtest_start_datetime: str | None = None
    backtest_end_datetime: str | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "pattern": self.pattern,
            "model_file": self.model_file,
            "backtest_start_datetime": self.backtest_start_datetime,
            "backtest_end_datetime": self.backtest_end_datetime,
            "trades": self.trades,
            "win_rate": self.win_rate,
            "win_rate_trade": self.win_rate_trade if self.win_rate_trade is not None else self.win_rate,
            "win_rate_period": self.win_rate_period if self.win_rate_period is not None else self.win_rate,
            "avg_trade_return": self.avg_trade_return,
            "cumulative_return": self.cumulative_return,
            "sharpe": self.sharpe,
            "sortino": float(self.sortino) if self.sortino is not None else np.nan,
            "max_drawdown": float(self.max_drawdown) if self.max_drawdown is not None else np.nan,
            "profit_factor": float(self.profit_factor) if self.profit_factor is not None else np.nan,
            "annualized_return": float(self.annualized_return) if self.annualized_return is not None else np.nan,
            "annualized_volatility": float(self.annualized_volatility) if self.annualized_volatility is not None else np.nan,
        }


def _bars_per_day(interval: str) -> float:
    mapping = {
        "1d": 1.0,
        "1h": 6.5,
        "60m": 6.5,
        "30m": 13.0,
        "15m": 26.0,
        "5m": 78.0,
        "1m": 390.0,
    }
    return float(mapping.get(interval, 1.0))


def _holding_days(interval: str, horizon_bars: int, latency_bars: int) -> float:
    return float(max(1, horizon_bars + max(0, latency_bars)) / _bars_per_day(interval))


def _periods_per_year(interval: str) -> float:
    return float(252.0 * _bars_per_day(interval))


def _bar_timedelta(interval: str, bars: int) -> pd.Timedelta:
    step = max(0, int(bars))
    if step <= 0:
        return pd.Timedelta(0)
    token = str(interval).strip().lower()
    if token in {"1d", "d", "1day"}:
        return pd.Timedelta(days=step)
    if token in {"1h", "60m"}:
        return pd.Timedelta(hours=step)
    if token == "30m":
        return pd.Timedelta(minutes=30 * step)
    if token == "15m":
        return pd.Timedelta(minutes=15 * step)
    if token == "5m":
        return pd.Timedelta(minutes=5 * step)
    if token == "1m":
        return pd.Timedelta(minutes=step)
    return pd.Timedelta(days=step)


def _build_meta_filter_features(
    frame: pd.DataFrame,
    base_feature_cols: list[str],
    prob_up: np.ndarray,
    positions: np.ndarray,
) -> pd.DataFrame:
    out = frame.reindex(columns=base_feature_cols).copy()
    p = pd.Series(prob_up, index=frame.index, dtype=float)
    side = pd.Series(np.sign(positions), index=frame.index, dtype=float)
    out["meta_prob_up"] = p
    out["meta_abs_edge"] = (p - 0.5).abs()
    out["meta_side"] = side
    out["meta_signed_edge"] = np.where(side >= 0, p - 0.5, 0.5 - p)
    return out


def _apply_payload_meta_filter(
    payload: dict[str, object],
    subset: pd.DataFrame,
    feature_cols: list[str],
    prob_up: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    meta = payload.get("meta_filter", {}) if isinstance(payload, dict) else {}
    if not isinstance(meta, dict) or not bool(meta.get("enabled", False)):
        return positions, np.full(len(positions), np.nan, dtype=float)

    model = meta.get("model")
    meta_cols = [str(c) for c in list(meta.get("feature_columns", []))]
    if model is None or not meta_cols:
        return positions, np.full(len(positions), np.nan, dtype=float)

    active = positions != 0
    if not bool(active.any()):
        return positions, np.full(len(positions), np.nan, dtype=float)

    meta_frame = _build_meta_filter_features(subset, base_feature_cols=feature_cols, prob_up=prob_up, positions=positions)
    x_meta = meta_frame.loc[active].reindex(columns=meta_cols).copy()
    x_meta = x_meta.fillna(x_meta.median(numeric_only=True)).fillna(0.0)
    try:
        meta_prob_active = model.predict_proba(x_meta)[:, 1]
    except Exception:
        return positions, np.full(len(positions), np.nan, dtype=float)

    threshold = float(np.clip(float(meta.get("threshold", 0.55)), 0.5, 0.95))
    keep = meta_prob_active >= threshold
    out_pos = positions.copy()
    active_idx = np.where(active)[0]
    out_pos[active_idx[~keep]] = 0

    meta_prob_all = np.full(len(positions), np.nan, dtype=float)
    meta_prob_all[active_idx] = meta_prob_active
    return out_pos, meta_prob_all


def _aggregate_returns_by_datetime(subset: pd.DataFrame, column: str = "strategy_return") -> pd.Series:
    if subset.empty or column not in subset.columns:
        return pd.Series(dtype=float)

    returns = pd.to_numeric(subset[column], errors="coerce")
    if "datetime" not in subset.columns:
        return returns.dropna().astype(float).reset_index(drop=True)

    dt = pd.to_datetime(subset["datetime"], utc=True, errors="coerce")
    valid = dt.notna() & returns.notna()
    if not bool(valid.any()):
        return pd.Series(dtype=float)

    frame = pd.DataFrame({"datetime": dt.loc[valid], "strategy_return": returns.loc[valid].astype(float)})
    # Aggregate concurrent symbols/signals into one portfolio return per timestamp.
    return frame.groupby("datetime", sort=True)["strategy_return"].mean()


def _safe_cumulative_return(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    # Guard log1p domain while preserving normal return magnitudes.
    clean = clean.clip(lower=-0.999999999)
    return float(np.expm1(np.log1p(clean).sum()))


def _safe_sharpe_for_interval(series: pd.Series, interval: str) -> float:
    std = series.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((series.mean() / std) * np.sqrt(_periods_per_year(interval)))


def _safe_sortino_for_interval(series: pd.Series, interval: str) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    downside = clean.loc[clean < 0]
    downside_std = float(downside.std(ddof=1)) if not downside.empty else 0.0
    if downside_std <= 0 or np.isnan(downside_std):
        return 0.0
    return float((clean.mean() / downside_std) * np.sqrt(_periods_per_year(interval)))


def _max_drawdown_from_period_returns(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    growth = np.exp(np.log1p(clean.clip(lower=-0.999999999)).cumsum())
    drawdown = growth / np.maximum.accumulate(growth) - 1.0
    return float(np.nanmin(drawdown)) if len(drawdown) else 0.0


def _profit_factor(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    gross_profit = float(clean.loc[clean > 0].sum())
    gross_loss = float(-clean.loc[clean < 0].sum())
    if gross_loss <= 1e-12:
        return float("nan")
    return float(gross_profit / gross_loss)


def _annualized_return_from_period_returns(series: pd.Series, interval: str) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    growth = float(np.exp(np.log1p(clean.clip(lower=-0.999999999)).sum()))
    years = _annualization_years_from_period_returns(clean, interval=interval)
    if years <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def _annualized_volatility(series: pd.Series, interval: str) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or len(clean) < 2:
        return 0.0
    years = _annualization_years_from_period_returns(clean, interval=interval)
    if years <= 0:
        return 0.0
    realized_periods_per_year = max(1e-9, float(len(clean)) / years)
    std = float(clean.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return float(std * np.sqrt(realized_periods_per_year))


def _annualization_years_from_period_returns(series: pd.Series, interval: str) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0

    min_years = 1.0 / max(1e-9, _periods_per_year(interval))
    idx = clean.index
    if isinstance(idx, pd.DatetimeIndex):
        dt = pd.to_datetime(idx, utc=True, errors="coerce")
        dt = dt[~pd.isna(dt)]
        if len(dt) >= 2:
            elapsed_years = float((dt.max() - dt.min()).total_seconds()) / (365.25 * 24.0 * 3600.0)
            # Include roughly one bar to avoid zero-year annualization on narrow spans.
            return max(float(elapsed_years + min_years), min_years)

    # Fallback for non-datetime indexes.
    return max(float(len(clean)) / max(1e-9, _periods_per_year(interval)), min_years)


def _win_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float((clean > 0).mean())


def _to_iso_utc(value: object) -> str | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _dedupe_by_symbol_datetime(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "datetime" not in frame.columns:
        return frame

    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    out = out.dropna(subset=["datetime"]).copy()
    if out.empty:
        return out

    sort_cols: list[str] = []
    if "window_start" in out.columns:
        out["window_start"] = pd.to_datetime(out["window_start"], utc=True, errors="coerce")
        sort_cols.append("window_start")
    sort_cols.append("datetime")
    has_confidence = False
    if "pred_prob_up" in out.columns:
        out["_confidence"] = (pd.to_numeric(out["pred_prob_up"], errors="coerce") - 0.5).abs()
        sort_cols.append("_confidence")
        has_confidence = True

    ascending = [True] * len(sort_cols)
    if has_confidence:
        ascending[-1] = False
    out = out.sort_values(sort_cols, ascending=ascending)
    dedupe_keys = ["datetime"]
    if "symbol" in out.columns:
        dedupe_keys = ["symbol", "datetime"]
    out = out.drop_duplicates(subset=dedupe_keys, keep="first").copy()
    if "_confidence" in out.columns:
        out = out.drop(columns=["_confidence"])
    return out


def _compute_label_end_datetime(frame: pd.DataFrame, bars_ahead: int) -> pd.Series:
    if frame.empty or "datetime" not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")

    steps = max(1, int(bars_ahead))
    work = frame.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    out = pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")

    if "symbol" not in work.columns:
        return work["datetime"].shift(-steps)

    for _, g in work.groupby("symbol", sort=False):
        g_sorted = g.sort_values("datetime")
        out.loc[g_sorted.index] = g_sorted["datetime"].shift(-steps)
    return out


def _build_equity_curve_frame(
    period_returns: pd.Series,
    pattern: str,
    model_file: str,
    initial_investment: float,
    curve_variant: str = "ml_model",
) -> pd.DataFrame:
    if period_returns.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "pattern",
                "model_file",
                "period_return",
                "cumulative_return",
                "equity_value",
                "initial_investment",
                "curve_variant",
            ]
        )

    clean = pd.to_numeric(period_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "pattern",
                "model_file",
                "period_return",
                "cumulative_return",
                "equity_value",
                "initial_investment",
                "curve_variant",
            ]
        )

    if isinstance(clean.index, pd.DatetimeIndex):
        dt_index = pd.to_datetime(clean.index, utc=True, errors="coerce")
    else:
        dt_index = pd.to_datetime(clean.index, utc=True, errors="coerce")
    valid = ~pd.isna(dt_index)
    if not bool(valid.any()):
        return pd.DataFrame(
            columns=[
                "datetime",
                "pattern",
                "model_file",
                "period_return",
                "cumulative_return",
                "equity_value",
                "initial_investment",
                "curve_variant",
            ]
        )

    values = clean.loc[valid].astype(float).clip(lower=-0.999999999)
    dt = pd.to_datetime(dt_index[valid], utc=True, errors="coerce")
    growth = np.exp(np.log1p(values).cumsum())
    equity = float(initial_investment) * growth
    cumulative = growth - 1.0

    return pd.DataFrame(
        {
            "datetime": dt,
            "pattern": pattern,
            "model_file": model_file,
            "period_return": values.to_numpy(dtype=float),
            "cumulative_return": cumulative.to_numpy(dtype=float),
            "equity_value": equity.to_numpy(dtype=float),
            "initial_investment": float(initial_investment),
            "curve_variant": str(curve_variant),
        }
    )


def _build_portfolio_summary_from_signals(
    signals: pd.DataFrame,
    interval: str,
    fee_bps: float,
    spread_bps: float,
    slippage_bps: float,
    short_borrow_bps_per_day: float,
    horizon_bars: int,
    latency_bars: int,
    top_k_per_side: int,
    max_gross_exposure: float,
    model_file: str,
    return_period_series: bool = False,
    min_abs_score: float = 0.0,
    rebalance_every_n_bars: int = 1,
    symbol_cooldown_bars: int = 0,
    volatility_scaling: bool = True,
    max_symbol_weight: float = 0.35,
) -> dict[str, float | int | str | None] | tuple[dict[str, float | int | str | None], pd.Series] | None:
    if signals.empty:
        return None

    work = signals.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    work["score"] = pd.to_numeric(work.get("score"), errors="coerce")
    work["realized_return"] = pd.to_numeric(work.get("realized_return"), errors="coerce")
    if "volatility_20" in work.columns:
        work["volatility_20"] = pd.to_numeric(work.get("volatility_20"), errors="coerce")
    else:
        work["volatility_20"] = np.nan
    work = work.dropna(subset=["datetime", "symbol", "score", "realized_return"]).copy()
    min_edge = max(0.0, float(min_abs_score))
    if min_edge > 0:
        work = work.loc[work["score"].abs() >= min_edge].copy()
    if work.empty:
        return None

    fee = float(fee_bps) / 10000.0
    spread = float(spread_bps) / 10000.0
    slippage = float(slippage_bps) / 10000.0
    trade_cost = fee + spread + (2.0 * slippage)
    borrow = (float(short_borrow_bps_per_day) / 10000.0) * _holding_days(
        interval=interval,
        horizon_bars=horizon_bars,
        latency_bars=latency_bars,
    )
    gross_cap = max(0.0, float(max_gross_exposure))
    per_symbol_cap = min(max(0.01, float(max_symbol_weight)), max(0.01, gross_cap))
    k = max(1, int(top_k_per_side))
    rebalance_n = max(1, int(rebalance_every_n_bars))
    cooldown_n = max(0, int(symbol_cooldown_bars))

    period_rows: list[tuple[pd.Timestamp, float]] = []
    trade_returns: list[float] = []
    trades = 0
    last_trade_bar_idx_by_symbol: dict[str, int] = {}

    grouped = list(work.groupby("datetime", sort=True))
    for bar_idx, (dt, g) in enumerate(grouped):
        if rebalance_n > 1 and (bar_idx % rebalance_n) != 0:
            continue

        by_symbol = (
            g.groupby("symbol", as_index=False)
            .agg(
                score=("score", "mean"),
                realized_return=("realized_return", "mean"),
                volatility_20=("volatility_20", "mean"),
            )
            .sort_values("score", ascending=False)
        )
        if cooldown_n > 0 and not by_symbol.empty:
            eligible = by_symbol["symbol"].astype(str).apply(
                lambda sym: (bar_idx - last_trade_bar_idx_by_symbol.get(sym, -10_000_000)) > cooldown_n
            )
            by_symbol = by_symbol.loc[eligible].copy()
        if by_symbol.empty:
            continue

        longs = by_symbol.loc[by_symbol["score"] > 0].head(k).copy()
        shorts = by_symbol.loc[by_symbol["score"] < 0].sort_values("score", ascending=True).head(k).copy()

        selected_parts: list[pd.DataFrame] = []
        def _assign_side_weights(side_frame: pd.DataFrame, side_budget: float, sign: float) -> pd.DataFrame:
            if side_frame.empty or side_budget <= 0:
                return side_frame.iloc[0:0].copy()
            x = side_frame.copy()
            if volatility_scaling:
                vol = pd.to_numeric(x.get("volatility_20"), errors="coerce").replace(0, np.nan).abs()
                inv_vol = 1.0 / vol.clip(lower=0.005, upper=2.0)
                quality = pd.to_numeric(x["score"], errors="coerce").abs().clip(lower=1e-6)
                raw = (quality * inv_vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                raw = pd.Series(1.0, index=x.index, dtype=float)
            if float(raw.sum()) <= 0:
                raw = pd.Series(1.0, index=x.index, dtype=float)
            w = side_budget * (raw / raw.sum())
            w = w.clip(upper=per_symbol_cap)
            if float(w.sum()) > side_budget and side_budget > 0:
                w = side_budget * (w / w.sum())
            x["weight"] = float(sign) * w
            return x

        if not longs.empty and not shorts.empty:
            selected_parts.append(_assign_side_weights(longs, side_budget=(gross_cap * 0.5), sign=1.0))
            selected_parts.append(_assign_side_weights(shorts, side_budget=(gross_cap * 0.5), sign=-1.0))
        elif not longs.empty:
            selected_parts.append(_assign_side_weights(longs, side_budget=gross_cap, sign=1.0))
        elif not shorts.empty:
            selected_parts.append(_assign_side_weights(shorts, side_budget=gross_cap, sign=-1.0))

        if not selected_parts:
            continue

        selected = pd.concat(selected_parts, ignore_index=True)
        if selected.empty or "weight" not in selected.columns:
            continue
        if selected["weight"].abs().sum() > gross_cap and gross_cap > 0:
            selected["weight"] = selected["weight"] * (gross_cap / selected["weight"].abs().sum())
        selected["trade_return"] = (
            selected["weight"] * selected["realized_return"]
            - selected["weight"].abs() * trade_cost
            - np.where(selected["weight"] < 0, selected["weight"].abs() * borrow, 0.0)
        )

        period_rows.append((dt, float(selected["trade_return"].sum())))
        trade_returns.extend(selected["trade_return"].astype(float).tolist())
        trades += int(len(selected))
        for sym in selected["symbol"].astype(str).tolist():
            last_trade_bar_idx_by_symbol[sym] = int(bar_idx)

    if trades == 0 or not period_rows:
        return None

    period = pd.Series(
        [ret for _, ret in period_rows],
        index=pd.to_datetime([ts for ts, _ in period_rows], utc=True, errors="coerce"),
        dtype=float,
    ).sort_index()
    trade = pd.Series(trade_returns, dtype=float)

    summary = {
        "pattern": "portfolio_combined",
        "model_file": model_file,
        "backtest_start_datetime": _to_iso_utc(period.index.min()),
        "backtest_end_datetime": _to_iso_utc(period.index.max()),
        "trades": int(trades),
        "win_rate": _win_rate(period),
        "win_rate_trade": _win_rate(trade),
        "win_rate_period": _win_rate(period),
        "avg_trade_return": float(trade.mean()) if not trade.empty else 0.0,
        "cumulative_return": _safe_cumulative_return(period),
        "sharpe": _safe_sharpe_for_interval(period, interval=interval),
        "sortino": _safe_sortino_for_interval(period, interval=interval),
        "max_drawdown": _max_drawdown_from_period_returns(period),
        "profit_factor": _profit_factor(trade),
        "annualized_return": _annualized_return_from_period_returns(period, interval=interval),
        "annualized_volatility": _annualized_volatility(period, interval=interval),
    }
    if return_period_series:
        return summary, period
    return summary


def _extract_pattern_subset(frame: pd.DataFrame, pattern: str) -> pd.DataFrame:
    if pattern == "none":
        return frame.loc[frame["pattern"].isna()].copy()
    pattern_col = f"pattern_{pattern}"
    if pattern_col not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame[pattern_col] == 1].copy()


def _cap_rows_for_speed(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None:
        return frame
    cap = int(max_rows)
    if cap <= 0 or len(frame) <= cap:
        return frame
    if "datetime" not in frame.columns:
        return frame.tail(cap).copy()
    return frame.sort_values("datetime").tail(cap).copy()


def _compute_realized_return(
    frame: pd.DataFrame,
    horizon_bars: int,
    latency_bars: int,
) -> pd.Series:
    price_col = "adj_close" if "adj_close" in frame.columns else "close"
    if price_col not in frame.columns:
        if "future_return" in frame.columns and int(latency_bars) == 0:
            return frame["future_return"].astype(float)
        return pd.Series(np.nan, index=frame.index, dtype=float)

    out = pd.Series(index=frame.index, dtype=float)
    lag = max(0, int(latency_bars))
    horizon = max(1, int(horizon_bars))
    for _, g in frame.groupby("symbol", sort=False):
        if "datetime" in g.columns:
            g = g.sort_values("datetime")
        entry = g[price_col].shift(-lag)
        exit_ = g[price_col].shift(-(lag + horizon))
        out.loc[g.index] = (exit_ / entry) - 1.0
    return out


def _filter_extreme_realized_returns(
    subset: pd.DataFrame,
    max_abs_return: float = 3.0,
) -> pd.DataFrame:
    if subset.empty or "realized_return" not in subset.columns:
        return subset
    bound = float(max_abs_return)
    if bound <= 0:
        return subset
    rr = pd.to_numeric(subset["realized_return"], errors="coerce")
    # Drops extreme single-trade returns that are typically split/corporate-action artifacts.
    mask = rr.abs() <= bound
    return subset.loc[mask].copy()


def _select_best_patterns_for_portfolio(
    rows: list[dict[str, float | int | str | None]],
    top_n: int = 6,
    min_trades: int = 40,
    min_win_rate_trade: float = 0.5,
) -> list[str]:
    if not rows:
        return []

    df = pd.DataFrame(rows)
    if df.empty or "pattern" not in df.columns:
        return []

    work = df.copy()
    work["pattern"] = work["pattern"].astype(str)
    work = work.loc[~work["pattern"].str.startswith("portfolio_")].copy()
    if work.empty:
        return []

    work["trades"] = pd.to_numeric(work.get("trades"), errors="coerce").fillna(0.0)
    work["cumulative_return"] = pd.to_numeric(work.get("cumulative_return"), errors="coerce")
    work["sharpe"] = pd.to_numeric(work.get("sharpe"), errors="coerce")
    work["max_drawdown"] = pd.to_numeric(work.get("max_drawdown"), errors="coerce")
    work["win_rate_trade"] = pd.to_numeric(work.get("win_rate_trade"), errors="coerce")

    if int(min_trades) > 0:
        work = work.loc[work["trades"] >= float(min_trades)].copy()
    if float(min_win_rate_trade) > 0:
        work = work.loc[work["win_rate_trade"] >= float(min_win_rate_trade)].copy()

    if work.empty:
        return []

    work["objective_score"] = (
        work["cumulative_return"].replace([np.inf, -np.inf], np.nan).fillna(-1e9)
        + (0.20 * work["sharpe"].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        - (0.15 * work["max_drawdown"].abs().replace([np.inf, -np.inf], np.nan).fillna(0.0))
        + (0.20 * work["win_rate_trade"].replace([np.inf, -np.inf], np.nan).fillna(0.0))
        - (0.01 * np.log1p(work["trades"].clip(lower=0.0)))
    )
    # Keep the best model variant per pattern first, then rank patterns.
    best_per_pattern = work.sort_values(["objective_score", "cumulative_return"], ascending=[False, False]).drop_duplicates(
        subset=["pattern"],
        keep="first",
    )
    keep_n = max(1, int(top_n))
    return best_per_pattern.head(keep_n)["pattern"].dropna().astype(str).tolist()


def _apply_execution_realism(
    subset: pd.DataFrame,
    interval: str,
    horizon_bars: int,
    fee_bps: float,
    spread_bps: float,
    slippage_bps: float,
    short_borrow_bps_per_day: float,
    latency_bars: int,
) -> pd.Series:
    holding_days = _holding_days(interval, horizon_bars=horizon_bars, latency_bars=latency_bars)
    fee = float(fee_bps) / 10000.0
    spread = float(spread_bps) / 10000.0
    slippage = float(slippage_bps) / 10000.0
    borrow = (float(short_borrow_bps_per_day) / 10000.0) * holding_days

    trade_cost = fee + spread + (2.0 * slippage)
    short_cost = np.where(subset["position"] < 0, borrow, 0.0)
    return subset["position"] * subset["realized_return"] - trade_cost - short_cost


def _infer_blind_pattern_side(pattern: str) -> int:
    token = str(pattern or "").strip().lower()
    bearish_tokens = ("bearish", "shooting_star", "evening_star", "dark_cloud_cover")
    bullish_tokens = ("bullish", "hammer", "morning_star", "piercing_line")
    if any(t in token for t in bearish_tokens):
        return -1
    if any(t in token for t in bullish_tokens):
        return 1
    return 1


def _build_blind_pattern_period_returns(
    pattern_subset: pd.DataFrame,
    pattern: str,
    interval: str,
    horizon_bars: int,
    fee_bps: float,
    spread_bps: float,
    slippage_bps: float,
    short_borrow_bps_per_day: float,
    latency_bars: int,
) -> pd.Series:
    if pattern_subset.empty:
        return pd.Series(dtype=float)

    blind = pattern_subset.copy()
    if "realized_return" not in blind.columns:
        return pd.Series(dtype=float)
    blind = blind.dropna(subset=["realized_return"]).copy()
    blind = _filter_extreme_realized_returns(blind, max_abs_return=3.0)
    if blind.empty:
        return pd.Series(dtype=float)

    blind["position"] = int(_infer_blind_pattern_side(pattern))
    blind["strategy_return"] = _apply_execution_realism(
        subset=blind,
        interval=interval,
        horizon_bars=horizon_bars,
        fee_bps=fee_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        short_borrow_bps_per_day=short_borrow_bps_per_day,
        latency_bars=latency_bars,
    )
    return _aggregate_returns_by_datetime(blind, column="strategy_return")


def _build_universe_period_returns(universe_frame: pd.DataFrame) -> pd.Series:
    if universe_frame.empty:
        return pd.Series(dtype=float)

    frame = universe_frame.copy()
    rr_col = None
    if "realized_return" in frame.columns:
        rr_col = "realized_return"
    elif "_realized_return_full" in frame.columns:
        rr_col = "_realized_return_full"
    if rr_col is None or "datetime" not in frame.columns:
        return pd.Series(dtype=float)

    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
    frame["realized_return"] = pd.to_numeric(frame[rr_col], errors="coerce")
    frame = frame.dropna(subset=["datetime", "realized_return"]).copy()
    if frame.empty:
        return pd.Series(dtype=float)

    frame = frame.loc[frame["realized_return"].abs() <= 3.0].copy()
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.groupby("datetime", sort=True)["realized_return"].mean()


def _build_component_period_table_from_signals(
    signals: pd.DataFrame,
    component_col: str,
    interval: str,
    horizon_bars: int,
    fee_bps: float,
    spread_bps: float,
    slippage_bps: float,
    short_borrow_bps_per_day: float,
    latency_bars: int,
) -> pd.DataFrame:
    required = {"datetime", "realized_return", "score"}
    if signals.empty or component_col not in signals.columns or not required.issubset(signals.columns):
        return pd.DataFrame(
            columns=[
                "datetime",
                "component",
                "period_return",
                "component_beta_63",
                "component_size_z",
                "component_signal_count",
            ]
        )

    work = signals.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    work["component"] = work[component_col].astype(str).str.strip()
    work["realized_return"] = pd.to_numeric(work["realized_return"], errors="coerce")
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work["beta_63"] = pd.to_numeric(work.get("beta_63"), errors="coerce")
    work["market_cap"] = pd.to_numeric(work.get("market_cap"), errors="coerce")
    work = work.dropna(subset=["datetime", "component", "realized_return", "score"]).copy()
    work = work.loc[work["component"] != ""].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "component",
                "period_return",
                "component_beta_63",
                "component_size_z",
                "component_signal_count",
            ]
        )

    work = _filter_extreme_realized_returns(work, max_abs_return=3.0)
    if work.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "component",
                "period_return",
                "component_beta_63",
                "component_size_z",
                "component_signal_count",
            ]
        )

    work["position"] = np.where(work["score"] >= 0, 1, -1).astype(int)
    work["strategy_return"] = _apply_execution_realism(
        subset=work,
        interval=interval,
        horizon_bars=horizon_bars,
        fee_bps=fee_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        short_borrow_bps_per_day=short_borrow_bps_per_day,
        latency_bars=latency_bars,
    )
    work["log_market_cap"] = np.log1p(work["market_cap"].clip(lower=0.0))

    period = (
        work.groupby(["datetime", "component"], as_index=False)
        .agg(
            period_return=("strategy_return", "mean"),
            component_beta_63=("beta_63", "mean"),
            component_log_mcap=("log_market_cap", "mean"),
            component_signal_count=("symbol", "nunique"),
        )
        .sort_values(["datetime", "component"])
        .reset_index(drop=True)
    )

    if period.empty:
        return pd.DataFrame(
            columns=[
                "datetime",
                "component",
                "period_return",
                "component_beta_63",
                "component_size_z",
                "component_signal_count",
            ]
        )

    period["component_signal_count"] = pd.to_numeric(period["component_signal_count"], errors="coerce").fillna(0).astype(int)
    period["component_size_z"] = 0.0
    for dt, idx in period.groupby("datetime").groups.items():
        vals = pd.to_numeric(period.loc[idx, "component_log_mcap"], errors="coerce")
        std = float(vals.std(ddof=0)) if vals.notna().sum() >= 2 else 0.0
        if std > 0 and np.isfinite(std):
            period.loc[idx, "component_size_z"] = (vals - float(vals.mean())) / std
        else:
            period.loc[idx, "component_size_z"] = 0.0

    return period.drop(columns=["component_log_mcap"], errors="ignore")


def _rolling_component_beta_to_market(
    returns_wide: pd.DataFrame,
    market_period_returns: pd.Series,
    lookback_bars: int,
) -> pd.DataFrame:
    if returns_wide.empty:
        return pd.DataFrame(index=returns_wide.index, columns=returns_wide.columns)
    market = pd.to_numeric(market_period_returns, errors="coerce").reindex(returns_wide.index)
    min_periods = max(5, int(lookback_bars) // 3)
    var = market.rolling(int(lookback_bars), min_periods=min_periods).var()
    out = pd.DataFrame(index=returns_wide.index, columns=returns_wide.columns, dtype=float)
    if not bool(var.notna().any()):
        return out
    for col in returns_wide.columns:
        cov = pd.to_numeric(returns_wide[col], errors="coerce").rolling(int(lookback_bars), min_periods=min_periods).cov(market)
        out[col] = cov / var.replace(0, np.nan)
    return out


def _blend_neutral_weights(
    beta_long: float | None,
    beta_short: float | None,
    size_long: float | None,
    size_short: float | None,
    beta_neutral: bool,
    size_neutral: bool,
) -> tuple[float, float]:
    w_long = 0.5
    w_short = 0.5

    if beta_neutral:
        b_l = float(beta_long) if beta_long is not None else np.nan
        b_s = float(beta_short) if beta_short is not None else np.nan
        if np.isfinite(b_l) and np.isfinite(b_s) and (abs(b_l) + abs(b_s)) > 1e-9:
            w_long = abs(b_s) / (abs(b_l) + abs(b_s))
            w_short = abs(b_l) / (abs(b_l) + abs(b_s))

    if size_neutral:
        s_l = float(size_long) if size_long is not None else np.nan
        s_s = float(size_short) if size_short is not None else np.nan
        if np.isfinite(s_l) and np.isfinite(s_s) and (abs(s_l) + abs(s_s)) > 1e-9:
            w_long_sz = abs(s_s) / (abs(s_l) + abs(s_s))
            w_short_sz = abs(s_l) / (abs(s_l) + abs(s_s))
            w_long = 0.5 * (w_long + w_long_sz)
            w_short = 0.5 * (w_short + w_short_sz)

    total = float(w_long + w_short)
    if total <= 0:
        return 0.5, 0.5
    return float(w_long / total), float(w_short / total)


def _build_relative_strength_spread_period_returns(
    component_period_table: pd.DataFrame,
    interval: str,
    lookback_bars: int,
    top_components: int,
    min_edge: float,
    switch_cost_bps: float,
    beta_neutral: bool,
    size_neutral: bool,
    target_vol_annual: float,
    market_period_returns: pd.Series | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    # Relative-strength winner/loser spread construction:
    # aligns with cross-sectional momentum + relative-value spread literature.
    if component_period_table.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    table = component_period_table.copy()
    table["datetime"] = pd.to_datetime(table["datetime"], utc=True, errors="coerce")
    table["period_return"] = pd.to_numeric(table["period_return"], errors="coerce")
    table["component"] = table["component"].astype(str)
    table = table.dropna(subset=["datetime", "component", "period_return"]).copy()
    if table.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    returns_wide = (
        table.pivot_table(index="datetime", columns="component", values="period_return", aggfunc="mean")
        .sort_index()
        .copy()
    )
    if returns_wide.shape[1] < 2:
        return pd.Series(dtype=float), pd.DataFrame()

    size_wide = table.pivot_table(index="datetime", columns="component", values="component_size_z", aggfunc="mean").reindex(
        returns_wide.index
    )
    if market_period_returns is None or market_period_returns.empty:
        market = returns_wide.mean(axis=1, skipna=True).astype(float)
    else:
        market = pd.to_numeric(market_period_returns, errors="coerce").reindex(returns_wide.index)
    beta_wide = _rolling_component_beta_to_market(
        returns_wide=returns_wide,
        market_period_returns=market,
        lookback_bars=max(20, int(lookback_bars)),
    )

    lookback = max(10, int(lookback_bars))
    top_k = max(1, int(top_components))
    edge_floor = float(max(0.0, float(min_edge)))
    switch_cost = float(max(0.0, float(switch_cost_bps)) / 10000.0)
    history_min = max(8, lookback // 3)

    period_rows: list[tuple[pd.Timestamp, float]] = []
    decision_rows: list[dict[str, object]] = []
    prev_pair: tuple[str, str] | None = None

    dt_index = returns_wide.index
    for i in range(lookback, len(dt_index)):
        dt = dt_index[i]
        hist = returns_wide.iloc[i - lookback : i]
        current = returns_wide.iloc[i]
        hist_count = hist.notna().sum(axis=0)
        valid_components = [c for c in returns_wide.columns if pd.notna(current[c]) and int(hist_count.get(c, 0)) >= history_min]
        if len(valid_components) < 2:
            continue

        hist_valid = hist[valid_components]
        mu = hist_valid.mean(axis=0, skipna=True)
        sigma = hist_valid.std(axis=0, ddof=1, skipna=True)
        score = (mu / sigma.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        score = score.fillna(mu).replace([np.inf, -np.inf], np.nan).dropna()
        if len(score) < 2:
            continue

        ranked = score.sort_values(ascending=False)
        long_candidates = ranked.head(top_k).index.tolist()
        short_candidates = ranked.tail(top_k).index.tolist()

        best_choice: dict[str, object] | None = None
        for long_comp in long_candidates:
            for short_comp in short_candidates:
                if long_comp == short_comp:
                    continue
                edge = float(ranked.get(long_comp, np.nan) - ranked.get(short_comp, np.nan))
                if not np.isfinite(edge) or edge < edge_floor:
                    continue

                beta_long = beta_wide.at[dt, long_comp] if long_comp in beta_wide.columns else np.nan
                beta_short = beta_wide.at[dt, short_comp] if short_comp in beta_wide.columns else np.nan
                size_long = size_wide.at[dt, long_comp] if long_comp in size_wide.columns else np.nan
                size_short = size_wide.at[dt, short_comp] if short_comp in size_wide.columns else np.nan
                penalty = 0.0
                if beta_neutral and np.isfinite(beta_long) and np.isfinite(beta_short):
                    penalty += 0.25 * abs(float(beta_long) - float(beta_short))
                if size_neutral and np.isfinite(size_long) and np.isfinite(size_short):
                    penalty += 0.15 * abs(float(size_long) - float(size_short))
                objective = edge - penalty

                if best_choice is None or objective > float(best_choice["objective"]):
                    best_choice = {
                        "long_component": str(long_comp),
                        "short_component": str(short_comp),
                        "edge": float(edge),
                        "objective": float(objective),
                        "beta_long": float(beta_long) if np.isfinite(beta_long) else np.nan,
                        "beta_short": float(beta_short) if np.isfinite(beta_short) else np.nan,
                        "size_long": float(size_long) if np.isfinite(size_long) else np.nan,
                        "size_short": float(size_short) if np.isfinite(size_short) else np.nan,
                    }

        if best_choice is None:
            continue

        long_comp = str(best_choice["long_component"])
        short_comp = str(best_choice["short_component"])
        ret_long = float(current[long_comp])
        ret_short = float(current[short_comp])
        w_long, w_short = _blend_neutral_weights(
            beta_long=best_choice.get("beta_long"),
            beta_short=best_choice.get("beta_short"),
            size_long=best_choice.get("size_long"),
            size_short=best_choice.get("size_short"),
            beta_neutral=bool(beta_neutral),
            size_neutral=bool(size_neutral),
        )
        spread_ret = (w_long * ret_long) - (w_short * ret_short)
        pair_now = (long_comp, short_comp)
        pair_changed = prev_pair is not None and pair_now != prev_pair
        if pair_changed and switch_cost > 0:
            spread_ret -= switch_cost
        prev_pair = pair_now

        period_rows.append((dt, float(spread_ret)))
        decision_rows.append(
            {
                "datetime": dt,
                "long_component": long_comp,
                "short_component": short_comp,
                "edge": float(best_choice["edge"]),
                "objective": float(best_choice["objective"]),
                "long_weight": float(w_long),
                "short_weight": float(w_short),
                "beta_exposure": (float(w_long) * float(best_choice.get("beta_long", np.nan)))
                - (float(w_short) * float(best_choice.get("beta_short", np.nan))),
                "size_exposure": (float(w_long) * float(best_choice.get("size_long", np.nan)))
                - (float(w_short) * float(best_choice.get("size_short", np.nan))),
                "pair_changed": int(bool(pair_changed)),
            }
        )

    if not period_rows:
        return pd.Series(dtype=float), pd.DataFrame()

    period = pd.Series(
        [v for _, v in period_rows],
        index=pd.to_datetime([dt for dt, _ in period_rows], utc=True, errors="coerce"),
        dtype=float,
    ).sort_index()
    decision = pd.DataFrame(decision_rows).sort_values("datetime").reset_index(drop=True)

    target_vol = float(target_vol_annual)
    if target_vol > 0 and not period.empty:
        vol_window = max(10, lookback // 2)
        realized = period.rolling(vol_window, min_periods=max(5, vol_window // 3)).std().shift(1)
        target_period_vol = target_vol / np.sqrt(_periods_per_year(interval))
        scale = (target_period_vol / realized.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=3.0)
        scale = scale.fillna(1.0)
        period = period * scale
        if not decision.empty:
            decision["vol_scale"] = pd.to_numeric(scale.reindex(pd.to_datetime(decision["datetime"], utc=True, errors="coerce")), errors="coerce").fillna(1.0).to_numpy(dtype=float)

    return period, decision


def _derive_regime_state_series(frame: pd.DataFrame) -> pd.Series:
    # Simple regime proxy that maps market/macro state into risk-on/off buckets,
    # inspired by regime-switching and volatility-managed allocation research.
    if frame.empty or "datetime" not in frame.columns:
        return pd.Series(dtype="string")

    keys = [
        "macro_risk_off_score",
        "macro_regime_high_stress",
        "regime_high_vol_downtrend",
        "regime_low_vol_uptrend",
        "market_volatility_20",
        "market_return_20",
        "market_drawdown_63",
    ]
    available = [k for k in keys if k in frame.columns]
    if not available:
        return pd.Series(dtype="string")

    work = frame[["datetime", *available]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    work = work.dropna(subset=["datetime"]).copy()
    if work.empty:
        return pd.Series(dtype="string")

    for col in available:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    by_dt = work.groupby("datetime", as_index=True).mean(numeric_only=True).sort_index()
    if by_dt.empty:
        return pd.Series(dtype="string")

    vol = pd.to_numeric(by_dt.get("market_volatility_20"), errors="coerce")
    vol_thresh = vol.rolling(252, min_periods=20).median()

    state_rows: list[tuple[pd.Timestamp, str]] = []
    for dt, row in by_dt.iterrows():
        risk_off_votes = 0
        risk_on_votes = 0

        if np.isfinite(float(row.get("macro_regime_high_stress", np.nan))) and float(row["macro_regime_high_stress"]) > 0.5:
            risk_off_votes += 1
        if np.isfinite(float(row.get("regime_high_vol_downtrend", np.nan))) and float(row["regime_high_vol_downtrend"]) > 0.5:
            risk_off_votes += 1
        if np.isfinite(float(row.get("macro_risk_off_score", np.nan))) and float(row["macro_risk_off_score"]) > 0.5:
            risk_off_votes += 1
        if np.isfinite(float(row.get("market_drawdown_63", np.nan))) and float(row["market_drawdown_63"]) < -0.08:
            risk_off_votes += 1
        if dt in vol.index and np.isfinite(float(vol.get(dt, np.nan))) and np.isfinite(float(vol_thresh.get(dt, np.nan))):
            if float(vol.loc[dt]) > (1.1 * float(vol_thresh.loc[dt])):
                risk_off_votes += 1

        if np.isfinite(float(row.get("regime_low_vol_uptrend", np.nan))) and float(row["regime_low_vol_uptrend"]) > 0.5:
            risk_on_votes += 1
        if np.isfinite(float(row.get("market_return_20", np.nan))) and float(row["market_return_20"]) > 0:
            risk_on_votes += 1
        if dt in vol.index and np.isfinite(float(vol.get(dt, np.nan))) and np.isfinite(float(vol_thresh.get(dt, np.nan))):
            if float(vol.loc[dt]) < (0.9 * float(vol_thresh.loc[dt])):
                risk_on_votes += 1

        if risk_off_votes >= 2:
            state = "risk_off"
        elif risk_on_votes >= 2:
            state = "risk_on"
        else:
            state = "neutral"
        state_rows.append((dt, state))

    return pd.Series(
        [s for _, s in state_rows],
        index=pd.to_datetime([dt for dt, _ in state_rows], utc=True, errors="coerce"),
        dtype="string",
    ).sort_index()


def _build_summary_from_period_returns(
    period_returns: pd.Series,
    interval: str,
    pattern: str,
    model_file: str,
    trades: int | None = None,
    windows_used: int | None = None,
    extras: dict[str, object] | None = None,
) -> dict[str, float | int | str | None] | None:
    clean = pd.to_numeric(period_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return None
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index, utc=True, errors="coerce")
        clean = clean.loc[~pd.isna(clean.index)]
    if clean.empty:
        return None

    trade_count = int(trades) if trades is not None else int(len(clean))
    result = BacktestResult(
        pattern=str(pattern),
        model_file=str(model_file),
        trades=trade_count,
        win_rate=_win_rate(clean),
        win_rate_trade=_win_rate(clean),
        win_rate_period=_win_rate(clean),
        avg_trade_return=float(clean.mean()),
        cumulative_return=_safe_cumulative_return(clean),
        sharpe=_safe_sharpe_for_interval(clean, interval=interval),
        sortino=_safe_sortino_for_interval(clean, interval=interval),
        max_drawdown=_max_drawdown_from_period_returns(clean),
        profit_factor=_profit_factor(clean),
        annualized_return=_annualized_return_from_period_returns(clean, interval=interval),
        annualized_volatility=_annualized_volatility(clean, interval=interval),
        backtest_start_datetime=_to_iso_utc(clean.index.min()),
        backtest_end_datetime=_to_iso_utc(clean.index.max()),
    )
    row = result.to_dict()
    if windows_used is not None:
        row["windows_used"] = int(windows_used)
    if extras:
        row.update(extras)
    return row


def _build_spread_strategy_rows_and_curves(
    signals_all: pd.DataFrame,
    regime_frame: pd.DataFrame,
    universe_period_returns: pd.Series,
    interval: str,
    horizon_bars: int,
    fee_bps: float,
    spread_bps: float,
    slippage_bps: float,
    short_borrow_bps_per_day: float,
    latency_bars: int,
    lookback_bars: int,
    top_components: int,
    min_edge: float,
    switch_cost_bps: float,
    include_neutral_overlay: bool,
    include_regime_switch: bool,
    target_vol_annual: float,
    return_curves: bool,
    initial_investment: float,
    windows_used: int | None = None,
) -> tuple[list[dict[str, float | int | str | None]], list[pd.DataFrame]]:
    rows: list[dict[str, float | int | str | None]] = []
    curves: list[pd.DataFrame] = []
    if signals_all.empty:
        return rows, curves

    signal_frame = signals_all.copy()
    signal_frame["datetime"] = pd.to_datetime(signal_frame.get("datetime"), utc=True, errors="coerce")
    signal_frame = signal_frame.dropna(subset=["datetime"]).copy()
    if signal_frame.empty:
        return rows, curves

    def _build_strategy(
        strategy_name: str,
        component_col: str,
        beta_neutral: bool,
        size_neutral: bool,
    ) -> tuple[pd.Series, pd.DataFrame]:
        component_period = _build_component_period_table_from_signals(
            signals=signal_frame,
            component_col=component_col,
            interval=interval,
            horizon_bars=horizon_bars,
            fee_bps=fee_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            short_borrow_bps_per_day=short_borrow_bps_per_day,
            latency_bars=latency_bars,
        )
        if component_period.empty or component_period["component"].nunique() < 2:
            return pd.Series(dtype=float), pd.DataFrame()
        return _build_relative_strength_spread_period_returns(
            component_period_table=component_period,
            interval=interval,
            lookback_bars=lookback_bars,
            top_components=top_components,
            min_edge=min_edge,
            switch_cost_bps=switch_cost_bps,
            beta_neutral=beta_neutral,
            size_neutral=size_neutral,
            target_vol_annual=target_vol_annual,
            market_period_returns=universe_period_returns,
        )

    strategy_series: dict[str, pd.Series] = {}
    strategy_decisions: dict[str, pd.DataFrame] = {}

    model_component_col: str | None = None
    for candidate in ("source_model_file", "model_file"):
        if candidate in signal_frame.columns and signal_frame[candidate].astype(str).nunique() >= 2:
            model_component_col = candidate
            break
    if model_component_col is not None:
        model_plain, model_plain_dec = _build_strategy(
            strategy_name="spread_model_vs_model",
            component_col=model_component_col,
            beta_neutral=False,
            size_neutral=False,
        )
        if not model_plain.empty:
            strategy_series["spread_model_vs_model"] = model_plain
            strategy_decisions["spread_model_vs_model"] = model_plain_dec
        if include_neutral_overlay:
            model_neutral, model_neutral_dec = _build_strategy(
                strategy_name="spread_model_vs_model_neutral",
                component_col=model_component_col,
                beta_neutral=True,
                size_neutral=True,
            )
            if not model_neutral.empty:
                strategy_series["spread_model_vs_model_neutral"] = model_neutral
                strategy_decisions["spread_model_vs_model_neutral"] = model_neutral_dec

    if "pattern" in signal_frame.columns and signal_frame["pattern"].astype(str).nunique() >= 2:
        pattern_plain, pattern_plain_dec = _build_strategy(
            strategy_name="spread_pattern_vs_pattern",
            component_col="pattern",
            beta_neutral=False,
            size_neutral=False,
        )
        if not pattern_plain.empty:
            strategy_series["spread_pattern_vs_pattern"] = pattern_plain
            strategy_decisions["spread_pattern_vs_pattern"] = pattern_plain_dec
        if include_neutral_overlay:
            pattern_neutral, pattern_neutral_dec = _build_strategy(
                strategy_name="spread_pattern_vs_pattern_neutral",
                component_col="pattern",
                beta_neutral=True,
                size_neutral=True,
            )
            if not pattern_neutral.empty:
                strategy_series["spread_pattern_vs_pattern_neutral"] = pattern_neutral
                strategy_decisions["spread_pattern_vs_pattern_neutral"] = pattern_neutral_dec

    if include_regime_switch:
        model_key = "spread_model_vs_model_neutral" if "spread_model_vs_model_neutral" in strategy_series else "spread_model_vs_model"
        pattern_key = (
            "spread_pattern_vs_pattern_neutral"
            if "spread_pattern_vs_pattern_neutral" in strategy_series
            else "spread_pattern_vs_pattern"
        )
        model_series = strategy_series.get(model_key, pd.Series(dtype=float))
        pattern_series = strategy_series.get(pattern_key, pd.Series(dtype=float))
        if not model_series.empty or not pattern_series.empty:
            regime = _derive_regime_state_series(regime_frame)
            combined_index = model_series.index.union(pattern_series.index).sort_values()
            if len(combined_index) > 0:
                regime_aligned = regime.reindex(combined_index).ffill().fillna("neutral")
                model_aligned = model_series.reindex(combined_index)
                pattern_aligned = pattern_series.reindex(combined_index)
                switch_rows: list[tuple[pd.Timestamp, float]] = []
                switch_meta: list[dict[str, object]] = []
                for dt in combined_index:
                    state = str(regime_aligned.loc[dt])
                    model_ret = model_aligned.loc[dt]
                    pattern_ret = pattern_aligned.loc[dt]
                    chosen = "none"
                    out_ret = np.nan
                    if state == "risk_off":
                        if pd.notna(pattern_ret):
                            out_ret = float(pattern_ret)
                            chosen = pattern_key
                        elif pd.notna(model_ret):
                            out_ret = float(model_ret)
                            chosen = model_key
                    elif state == "risk_on":
                        if pd.notna(model_ret):
                            out_ret = float(model_ret)
                            chosen = model_key
                        elif pd.notna(pattern_ret):
                            out_ret = float(pattern_ret)
                            chosen = pattern_key
                    else:
                        vals = [float(v) for v in (model_ret, pattern_ret) if pd.notna(v)]
                        if vals:
                            out_ret = float(np.mean(vals))
                            chosen = "blend"
                    if pd.notna(out_ret):
                        switch_rows.append((dt, float(out_ret)))
                        switch_meta.append({"datetime": dt, "regime_state": state, "regime_choice": chosen})
                if switch_rows:
                    switch_series = pd.Series(
                        [v for _, v in switch_rows],
                        index=pd.to_datetime([dt for dt, _ in switch_rows], utc=True, errors="coerce"),
                        dtype=float,
                    ).sort_index()
                    strategy_series["spread_regime_switch"] = switch_series
                    strategy_decisions["spread_regime_switch"] = pd.DataFrame(switch_meta)

    for strategy_name, period in strategy_series.items():
        decisions = strategy_decisions.get(strategy_name, pd.DataFrame())
        if not decisions.empty and "pair_changed" in decisions.columns:
            switches = int(pd.to_numeric(decisions["pair_changed"], errors="coerce").fillna(0).sum())
        else:
            switches = 0
        components_used = 0
        if not decisions.empty and {"long_component", "short_component"}.issubset(decisions.columns):
            components_used = int(
                pd.concat([decisions["long_component"].astype(str), decisions["short_component"].astype(str)], ignore_index=True)
                .dropna()
                .nunique()
            )
        row = _build_summary_from_period_returns(
            period_returns=period,
            interval=interval,
            pattern=strategy_name,
            model_file=strategy_name,
            trades=int(len(period)),
            windows_used=windows_used,
            extras={
                "spread_strategy": strategy_name,
                "spread_switches": switches,
                "spread_components_used": components_used,
            },
        )
        if row is not None:
            rows.append(row)

        if return_curves and not period.empty:
            spread_curve = _build_equity_curve_frame(
                period_returns=period,
                pattern=strategy_name,
                model_file=strategy_name,
                initial_investment=initial_investment,
                curve_variant="ml_model",
            )
            if not spread_curve.empty:
                curves.append(spread_curve)
            if not universe_period_returns.empty:
                baseline = pd.to_numeric(universe_period_returns, errors="coerce").dropna().reindex(period.index).dropna()
                baseline_curve = _build_equity_curve_frame(
                    period_returns=baseline,
                    pattern=strategy_name,
                    model_file=strategy_name,
                    initial_investment=initial_investment,
                    curve_variant="baseline_universe_eqw",
                )
                if not baseline_curve.empty:
                    curves.append(baseline_curve)

    return rows, curves


def run_pattern_backtests(
    dataset: pd.DataFrame,
    model_io: PatternModelIO,
    interval: str,
    horizon_bars: int,
    long_threshold: float | None = 0.65,
    short_threshold: float | None = 0.35,
    fee_bps: float = 1.0,
    include_patterns: set[str] | None = None,
    include_model_files: set[str] | None = None,
    use_model_thresholds: bool = False,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    short_borrow_bps_per_day: float = 0.0,
    latency_bars: int = 0,
    embargo_bars: int = 0,
    parallel_models: int = 1,
    max_eval_rows_per_pattern: int | None = None,
    include_portfolio: bool = True,
    portfolio_top_k_per_side: int = 5,
    portfolio_max_gross_exposure: float = 1.0,
    portfolio_pattern_selection: str = "all",
    portfolio_best_patterns_top_n: int = 6,
    portfolio_min_pattern_trades: int = 40,
    portfolio_min_pattern_win_rate_trade: float = 0.55,
    portfolio_min_abs_score: float = 0.15,
    portfolio_rebalance_every_n_bars: int = 3,
    portfolio_symbol_cooldown_bars: int = 5,
    portfolio_volatility_scaling: bool = True,
    portfolio_max_symbol_weight: float = 0.35,
    include_spread_strategies: bool = False,
    spread_lookback_bars: int = 63,
    spread_top_components: int = 3,
    spread_min_edge: float = 0.02,
    spread_switch_cost_bps: float = 0.0,
    spread_include_neutral_overlay: bool = True,
    spread_include_regime_switch: bool = True,
    spread_target_vol_annual: float = 0.0,
    return_curves: bool = False,
    initial_investment: float = 10000.0,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    output_cols = [
        "pattern",
        "model_file",
        "backtest_start_datetime",
        "backtest_end_datetime",
        "trades",
        "win_rate",
        "win_rate_trade",
        "win_rate_period",
        "avg_trade_return",
        "cumulative_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "profit_factor",
        "annualized_return",
        "annualized_volatility",
        "portfolio_selection_mode",
        "portfolio_pattern_count",
        "portfolio_patterns_used",
        "spread_strategy",
        "spread_switches",
        "spread_components_used",
    ]
    if dataset.empty:
        empty_table = pd.DataFrame(columns=output_cols)
        if return_curves:
            return empty_table, pd.DataFrame(
                columns=[
                    "datetime",
                    "pattern",
                    "model_file",
                    "period_return",
                    "cumulative_return",
                    "equity_value",
                    "initial_investment",
                    "curve_variant",
                ]
            )
        return empty_table

    full = dataset.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
    full_dt = pd.to_datetime(full["datetime"], utc=True, errors="coerce")
    full["_realized_return_full"] = _compute_realized_return(
        full,
        horizon_bars=horizon_bars,
        latency_bars=latency_bars,
    )

    default_test = full.loc[full["split"] == "test"].copy() if "split" in full.columns else pd.DataFrame()
    if default_test.empty:
        cutoff = int(len(full) * 0.8)
        default_test = full.iloc[cutoff:].copy()
    universe_period_global = pd.Series(dtype=float)
    if not default_test.empty:
        default_test_rr = default_test.copy()
        default_test_rr["realized_return"] = pd.to_numeric(
            full.loc[default_test_rr.index, "_realized_return_full"],
            errors="coerce",
        )
        universe_period_global = _build_universe_period_returns(default_test_rr)

    default_train_end = pd.NaT
    if "split" in full.columns:
        train_dt = pd.to_datetime(full.loc[full["split"] == "train", "datetime"], utc=True, errors="coerce")
        if not train_dt.empty:
            default_train_end = train_dt.max()

    candidate_models: list[tuple] = []
    for model_path in model_io.list_models():
        model_file = model_path.name
        if include_model_files and model_file not in include_model_files:
            continue

        parsed = model_io.parse_model_filename(model_file)
        parsed_interval = parsed.get("interval")
        parsed_horizon = parsed.get("horizon_bars")
        parsed_pattern = parsed.get("pattern")
        if parsed_interval is not None and str(parsed_interval) != str(interval):
            continue
        if parsed_horizon is not None and int(parsed_horizon) != int(horizon_bars):
            continue
        if include_patterns and parsed_pattern is not None and str(parsed_pattern) not in include_patterns:
            continue
        candidate_models.append((model_path, model_file))

    if not candidate_models:
        empty_table = pd.DataFrame(columns=output_cols)
        if return_curves:
            return empty_table, pd.DataFrame(
                columns=[
                    "datetime",
                    "pattern",
                    "model_file",
                    "period_return",
                    "cumulative_return",
                    "equity_value",
                    "initial_investment",
                    "curve_variant",
                ]
            )
        return empty_table

    def evaluate_model(
        model_path,
        model_file: str,
    ) -> tuple[dict[str, float | int | str | None] | None, pd.DataFrame, pd.DataFrame]:
        try:
            payload = model_io.load_from_path(model_path)
        except Exception:
            return None, pd.DataFrame(), pd.DataFrame()

        if payload.get("interval") != interval or int(payload.get("horizon_bars", -1)) != int(horizon_bars):
            return None, pd.DataFrame(), pd.DataFrame()

        pattern = str(payload["pattern"])
        if include_patterns and pattern not in include_patterns:
            return None, pd.DataFrame(), pd.DataFrame()

        model_train_end = pd.to_datetime(payload.get("train_end_datetime"), utc=True, errors="coerce")
        model_test_start = pd.to_datetime(payload.get("test_start_datetime"), utc=True, errors="coerce")
        if pd.isna(model_train_end):
            model_train_end = default_train_end

        eval_cut_candidates = [t for t in [model_train_end, model_test_start] if pd.notna(t)]
        eval_cutoff = max(eval_cut_candidates) if eval_cut_candidates else pd.NaT
        if pd.notna(eval_cutoff):
            eval_cutoff = eval_cutoff + _bar_timedelta(interval, bars=max(0, int(embargo_bars)))
            eval_pool = full.loc[full_dt >= eval_cutoff].copy()
        else:
            eval_pool = default_test.copy()

        if eval_pool.empty:
            eval_pool = default_test.copy()
        if eval_pool.empty:
            return None, pd.DataFrame(), pd.DataFrame()

        eval_dt = pd.to_datetime(eval_pool["datetime"], utc=True, errors="coerce")
        eval_start_iso = _to_iso_utc(eval_dt.min())
        eval_end_iso = _to_iso_utc(eval_dt.max())

        eval_pool = eval_pool.sort_values(["symbol", "datetime"]).copy()
        eval_pool["realized_return"] = pd.to_numeric(
            full.loc[eval_pool.index, "_realized_return_full"],
            errors="coerce",
        )

        model = payload["model"]
        feature_cols = payload["feature_columns"]

        pattern_subset = _extract_pattern_subset(eval_pool, pattern=pattern)
        pattern_subset = _cap_rows_for_speed(pattern_subset, max_rows=max_eval_rows_per_pattern)
        if pattern_subset.empty:
            return None, pd.DataFrame(), pd.DataFrame()
        subset = pattern_subset.copy()

        x = subset.reindex(columns=feature_cols).fillna(subset.median(numeric_only=True)).fillna(0.0)
        prob_raw = model.predict_proba(x)[:, 1]
        prob = apply_probability_calibration(prob_raw, payload.get("probability_calibration"))
        long_t, short_t = resolve_thresholds(
            payload=payload,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            use_model_thresholds=use_model_thresholds,
        )

        subset["pred_prob_up"] = prob
        subset["position"] = 0
        subset.loc[subset["pred_prob_up"] >= long_t, "position"] = 1
        subset.loc[subset["pred_prob_up"] <= short_t, "position"] = -1

        pos_filtered, meta_prob = _apply_payload_meta_filter(
            payload=payload,
            subset=subset,
            feature_cols=feature_cols,
            prob_up=prob,
            positions=subset["position"].to_numpy(dtype=int),
        )
        subset["position"] = pos_filtered
        subset["meta_prob_trade"] = meta_prob
        subset = subset.loc[subset["position"] != 0].copy()
        if subset.empty:
            return None, pd.DataFrame(), pd.DataFrame()

        subset = subset.dropna(subset=["realized_return"]).copy()
        subset = _filter_extreme_realized_returns(subset, max_abs_return=3.0)
        if subset.empty:
            return None, pd.DataFrame(), pd.DataFrame()

        subset["strategy_return"] = _apply_execution_realism(
            subset=subset,
            interval=interval,
            horizon_bars=horizon_bars,
            fee_bps=fee_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            short_borrow_bps_per_day=short_borrow_bps_per_day,
            latency_bars=latency_bars,
        )
        signal_candidates = subset[["datetime", "symbol", "pred_prob_up", "realized_return"]].copy()
        signal_candidates["score"] = signal_candidates["pred_prob_up"] - 0.5
        signal_candidates["pattern"] = pattern
        signal_candidates["model_file"] = model_file
        signal_candidates["source_model_file"] = model_file
        signal_candidates["volatility_20"] = pd.to_numeric(subset.get("volatility_20"), errors="coerce")
        signal_candidates["beta_63"] = pd.to_numeric(subset.get("beta_63"), errors="coerce")
        signal_candidates["market_cap"] = pd.to_numeric(subset.get("market_cap"), errors="coerce")
        signal_candidates["meta_prob_trade"] = pd.to_numeric(subset.get("meta_prob_trade"), errors="coerce")

        period_returns = _aggregate_returns_by_datetime(subset, column="strategy_return")
        result = BacktestResult(
            pattern=pattern,
            model_file=model_file,
            trades=int(len(subset)),
            win_rate=_win_rate(period_returns),
            win_rate_trade=_win_rate(subset["strategy_return"]),
            win_rate_period=_win_rate(period_returns),
            avg_trade_return=float(subset["strategy_return"].mean()),
            cumulative_return=_safe_cumulative_return(period_returns),
            sharpe=_safe_sharpe_for_interval(period_returns, interval=interval),
            sortino=_safe_sortino_for_interval(period_returns, interval=interval),
            max_drawdown=_max_drawdown_from_period_returns(period_returns),
            profit_factor=_profit_factor(subset["strategy_return"]),
            annualized_return=_annualized_return_from_period_returns(period_returns, interval=interval),
            annualized_volatility=_annualized_volatility(period_returns, interval=interval),
            backtest_start_datetime=eval_start_iso,
            backtest_end_datetime=eval_end_iso,
        )
        curve_frames_local: list[pd.DataFrame] = []
        if return_curves:
            ml_curve = _build_equity_curve_frame(
                period_returns=period_returns,
                pattern=pattern,
                model_file=model_file,
                initial_investment=initial_investment,
                curve_variant="ml_model",
            )
            if not ml_curve.empty:
                curve_frames_local.append(ml_curve)

            blind_period = _build_blind_pattern_period_returns(
                pattern_subset=pattern_subset,
                pattern=pattern,
                interval=interval,
                horizon_bars=horizon_bars,
                fee_bps=fee_bps,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                short_borrow_bps_per_day=short_borrow_bps_per_day,
                latency_bars=latency_bars,
            )
            blind_curve = _build_equity_curve_frame(
                period_returns=blind_period,
                pattern=pattern,
                model_file=model_file,
                initial_investment=initial_investment,
                curve_variant="baseline_blind_pattern",
            )
            if not blind_curve.empty:
                curve_frames_local.append(blind_curve)

            universe_period = _build_universe_period_returns(eval_pool)
            universe_curve = _build_equity_curve_frame(
                period_returns=universe_period,
                pattern=pattern,
                model_file=model_file,
                initial_investment=initial_investment,
                curve_variant="baseline_universe_eqw",
            )
            if not universe_curve.empty:
                curve_frames_local.append(universe_curve)

        curve = pd.concat(curve_frames_local, ignore_index=True) if curve_frames_local else pd.DataFrame()
        return result.to_dict(), signal_candidates, curve

    rows: list[dict[str, float | int | str | None]] = []
    portfolio_signals: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    worker_count = max(1, int(parallel_models))
    if worker_count <= 1:
        for model_path, model_file in candidate_models:
            row, signals, curve = evaluate_model(model_path, model_file=model_file)
            if row is not None:
                rows.append(row)
            if not signals.empty:
                portfolio_signals.append(signals)
            if not curve.empty:
                curve_frames.append(curve)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(evaluate_model, model_path, model_file): model_file
                for model_path, model_file in candidate_models
            }
            for future in as_completed(futures):
                try:
                    row, signals, curve = future.result()
                except Exception:
                    continue
                if row is not None:
                    rows.append(row)
                if not signals.empty:
                    portfolio_signals.append(signals)
                if not curve.empty:
                    curve_frames.append(curve)

    if include_portfolio and portfolio_signals:
        signals_all = pd.concat(portfolio_signals, ignore_index=True)
        selection_mode = str(portfolio_pattern_selection or "all").strip().lower()
        if selection_mode not in {"all", "best", "both"}:
            selection_mode = "all"

        portfolio_jobs: list[tuple[str, pd.DataFrame, list[str] | None, str]] = []
        if selection_mode in {"all", "both"}:
            portfolio_jobs.append(
                (
                    "portfolio_combined",
                    signals_all,
                    None,
                    "all",
                )
            )
        if selection_mode in {"best", "both"}:
            best_patterns = _select_best_patterns_for_portfolio(
                rows=rows,
                top_n=int(portfolio_best_patterns_top_n),
                min_trades=int(portfolio_min_pattern_trades),
                min_win_rate_trade=float(portfolio_min_pattern_win_rate_trade),
            )
            if best_patterns:
                best_signals = signals_all.loc[signals_all["pattern"].astype(str).isin(set(best_patterns))].copy()
                if not best_signals.empty:
                    portfolio_jobs.append(
                        (
                            "portfolio_best_patterns",
                            best_signals,
                            best_patterns,
                            "best",
                        )
                    )

        for model_file, signal_frame, used_patterns, mode_label in portfolio_jobs:
            portfolio_payload = _build_portfolio_summary_from_signals(
                signals=signal_frame,
                interval=interval,
                fee_bps=fee_bps,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                short_borrow_bps_per_day=short_borrow_bps_per_day,
                horizon_bars=horizon_bars,
                latency_bars=latency_bars,
                top_k_per_side=portfolio_top_k_per_side,
                max_gross_exposure=portfolio_max_gross_exposure,
                model_file=model_file,
                return_period_series=return_curves,
                min_abs_score=float(portfolio_min_abs_score),
                rebalance_every_n_bars=int(portfolio_rebalance_every_n_bars),
                symbol_cooldown_bars=int(portfolio_symbol_cooldown_bars),
                volatility_scaling=bool(portfolio_volatility_scaling),
                max_symbol_weight=float(portfolio_max_symbol_weight),
            )
            if return_curves:
                portfolio_row = None
                portfolio_period = pd.Series(dtype=float)
                if isinstance(portfolio_payload, tuple):
                    portfolio_row, portfolio_period = portfolio_payload
                if portfolio_row is not None:
                    portfolio_row["pattern"] = model_file
                    portfolio_row["model_file"] = model_file
                    portfolio_row["portfolio_selection_mode"] = mode_label
                    portfolio_row["portfolio_pattern_count"] = int(len(used_patterns or []))
                    portfolio_row["portfolio_patterns_used"] = ",".join(sorted(used_patterns or []))
                    rows.append(portfolio_row)
                    portfolio_curve = _build_equity_curve_frame(
                        period_returns=portfolio_period,
                        pattern=str(portfolio_row.get("pattern", model_file)),
                        model_file=str(portfolio_row.get("model_file", model_file)),
                        initial_investment=initial_investment,
                    )
                    if not portfolio_curve.empty:
                        curve_frames.append(portfolio_curve)
            else:
                if isinstance(portfolio_payload, dict):
                    portfolio_payload["pattern"] = model_file
                    portfolio_payload["model_file"] = model_file
                    portfolio_payload["portfolio_selection_mode"] = mode_label
                    portfolio_payload["portfolio_pattern_count"] = int(len(used_patterns or []))
                    portfolio_payload["portfolio_patterns_used"] = ",".join(sorted(used_patterns or []))
                    rows.append(portfolio_payload)

    if include_spread_strategies and portfolio_signals:
        spread_rows, spread_curves = _build_spread_strategy_rows_and_curves(
            signals_all=pd.concat(portfolio_signals, ignore_index=True),
            regime_frame=full,
            universe_period_returns=universe_period_global,
            interval=interval,
            horizon_bars=horizon_bars,
            fee_bps=fee_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            short_borrow_bps_per_day=short_borrow_bps_per_day,
            latency_bars=latency_bars,
            lookback_bars=int(spread_lookback_bars),
            top_components=int(spread_top_components),
            min_edge=float(spread_min_edge),
            switch_cost_bps=float(spread_switch_cost_bps),
            include_neutral_overlay=bool(spread_include_neutral_overlay),
            include_regime_switch=bool(spread_include_regime_switch),
            target_vol_annual=float(spread_target_vol_annual),
            return_curves=bool(return_curves),
            initial_investment=float(initial_investment),
            windows_used=None,
        )
        if spread_rows:
            rows.extend(spread_rows)
        if spread_curves:
            curve_frames.extend(spread_curves)

    if not rows:
        empty_table = pd.DataFrame(columns=output_cols)
        if return_curves:
            return empty_table, pd.DataFrame(
                columns=[
                    "datetime",
                    "pattern",
                    "model_file",
                    "period_return",
                    "cumulative_return",
                    "equity_value",
                    "initial_investment",
                    "curve_variant",
                ]
            )
        return empty_table

    summary = pd.DataFrame(rows).sort_values("cumulative_return", ascending=False).reset_index(drop=True)
    if not return_curves:
        return summary

    curves = (
        pd.concat(curve_frames, ignore_index=True)
        if curve_frames
        else pd.DataFrame(
            columns=[
                "datetime",
                "pattern",
                "model_file",
                "period_return",
                "cumulative_return",
                "equity_value",
                "initial_investment",
                "curve_variant",
            ]
        )
    )
    if not curves.empty:
        curves["datetime"] = pd.to_datetime(curves["datetime"], utc=True, errors="coerce")
        curves = curves.dropna(subset=["datetime"]).sort_values(["model_file", "datetime"]).reset_index(drop=True)
    return summary, curves


def optimize_saved_model_thresholds(
    dataset: pd.DataFrame,
    model_io: PatternModelIO,
    interval: str,
    horizon_bars: int,
    fee_bps: float = 1.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    short_borrow_bps_per_day: float = 0.0,
    latency_bars: int = 1,
    embargo_bars: int = 1,
    include_patterns: set[str] | None = None,
    include_model_files: set[str] | None = None,
    long_grid: np.ndarray | None = None,
    short_grid: np.ndarray | None = None,
    min_trades: int = 40,
    max_eval_rows_per_pattern: int | None = None,
    parallel_models: int = 1,
    persist: bool = True,
) -> pd.DataFrame:
    output_cols = [
        "pattern",
        "model_file",
        "prev_long_threshold",
        "prev_short_threshold",
        "best_long_threshold",
        "best_short_threshold",
        "trades",
        "win_rate_period",
        "win_rate_trade",
        "avg_trade_return",
        "cumulative_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "profit_factor",
        "annualized_return",
        "annualized_volatility",
        "threshold_updated",
        "improvement_cumulative_return",
        "objective_score",
        "backtest_start_datetime",
        "backtest_end_datetime",
    ]
    if dataset.empty:
        return pd.DataFrame(columns=output_cols)

    full = dataset.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
    full_dt = pd.to_datetime(full["datetime"], utc=True, errors="coerce")
    full["_realized_return_full"] = _compute_realized_return(
        full,
        horizon_bars=horizon_bars,
        latency_bars=latency_bars,
    )

    default_test = full.loc[full["split"] == "test"].copy() if "split" in full.columns else pd.DataFrame()
    if default_test.empty:
        cutoff = int(len(full) * 0.8)
        default_test = full.iloc[cutoff:].copy()

    default_train_end = pd.NaT
    if "split" in full.columns:
        train_dt = pd.to_datetime(full.loc[full["split"] == "train", "datetime"], utc=True, errors="coerce")
        if not train_dt.empty:
            default_train_end = train_dt.max()

    candidate_models: list[tuple[Path, str]] = []
    for model_path in model_io.list_models():
        model_file = model_path.name
        if include_model_files and model_file not in include_model_files:
            continue
        parsed = model_io.parse_model_filename(model_file)
        parsed_interval = parsed.get("interval")
        parsed_horizon = parsed.get("horizon_bars")
        parsed_pattern = parsed.get("pattern")
        if parsed_interval is not None and str(parsed_interval) != str(interval):
            continue
        if parsed_horizon is not None and int(parsed_horizon) != int(horizon_bars):
            continue
        if include_patterns and parsed_pattern is not None and str(parsed_pattern) not in include_patterns:
            continue
        candidate_models.append((model_path, model_file))

    if not candidate_models:
        return pd.DataFrame(columns=output_cols)

    if long_grid is None:
        base_longs = np.arange(0.52, 0.82, 0.02)
    else:
        base_longs = np.asarray(long_grid, dtype=float)
    if short_grid is None:
        base_shorts = np.arange(0.18, 0.50, 0.02)
    else:
        base_shorts = np.asarray(short_grid, dtype=float)

    fee = float(fee_bps) / 10000.0
    spread = float(spread_bps) / 10000.0
    slippage = float(slippage_bps) / 10000.0
    trade_cost = fee + spread + (2.0 * slippage)
    borrow = (float(short_borrow_bps_per_day) / 10000.0) * _holding_days(
        interval=interval,
        horizon_bars=horizon_bars,
        latency_bars=latency_bars,
    )

    def _metrics_for_thresholds(
        dt: pd.Series,
        rr: np.ndarray,
        prob: np.ndarray,
        long_t: float,
        short_t: float,
        payload: dict[str, object],
        feature_frame: pd.DataFrame,
        feature_cols: list[str],
    ) -> dict[str, float | int | str | None] | None:
        pos = np.where(prob >= long_t, 1, np.where(prob <= short_t, -1, 0))
        pos, _ = _apply_payload_meta_filter(
            payload=payload,
            subset=feature_frame,
            feature_cols=feature_cols,
            prob_up=prob,
            positions=pos,
        )
        mask = pos != 0
        trades = int(mask.sum())
        if trades < int(min_trades):
            return None

        rr_sel = rr[mask]
        pos_sel = pos[mask]
        dt_sel = pd.to_datetime(dt.loc[mask], utc=True, errors="coerce")
        valid = pd.notna(dt_sel) & np.isfinite(rr_sel)
        if not bool(valid.any()):
            return None

        rr_sel = rr_sel[valid]
        pos_sel = pos_sel[valid]
        dt_sel = pd.to_datetime(dt_sel[valid], utc=True, errors="coerce")
        strat = (pos_sel * rr_sel) - trade_cost - np.where(pos_sel < 0, borrow, 0.0)
        strat_series = pd.Series(strat, index=dt_sel).sort_index()
        period = strat_series.groupby(level=0).mean()
        if period.empty:
            return None

        cum = _safe_cumulative_return(period)
        sharpe = _safe_sharpe_for_interval(period, interval=interval)
        sortino = _safe_sortino_for_interval(period, interval=interval)
        max_dd = _max_drawdown_from_period_returns(period)
        ann_ret = _annualized_return_from_period_returns(period, interval=interval)
        ann_vol = _annualized_volatility(period, interval=interval)
        trade_series = pd.Series(strat, dtype=float)
        pf = _profit_factor(trade_series)
        win_trade = _win_rate(trade_series)
        win_period = _win_rate(period)

        # Return-first objective: prioritize compounded return, then annualized return, then risk-adjusted quality.
        objective_tuple = (
            float(cum if np.isfinite(cum) else -1e12),
            float(ann_ret if np.isfinite(ann_ret) else -1e12),
            float(sharpe if np.isfinite(sharpe) else -1e12),
            float(sortino if np.isfinite(sortino) else -1e12),
            float(-abs(max_dd) if np.isfinite(max_dd) else -1e12),
            float(win_period if np.isfinite(win_period) else -1e12),
            float(trades),
        )

        return {
            "long_t": float(long_t),
            "short_t": float(short_t),
            "trades": int(trades),
            "win_rate_period": float(win_period),
            "win_rate_trade": float(win_trade),
            "avg_trade_return": float(trade_series.mean()) if not trade_series.empty else 0.0,
            "cumulative_return": float(cum),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_dd),
            "profit_factor": float(pf) if np.isfinite(pf) else np.nan,
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "backtest_start_datetime": _to_iso_utc(period.index.min()),
            "backtest_end_datetime": _to_iso_utc(period.index.max()),
            "objective_tuple": objective_tuple,
        }

    def _evaluate(model_path: Path, model_file: str) -> dict[str, float | int | str | None] | None:
        try:
            payload = model_io.load_from_path(model_path)
        except Exception:
            return None

        if payload.get("interval") != interval or int(payload.get("horizon_bars", -1)) != int(horizon_bars):
            return None
        pattern = str(payload.get("pattern", ""))
        if include_patterns and pattern not in include_patterns:
            return None

        model_train_end = pd.to_datetime(payload.get("train_end_datetime"), utc=True, errors="coerce")
        model_test_start = pd.to_datetime(payload.get("test_start_datetime"), utc=True, errors="coerce")
        if pd.isna(model_train_end):
            model_train_end = default_train_end
        eval_cut_candidates = [t for t in [model_train_end, model_test_start] if pd.notna(t)]
        eval_cutoff = max(eval_cut_candidates) if eval_cut_candidates else pd.NaT
        if pd.notna(eval_cutoff):
            eval_cutoff = eval_cutoff + _bar_timedelta(interval, bars=max(0, int(embargo_bars)))
            eval_pool = full.loc[full_dt >= eval_cutoff].copy()
        else:
            eval_pool = default_test.copy()
        if eval_pool.empty:
            eval_pool = default_test.copy()
        if eval_pool.empty:
            return None

        eval_pool = eval_pool.sort_values(["symbol", "datetime"]).copy()
        eval_pool["realized_return"] = pd.to_numeric(
            full.loc[eval_pool.index, "_realized_return_full"],
            errors="coerce",
        )
        subset = _extract_pattern_subset(eval_pool, pattern=pattern)
        subset = _cap_rows_for_speed(subset, max_rows=max_eval_rows_per_pattern)
        subset = subset.dropna(subset=["realized_return"]).copy()
        subset = _filter_extreme_realized_returns(subset, max_abs_return=3.0)
        if subset.empty:
            return None

        feature_cols = payload.get("feature_columns", [])
        model = payload.get("model")
        if model is None or not feature_cols:
            return None
        x = subset.reindex(columns=feature_cols).fillna(subset.median(numeric_only=True)).fillna(0.0)
        prob_raw = model.predict_proba(x)[:, 1]
        prob = apply_probability_calibration(prob_raw, payload.get("probability_calibration"))
        dt = pd.to_datetime(subset["datetime"], utc=True, errors="coerce")
        rr = pd.to_numeric(subset["realized_return"], errors="coerce").to_numpy(dtype=float)

        prev_long, prev_short = resolve_thresholds(
            payload=payload,
            long_threshold=None,
            short_threshold=None,
            use_model_thresholds=True,
        )
        local_longs = np.unique(
            np.clip(
                np.concatenate(
                    [
                        base_longs,
                        np.array([prev_long - 0.08, prev_long - 0.04, prev_long, prev_long + 0.04, prev_long + 0.08]),
                    ]
                ),
                0.5,
                0.95,
            )
        )
        local_shorts = np.unique(
            np.clip(
                np.concatenate(
                    [
                        base_shorts,
                        np.array([prev_short - 0.08, prev_short - 0.04, prev_short, prev_short + 0.04, prev_short + 0.08]),
                    ]
                ),
                0.05,
                0.5,
            )
        )

        baseline = _metrics_for_thresholds(
            dt=dt,
            rr=rr,
            prob=prob,
            long_t=float(prev_long),
            short_t=float(prev_short),
            payload=payload,
            feature_frame=subset,
            feature_cols=feature_cols,
        )
        best = baseline
        for long_t in local_longs:
            for short_t in local_shorts:
                if long_t <= short_t or (long_t - short_t) < 0.03:
                    continue
                metrics = _metrics_for_thresholds(
                    dt=dt,
                    rr=rr,
                    prob=prob,
                    long_t=float(long_t),
                    short_t=float(short_t),
                    payload=payload,
                    feature_frame=subset,
                    feature_cols=feature_cols,
                )
                if metrics is None:
                    continue
                if best is None or tuple(metrics["objective_tuple"]) > tuple(best["objective_tuple"]):
                    best = metrics

        if best is None:
            return None

        prev_cum = float(baseline["cumulative_return"]) if baseline is not None else float("nan")
        new_cum = float(best["cumulative_return"])
        improvement = float(new_cum - prev_cum) if np.isfinite(prev_cum) else np.nan
        updated = False
        if persist and (float(best["long_t"]) != float(prev_long) or float(best["short_t"]) != float(prev_short)):
            model_io.update_model_thresholds(
                model_file=model_file,
                tuned_thresholds={
                    "long": float(best["long_t"]),
                    "short": float(best["short_t"]),
                    "objective": float(best["objective_tuple"][0]),
                    "trades": int(best["trades"]),
                },
                optimization_meta={
                    "source": "recursive_backtest_threshold_optimizer",
                    "interval": str(interval),
                    "horizon_bars": int(horizon_bars),
                    "fee_bps": float(fee_bps),
                    "spread_bps": float(spread_bps),
                    "slippage_bps": float(slippage_bps),
                    "short_borrow_bps_per_day": float(short_borrow_bps_per_day),
                    "latency_bars": int(latency_bars),
                },
            )
            updated = True

        return {
            "pattern": pattern,
            "model_file": model_file,
            "prev_long_threshold": float(prev_long),
            "prev_short_threshold": float(prev_short),
            "best_long_threshold": float(best["long_t"]),
            "best_short_threshold": float(best["short_t"]),
            "trades": int(best["trades"]),
            "win_rate_period": float(best["win_rate_period"]),
            "win_rate_trade": float(best["win_rate_trade"]),
            "avg_trade_return": float(best["avg_trade_return"]),
            "cumulative_return": float(best["cumulative_return"]),
            "sharpe": float(best["sharpe"]),
            "sortino": float(best["sortino"]),
            "max_drawdown": float(best["max_drawdown"]),
            "profit_factor": float(best["profit_factor"]) if np.isfinite(float(best["profit_factor"])) else np.nan,
            "annualized_return": float(best["annualized_return"]),
            "annualized_volatility": float(best["annualized_volatility"]),
            "threshold_updated": bool(updated),
            "improvement_cumulative_return": float(improvement) if np.isfinite(improvement) else np.nan,
            "objective_score": float(best["objective_tuple"][0]),
            "backtest_start_datetime": best["backtest_start_datetime"],
            "backtest_end_datetime": best["backtest_end_datetime"],
        }

    rows: list[dict[str, float | int | str | None]] = []
    worker_count = max(1, int(parallel_models))
    if worker_count <= 1:
        for model_path, model_file in candidate_models:
            row = _evaluate(model_path, model_file)
            if row is not None:
                rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_evaluate, model_path, model_file): model_file for model_path, model_file in candidate_models}
            for future in as_completed(futures):
                try:
                    row = future.result()
                except Exception:
                    continue
                if row is not None:
                    rows.append(row)

    if not rows:
        return pd.DataFrame(columns=output_cols)
    return pd.DataFrame(rows).sort_values(["objective_score", "cumulative_return"], ascending=False).reset_index(drop=True)


def run_walk_forward_retraining_backtests(
    dataset: pd.DataFrame,
    interval: str,
    horizon_bars: int,
    train_window_days: int = 504,
    test_window_days: int = 63,
    step_days: int = 21,
    min_pattern_rows: int = 120,
    include_patterns: set[str] | None = None,
    fee_bps: float = 1.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    short_borrow_bps_per_day: float = 0.0,
    latency_bars: int = 1,
    embargo_bars: int = 1,
    fast_mode: bool = False,
    parallel_patterns: int = 1,
    max_windows_per_pattern: int | None = None,
    max_train_rows_per_window: int | None = None,
    include_portfolio: bool = True,
    portfolio_top_k_per_side: int = 5,
    portfolio_max_gross_exposure: float = 1.0,
    portfolio_pattern_selection: str = "all",
    portfolio_best_patterns_top_n: int = 6,
    portfolio_min_pattern_trades: int = 40,
    portfolio_min_pattern_win_rate_trade: float = 0.55,
    portfolio_min_abs_score: float = 0.15,
    portfolio_rebalance_every_n_bars: int = 3,
    portfolio_symbol_cooldown_bars: int = 5,
    portfolio_volatility_scaling: bool = True,
    portfolio_max_symbol_weight: float = 0.35,
    include_spread_strategies: bool = False,
    spread_lookback_bars: int = 63,
    spread_top_components: int = 3,
    spread_min_edge: float = 0.02,
    spread_switch_cost_bps: float = 0.0,
    spread_include_neutral_overlay: bool = True,
    spread_include_regime_switch: bool = True,
    spread_target_vol_annual: float = 0.0,
    return_curves: bool = False,
    initial_investment: float = 10000.0,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    output_cols = [
        "pattern",
        "model_file",
        "backtest_start_datetime",
        "backtest_end_datetime",
        "windows_used",
        "trades",
        "win_rate",
        "win_rate_trade",
        "win_rate_period",
        "avg_trade_return",
        "cumulative_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "profit_factor",
        "annualized_return",
        "annualized_volatility",
        "portfolio_selection_mode",
        "portfolio_pattern_count",
        "portfolio_patterns_used",
        "spread_strategy",
        "spread_switches",
        "spread_components_used",
    ]
    if dataset.empty:
        empty = pd.DataFrame()
        if return_curves:
            return empty, pd.DataFrame(
                columns=[
                    "datetime",
                    "pattern",
                    "model_file",
                    "period_return",
                    "cumulative_return",
                    "equity_value",
                    "initial_investment",
                    "curve_variant",
                ]
            )
        return empty

    work = dataset.copy().sort_values(["datetime", "symbol"]).reset_index(drop=True)
    work["datetime"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
    work = work.dropna(subset=["datetime"]).copy()
    if work.empty:
        empty = pd.DataFrame()
        if return_curves:
            return empty, pd.DataFrame(
                columns=[
                    "datetime",
                    "pattern",
                    "model_file",
                    "period_return",
                    "cumulative_return",
                    "equity_value",
                    "initial_investment",
                    "curve_variant",
                ]
            )
        return empty
    work["_realized_return_full"] = _compute_realized_return(
        work,
        horizon_bars=horizon_bars,
        latency_bars=latency_bars,
    )
    universe_period_global = _build_universe_period_returns(
        work[["datetime", "_realized_return_full"]].rename(columns={"_realized_return_full": "realized_return"})
    )
    work["_label_end_datetime"] = _compute_label_end_datetime(
        work,
        bars_ahead=max(1, int(horizon_bars) + max(0, int(latency_bars)) + max(0, int(embargo_bars))),
    )

    min_dt = work["datetime"].min()
    max_dt = work["datetime"].max()
    test_start = min_dt + pd.Timedelta(days=int(train_window_days))
    if test_start >= max_dt:
        empty = pd.DataFrame()
        if return_curves:
            return empty, pd.DataFrame(
                columns=[
                    "datetime",
                    "pattern",
                    "model_file",
                    "period_return",
                    "cumulative_return",
                    "equity_value",
                    "initial_investment",
                    "curve_variant",
                ]
            )
        return empty

    pattern_names = [c.replace("pattern_", "") for c in PATTERN_COLUMNS] + ["none"]
    if include_patterns:
        pattern_names = [p for p in pattern_names if p in include_patterns]

    effective_step_days = max(1, int(step_days))
    if effective_step_days < int(test_window_days):
        # Prevent overlapping test windows by default to avoid duplicated test observations.
        effective_step_days = int(test_window_days)

    windows_master: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = test_start
    while cursor < max_dt:
        train_start = cursor - pd.Timedelta(days=int(train_window_days))
        test_end = cursor + pd.Timedelta(days=int(test_window_days))
        windows_master.append((train_start, cursor, test_end))
        cursor = cursor + pd.Timedelta(days=effective_step_days)
    if max_windows_per_pattern is not None and int(max_windows_per_pattern) > 0 and len(windows_master) > int(max_windows_per_pattern):
        windows_master = windows_master[-int(max_windows_per_pattern) :]

    universe_period_by_window: dict[tuple[pd.Timestamp, pd.Timestamp], pd.Series] = {}
    if return_curves and windows_master:
        embargo_shift = _bar_timedelta(interval, bars=max(0, int(embargo_bars)))
        for _, window_start, test_end in windows_master:
            window_test_start = pd.to_datetime(window_start, utc=True, errors="coerce") + embargo_shift
            window_key = (window_test_start, pd.to_datetime(test_end, utc=True, errors="coerce"))
            universe_win = work.loc[
                (work["datetime"] >= window_test_start)
                & (work["datetime"] < test_end)
                & (work["_label_end_datetime"] < test_end)
            ][["datetime", "_realized_return_full"]].copy()
            universe_win = universe_win.rename(columns={"_realized_return_full": "realized_return"})
            universe_period_by_window[window_key] = _build_universe_period_returns(universe_win)

    rows: list[dict[str, float | int | str | None]] = []
    portfolio_signals: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    def _evaluate_pattern(pattern: str) -> tuple[dict[str, float | int | str | None] | None, pd.DataFrame, pd.DataFrame]:
        subset_pattern = _extract_pattern_subset(work, pattern=pattern)
        if subset_pattern.empty:
            return None, pd.DataFrame(), pd.DataFrame()
        subset_pattern["realized_return"] = pd.to_numeric(
            work.loc[subset_pattern.index, "_realized_return_full"],
            errors="coerce",
        )

        window_trades: list[pd.DataFrame] = []
        blind_window_frames: list[pd.DataFrame] = []
        used_window_keys: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        used_window_starts: list[pd.Timestamp] = []
        used_window_ends: list[pd.Timestamp] = []
        windows_used = 0
        cached_feature_cols: list[str] | None = None

        for train_start, window_start, test_end in windows_master:
            window_test_start = pd.to_datetime(window_start, utc=True, errors="coerce") + _bar_timedelta(
                interval,
                bars=max(0, int(embargo_bars)),
            )
            train_win = subset_pattern.loc[
                (subset_pattern["datetime"] >= train_start)
                & (subset_pattern["datetime"] < window_start)
                & (subset_pattern["_label_end_datetime"] < window_test_start)
            ].copy()
            test_win = subset_pattern.loc[
                (subset_pattern["datetime"] >= window_test_start)
                & (subset_pattern["datetime"] < test_end)
                & (subset_pattern["_label_end_datetime"] < test_end)
            ].copy()

            train_win = _cap_rows_for_speed(train_win, max_rows=max_train_rows_per_window)
            if len(train_win) < int(min_pattern_rows) or test_win.empty:
                continue
            baseline_test_win = test_win.copy()

            train_win = train_win.sort_values("datetime").reset_index(drop=True)
            x_train_raw, y_train, _ = split_xy(train_win)

            feature_cols: list[str] = []
            if fast_mode and cached_feature_cols:
                available = [c for c in cached_feature_cols if c in x_train_raw.columns]
                if len(available) >= 10:
                    feature_cols = available

            if not feature_cols:
                selection = select_features(
                    x_train=x_train_raw,
                    y_train=y_train,
                    config=FeatureSelectionConfig(
                        min_features=15 if fast_mode else 25,
                        max_features=120 if fast_mode else 260,
                        random_state=42,
                    ),
                )
                feature_cols = selection.selected_features
                if fast_mode and feature_cols:
                    cached_feature_cols = feature_cols

            if not feature_cols:
                continue

            split_idx = int(round(len(train_win) * 0.8))
            split_idx = max(1, min(split_idx, len(train_win) - 1))
            fit_win = train_win.iloc[:split_idx].copy()
            cal_win = train_win.iloc[split_idx:].copy()

            x_fit = fit_win.reindex(columns=feature_cols).fillna(x_train_raw.median(numeric_only=True)).fillna(0.0)
            y_fit = fit_win["future_direction"].astype(int)
            if y_fit.nunique() < 2:
                continue

            model = build_stacking_classifier(random_state=42, n_jobs=1, fast_mode=fast_mode)
            model.fit(x_fit, y_fit)

            calibration = None
            tuned_long = 0.65
            tuned_short = 0.35
            meta_model = None
            meta_threshold = 0.55
            meta_feature_cols: list[str] = []
            if not cal_win.empty and not fast_mode:
                x_cal = cal_win.reindex(columns=feature_cols).fillna(x_fit.median(numeric_only=True)).fillna(0.0)
                y_cal = cal_win["future_direction"].astype(int)
                if y_cal.nunique() >= 2:
                    prob_cal_raw = model.predict_proba(x_cal)[:, 1]
                    calibration = fit_probability_calibration(prob_cal_raw, y_cal)
                    prob_cal = apply_probability_calibration(prob_cal_raw, calibration)
                    tuned = optimize_thresholds_from_validation(
                        prob_up=prob_cal,
                        future_return=cal_win["future_return"],
                        future_excess_return=cal_win["future_excess_return"] if "future_excess_return" in cal_win.columns else None,
                        timestamps=cal_win["datetime"] if "datetime" in cal_win.columns else None,
                        min_trades=max(10, int(len(cal_win) * 0.05)),
                        min_trades_per_side=max(4, int(len(cal_win) * 0.01)),
                        transaction_cost_bps=float(fee_bps) + float(spread_bps) + (2.0 * float(slippage_bps)),
                        periods_per_year=_periods_per_year(interval),
                        exposure_penalty=0.4,
                        imbalance_penalty=0.25,
                    )
                    tuned_long = float(tuned.long_threshold)
                    tuned_short = float(tuned.short_threshold)

                    pos_cal = np.where(prob_cal >= tuned_long, 1, np.where(prob_cal <= tuned_short, -1, 0))
                    active_cal = pos_cal != 0
                    if int(active_cal.sum()) >= 120:
                        rr_cal = pd.to_numeric(cal_win["future_return"], errors="coerce").to_numpy(dtype=float)
                        per_trade_cost = (
                            float(fee_bps) + float(spread_bps) + (2.0 * float(slippage_bps))
                        ) / 10000.0
                        net_cal = (pos_cal * rr_cal) - per_trade_cost
                        y_meta = (net_cal > 0).astype(int)
                        y_meta_active = y_meta[active_cal]
                        pos_count = int((y_meta_active == 1).sum())
                        neg_count = int((y_meta_active == 0).sum())
                        if pos_count >= 20 and neg_count >= 20:
                            meta_frame = _build_meta_filter_features(
                                cal_win,
                                base_feature_cols=feature_cols,
                                prob_up=prob_cal,
                                positions=pos_cal,
                            )
                            x_meta = meta_frame.loc[active_cal].copy()
                            meta_feature_cols = list(x_meta.columns)
                            x_meta = x_meta.fillna(x_meta.median(numeric_only=True)).fillna(0.0)
                            y_meta_series = pd.Series(y_meta_active, index=x_meta.index, dtype=int)
                            if y_meta_series.nunique() >= 2:
                                candidate_meta = LogisticRegression(
                                    max_iter=600,
                                    class_weight="balanced",
                                    solver="lbfgs",
                                    random_state=42,
                                )
                                try:
                                    candidate_meta.fit(x_meta, y_meta_series)
                                    meta_model = candidate_meta
                                    meta_threshold = 0.55
                                except Exception:
                                    meta_model = None

            x_test = test_win.reindex(columns=feature_cols).fillna(x_fit.median(numeric_only=True)).fillna(0.0)
            prob_test_raw = model.predict_proba(x_test)[:, 1]
            prob_test = apply_probability_calibration(prob_test_raw, calibration)

            test_win["pred_prob_up"] = prob_test
            test_win["position"] = 0
            test_win.loc[test_win["pred_prob_up"] >= tuned_long, "position"] = 1
            test_win.loc[test_win["pred_prob_up"] <= tuned_short, "position"] = -1
            if meta_model is not None and meta_feature_cols:
                pos_test = test_win["position"].to_numpy(dtype=int)
                active_test = pos_test != 0
                if bool(active_test.any()):
                    meta_test = _build_meta_filter_features(
                        test_win,
                        base_feature_cols=feature_cols,
                        prob_up=prob_test,
                        positions=pos_test,
                    )
                    x_meta_test = meta_test.loc[active_test].reindex(columns=meta_feature_cols).copy()
                    x_meta_test = x_meta_test.fillna(x_meta_test.median(numeric_only=True)).fillna(0.0)
                    try:
                        meta_prob_test = meta_model.predict_proba(x_meta_test)[:, 1]
                        keep_test = meta_prob_test >= float(meta_threshold)
                        active_idx = np.where(active_test)[0]
                        pos_test[active_idx[~keep_test]] = 0
                        test_win["position"] = pos_test
                        meta_prob_all = np.full(len(test_win), np.nan, dtype=float)
                        meta_prob_all[active_idx] = meta_prob_test
                        test_win["meta_prob_trade"] = meta_prob_all
                    except Exception:
                        pass
            test_win = test_win.loc[test_win["position"] != 0].copy()
            if test_win.empty:
                continue

            test_win = test_win.dropna(subset=["realized_return"]).copy()
            test_win = _filter_extreme_realized_returns(test_win, max_abs_return=3.0)
            if test_win.empty:
                continue

            test_win["strategy_return"] = _apply_execution_realism(
                subset=test_win,
                interval=interval,
                horizon_bars=horizon_bars,
                fee_bps=fee_bps,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                short_borrow_bps_per_day=short_borrow_bps_per_day,
                latency_bars=latency_bars,
            )
            test_win["window_start"] = pd.to_datetime(window_test_start, utc=True, errors="coerce")
            export_cols = [
                "datetime",
                "symbol",
                "pred_prob_up",
                "realized_return",
                "strategy_return",
                "window_start",
                "volatility_20",
                "beta_63",
                "market_cap",
                "meta_prob_trade",
            ]
            for col in export_cols:
                if col not in test_win.columns:
                    test_win[col] = np.nan
            window_trades.append(test_win[export_cols].copy())
            baseline_test_win = baseline_test_win.dropna(subset=["realized_return"]).copy()
            baseline_test_win = _filter_extreme_realized_returns(baseline_test_win, max_abs_return=3.0)
            if not baseline_test_win.empty:
                blind_window_frames.append(baseline_test_win[["datetime", "symbol", "realized_return"]].copy())
            used_window_starts.append(window_test_start)
            used_window_ends.append(test_end)
            used_window_keys.append((window_test_start, pd.to_datetime(test_end, utc=True, errors="coerce")))
            windows_used += 1

        if not window_trades:
            return None, pd.DataFrame(), pd.DataFrame()

        trades_frame = pd.concat(window_trades, ignore_index=True)
        trades_frame = _dedupe_by_symbol_datetime(trades_frame)
        if trades_frame.empty:
            return None, pd.DataFrame(), pd.DataFrame()

        strat = pd.to_numeric(trades_frame["strategy_return"], errors="coerce").dropna().astype(float)
        if strat.empty:
            return None, pd.DataFrame(), pd.DataFrame()
        trades = int(len(strat))

        period_returns = _aggregate_returns_by_datetime(trades_frame, column="strategy_return")
        trade_dt = pd.to_datetime(trades_frame["datetime"], utc=True, errors="coerce")
        backtest_start_iso = _to_iso_utc(trade_dt.min()) if not trade_dt.empty else _to_iso_utc(min(used_window_starts))
        backtest_end_iso = _to_iso_utc(trade_dt.max()) if not trade_dt.empty else _to_iso_utc(max(used_window_ends))
        row = {
            "pattern": pattern,
            "model_file": "walk_forward_retrain",
            "backtest_start_datetime": backtest_start_iso,
            "backtest_end_datetime": backtest_end_iso,
            "windows_used": int(windows_used),
            "trades": int(trades),
            "win_rate": _win_rate(period_returns),
            "win_rate_trade": _win_rate(strat),
            "win_rate_period": _win_rate(period_returns),
            "avg_trade_return": float(strat.mean()),
            "cumulative_return": _safe_cumulative_return(period_returns),
            "sharpe": _safe_sharpe_for_interval(period_returns, interval=interval),
            "sortino": _safe_sortino_for_interval(period_returns, interval=interval),
            "max_drawdown": _max_drawdown_from_period_returns(period_returns),
            "profit_factor": _profit_factor(strat),
            "annualized_return": _annualized_return_from_period_returns(period_returns, interval=interval),
            "annualized_volatility": _annualized_volatility(period_returns, interval=interval),
        }
        curve_frames_local: list[pd.DataFrame] = []
        if return_curves:
            ml_curve = _build_equity_curve_frame(
                period_returns=period_returns,
                pattern=pattern,
                model_file="walk_forward_retrain",
                initial_investment=initial_investment,
                curve_variant="ml_model",
            )
            if not ml_curve.empty:
                curve_frames_local.append(ml_curve)

            if blind_window_frames:
                blind_source = pd.concat(blind_window_frames, ignore_index=True)
                blind_period = _build_blind_pattern_period_returns(
                    pattern_subset=blind_source,
                    pattern=pattern,
                    interval=interval,
                    horizon_bars=horizon_bars,
                    fee_bps=fee_bps,
                    spread_bps=spread_bps,
                    slippage_bps=slippage_bps,
                    short_borrow_bps_per_day=short_borrow_bps_per_day,
                    latency_bars=latency_bars,
                )
                blind_curve = _build_equity_curve_frame(
                    period_returns=blind_period,
                    pattern=pattern,
                    model_file="walk_forward_retrain",
                    initial_investment=initial_investment,
                    curve_variant="baseline_blind_pattern",
                )
                if not blind_curve.empty:
                    curve_frames_local.append(blind_curve)

            universe_period_parts = [
                universe_period_by_window.get(key, pd.Series(dtype=float))
                for key in used_window_keys
            ]
            universe_period_parts = [series for series in universe_period_parts if not series.empty]
            if universe_period_parts:
                universe_period = pd.concat(universe_period_parts).groupby(level=0).mean().sort_index()
                universe_curve = _build_equity_curve_frame(
                    period_returns=universe_period,
                    pattern=pattern,
                    model_file="walk_forward_retrain",
                    initial_investment=initial_investment,
                    curve_variant="baseline_universe_eqw",
                )
                if not universe_curve.empty:
                    curve_frames_local.append(universe_curve)

        curve = pd.concat(curve_frames_local, ignore_index=True) if curve_frames_local else pd.DataFrame()
        signals = trades_frame[["datetime", "symbol", "pred_prob_up", "realized_return"]].copy()
        signals["score"] = signals["pred_prob_up"] - 0.5
        signals["pattern"] = pattern
        signals["model_file"] = "walk_forward_retrain"
        signals["source_model_file"] = "walk_forward_retrain"
        signals["volatility_20"] = pd.to_numeric(trades_frame.get("volatility_20"), errors="coerce")
        signals["beta_63"] = pd.to_numeric(trades_frame.get("beta_63"), errors="coerce")
        signals["market_cap"] = pd.to_numeric(trades_frame.get("market_cap"), errors="coerce")
        signals["meta_prob_trade"] = pd.to_numeric(trades_frame.get("meta_prob_trade"), errors="coerce")
        return row, signals, curve

    worker_count = max(1, int(parallel_patterns))
    if worker_count <= 1:
        for pattern in pattern_names:
            row, signals, curve = _evaluate_pattern(pattern)
            if row is not None:
                rows.append(row)
            if not signals.empty:
                portfolio_signals.append(signals)
            if not curve.empty:
                curve_frames.append(curve)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(_evaluate_pattern, pattern): pattern for pattern in pattern_names}
            for future in as_completed(futures):
                try:
                    row, signals, curve = future.result()
                except Exception:
                    continue
                if row is not None:
                    rows.append(row)
                if not signals.empty:
                    portfolio_signals.append(signals)
                if not curve.empty:
                    curve_frames.append(curve)

    if include_portfolio and portfolio_signals:
        signals_all = pd.concat(portfolio_signals, ignore_index=True)
        selection_mode = str(portfolio_pattern_selection or "all").strip().lower()
        if selection_mode not in {"all", "best", "both"}:
            selection_mode = "all"

        portfolio_jobs: list[tuple[str, pd.DataFrame, list[str] | None, str]] = []
        if selection_mode in {"all", "both"}:
            portfolio_jobs.append(
                (
                    "portfolio_combined_walk_forward",
                    signals_all,
                    None,
                    "all",
                )
            )
        if selection_mode in {"best", "both"}:
            best_patterns = _select_best_patterns_for_portfolio(
                rows=rows,
                top_n=int(portfolio_best_patterns_top_n),
                min_trades=int(portfolio_min_pattern_trades),
                min_win_rate_trade=float(portfolio_min_pattern_win_rate_trade),
            )
            if best_patterns:
                best_signals = signals_all.loc[signals_all["pattern"].astype(str).isin(set(best_patterns))].copy()
                if not best_signals.empty:
                    portfolio_jobs.append(
                        (
                            "portfolio_best_patterns_walk_forward",
                            best_signals,
                            best_patterns,
                            "best",
                        )
                    )

        for model_file, signal_frame, used_patterns, mode_label in portfolio_jobs:
            portfolio_payload = _build_portfolio_summary_from_signals(
                signals=signal_frame,
                interval=interval,
                fee_bps=fee_bps,
                spread_bps=spread_bps,
                slippage_bps=slippage_bps,
                short_borrow_bps_per_day=short_borrow_bps_per_day,
                horizon_bars=horizon_bars,
                latency_bars=latency_bars,
                top_k_per_side=portfolio_top_k_per_side,
                max_gross_exposure=portfolio_max_gross_exposure,
                model_file=model_file,
                return_period_series=return_curves,
                min_abs_score=float(portfolio_min_abs_score),
                rebalance_every_n_bars=int(portfolio_rebalance_every_n_bars),
                symbol_cooldown_bars=int(portfolio_symbol_cooldown_bars),
                volatility_scaling=bool(portfolio_volatility_scaling),
                max_symbol_weight=float(portfolio_max_symbol_weight),
            )
            unique_periods = int(pd.to_datetime(signal_frame["datetime"], utc=True, errors="coerce").dropna().nunique())
            if return_curves:
                portfolio_row = None
                portfolio_period = pd.Series(dtype=float)
                if isinstance(portfolio_payload, tuple):
                    portfolio_row, portfolio_period = portfolio_payload
                if portfolio_row is not None:
                    portfolio_row["pattern"] = model_file
                    portfolio_row["model_file"] = model_file
                    portfolio_row["windows_used"] = unique_periods
                    portfolio_row["portfolio_selection_mode"] = mode_label
                    portfolio_row["portfolio_pattern_count"] = int(len(used_patterns or []))
                    portfolio_row["portfolio_patterns_used"] = ",".join(sorted(used_patterns or []))
                    rows.append(portfolio_row)
                    portfolio_curve = _build_equity_curve_frame(
                        period_returns=portfolio_period,
                        pattern=str(portfolio_row.get("pattern", model_file)),
                        model_file=str(portfolio_row.get("model_file", model_file)),
                        initial_investment=initial_investment,
                    )
                    if not portfolio_curve.empty:
                        curve_frames.append(portfolio_curve)
            else:
                portfolio_row = portfolio_payload if isinstance(portfolio_payload, dict) else None
                if portfolio_row is not None:
                    portfolio_row["pattern"] = model_file
                    portfolio_row["model_file"] = model_file
                    portfolio_row["windows_used"] = unique_periods
                    portfolio_row["portfolio_selection_mode"] = mode_label
                    portfolio_row["portfolio_pattern_count"] = int(len(used_patterns or []))
                    portfolio_row["portfolio_patterns_used"] = ",".join(sorted(used_patterns or []))
                    rows.append(portfolio_row)

    if include_spread_strategies and portfolio_signals:
        spread_signals = pd.concat(portfolio_signals, ignore_index=True)
        spread_windows_used = int(pd.to_datetime(spread_signals["datetime"], utc=True, errors="coerce").dropna().nunique())
        spread_rows, spread_curves = _build_spread_strategy_rows_and_curves(
            signals_all=spread_signals,
            regime_frame=work,
            universe_period_returns=universe_period_global,
            interval=interval,
            horizon_bars=horizon_bars,
            fee_bps=fee_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            short_borrow_bps_per_day=short_borrow_bps_per_day,
            latency_bars=latency_bars,
            lookback_bars=int(spread_lookback_bars),
            top_components=int(spread_top_components),
            min_edge=float(spread_min_edge),
            switch_cost_bps=float(spread_switch_cost_bps),
            include_neutral_overlay=bool(spread_include_neutral_overlay),
            include_regime_switch=bool(spread_include_regime_switch),
            target_vol_annual=float(spread_target_vol_annual),
            return_curves=bool(return_curves),
            initial_investment=float(initial_investment),
            windows_used=spread_windows_used,
        )
        if spread_rows:
            rows.extend(spread_rows)
        if spread_curves:
            curve_frames.extend(spread_curves)

    if not rows:
        empty = pd.DataFrame(columns=output_cols)
        if return_curves:
            return empty, pd.DataFrame(
                columns=[
                    "datetime",
                    "pattern",
                    "model_file",
                    "period_return",
                    "cumulative_return",
                    "equity_value",
                    "initial_investment",
                    "curve_variant",
                ]
            )
        return empty

    summary = pd.DataFrame(rows).sort_values("cumulative_return", ascending=False).reset_index(drop=True)
    if not return_curves:
        return summary

    curves = (
        pd.concat(curve_frames, ignore_index=True)
        if curve_frames
        else pd.DataFrame(
            columns=[
                "datetime",
                "pattern",
                "model_file",
                "period_return",
                "cumulative_return",
                "equity_value",
                "initial_investment",
                "curve_variant",
            ]
        )
    )
    if not curves.empty:
        curves["datetime"] = pd.to_datetime(curves["datetime"], utc=True, errors="coerce")
        curves = curves.dropna(subset=["datetime"]).sort_values(["model_file", "datetime"]).reset_index(drop=True)
    return summary, curves


def summarize_pattern_coverage(dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame(columns=["pattern", "count", "share"])

    total = len(dataset)
    rows: list[dict[str, float | int | str]] = []
    for col in PATTERN_COLUMNS:
        count = int(dataset[col].sum()) if col in dataset.columns else 0
        rows.append({"pattern": col.replace("pattern_", ""), "count": count, "share": count / total})

    none_count = int(dataset["pattern"].isna().sum()) if "pattern" in dataset.columns else 0
    rows.append({"pattern": "none", "count": none_count, "share": none_count / total})
    return pd.DataFrame(rows).sort_values("count", ascending=False)

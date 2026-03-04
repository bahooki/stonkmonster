from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class ThresholdOptimizationResult:
    long_threshold: float
    short_threshold: float
    objective: float
    trades: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "long": float(self.long_threshold),
            "short": float(self.short_threshold),
            "objective": float(self.objective),
            "trades": int(self.trades),
        }


def _clip_probs(prob_up: np.ndarray) -> np.ndarray:
    return np.clip(prob_up.astype(float), 1e-6, 1 - 1e-6)


def fit_probability_calibration(prob_up: np.ndarray, y_true: pd.Series) -> dict[str, float] | None:
    y = pd.Series(y_true).astype(int)
    if len(y) < 40 or y.nunique() < 2:
        return None

    clipped = _clip_probs(np.asarray(prob_up))
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(max_iter=500)
        model.fit(logits, y.values)
    except Exception:
        return None

    return {
        "method": "platt_logit",
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
    }


def apply_probability_calibration(prob_up: np.ndarray, calibration: dict[str, float] | None) -> np.ndarray:
    clipped = _clip_probs(np.asarray(prob_up))
    if not calibration:
        return clipped

    method = str(calibration.get("method", ""))
    if method != "platt_logit":
        return clipped

    coef = float(calibration.get("coef", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    logits = np.log(clipped / (1.0 - clipped))
    z = coef * logits + intercept
    calibrated = 1.0 / (1.0 + np.exp(-z))
    return _clip_probs(calibrated)


def resolve_thresholds(
    payload: dict[str, object],
    long_threshold: float | None,
    short_threshold: float | None,
    use_model_thresholds: bool = False,
) -> tuple[float, float]:
    tuned = payload.get("tuned_thresholds", {}) if isinstance(payload, dict) else {}
    tuned_long = float(tuned.get("long", 0.55)) if isinstance(tuned, dict) else 0.55
    tuned_short = float(tuned.get("short", 0.45)) if isinstance(tuned, dict) else 0.45

    if use_model_thresholds:
        return tuned_long, tuned_short

    resolved_long = tuned_long if long_threshold is None else float(long_threshold)
    resolved_short = tuned_short if short_threshold is None else float(short_threshold)
    return resolved_long, resolved_short


def optimize_thresholds_from_validation(
    prob_up: np.ndarray,
    future_return: pd.Series,
    future_excess_return: pd.Series | None = None,
    timestamps: pd.Series | None = None,
    long_grid: np.ndarray | None = None,
    short_grid: np.ndarray | None = None,
    min_trades: int = 20,
    min_trades_per_side: int | None = None,
    transaction_cost_bps: float = 0.0,
    periods_per_year: float = 252.0,
    exposure_penalty: float = 0.35,
    imbalance_penalty: float = 0.2,
) -> ThresholdOptimizationResult:
    probs = _clip_probs(np.asarray(prob_up))
    target = pd.Series(future_excess_return if future_excess_return is not None else future_return).astype(float).to_numpy()
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce") if timestamps is not None else None

    long_values = long_grid if long_grid is not None else np.arange(0.53, 0.71, 0.01)
    short_values = short_grid if short_grid is not None else np.arange(0.30, 0.48, 0.01)
    per_trade_cost = float(transaction_cost_bps) / 10000.0
    min_side = int(min_trades_per_side) if min_trades_per_side is not None else max(5, int(min_trades) // 4)

    best = ThresholdOptimizationResult(long_threshold=0.55, short_threshold=0.45, objective=float("-inf"), trades=0)

    for long_t in long_values:
        for short_t in short_values:
            if long_t <= short_t:
                continue
            positions = np.where(probs >= long_t, 1, np.where(probs <= short_t, -1, 0))
            trade_mask = positions != 0
            trades = int(trade_mask.sum())
            if trades < min_trades:
                continue
            long_trades = int((positions[trade_mask] > 0).sum())
            short_trades = int((positions[trade_mask] < 0).sum())
            if long_trades < min_side or short_trades < min_side:
                continue

            strat = positions[trade_mask] * target[trade_mask] - per_trade_cost
            if len(strat) == 0:
                continue
            if ts is None:
                period = pd.Series(strat, dtype=float)
            else:
                ts_sel = pd.to_datetime(ts.loc[trade_mask], utc=True, errors="coerce")
                valid = np.isfinite(strat) & pd.notna(ts_sel).to_numpy(dtype=bool)
                if not bool(valid.any()):
                    continue
                strat_valid = strat[valid]
                ts_valid = pd.to_datetime(ts_sel[valid], utc=True, errors="coerce")
                period = pd.Series(strat_valid, index=ts_valid, dtype=float).groupby(level=0).mean().sort_index()
                if period.empty:
                    continue

            period_clean = pd.to_numeric(period, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if period_clean.empty:
                continue
            clipped = period_clean.clip(lower=-0.999999999)
            cumulative_return = float(np.expm1(np.log1p(clipped).sum()))
            mean_ret = float(period_clean.mean())
            std_ret = float(period_clean.std(ddof=1))
            sharpe = float((mean_ret / std_ret) * np.sqrt(float(periods_per_year))) if std_ret > 1e-12 else 0.0
            downside = period_clean.loc[period_clean < 0]
            downside_std = float(downside.std(ddof=1)) if len(downside) >= 2 else 0.0
            sortino = float((mean_ret / downside_std) * np.sqrt(float(periods_per_year))) if downside_std > 1e-12 else 0.0
            growth = np.exp(np.log1p(clipped).cumsum())
            max_dd = float(np.nanmin(growth / np.maximum.accumulate(growth) - 1.0)) if len(growth) else 0.0
            win_period = float((period_clean > 0).mean())
            net_exposure = float(np.nanmean(positions[trade_mask]))
            long_share = long_trades / max(1, trades)
            short_share = short_trades / max(1, trades)
            imbalance = abs(long_share - short_share)

            # Return-first objective on net (post-cost) compounded returns.
            objective = (
                (1.00 * cumulative_return)
                + (0.20 * sharpe)
                + (0.10 * sortino)
                + (0.10 * win_period)
                - (0.12 * abs(max_dd))
                - (float(exposure_penalty) * abs(net_exposure))
                - (float(imbalance_penalty) * imbalance)
                - (0.01 * np.log1p(max(0, trades)))
            )
            if not np.isfinite(objective):
                continue
            if objective > best.objective:
                best = ThresholdOptimizationResult(
                    long_threshold=float(long_t),
                    short_threshold=float(short_t),
                    objective=float(objective),
                    trades=trades,
                )

    if best.objective == float("-inf"):
        return ThresholdOptimizationResult(long_threshold=0.55, short_threshold=0.45, objective=0.0, trades=0)
    return best

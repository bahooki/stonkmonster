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
    long_grid: np.ndarray | None = None,
    short_grid: np.ndarray | None = None,
    min_trades: int = 20,
) -> ThresholdOptimizationResult:
    probs = _clip_probs(np.asarray(prob_up))
    future = pd.Series(future_return).astype(float).to_numpy()

    long_values = long_grid if long_grid is not None else np.arange(0.52, 0.81, 0.02)
    short_values = short_grid if short_grid is not None else np.arange(0.20, 0.49, 0.02)

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
            strat = positions[trade_mask] * future[trade_mask]
            objective = float(np.nanmean(strat)) if len(strat) else float("-inf")
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

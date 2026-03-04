from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import stonkmodel.backtest.walk_forward as walk_forward


class _AlwaysLongModel:
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "_AlwaysLongModel":
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), 0.9, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class _Selection:
    selected_features: list[str]


def test_walk_forward_clamps_step_and_avoids_overlapping_duplicate_trades(monkeypatch) -> None:
    monkeypatch.setattr(walk_forward, "build_stacking_classifier", lambda **kwargs: _AlwaysLongModel())
    monkeypatch.setattr(
        walk_forward,
        "select_features",
        lambda x_train, y_train, config=None: _Selection(selected_features=["feat"]),
    )

    dt = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    close = np.linspace(100.0, 129.0, num=30)
    frame = pd.DataFrame(
        {
            "symbol": "AAA",
            "datetime": dt,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
            "pattern": "doji",
            "pattern_doji": 1,
            "future_direction": [i % 2 for i in range(len(dt))],
            "future_return": 0.01,
            "future_excess_return": 0.0,
            "feat": np.linspace(0.0, 1.0, num=len(dt)),
        }
    )

    out = walk_forward.run_walk_forward_retraining_backtests(
        dataset=frame,
        interval="1d",
        horizon_bars=1,
        train_window_days=10,
        test_window_days=6,
        step_days=2,  # intentionally overlapping request; implementation should clamp to 6
        min_pattern_rows=5,
        include_patterns={"doji"},
        fee_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
        latency_bars=0,
        embargo_bars=0,
        fast_mode=True,
        include_portfolio=False,
    )

    assert not out.empty
    row = out.iloc[0]
    assert int(row["windows_used"]) == 4
    # One symbol with label-end purging inside each test window.
    assert int(row["trades"]) == 16
    assert float(row["win_rate"]) == 1.0
    assert float(row["win_rate_trade"]) == 1.0
    assert float(row["win_rate_period"]) == 1.0

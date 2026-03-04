from __future__ import annotations

import numpy as np
import pandas as pd

from stonkmodel.models.calibration import optimize_thresholds_from_validation


def test_threshold_optimization_enforces_two_sided_activity() -> None:
    probs = np.array([0.90, 0.85, 0.88, 0.92, 0.89, 0.87], dtype=float)
    future = pd.Series([0.02, 0.01, 0.03, 0.02, 0.01, 0.02], dtype=float)
    out = optimize_thresholds_from_validation(
        prob_up=probs,
        future_return=future,
        min_trades=3,
        min_trades_per_side=2,
    )
    # With no low-probability points, valid two-sided thresholds should not be found.
    assert out.long_threshold == 0.55
    assert out.short_threshold == 0.45
    assert out.trades == 0


def test_threshold_optimization_prefers_risk_adjusted_signal() -> None:
    probs = np.array(
        [
            0.88,
            0.84,
            0.81,
            0.76,
            0.23,
            0.19,
            0.14,
            0.11,
            0.52,
            0.49,
            0.55,
            0.45,
        ],
        dtype=float,
    )
    # Positive returns when long high-prob points and when short low-prob points.
    future_excess = pd.Series([0.03, 0.02, 0.02, 0.01, -0.02, -0.03, -0.02, -0.01, 0.0, 0.0, 0.0, 0.0], dtype=float)
    out = optimize_thresholds_from_validation(
        prob_up=probs,
        future_return=future_excess,
        future_excess_return=future_excess,
        min_trades=6,
        min_trades_per_side=2,
        transaction_cost_bps=1.0,
    )
    assert out.trades >= 6
    assert out.long_threshold > out.short_threshold
    assert out.long_threshold >= 0.53
    assert out.short_threshold <= 0.47


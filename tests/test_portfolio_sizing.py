from __future__ import annotations

import math

import pandas as pd

from stonkmodel.backtest.walk_forward import _build_portfolio_summary_from_signals


def test_portfolio_sizing_respects_per_symbol_cap() -> None:
    signals = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"],
                utc=True,
            ),
            "symbol": ["AAA", "AAA", "AAA"],
            "score": [0.25, 0.25, 0.25],
            "realized_return": [0.10, 0.10, 0.10],
            "volatility_20": [0.20, 0.20, 0.20],
        }
    )

    out = _build_portfolio_summary_from_signals(
        signals=signals,
        interval="1d",
        fee_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
        horizon_bars=1,
        latency_bars=0,
        top_k_per_side=1,
        max_gross_exposure=1.0,
        model_file="portfolio_test",
        min_abs_score=0.0,
        rebalance_every_n_bars=1,
        symbol_cooldown_bars=0,
        volatility_scaling=True,
        max_symbol_weight=0.20,
    )
    assert isinstance(out, dict)
    assert int(out["trades"]) == 3
    assert math.isclose(float(out["avg_trade_return"]), 0.02, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(float(out["cumulative_return"]), (1.02**3) - 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_portfolio_with_zero_gross_exposure_skips_trading_without_error() -> None:
    signals = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
            "symbol": ["AAA"],
            "score": [0.2],
            "realized_return": [0.1],
            "volatility_20": [0.2],
        }
    )
    out = _build_portfolio_summary_from_signals(
        signals=signals,
        interval="1d",
        fee_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
        horizon_bars=1,
        latency_bars=0,
        top_k_per_side=1,
        max_gross_exposure=0.0,
        model_file="portfolio_test",
    )
    assert out is None

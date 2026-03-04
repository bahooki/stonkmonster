from __future__ import annotations

from stonkmodel.api.schemas import AutoImproveRequest, BacktestRequest


def test_backtest_request_defaults_and_portfolio_fields() -> None:
    payload = BacktestRequest()
    assert int(payload.latency_bars) == 1
    assert int(payload.embargo_bars) == 1
    assert float(payload.long_threshold) == 0.65
    assert float(payload.short_threshold) == 0.35
    assert payload.portfolio_pattern_selection == "all"
    assert int(payload.portfolio_best_patterns_top_n) == 6
    assert int(payload.portfolio_min_pattern_trades) == 40
    assert float(payload.portfolio_min_pattern_win_rate_trade) == 0.55
    assert float(payload.portfolio_min_abs_score) == 0.15
    assert int(payload.portfolio_rebalance_every_n_bars) == 3
    assert int(payload.portfolio_symbol_cooldown_bars) == 5
    assert bool(payload.portfolio_volatility_scaling) is True
    assert float(payload.portfolio_max_symbol_weight) == 0.35
    assert bool(payload.include_spread_strategies) is False
    assert int(payload.spread_lookback_bars) == 63
    assert int(payload.spread_top_components) == 3
    assert float(payload.spread_min_edge) == 0.02
    assert float(payload.spread_switch_cost_bps) == 0.0
    assert bool(payload.spread_include_neutral_overlay) is True
    assert bool(payload.spread_include_regime_switch) is True
    assert float(payload.spread_target_vol_annual) == 0.0


def test_auto_improve_request_defaults() -> None:
    payload = AutoImproveRequest()
    assert payload.dataset_name == "model_dataset"
    assert int(payload.iterations) == 8
    assert int(payload.max_minutes) == 180
    assert int(payload.patience) == 3
    assert float(payload.min_significant_improvement) == 0.10
    assert int(payload.min_iteration_trades) == 40
    assert float(payload.fee_bps) == 1.0
    assert float(payload.spread_bps) == 0.5
    assert float(payload.slippage_bps) == 0.5
    assert int(payload.latency_bars) == 1
    assert int(payload.parallel_patterns) == 4
    assert bool(payload.include_spread_strategies) is True
    assert int(payload.random_seed) == 42

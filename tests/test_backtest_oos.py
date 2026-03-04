from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from stonkmodel.backtest.walk_forward import run_pattern_backtests
from stonkmodel.models.stacking import ModelMetrics, PatternModelArtifact, PatternModelIO


class _FeatureProbabilityModel:
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "_FeatureProbabilityModel":
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        feat = pd.to_numeric(x.iloc[:, 0], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        p = np.clip(feat, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def test_backtest_uses_only_post_train_days(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name=None,
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2025-01-02T00:00:00+00:00",
        test_start_datetime="2025-01-03T00:00:00+00:00",
    )
    model_io.save(artifact)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "datetime": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z"],
                utc=True,
            ),
            "split": ["test", "test", "test"],
            "pattern": ["doji", "doji", "doji"],
            "pattern_doji": [1, 1, 1],
            "future_return": [0.01, 0.02, 0.03],
            "future_direction": [1, 1, 1],
            "feat": [0.1, 0.2, 0.3],
        }
    )

    out = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
    )

    assert not out.empty
    model_row = out.loc[out["model_file"] != "portfolio_combined"].iloc[0]
    assert int(model_row["trades"]) == 1
    assert str(model_row["backtest_start_datetime"]) == "2025-01-03T00:00:00+00:00"
    assert str(model_row["backtest_end_datetime"]) == "2025-01-03T00:00:00+00:00"


def test_backtest_compounds_returns_by_timestamp_not_by_trade(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name=None,
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    model_io.save(artifact)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "datetime": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "split": ["test", "test", "test", "test"],
            "pattern": ["doji", "doji", "doji", "doji"],
            "pattern_doji": [1, 1, 1, 1],
            "future_return": [0.1, 0.1, 0.1, 0.1],
            "future_direction": [1, 1, 1, 1],
            "feat": [0.1, 0.2, 0.3, 0.4],
        }
    )

    out = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
    )

    assert not out.empty
    model_row = out.loc[out["model_file"] != "portfolio_combined"].iloc[0]
    assert int(model_row["trades"]) == 4
    assert str(model_row["backtest_start_datetime"]) == "2025-01-01T00:00:00+00:00"
    assert str(model_row["backtest_end_datetime"]) == "2025-01-02T00:00:00+00:00"
    # 10% on each of 2 days => (1.1 * 1.1) - 1 = 0.21
    assert math.isclose(float(model_row["cumulative_return"]), 0.21, rel_tol=1e-9, abs_tol=1e-9)


def test_backtest_adds_portfolio_combined_row(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name=None,
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    model_io.save(artifact)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "datetime": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "split": ["test", "test", "test", "test"],
            "pattern": ["doji", "doji", "doji", "doji"],
            "pattern_doji": [1, 1, 1, 1],
            "future_return": [0.01, -0.01, 0.01, -0.01],
            "future_direction": [1, 0, 1, 0],
            "feat": [0.1, 0.2, 0.3, 0.4],
        }
    )

    out = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
        include_portfolio=True,
        portfolio_top_k_per_side=1,
        portfolio_max_gross_exposure=1.0,
    )

    assert not out.empty
    portfolio = out.loc[out["model_file"] == "portfolio_combined"]
    assert not portfolio.empty


def test_backtest_returns_equity_curve_when_requested(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name=None,
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    model_io.save(artifact)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "datetime": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                ],
                utc=True,
            ),
            "split": ["test", "test", "test", "test"],
            "pattern": ["doji", "doji", "doji", "doji"],
            "pattern_doji": [1, 1, 1, 1],
            "future_return": [0.1, 0.1, 0.1, 0.1],
            "future_direction": [1, 1, 1, 1],
            "feat": [0.1, 0.2, 0.3, 0.4],
        }
    )

    out, curves = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
        return_curves=True,
        initial_investment=10_000.0,
    )

    assert not out.empty
    assert not curves.empty
    curve_all = curves.loc[curves["model_file"] != "portfolio_combined"].sort_values("datetime")
    if "curve_variant" in curve_all.columns:
        variants = set(curve_all["curve_variant"].dropna().astype(str).tolist())
        assert {"ml_model", "baseline_blind_pattern", "baseline_universe_eqw"}.issubset(variants)
    curve = curve_all.copy()
    if "curve_variant" in curve.columns:
        curve = curve.loc[curve["curve_variant"].astype(str) == "ml_model"].sort_values("datetime")
    assert not curve.empty
    assert float(curve["initial_investment"].iloc[0]) == 10000.0
    assert math.isclose(float(curve["equity_value"].iloc[-1]), 12100.0, rel_tol=1e-9, abs_tol=1e-9)


def test_backtest_filters_extreme_realized_returns(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name=None,
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    model_io.save(artifact)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "datetime": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "split": ["test", "test", "test"],
            "pattern": ["doji", "doji", "doji"],
            "pattern_doji": [1, 1, 1],
            "future_return": [0.0, 0.0, 0.0],
            "future_direction": [1, 1, 1],
            "feat": [0.1, 0.2, 0.3],
            "close": [10.0, 100.0, 101.0],  # 900% jump then ~1%
        }
    )

    out = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
    )

    assert not out.empty
    model_row = out.loc[out["model_file"] != "portfolio_combined"].iloc[0]
    # The extreme 900% return trade should be filtered out by sanity limits.
    assert int(model_row["trades"]) == 1


def test_backtest_reports_risk_diagnostics(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name=None,
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    model_io.save(artifact)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "datetime": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "split": ["test", "test", "test"],
            "pattern": ["doji", "doji", "doji"],
            "pattern_doji": [1, 1, 1],
            "future_return": [0.10, -0.05, 0.00],
            "future_direction": [1, 0, 1],
            "feat": [0.1, 0.2, 0.3],
        }
    )

    out = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
    )

    assert not out.empty
    row = out.loc[out["model_file"] != "portfolio_combined"].iloc[0]
    assert "max_drawdown" in out.columns
    assert "sortino" in out.columns
    assert "profit_factor" in out.columns
    assert "annualized_return" in out.columns
    assert "annualized_volatility" in out.columns
    assert math.isclose(float(row["max_drawdown"]), -0.05, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(float(row["profit_factor"]), 2.0, rel_tol=1e-9, abs_tol=1e-9)


def test_backtest_annualization_uses_elapsed_time_for_sparse_signals(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name=None,
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    model_io.save(artifact)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "datetime": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-12-31T00:00:00Z",
                ],
                utc=True,
            ),
            "split": ["test", "test"],
            "pattern": ["doji", "doji"],
            "pattern_doji": [1, 1],
            "future_return": [0.10, 0.10],
            "future_direction": [1, 1],
            "feat": [0.9, 0.9],
        }
    )

    out = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
        include_portfolio=False,
    )

    assert not out.empty
    row = out.iloc[0]
    # Sparse signals over ~1 year should not explode annualization.
    assert float(row["annualized_return"]) < 1.0
    assert float(row["annualized_return"]) >= -1.0


def test_backtest_best_pattern_portfolio_mode_selects_top_patterns(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    artifact_doji = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="test_best",
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    artifact_hammer = PatternModelArtifact(
        pattern="hammer",
        interval="1d",
        horizon_bars=1,
        model_name="test_best",
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    model_io.save(artifact_doji)
    model_io.save(artifact_hammer)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
            "datetime": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "split": ["test"] * 6,
            "pattern": ["doji", "doji", "doji", "hammer", "hammer", "hammer"],
            "pattern_doji": [1, 1, 1, 0, 0, 0],
            "pattern_hammer": [0, 0, 0, 1, 1, 1],
            "future_return": [0.0] * 6,
            "future_direction": [1, 1, 1, 1, 1, 1],
            "feat": [0.9] * 6,
            "close": [100.0, 110.0, 121.0, 100.0, 90.0, 81.0],
        }
    )

    out = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
        include_portfolio=True,
        portfolio_pattern_selection="best",
        portfolio_best_patterns_top_n=1,
        portfolio_min_pattern_trades=1,
        portfolio_min_pattern_win_rate_trade=0.0,
        portfolio_top_k_per_side=1,
        portfolio_max_gross_exposure=1.0,
    )

    portfolio = out.loc[out["model_file"] == "portfolio_best_patterns"]
    assert not portfolio.empty
    row = portfolio.iloc[0]
    assert int(row["portfolio_pattern_count"]) == 1
    assert str(row["portfolio_patterns_used"]) == "doji"
    assert float(row["cumulative_return"]) > 0.0


def test_backtest_portfolio_edge_and_rebalance_controls_reduce_trades(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = _FeatureProbabilityModel().fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="test_turnover",
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    model_io.save(artifact)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA"] * 6,
            "datetime": pd.to_datetime(
                [
                    "2025-01-01T00:00:00Z",
                    "2025-01-02T00:00:00Z",
                    "2025-01-03T00:00:00Z",
                    "2025-01-04T00:00:00Z",
                    "2025-01-05T00:00:00Z",
                    "2025-01-06T00:00:00Z",
                ],
                utc=True,
            ),
            "split": ["test"] * 6,
            "pattern": ["doji"] * 6,
            "pattern_doji": [1] * 6,
            "future_return": [0.0] * 6,
            "future_direction": [1] * 6,
            "feat": [0.52, 0.90, 0.51, 0.92, 0.53, 0.93],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )

    out_loose = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
        include_portfolio=True,
        portfolio_pattern_selection="all",
        portfolio_top_k_per_side=1,
        portfolio_max_gross_exposure=1.0,
        portfolio_min_abs_score=0.0,
        portfolio_rebalance_every_n_bars=1,
        portfolio_symbol_cooldown_bars=0,
    )
    out_strict = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
        include_portfolio=True,
        portfolio_pattern_selection="all",
        portfolio_top_k_per_side=1,
        portfolio_max_gross_exposure=1.0,
        portfolio_min_abs_score=0.20,
        portfolio_rebalance_every_n_bars=2,
        portfolio_symbol_cooldown_bars=1,
    )

    loose_row = out_loose.loc[out_loose["model_file"] == "portfolio_combined"].iloc[0]
    strict_row = out_strict.loc[out_strict["model_file"] == "portfolio_combined"].iloc[0]
    assert int(strict_row["trades"]) < int(loose_row["trades"])

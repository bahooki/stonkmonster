from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import stonkmodel.backtest.walk_forward as walk_forward
from stonkmodel.backtest.walk_forward import run_pattern_backtests
from stonkmodel.models.stacking import ModelMetrics, PatternModelArtifact, PatternModelIO


def _constant_long_model() -> Pipeline:
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))
    return model


def _save_pattern_model(model_io: PatternModelIO, pattern: str) -> None:
    artifact = PatternModelArtifact(
        pattern=pattern,
        interval="1d",
        horizon_bars=1,
        model_name=f"spread_{pattern}",
        feature_columns=["feat"],
        model=_constant_long_model(),
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=50,
        test_rows=30,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
    )
    model_io.save(artifact)


def test_saved_model_backtest_builds_spread_rows_and_curves(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    _save_pattern_model(model_io, "doji")
    _save_pattern_model(model_io, "hammer")

    dt = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dt):
        rows.append(
            {
                "symbol": "AAA",
                "datetime": date,
                "split": "test",
                "pattern": "doji",
                "pattern_doji": 1,
                "pattern_hammer": 0,
                "future_return": 0.02,
                "future_direction": 1,
                "feat": 0.9,
                "close": 100.0 + i,
            }
        )
        rows.append(
            {
                "symbol": "BBB",
                "datetime": date,
                "split": "test",
                "pattern": "hammer",
                "pattern_doji": 0,
                "pattern_hammer": 1,
                "future_return": -0.01,
                "future_direction": 0,
                "feat": 0.9,
                "close": 100.0 - i,
            }
        )
    frame = pd.DataFrame(rows)

    out, curves = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        long_threshold=0.5,
        short_threshold=0.45,
        fee_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
        latency_bars=0,
        include_portfolio=False,
        include_spread_strategies=True,
        spread_lookback_bars=5,
        spread_top_components=1,
        spread_min_edge=0.0,
        spread_include_neutral_overlay=True,
        spread_include_regime_switch=True,
        return_curves=True,
    )

    assert not out.empty
    spread_names = set(out["model_file"].dropna().astype(str).tolist())
    assert "spread_model_vs_model" in spread_names
    assert "spread_pattern_vs_pattern" in spread_names
    assert "spread_regime_switch" in spread_names

    spread_row = out.loc[out["model_file"] == "spread_model_vs_model"].iloc[0]
    assert int(spread_row["trades"]) > 0

    curve_slice = curves.loc[curves["model_file"] == "spread_model_vs_model"].copy()
    assert not curve_slice.empty
    variants = set(curve_slice["curve_variant"].dropna().astype(str).tolist())
    assert {"ml_model", "baseline_universe_eqw"}.issubset(variants)


class _AlwaysLongModel:
    def fit(self, x: pd.DataFrame, y: pd.Series) -> "_AlwaysLongModel":
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), 0.9, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class _Selection:
    selected_features: list[str]


def test_walk_forward_backtest_builds_pattern_spread_rows(monkeypatch) -> None:
    monkeypatch.setattr(walk_forward, "build_stacking_classifier", lambda **kwargs: _AlwaysLongModel())
    monkeypatch.setattr(
        walk_forward,
        "select_features",
        lambda x_train, y_train, config=None: _Selection(selected_features=["feat"]),
    )

    dt = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dt):
        rows.append(
            {
                "symbol": "AAA",
                "datetime": date,
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1_000.0,
                "pattern": "doji",
                "pattern_doji": 1,
                "pattern_hammer": 0,
                "future_direction": i % 2,
                "future_return": 0.015,
                "future_excess_return": 0.0,
                "feat": 0.6,
            }
        )
        rows.append(
            {
                "symbol": "BBB",
                "datetime": date,
                "open": 100.0 - i,
                "high": 101.0 - i,
                "low": 99.0 - i,
                "close": 100.0 - i,
                "volume": 1_000.0,
                "pattern": "hammer",
                "pattern_doji": 0,
                "pattern_hammer": 1,
                "future_direction": (i + 1) % 2,
                "future_return": -0.01,
                "future_excess_return": 0.0,
                "feat": 0.6,
            }
        )
    frame = pd.DataFrame(rows)

    out = walk_forward.run_walk_forward_retraining_backtests(
        dataset=frame,
        interval="1d",
        horizon_bars=1,
        train_window_days=12,
        test_window_days=6,
        step_days=6,
        min_pattern_rows=8,
        include_patterns={"doji", "hammer"},
        fee_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
        latency_bars=0,
        embargo_bars=0,
        fast_mode=True,
        include_portfolio=False,
        include_spread_strategies=True,
        spread_lookback_bars=5,
        spread_top_components=1,
        spread_min_edge=0.0,
        spread_include_neutral_overlay=True,
        spread_include_regime_switch=True,
        return_curves=False,
    )

    assert not out.empty
    names = set(out["model_file"].dropna().astype(str).tolist())
    assert "spread_pattern_vs_pattern" in names
    assert "spread_regime_switch" in names

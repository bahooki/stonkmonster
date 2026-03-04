from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from stonkmodel.backtest.walk_forward import run_pattern_backtests
from stonkmodel.models.stacking import ModelMetrics, PatternModelArtifact, PatternModelIO


def test_backtest_respects_saved_meta_filter_gate(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)

    base_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    base_model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))

    meta_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=0))])
    meta_train = pd.DataFrame(
        {
            "feat": [0.0, 1.0],
            "meta_prob_up": [0.2, 0.8],
            "meta_abs_edge": [0.3, 0.3],
            "meta_side": [1.0, 1.0],
            "meta_signed_edge": [0.3, 0.3],
        }
    )
    meta_model.fit(meta_train, pd.Series([0, 1]))

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="meta_gate",
        feature_columns=["feat"],
        model=base_model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
        meta_filter={
            "enabled": True,
            "threshold": 0.99,
            "feature_columns": ["feat", "meta_prob_up", "meta_abs_edge", "meta_side", "meta_signed_edge"],
            "model": meta_model,
            "train_rows": 2,
            "positive_rate": 0.5,
        },
    )
    model_path = model_io.save(artifact)

    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "datetime": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
            "split": ["test", "test"],
            "pattern": ["doji", "doji"],
            "pattern_doji": [1, 1],
            "future_return": [0.01, 0.01],
            "future_direction": [1, 1],
            "feat": [0.9, 0.9],
        }
    )

    out = run_pattern_backtests(
        dataset=frame,
        model_io=model_io,
        interval="1d",
        horizon_bars=1,
        include_model_files={model_path.name},
        include_portfolio=False,
        fee_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        latency_bars=0,
        embargo_bars=0,
    )

    assert out.empty

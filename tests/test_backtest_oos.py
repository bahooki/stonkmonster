from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from stonkmodel.backtest.walk_forward import run_pattern_backtests
from stonkmodel.models.stacking import ModelMetrics, PatternModelArtifact, PatternModelIO


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
    assert int(out.iloc[0]["trades"]) == 1

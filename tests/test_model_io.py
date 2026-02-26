from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from stonkmodel.models.stacking import ModelMetrics, PatternModelArtifact, PatternModelIO


def test_model_io_roundtrip(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))])

    artifact = PatternModelArtifact(
        pattern="bearish_engulfing",
        interval="1d",
        horizon_bars=1,
        feature_columns=["x1", "x2"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=100,
        test_rows=20,
    )
    path = model_io.save(artifact)

    loaded = model_io.load_from_path(path)
    assert loaded["pattern"] == "bearish_engulfing"
    assert loaded["interval"] == "1d"
    assert loaded["feature_columns"] == ["x1", "x2"]


def test_feature_importance_io_roundtrip(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    frame = pd.DataFrame(
        {
            "feature": ["x1", "x2"],
            "importance_mean": [0.03, 0.01],
            "importance_std": [0.005, 0.004],
            "importance_abs": [0.03, 0.01],
            "rank": [1, 2],
        }
    )
    model_io.save_feature_importance("bearish_engulfing", "1d", 1, frame)

    loaded = model_io.load_feature_importance("bearish_engulfing", "1d", 1)
    assert not loaded.empty
    assert set(["pattern", "interval", "horizon_bars", "feature"]).issubset(loaded.columns)

    summary = model_io.get_feature_importance_summary(interval="1d", horizon_bars=1, top_n_per_pattern=1)
    assert len(summary) == 1


def test_delete_model_removes_related_files(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))])

    artifact = PatternModelArtifact(
        pattern="bearish_engulfing",
        interval="1d",
        horizon_bars=1,
        feature_columns=["x1", "x2"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=100,
        test_rows=20,
    )
    model_path = model_io.save(artifact)
    importance = pd.DataFrame(
        {
            "feature": ["x1"],
            "importance_mean": [0.01],
            "importance_std": [0.001],
            "importance_abs": [0.01],
            "rank": [1],
        }
    )
    imp_path = model_io.save_feature_importance("bearish_engulfing", "1d", 1, importance)
    assert model_path.exists()
    assert imp_path.exists()

    result = model_io.delete_model(model_path.name)
    assert result["deleted"] is True
    assert not model_path.exists()
    assert not imp_path.exists()
    assert not (tmp_path / "bearish_engulfing__1d__h1.meta.json").exists()

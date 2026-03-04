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
        model_name=None,
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
        model_name=None,
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


def test_bulk_delete_models(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))])

    artifact_a = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="bulk_a",
        feature_columns=["x1"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
    )
    artifact_b = PatternModelArtifact(
        pattern="hammer",
        interval="1d",
        horizon_bars=1,
        model_name="bulk_b",
        feature_columns=["x1"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
    )
    path_a = model_io.save(artifact_a)
    path_b = model_io.save(artifact_b)
    assert path_a.exists()
    assert path_b.exists()

    result = model_io.delete_models([path_a.name, path_b.name, "does_not_exist.joblib", path_a.name])
    assert int(result["requested_count"]) == 3
    assert int(result["deleted_count"]) == 2
    assert int(result["missing_or_invalid_count"]) == 1
    assert not path_a.exists()
    assert not path_b.exists()


def test_named_model_roundtrip_and_registry(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))])

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="My alpha run 01",
        feature_columns=["x1"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
    )
    model_path = model_io.save(artifact)
    model_io.save_feature_importance(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="My alpha run 01",
        frame=pd.DataFrame(
            {
                "feature": ["x1"],
                "importance_mean": [0.1],
                "importance_std": [0.01],
                "importance_abs": [0.1],
                "rank": [1],
            }
        ),
    )

    assert "__nmy_alpha_run_01.joblib" in model_path.name

    parsed = model_io.parse_model_filename(model_path.name)
    assert parsed["model_name"] == "my_alpha_run_01"

    registry = model_io.get_model_registry(interval="1d", pattern="doji")
    assert not registry.empty
    assert registry.iloc[0]["model_name"] == "my_alpha_run_01"

    details = model_io.get_model_details(model_file=model_path.name, top_n_importance=5)
    assert details["model_name"] == "my_alpha_run_01"


def test_save_does_not_overwrite_existing_named_model(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))])

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="repeatable",
        feature_columns=["x1"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
    )
    path_one = model_io.save(artifact)
    path_two = model_io.save(artifact)

    assert path_one.exists()
    assert path_two.exists()
    assert path_one.name != path_two.name
    assert "__v" in path_two.stem

    registry = model_io.get_model_registry(interval="1d", pattern="doji")
    assert len(registry) == 2


def test_model_details_exposes_meta_filter_summary_without_model_object(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    base_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))])
    meta_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))])

    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="meta_summary",
        feature_columns=["x1"],
        model=base_model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        meta_filter={
            "enabled": True,
            "threshold": 0.62,
            "train_rows": 123,
            "positive_rate": 0.54,
            "feature_columns": ["x1", "meta_prob_up"],
            "model": meta_model,
        },
    )
    model_path = model_io.save(artifact)

    details = model_io.get_model_details(model_file=model_path.name, top_n_importance=5)
    meta = details.get("meta_filter", {})
    assert bool(meta.get("enabled")) is True
    assert float(meta.get("threshold")) == 0.62
    assert int(meta.get("train_rows")) == 123
    assert "model" not in meta

    registry = model_io.get_model_registry(interval="1d", pattern="doji")
    assert bool(registry.iloc[0]["meta_filter_enabled"]) is True

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from stonkmodel.backtest.walk_forward import optimize_saved_model_thresholds
from stonkmodel.models.stacking import ModelMetrics, PatternModelArtifact, PatternModelIO


class _LinearProbModel:
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = pd.to_numeric(x["feat"], errors="coerce").fillna(0.5).clip(0.01, 0.99).to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])


def test_model_io_update_thresholds_persists(tmp_path: Path) -> None:
    io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="constant", constant=1))])
    model.fit(pd.DataFrame({"feat": [0.0, 1.0]}), pd.Series([0, 1]))
    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="unit",
        feature_columns=["feat"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        tuned_thresholds={"long": 0.55, "short": 0.45},
    )
    model_path = io.save(artifact)
    result = io.update_model_thresholds(
        model_file=model_path.name,
        tuned_thresholds={"long": 0.61, "short": 0.39, "objective": 1.0, "trades": 42},
        optimization_meta={"source": "test"},
    )
    assert bool(result["updated"]) is True

    payload = io.load_from_path(model_path)
    tuned = payload.get("tuned_thresholds", {})
    assert float(tuned.get("long")) == 0.61
    assert float(tuned.get("short")) == 0.39
    assert payload.get("threshold_optimization", {}).get("source") == "test"


def test_optimize_saved_model_thresholds_updates_model(tmp_path: Path) -> None:
    io = PatternModelIO(tmp_path)
    artifact = PatternModelArtifact(
        pattern="doji",
        interval="1d",
        horizon_bars=1,
        model_name="retopt",
        feature_columns=["feat"],
        model=_LinearProbModel(),
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=20,
        test_rows=20,
        train_end_datetime="2024-12-31T00:00:00+00:00",
        test_start_datetime="2025-01-01T00:00:00+00:00",
        tuned_thresholds={"long": 0.90, "short": 0.10},
    )
    model_path = io.save(artifact)

    dt = pd.date_range("2025-01-01", periods=24, freq="D", tz="UTC")
    feat = np.linspace(0.2, 0.8, num=len(dt))
    future_return = np.where(feat >= 0.55, 0.02, -0.01)
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"] * len(dt),
            "datetime": dt,
            "split": ["test"] * len(dt),
            "pattern": ["doji"] * len(dt),
            "pattern_doji": [1] * len(dt),
            "future_return": future_return,
            "future_direction": (future_return > 0).astype(int),
            "feat": feat,
        }
    )

    out = optimize_saved_model_thresholds(
        dataset=frame,
        model_io=io,
        interval="1d",
        horizon_bars=1,
        fee_bps=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        short_borrow_bps_per_day=0.0,
        latency_bars=0,
        embargo_bars=0,
        include_model_files={model_path.name},
        min_trades=5,
        persist=True,
    )
    assert not out.empty
    row = out.iloc[0]
    assert bool(row["threshold_updated"]) is True
    assert float(row["best_long_threshold"]) <= 0.90
    assert float(row["best_short_threshold"]) >= 0.10
    improvement = pd.to_numeric(pd.Series([row["improvement_cumulative_return"]]), errors="coerce").iloc[0]
    if pd.notna(improvement):
        assert float(improvement) >= 0.0

    payload = io.load_from_path(model_path)
    tuned = payload.get("tuned_thresholds", {})
    assert float(tuned.get("long", 0.0)) == float(row["best_long_threshold"])
    assert float(tuned.get("short", 0.0)) == float(row["best_short_threshold"])

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from stonkmodel.models.stacking import ModelMetrics, PatternModelArtifact, PatternModelIO
from stonkmodel.scanner.scanner import ScanConfig, SignalScanner


def test_scanner_skips_corrupt_model_files(tmp_path: Path) -> None:
    model_io = PatternModelIO(tmp_path)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="prior"))])
    model.fit(pd.DataFrame({"ret_1": [0.0, 1.0]}), pd.Series([0, 1]))
    artifact = PatternModelArtifact(
        pattern="none",
        interval="1d",
        horizon_bars=1,
        model_name="scan",
        feature_columns=["ret_1"],
        model=model,
        metrics=ModelMetrics(accuracy=0.5, precision=0.5, recall=0.5, f1=0.5, roc_auc=0.5),
        train_rows=10,
        test_rows=5,
        tuned_thresholds={"long": 0.49, "short": 0.25},
    )
    model_io.save(artifact)
    (tmp_path / "corrupt.joblib").write_text("not-a-joblib-payload", encoding="utf-8")

    frames = {
        "AAA": pd.DataFrame(
            {
                "symbol": ["AAA", "AAA"],
                "datetime": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1200.0],
            }
        )
    }

    scanner = SignalScanner(model_io)
    out = scanner.scan(
        latest_frames=frames,
        config=ScanConfig(interval="1d", horizon_bars=1, min_confidence=0.5),
    )
    assert isinstance(out, pd.DataFrame)
    assert not out.empty

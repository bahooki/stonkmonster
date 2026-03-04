from __future__ import annotations

from pathlib import Path

import pandas as pd

from stonkmodel.config import Settings
from stonkmodel.pipeline import StonkService


def _service(tmp_path: Path) -> StonkService:
    settings = Settings(
        data_dir=tmp_path / "data",
        raw_data_dir=tmp_path / "data" / "raw",
        processed_data_dir=tmp_path / "data" / "processed",
        models_dir=tmp_path / "models",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    return StonkService(settings)


def test_recommend_thresholds_uses_weighted_model_metadata(tmp_path: Path) -> None:
    service = _service(tmp_path)
    registry = pd.DataFrame(
        {
            "model_file": ["m1.joblib", "m2.joblib", "m3.joblib"],
            "pattern": ["doji", "hammer", "doji"],
            "roc_auc": [0.62, 0.58, 0.53],
            "train_rows": [1000, 800, 400],
            "tuned_long_threshold": [0.64, 0.58, 0.52],
            "tuned_short_threshold": [0.36, 0.42, 0.48],
        }
    )
    service.model_io.get_model_registry = lambda interval=None, pattern=None: registry.copy()  # type: ignore[method-assign]

    out = service.recommend_thresholds(interval="1d")
    assert out["models_considered"] == 3
    assert 0.60 <= float(out["recommended_long_threshold"]) <= 0.64
    assert 0.36 <= float(out["recommended_short_threshold"]) <= 0.40


def test_recommend_thresholds_falls_back_without_models(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.model_io.get_model_registry = lambda interval=None, pattern=None: pd.DataFrame()  # type: ignore[method-assign]
    out = service.recommend_thresholds(interval="1d")
    assert float(out["recommended_long_threshold"]) == 0.65
    assert float(out["recommended_short_threshold"]) == 0.35

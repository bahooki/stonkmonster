from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import stonkmodel.models.trainer as trainer_mod
from stonkmodel.models.stacking import PatternModelIO
from stonkmodel.models.trainer import PatternTrainer, TrainConfig


class _Selection:
    def __init__(self, selected_features: list[str]) -> None:
        self.selected_features = selected_features
        self.raw_feature_count = len(selected_features)
        self.prefiltered_feature_count = len(selected_features)
        self.selected_feature_count = len(selected_features)
        self.dropped_missing = 0
        self.dropped_constant = 0
        self.dropped_correlated = 0
        self.dropped_low_importance = 0


def _dummy_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="prior")),
        ]
    )


def test_train_one_purges_train_rows_touching_test_window(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(trainer_mod, "build_stacking_classifier", lambda **kwargs: _dummy_pipeline())
    monkeypatch.setattr(trainer_mod, "select_features", lambda x_train, y_train, config=None: _Selection(["feat"]))
    monkeypatch.setattr(trainer_mod, "timeseries_cv_score", lambda *args, **kwargs: float("nan"))

    rows = 8
    dt = pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"] * rows,
            "datetime": dt,
            "split": ["train"] * 5 + ["test"] * 3,
            "pattern": ["doji"] * rows,
            "pattern_doji": [1] * rows,
            "future_direction": [0, 1, 0, 1, 0, 1, 0, 1],
            "future_return": np.linspace(-0.01, 0.01, rows),
            "future_excess_return": np.linspace(-0.005, 0.005, rows),
            "feat": np.linspace(0.0, 1.0, rows),
        }
    )

    trainer = PatternTrainer(PatternModelIO(tmp_path))
    cfg = TrainConfig(
        interval="1d",
        horizon_bars=1,
        min_pattern_rows=4,
        enable_permutation_importance=False,
        enable_timeseries_cv=False,
        purge_overlap=True,
    )

    out = trainer.train_one("doji", frame, cfg)
    assert out is not None
    # Last train row's label endpoint touches test start and should be purged.
    assert int(out["purged_train_rows"]) == 1
    assert int(out["train_rows"]) == 4

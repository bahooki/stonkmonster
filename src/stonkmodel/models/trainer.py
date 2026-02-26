from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stonkmodel.features.dataset import get_pattern_datasets, split_xy
from stonkmodel.models.feature_selection import FeatureSelectionConfig, select_features
from stonkmodel.models.importance import ImportanceConfig, compute_permutation_feature_importance
from stonkmodel.models.stacking import (
    PatternModelArtifact,
    PatternModelIO,
    build_stacking_classifier,
    evaluate_binary,
    timeseries_cv_score,
)


@dataclass
class TrainConfig:
    interval: str
    horizon_bars: int
    min_pattern_rows: int = 100
    max_feature_missing_rate: float = 0.65
    max_feature_correlation: float = 0.995
    min_selected_features: int = 25
    max_selected_features: int = 260


class PatternTrainer:
    def __init__(self, model_io: PatternModelIO) -> None:
        self.model_io = model_io

    def train_all(self, dataset: pd.DataFrame, config: TrainConfig) -> pd.DataFrame:
        pattern_frames = get_pattern_datasets(dataset, min_rows=config.min_pattern_rows)
        rows: list[dict[str, float | int | str]] = []

        for pattern_name, pattern_data in pattern_frames.items():
            result = self.train_one(pattern_name, pattern_data, config)
            if result is None:
                continue
            rows.append(result)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "pattern",
                    "train_rows",
                    "test_rows",
                    "accuracy",
                    "f1",
                    "roc_auc",
                    "cv_roc_auc",
                    "raw_features",
                    "prefilter_features",
                    "selected_features",
                    "dropped_missing",
                    "dropped_constant",
                    "dropped_correlated",
                    "dropped_low_importance",
                    "top_features",
                    "importance_path",
                ]
            )

        return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    def train_one(self, pattern_name: str, data: pd.DataFrame, config: TrainConfig) -> dict[str, float | int | str] | None:
        if len(data) < config.min_pattern_rows:
            return None

        train = data.loc[data["split"] == "train"].copy()
        test = data.loc[data["split"] == "test"].copy()
        if train.empty or test.empty:
            cutoff = int(len(data) * 0.8)
            train = data.iloc[:cutoff].copy()
            test = data.iloc[cutoff:].copy()

        if train.empty or test.empty:
            return None

        x_train_raw, y_train, feature_cols_raw = split_xy(train)
        selection = select_features(
            x_train=x_train_raw,
            y_train=y_train,
            config=FeatureSelectionConfig(
                max_missing_rate=config.max_feature_missing_rate,
                corr_threshold=config.max_feature_correlation,
                min_features=config.min_selected_features,
                max_features=config.max_selected_features,
                random_state=42,
            ),
        )
        feature_cols = selection.selected_features
        if not feature_cols:
            return None

        x_train = x_train_raw[feature_cols].copy()
        x_test = test.reindex(columns=feature_cols).copy()
        x_test = x_test.fillna(x_train.median(numeric_only=True)).fillna(0.0)
        y_test = test["future_direction"].astype(int)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            return None

        model = build_stacking_classifier(random_state=42)
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]
        metrics = evaluate_binary(y_test, pred, prob)

        artifact = PatternModelArtifact(
            pattern=pattern_name,
            interval=config.interval,
            horizon_bars=config.horizon_bars,
            feature_columns=feature_cols,
            model=model,
            metrics=metrics,
            train_rows=len(train),
            test_rows=len(test),
            feature_selection={
                "raw_feature_count": int(selection.raw_feature_count),
                "prefilter_feature_count": int(selection.prefiltered_feature_count),
                "selected_feature_count": int(selection.selected_feature_count),
                "dropped_missing": int(selection.dropped_missing),
                "dropped_constant": int(selection.dropped_constant),
                "dropped_correlated": int(selection.dropped_correlated),
                "dropped_low_importance": int(selection.dropped_low_importance),
            },
        )
        self.model_io.save(artifact)

        importance = compute_permutation_feature_importance(
            model=model,
            x=x_test,
            y=y_test,
            config=ImportanceConfig(n_repeats=6, max_samples=5000, random_state=42, scoring="roc_auc"),
        )
        importance_path = ""
        top_features = ""
        if not importance.empty:
            importance_path = str(
                self.model_io.save_feature_importance(
                    pattern=pattern_name,
                    interval=config.interval,
                    horizon_bars=config.horizon_bars,
                    frame=importance,
                )
            )
            top_features = ", ".join(importance["feature"].head(8).tolist())

        cv_auc = timeseries_cv_score(x_train, y_train, n_splits=4)

        return {
            "pattern": pattern_name,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "cv_roc_auc": float(cv_auc) if not np.isnan(cv_auc) else np.nan,
            "raw_features": int(selection.raw_feature_count),
            "prefilter_features": int(selection.prefiltered_feature_count),
            "selected_features": int(selection.selected_feature_count),
            "dropped_missing": int(selection.dropped_missing),
            "dropped_constant": int(selection.dropped_constant),
            "dropped_correlated": int(selection.dropped_correlated),
            "dropped_low_importance": int(selection.dropped_low_importance),
            "top_features": top_features,
            "importance_path": importance_path,
        }

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from stonkmodel.models.importance import ImportanceConfig, compute_permutation_feature_importance


@dataclass
class FeatureSelectionConfig:
    max_missing_rate: float = 0.65
    corr_threshold: float = 0.995
    min_features: int = 25
    max_features: int = 260
    min_train_rows_for_importance: int = 350
    min_validation_rows: int = 80
    validation_fraction: float = 0.2
    min_importance: float = 0.0
    sample_for_correlation: int = 8000
    random_state: int = 42


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    raw_feature_count: int
    prefiltered_feature_count: int
    selected_feature_count: int
    dropped_missing: int
    dropped_constant: int
    dropped_correlated: int
    dropped_low_importance: int
    importance: pd.DataFrame


def _drop_high_correlation(
    x: pd.DataFrame,
    threshold: float,
    sample_for_correlation: int,
    random_state: int,
) -> tuple[list[str], int]:
    if x.shape[1] <= 1:
        return list(x.columns), 0

    sample = x
    if len(sample) > sample_for_correlation:
        sample = sample.sample(n=sample_for_correlation, random_state=random_state)

    sample = sample.fillna(sample.median(numeric_only=True)).fillna(0.0)
    corr = sample.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if bool((upper[col] > threshold).any())]
    kept = [col for col in x.columns if col not in set(to_drop)]
    return kept, len(to_drop)


def _build_selector_model(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=350,
                    max_depth=14,
                    min_samples_leaf=10,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _ordered_intersection(ordered_columns: list[str], selected: list[str]) -> list[str]:
    selected_set = set(selected)
    return [col for col in ordered_columns if col in selected_set]


def select_features(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: FeatureSelectionConfig | None = None,
) -> FeatureSelectionResult:
    cfg = config or FeatureSelectionConfig()
    if x_train.empty:
        return FeatureSelectionResult(
            selected_features=[],
            raw_feature_count=0,
            prefiltered_feature_count=0,
            selected_feature_count=0,
            dropped_missing=0,
            dropped_constant=0,
            dropped_correlated=0,
            dropped_low_importance=0,
            importance=pd.DataFrame(),
        )

    raw_cols = list(x_train.columns)
    missing_rate = x_train.isna().mean()
    kept_missing = [c for c in raw_cols if float(missing_rate.get(c, 1.0)) <= cfg.max_missing_rate]
    dropped_missing = len(raw_cols) - len(kept_missing)

    x_prefilter = x_train[kept_missing].copy()
    nunique = x_prefilter.nunique(dropna=True)
    kept_nonconstant = [c for c in kept_missing if int(nunique.get(c, 0)) > 1]
    dropped_constant = len(kept_missing) - len(kept_nonconstant)

    x_prefilter = x_train[kept_nonconstant].copy()
    kept_uncorrelated, dropped_correlated = _drop_high_correlation(
        x_prefilter,
        threshold=cfg.corr_threshold,
        sample_for_correlation=cfg.sample_for_correlation,
        random_state=cfg.random_state,
    )
    x_prefilter = x_prefilter[kept_uncorrelated].copy()

    if x_prefilter.empty:
        return FeatureSelectionResult(
            selected_features=[],
            raw_feature_count=len(raw_cols),
            prefiltered_feature_count=0,
            selected_feature_count=0,
            dropped_missing=dropped_missing,
            dropped_constant=dropped_constant,
            dropped_correlated=dropped_correlated,
            dropped_low_importance=0,
            importance=pd.DataFrame(),
        )

    importance = pd.DataFrame()
    selected = list(x_prefilter.columns)

    can_score_importance = (
        len(x_prefilter) >= cfg.min_train_rows_for_importance
        and len(x_prefilter.columns) > cfg.min_features
        and y_train.nunique() >= 2
    )
    if can_score_importance:
        split_idx = int(round(len(x_prefilter) * (1 - cfg.validation_fraction)))
        split_idx = max(1, min(split_idx, len(x_prefilter) - 1))
        x_fit = x_prefilter.iloc[:split_idx].copy()
        y_fit = y_train.iloc[:split_idx].astype(int)
        x_val = x_prefilter.iloc[split_idx:].copy()
        y_val = y_train.iloc[split_idx:].astype(int)

        if len(x_val) >= cfg.min_validation_rows and y_fit.nunique() >= 2 and y_val.nunique() >= 2:
            selector_model = _build_selector_model(random_state=cfg.random_state)
            selector_model.fit(x_fit, y_fit)
            importance = compute_permutation_feature_importance(
                model=selector_model,
                x=x_val,
                y=y_val,
                config=ImportanceConfig(
                    n_repeats=4,
                    max_samples=3000,
                    random_state=cfg.random_state,
                    scoring="roc_auc",
                ),
            )
            if not importance.empty:
                positive = importance.loc[importance["importance_mean"] > cfg.min_importance, "feature"].tolist()
                if len(positive) >= cfg.min_features:
                    selected = positive
                else:
                    top_n = max(cfg.min_features, min(cfg.max_features, len(importance)))
                    selected = importance["feature"].head(top_n).tolist()

    if cfg.max_features > 0 and len(selected) > cfg.max_features:
        selected = selected[: cfg.max_features]

    if len(selected) < cfg.min_features:
        floor = min(len(x_prefilter.columns), cfg.min_features)
        selected = list(x_prefilter.columns[:floor])

    selected = _ordered_intersection(list(x_prefilter.columns), selected)
    dropped_low_importance = max(0, len(x_prefilter.columns) - len(selected))

    return FeatureSelectionResult(
        selected_features=selected,
        raw_feature_count=len(raw_cols),
        prefiltered_feature_count=len(x_prefilter.columns),
        selected_feature_count=len(selected),
        dropped_missing=dropped_missing,
        dropped_constant=dropped_constant,
        dropped_correlated=dropped_correlated,
        dropped_low_importance=dropped_low_importance,
        importance=importance,
    )

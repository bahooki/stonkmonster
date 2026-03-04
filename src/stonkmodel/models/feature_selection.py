from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

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
    importance_cum_share: float = 0.9
    sample_for_correlation: int = 8000
    random_state: int = 42
    stability_n_splits: int = 4
    stability_min_support: float = 0.55
    stability_top_feature_fraction: float = 0.35
    stability_min_contexts: int = 2


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
    dropped_unstable: int
    stability_contexts: int
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


def _top_features_by_importance(
    importance_values: pd.Series,
    top_n: int,
) -> list[str]:
    work = pd.to_numeric(importance_values, errors="coerce").fillna(0.0)
    if work.empty or top_n <= 0:
        return []
    top_n = min(int(top_n), len(work))
    ranked = work.sort_values(ascending=False)
    positive = ranked.loc[ranked > 0].head(top_n)
    if not positive.empty:
        return positive.index.tolist()
    return ranked.head(top_n).index.tolist()


def _stability_support_scores(
    x: pd.DataFrame,
    y: pd.Series,
    cfg: FeatureSelectionConfig,
) -> tuple[pd.Series, int]:
    if x.empty or len(x.columns) == 0 or len(y) < max(120, cfg.min_validation_rows * 2):
        return pd.Series(dtype=float), 0
    if y.nunique() < 2:
        return pd.Series(dtype=float), 0

    columns = list(x.columns)
    top_n = max(
        cfg.min_features,
        min(cfg.max_features, int(round(len(columns) * float(cfg.stability_top_feature_fraction)))),
    )
    support_counts = pd.Series(0.0, index=columns, dtype=float)
    contexts = 0

    n_splits = max(2, int(cfg.stability_n_splits))
    if len(x) >= (n_splits + 1) * max(25, cfg.min_validation_rows // 2):
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for split_idx, (train_idx, val_idx) in enumerate(splitter.split(x)):
            if len(train_idx) < max(100, cfg.min_validation_rows) or len(val_idx) < max(20, cfg.min_validation_rows // 2):
                continue
            y_train = y.iloc[train_idx].astype(int)
            if y_train.nunique() < 2:
                continue
            model = _build_selector_model(random_state=cfg.random_state + split_idx)
            x_train = x.iloc[train_idx].fillna(x.iloc[train_idx].median(numeric_only=True)).fillna(0.0)
            try:
                model.fit(x_train, y_train)
                feature_imp = pd.Series(
                    model.named_steps["model"].feature_importances_,  # type: ignore[index]
                    index=columns,
                    dtype=float,
                )
            except Exception:
                continue
            selected_fold = _top_features_by_importance(feature_imp, top_n=top_n)
            if not selected_fold:
                continue
            support_counts.loc[selected_fold] += 1.0
            contexts += 1

    regime_col = next((c for c in ("market_volatility_20", "volatility_20") if c in x.columns), None)
    if regime_col is not None:
        regime_series = pd.to_numeric(x[regime_col], errors="coerce")
        regime_median = float(regime_series.median()) if regime_series.notna().any() else np.nan
        if np.isfinite(regime_median):
            for regime_offset, mask in enumerate([regime_series <= regime_median, regime_series > regime_median], start=100):
                idx = x.index[mask.fillna(False)]
                if len(idx) < max(100, cfg.min_validation_rows * 2):
                    continue
                x_reg = x.loc[idx].fillna(x.loc[idx].median(numeric_only=True)).fillna(0.0)
                y_reg = y.loc[idx].astype(int)
                if y_reg.nunique() < 2:
                    continue
                model = _build_selector_model(random_state=cfg.random_state + regime_offset)
                try:
                    model.fit(x_reg, y_reg)
                    feature_imp = pd.Series(
                        model.named_steps["model"].feature_importances_,  # type: ignore[index]
                        index=columns,
                        dtype=float,
                    )
                except Exception:
                    continue
                selected_regime = _top_features_by_importance(feature_imp, top_n=top_n)
                if not selected_regime:
                    continue
                support_counts.loc[selected_regime] += 1.0
                contexts += 1

    if contexts <= 0:
        return pd.Series(dtype=float), 0
    return support_counts / float(contexts), contexts


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
            dropped_unstable=0,
            stability_contexts=0,
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
            dropped_unstable=0,
            stability_contexts=0,
            importance=pd.DataFrame(),
        )

    importance = pd.DataFrame()
    selected = list(x_prefilter.columns)
    stability_contexts = 0
    dropped_unstable = 0

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
                imp = pd.to_numeric(importance["importance_mean"], errors="coerce").fillna(0.0).clip(lower=0.0)
                importance_work = importance.copy()
                importance_work["_imp"] = imp
                positive = importance_work.loc[importance_work["_imp"] > max(0.0, float(cfg.min_importance)), "feature"].tolist()
                if positive:
                    pos_work = importance_work.loc[importance_work["feature"].isin(set(positive))].copy()
                    total_imp = float(pos_work["_imp"].sum())
                    if total_imp > 0:
                        pos_work["_cum_share"] = (pos_work["_imp"] / total_imp).cumsum()
                        keep = pos_work.loc[pos_work["_cum_share"] <= float(cfg.importance_cum_share), "feature"].tolist()
                    else:
                        keep = []
                    floor_n = min(len(pos_work), max(1, int(cfg.min_features)))
                    if len(keep) < floor_n:
                        keep = pos_work["feature"].head(floor_n).tolist()
                    selected = keep
                else:
                    top_n = max(cfg.min_features, min(cfg.max_features, len(importance)))
                    selected = importance["feature"].head(top_n).tolist()

    support_rate, stability_contexts = _stability_support_scores(x_prefilter, y_train, cfg)
    if not support_rate.empty and int(stability_contexts) >= int(cfg.stability_min_contexts):
        stable_features = support_rate.loc[support_rate >= float(cfg.stability_min_support)].index.tolist()
        if stable_features:
            before = len(selected)
            selected = [c for c in selected if c in set(stable_features)]
            dropped_unstable = max(0, before - len(selected))

            if len(selected) < cfg.min_features:
                # Backfill using most stable features, then importance order.
                rank_stable = support_rate.sort_values(ascending=False).index.tolist()
                rank_importance = (
                    importance["feature"].tolist()
                    if not importance.empty and "feature" in importance.columns
                    else list(x_prefilter.columns)
                )
                for col in [*rank_stable, *rank_importance]:
                    if col in selected or col not in x_prefilter.columns:
                        continue
                    selected.append(col)
                    if len(selected) >= cfg.min_features:
                        break

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
        dropped_unstable=dropped_unstable,
        stability_contexts=int(stability_contexts),
        importance=importance,
    )

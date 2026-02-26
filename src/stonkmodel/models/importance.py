from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


@dataclass
class ImportanceConfig:
    n_repeats: int = 6
    max_samples: int = 5000
    random_state: int = 42
    scoring: str = "roc_auc"


def _stratified_sample(x: pd.DataFrame, y: pd.Series, max_samples: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    if len(x) <= max_samples:
        return x, y

    frame = x.copy()
    frame["__target__"] = y.values

    sampled_groups: list[pd.DataFrame] = []
    for _, group in frame.groupby("__target__", sort=False):
        frac = len(group) / len(frame)
        n = max(1, int(round(max_samples * frac)))
        sampled_groups.append(group.sample(n=min(n, len(group)), random_state=random_state))

    sampled = pd.concat(sampled_groups, ignore_index=True)
    # Safety cap in case rounding pushes over the limit.
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=random_state)

    y_sample = sampled["__target__"].astype(int)
    x_sample = sampled.drop(columns=["__target__"])
    return x_sample, y_sample


def compute_permutation_feature_importance(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    config: ImportanceConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ImportanceConfig()
    if x.empty or y.empty or y.nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std", "importance_abs", "rank"])

    x_eval, y_eval = _stratified_sample(x, y, max_samples=cfg.max_samples, random_state=cfg.random_state)

    scoring = cfg.scoring if y_eval.nunique() > 1 else "accuracy"
    try:
        result = permutation_importance(
            estimator=model,
            X=x_eval,
            y=y_eval,
            scoring=scoring,
            n_repeats=cfg.n_repeats,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    except Exception:
        # Fallback to a simpler metric if ROC AUC fails due edge cases.
        result = permutation_importance(
            estimator=model,
            X=x_eval,
            y=y_eval,
            scoring="accuracy",
            n_repeats=max(3, cfg.n_repeats // 2),
            random_state=cfg.random_state,
            n_jobs=-1,
        )

    out = pd.DataFrame(
        {
            "feature": x_eval.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    out["importance_abs"] = out["importance_mean"].abs()
    out = out.sort_values("importance_abs", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out

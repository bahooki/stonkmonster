from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline


def _maybe_lightgbm(random_state: int):
    try:
        from lightgbm import LGBMClassifier

        return (
            "lgbm",
            LGBMClassifier(
                n_estimators=450,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
            ),
        )
    except Exception:
        return None


def _maybe_xgboost(random_state: int):
    try:
        from xgboost import XGBClassifier

        return (
            "xgb",
            XGBClassifier(
                n_estimators=350,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=1,
            ),
        )
    except Exception:
        return None


def _maybe_catboost(random_state: int):
    try:
        from catboost import CatBoostClassifier

        return (
            "catboost",
            CatBoostClassifier(
                depth=6,
                learning_rate=0.04,
                iterations=400,
                loss_function="Logloss",
                random_seed=random_state,
                verbose=False,
            ),
        )
    except Exception:
        return None


def build_stacking_classifier(random_state: int = 42) -> Pipeline:
    base_estimators: list[tuple[str, Any]] = [
        (
            "hgb",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_depth=8,
                min_samples_leaf=50,
                random_state=random_state,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "et",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=15,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]

    for optional_builder in (_maybe_lightgbm, _maybe_xgboost, _maybe_catboost):
        built = optional_builder(random_state)
        if built is not None:
            base_estimators.append(built)

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=500),
        passthrough=True,
        n_jobs=-1,
        cv=3,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", stack),
        ]
    )
    return pipeline


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


@dataclass
class PatternModelArtifact:
    pattern: str
    interval: str
    horizon_bars: int
    feature_columns: list[str]
    model: Pipeline
    metrics: ModelMetrics
    train_rows: int
    test_rows: int
    feature_selection: dict[str, Any] | None = None
    train_end_datetime: str | None = None
    test_start_datetime: str | None = None
    probability_calibration: dict[str, Any] | None = None
    tuned_thresholds: dict[str, Any] | None = None


class PatternModelIO:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, pattern: str, interval: str, horizon_bars: int) -> Path:
        safe = f"{pattern}__{interval}__h{horizon_bars}.joblib".replace("/", "_")
        return self.models_dir / safe

    def _importance_path(self, pattern: str, interval: str, horizon_bars: int) -> Path:
        safe = f"{pattern}__{interval}__h{horizon_bars}.importance.parquet".replace("/", "_")
        return self.models_dir / safe

    def _meta_path(self, pattern: str, interval: str, horizon_bars: int) -> Path:
        safe = f"{pattern}__{interval}__h{horizon_bars}.meta.json".replace("/", "_")
        return self.models_dir / safe

    def save(self, artifact: PatternModelArtifact) -> Path:
        path = self._path(artifact.pattern, artifact.interval, artifact.horizon_bars)
        payload = {
            "pattern": artifact.pattern,
            "interval": artifact.interval,
            "horizon_bars": artifact.horizon_bars,
            "feature_columns": artifact.feature_columns,
            "metrics": artifact.metrics.to_dict(),
            "train_rows": artifact.train_rows,
            "test_rows": artifact.test_rows,
            "feature_selection": artifact.feature_selection or {},
            "train_end_datetime": artifact.train_end_datetime,
            "test_start_datetime": artifact.test_start_datetime,
            "probability_calibration": artifact.probability_calibration or {},
            "tuned_thresholds": artifact.tuned_thresholds or {},
            "model": artifact.model,
        }
        joblib.dump(payload, path)

        meta = {
            "pattern": artifact.pattern,
            "interval": artifact.interval,
            "horizon_bars": artifact.horizon_bars,
            "metrics": artifact.metrics.to_dict(),
            "train_rows": artifact.train_rows,
            "test_rows": artifact.test_rows,
            "feature_count": len(artifact.feature_columns),
            "feature_selection": artifact.feature_selection or {},
            "train_end_datetime": artifact.train_end_datetime,
            "test_start_datetime": artifact.test_start_datetime,
            "probability_calibration": artifact.probability_calibration or {},
            "tuned_thresholds": artifact.tuned_thresholds or {},
        }
        self._meta_path(artifact.pattern, artifact.interval, artifact.horizon_bars).write_text(
            json.dumps(meta),
            encoding="utf-8",
        )
        return path

    def load(self, pattern: str, interval: str, horizon_bars: int) -> dict[str, Any]:
        path = self._path(pattern, interval, horizon_bars)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found for pattern={pattern} interval={interval} horizon={horizon_bars}")
        return joblib.load(path)

    def load_from_path(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        return joblib.load(path)

    def list_models(self) -> list[Path]:
        return sorted(self.models_dir.glob("*.joblib"))

    def delete_model(self, model_file: str | Path) -> dict[str, object]:
        safe_file = Path(model_file).name
        model_path = self.models_dir / safe_file
        if model_path.suffix != ".joblib":
            return {"deleted": False, "removed_files": []}
        if not model_path.exists():
            return {"deleted": False, "removed_files": []}

        removed_files: list[str] = []
        model_path.unlink()
        removed_files.append(str(model_path))

        parsed = self.parse_model_filename(model_path)
        pattern = parsed.get("pattern")
        interval = parsed.get("interval")
        horizon = parsed.get("horizon_bars")
        if pattern and interval and horizon is not None:
            for extra_path in (
                self._meta_path(str(pattern), str(interval), int(horizon)),
                self._importance_path(str(pattern), str(interval), int(horizon)),
            ):
                if extra_path.exists():
                    extra_path.unlink()
                    removed_files.append(str(extra_path))

        return {"deleted": True, "removed_files": removed_files}

    def save_feature_importance(
        self,
        pattern: str,
        interval: str,
        horizon_bars: int,
        frame: pd.DataFrame,
    ) -> Path:
        path = self._importance_path(pattern, interval, horizon_bars)
        out = frame.copy()
        out["pattern"] = pattern
        out["interval"] = interval
        out["horizon_bars"] = int(horizon_bars)
        out.to_parquet(path, index=False)
        return path

    def load_feature_importance(self, pattern: str, interval: str, horizon_bars: int) -> pd.DataFrame:
        path = self._importance_path(pattern, interval, horizon_bars)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def list_feature_importance_files(self) -> list[Path]:
        return sorted(self.models_dir.glob("*.importance.parquet"))

    def parse_model_filename(self, model_file: str | Path) -> dict[str, Any]:
        stem = Path(model_file).stem
        parts = stem.split("__")
        if len(parts) < 3:
            return {"pattern": stem, "interval": None, "horizon_bars": None}
        pattern, interval, h_token = parts[0], parts[1], parts[2]
        horizon = None
        if h_token.startswith("h"):
            try:
                horizon = int(h_token[1:])
            except Exception:
                horizon = None
        return {
            "pattern": pattern,
            "interval": interval,
            "horizon_bars": horizon,
        }

    def get_feature_importance_summary(
        self,
        interval: str | None = None,
        horizon_bars: int | None = None,
        top_n_per_pattern: int = 20,
    ) -> pd.DataFrame:
        files = self.list_feature_importance_files()
        if not files:
            return pd.DataFrame(
                columns=[
                    "pattern",
                    "interval",
                    "horizon_bars",
                    "feature",
                    "importance_mean",
                    "importance_std",
                    "importance_abs",
                    "rank",
                ]
            )

        frames: list[pd.DataFrame] = []
        for path in files:
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            if frame.empty:
                continue
            if interval is not None and "interval" in frame.columns:
                frame = frame.loc[frame["interval"] == interval]
            if horizon_bars is not None and "horizon_bars" in frame.columns:
                frame = frame.loc[frame["horizon_bars"] == int(horizon_bars)]
            if frame.empty:
                continue
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        if "rank" not in merged.columns:
            merged = merged.sort_values("importance_abs", ascending=False).reset_index(drop=True)
            merged["rank"] = merged.groupby(["pattern", "interval", "horizon_bars"]).cumcount() + 1

        merged = merged.loc[merged["rank"] <= int(top_n_per_pattern)]
        return merged.sort_values(["pattern", "rank"]).reset_index(drop=True)

    def get_model_registry(self, interval: str | None = None, pattern: str | None = None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for path in self.list_models():
            meta = self.parse_model_filename(path)
            if interval is not None and meta.get("interval") != interval:
                continue
            if pattern is not None and meta.get("pattern") != pattern:
                continue

            stat = path.stat()
            row: dict[str, Any] = {
                "model_file": path.name,
                "model_path": str(path),
                "pattern": meta.get("pattern"),
                "interval": meta.get("interval"),
                "horizon_bars": meta.get("horizon_bars"),
                "size_kb": round(stat.st_size / 1024, 2),
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "accuracy": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "f1": np.nan,
                "roc_auc": np.nan,
                "train_rows": np.nan,
                "test_rows": np.nan,
                "feature_count": np.nan,
                "raw_feature_count": np.nan,
                "train_end_datetime": None,
                "test_start_datetime": None,
                "tuned_long_threshold": np.nan,
                "tuned_short_threshold": np.nan,
                "importance_file": None,
            }
            meta_loaded = False
            if meta.get("interval") is not None and meta.get("horizon_bars") is not None:
                meta_path = self._meta_path(str(meta.get("pattern")), str(meta.get("interval")), int(meta.get("horizon_bars")))
                if meta_path.exists():
                    try:
                        payload_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        metrics = payload_meta.get("metrics", {})
                        row["accuracy"] = metrics.get("accuracy", np.nan)
                        row["precision"] = metrics.get("precision", np.nan)
                        row["recall"] = metrics.get("recall", np.nan)
                        row["f1"] = metrics.get("f1", np.nan)
                        row["roc_auc"] = metrics.get("roc_auc", np.nan)
                        row["train_rows"] = payload_meta.get("train_rows", np.nan)
                        row["test_rows"] = payload_meta.get("test_rows", np.nan)
                        row["feature_count"] = payload_meta.get("feature_count", np.nan)
                        fs = payload_meta.get("feature_selection", {}) or {}
                        row["raw_feature_count"] = fs.get("raw_feature_count", np.nan)
                        row["train_end_datetime"] = payload_meta.get("train_end_datetime")
                        row["test_start_datetime"] = payload_meta.get("test_start_datetime")
                        tuned = payload_meta.get("tuned_thresholds", {}) or {}
                        row["tuned_long_threshold"] = tuned.get("long", np.nan)
                        row["tuned_short_threshold"] = tuned.get("short", np.nan)
                        meta_loaded = True
                    except Exception:
                        meta_loaded = False

            if not meta_loaded:
                try:
                    payload = self.load_from_path(path)
                    metrics = payload.get("metrics", {})
                    row["accuracy"] = metrics.get("accuracy", np.nan)
                    row["precision"] = metrics.get("precision", np.nan)
                    row["recall"] = metrics.get("recall", np.nan)
                    row["f1"] = metrics.get("f1", np.nan)
                    row["roc_auc"] = metrics.get("roc_auc", np.nan)
                    row["train_rows"] = payload.get("train_rows", np.nan)
                    row["test_rows"] = payload.get("test_rows", np.nan)
                    row["feature_count"] = len(payload.get("feature_columns", []))
                    fs = payload.get("feature_selection", {}) or {}
                    row["raw_feature_count"] = fs.get("raw_feature_count", np.nan)
                    row["train_end_datetime"] = payload.get("train_end_datetime")
                    row["test_start_datetime"] = payload.get("test_start_datetime")
                    tuned = payload.get("tuned_thresholds", {}) or {}
                    row["tuned_long_threshold"] = tuned.get("long", np.nan)
                    row["tuned_short_threshold"] = tuned.get("short", np.nan)
                except Exception:
                    pass

            if meta.get("interval") is not None and meta.get("horizon_bars") is not None:
                imp_path = self._importance_path(str(meta.get("pattern")), str(meta.get("interval")), int(meta.get("horizon_bars")))
                if imp_path.exists():
                    row["importance_file"] = imp_path.name

            rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "model_file",
                    "model_path",
                    "pattern",
                    "interval",
                    "horizon_bars",
                    "size_kb",
                    "modified_utc",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "train_rows",
                    "test_rows",
                    "feature_count",
                    "raw_feature_count",
                    "train_end_datetime",
                    "test_start_datetime",
                    "tuned_long_threshold",
                    "tuned_short_threshold",
                    "importance_file",
                ]
            )

        return pd.DataFrame(rows).sort_values("modified_utc", ascending=False).reset_index(drop=True)

    def get_model_details(self, model_file: str, top_n_importance: int = 30) -> dict[str, Any]:
        safe_file = Path(model_file).name
        path = self.models_dir / safe_file
        payload = self.load_from_path(path)
        parsed = self.parse_model_filename(path)

        importance = pd.DataFrame()
        pattern = parsed.get("pattern")
        interval = parsed.get("interval")
        horizon = parsed.get("horizon_bars")
        if pattern and interval and horizon is not None:
            importance = self.load_feature_importance(pattern, interval, int(horizon))
            if not importance.empty:
                importance = importance.sort_values("rank").head(top_n_importance)

        return {
            "model_file": safe_file,
            "model_path": str(path),
            "pattern": payload.get("pattern", pattern),
            "interval": payload.get("interval", interval),
            "horizon_bars": payload.get("horizon_bars", horizon),
            "metrics": payload.get("metrics", {}),
            "train_rows": payload.get("train_rows"),
            "test_rows": payload.get("test_rows"),
            "feature_columns": payload.get("feature_columns", []),
            "feature_count": len(payload.get("feature_columns", [])),
            "feature_selection": payload.get("feature_selection", {}),
            "train_end_datetime": payload.get("train_end_datetime"),
            "test_start_datetime": payload.get("test_start_datetime"),
            "probability_calibration": payload.get("probability_calibration", {}),
            "tuned_thresholds": payload.get("tuned_thresholds", {}),
            "importance": importance,
        }


def evaluate_binary(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> ModelMetrics:
    return ModelMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5,
    )


def timeseries_cv_score(x: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    if len(x) < 500:
        return float("nan")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    scores: list[float] = []
    for train_idx, test_idx in splitter.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = build_stacking_classifier(random_state=42)
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        if len(np.unique(y_test)) < 2:
            continue
        scores.append(float(roc_auc_score(y_test, prob)))

    if not scores:
        return float("nan")
    return float(np.mean(scores))

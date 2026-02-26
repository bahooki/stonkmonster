from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
import pandas as pd

from stonkmodel.features.dataset import split_xy
from stonkmodel.features.patterns import PATTERN_COLUMNS
from stonkmodel.models.calibration import (
    apply_probability_calibration,
    fit_probability_calibration,
    optimize_thresholds_from_validation,
)
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
    model_name: str | None = None
    parallel_patterns: int = 1
    model_n_jobs: int = -1
    fast_mode: bool = False
    max_rows_per_pattern: int | None = None
    enable_permutation_importance: bool = True
    enable_timeseries_cv: bool = True
    min_pattern_rows: int = 100
    max_feature_missing_rate: float = 0.65
    max_feature_correlation: float = 0.995
    min_selected_features: int = 25
    max_selected_features: int = 260
    calibration_fraction: float = 0.2


class PatternTrainer:
    def __init__(self, model_io: PatternModelIO) -> None:
        self.model_io = model_io

    @staticmethod
    def _summary_columns() -> list[str]:
        return [
            "model_name",
            "pattern",
            "model_file",
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
            "tuned_long_threshold",
            "tuned_short_threshold",
            "top_features",
            "importance_path",
        ]

    @staticmethod
    def _eligible_patterns(dataset: pd.DataFrame, min_rows: int) -> list[tuple[str, int]]:
        eligible: list[tuple[str, int]] = []
        if dataset.empty:
            return eligible

        for col in PATTERN_COLUMNS:
            if col not in dataset.columns:
                continue
            count = int(pd.to_numeric(dataset[col], errors="coerce").fillna(0).sum())
            if count >= int(min_rows):
                eligible.append((col.replace("pattern_", ""), count))

        if "pattern" in dataset.columns:
            none_count = int(dataset["pattern"].isna().sum())
            if none_count >= int(min_rows):
                eligible.append(("none", none_count))

        return eligible

    @staticmethod
    def _extract_pattern_frame(dataset: pd.DataFrame, pattern_name: str) -> pd.DataFrame:
        if dataset.empty:
            return pd.DataFrame()
        if pattern_name == "none":
            if "pattern" not in dataset.columns:
                return pd.DataFrame()
            return dataset.loc[dataset["pattern"].isna()].copy()
        col = f"pattern_{pattern_name}"
        if col not in dataset.columns:
            return pd.DataFrame()
        return dataset.loc[dataset[col] == 1].copy()

    @staticmethod
    def _cap_rows_for_speed(data: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
        if max_rows is None:
            return data
        cap = int(max_rows)
        if cap <= 0 or len(data) <= cap:
            return data
        if "datetime" in data.columns:
            return data.sort_values("datetime").tail(cap).copy()
        return data.tail(cap).copy()

    def train_all(
        self,
        dataset: pd.DataFrame,
        config: TrainConfig,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> pd.DataFrame:
        def emit(pct: float, message: str) -> None:
            if progress_callback is None:
                return
            progress_callback(max(0.0, min(100.0, float(pct))), str(message))

        emit(2.0, "Counting eligible candlestick pattern groups")
        eligible = self._eligible_patterns(dataset, min_rows=config.min_pattern_rows)
        total_patterns = len(eligible)
        emit(5.0, f"Eligible patterns: {total_patterns}")
        rows: list[dict[str, float | int | str]] = []
        if total_patterns == 0:
            emit(100.0, "Training finished (no eligible patterns)")
            return pd.DataFrame(columns=self._summary_columns())

        worker_count = max(1, int(config.parallel_patterns))
        if worker_count <= 1:
            for idx, (pattern_name, pattern_rows) in enumerate(eligible, start=1):
                start_pct = 5.0 + ((idx - 1) / max(1, total_patterns)) * 90.0
                emit(
                    start_pct,
                    f"Training pattern `{pattern_name}` ({idx}/{total_patterns}, rows={pattern_rows})",
                )
                pattern_data = self._extract_pattern_frame(dataset, pattern_name=pattern_name)
                pattern_data = self._cap_rows_for_speed(pattern_data, max_rows=config.max_rows_per_pattern)
                result = self.train_one(pattern_name, pattern_data, config)
                if result is None:
                    continue
                rows.append(result)
        else:
            emit(6.0, f"Training in parallel with {worker_count} workers")
            completed = 0
            launched = 0
            failures = 0
            started = time.time()
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures: dict[object, tuple[str, int]] = {}
                next_idx = 0

                def submit_one(i: int) -> None:
                    nonlocal launched
                    pattern_name, pattern_rows = eligible[i]
                    pattern_data = self._extract_pattern_frame(dataset, pattern_name=pattern_name)
                    pattern_data = self._cap_rows_for_speed(pattern_data, max_rows=config.max_rows_per_pattern)
                    future = pool.submit(self.train_one, pattern_name, pattern_data, config)
                    futures[future] = (pattern_name, pattern_rows)
                    launched += 1

                while next_idx < total_patterns and len(futures) < worker_count:
                    submit_one(next_idx)
                    next_idx += 1

                while futures:
                    done, _ = wait(list(futures.keys()), timeout=8.0, return_when=FIRST_COMPLETED)
                    if not done:
                        running = len(futures)
                        elapsed = int(time.time() - started)
                        pseudo = completed + (0.35 * running)
                        progress = 5.0 + (pseudo / max(1, total_patterns)) * 90.0
                        emit(
                            progress,
                            f"Running {running} pattern job(s), completed {completed}/{total_patterns}, elapsed {elapsed}s",
                        )
                        continue

                    for future in done:
                        pattern_name, _pattern_rows = futures.pop(future)
                        completed += 1
                        try:
                            result = future.result()
                        except Exception:
                            failures += 1
                            result = None
                        progress = 5.0 + (completed / max(1, total_patterns)) * 90.0
                        emit(
                            progress,
                            f"Finished pattern `{pattern_name}` ({completed}/{total_patterns})",
                        )
                        if result is not None:
                            rows.append(result)

                        if next_idx < total_patterns:
                            submit_one(next_idx)
                            next_idx += 1

                if failures > 0:
                    emit(96.0, f"Training completed with {failures} pattern failure(s)")

        if not rows:
            emit(100.0, "Training finished (no models met criteria)")
            return pd.DataFrame(columns=self._summary_columns())

        emit(100.0, f"Training finished ({len(rows)} models saved)")
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

        train_dt = pd.to_datetime(train["datetime"], utc=True, errors="coerce")
        test_dt = pd.to_datetime(test["datetime"], utc=True, errors="coerce")
        train_end_datetime = train_dt.max()
        test_start_datetime = test_dt.min()

        train = train.sort_values("datetime").reset_index(drop=True)
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

        split_idx = int(round(len(train) * (1.0 - float(config.calibration_fraction))))
        split_idx = max(1, min(split_idx, len(train) - 1))

        fit = train.iloc[:split_idx].copy()
        calibration = train.iloc[split_idx:].copy()
        if len(calibration) < 50:
            fit = train.copy()
            calibration = pd.DataFrame()

        x_fit = fit.reindex(columns=feature_cols).copy()
        x_fit = x_fit.fillna(x_fit.median(numeric_only=True)).fillna(0.0)
        y_fit = fit["future_direction"].astype(int)

        if y_fit.nunique() < 2:
            return None

        model = build_stacking_classifier(random_state=42, n_jobs=config.model_n_jobs, fast_mode=config.fast_mode)
        model.fit(x_fit, y_fit)

        calibration_params: dict[str, float] | None = None
        tuned_thresholds: dict[str, float | int] = {"long": 0.55, "short": 0.45, "objective": 0.0, "trades": 0}
        if not calibration.empty:
            x_cal = calibration.reindex(columns=feature_cols).copy()
            x_cal = x_cal.fillna(x_fit.median(numeric_only=True)).fillna(0.0)
            y_cal = calibration["future_direction"].astype(int)
            if y_cal.nunique() >= 2:
                prob_cal_raw = model.predict_proba(x_cal)[:, 1]
                calibration_params = fit_probability_calibration(prob_cal_raw, y_cal)
                prob_cal = apply_probability_calibration(prob_cal_raw, calibration_params)
                tuned = optimize_thresholds_from_validation(
                    prob_up=prob_cal,
                    future_return=calibration["future_return"],
                    min_trades=max(10, int(len(calibration) * 0.05)),
                )
                tuned_thresholds = tuned.to_dict()

        x_test = test.reindex(columns=feature_cols).copy()
        x_test = x_test.fillna(x_fit.median(numeric_only=True)).fillna(0.0)
        y_test = test["future_direction"].astype(int)

        if y_test.nunique() < 2:
            return None

        prob_raw = model.predict_proba(x_test)[:, 1]
        prob = apply_probability_calibration(prob_raw, calibration_params)
        pred = (prob >= 0.5).astype(int)
        metrics = evaluate_binary(y_test, pred, prob)

        artifact = PatternModelArtifact(
            pattern=pattern_name,
            interval=config.interval,
            horizon_bars=config.horizon_bars,
            model_name=config.model_name,
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
            train_end_datetime=train_end_datetime.isoformat() if pd.notna(train_end_datetime) else None,
            test_start_datetime=test_start_datetime.isoformat() if pd.notna(test_start_datetime) else None,
            probability_calibration=calibration_params or {},
            tuned_thresholds=tuned_thresholds,
        )
        model_path = self.model_io.save(artifact)
        saved_model_name = self.model_io.parse_model_filename(model_path.name).get("model_name")

        importance_path = ""
        top_features = ""
        if config.enable_permutation_importance:
            importance = compute_permutation_feature_importance(
                model=model,
                x=x_test,
                y=y_test,
                config=ImportanceConfig(n_repeats=6, max_samples=5000, random_state=42, scoring="roc_auc"),
            )
            if not importance.empty:
                importance_path = str(
                    self.model_io.save_feature_importance(
                        pattern=pattern_name,
                        interval=config.interval,
                        horizon_bars=config.horizon_bars,
                        frame=importance,
                        model_name=config.model_name,
                    )
                )
                top_features = ", ".join(importance["feature"].head(8).tolist())

        cv_auc = (
            timeseries_cv_score(x_fit, y_fit, n_splits=4, model_n_jobs=config.model_n_jobs)
            if config.enable_timeseries_cv
            else float("nan")
        )

        return {
            "model_name": saved_model_name,
            "pattern": pattern_name,
            "model_file": model_path.name,
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
            "tuned_long_threshold": float(tuned_thresholds.get("long", 0.55)),
            "tuned_short_threshold": float(tuned_thresholds.get("short", 0.45)),
            "top_features": top_features,
            "importance_path": importance_path,
        }

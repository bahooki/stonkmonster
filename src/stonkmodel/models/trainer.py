from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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


def _periods_per_year(interval: str) -> float:
    mapping = {
        "1d": 252.0,
        "1h": 252.0 * 6.5,
        "60m": 252.0 * 6.5,
        "30m": 252.0 * 13.0,
        "15m": 252.0 * 26.0,
        "5m": 252.0 * 78.0,
        "1m": 252.0 * 390.0,
    }
    return float(mapping.get(str(interval), 252.0))


def _safe_cumulative_return(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    clean = clean.clip(lower=-0.999999999)
    return float(np.expm1(np.log1p(clean).sum()))


def _safe_sharpe(series: pd.Series, interval: str) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    std = float(clean.std(ddof=1))
    if std <= 0 or np.isnan(std):
        return 0.0
    return float((clean.mean() / std) * np.sqrt(_periods_per_year(interval)))


def _max_drawdown(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    growth = np.exp(np.log1p(clean.clip(lower=-0.999999999)).cumsum())
    dd = growth / np.maximum.accumulate(growth) - 1.0
    return float(np.nanmin(dd)) if len(dd) else 0.0


def _net_objective_tuple(
    net_trade_returns: pd.Series,
    interval: str,
    trade_count: int,
) -> tuple[float, float, float, float, float]:
    clean = pd.to_numeric(net_trade_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return (-1e12, -1e12, -1e12, -1e12, -1e12)
    cum = _safe_cumulative_return(clean)
    sharpe = _safe_sharpe(clean, interval=interval)
    max_dd = _max_drawdown(clean)
    win_trade = float((clean > 0).mean())
    trade_penalty = -float(np.log1p(max(0, int(trade_count))))
    return (
        float(cum if np.isfinite(cum) else -1e12),
        float(sharpe if np.isfinite(sharpe) else -1e12),
        float(win_trade if np.isfinite(win_trade) else -1e12),
        float(-abs(max_dd) if np.isfinite(max_dd) else -1e12),
        float(trade_penalty),
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
    purge_overlap: bool = True
    embargo_bars: int = 0
    purged_cv_embargo_bars: int = 1
    include_patterns: set[str] | None = None
    candidate_random_states: tuple[int, ...] = (42,)
    transaction_cost_bps: float = 2.0
    meta_filter_enabled: bool = True
    meta_filter_threshold: float = 0.55
    meta_filter_min_trades: int = 120
    meta_filter_min_positive: int = 20


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
            "purged_train_rows",
            "accuracy",
            "f1",
            "roc_auc",
            "cv_roc_auc",
            "raw_features",
            "prefilter_features",
            "selected_features",
            "candidate_models_tested",
            "selected_random_seed",
            "dropped_missing",
            "dropped_constant",
            "dropped_correlated",
            "dropped_low_importance",
            "dropped_unstable",
            "stability_contexts",
            "tuned_long_threshold",
            "tuned_short_threshold",
            "meta_filter_enabled",
            "meta_filter_acceptance",
            "top_features",
            "importance_path",
        ]

    @staticmethod
    def _compute_label_end_datetime(frame: pd.DataFrame, bars_ahead: int) -> pd.Series:
        if frame.empty or "datetime" not in frame.columns:
            return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")

        steps = max(1, int(bars_ahead))
        out = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        work = frame.copy()
        work["datetime"] = pd.to_datetime(work["datetime"], utc=True, errors="coerce")
        if "symbol" not in work.columns:
            return work["datetime"].shift(-steps)

        for _, g in work.groupby("symbol", sort=False):
            g_sorted = g.sort_values("datetime")
            out.loc[g_sorted.index] = g_sorted["datetime"].shift(-steps)
        return out

    @staticmethod
    def _eligible_patterns(
        dataset: pd.DataFrame,
        min_rows: int,
        include_patterns: set[str] | None = None,
    ) -> list[tuple[str, int]]:
        eligible: list[tuple[str, int]] = []
        if dataset.empty:
            return eligible

        allowed = set(include_patterns) if include_patterns else None
        for col in PATTERN_COLUMNS:
            if col not in dataset.columns:
                continue
            pattern_name = col.replace("pattern_", "")
            if allowed and pattern_name not in allowed:
                continue
            count = int(pd.to_numeric(dataset[col], errors="coerce").fillna(0).sum())
            if count >= int(min_rows):
                eligible.append((pattern_name, count))

        if "pattern" in dataset.columns:
            none_count = int(dataset["pattern"].isna().sum())
            if none_count >= int(min_rows) and (allowed is None or "none" in allowed):
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

    @staticmethod
    def _meta_feature_frame(
        frame: pd.DataFrame,
        base_feature_cols: list[str],
        prob_up: np.ndarray,
        positions: np.ndarray,
    ) -> pd.DataFrame:
        meta = frame.reindex(columns=base_feature_cols).copy()
        p = pd.Series(prob_up, index=frame.index, dtype=float)
        side = pd.Series(np.sign(positions), index=frame.index, dtype=float)
        edge = (p - 0.5).abs()
        signed_edge = np.where(side >= 0, p - 0.5, 0.5 - p)
        meta["meta_prob_up"] = p
        meta["meta_abs_edge"] = edge
        meta["meta_side"] = side
        meta["meta_signed_edge"] = pd.Series(signed_edge, index=frame.index, dtype=float)
        return meta

    def _fit_meta_filter(
        self,
        calibration: pd.DataFrame,
        feature_cols: list[str],
        prob_cal: np.ndarray,
        long_t: float,
        short_t: float,
        config: TrainConfig,
        random_state: int,
    ) -> dict[str, object] | None:
        if calibration.empty or not config.meta_filter_enabled:
            return None
        if "future_return" not in calibration.columns:
            return None
        if len(prob_cal) != len(calibration):
            return None

        pos = np.where(prob_cal >= float(long_t), 1, np.where(prob_cal <= float(short_t), -1, 0))
        active = pos != 0
        if int(active.sum()) < int(config.meta_filter_min_trades):
            return None

        per_trade_cost = float(config.transaction_cost_bps) / 10000.0
        realized = pd.to_numeric(calibration["future_return"], errors="coerce").to_numpy(dtype=float)
        net_trade = (pos * realized) - per_trade_cost
        y_meta = (net_trade > 0).astype(int)
        y_active = y_meta[active]
        if len(y_active) < int(config.meta_filter_min_trades):
            return None

        positive = int((y_active == 1).sum())
        negative = int((y_active == 0).sum())
        min_pos = max(1, int(config.meta_filter_min_positive))
        if positive < min_pos or negative < min_pos:
            return None

        meta_all = self._meta_feature_frame(calibration, base_feature_cols=feature_cols, prob_up=prob_cal, positions=pos)
        meta_x = meta_all.loc[active].copy()
        meta_feature_cols = list(meta_x.columns)
        meta_x = meta_x.fillna(meta_x.median(numeric_only=True)).fillna(0.0)
        y_series = pd.Series(y_active, index=meta_x.index, dtype=int)
        if y_series.nunique() < 2:
            return None

        model = LogisticRegression(
            max_iter=600,
            class_weight="balanced",
            solver="lbfgs",
            random_state=int(random_state),
        )
        try:
            model.fit(meta_x, y_series)
        except Exception:
            return None

        return {
            "enabled": True,
            "threshold": float(np.clip(float(config.meta_filter_threshold), 0.50, 0.95)),
            "feature_columns": meta_feature_cols,
            "model": model,
            "train_rows": int(len(meta_x)),
            "positive_rate": float(y_series.mean()),
        }

    def _apply_meta_filter(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        prob_up: np.ndarray,
        positions: np.ndarray,
        meta_filter: dict[str, object] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not meta_filter or not bool(meta_filter.get("enabled", False)):
            return positions, np.full(len(positions), np.nan, dtype=float)

        model = meta_filter.get("model")
        meta_cols = [str(c) for c in list(meta_filter.get("feature_columns", []))]
        if model is None or not meta_cols:
            return positions, np.full(len(positions), np.nan, dtype=float)

        active = positions != 0
        if not bool(active.any()):
            return positions, np.full(len(positions), np.nan, dtype=float)

        meta_all = self._meta_feature_frame(frame, base_feature_cols=feature_cols, prob_up=prob_up, positions=positions)
        x_meta = meta_all.loc[active].reindex(columns=meta_cols).copy()
        x_meta = x_meta.fillna(x_meta.median(numeric_only=True)).fillna(0.0)
        try:
            meta_prob_active = model.predict_proba(x_meta)[:, 1]
        except Exception:
            return positions, np.full(len(positions), np.nan, dtype=float)

        threshold = float(np.clip(float(meta_filter.get("threshold", 0.55)), 0.5, 0.95))
        keep = meta_prob_active >= threshold
        out_pos = positions.copy()
        active_idx = np.where(active)[0]
        out_pos[active_idx[~keep]] = 0

        meta_prob_all = np.full(len(positions), np.nan, dtype=float)
        meta_prob_all[active_idx] = meta_prob_active
        return out_pos, meta_prob_all

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
        eligible = self._eligible_patterns(
            dataset,
            min_rows=config.min_pattern_rows,
            include_patterns=config.include_patterns,
        )
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

        work = data.copy().sort_values(["symbol", "datetime"]).reset_index(drop=True)
        work["_label_end_datetime"] = self._compute_label_end_datetime(
            work,
            bars_ahead=int(config.horizon_bars) + max(0, int(config.embargo_bars)),
        )

        train = work.loc[work["split"] == "train"].copy()
        test = work.loc[work["split"] == "test"].copy()
        if train.empty or test.empty:
            cutoff = int(len(work) * 0.8)
            train = work.iloc[:cutoff].copy()
            test = work.iloc[cutoff:].copy()

        if train.empty or test.empty:
            return None

        test_dt = pd.to_datetime(test["datetime"], utc=True, errors="coerce")
        test_start_datetime = test_dt.min()
        train_raw_rows = int(len(train))

        if config.purge_overlap and pd.notna(test_start_datetime):
            label_end = pd.to_datetime(train["_label_end_datetime"], utc=True, errors="coerce")
            train = train.loc[label_end < test_start_datetime].copy()

        purged_train_rows = max(0, train_raw_rows - int(len(train)))
        if train.empty:
            return None

        train_dt = pd.to_datetime(train["datetime"], utc=True, errors="coerce")
        train_end_datetime = train_dt.max()

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

        x_test = test.reindex(columns=feature_cols).copy()
        x_test = x_test.fillna(x_fit.median(numeric_only=True)).fillna(0.0)
        y_test = test["future_direction"].astype(int)

        if y_test.nunique() < 2:
            return None

        candidate_states = tuple(dict.fromkeys(int(s) for s in (config.candidate_random_states or (42,))))
        best_obj: tuple[float, float, float, float, float] | None = None
        best_model = None
        best_prob = None
        best_metrics = None
        best_calibration: dict[str, float] | None = None
        best_thresholds: dict[str, float | int] = {"long": 0.55, "short": 0.45, "objective": 0.0, "trades": 0}
        best_meta_filter: dict[str, object] | None = None
        best_meta_acceptance = np.nan
        best_seed = int(candidate_states[0])

        y_test_return = pd.to_numeric(test.get("future_return"), errors="coerce")
        period_scale = _periods_per_year(config.interval)
        per_trade_cost = float(config.transaction_cost_bps) / 10000.0

        for seed in candidate_states:
            model = build_stacking_classifier(random_state=int(seed), n_jobs=config.model_n_jobs, fast_mode=config.fast_mode)
            model.fit(x_fit, y_fit)

            calibration_params: dict[str, float] | None = None
            tuned_thresholds: dict[str, float | int] = {"long": 0.55, "short": 0.45, "objective": 0.0, "trades": 0}
            prob_cal: np.ndarray | None = None
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
                        future_excess_return=calibration["future_excess_return"] if "future_excess_return" in calibration.columns else None,
                        timestamps=calibration["datetime"] if "datetime" in calibration.columns else None,
                        min_trades=max(10, int(len(calibration) * 0.05)),
                        min_trades_per_side=max(4, int(len(calibration) * 0.01)),
                        transaction_cost_bps=float(config.transaction_cost_bps),
                        periods_per_year=period_scale,
                        exposure_penalty=0.4,
                        imbalance_penalty=0.25,
                    )
                    tuned_thresholds = tuned.to_dict()

            prob_raw = model.predict_proba(x_test)[:, 1]
            prob = apply_probability_calibration(prob_raw, calibration_params)
            pred = (prob >= 0.5).astype(int)
            metrics = evaluate_binary(y_test, pred, prob)

            long_t = float(tuned_thresholds.get("long", 0.55))
            short_t = float(tuned_thresholds.get("short", 0.45))
            pos = np.where(prob >= long_t, 1, np.where(prob <= short_t, -1, 0))

            meta_filter = self._fit_meta_filter(
                calibration=calibration,
                feature_cols=feature_cols,
                prob_cal=prob_cal if prob_cal is not None else np.array([]),
                long_t=long_t,
                short_t=short_t,
                config=config,
                random_state=int(seed),
            )
            pos, meta_prob = self._apply_meta_filter(
                frame=test,
                feature_cols=feature_cols,
                prob_up=prob,
                positions=pos,
                meta_filter=meta_filter,
            )

            mask = pos != 0
            if bool(mask.any()):
                strat = pd.Series((pos[mask] * y_test_return.to_numpy(dtype=float)[mask]) - per_trade_cost, dtype=float)
            else:
                strat = pd.Series(dtype=float)
            trades = int(mask.sum())
            obj = _net_objective_tuple(strat, interval=config.interval, trade_count=trades)
            meta_active = np.isfinite(meta_prob)
            meta_acceptance = float(trades / int(meta_active.sum())) if bool(meta_active.any()) else np.nan

            if best_obj is None or obj > best_obj:
                best_obj = obj
                best_model = model
                best_prob = prob
                best_metrics = metrics
                best_calibration = calibration_params
                best_thresholds = tuned_thresholds
                best_meta_filter = meta_filter
                best_meta_acceptance = meta_acceptance
                best_seed = int(seed)

        if best_model is None or best_prob is None or best_metrics is None:
            return None

        model = best_model
        prob = best_prob
        metrics = best_metrics
        calibration_params = best_calibration
        tuned_thresholds = best_thresholds
        meta_filter = best_meta_filter

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
                "candidate_models_tested": int(len(candidate_states)),
                "selected_random_seed": int(best_seed),
                "dropped_missing": int(selection.dropped_missing),
                "dropped_constant": int(selection.dropped_constant),
                "dropped_correlated": int(selection.dropped_correlated),
                "dropped_low_importance": int(selection.dropped_low_importance),
                "dropped_unstable": int(getattr(selection, "dropped_unstable", 0)),
                "stability_contexts": int(getattr(selection, "stability_contexts", 0)),
                "purged_train_rows": int(purged_train_rows),
            },
            train_end_datetime=train_end_datetime.isoformat() if pd.notna(train_end_datetime) else None,
            test_start_datetime=test_start_datetime.isoformat() if pd.notna(test_start_datetime) else None,
            probability_calibration=calibration_params or {},
            tuned_thresholds=tuned_thresholds,
            meta_filter=meta_filter or {},
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
                        model_file=model_path.name,
                    )
                )
                top_features = ", ".join(importance["feature"].head(8).tolist())

        cv_auc = (
            timeseries_cv_score(
                x_fit,
                y_fit,
                n_splits=4,
                model_n_jobs=config.model_n_jobs,
                purge_bars=max(1, int(config.horizon_bars)),
                embargo_bars=max(0, int(config.purged_cv_embargo_bars)),
            )
            if config.enable_timeseries_cv
            else float("nan")
        )

        return {
            "model_name": saved_model_name,
            "pattern": pattern_name,
            "model_file": model_path.name,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "purged_train_rows": int(purged_train_rows),
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
            "candidate_models_tested": int(len(candidate_states)),
            "selected_random_seed": int(best_seed),
            "dropped_missing": int(selection.dropped_missing),
            "dropped_constant": int(selection.dropped_constant),
            "dropped_correlated": int(selection.dropped_correlated),
            "dropped_low_importance": int(selection.dropped_low_importance),
            "dropped_unstable": int(getattr(selection, "dropped_unstable", 0)),
            "stability_contexts": int(getattr(selection, "stability_contexts", 0)),
            "tuned_long_threshold": float(tuned_thresholds.get("long", 0.55)),
            "tuned_short_threshold": float(tuned_thresholds.get("short", 0.45)),
            "meta_filter_enabled": bool((meta_filter or {}).get("enabled", False)),
            "meta_filter_acceptance": float(best_meta_acceptance) if np.isfinite(best_meta_acceptance) else np.nan,
            "top_features": top_features,
            "importance_path": importance_path,
        }
